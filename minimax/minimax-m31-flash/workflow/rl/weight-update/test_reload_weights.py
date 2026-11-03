#!/usr/bin/env python3
r"""在线 checkpoint 权重重载验证（等价于 npu_ipc / hccl 用例的 checkpoint 版）。

对照 test_npu_ipc_weight_update.py 与 test_hccl_weight_update.py：它们分别通过
NPU-IPC/HCCL 从 checkpoint producer 向推理 worker 传输权重；IPC 使用同卡
torchrun producer，HCCL 使用独立 NPU 上的一个 producer。
checkpoint 路径则完全不同——它不搬运权重，而是在已运行的推理服务内就地重载
checkpoint：在已运行的真实权重服务内通过 dev-mode 的 /collective_rpc 调用
reload_weights(is_checkpoint_format=True) 重载同一真实 checkpoint。因此：

  - 无需 torchrun 训练进程、无需训练卡、无需 weight-transfer-config /
    rendezvous 端口；
  - 单进程客户端即可，直接连接已启动的服务；
  - 全部卡都可交给推理服务。

更新流程为：服务先以真实 checkpoint 启动并 generate 基线 -> sleep(level=2)
释放权重和 KV cache -> wake(weights) -> reload_weights(同一真实 checkpoint) ->
wake(kv_cache) -> 再次 generate，并要求
两个接口的结果与基线完全一致。

多机传入 --inference-nodes、--expected-world-size 和 --data-parallel-size；
API 必须通过 collective RPC 管理全部 worker，checkpoint 路径必须在各节点可读。
脚本逐 DP 副本验证两个接口，不以 RPC 仅返回某个副本的结果列表推断全局覆盖。

使用示例（在本脚本所在目录执行，模型路径按实际环境替换）：
  1. 启动推理服务，真实 checkpoint 权重 + sleep mode + dev mode（服务端配置见
     下方示例，与 IPC/HCCL 一样也请设置
     --additional-config '{"weight_nz_mode":0}')：

     ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 \
       VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /mnt/weight/Qwen3-30B-A3B \
       --served-model-name policy --tensor-parallel-size 2 --dtype bfloat16 \
       --load-format auto --enforce-eager --enable-sleep-mode --max-model-len 2048 \
       --gpu-memory-utilization 0.7 --additional-config '{"weight_nz_mode":0}' --port 8000

  2. 等待 /health 就绪后执行客户端：

     echo '{"prompts":["The capital of France is"]}' > prompts.json
     python test_reload_weights.py \
       --server-url http://127.0.0.1:8000 \
       --updated-model /mnt/weight/Qwen3-30B-A3B \
       --prompts-json prompts.json \
       --report checkpoint_report.json

参数说明与判定：
  - --served-model-name auto 会查询 /v1/models；多模型时须明确指定。
  - --load-format 必须为 auto，表示服务已经以真实 checkpoint 启动；脚本不接受
    dummy 初始权重，也不会切换 loader。
  - --timeout 控制 HTTP 等待。服务必须先以真实 checkpoint 启动，更新前后
    两个接口的 text、token 和 logprob 结果必须一致；不一致即失败。
  - 写 JSON 报告，记录 baseline_by_endpoint、post_update_by_endpoint、checks、status；返回 0 表示
    重载及更新后生成成功，失败返回 2。
  - level-2 sleep 丢弃原权重；更新失败时不唤醒 KV cache，报告
    server_left_sleeping；休眠状态未确认时报告 server_state_unknown。
    权重可能不完整，需要重启服务后重试。脚本不会自动回滚权重。
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

WEIGHT_UPDATE_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = WEIGHT_UPDATE_DIRECTORY.parent
WEIGHT_UPDATE_COMMON_DIRECTORY = WEIGHT_UPDATE_DIRECTORY / "common"
for directory in (WEIGHT_UPDATE_COMMON_DIRECTORY, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from common.feature_arguments import (
    add_generation_arguments,
    add_report_arguments,
    add_service_arguments,
    add_timeout_argument,
    add_weight_update_cluster_arguments,
    create_feature_parser,
)
from weight_update_validation import (
    WeightUpdateValidation,
    load_prompts,
    request_json,
    write_validation_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser)
    add_generation_arguments(parser, prompt_input="json")
    add_report_arguments(parser)
    add_timeout_argument(parser)
    add_weight_update_cluster_arguments(parser)
    parser.add_argument("--updated-model", required=True, help="要重载的 checkpoint 路径或模型 ID")
    parser.add_argument("--load-format", choices=("auto",), default="auto",
                   help="必须为 auto：服务需先以真实 checkpoint 启动")
    parser.add_argument("--model", default=None, help="兼容旧参数，作为服务模型名用于更新前后请求")
    arguments = parser.parse_args(argv)
    if arguments.model and arguments.served_model_name != "auto" and arguments.model != arguments.served_model_name:
        raise ValueError("--model 与 --served-model-name 必须指定同一服务模型")
    return arguments


def reload_checkpoint_weights(arguments: argparse.Namespace) -> Any:
    """通过 dev-mode /collective_rpc 在所有 worker 上就地重载 checkpoint。"""
    return request_json(arguments.server_url, "POST", "/collective_rpc", {
        "method": "reload_weights",
        "args": [],
        "kwargs": {
            "weights_path": arguments.updated_model,
            "is_checkpoint_format": True,
        },
        "timeout": arguments.timeout,
    }, timeout=arguments.timeout)


def run_validation(arguments: argparse.Namespace, report: dict[str, Any]) -> None:
    prompts = load_prompts(arguments.prompts_json)
    validation = WeightUpdateValidation(
        arguments.server_url, arguments.model or arguments.served_model_name, prompts,
        arguments.max_tokens, arguments.timeout, report,
    )
    validation.configure_cluster(arguments)
    validation.record_baseline()
    try:
        validation.sleep_service()
        validation.wake_weight_buffers()
        report["checkpoint_reload_result"] = reload_checkpoint_weights(arguments)
        report["checks"]["checkpoint_reload"] = True
        validation.wake_kv_cache()
    finally:
        validation.record_incomplete_update()
    validation.verify_updated_outputs()


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    command = arguments.verification_command or shlex.join([sys.executable, *sys.argv])
    print(f"[verification command] {command}", flush=True)
    report: dict[str, Any] = {
        "mode": "checkpoint_weight_update",
        "server_url": arguments.server_url,
        # 公共验证方法会解析 auto 并记录真实服务模型名。
        "served_model_name": arguments.served_model_name,
        "updated_model": arguments.updated_model,
        "load_format": arguments.load_format,
        "weight_update_acceptance": "real_checkpoint_outputs_identical",
        "checks": {},
        "status": "failed",
        "verification_command": command,
    }
    try:
        run_validation(arguments, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = repr(exc)
        print(f"checkpoint 权重更新验证失败：{exc}", file=sys.stderr, flush=True)
    finally:
        write_validation_report(arguments.report, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
