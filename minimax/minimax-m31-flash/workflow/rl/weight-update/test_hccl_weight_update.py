#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
r"""Checkpoint-native -> vLLM HCCL 权重更新验证。

本脚本不导入 Transformers、不构造模型，也不使用 FSDP。独立 producer NPU
直接扫描 safetensors/PyTorch checkpoint，按 checkpoint 原名逐张量搬到 NPU，
再通过 HCCL 广播给全部 vLLM worker。接收端使用已适配模型自身的
``load_weights`` 完成名称映射、融合、量化处理和 TP/PP/EP 本地切分。

推理服务示例：

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_DEV_MODE=1 \
    vllm serve /models/new-model --served-model-name policy \
    --tensor-parallel-size 4 --load-format auto --enable-sleep-mode \
    --gpu-memory-utilization 0.3 --additional-config '{"weight_nz_mode":0}' \
    --weight-transfer-config '{"backend":"hccl"}' --port 8000

使用另一张卡启动 checkpoint producer：

  ASCEND_RT_VISIBLE_DEVICES=4 python test_hccl_weight_update.py \
    --server-url http://127.0.0.1:8000 --served-model-name policy \
    --checkpoint /models/new-model --device 0 \
    --master-address 127.0.0.1 --master-port 29501 \
    --prompts-json prompts.json --report hccl_report.json

要求 checkpoint 与服务模型架构、词表、dtype 和量化格式兼容。producer 与
推理服务必须使用不同物理 NPU。packed buffer 自动取显式下限和最大单张量
字节数中的较大值。服务必须先以同一真实 checkpoint 启动；本脚本验证该
checkpoint 在 ``sleep(level=2) -> wake(weights) -> update -> wake(kv_cache)``
之后的双接口输出与真实权重基线完全一致。

多机仍只需一个独立 producer。使用 --inference-nodes、--expected-world-size
和 --data-parallel-size 核对推理集群；master-address 必须为全部推理节点可达
的 producer 地址。广播组包含全局 TP×PP×DP worker，逐 DP 副本检查双接口输出。
"""
from __future__ import annotations

import argparse
import gc
import shlex
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

WEIGHT_UPDATE_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = WEIGHT_UPDATE_DIRECTORY.parent
WEIGHT_UPDATE_COMMON_DIRECTORY = WEIGHT_UPDATE_DIRECTORY / "common"
for directory in (WEIGHT_UPDATE_COMMON_DIRECTORY, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from checkpoint_weight_source import CheckpointWeightSource
from common.feature_arguments import (
    add_checkpoint_transfer_arguments,
    add_generation_arguments,
    add_report_arguments,
    add_service_arguments,
    add_timeout_argument,
    add_weight_update_cluster_arguments,
    create_feature_parser,
    port_number,
    nonnegative_integer,
    positive_integer,
)
from weight_update_validation import (
    WeightUpdateValidation,
    load_prompts,
    write_validation_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser)
    add_generation_arguments(parser, prompt_input="json")
    add_report_arguments(parser)
    add_timeout_argument(parser)
    add_checkpoint_transfer_arguments(parser)
    add_weight_update_cluster_arguments(parser)
    parser.add_argument("--device", type=nonnegative_integer, default=0, help="producer 使用的逻辑 NPU")
    parser.add_argument("--master-address", required=True, help="全部推理 worker 可达的 producer 地址")
    parser.add_argument("--master-port", type=port_number, required=True)
    parser.add_argument("--packed-num-buffers", type=positive_integer, default=1)
    return parser.parse_args(argv)


class BackgroundCall:
    def __init__(self, action: Callable[..., Any], *arguments: Any):
        self.future: Future = Future()
        self.name = getattr(action, "__name__", "HTTP request")

        def run() -> None:
            try:
                self.future.set_result(action(*arguments))
            except BaseException as exc:
                self.future.set_exception(exc)

        threading.Thread(target=run, name=self.name, daemon=True).start()

    def check(self) -> None:
        if self.future.done():
            self.future.result()

    def wait(self, timeout: float) -> None:
        try:
            self.future.result(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"等待 {self.name} 完成超时（{timeout} 秒）") from exc


def initialize_hccl_transfer(arguments: argparse.Namespace, client: Any, backend: Any, inference_worker_count: int) -> Any:
    # The worker-side HTTP endpoint needs ``rank_offset`` to place inference
    # workers after trainer rank 0.  The static trainer API only consumes the
    # rendezvous fields; keep the two payloads separate instead of relying on
    # HCCLWeightTransferEngine.trainer_init silently ignoring extra keys.
    worker_init_info = {
        "master_address": arguments.master_address,
        "master_port": arguments.master_port,
        "rank_offset": 1,
        "world_size": inference_worker_count + 1,
    }
    trainer_init_info = {
        key: worker_init_info[key]
        for key in ("master_address", "master_port", "world_size")
    }
    background_request = BackgroundCall(
        client.init_weight_transfer_engine,
        worker_init_info,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if background_request.future.done():
            background_request.future.result()
        time.sleep(0.1)
    transfer_group = backend.trainer_init(trainer_init_info)
    if not transfer_group.available or transfer_group.disabled:
        raise RuntimeError("HCCL communicator 不可用，请检查 CANN/HCCL 动态库")
    background_request.wait(arguments.timeout)
    return transfer_group


def transfer_checkpoint_weights(
    arguments: argparse.Namespace, weight_source: CheckpointWeightSource,
    client: Any, transfer_group: Any,
) -> dict[str, int]:
    import torch
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    tensor_metadata = weight_source.metadata()
    checkpoint_summary = weight_source.summary()
    packed_buffer_bytes = max(
        int(checkpoint_summary["max_tensor_bytes"]), arguments.packed_buffer_mb * 1024 * 1024,
    )
    packing = {
        "packed": True,
        "packed_buffer_size_bytes": packed_buffer_bytes,
        "packed_num_buffers": arguments.packed_num_buffers,
    }
    update_info = {
        "names": [item.name for item in tensor_metadata],
        "dtype_names": [str(item.dtype).split(".")[-1] for item in tensor_metadata],
        "shapes": [list(item.shape) for item in tensor_metadata],
        **packing,
    }
    print(
        f"[checkpoint producer] {len(tensor_metadata)} tensors, "
        f"max={checkpoint_summary['max_tensor_bytes'] / 2**30:.2f} GiB, "
        f"packed={packed_buffer_bytes / 2**30:.2f} GiB x {arguments.packed_num_buffers}",
        flush=True,
    )
    background_request = BackgroundCall(client.update_weights, update_info)
    background_request.check()
    with torch.no_grad():
        HCCLWeightTransferEngine.trainer_send_weights(
            iter(weight_source), HCCLTrainerSendWeightsArgs(group=transfer_group, **packing),
        )
    background_request.wait(arguments.timeout)
    torch.npu.synchronize()
    torch.npu.empty_cache()
    return {"packed_buffer_bytes": packed_buffer_bytes, "tensor_count": len(tensor_metadata)}


def run_validation(arguments: argparse.Namespace, report: dict[str, Any]) -> None:
    import torch
    from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

    prompts = load_prompts(arguments.prompts_json)
    validation = WeightUpdateValidation(
        arguments.server_url, arguments.served_model_name, prompts,
        arguments.max_tokens, arguments.timeout, report,
    )
    client = HTTPVLLMWeightSyncClient(arguments.server_url, timeout=arguments.timeout)
    validation.resolve_served_model_name()
    inference_worker_count = validation.configure_cluster(arguments)
    if len(report["cluster"]["inference_nodes"]) > 1:
        address = arguments.master_address.lower()
        if address in {"localhost", "0.0.0.0", "::", "::1"} or address.startswith("127."):
            raise ValueError("多机 HCCL --master-address 必须是所有推理 worker 可达的 producer 地址")
    report["transfer_world_size"] = inference_worker_count + 1
    validation.record_baseline()

    weight_source = CheckpointWeightSource(arguments.checkpoint, torch.device(f"npu:{arguments.device}"))
    report["checkpoint"] = weight_source.summary()
    transfer_group = None
    try:
        validation.sleep_service()
        transfer_group = initialize_hccl_transfer(
            arguments, client, HCCLWeightTransferEngine, inference_worker_count,
        )
        validation.wake_weight_buffers()
        client.start_weight_update()
        report["transfer"] = transfer_checkpoint_weights(arguments, weight_source, client, transfer_group)
        client.finish_weight_update()
        report["checks"]["checkpoint_stream_transfer"] = True
        report["checks"]["init_start_update_finish"] = True

        transfer_group = None
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        validation.wake_kv_cache()
    finally:
        transfer_group = None
        validation.record_incomplete_update()
    validation.verify_updated_outputs()


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    command = arguments.verification_command or shlex.join([sys.executable, *sys.argv])
    print(f"[verification command] {command}", flush=True)
    report: dict[str, Any] = {
        "mode": "checkpoint_native_hccl_weight_update",
        "server_url": arguments.server_url,
        "served_model_name": arguments.served_model_name,
        "checkpoint_path": str(Path(arguments.checkpoint).expanduser()),
        "producer_device": arguments.device,
        "packed": True,
        "packed_buffer_mb": arguments.packed_buffer_mb,
        "packed_num_buffers": arguments.packed_num_buffers,
        "weight_update_acceptance": "real_checkpoint_outputs_identical",
        "checks": {},
        "status": "failed",
        "verification_command": command,
    }
    try:
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available() or arguments.device >= torch.npu.device_count():
            raise RuntimeError(f"逻辑 NPU {arguments.device} 不可用，请检查可见设备设置")
        torch.npu.set_device(arguments.device)
        run_validation(arguments, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = repr(exc)
        print(f"HCCL checkpoint-native 更新失败：{exc}", file=sys.stderr, flush=True)
    finally:
        write_validation_report(arguments.report, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
