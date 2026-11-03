#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
r"""Checkpoint-native 同卡 NPU IPC 权重更新验证。

每个推理 worker 需要一个位于同主机、同物理 NPU 的 checkpoint producer。
producer 不导入 Transformers、不构造模型、不使用 FSDP；它直接按 checkpoint
原名流式读取 safetensors/PyTorch 权重，每次只把一个张量或 packed chunk
放到本地 NPU 并导出 IPC handle。vLLM worker 使用模型自身的 ``load_weights``
完成名称映射、融合、量化处理及 TP/PP/EP 本地切分。

两卡示例：

  ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 \
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    vllm serve /models/new-model --served-model-name policy \
    --tensor-parallel-size 2 --load-format auto --enable-sleep-mode \
    --gpu-memory-utilization 0.3 --additional-config '{"weight_nz_mode":0}' \
    --weight-transfer-config '{"backend":"npu_ipc"}' --port 8000

  ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
    test_npu_ipc_weight_update.py --server-url http://127.0.0.1:8000 \
    --checkpoint /models/new-model --devices 0,1 \
    --prompts-json prompts.json --report ipc_report.json

``WORLD_SIZE`` 必须与推理 worker 数一致。单卡可直接用 python 并指定
``--devices 0``。packed buffer 必须至少容纳 checkpoint 中最大的单个张量。
服务必须先以同一真实 checkpoint 启动。本脚本验证该 checkpoint 在
``sleep(level=2) -> wake(weights) -> update -> wake(kv_cache)`` 之后，两个
接口和每个 prompt 的推理结果与真实权重基线完全一致。仅在可信环境启用
IPC 不安全序列化。

多机使用每节点 torchrun --nnodes/--node-rank 启动；--node-size/--node-rank
与运行时全局拓扑核对。传入 --inference-devices-json 审计推理物理卡对应关系，
休眠前统一校验全部节点的 checkpoint、UUID 和 packed 大小。只有全局 rank 0
发出服务生命周期控制请求，所有 rank 参与 handle 汇总和数据传输。
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import shlex
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

WEIGHT_UPDATE_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = WEIGHT_UPDATE_DIRECTORY.parent
WEIGHT_UPDATE_COMMON_DIRECTORY = WEIGHT_UPDATE_DIRECTORY / "common"
for directory in (WEIGHT_UPDATE_COMMON_DIRECTORY, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from checkpoint_weight_source import CheckpointWeightSource
from producer_topology import load_inference_device_map, resolve_ipc_topology, validate_producer_records
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


def _uses_unsupported_vllm_version() -> bool:
    try:
        import vllm

        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(vllm.__version__))
        return match is not None and tuple(map(int, match.groups())) <= (0, 26, 0)
    except Exception:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser)
    add_generation_arguments(parser, prompt_input="json")
    add_report_arguments(parser)
    add_timeout_argument(parser)
    add_checkpoint_transfer_arguments(parser)
    add_weight_update_cluster_arguments(parser)
    parser.add_argument("--devices", required=True, help="按 LOCAL_RANK 排列的本地逻辑 NPU 编号")
    parser.add_argument("--master-addr", default=None)
    parser.add_argument("--master-port", type=port_number, default=29501)
    parser.add_argument("--node-size", type=positive_integer, default=None,
                        help="producer 节点总数；与 torchrun --nnodes 核对")
    parser.add_argument("--node-rank", type=nonnegative_integer, default=None,
                        help="本节点编号；与 torchrun --node-rank 核对")
    parser.add_argument("--inference-devices-json", default=None,
                        help="推理物理设备映射 JSON；多机必填，格式为 [{host, devices}]，host 与 NPU UUID 一致")
    return parser.parse_args(argv)


def resolve_distributed_config(arguments: argparse.Namespace) -> tuple[int, int, int, list[int], int]:
    topology = resolve_ipc_topology(arguments)
    arguments.producer_topology = topology
    if arguments.master_addr:
        os.environ.setdefault("MASTER_ADDR", arguments.master_addr)
    elif topology["node_size"] == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(arguments.master_port))
    return (topology["rank"], topology["world_size"], topology["local_world_size"],
            topology["devices"], topology["device"])


def run_distributed_step(
    arguments: argparse.Namespace, distributed: Any, rank: int, name: str,
    action: Callable[[], Any],
) -> Any:
    result = None
    error = None
    try:
        result = action()
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [error]
    if distributed.is_initialized():
        errors = [None] * distributed.get_world_size()
        distributed.all_gather_object(errors, error, group=arguments.control_group)
    failures = [item for item in errors if item]
    if failures:
        raise RuntimeError(f"{name} 失败：" + "; ".join(failures))
    return result


def register_ipc_producer_backend(factory: Any) -> None:
    try:
        factory.register_engine(
            "npu_ipc", "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
            "NPUIPCTrainerWeightTransferEngine",
        )
    except ValueError as exc:
        if "'npu_ipc' is already registered" not in str(exc):
            raise


def create_checkpoint_producer(
    arguments: argparse.Namespace, rank: int, weight_source: CheckpointWeightSource,
) -> Any:
    checkpoint_summary = weight_source.summary()
    packed_buffer_bytes = arguments.packed_buffer_mb * 1024 * 1024
    if int(checkpoint_summary["max_tensor_bytes"]) > packed_buffer_bytes:
        required_mb = (int(checkpoint_summary["max_tensor_bytes"]) + 1024 * 1024 - 1) // (1024 * 1024)
        raise ValueError(
            f"最大张量 {checkpoint_summary['max_tensor_name']} 需要 {checkpoint_summary['max_tensor_bytes']} 字节；"
            f"请设置 --packed-buffer-mb 至少为 {required_mb}"
        )

    if _uses_unsupported_vllm_version():
        raise RuntimeError(
            "checkpoint-native NPU IPC requires the factory API introduced after "
            "vLLM 0.26.0; this checkout has no compatible legacy producer API"
        )
    from vllm.distributed.weight_transfer import (
        HTTPVLLMWeightSyncClient,
        WeightTransferTrainerFactory,
    )
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        NPUIPCTrainerInitInfo,
    )

    register_ipc_producer_backend(WeightTransferTrainerFactory)
    producer = WeightTransferTrainerFactory.trainer_init(
        NPUIPCTrainerInitInfo(
            rank=rank, packed=True, packed_buffer_size_bytes=packed_buffer_bytes,
        ),
        client=HTTPVLLMWeightSyncClient(arguments.server_url, timeout=arguments.timeout),
        source=weight_source,
    )
    return producer


def run_validation(
    arguments: argparse.Namespace, report: dict[str, Any], distributed: Any,
    rank: int, world_size: int, device_index: int,
) -> None:
    import torch

    def run_on_all_ranks(step_name: str, action: Callable[[], Any]) -> Any:
        print(f"[rank {rank}] {step_name}", flush=True)
        return run_distributed_step(arguments, distributed, rank, step_name, action)

    def run_on_coordinator(step_name: str, action: Callable[[], Any]) -> Any:
        return run_on_all_ranks(step_name, lambda: action() if rank == 0 else None)

    prompts = run_on_all_ranks("读取提示词", lambda: load_prompts(arguments.prompts_json))
    validation = WeightUpdateValidation(
        arguments.server_url, arguments.served_model_name, prompts,
        arguments.max_tokens, arguments.timeout, report,
    )

    def validate_service_topology() -> None:
        validation.resolve_served_model_name()
        inference_worker_count = validation.configure_cluster(arguments)
        if inference_worker_count != world_size:
            raise ValueError(
                f"推理 worker 数 {inference_worker_count} 与 checkpoint producer 数 {world_size} 不一致"
            )
        nodes = report["cluster"]["inference_nodes"]
        if nodes and len(nodes) != arguments.producer_topology["node_size"]:
            raise ValueError("IPC producer 节点数必须与 --inference-nodes 数量一致")

    run_on_coordinator("查询服务拓扑", validate_service_topology)
    run_on_coordinator("生成更新前输出", validation.record_baseline)

    # Validate every producer manifest before sleep discards service weights.
    weight_source = run_on_all_ranks(
        "扫描 checkpoint manifest",
        lambda: CheckpointWeightSource(arguments.checkpoint, torch.device(f"npu:{device_index}")),
    )
    def build_producer_record() -> dict:
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        return {**arguments.producer_topology, "npu_uuid": npu_generate_uuid(device_index),
                "checkpoint": weight_source.summary(),
                "packed_buffer_bytes": arguments.packed_buffer_mb * 1024 * 1024}

    local_record = run_on_all_ranks("核验本地 producer 设备与 checkpoint", build_producer_record)
    producer_records = [local_record]
    if distributed.is_initialized():
        producer_records = [None] * world_size
        distributed.all_gather_object(producer_records, local_record, group=arguments.control_group)
    if rank == 0:
        report["producer_nodes"] = producer_records
    def validate_placement() -> None:
        expected_uuids = load_inference_device_map(arguments.inference_devices_json,
                                                   arguments.producer_topology["node_size"])
        validate_producer_records(producer_records, world_size, expected_uuids)
    run_on_all_ranks("核验全节点 producer 映射与 packed 配置", validate_placement)
    def validate_factory_api() -> None:
        if _uses_unsupported_vllm_version():
            raise RuntimeError("checkpoint-native NPU IPC requires the post-0.26 factory API")
        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
        if not callable(getattr(WeightTransferTrainerFactory, "trainer_init", None)):
            raise RuntimeError("目标 vLLM 缺少 WeightTransferTrainerFactory.trainer_init")
    run_on_all_ranks("核验 IPC factory API", validate_factory_api)
    if rank == 0:
        report["checkpoint"] = local_record["checkpoint"]
        report["checks"]["checkpoint_manifest_consistent"] = True
        report["checks"]["producer_placement_and_buffers"] = True

    producer = None
    try:
        run_on_coordinator("休眠 level-2", validation.sleep_service)
        producer = run_on_all_ranks(
            "准备 checkpoint producer",
            lambda: create_checkpoint_producer(arguments, rank, weight_source),
        )
        run_on_coordinator("唤醒权重分配", validation.wake_weight_buffers)
        run_on_all_ranks("发送 checkpoint 权重流", producer.send_weights)
        if rank == 0:
            report["checks"]["checkpoint_stream_transfer"] = True

        producer = None
        weight_source = None
        def release_producer_buffers() -> None:
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
        run_on_all_ranks("释放 producer 缓冲", release_producer_buffers)
        run_on_coordinator("唤醒 KV cache", validation.wake_kv_cache)
    finally:
        producer = None
        weight_source = None
        validation.record_incomplete_update()

    # Propagate coordinator acceptance failures to every producer rank.
    run_on_coordinator("验证更新后输出", validation.verify_updated_outputs)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    command = arguments.verification_command or shlex.join([sys.executable, *sys.argv])
    print(f"[verification command] {command}", flush=True)
    rank = int(os.environ.get("RANK", "0"))
    report: dict[str, Any] = {
        "mode": "checkpoint_native_npu_ipc_weight_update",
        "rank": rank,
        "server_url": arguments.server_url,
        "served_model_name": arguments.served_model_name,
        "checkpoint_path": str(Path(arguments.checkpoint).expanduser()),
        "packed": True,
        "packed_buffer_mb": arguments.packed_buffer_mb,
        "weight_update_acceptance": "real_checkpoint_outputs_identical",
        "checks": {},
        "status": "failed",
        "verification_command": command,
    }
    distributed = None
    try:
        rank, world_size, local_world_size, devices, device_index = resolve_distributed_config(arguments)
        report.update(
            world_size=world_size, local_world_size=local_world_size,
            multi_node=world_size > local_world_size, devices=devices,
            node_size=arguments.producer_topology["node_size"],
            node_rank=arguments.producer_topology["node_rank"],
        )
        import torch
        import torch.distributed as distributed
        import torch_npu  # noqa: F401

        if not torch.npu.is_available() or device_index >= torch.npu.device_count():
            raise RuntimeError(f"逻辑 NPU {device_index} 不可用，请检查可见设备设置")
        torch.npu.set_device(device_index)
        arguments.control_group = None
        if world_size > 1:
            timeout = timedelta(seconds=arguments.timeout)
            distributed.init_process_group(
                backend="cpu:gloo,npu:hccl", rank=rank, world_size=world_size, timeout=timeout,
            )
            arguments.control_group = distributed.new_group(backend="gloo", timeout=timeout)
        run_validation(arguments, report, distributed, rank, world_size, device_index)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = repr(exc)
        print(f"[rank {rank}] IPC checkpoint-native 更新失败：{exc}", file=sys.stderr, flush=True)
    finally:
        if distributed is not None and distributed.is_initialized():
            try:
                distributed.destroy_process_group()
            except Exception as exc:
                report["cleanup_error"] = repr(exc)
                report["status"] = "failed"
        gc.collect()
        if rank == 0:
            write_validation_report(arguments.report, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
