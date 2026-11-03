"""Validate torchrun placement and checkpoint agreement before IPC sleep."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping


def resolve_ipc_topology(arguments: Any, environment: Mapping[str, str] | None = None) -> dict:
    environment = os.environ if environment is None else environment
    devices = [int(item.strip()) for item in arguments.devices.split(",")]
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise ValueError("--devices 必须包含互不重复的非负本地逻辑 NPU 编号")
    rank = int(environment.get("RANK", "0"))
    world_size = int(environment.get("WORLD_SIZE", "1"))
    local_world_size = int(environment.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    if world_size < 1 or local_world_size < 1 or world_size % local_world_size:
        raise ValueError("WORLD_SIZE 必须为 LOCAL_WORLD_SIZE 的正整数倍；各节点使用相同 producer 数")
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise ValueError("RANK/LOCAL_RANK 越界")
    if local_world_size != len(devices) or rank % local_world_size != local_rank:
        raise ValueError("LOCAL_WORLD_SIZE、LOCAL_RANK 与 --devices / 全局 RANK 不匹配")
    node_size = world_size // local_world_size
    node_rank = rank // local_world_size
    if "GROUP_RANK" in environment and int(environment["GROUP_RANK"]) != node_rank:
        raise ValueError("torchrun GROUP_RANK 与推导的节点 rank 不一致")
    if arguments.node_size is not None and arguments.node_size != node_size:
        raise ValueError("--node-size 与 torchrun 全局拓扑不一致")
    if arguments.node_rank is not None and arguments.node_rank != node_rank:
        raise ValueError("--node-rank 与 torchrun 全局 rank 不一致")
    if arguments.master_addr and environment.get("MASTER_ADDR") and arguments.master_addr != environment["MASTER_ADDR"]:
        raise ValueError("--master-addr 与 torchrun MASTER_ADDR 不一致")
    if node_size > 1:
        master = environment.get("MASTER_ADDR", arguments.master_addr or "")
        if not master or master.lower() in {"localhost", "0.0.0.0", "::", "::1"} or master.startswith("127."):
            raise ValueError("多机 IPC 必须使用各节点可达的 torchrun MASTER_ADDR，不能使用 loopback")
    return {"rank": rank, "world_size": world_size, "local_world_size": local_world_size,
            "local_rank": local_rank, "node_size": node_size, "node_rank": node_rank,
            "devices": devices, "device": devices[local_rank]}


def load_inference_device_map(path: str | None, node_size: int) -> set[str] | None:
    """Read the physical placement audited from serving launch/state evidence."""
    if path is None:
        if node_size > 1:
            raise ValueError("多机 IPC 需要 --inference-devices-json，核对每个 producer 与推理物理卡的对应关系")
        return None
    nodes = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(nodes, list) or len(nodes) != node_size:
        raise ValueError("inference-devices-json 的节点数必须与 producer 节点数一致")
    uuids = set()
    hosts = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("推理设备映射的节点必须为对象")
        host, devices = node.get("host"), node.get("devices")
        if not isinstance(host, str) or not host.strip() or host in hosts:
            raise ValueError("推理节点地址缺失或重复")
        if (not isinstance(devices, list) or not devices
                or any(type(device) is not int or device < 0 for device in devices)
                or len(set(devices)) != len(devices)):
            raise ValueError("推理节点 devices 必须包含互不重复的物理 NPU 编号")
        hosts.add(host)
        uuids.update(f"{host}-{device}" for device in devices)
    return uuids


def validate_producer_records(records: list[dict], world_size: int, expected_uuids: set[str] | None = None) -> None:
    """Prevent node/rank collisions, differing manifests, or divergent packed rounds."""
    if len(records) != world_size or any(not isinstance(record, dict) for record in records):
        raise ValueError("缺少 producer 节点记录")
    if sorted(record["rank"] for record in records) != list(range(world_size)):
        raise ValueError("producer 全局 rank 缺失或重复")
    if len({record["npu_uuid"] for record in records}) != world_size:
        raise ValueError("producer 物理 NPU UUID 重复；每个 worker 必须对应同节点同物理卡的独立 producer")
    if expected_uuids is not None and {record["npu_uuid"] for record in records} != expected_uuids:
        raise ValueError("producer 物理 NPU 与推理设备映射不一致；检查每个节点的可见设备和主机地址")
    manifests = {(record["checkpoint"]["manifest_sha256"], record["checkpoint"]["tensor_count"],
                  record["checkpoint"]["total_tensor_bytes"]) for record in records}
    if len(manifests) != 1:
        raise ValueError("各节点 checkpoint manifest 不一致")
    buffers = {record["packed_buffer_bytes"] for record in records}
    if len(buffers) != 1:
        raise ValueError("各节点 packed buffer 大小不一致，会导致 chunk 集合通信失配")
    buffer_size = next(iter(buffers))
    if any(record["checkpoint"]["max_tensor_bytes"] > buffer_size for record in records):
        raise ValueError("packed buffer 小于 checkpoint 最大张量；请增大 --packed-buffer-mb")
    node_sizes = {record["node_size"] for record in records}
    if len(node_sizes) != 1:
        raise ValueError("producer 节点数量配置不一致")
    node_size = next(iter(node_sizes))
    if node_size < 1 or world_size % node_size:
        raise ValueError("producer 节点数与进程总数不匹配")
    per_node = world_size // node_size
    hosts = []
    for node_rank in range(node_size):
        node_records = [record for record in records if record["node_rank"] == node_rank]
        if sorted(record["local_rank"] for record in node_records) != list(range(per_node)):
            raise ValueError(f"节点 {node_rank} 的 LOCAL_RANK 缺失或重复")
        if any(record["rank"] != node_rank * per_node + record["local_rank"] for record in node_records):
            raise ValueError("producer 的节点 rank 与全局 rank 不匹配")
        host_ids = {record["npu_uuid"].rsplit("-", 1)[0] for record in node_records}
        if len(host_ids) != 1:
            raise ValueError(f"节点 {node_rank} 的 producer 分布在多个物理主机")
        hosts.append(next(iter(host_ids)))
    if len(set(hosts)) != node_size:
        raise ValueError("不同 node rank 落在同一物理主机，不能作为多机验证")
