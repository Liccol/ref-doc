#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native weight source for vLLM weight-transfer validation.

The source intentionally does not construct a model.  It reads checkpoint
metadata on CPU, then materializes one checkpoint tensor at a time on the
requested device.  Names remain checkpoint names so the receiving vLLM
model's ``load_weights`` implementation performs model-specific mapping,
fusion, quantization handling, and parallel slicing.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class CheckpointParamMeta:
    name: str
    dtype: Any
    shape: tuple[int, ...]
    file: str


_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def _layer_order(names: list[str]) -> list[str]:
    """Keep non-layer tensors at the edges and decoder layers contiguous."""
    pre: list[str] = []
    post: list[str] = []
    stacks: dict[tuple[str, int], list[str]] = {}
    order: list[tuple[str, int]] = []
    for name in names:
        parts = name.split(".")
        key = next(
            ((".".join(parts[:index]), int(part))
             for index, part in enumerate(parts) if part.isdigit()),
            None,
        )
        if key is None:
            (post if order else pre).append(name)
            continue
        if key not in stacks:
            stacks[key] = []
            order.append(key)
        stacks[key].append(name)
    prefix_rank: dict[str, int] = {}
    for prefix, _ in order:
        prefix_rank.setdefault(prefix, len(prefix_rank))
    order.sort(key=lambda item: (prefix_rank[item[0]], item[1]))
    result = list(pre)
    for key in order:
        result.extend(stacks[key])
    result.extend(post)
    return result


def _resolve_files(checkpoint: Path) -> tuple[str, list[Path], dict[str, str] | None]:
    if checkpoint.is_file():
        suffix = checkpoint.suffix.lower()
        if suffix == ".safetensors":
            return "safetensors", [checkpoint], None
        if suffix in {".bin", ".pt", ".pth"}:
            return "pt", [checkpoint], None
        raise ValueError(f"不支持的 checkpoint 文件类型：{checkpoint}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint 路径不存在：{checkpoint}")

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"无效的 safetensors 索引：{index_path}")
        normalized = {str(name): str(file) for name, file in weight_map.items()}
        files = [checkpoint / name for name in sorted(set(normalized.values()))]
        return "safetensors", files, normalized

    safetensors_files = sorted(checkpoint.glob("model*.safetensors"))
    if not safetensors_files:
        safetensors_files = sorted(checkpoint.glob("*.safetensors"))
    if safetensors_files:
        return "safetensors", safetensors_files, None

    pt_index = checkpoint / "pytorch_model.bin.index.json"
    if pt_index.is_file():
        data = json.loads(pt_index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"无效的 PyTorch 权重索引：{pt_index}")
        normalized = {str(name): str(file) for name, file in weight_map.items()}
        files = [checkpoint / name for name in sorted(set(normalized.values()))]
        return "pt", files, normalized

    pt_files = sorted(checkpoint.glob("pytorch_model*.bin"))
    if not pt_files:
        pt_files = sorted(checkpoint.glob("*.bin"))
    if not pt_files:
        pt_files = sorted(checkpoint.glob("*.pt")) + sorted(checkpoint.glob("*.pth"))
    if pt_files:
        return "pt", pt_files, None
    raise FileNotFoundError(f"checkpoint 中未找到 safetensors 或 PyTorch 权重：{checkpoint}")


class CheckpointWeightSource:
    """Re-iterable checkpoint source compatible with vLLM ``WeightSource``."""

    def __init__(self, checkpoint: str, device: Any) -> None:
        import torch

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.format, self.files, weight_map = _resolve_files(self.checkpoint)
        missing = [str(path) for path in self.files if not path.is_file()]
        if missing:
            raise FileNotFoundError("索引引用的权重分片不存在：" + ", ".join(missing))
        self._metadata = self._scan_metadata(weight_map)
        if not self._metadata:
            raise ValueError("checkpoint 中没有可发送的张量")
        names = [item.name for item in self._metadata]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint 存在重复权重名")
        # safetensors permits cheap random access through mmap, so reorder it
        # layer-wise to bound vLLM's incremental loader state. PyTorch shards
        # are loaded as whole dictionaries and must stay file-contiguous.
        if self.format == "safetensors":
            by_name = {item.name: item for item in self._metadata}
            ordered = _layer_order(names)
            self._metadata = [by_name[name] for name in ordered]

    def _scan_metadata(self, weight_map: dict[str, str] | None) -> list[CheckpointParamMeta]:
        if self.format == "safetensors":
            return self._scan_safetensors(weight_map)
        return self._scan_pt(weight_map)

    def _scan_safetensors(
        self, weight_map: dict[str, str] | None,
    ) -> list[CheckpointParamMeta]:
        import torch
        from safetensors import safe_open

        result: list[CheckpointParamMeta] = []
        allowed = set(weight_map) if weight_map is not None else None
        for path in self.files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if allowed is not None and (
                        name not in allowed or Path(weight_map[name]).name != path.name
                    ):
                        continue
                    tensor_slice = handle.get_slice(name)
                    dtype_name = _SAFETENSORS_DTYPES.get(tensor_slice.get_dtype())
                    if dtype_name is None or not hasattr(torch, dtype_name):
                        raise ValueError(
                            f"权重 {name} 使用了当前 torch 不支持的 dtype "
                            f"{tensor_slice.get_dtype()}"
                        )
                    result.append(CheckpointParamMeta(
                        name=name,
                        dtype=getattr(torch, dtype_name),
                        shape=tuple(tensor_slice.get_shape()),
                        file=str(path),
                    ))
        if allowed is not None and {item.name for item in result} != allowed:
            missing = sorted(allowed - {item.name for item in result})
            raise ValueError(f"索引与 safetensors 内容不一致，缺失：{missing[:5]}")
        return result

    def _scan_pt(self, weight_map: dict[str, str] | None) -> list[CheckpointParamMeta]:
        import torch

        result: list[CheckpointParamMeta] = []
        allowed = set(weight_map) if weight_map is not None else None
        for path in self.files:
            state = self._load_pt_file(path)
            if not isinstance(state, dict):
                raise ValueError(f"PyTorch 权重分片不是 state_dict：{path}")
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if allowed is not None and (
                    name not in allowed or Path(weight_map[name]).name != path.name
                ):
                    continue
                result.append(CheckpointParamMeta(
                    name=name, dtype=tensor.dtype, shape=tuple(tensor.shape), file=str(path),
                ))
            del state
        if allowed is not None and {item.name for item in result} != allowed:
            missing = sorted(allowed - {item.name for item in result})
            raise ValueError(f"索引与 PyTorch 权重内容不一致，缺失：{missing[:5]}")
        return result

    def metadata(self) -> list[CheckpointParamMeta]:
        return list(self._metadata)

    def held_names(self) -> None:
        return None

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        if self.format == "safetensors":
            yield from self._iter_safetensors()
        else:
            yield from self._iter_pt()

    def _iter_safetensors(self) -> Iterator[tuple[str, Any]]:
        from safetensors import safe_open

        with ExitStack() as stack:
            handles = {
                str(path): stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                for path in self.files
            }
            for meta in self._metadata:
                cpu_tensor = handles[meta.file].get_tensor(meta.name)
                tensor = cpu_tensor.to(self.device).contiguous()
                yield meta.name, tensor

    def _iter_pt(self) -> Iterator[tuple[str, Any]]:
        loaded: dict[str, dict[str, Any]] = {}
        try:
            # PT shards cannot be randomly memory-mapped tensor-by-tensor. Keep
            # only the shard needed by the current metadata run.
            current_file = None
            state: dict[str, Any] | None = None
            for meta in self._metadata:
                if meta.file != current_file:
                    state = None
                    loaded.clear()
                    state = self._load_pt_file(Path(meta.file))
                    current_file = meta.file
                    loaded[current_file] = state
                assert state is not None
                tensor = state[meta.name].to(self.device).contiguous()
                yield meta.name, tensor
        finally:
            loaded.clear()

    @staticmethod
    def _load_pt_file(path: Path) -> dict[str, Any]:
        import torch

        try:
            return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as exc:
            if "mmap" not in str(exc).lower():
                raise
            return torch.load(str(path), map_location="cpu", weights_only=True)

    def summary(self) -> dict[str, Any]:
        file_facts = [(path.name, path.stat().st_size) for path in self.files]
        total_bytes = sum(
            item.dtype.itemsize
            * math.prod(item.shape)
            for item in self._metadata
        )
        max_item = max(
            self._metadata,
            key=lambda item: item.dtype.itemsize * math.prod(item.shape),
        )
        digest = hashlib.sha256()
        for name, size in file_facts:
            digest.update(f"file\0{name}\0{size}\n".encode())
        for item in self._metadata:
            digest.update(
                f"{item.name}\0{item.dtype}\0{item.shape}\0{Path(item.file).name}\n".encode()
            )
        return {
            "path": str(self.checkpoint),
            "format": self.format,
            "file_count": len(self.files),
            "total_file_bytes": sum(size for _, size in file_facts),
            "tensor_count": len(self._metadata),
            "total_tensor_bytes": total_bytes,
            "max_tensor_name": max_item.name,
            "max_tensor_bytes": max_item.dtype.itemsize
            * math.prod(max_item.shape),
            "manifest_sha256": digest.hexdigest(),
        }
