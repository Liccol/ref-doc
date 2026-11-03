#!/usr/bin/env python3
"""Create an offline, randomly initialized checkpoint with fewer decoder layers.

Uses the installed Transformers architecture's own initializer, including its
persistent buffers. Does not download weights, execute remote code, or start a
service. Unsupported layer schedules require a model-specific reduction rule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


METADATA_FILENAME = "reduced_checkpoint.json"
LAYER_SEQUENCE_FIELDS = ("layer_types",)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_layer_coverage(original: list[Any], reduced: list[Any], field: str) -> None:
    encode = lambda value: json.dumps(value, sort_keys=True)
    missing = set(map(encode, original)) - set(map(encode, reduced))
    if missing:
        raise ValueError(f"减层会遗漏 {field} 中的层类型 {sorted(missing)}；请增加 --layers")


def build_reduced_config(source: dict[str, Any], layers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep a decoder prefix and retain every supported structural layer type."""
    reduced = copy.deepcopy(source)
    text = reduced.get("text_config", reduced)
    if not isinstance(text, dict):
        raise ValueError("text_config 必须是对象")
    if text is not reduced and "num_hidden_layers" in reduced:
        raise ValueError("根配置和 text_config 均定义层数，需要按模型实现处理，不能猜测")
    original_layers = text.get("num_hidden_layers")
    if type(original_layers) is not int or not 0 < layers < original_layers:
        raise ValueError("需要 num_hidden_layers，且 --layers 必须大于 0、小于原层数")
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or not isinstance(architectures[0], str):
        raise ValueError("config 必须明确指定一个 architectures 模型类")
    for section in (source, text):
        if section.get("quantization_config") or section.get("compression_config"):
            raise ValueError("量化 checkpoint 需要专用生成器；不能生成浮点权重后保留量化标记")

    changes: dict[str, Any] = {"num_hidden_layers": {"before": original_layers, "after": layers}}
    for field in LAYER_SEQUENCE_FIELDS:
        if field not in text:
            continue
        sequence = text[field]
        if not isinstance(sequence, list) or len(sequence) != original_layers:
            raise ValueError(f"{field} 必须与原始层数等长")
        require_layer_coverage(sequence, sequence[:layers], field)
        text[field] = sequence[:layers]
        changes[field] = {"before": sequence, "after": text[field]}

    model_type = text.get("model_type", source.get("model_type"))
    dense_layers = text.get("mlp_only_layers", [])
    sparse_step = text.get("decoder_sparse_step", 1)
    if "mlp_only_layers" in text or "decoder_sparse_step" in text:
        if model_type not in {"qwen2_moe", "qwen3_moe", "qwen3_vl_moe_text"}:
            raise ValueError(f"{model_type} 的 MoE 分层规则需要模型专用适配")
        if (not isinstance(dense_layers, list)
                or any(type(index) is not int or not 0 <= index < original_layers for index in dense_layers)
                or type(sparse_step) is not int or sparse_step < 1):
            raise ValueError("无效的 mlp_only_layers / decoder_sparse_step")
        types = ["moe" if index not in dense_layers and (index + 1) % sparse_step == 0 else "dense"
                 for index in range(original_layers)]
        require_layer_coverage(types, types[:layers], "MoE/dense")
        if "mlp_only_layers" in text:
            text["mlp_only_layers"] = [index for index in dense_layers if index < layers]
            if text["mlp_only_layers"] != dense_layers:
                changes["mlp_only_layers"] = {"before": dense_layers, "after": text["mlp_only_layers"]}

    # These fields have model-specific index/boundary semantics. Keeping them
    # unchanged is not enough to prove the reduced model retains its structure.
    for field in ("first_k_dense_replace", "moe_layer_freq", "full_attention_interval",
                  "attention_types", "num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if text.get(field):
            raise ValueError(f"{field} 需要模型专用减层规则；请先阅读建模代码并适配，不能直接截断")
    if text.get("use_sliding_window") and "layer_types" not in text:
        raise ValueError("滑窗/全注意力分层需要显式 layer_types 或模型专用减层规则")
    for field, value in text.items():
        if isinstance(value, list) and len(value) == original_layers and field not in {
            *LAYER_SEQUENCE_FIELDS, "mlp_only_layers", "architectures", "eos_token_id",
        }:
            raise ValueError(f"发现可能与层数关联的字段 {field}；需确认其语义后添加减层规则")
    text["num_hidden_layers"] = layers
    return reduced, {
        "model_type": model_type,
        "architecture": architectures[0],
        "decoder_config_path": "text_config" if text is not reduced else ".",
        "original_layers": original_layers,
        "reduced_layers": layers,
        "retained_layer_indices": list(range(layers)),
        "config_changes": changes,
    }


def validate_state_dict(state: dict[str, Any], text_config: dict[str, Any]) -> None:
    import torch

    for name, tensor in state.items():
        if tensor.is_meta:
            raise ValueError(f"{name} 仍在 meta device，没有实际权重")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} 初始化后含 NaN/Inf，需要模型专用初始化")
        if name.endswith("tid2eid"):
            experts = text_config.get("n_routed_experts")
            if not experts or tensor.ndim != 2 or tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} 的路由表结构无法确认，需要模型专用初始化")
            ordered = tensor.sort(dim=-1).values
            if (torch.any(tensor < 0) or torch.any(tensor >= experts)
                    or torch.any(ordered[:, 1:] == ordered[:, :-1])):
                raise ValueError(f"{name} 含越界/重复专家编号，需要模型专用初始化")


def generate_checkpoint(config_path: Path, output: Path, reduced: dict, plan: dict,
                        *, seed: int, max_weight_gib: float, max_shard_size: str) -> dict:
    import torch
    import transformers
    from transformers import AutoConfig, PreTrainedModel

    if output.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling staging directory so a failed initialization/export
    # cannot be mistaken for a complete checkpoint.
    with tempfile.TemporaryDirectory(prefix=".reduced-checkpoint-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        write_json(staging / "config.json", reduced)
        config = AutoConfig.from_pretrained(staging, local_files_only=True, trust_remote_code=False)
        architecture = getattr(transformers, plan["architecture"], None)
        if not isinstance(architecture, type) or not issubclass(architecture, PreTrainedModel):
            raise ValueError(f"当前 Transformers 未提供 {plan['architecture']}；请在目标模型环境适配生成器")
        text_config = reduced.get("text_config", reduced)
        dtype_name = reduced.get("dtype", reduced.get("torch_dtype")) or text_config.get("dtype", text_config.get("torch_dtype")) or "float32"
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"不支持生成 {dtype_name} 权重，需要模型专用生成器")
        dtype = getattr(torch, dtype_name)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device("meta"):
                schema_model = architecture(config)
            weight_bytes = sum(tensor.numel() * tensor.element_size()
                               for tensor in schema_model.state_dict().values())
            del schema_model
            if weight_bytes > max_weight_gib * 1024**3:
                raise ValueError(f"减层后权重仍约 {weight_bytes / 1024**3:.2f} GiB，超过 --max-weight-gib；不自动缩小维度或专家数")
            if shutil.disk_usage(staging).free < weight_bytes * 1.1:
                raise ValueError("输出磁盘空间不足")
            # Preserve model-defined normalization constants, tied parameters
            # and integer buffers rather than filling every tensor with noise.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                with torch.device("cpu"):
                    model = architecture(config)
            model.eval()
            validate_state_dict(model.state_dict(), text_config)
            model.save_pretrained(staging, max_shard_size=max_shard_size)
        finally:
            torch.set_default_dtype(previous_dtype)
        # save_pretrained adds defaults/metadata; preserve the source config
        # verbatim except for the audited depth-dependent changes above.
        write_json(staging / "config.json", reduced)
        shutil.copyfile(config_path, staging / "source_config.json")
        weight_files = sorted(staging.glob("*.safetensors"))
        if not weight_files:
            raise ValueError("模型未导出 safetensors 权重")
        metadata = {
            "schema_version": 1,
            "kind": "synthetic_reduced_checkpoint",
            "validation_scope": "functional_smoke",
            "seed": seed,
            "dtype": dtype_name,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "source_config": str(config_path),
            "source_config_sha256": file_sha256(config_path),
            "generated_config_sha256": file_sha256(staging / "config.json"),
            "estimated_weight_bytes": weight_bytes,
            "weight_files": {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in weight_files},
            "tokenizer_included": False,
            **plan,
        }
        write_json(staging / METADATA_FILENAME, metadata)
        if output.exists():
            raise FileExistsError(f"输出目录已存在，不覆盖：{output}")
        staging.rename(output)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="本地 config.json 或其目录")
    parser.add_argument("--output", type=Path, required=True, help="尚不存在的输出 checkpoint 目录")
    parser.add_argument("--layers", type=int, required=True, help="保留的前缀层数，须覆盖全部结构层类型")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-weight-gib", type=float, default=8, help="CPU 分配前的权重大小上限，默认 8 GiB")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--dry-run", action="store_true", help="只输出配置差异，不分配权重，不需要 torch/Transformers")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.max_weight_gib) or args.max_weight_gib <= 0:
            raise ValueError("--max-weight-gib 必须是有限正数")
        config_path = args.config.expanduser().resolve()
        if config_path.is_dir():
            config_path /= "config.json"
        source = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError("config 必须是 JSON 对象")
        reduced, plan = build_reduced_config(source, args.layers)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            metadata = generate_checkpoint(config_path, args.output.expanduser().resolve(), reduced, plan,
                                           seed=args.seed, max_weight_gib=args.max_weight_gib,
                                           max_shard_size=args.max_shard_size)
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"生成减层 checkpoint 失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
