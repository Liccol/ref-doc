"""Identify generated reduced checkpoints without importing a model runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METADATA_FILENAME = "reduced_checkpoint.json"


def load_checkpoint_provenance(checkpoint: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint).expanduser()
    metadata_path = (path.parent if path.is_file() else path) / METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or metadata.get("kind") != "synthetic_reduced_checkpoint"
            or metadata.get("schema_version") != 1
            or any(field not in metadata for field in ("original_layers", "reduced_layers", "seed", "source_config_sha256"))):
        raise ValueError(f"无效的减层 checkpoint 元数据：{metadata_path}")
    return {**metadata, "metadata_path": str(metadata_path.resolve())}


def annotate_weight_update_report(report: dict[str, Any]) -> None:
    checkpoint = report.get("checkpoint_path") or report.get("updated_model")
    if not checkpoint:
        return
    provenance = load_checkpoint_provenance(checkpoint)
    if provenance:
        report["checkpoint_provenance"] = provenance
        report["validation_scope"] = "functional_smoke"
        report["weight_update_acceptance"] = "synthetic_reduced_checkpoint_outputs_identical"
