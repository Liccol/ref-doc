#!/usr/bin/env python3
"""Estimate a single device's compute-kernel efficiency and export an XLSX audit."""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def positive_number(value: str | float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Expected a finite positive number, got {value!r}")
    return number


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_column(columns: list[str], requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Missing column: {requested}")
        return requested
    matches = [name for name in columns if normalize_column(name) in candidates]
    if len(matches) != 1:
        raise ValueError(f"Specify a column explicitly; candidates {candidates}, found {matches}")
    return matches[0]


def load_rules(path: Path | None) -> list[dict]:
    rules = json.loads(path.read_text(encoding="utf-8")) if path else []
    if not isinstance(rules, list):
        raise ValueError("Ceiling rules must be a JSON list")
    for rule in rules:
        if not isinstance(rule.get("operator"), str) or not rule["operator"]:
            raise ValueError("Each ceiling rule needs an operator glob")
        if not isinstance(rule.get("metric"), str) or not rule["metric"]:
            raise ValueError("Each ceiling rule needs a metric column name")
        if not isinstance(rule.get("source"), str) or not rule["source"].strip():
            raise ValueError("Each ceiling rule needs its empirical source")
        rule["ceiling"] = positive_number(rule["ceiling"])
        if rule["ceiling"] > 1:
            raise ValueError("Ceilings use fractions in (0, 1]")
    return rules


def analyze_kernels(source: Path, *, device: str, ratio_unit: str = "fraction",
                    default_ceiling: float = 0.8, short_kernel_us: float = 10,
                    priority_duration_us: float = 20, priority_ratio: float = 0.6,
                    rules: list[dict] | None = None, columns: dict | None = None,
                    validation_scope: str = "full_model") -> dict:
    """Weight individual calls before grouping, preserving all layers and short calls."""
    default_ceiling = positive_number(default_ceiling)
    short_kernel_us = positive_number(short_kernel_us)
    priority_duration_us = positive_number(priority_duration_us)
    if default_ceiling > 1 or not 0 <= priority_ratio <= 1:
        raise ValueError("Ceiling and priority ratio must be fractions")
    if ratio_unit not in ("fraction", "percent"):
        raise ValueError("ratio_unit must be fraction or percent")
    if validation_scope not in ("full_model", "functional_smoke"):
        raise ValueError("Unknown validation scope")
    if not str(device).strip() or "," in str(device):
        raise ValueError("Select exactly one device ID")
    columns = columns or {}
    rules = rules or []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        if len(set(names)) != len(names):
            raise ValueError("Duplicate CSV headers")
        name_column = find_column(names, columns.get("name"), ("name", "op_name", "kernel_name"))
        type_column = find_column(names, columns.get("type"), ("task_type",))
        duration_column = find_column(names, columns.get("duration"), ("duration_us",))
        device_column = columns.get("device")
        if device_column:
            find_column(names, device_column, ())
        else:
            device_column = next((name for name in names if normalize_column(name) == "device_id"), None)
        ratio_columns = [name for name in names if normalize_column(name).endswith("_ratio")
                         and normalize_column(name) != "dur_ratio"]
        if not ratio_columns:
            raise ValueError("No *_ratio columns; collect PipeUtilization metrics before estimating")
        for rule in rules:
            if rule["metric"] not in ratio_columns:
                raise ValueError(f"Ceiling rule refers to missing ratio column: {rule['metric']}")
        kernels, excluded = [], []
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {row_number}")
            reason = None
            operator = row[name_column]
            task_type = normalize_column(row[type_column]).upper()
            if device_column and row[device_column].strip() != str(device):
                reason = "other_device"
            elif re.search(r"hccl|hcom|lccl|all[_ -]?reduce|all[_ -]?gather|reduce[_ -]?scatter|send[_ -]?recv|memcpy|memset|moedistribute(?:dispatch|combine)", operator, re.I):
                reason = "communication_or_memory_copy"
            elif task_type not in {"AI_CORE", "AI_VECTOR", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "MIX_AIC_AIV", "MIX_AIV_AIC"}:
                reason = "non_compute_task_type"
            if reason:
                excluded.append({"row": row_number, "reason": reason, "source": row})
                continue
            try:
                duration = float(row[duration_column])
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("Duration must be finite and nonnegative (μs)")
                ratios = {}
                for metric in ratio_columns:
                    text = row[metric].strip()
                    if not text or text.lower() in {"n/a", "na", "nan", "none", "-"}:
                        continue
                    value = float(text.rstrip("%"))
                    if text.endswith("%") or ratio_unit == "percent":
                        value /= 100
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid ratio {metric}={text}; check --ratio-unit")
                    ratios[metric] = value
                if not ratios:
                    raise ValueError("No valid ratios on compute kernel")
                max_metric = max(ratios, key=ratios.get)
                matches = [rule for rule in rules if fnmatch.fnmatchcase(operator, rule["operator"])]
                if len(matches) > 1:
                    raise ValueError(f"Overlapping ceiling rules for {operator}")
                rule = matches[0] if matches else None
                metric = rule["metric"] if rule else max_metric
                if metric not in ratios:
                    raise ValueError(f"Missing selected metric: {metric}")
                ceiling = positive_number(rule["ceiling"]) if rule else default_ceiling
                if ceiling > 1:
                    raise ValueError("Ceiling must be <=1")
            except ValueError as exc:
                raise ValueError(f"CSV row {row_number}: {exc}") from exc
            kernels.append({"row": row_number, "operator": operator, "task_type": task_type,
                            "duration_us": duration, "ratios": ratios, "max_metric": max_metric,
                            "max_ratio": ratios[max_metric], "selected_metric": metric,
                            "selected_ratio": ratios[metric], "ceiling": ceiling,
                            "ceiling_source": rule["source"] if rule else "skill 默认经验值",
                            "source": row})
    if not kernels:
        raise ValueError("No compute kernels on selected device; check task types and device ID")
    total_duration = sum(kernel["duration_us"] for kernel in kernels)
    if total_duration <= 0:
        raise ValueError("Selected compute duration is zero; cannot calculate duration weights")
    groups: dict[str, dict] = {}
    for kernel in kernels:
        duration = kernel["duration_us"]
        kernel["dur_ratio"] = duration / total_duration
        kernel["attainment"] = kernel["selected_ratio"] / kernel["ceiling"]
        kernel["contribution"] = kernel["dur_ratio"] * (kernel["attainment"] if duration >= short_kernel_us else 1)
        kernel["priority"] = duration > priority_duration_us and kernel["max_ratio"] < priority_ratio
        group = groups.setdefault(kernel["operator"], {"operator": kernel["operator"], "calls": 0,
                                 "duration_us": 0, "dur_ratio": 0, "contribution": 0, "priority_calls": 0})
        group["calls"] += 1
        for key in ("duration_us", "dur_ratio", "contribution"):
            group[key] += kernel[key]
        group["priority_calls"] += int(kernel["priority"])
    long_contribution = sum(kernel["contribution"] for kernel in kernels if kernel["duration_us"] >= short_kernel_us)
    short_contribution = sum(kernel["contribution"] for kernel in kernels if kernel["duration_us"] < short_kernel_us)
    return {"feature": "hardware_efficiency", "status": "analysis_complete", "validation_scope": validation_scope,
            "source_csv": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "device": str(device), "device_selection": "CSV filter" if device_column else "single-device file asserted by caller",
            "ratio_unit": ratio_unit, "ratio_columns": ratio_columns, "ceiling_rules": rules,
            "parameters": {"default_ceiling": default_ceiling, "short_kernel_us": short_kernel_us,
                           "priority_duration_us": priority_duration_us, "priority_ratio": priority_ratio},
            "compute_duration_us": total_duration, "long_contribution": long_contribution,
            "short_contribution": short_contribution, "weighted_attainment": long_contribution + short_contribution,
            "priority_calls": sum(kernel["priority"] for kernel in kernels), "kernels": kernels,
            "operator_summary": sorted(groups.values(), key=lambda group: -group["duration_us"]),
            "excluded": excluded,
            "note": "计算算子耗时加权的经验达成率；未计通信、空闲和重叠，不等同端到端吞吐或加速比。"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-details", type=Path, required=True)
    parser.add_argument("--device", required=True, help="One device ID; without a device column, input must be one device's file")
    parser.add_argument("--output", type=Path, required=True, help="New .xlsx output path")
    parser.add_argument("--validation-scope", choices=("full_model", "functional_smoke"), required=True)
    parser.add_argument("--ratio-unit", choices=("fraction", "percent"), default="fraction")
    parser.add_argument("--ceilings", type=Path, help="JSON rules: operator glob, metric, ceiling fraction, source")
    parser.add_argument("--default-ceiling", type=float, default=0.8)
    parser.add_argument("--short-kernel-us", type=float, default=10)
    parser.add_argument("--priority-duration-us", type=float, default=20)
    parser.add_argument("--priority-ratio", type=float, default=0.6)
    for column in ("name", "type", "duration", "device"):
        parser.add_argument(f"--{column}-column")
    parser.add_argument("--node", default="node", help="Node.js executable")
    parser.add_argument("--node-modules", type=Path, help="Optional directory containing @oai/artifact-tool")
    parser.add_argument("--preview-dir", type=Path, help="Optional PNG previews for workbook review")
    args = parser.parse_args(argv)
    report_path = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".xlsx":
        parser.error("--output must end in .xlsx")
    if args.output.exists() or report_path.exists():
        parser.error("Output already exists; choose a new run directory")
    try:
        report = analyze_kernels(args.kernel_details, device=args.device, ratio_unit=args.ratio_unit,
                                 default_ceiling=args.default_ceiling, short_kernel_us=args.short_kernel_us,
                                 priority_duration_us=args.priority_duration_us, priority_ratio=args.priority_ratio,
                                 validation_scope=args.validation_scope, rules=load_rules(args.ceilings),
                                 columns={column: getattr(args, f"{column}_column") for column in ("name", "type", "duration", "device")})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report["status"] = "export_pending"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [args.node, str(Path(__file__).with_name("export_hardware_efficiency.mjs")),
                   str(report_path), str(args.output)]
        command.extend([str(args.node_modules) if args.node_modules else "", str(args.preview_dir or "")])
        subprocess.run(command, check=True)
        report.update(status="passed", workbook_path=str(args.output.resolve()))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        if report_path.exists():
            report.update(status="failed", error=str(exc))
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        parser.exit(1, f"Hardware efficiency estimation failed: {exc}\n")
    print(json.dumps({"report": str(report_path), "workbook": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
