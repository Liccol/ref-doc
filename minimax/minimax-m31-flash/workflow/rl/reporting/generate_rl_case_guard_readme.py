#!/usr/bin/env python3
"""Generate a readable Markdown README from model-rl-case-guard JSON evidence.

The generated document is intentionally plain Markdown: it renders well in
GitHub, GitLab, VS Code and the Codex file preview without a PDF font/runtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.checkpoint_provenance import METADATA_FILENAME, load_checkpoint_provenance

ENDPOINTS = ("/v1/completions", "/v1/chat/completions")
STATUS_LABELS = {
    "passed": ("✅", "passed"),
    "passed_http_only": ("🟡", "passed (HTTP only)"),
    "failed": ("❌", "failed"),
    "blocked": ("⛔", "blocked"),
    "not_run": ("⏭️", "not run"),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def collect_reports(report_dir: Path | None, explicit: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    paths = list(explicit)
    if report_dir:
        paths.extend(sorted(report_dir.rglob("*.json")))
    reports: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or path.name in {"manifest.json", "environment.json", METADATA_FILENAME}:
            continue
        # Generated checkpoint config/index/metadata are assets, not test reports.
        if any((parent / METADATA_FILENAME).is_file()
               and (parent / "config.json").is_file()
               and any(parent.glob("*.safetensors")) for parent in path.parents):
            continue
        seen.add(path)
        data = _read_json(path)
        if data is not None:
            reports.append((path, data))
    return reports


def _status(value: Any) -> tuple[str, str]:
    return STATUS_LABELS.get(str(value or "unknown").lower(), ("❔", str(value or "unknown")))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ").strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _inline(value: Any) -> str:
    return _text(value).replace("`", "\\`").replace("|", "\\|")


def _code(value: Any) -> str:
    return str(value or "").replace("```", "``\\`")


def _output_pairs(data: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    if "weight_update_acceptance" in data:
        before = data.get("baseline_by_endpoint", {})
        after = data.get("post_update_by_endpoint", {})
    else:
        before = data.get("baseline_outputs_by_endpoint") or data.get("baseline_by_endpoint")
        after = data.get("feature_outputs_by_endpoint") or data.get("post_update_by_endpoint")
    if not isinstance(before, dict) and not isinstance(after, dict):
        before = data.get("baseline_outputs") or data.get("baseline") or []
        after = data.get("feature_outputs") or data.get("post_update") or []
        prompts = data.get("prompts") or []
        count = max(len(before) if isinstance(before, list) else 0,
                    len(after) if isinstance(after, list) else 0)
        return [
            (_text(prompts[i] if i < len(prompts) else f"prompt-{i + 1}"),
             before[i] if isinstance(before, list) and i < len(before) else "<missing>",
             after[i] if isinstance(after, list) and i < len(after) else "<missing>")
            for i in range(count)
        ]
    pairs: list[tuple[str, Any, Any]] = []
    prompts = data.get("prompts") or []
    for endpoint in ENDPOINTS:
        left = before.get(endpoint, []) if isinstance(before, dict) else []
        right = after.get(endpoint, []) if isinstance(after, dict) else []
        count = max(len(left) if isinstance(left, list) else 0,
                    len(right) if isinstance(right, list) else 0)
        for i in range(count):
            output = left[i] if i < len(left) else right[i]
            prompt_index = output.get("prompt_index", i) if isinstance(output, dict) else i
            prompt = prompts[prompt_index] if isinstance(prompt_index, int) and 0 <= prompt_index < len(prompts) else f"prompt-{i + 1}"
            replica = f" · DP {output['dp_rank']}" if isinstance(output, dict) and "dp_rank" in output else ""
            pairs.append((f"{endpoint}{replica} · {_text(prompt)}",
                          left[i] if isinstance(left, list) and i < len(left) else "<missing>",
                          right[i] if isinstance(right, list) and i < len(right) else "<missing>"))
    return pairs


def _details(lines: list[str], title: str, value: Any) -> None:
    if value:
        lines.extend(["", "<details>", f"<summary>{title}</summary>", "", "```text", _code(value), "```", "", "</details>"])


def _relative_path(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir)).as_posix()


def append_launch_context(lines: list[str], task_manifest: dict[str, Any]) -> None:
    """Show prerequisite PRs and the shared baseline configuration as evidence."""
    changes = task_manifest.get("environment_changes")
    if isinstance(changes, dict):
        assessment = {"required": "需要前置修改", "not_required": "无需前置修改", "unknown": "待核实"}
        lines.extend(["", "## 环境前置修改", "",
                      f"核查结论：{assessment.get(changes.get('assessment'), '待核实')}"])
        items = changes.get("items", [])
        if items:
            lines.extend(["", "| 仓库/组件 | PR 号与链接 | 原因/适用范围 | 目标版本 | 状态 | 操作 | 节点 | 证据 |",
                          "|---|---|---|---|---|---|---|---|"])
            fields = ("component", "pr", "reason", "target_revision", "status", "action", "nodes", "evidence")
            for item in items:
                lines.append("| " + " | ".join(_inline(item.get(field, "")) for field in fields) + " |")
        elif changes.get("assessment") != "not_required":
            lines.extend(["", "尚未提供修改明细；不能据此认定必要前置项已满足。"])
    special_config = task_manifest.get("baseline_special_config")
    if isinstance(special_config, dict):
        confirmation = "已明确" if special_config.get("confirmation") == "confirmed" else "待补充"
        lines.extend(["", "## Baseline 特殊配置", "", f"用户答复：{confirmation}；配置版本：`{_inline(special_config.get('revision'))}`。",
                      "", "以下公共配置应由后续特性继承；实际继承情况以各特性的命令和生效证据为准。",
                      "", "```json", _code(json.dumps(special_config, ensure_ascii=False, indent=2)), "```"])


def build_readme(
    reports: list[tuple[Path, dict[str, Any]]],
    commands: str,
    *,
    title: str,
    model: str | None,
    output_dir: Path,
    checkpoint_provenance: dict[str, Any] | None = None,
    task_manifest: dict[str, Any] | None = None,
) -> str:
    statuses = [_status(data.get("status")) for _, data in reports]
    passed = sum(label.startswith("passed") for _, label in statuses)
    lines = [f"# {title}", "", "> 自动汇总的模型 RL 用例看护证据。原始日志、JSON 报告和启动命令应与本文件一起归档。", ""]
    lines.extend(["## Summary", "", f"- **Generated:** {dt.datetime.now().astimezone().isoformat(timespec='seconds')}"])
    if model:
        lines.append(f"- **Model / checkpoint:** `{_inline(model)}`")
    provenance_entries = [checkpoint_provenance] if checkpoint_provenance else []
    for _, data in reports:
        provenance = data.get("checkpoint_provenance")
        if provenance and provenance not in provenance_entries:
            provenance_entries.append(provenance)
    if provenance_entries:
        lines.extend(["- **Validation scope:** 包含减层随机权重功能验证（functional_smoke）",
                      "- **结论边界：** 仅覆盖实际执行的功能路径，不能代表真实权重精度、完整层数行为或原模型性能。"])
        for provenance in provenance_entries:
            lines.append(f"- **Checkpoint source:** `{_inline(provenance.get('source_config'))}`; "
                         f"layers: {provenance.get('original_layers')} → {provenance.get('reduced_layers')}; "
                         f"seed: {provenance.get('seed')}; config SHA256: `{_inline(provenance.get('source_config_sha256'))}`")
    lines.append(f"- **Result:** **{passed}/{len(statuses)}** report(s) passed" if statuses else "- **Result:** no JSON reports found")
    lines.extend(["", "| Feature | Status | Evidence |", "|---|---|---|"])
    if not reports:
        lines.append("| — | ❔ unknown | No JSON report found |")
    for path, data in reports:
        icon, label = _status(data.get("status"))
        feature = data.get("feature") or data.get("mode") or path.stem
        relative = _relative_path(path, output_dir)
        lines.append(f"| `{_inline(feature)}` | {icon} **{label}** | [`{relative}`]({relative}) |")

    if task_manifest:
        append_launch_context(lines, task_manifest)

    if commands.strip():
        lines.extend(["", "## Commands", "", "The following commands were recorded in `logs/launch_commands.txt`:", "", "```bash", _code(commands.rstrip()), "```"])

    lines.extend(["", "## Feature evidence"])
    for index, (path, data) in enumerate(reports, 1):
        icon, label = _status(data.get("status"))
        feature = data.get("feature") or data.get("mode") or path.stem
        relative = _relative_path(path, output_dir)
        lines.extend(["", f"### {index}. {icon} {feature} — {label}", "", f"Evidence file: [`{relative}`]({relative})"])
        if checkpoint_provenance or data.get("checkpoint_provenance"):
            lines.append("- **Scope:** 减层随机权重功能验证；通过结论仅适用于报告中的 checkpoint。")
        if "baseline_config_revision" in data:
            lines.append(f"- **Baseline 配置版本：** `{_inline(data['baseline_config_revision'])}`")
        for key, heading in (("inherited_special_config", "继承的特殊配置"), ("config_differences", "本特性配置差异")):
            if key in data:
                _details(lines, heading, json.dumps(data[key], ensure_ascii=False, indent=2))
        checks = data.get("checks")
        if isinstance(checks, dict):
            lines.extend(["", "#### Checks", "", "| Check | Result |", "|---|---|"])
            for name, value in checks.items():
                result = ("✅ pass" if value is True else "❌ fail" if value is False
                          else "⏭️ not_run" if value is None else f"`{_inline(value)}`")
                lines.append(f"| `{_inline(name)}` | {result} |")
        endpoint_checks = data.get("endpoint_checks")
        if isinstance(endpoint_checks, dict):
            lines.extend(["", "#### Endpoint checks", "", "| Endpoint | Result |", "|---|---|"])
            for endpoint, value in endpoint_checks.items():
                result = "✅ pass" if value is True else "⏭️ not_run" if value is None else "❌ fail"
                lines.append(f"| `{_inline(endpoint)}` | {result} |")
        if data.get("mode") == "sleep_mode":
            _details(lines, "Sleep 内存验证", json.dumps(data.get("memory_validation"), ensure_ascii=False, indent=2))
            _details(lines, "Sleep 各轮状态与输出", json.dumps(data.get("cycles", []), ensure_ascii=False, indent=2))
        if data.get("cluster"):
            _details(lines, "权重更新集群拓扑", json.dumps(data["cluster"], ensure_ascii=False, indent=2))
            _details(lines, "逐 DP 副本一致性", json.dumps(data.get("outputs_identical_by_dp_rank"), ensure_ascii=False, indent=2))
        if data.get("producer_nodes"):
            _details(lines, "全节点 IPC producer 映射", json.dumps(data["producer_nodes"], ensure_ascii=False, indent=2))
        if data.get("feature") == "hardware_efficiency":
            score = data.get("weighted_attainment")
            if isinstance(score, (float, int)):
                lines.append(f"- **计算算子加权极致性能达成率：** {score:.4%}（经验估算）")
            lines.append(f"- **设备 / 范围：** `{_inline(data.get('device'))}` / `{_inline(data.get('validation_scope'))}`")
            lines.append(f"- **重点优化调用数：** `{_inline(data.get('priority_calls'))}`")
            _details(lines, "测算参数", json.dumps(data.get("parameters"), ensure_ascii=False, indent=2))
        if data.get("feature") == "dataset_evaluation":
            lines.append(f"- **数据集 / 输出长度：** `{_inline(data.get('dataset'))}` / `{_inline(data.get('max_out_len'))}`")
            _details(lines, "评测指标", json.dumps(data.get("metrics"), ensure_ascii=False, indent=2))
        for key, heading in (("workbook_path", "性能测算 Excel"), ("source_csv", "单卡 kernel_details"),
                             ("result_path", "原始评测结果"), ("profiling_manifest", "Profiling 产物")):
            if data.get(key):
                artifact_path = Path(data[key])
                if artifact_path.is_absolute():
                    artifact_path = Path(_relative_path(artifact_path, output_dir))
                target = artifact_path.as_posix().replace(" ", "%20").replace("(", "%28").replace(")", "%29")
                lines.append(f"- **{heading}：** [{_inline(artifact_path.as_posix())}]({target})")
        for key, heading in (("checkpoint", "Checkpoint"), ("service_url", "Service"), ("log", "Log"), ("log_path", "Log path"), ("server_log", "Server log"), ("client_log", "Client log")):
            if data.get(key):
                lines.append(f"- **{heading}:** `{_inline(data[key])}`")
        for key, heading in (("reason", "Reason"), ("error", "Error"), ("recovery", "Recovery"), ("note", "Note")):
            if data.get(key):
                lines.append(f"- **{heading}:** {_inline(data[key])}")
        _details(lines, "Launch command", data.get("launch_command"))
        _details(lines, "Verification command", data.get("verification_command"))
        pairs = _output_pairs(data)
        if pairs:
            lines.extend(["", "#### Output comparison", "", "| Endpoint / prompt | Baseline | After update |", "|---|---|---|"])
            for prompt, before, after in pairs:
                lines.append(f"| {_inline(prompt)} | {_inline(before)} | {_inline(after)} |")
        elif data.get("outputs"):
            lines.extend(["", "#### Outputs", ""])
            lines.extend(f"- {_inline(item)}" for item in data["outputs"])

    lines.extend(["", "## Interpretation", "", "- `passed` means the recorded checks completed; it is not a substitute for customer precision acceptance.", "- `failed`, `blocked` and `not_run` must be reviewed together with the original logs and environment evidence.", "- Text output is runtime evidence only; it does not prove numerical equivalence.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown README from model-rl-case-guard JSON evidence")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--commands-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="模型 RL 用例看护报告")
    parser.add_argument("--model", default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="任务 manifest.json；默认读取 --report-dir/manifest.json，展示环境修改与 baseline 特殊配置")
    parser.add_argument("--checkpoint-metadata", type=Path, default=None,
                        help="减层权重的 reduced_checkpoint.json；checkpoint 不在本机时传入归档副本")
    args = parser.parse_args(argv)
    manifest_path = args.manifest or (args.report_dir / "manifest.json" if args.report_dir else None)
    task_manifest = None
    if manifest_path and (args.manifest or manifest_path.exists()):
        task_manifest = _read_json(manifest_path)
        if task_manifest is None:
            parser.error(f"无法读取有效的任务 manifest：{manifest_path}")
    commands = args.commands_file.read_text(encoding="utf-8", errors="replace") if args.commands_file and args.commands_file.is_file() else ""
    provenance = None
    if args.checkpoint_metadata:
        if args.checkpoint_metadata.name != METADATA_FILENAME or not args.checkpoint_metadata.is_file():
            parser.error(f"--checkpoint-metadata 必须指向已存在的 {METADATA_FILENAME}")
        provenance = load_checkpoint_provenance(args.checkpoint_metadata)
    elif args.model:
        provenance = load_checkpoint_provenance(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = collect_reports(args.report_dir, args.report)
    if manifest_path:
        reports = [(path, data) for path, data in reports if path != manifest_path.resolve()]
    content = build_readme(
        reports,
        commands,
        title=args.title,
        model=args.model,
        output_dir=args.output.parent.resolve(),
        checkpoint_provenance=provenance,
        task_manifest=task_manifest,
    )
    args.output.write_text(content + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
