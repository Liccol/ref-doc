#!/usr/bin/env python3
"""Build a self-contained PDF report from model-openbox JSON evidence.

The writer intentionally uses only the Python standard library.  It uses the
standard PDF ``STSong-Light`` CID font so Chinese text remains readable on
viewers that implement the Adobe CJK fonts, without requiring reportlab or a
font download on the target NPU host.

Example::

    python generate_openbox_pdf.py \
      --report-dir model_openbox/qwen3-30b-a3b \
      --commands-file model_openbox/qwen3-30b-a3b/logs/launch_commands.txt \
      --output model_openbox/qwen3-30b-a3b/openbox_report.pdf \
      --model /path/to/artifact
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 42
TOP = 800
BOTTOM = 48
FONT_SIZE = 9
LINE_HEIGHT = 13
MAX_CHARS = 92


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def collect_reports(report_dir: Path | None, explicit: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    paths = list(explicit)
    if report_dir:
        paths.extend(sorted(report_dir.rglob("*.json")))
    seen: set[Path] = set()
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        path = path.resolve()
        if path in seen or path.name in {"manifest.json", "environment.json"}:
            continue
        seen.add(path)
        data = _read_json(path)
        if data is not None:
            reports.append((path, data))
    return reports


def _status(value: Any) -> str:
    value = str(value or "unknown").lower()
    return {
        "passed": "PASS",
        "passed_http_only": "PASS (HTTP)",
        "failed": "FAIL",
        "blocked": "BLOCKED",
        "not_run": "NOT RUN",
    }.get(value, value.upper())


def _one_line(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ").strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _output_pairs(data: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    pairs: list[tuple[str, Any, Any]] = []
    if "weight_update_acceptance" in data:
        before_by_endpoint = data.get("baseline_by_endpoint", {})
        after_by_endpoint = data.get("post_update_by_endpoint", {})
    else:
        before_by_endpoint = data.get("baseline_outputs_by_endpoint") or data.get("baseline_by_endpoint")
        after_by_endpoint = data.get("feature_outputs_by_endpoint") or data.get("post_update_by_endpoint")
    if isinstance(before_by_endpoint, dict) or isinstance(after_by_endpoint, dict):
        for endpoint in ("/v1/completions", "/v1/chat/completions"):
            before = (before_by_endpoint or {}).get(endpoint, [])
            after = (after_by_endpoint or {}).get(endpoint, [])
            count = max(len(before) if isinstance(before, list) else 0,
                        len(after) if isinstance(after, list) else 0)
            prompts = data.get("prompts") or []
            for index in range(count):
                prompt = prompts[index] if index < len(prompts) else f"prompt-{index + 1}"
                left = before[index] if isinstance(before, list) and index < len(before) else "<missing>"
                right = after[index] if isinstance(after, list) and index < len(after) else "<missing>"
                pairs.append((f"{endpoint} | {prompt}", left, right))
        return pairs
    if "baseline_outputs" in data or "feature_outputs" in data:
        before = data.get("baseline_outputs") or []
        after = data.get("feature_outputs") or []
    elif "baseline" in data or "post_update" in data:
        before = data.get("baseline") or []
        after = data.get("post_update") or []
    else:
        return pairs
    count = max(len(before) if isinstance(before, list) else 0,
                len(after) if isinstance(after, list) else 0)
    prompts = data.get("prompts") or []
    for index in range(count):
        prompt = prompts[index] if index < len(prompts) else f"prompt-{index + 1}"
        left = before[index] if isinstance(before, list) and index < len(before) else "<missing>"
        right = after[index] if isinstance(after, list) and index < len(after) else "<missing>"
        pairs.append((_one_line(prompt), left, right))
    return pairs


def build_lines(
    reports: list[tuple[Path, dict[str, Any]]],
    commands: str,
    *,
    title: str,
    model: str | None,
) -> list[str]:
    lines = [title, "=" * min(len(title), MAX_CHARS),
             f"生成时间: {_dt.datetime.now().astimezone().isoformat(timespec='seconds')}"]
    if model:
        lines.append(f"模型/权重: {model}")
    statuses = [_status(data.get("status")) for _, data in reports]
    passed = sum(status.startswith("PASS") for status in statuses)
    lines.append(f"验证汇总: {passed}/{len(statuses)} 项通过" if statuses else "验证汇总: 未发现 JSON 验证报告")

    if commands.strip():
        lines.extend(["", "一、实际拉起与验证命令", "-" * 28])
        for raw in commands.splitlines():
            if raw.strip():
                lines.extend(textwrap.wrap(raw.rstrip(), width=MAX_CHARS,
                                           subsequent_indent="    ") or [""])

    lines.extend(["", "二、逐项验证结果", "-" * 28])
    if not reports:
        lines.append("没有可汇总的验证报告。")
    for index, (path, data) in enumerate(reports, 1):
        feature = data.get("feature") or data.get("mode") or path.stem
        lines.extend(["", f"{index}. {feature} [{_status(data.get('status'))}]", f"报告文件: {path}"])
        checks = data.get("checks")
        if isinstance(checks, dict):
            lines.append("检查项:")
            for name, value in checks.items():
                lines.append(f"  [{'PASS' if value is True else 'FAIL'}] {name}: {value}")
        endpoint_checks = data.get("endpoint_checks")
        if isinstance(endpoint_checks, dict):
            lines.append("接口检查:")
            for endpoint, value in endpoint_checks.items():
                lines.append(f"  [{'PASS' if value is True else 'FAIL'}] {endpoint}")
        for key in ("reason", "error", "recovery", "note"):
            if data.get(key):
                lines.extend(textwrap.wrap(f"{key}: {_one_line(data[key])}", width=MAX_CHARS,
                                           subsequent_indent="    "))
        for key in ("launch_command", "verification_command", "service_url",
                    "log", "log_path", "server_log", "client_log", "checkpoint"):
            if data.get(key):
                label = "实际拉起命令" if key == "launch_command" else (
                    "实际验证命令" if key == "verification_command" else key
                )
                lines.extend(textwrap.wrap(f"{label}: {_one_line(data[key])}", width=MAX_CHARS,
                                           subsequent_indent="    "))
        pairs = _output_pairs(data)
        if pairs:
            lines.append("推理结果对比（baseline / 使能特性后）:")
            for prompt, before, after in pairs:
                lines.append(f"  Prompt: {prompt}")
                lines.extend(textwrap.wrap(f"    baseline: {_one_line(before)}", width=MAX_CHARS,
                                           subsequent_indent="      ") or ["      baseline: "])
                lines.extend(textwrap.wrap(f"    使能特性后: {_one_line(after)}", width=MAX_CHARS,
                                           subsequent_indent="      ") or ["      使能特性后: "])
        elif data.get("outputs"):
            lines.append("推理结果:")
            for output in data["outputs"]:
                lines.extend(textwrap.wrap(f"  {_one_line(output)}", width=MAX_CHARS,
                                           subsequent_indent="    "))
    lines.extend(["", "三、结论", "-" * 28,
                  "PASS 表示脚本声明的检查项均满足；FAIL/BLOCKED/NOT RUN 需结合原始日志和源码诊断。",
                  "推理文本仅作为运行证据展示，不替代客户的精度验收。"])
    return lines


def _pdf_string(value: str) -> bytes:
    # Hex strings avoid escaping NUL bytes in UTF-16BE text.
    return ("<" + ("\ufeff" + value).encode("utf-16-be").hex().upper() + ">\n").encode("ascii")


def write_pdf(lines: list[str], output: Path) -> None:
    pages: list[list[str]] = []
    current: list[str] = []
    max_lines = int((TOP - BOTTOM) // LINE_HEIGHT)
    for line in lines:
        if len(current) >= max_lines:
            pages.append(current)
            current = []
        current.append(line)
    if current or not pages:
        pages.append(current)

    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    # Reserve the descendant object first so the Type0 font can reference its
    # actual object number (object numbers must not be guessed in a PDF).
    descendant_placeholder = add(b"")
    font = add(
        f"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light "
        f"/Encoding /UniGB-UCS2-H /DescendantFonts [{descendant_placeholder} 0 R] >>".encode()
    )
    page_ids: list[int] = []
    content_ids: list[int] = []
    for page_lines in pages:
        stream = bytearray()
        stream.extend(b"BT\n/F1 9 Tf\n")
        y = TOP
        for line in page_lines:
            stream.extend(f"1 0 0 1 {LEFT} {y} Tm\n".encode("ascii"))
            stream.extend(_pdf_string(line))
            stream.extend(b" Tj\n")
            y -= LINE_HEIGHT
        stream.extend(b"ET\n")
        content_ids.append(add(f"<< /Length {len(stream)} >>\nstream\n".encode() + bytes(stream) + b"endstream"))
        page_ids.append(0)
    pages_id = add(b"")
    for index, content_id in enumerate(content_ids):
        page_ids[index] = add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] "
        f"/Count {len(page_ids)} >>".encode()
    )
    objects[descendant_placeholder - 1] = (
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> "
        b"/DW 1000 >>"
    )
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    info_id = add(b"<< /Title (Model RL Case Guard Verification Report) /Producer (model-rl-case-guard) >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R /Info {info_id} 0 R >>\n"
               f"startxref\n{xref}\n%%EOF\n".encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a PDF openbox report from JSON evidence")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--commands-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="模型开箱验证报告")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    commands = args.commands_file.read_text(encoding="utf-8", errors="replace") if args.commands_file and args.commands_file.is_file() else ""
    reports = collect_reports(args.report_dir, args.report)
    write_pdf(build_lines(reports, commands, title=args.title, model=args.model), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
