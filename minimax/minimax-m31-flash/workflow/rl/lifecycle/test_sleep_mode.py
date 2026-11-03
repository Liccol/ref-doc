#!/usr/bin/env python3
"""Validate level-1 sleep/wake determinism; optionally measure server-device HBM.

Level 2 discards weights and requires a separate weight-reload/transfer test.
Without a remote memory probe, success is reported as passed_http_only.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
from common.dual_api import ENDPOINTS, generate_endpoint, validate_generation_output
from common.feature_arguments import (
    add_generation_arguments, add_report_arguments, add_service_arguments,
    add_timeout_argument, create_feature_parser, positive_integer, positive_number,
)

DEFAULT_PROMPT = (
    "Please explain in detail how distributed inference systems coordinate "
    "data parallelism, tensor parallelism, expert parallelism, weight updates, "
    "and service recovery while preserving deterministic outputs across repeated validation runs."
)
MAX_MEMORY_GROWTH_RATIO = 0.05


def request_json(server_url: str, method: str, endpoint: str,
                 payload: dict[str, Any] | None = None, *, timeout: float = 600) -> Any:
    request_body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + endpoint, data=request_body,
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {endpoint}: HTTP {error.code}: {detail}") from error
    return json.loads(body) if body.strip() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser, model_option="--model", require_model=True)
    add_generation_arguments(parser, prompt_input="repeat")
    add_report_arguments(parser)
    add_timeout_argument(parser, option="--request-timeout", default=600, value_type=positive_number)
    parser.add_argument("--baseline-result", required=True, help="Baseline JSON; only its prompts are reused")
    parser.add_argument("--cycles", type=positive_integer, default=3)
    parser.add_argument("--poll-interval", type=positive_number, default=2)
    parser.add_argument("--memory-probe-command", default=None,
                        help="Remote serve_probe_npus.py command returning fresh JSON; executed without a shell")
    parser.add_argument("--memory-device", action="append", default=[], dest="memory_devices",
                        help="Inference device to measure as HOST/physical-NPU-ID; repeat for all inference devices")
    arguments = parser.parse_args(argv)
    if bool(arguments.memory_probe_command) != bool(arguments.memory_devices):
        parser.error("--memory-probe-command 和 --memory-device 必须同时提供")
    if len(arguments.memory_devices) != len(set(arguments.memory_devices)):
        parser.error("--memory-device 不得重复")
    for identity in arguments.memory_devices:
        host, separator, device = identity.rpartition("/")
        if not separator or not host or not device.isdigit():
            parser.error("--memory-device 格式必须为 HOST/physical-NPU-ID，例如 192.168.1.100/0")
    return arguments


def load_validation_prompts(arguments: argparse.Namespace) -> list[str]:
    saved = json.loads(Path(arguments.baseline_result).read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("baseline-result 必须为 JSON 对象")
    prompts = saved.get("prompts") or arguments.prompts or [DEFAULT_PROMPT]
    if not isinstance(prompts, list) or not prompts or any(
        not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
    ):
        raise ValueError("prompts 必须是非空字符串数组")
    return prompts


def collect_generation_outputs(arguments: argparse.Namespace, prompts: list[str],
                               outputs_by_endpoint: dict[str, list[dict[str, Any]]]) -> None:
    """Append each successful response before requesting the next."""
    for prompt in prompts:
        for endpoint in ENDPOINTS:
            output = generate_endpoint(
                request_json, arguments.server_url, arguments.model, prompt,
                arguments.max_tokens, endpoint=endpoint, timeout=arguments.request_timeout,
            )
            outputs_by_endpoint[endpoint].append(output)


def read_sleep_state(arguments: argparse.Namespace, *, timeout: float | None = None) -> bool:
    state = request_json(arguments.server_url, "GET", "/is_sleeping",
                         timeout=arguments.request_timeout if timeout is None else timeout)
    if not isinstance(state, dict) or type(state.get("is_sleeping")) is not bool:
        raise RuntimeError("/is_sleeping 未返回有效的布尔状态")
    return state["is_sleeping"]


def wait_until_awake(arguments: argparse.Namespace) -> None:
    deadline = time.monotonic() + arguments.request_timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"service remained sleeping after {arguments.request_timeout:g}s")
        if not read_sleep_state(arguments, timeout=min(30, remaining)):
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(arguments.poll_interval, remaining))


def collect_server_memory(arguments: argparse.Namespace) -> dict[str, dict[str, int]] | None:
    """Read selected physical devices from the serving probe, never from this client."""
    if not arguments.memory_probe_command:
        return None
    command = shlex.split(arguments.memory_probe_command)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            timeout=arguments.request_timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"远端内存采集失败，退出码 {result.returncode}: {result.stderr[-1000:]}")
    # serve_probe_npus.py emits one indented JSON object on stdout; progress
    # messages go to stderr. Parsing one line would lose the actual probe output.
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or data.get("status") != "ok":
        raise ValueError("内存采集未返回 status=ok 的 serve_probe_npus JSON")
    nodes = data.get("nodes") if "nodes" in data else {data.get("host"): data}
    if not isinstance(nodes, dict):
        raise ValueError("内存采集的 nodes 必须为对象")
    snapshot = {}
    for identity in arguments.memory_devices:
        host, separator, device = identity.rpartition("/")
        if not separator or not host or not device:
            raise ValueError(f"内存设备必须为 HOST/physical-NPU-ID：{identity}")
        reading = nodes.get(host, {}).get("hbm", {}).get(device)
        if not isinstance(reading, dict):
            raise ValueError(f"远端内存采集缺少推理设备 {identity}")
        used, total = reading.get("used_mb"), reading.get("total_mb")
        if type(used) is not int or type(total) is not int or not 0 <= used <= total or total <= 0:
            raise ValueError(f"推理设备 {identity} 的 HBM 数据无效：{reading}")
        snapshot[identity] = {"used_bytes": used * 1024**2, "total_bytes": total * 1024**2}
    return snapshot


def compare_memory(before: dict, after: dict) -> dict[str, int]:
    if before.keys() != after.keys() or any(
        before[device]["total_bytes"] != after[device]["total_bytes"] for device in before
    ):
        raise ValueError("内存采集前后的设备或 HBM 总容量发生变化，无法比较")
    return {device: after[device]["used_bytes"] - before[device]["used_bytes"] for device in before}


def compare_cycle_outputs(baseline: dict, prompts: list[str], cycle: dict) -> None:
    for endpoint in ENDPOINTS:
        before, after = baseline[endpoint], cycle["outputs_by_endpoint"][endpoint]
        valid_count = len(before) == len(after) == len(prompts)
        matches = []
        if valid_count:
            for left, right in zip(before, after):
                validate_generation_output(endpoint, left)
                validate_generation_output(endpoint, right)
                matches.append(left == right)
        cycle["outputs_identical_by_endpoint"][endpoint] = matches
        cycle["endpoint_checks"][endpoint] = valid_count and all(matches)
    if not all(cycle["endpoint_checks"].values()):
        raise RuntimeError(f"cycle {cycle['cycle']} 双接口 text/token/logprob 与基线不一致")


def run_sleep_cycle(arguments: argparse.Namespace, prompts: list[str], baseline: dict,
                    memory_before: dict | None, cycle: dict) -> None:
    checks = cycle["checks"]
    cycle["phase"] = "sleep"
    cycle["sleep_attempted"] = True  # A timed-out POST may already have put the service to sleep.
    request_json(arguments.server_url, "POST", "/sleep?level=1&mode=abort", timeout=arguments.request_timeout)
    cycle["sleep_state_after_sleep"] = read_sleep_state(arguments)
    checks["sleep_state_confirmed"] = cycle["sleep_state_after_sleep"] is True
    if not checks["sleep_state_confirmed"]:
        raise RuntimeError("sleep 返回成功，但服务未进入休眠状态")
    cycle["phase"] = "memory_after_sleep"
    cycle["memory_after_sleep"] = collect_server_memory(arguments)
    if memory_before is not None:
        deltas = compare_memory(memory_before, cycle["memory_after_sleep"])
        cycle["released_bytes_by_device"] = {device: -delta for device, delta in deltas.items()}
        checks["memory_released"] = all(delta < 0 for delta in deltas.values())
    cycle["phase"] = "wake_weights"
    request_json(arguments.server_url, "POST", "/wake_up?tags=weights", timeout=arguments.request_timeout)
    cycle["sleep_state_after_weights"] = read_sleep_state(arguments)
    checks["partial_wake_still_sleeping"] = cycle["sleep_state_after_weights"] is True
    if not checks["partial_wake_still_sleeping"]:
        raise RuntimeError("仅唤醒 weights 后服务提前退出休眠，KV cache 分阶段唤醒未被证明")
    cycle["phase"] = "wake_kv_cache"
    request_json(arguments.server_url, "POST", "/wake_up?tags=kv_cache", timeout=arguments.request_timeout)
    wait_until_awake(arguments)
    checks["fully_awake"] = True
    cycle["phase"] = "health_after_wake"
    request_json(arguments.server_url, "GET", "/health", timeout=arguments.request_timeout)
    checks["healthy_after_wake"] = True
    cycle["phase"] = "generation_after_wake"
    collect_generation_outputs(arguments, prompts, cycle["outputs_by_endpoint"])
    cycle["phase"] = "memory_after_generation"
    cycle["memory_after_generation"] = collect_server_memory(arguments)
    cycle["phase"] = "compare_outputs"
    compare_cycle_outputs(baseline, prompts, cycle)
    cycle["phase"] = "memory_acceptance"
    if checks["memory_released"] is False:
        raise RuntimeError("sleep 后至少一张推理设备未观察到 HBM 释放")
    cycle["phase"] = "complete"
    cycle["status"] = "passed"


def record_memory_result(report: dict, memory_enabled: bool) -> None:
    if not memory_enabled:
        report["memory_validation"] = {"status": "not_run", "reason": "未配置远端内存探测；不验证显存释放或泄漏"}
        return
    initial = report["memory_before"]
    cycles = report["cycles"]
    if not initial or any(cycle.get("memory_after_generation") is None for cycle in cycles):
        report["memory_validation"] = {"status": "failed", "reason": "远端内存证据不完整"}
        return
    growth = {device: 0.0 for device in initial}
    for cycle in cycles:
        deltas = compare_memory(initial, cycle["memory_after_generation"])
        for device, delta in deltas.items():
            if initial[device]["used_bytes"] <= 0:
                raise ValueError(f"推理设备 {device} 的基线 HBM 占用为零，无法判断内存增长")
            growth[device] = max(growth[device], delta / initial[device]["used_bytes"])
    report["memory_growth_by_device"] = growth
    report["memory_growth_ratio"] = max(growth.values())
    complete = len(cycles) == report["requested_cycles"] and all(cycle["status"] == "passed" for cycle in cycles)
    released = complete and all(cycle["checks"]["memory_released"] is True for cycle in cycles)
    within_limit = complete and report["memory_growth_ratio"] <= MAX_MEMORY_GROWTH_RATIO
    report["checks"].update(memory_released=released, memory_growth_within_5pct=within_limit)
    report["memory_validation"] = {"status": "passed" if released and within_limit else "failed",
                                   "source": "remote npu-smi device HBM", "devices": list(initial)}


def record_service_state_after_failure(arguments: argparse.Namespace, report: dict) -> None:
    if not any(cycle.get("sleep_attempted") for cycle in report["cycles"]):
        return
    try:
        sleeping = read_sleep_state(arguments, timeout=min(5, arguments.request_timeout))
        report["server_left_sleeping"] = sleeping
        if sleeping:
            report["recovery"] = "level-1 流程未完成，服务仍在休眠；请检查失败阶段及剩余 tags 后恢复或重启服务。"
    except Exception as error:
        report["server_state_unknown"] = True
        report["state_query_error"] = str(error)
        report["recovery"] = "无法确认服务休眠状态；保留服务日志，检查后恢复或重启服务。"


def run_validation(arguments: argparse.Namespace, verification_command: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": "sleep_mode", "status": "failed", "level": 1,
        "server_url": arguments.server_url, "model": arguments.model,
        "requested_cycles": arguments.cycles, "verification_command": verification_command,
        "baseline_outputs_by_endpoint": {endpoint: [] for endpoint in ENDPOINTS},
        "feature_outputs_by_endpoint": {endpoint: [] for endpoint in ENDPOINTS},
        "cycles": [], "memory_before": None, "memory_growth_ratio": None,
        "checks": {"cycles_complete": False, "output_tokens_logprobs_identical": None,
                   "both_generation_endpoints": None, "memory_released": None,
                   "memory_growth_within_5pct": None},
        "endpoint_checks": {endpoint: None for endpoint in ENDPOINTS},
        "memory_validation": {"status": "not_run", "reason": "尚未完成采集"},
    }
    try:
        report["phase"] = "load_prompts"
        prompts = load_validation_prompts(arguments)
        report["prompts"] = prompts
        report["phase"] = "baseline"
        if read_sleep_state(arguments):
            raise RuntimeError("服务初始处于休眠状态，不能采集推理基线")
        request_json(arguments.server_url, "GET", "/health", timeout=arguments.request_timeout)
        collect_generation_outputs(arguments, prompts, report["baseline_outputs_by_endpoint"])
        report["phase"] = "memory_baseline"
        report["memory_before"] = collect_server_memory(arguments)
        awake_memory = report["memory_before"]
        for number in range(1, arguments.cycles + 1):
            cycle = {"cycle": number, "status": "failed", "checks": {"memory_released": None},
                     "memory_before_sleep": awake_memory,
                     "outputs_by_endpoint": {endpoint: [] for endpoint in ENDPOINTS},
                     "outputs_identical_by_endpoint": {},
                     "endpoint_checks": {endpoint: None for endpoint in ENDPOINTS}}
            report["cycles"].append(cycle)
            report["feature_outputs_by_endpoint"] = cycle["outputs_by_endpoint"]
            report["phase"] = "sleep_cycle"
            run_sleep_cycle(arguments, prompts, report["baseline_outputs_by_endpoint"], awake_memory, cycle)
            awake_memory = cycle["memory_after_generation"]
        report["checks"]["cycles_complete"] = True
        report["phase"] = "memory_acceptance"
        record_memory_result(report, bool(arguments.memory_probe_command))
        if arguments.memory_probe_command and report["memory_validation"]["status"] != "passed":
            raise RuntimeError("远端设备 HBM 释放/增长检查未通过")
        report["status"] = "passed" if arguments.memory_probe_command else "passed_http_only"
        report["phase"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if report["phase"] == "sleep_cycle":
            report["failed_phase"] = report["cycles"][-1].get("phase", report["phase"])
        else:
            report["failed_phase"] = report["phase"]
        if arguments.memory_probe_command:
            report["memory_validation"] = {"status": "failed", "reason": "验证中断，不能给出完整内存验收结论"}
            if any(cycle["checks"]["memory_released"] is False for cycle in report["cycles"]):
                report["checks"]["memory_released"] = False
        record_service_state_after_failure(arguments, report)
    if not arguments.memory_probe_command:
        record_memory_result(report, False)
        report["note"] = "仅验证 HTTP 生命周期和双接口一致性；未采集服务端设备 HBM，不代表显存释放或无泄漏。"
    for endpoint in ENDPOINTS:
        results = [cycle["endpoint_checks"][endpoint] for cycle in report["cycles"]]
        report["endpoint_checks"][endpoint] = (
            False if False in results else
            (True if len(results) == arguments.cycles and all(value is True for value in results) else None)
        )
    endpoint_results = list(report["endpoint_checks"].values())
    identical = False if False in endpoint_results else (True if all(value is True for value in endpoint_results) else None)
    report["checks"].update(output_tokens_logprobs_identical=identical, both_generation_endpoints=identical)
    return report


def write_report(report_path: str, report: dict[str, Any]) -> None:
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[report] {output}: {report['status']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    command = arguments.verification_command or shlex.join([sys.executable, *sys.argv])
    print(f"[verification command] {command}", flush=True)
    report = run_validation(arguments, command)
    write_report(arguments.report, report)
    return 0 if report["status"] in {"passed", "passed_http_only"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
