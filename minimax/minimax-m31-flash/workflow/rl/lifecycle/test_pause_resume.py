#!/usr/bin/env python3
"""Validate pause/resume service lifecycle without claiming weight transfer."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
from common.feature_arguments import (
    add_generation_arguments,
    add_report_arguments,
    add_service_arguments,
    create_feature_parser,
)

GENERATION_ENDPOINTS = (
    ("/v1/completions", "inference"),
    ("/v1/chat/completions", "chat_inference"),
)
REQUIRED_CHECKS = (
    "healthy_before_pause",
    "inference_before_pause",
    "chat_inference_before_pause",
    "pause_accepted",
    "resume_accepted",
    "healthy_after_resume",
    "inference_after_resume",
    "chat_inference_after_resume",
)


def request_json_response(
    server_url: str, method: str, endpoint: str,
    payload: dict[str, Any] | None = None, timeout: float = 120,
) -> tuple[int, Any]:
    """Return HTTP status and body, including non-success HTTP responses."""
    request_body = None if payload is None else json.dumps(payload).encode()
    http_request = urllib.request.Request(
        server_url.rstrip("/") + endpoint,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            response_body = response.read()
            return response.status, json.loads(response_body) if response_body else None
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser, model_option="--model", require_model=True)
    add_generation_arguments(
        parser, prompt_input="single",
        default_prompt="Please explain how a distributed inference service pauses and resumes safely during maintenance.",
    )
    add_report_arguments(parser, include_verification_command=False)
    return parser.parse_args(argv)


def build_generation_payloads(arguments: argparse.Namespace) -> dict[str, dict[str, Any]]:
    generation_settings = {
        "model": arguments.model,
        "max_tokens": arguments.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
    }
    return {
        "/v1/completions": {**generation_settings, "prompt": arguments.prompt},
        "/v1/chat/completions": {
            **generation_settings,
            "messages": [{"role": "user", "content": arguments.prompt}],
        },
    }


def check_service_health(server_url: str, report: dict[str, Any], check_name: str) -> None:
    status_code, _ = request_json_response(server_url, "GET", "/health")
    report["checks"][check_name] = status_code == 200


def check_generation_endpoints(
    server_url: str, payloads: dict[str, dict[str, Any]], report: dict[str, Any],
    phase: Literal["before_pause", "after_resume"],
) -> None:
    for endpoint, check_prefix in GENERATION_ENDPOINTS:
        status_code, _ = request_json_response(server_url, "POST", endpoint, payloads[endpoint])
        report["checks"][f"{check_prefix}_{phase}"] = status_code == 200


def pause_service(server_url: str, report: dict[str, Any]) -> None:
    status_code, response_body = request_json_response(server_url, "POST", "/pause")
    report["pause_response"] = {"status": status_code, "body": response_body}
    report["checks"]["pause_accepted"] = status_code == 200

    status_code, pause_state = request_json_response(server_url, "GET", "/is_paused")
    report["pause_state"] = {"status": status_code, "body": pause_state}
    report["checks"]["paused_state_observed"] = (
        status_code == 200
        and isinstance(pause_state, dict)
        and bool(pause_state.get("is_paused", pause_state.get("paused", False)))
    )


def resume_service(server_url: str, report: dict[str, Any]) -> None:
    status_code, response_body = request_json_response(server_url, "POST", "/resume")
    report["resume_response"] = {"status": status_code, "body": response_body}
    report["checks"]["resume_accepted"] = status_code == 200


def run_validation(arguments: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": "pause_resume",
        "server_url": arguments.server_url,
        "checks": {},
    }
    check_service_health(arguments.server_url, report, "healthy_before_pause")
    payloads = build_generation_payloads(arguments)
    check_generation_endpoints(arguments.server_url, payloads, report, "before_pause")
    pause_service(arguments.server_url, report)
    resume_service(arguments.server_url, report)
    check_service_health(arguments.server_url, report, "healthy_after_resume")
    check_generation_endpoints(arguments.server_url, payloads, report, "after_resume")

    report["endpoint_checks"] = {
        endpoint: report["checks"][f"{check_prefix}_before_pause"]
        and report["checks"][f"{check_prefix}_after_resume"]
        for endpoint, check_prefix in GENERATION_ENDPOINTS
    }
    # Pause-state observation retains its existing diagnostic role.
    report["status"] = (
        "passed" if all(report["checks"].get(check_name) for check_name in REQUIRED_CHECKS)
        else "failed"
    )
    report["note"] = (
        "This validates lifecycle control only; it does not prove weights were transferred or changed."
    )
    return report


def write_report(report_path: str, report: dict[str, Any]) -> None:
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = run_validation(arguments)
    write_report(arguments.report, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
