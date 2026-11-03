"""Shared service lifecycle and output acceptance for weight-update verifiers.

Backend initialization, weight transfer and distributed synchronization belong
to each verifier. These helpers run on the coordinator and report both APIs.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from common.dual_api import ENDPOINTS, generate_both, validate_generation_output
from common.checkpoint_provenance import annotate_weight_update_report


def request_json(
    server_url: str, method: str, endpoint: str,
    payload: dict[str, Any] | None = None, timeout: float = 1800,
    *, headers: dict[str, str] | None = None,
) -> Any:
    """Send a JSON request and include endpoint details in transport errors."""
    request_body = None if payload is None else json.dumps(payload).encode("utf-8")
    http_request = urllib.request.Request(
        server_url.rstrip("/") + endpoint, data=request_body,
        headers={"Content-Type": "application/json", **(headers or {})}, method=method,
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            response_body = response.read()
            return json.loads(response_body) if response_body else None
    except urllib.error.HTTPError as error:
        error_detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {endpoint}: HTTP {error.code}: {error_detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{method} {endpoint}: {error.reason!r}") from error


def load_prompts(prompts_path: str) -> list[str]:
    prompt_data = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    prompts = prompt_data.get("prompts") if isinstance(prompt_data, dict) else None
    if not isinstance(prompts, list) or not prompts or any(
        not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
    ):
        raise ValueError("prompts-json 必须包含非空字符串数组 prompts")
    return prompts


def write_validation_report(report_path: str, report: dict[str, Any]) -> None:
    try:
        annotate_weight_update_report(report)
    except (OSError, ValueError) as error:
        report["status"] = "failed"
        report["provenance_error"] = str(error)
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"[report] {output_path}: {report['status']}", flush=True)


class WeightUpdateValidation:
    """Record a checkpoint baseline and verify idempotent service updates."""

    def __init__(
        self, server_url: str, served_model_name: str, prompts: list[str],
        max_tokens: int, timeout: float, report: dict[str, Any],
        *, request_function: Callable[..., Any] | None = None,
    ) -> None:
        self.server_url = server_url
        self.served_model_name = served_model_name
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.report = report
        self.request_function = request_function or request_json
        self.sleep_attempted = False
        self.data_parallel_size = 1
        report["prompts"] = prompts
        report.setdefault("checks", {})

    def request_service_json(self, method: str, endpoint: str) -> Any:
        return self.request_function(
            self.server_url, method, endpoint, timeout=self.timeout,
        )

    def resolve_served_model_name(self) -> str:
        if self.served_model_name == "auto":
            models = self.request_service_json("GET", "/v1/models")["data"]
            if len(models) != 1:
                raise ValueError("服务模型不唯一，请指定 --served-model-name")
            self.served_model_name = models[0]["id"]
        self.report["served_model_name"] = self.served_model_name
        return self.served_model_name

    def get_inference_worker_count(self) -> int:
        worker_count = self.request_service_json("GET", "/get_world_size?include_dp=true")["world_size"]
        if type(worker_count) is not int or worker_count < 1:
            raise ValueError("推理 worker 数必须大于 0")
        self.report["inference_world_size"] = worker_count
        return worker_count

    def configure_cluster(self, arguments: Any) -> int:
        """Verify global topology before sleep and explicitly probe every DP replica."""
        worker_count = self.get_inference_worker_count()
        group_size = self.request_service_json("GET", "/get_world_size?include_dp=false")["world_size"]
        if type(group_size) is not int or group_size < 1 or worker_count % group_size:
            raise ValueError("服务全局 world size 必须是单 DP 的 TP×PP world size 的整数倍")
        self.data_parallel_size = worker_count // group_size
        if arguments.expected_world_size is not None and arguments.expected_world_size != worker_count:
            raise ValueError(f"服务 worker 数 {worker_count} 与预期 {arguments.expected_world_size} 不一致")
        if arguments.data_parallel_size is not None and arguments.data_parallel_size != self.data_parallel_size:
            raise ValueError(f"服务 DP={self.data_parallel_size} 与预期 DP={arguments.data_parallel_size} 不一致")
        nodes = []
        if arguments.inference_nodes is not None:
            nodes = [node.strip() for node in arguments.inference_nodes.split(",")]
            if any(not node for node in nodes) or len(set(nodes)) != len(nodes):
                raise ValueError("--inference-nodes 必须包含非空且不重复的节点地址")
            if len(nodes) > worker_count:
                raise ValueError("推理节点数不能超过推理 worker 总数")
        self.report["cluster"] = {
            "inference_nodes": nodes, "world_size": worker_count,
            "workers_per_dp": group_size, "data_parallel_size": self.data_parallel_size,
            "control_server_url": self.server_url,
            "control_scope": "all_workers_via_collective_rpc",
            "generation_routing": "X-data-parallel-rank" if self.data_parallel_size > 1 else "single_replica",
        }
        self.report["checks"]["cluster_topology"] = True
        return worker_count

    def collect_generation_outputs(self, report_key: str) -> dict[str, list[dict[str, Any]]]:
        outputs = {endpoint: [] for endpoint in ENDPOINTS}
        self.report[report_key] = outputs
        for dp_rank in range(self.data_parallel_size):
            def request_replica(*args: Any, **kwargs: Any) -> Any:
                if self.data_parallel_size > 1:
                    kwargs["headers"] = {"X-data-parallel-rank": str(dp_rank)}
                return self.request_function(*args, **kwargs)

            for prompt_index, prompt in enumerate(self.prompts):
                generated = generate_both(
                    request_replica, self.server_url, self.served_model_name,
                    prompt, self.max_tokens, timeout=self.timeout,
                )
                for endpoint in ENDPOINTS:
                    outputs[endpoint].append({"dp_rank": dp_rank, "prompt_index": prompt_index,
                                              **generated[endpoint]})
        return outputs

    def record_baseline(self) -> None:
        self.resolve_served_model_name()
        self.collect_generation_outputs("baseline_by_endpoint")
        self.report["checks"]["both_generation_endpoints_baseline"] = True

    def sleep_service(self) -> None:
        # A timed-out request may already have discarded the service weights.
        self.sleep_attempted = True
        self.request_service_json("POST", "/sleep?level=2&mode=abort")
        sleep_state = self.request_service_json("GET", "/is_sleeping")
        if not isinstance(sleep_state, dict) or sleep_state.get("is_sleeping") is not True:
            raise RuntimeError("/sleep 返回成功，但未确认服务已进入休眠状态")
        self.report["checks"]["sleep_level2"] = True

    def wake_weight_buffers(self) -> None:
        self.request_service_json("POST", "/wake_up?tags=weights")
        self.report["checks"]["wake_weights"] = True

    def wake_kv_cache(self) -> None:
        self.request_service_json("POST", "/wake_up?tags=kv_cache")
        self.report["checks"]["wake_kv_cache"] = True

    def record_incomplete_update(self) -> None:
        """Record unsafe service state without waking partially updated weights."""
        if self.sleep_attempted and not self.report["checks"].get("wake_kv_cache"):
            state_key = (
                "server_left_sleeping" if self.report["checks"].get("sleep_level2")
                else "server_state_unknown"
            )
            self.report[state_key] = True
            self.report["recovery"] = (
                "level-2 权重更新流程未完成，服务可能仍休眠且权重可能不完整；"
                "请重启推理服务后重试。"
            )

    def verify_updated_outputs(self) -> None:
        self.collect_generation_outputs("post_update_by_endpoint")
        self.report["checks"]["both_generation_endpoints_post_update"] = True
        self.report["checks"]["post_update_available"] = True
        self.compare_outputs()

    def compare_outputs(self) -> None:
        """Require complete text/token/logprob equality for every prompt/API."""
        baseline_outputs = self.report["baseline_by_endpoint"]
        updated_outputs = self.report["post_update_by_endpoint"]
        outputs_identical_by_endpoint = {}
        for endpoint in ENDPOINTS:
            before_update = baseline_outputs.get(endpoint, [])
            after_update = updated_outputs.get(endpoint, [])
            expected_identities = [(rank, prompt_index) for rank in range(self.data_parallel_size)
                                   for prompt_index in range(len(self.prompts))]
            if ([(item.get("dp_rank"), item.get("prompt_index")) for item in before_update] != expected_identities
                    or [(item.get("dp_rank"), item.get("prompt_index")) for item in after_update] != expected_identities):
                self.report.setdefault("endpoint_checks", {})[endpoint] = False
                self.report["checks"]["outputs_identical"] = False
                raise RuntimeError(f"{endpoint} 更新前后必须完整覆盖每个 DP 副本的全部提示词")
            try:
                for baseline_output, updated_output in zip(before_update, after_update):
                    validate_generation_output(endpoint, baseline_output)
                    validate_generation_output(endpoint, updated_output)
            except RuntimeError:
                self.report.setdefault("endpoint_checks", {})[endpoint] = False
                self.report["checks"]["outputs_identical"] = False
                raise
            outputs_identical_by_endpoint[endpoint] = [
                baseline_output == updated_output
                for baseline_output, updated_output in zip(before_update, after_update)
            ]

        self.report["outputs_identical_by_endpoint"] = outputs_identical_by_endpoint
        self.report["endpoint_checks"] = {
            endpoint: all(matches)
            for endpoint, matches in outputs_identical_by_endpoint.items()
        }
        self.report["checks"]["outputs_identical"] = all(
            all(matches) for matches in outputs_identical_by_endpoint.values()
        )
        self.report["outputs_identical_by_dp_rank"] = {
            str(rank): all(
                matches[rank * len(self.prompts) + index]
                for matches in outputs_identical_by_endpoint.values()
                for index in range(len(self.prompts))
            ) for rank in range(self.data_parallel_size)
        }
        if not self.report["checks"]["outputs_identical"]:
            raise RuntimeError("权重更新前后两个接口的推理结果不一致")
