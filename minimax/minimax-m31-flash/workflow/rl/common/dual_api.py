"""Small shared probe for the two OpenAI-compatible generation APIs."""

from __future__ import annotations

import math
from typing import Any, Callable

ENDPOINTS = ("/v1/completions", "/v1/chat/completions")


def validate_generation_output(endpoint: str, output: dict[str, Any]) -> None:
    """Require generated text and aligned token/logprob evidence."""
    if not isinstance(output.get("text"), str) or not output["text"].strip():
        raise RuntimeError(f"{endpoint} 返回空输出")
    tokens = output.get("tokens")
    token_logprobs = output.get("token_logprobs")
    if not isinstance(tokens, list) or not tokens or any(
        not isinstance(token, str) for token in tokens
    ):
        raise RuntimeError(f"{endpoint} 缺少有效的生成 token 列表")
    if not isinstance(token_logprobs, list) or len(token_logprobs) != len(tokens):
        raise RuntimeError(f"{endpoint} token_logprobs 缺失或数量与 tokens 不一致")
    if any(
        isinstance(logprob, bool)
        or not isinstance(logprob, (int, float))
        or not math.isfinite(logprob)
        for logprob in token_logprobs
    ):
        raise RuntimeError(f"{endpoint} token_logprobs 必须包含有效的有限数值")


def generate_endpoint(
    request_fn: Callable[..., Any], base: str, model: str, prompt: str,
    max_tokens: int, *, endpoint: str, timeout: float,
) -> dict[str, Any]:
    """Generate one validated response, allowing callers to retain partial evidence."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"不支持的生成接口：{endpoint}")
    payload: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens,
        "temperature": 0, "top_p": 1, "seed": 0,
    }
    if endpoint == "/v1/chat/completions":
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    else:
        payload["prompt"] = prompt
        payload["logprobs"] = 1
    data = request_fn(base, "POST", endpoint, payload, timeout=timeout)
    choice = data["choices"][0]
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(f"{endpoint} 未返回 logprobs")
    if endpoint == "/v1/chat/completions":
        text = (choice.get("message") or {}).get("content", "")
        token_details = logprobs.get("content")
        if not isinstance(token_details, list) or not token_details or any(
            not isinstance(token_detail, dict) for token_detail in token_details
        ):
            raise RuntimeError(f"{endpoint} 未返回有效的 logprobs.content")
        tokens = [token_detail.get("token") for token_detail in token_details]
        token_logprobs = [token_detail.get("logprob") for token_detail in token_details]
    else:
        text = choice.get("text", "")
        tokens = logprobs.get("tokens")
        token_logprobs = logprobs.get("token_logprobs")
    output = {
        "text": text,
        "tokens": tokens,
        "token_logprobs": token_logprobs,
    }
    validate_generation_output(endpoint, output)
    return output


def generate_both(
    request_fn: Callable[..., Any], base: str, model: str, prompt: str,
    max_tokens: int, *, timeout: float,
) -> dict[str, Any]:
    """Call completions and chat/completions with identical generation settings."""
    return {
        endpoint: generate_endpoint(
            request_fn, base, model, prompt, max_tokens, endpoint=endpoint, timeout=timeout,
        )
        for endpoint in ENDPOINTS
    }
