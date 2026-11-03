"""Composable CLI argument groups for model RL feature verification."""
from __future__ import annotations

import argparse
import math
from typing import Callable, Literal


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return number


def positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为大于 0 的有限数值")
    return number


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 1..65535 内")
    return port


def create_feature_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_service_arguments(
    parser: argparse.ArgumentParser, *, require_server: bool = True,
    model_option: str | None = "--served-model-name", require_model: bool = False,
) -> None:
    """Add a service URL and, when needed, an explicit or auto-resolved model."""
    parser.add_argument("--server-url", required=require_server, default=None,
                        help="已启动服务的 HTTP 根地址；在线对照特性可使用它连接现有服务")
    if model_option is not None:
        parser.add_argument(model_option, required=require_model,
                            default=None if require_model else "auto", help="请求使用的服务模型名")


def add_generation_arguments(
    parser: argparse.ArgumentParser, *, prompt_input: Literal["json", "repeat", "single"],
    default_prompt: str | None = None,
) -> None:
    """Add one prompt-input form and the common generation length limit."""
    if prompt_input == "json":
        parser.add_argument("--prompts-json", required=True, help='格式：{"prompts":["提示词"]}')
    elif prompt_input == "repeat":
        parser.add_argument("--prompt", action="append", dest="prompts", default=None,
                            help="验证提示词，可重复传入")
    elif prompt_input == "single":
        parser.add_argument("--prompt", default=default_prompt, help="验证提示词")
    else:
        raise ValueError(f"未知提示词输入形式：{prompt_input}")
    parser.add_argument("--max-tokens", type=positive_integer, default=32,
                        help="每次请求的最大生成 token 数")


def add_report_arguments(
    parser: argparse.ArgumentParser, *, report_option: str = "--report",
    required: bool = True, include_verification_command: bool = True,
) -> None:
    parser.add_argument(report_option, required=required, default=None, help="输出的 JSON 报告路径")
    if include_verification_command:
        parser.add_argument("--verification-command", default=None, help="写入报告的完整验证命令")


def add_timeout_argument(
    parser: argparse.ArgumentParser, *, option: str = "--timeout", default: float = 1800,
    value_type: Callable[[str], int | float] = positive_integer,
) -> None:
    parser.add_argument(option, type=value_type, default=default, help="操作等待超时秒数")


def add_cluster_connection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared head/worker connection options without launching a service."""
    parser.add_argument("--host", default=None, help="HTTP 主机或集群 head 的 SSH 地址")
    parser.add_argument("--port", type=port_number, default=None, help="HTTP 服务端口")
    parser.add_argument("--worker-hosts", default=None, help="以逗号分隔的 worker 主机地址")
    parser.add_argument("--connection-mode", choices=("docker", "conda", "direct"), default="docker")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=None, help="SSH 密码")
    parser.add_argument("--key-file", default=None)
    parser.add_argument("--container", default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--worker-password", default=None)
    parser.add_argument("--worker-key-file", default=None)
    parser.add_argument("--worker-container", default=None)
    parser.add_argument("--worker-conda-env", default=None)


def add_checkpoint_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add checkpoint source and buffer options shared by HCCL and IPC."""
    parser.add_argument("--checkpoint", required=True, help="每个 producer 都可读取的 checkpoint 路径")
    parser.add_argument("--packed-buffer-mb", type=positive_integer, default=1024,
                        help="packed buffer 容量，单位 MiB")


def add_weight_update_cluster_arguments(parser: argparse.ArgumentParser) -> None:
    """Describe an existing cluster; service startup remains with serving."""
    parser.add_argument("--inference-nodes", default=None,
                        help="推理节点地址，逗号分隔；用于多机预检和报告，不执行 SSH 启动")
    parser.add_argument("--expected-world-size", type=positive_integer, default=None,
                        help="预期 TP×PP×DP worker 总数；与服务查询结果核对")
    parser.add_argument("--data-parallel-size", type=positive_integer, default=None,
                        help="预期 DP 副本数；省略时根据服务返回的全局和单 DP world size 推导")
