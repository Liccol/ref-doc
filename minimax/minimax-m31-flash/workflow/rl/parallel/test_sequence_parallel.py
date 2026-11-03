#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online HTTP validation for MoE sequence parallelism (SP).

The baseline and SP-enabled services are launched independently, so the
script works with a normal online vLLM endpoint. Supplying ``--worker-hosts``
uses the shared vLLM-Ascend multi-node service launcher.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from common.feature_arguments import create_feature_parser
from common.online_feature_common import (
    UnsupportedFeature,
    add_online_feature_arguments,
    parse_baseline,
    run_online_pair,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(
        "Online Sequence Parallel RL use-case verification (single or multi-node)"
    )
    add_online_feature_arguments(parser)
    parser.add_argument("--sp-baseline-all2all", default="nop",
                        help="All2All backend for the non-SP baseline (default: nop)")
    parser.add_argument("--sp-all2all", default="allgather_reducescatter",
                        help="All2All backend used to enable SP on the target version")
    parser.add_argument("--strict-output-match", action="store_true",
                        help="Require exact greedy text equality with the non-SP baseline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        baseline = parse_baseline(args.baseline_cmd)
        if baseline["tensor_parallel_size"] <= 1:
            raise UnsupportedFeature("SP requires tensor-parallel-size > 1")
        if baseline["data_parallel_size"] <= 1:
            raise UnsupportedFeature("SP requires data-parallel-size > 1")
        if not baseline["enable_expert_parallel"]:
            raise UnsupportedFeature("SP requires --enable-expert-parallel for an MoE model")

        return run_online_pair(
            baseline,
            args,
            feature_name="sp",
            remove_flags={"--all2all-backend"},
            feature_args=["--all2all-backend", args.sp_all2all],
            baseline_args=["--all2all-backend", args.sp_baseline_all2all],
            compare_outputs=args.strict_output_match,
        )
    except UnsupportedFeature as exc:
        print(f"[sp] BLOCKED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"[sp] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
