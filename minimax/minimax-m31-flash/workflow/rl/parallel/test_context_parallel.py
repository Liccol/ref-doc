#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online HTTP validation for decode/prefill context parallelism.

The script starts a baseline service and a CP variant, sends the same
completion requests to both, and compares the returned text. With
``--worker-hosts`` it delegates multi-node startup and cleanup to
``vllm-ascend-serving``.
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


CP_FLAGS = {
    "--prefill-context-parallel-size",
    "--decode-context-parallel-size",
}


def _valid_cp_size(tp: int) -> int:
    for size in range(2, tp + 1):
        if tp % size == 0:
            return size
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(
        "Online Context Parallel RL use-case verification (single or multi-node)"
    )
    add_online_feature_arguments(parser)
    parser.add_argument("--dim", choices=("dcp", "pcp", "both"), default="both")
    parser.add_argument("--dcp-size", type=int, default=0,
                        help="Decode CP size; defaults to the smallest divisor of TP >= 2")
    parser.add_argument("--pcp-size", type=int, default=0,
                        help="Prefill CP size; defaults to 2")
    parser.add_argument("--strict-output-match", action="store_true",
                        help="Require exact greedy text equality with the CP=1 baseline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        baseline = parse_baseline(args.baseline_cmd)
        tp = baseline["tensor_parallel_size"]
        if tp < 2:
            raise UnsupportedFeature("CP requires tensor-parallel-size >= 2")

        dcp_size = args.dcp_size or _valid_cp_size(tp)
        pcp_size = args.pcp_size or 2
        cases = []
        if args.dim in ("dcp", "both") and dcp_size > 1:
            cases.append(("cp-dcp", ["--prefill-context-parallel-size", "1",
                                      "--decode-context-parallel-size", str(dcp_size)]))
        if args.dim in ("pcp", "both") and pcp_size > 1:
            cases.append(("cp-pcp", ["--prefill-context-parallel-size", str(pcp_size),
                                      "--decode-context-parallel-size", "1"]))
        if not cases:
            raise UnsupportedFeature("no CP dimension has a size greater than 1")

        exit_codes = [
            run_online_pair(
                baseline,
                args,
                feature_name=name,
                remove_flags=CP_FLAGS,
                feature_args=feature_args,
                compare_outputs=args.strict_output_match,
            )
            for name, feature_args in cases
        ]
        return 0 if all(code == 0 for code in exit_codes) else (2 if any(code == 2 for code in exit_codes) else 1)
    except UnsupportedFeature as exc:
        print(f"[cp] BLOCKED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"[cp] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
