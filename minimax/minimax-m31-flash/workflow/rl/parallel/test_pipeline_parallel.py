#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online HTTP validation for pipeline parallelism (PP).

The script compares a baseline service with a service started using
``--pipeline-parallel-size N``. The shared ``--worker-hosts`` lifecycle
wrapper currently launches cross-node DP, not native cross-node PP, so this
script refuses that combination. For a real multi-node PP deployment, start
the service with the target version's MP/Ray/PP launcher and use
``--server-url`` to attach for HTTP validation.
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
        "Online Pipeline Parallel RL use-case verification (single or multi-node)"
    )
    add_online_feature_arguments(parser)
    parser.add_argument("--pp-size", type=int, default=2,
                        help="Pipeline stage count; must be greater than 1")
    parser.add_argument("--strict-output-match", action="store_true",
                        help="Require exact greedy text equality with the PP=1 baseline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.pp_size < 2:
            raise UnsupportedFeature("PP requires pipeline-parallel-size >= 2")
        if args.worker_hosts and not args.server_url:
            raise UnsupportedFeature(
                "--worker-hosts uses the serving wrapper's cross-node DP path, "
                "not native cross-node PP; start PP with the target version's "
                "MP/Ray/PP launcher and attach with --server-url"
            )
        baseline = parse_baseline(args.baseline_cmd)
        return run_online_pair(
            baseline,
            args,
            feature_name="pp",
            remove_flags={"--pipeline-parallel-size", "-pp"},
            feature_args=["--pipeline-parallel-size", str(args.pp_size)],
            compare_outputs=args.strict_output_match,
        )
    except UnsupportedFeature as exc:
        print(f"[pp] BLOCKED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"[pp] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
