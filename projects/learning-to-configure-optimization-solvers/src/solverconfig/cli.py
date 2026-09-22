from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instance-specific HiGHS configuration benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run the full configured experiment")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", default="results/latest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_benchmark(args.config, args.output)
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
