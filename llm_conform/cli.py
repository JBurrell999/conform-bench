"""Command-line entry point: `llm-conform <subcommand> ...`.

Subcommands:
  run          Run the test suite against one or more configured engines.
  report       Turn saved result JSON files into a markdown matrix.
  list-tests   Print the test case catalog.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from llm_conform.config import load_engine_configs
from llm_conform.report import build_full_matrix_markdown, load_reports, update_readme
from llm_conform.runner import run_many
from llm_conform.testcase_loader import load_structured_tests, load_tool_tests


def _cmd_run(args: argparse.Namespace) -> int:
    configs = load_engine_configs(args.config)
    if args.engine:
        configs = [c for c in configs if c.name in set(args.engine)]
        if not configs:
            print(f"no engines matching {args.engine} found in {args.config}", file=sys.stderr)
            return 2

    categories = args.category if args.category else None
    reports = run_many(configs, categories=categories, parallel=args.parallel)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        passed = sum(1 for r in report.results if r.outcome.value == "pass")
        total = len(report.results)
        print(f"{report.metadata.engine} ({report.metadata.engine_type}): {passed}/{total} passed")
        for r in report.results:
            if r.outcome.value != "pass":
                print(f"  [{r.outcome.value:11s}] {r.test_id}: {r.reason}")

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{report.metadata.engine}-{ts}.json"
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"  -> wrote {out_path}")

        if args.latest:
            latest_path = out_dir.parent / f"{report.metadata.engine}-latest.json"
            latest_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"  -> wrote {latest_path}")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.results]
    reports = load_reports(paths)
    if not reports:
        print("no result files given", file=sys.stderr)
        return 2
    matrix = build_full_matrix_markdown(reports)

    if args.update_readme:
        new_text = update_readme(Path(args.update_readme), matrix)
        print(f"updated matrix in {args.update_readme} ({len(new_text)} bytes)")
    else:
        print(matrix)
    return 0


def _cmd_list_tests(args: argparse.Namespace) -> int:
    structured = load_structured_tests()
    tools = load_tool_tests()
    print(f"structured_output ({len(structured)} tests):")
    for t in structured:
        print(f"  {t.id:30s} [{t.difficulty:10s}] {t.description}")
    print(f"\ntool_calling ({len(tools)} tests):")
    for t in tools:
        print(f"  {t.id:30s} [{t.difficulty:10s}] {t.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-conform")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the suite against configured engines")
    run_p.add_argument("--config", required=True, help="path to an engines.yaml file")
    run_p.add_argument("--engine", action="append", help="only run this engine name (repeatable)")
    run_p.add_argument(
        "--category",
        action="append",
        choices=["structured_output", "tool_calling"],
        help="only run this category (repeatable); default: both",
    )
    run_p.add_argument(
        "--output-dir",
        default="results/runs",
        help="directory to write timestamped result JSON into (default: results/runs, gitignored)",
    )
    run_p.add_argument(
        "--latest",
        action="store_true",
        help="also write results/<engine>-latest.json (one level above --output-dir) -- "
        "these are the files `llm-conform report` and the README matrix are meant to track in git",
    )
    run_p.add_argument(
        "--parallel", action="store_true", help="run each engine's suite concurrently"
    )
    run_p.set_defaults(func=_cmd_run)

    report_p = sub.add_parser("report", help="build a markdown compatibility matrix from result files")
    report_p.add_argument("results", nargs="+", help="one or more result JSON files")
    report_p.add_argument(
        "--update-readme", metavar="README.md", help="inject the matrix into this file in place"
    )
    report_p.set_defaults(func=_cmd_report)

    list_p = sub.add_parser("list-tests", help="print the test case catalog")
    list_p.set_defaults(func=_cmd_list_tests)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
