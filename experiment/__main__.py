"""CLI entry point: python -m experiment <command> [options]."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m experiment", description="OpenAI Batch API turnaround experiment")
    p.add_argument("--config", default="config/experiment.json", help="path to config/experiment.json")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="build inputs, upload files, write manifest, init DB, validate, cost-check (no batch creations)")
    s.add_argument("--runs-per-group", dest="runs_per_group", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--levels", nargs="+", type=int, help="production output-token levels (must be >= API minimum 16)")
    s.add_argument("--experiment-id", dest="experiment_id")
    s.add_argument("--model")
    s.add_argument("--file-mode", dest="file_mode", choices=["shared", "individual"])
    s.add_argument("--max-cost-usd", dest="max_cost_usd", type=float)
    s.add_argument("--launch-concurrency", dest="launch_concurrency", type=int)
    s.add_argument("--poll-mode", dest="poll_mode", choices=["list", "retrieve"])
    s.add_argument("--poll-interval-seconds", dest="poll_interval_seconds", type=float)
    s.add_argument("--reasoning-effort", dest="reasoning_effort")
    s.add_argument("--data-dir", dest="data_dir")
    s.add_argument("--no-upload", dest="no_upload", action="store_true", help="offline: skip uploads (tests/dry runs)")
    s.add_argument("--reset-config", dest="reset_config", action="store_true", help="ignore an existing config file")

    s = sub.add_parser("pilot", help="run the pilot batches and write reports/pilot_report.md")
    s.add_argument("--execute", action="store_true")
    s.add_argument("--max-cost-usd", dest="max_cost_usd", type=float, default=0.10)
    s.add_argument("--timeout-minutes", dest="timeout_minutes", type=float, default=240)
    s.add_argument("--again", action="store_true", help="create a fresh set of pilot batches even if some exist")
    s.add_argument("--report", default="reports/pilot_report.md")
    s.add_argument("--interval", type=float, help="poll interval override (seconds)")

    s = sub.add_parser("launch", help="timed wave: create every pending production batch")
    s.add_argument("--execute", action="store_true")
    s.add_argument("--concurrency", type=int)
    s.add_argument("--max-cost-usd", dest="max_cost_usd", type=float)
    s.add_argument("--wait", action="store_true", help="sleep until the rolling-hour creation limit allows the wave")
    s.add_argument("--resume", action="store_true", help="submit still-pending production jobs after a partial wave")
    s.add_argument("--skip-pilot-gate", dest="skip_pilot_gate", action="store_true")

    s = sub.add_parser("monitor", help="poll active batches until every job is terminal (resumable)")
    s.add_argument("--phases", nargs="+", default=None)
    s.add_argument("--once", action="store_true")
    s.add_argument("--max-minutes", dest="max_minutes", type=float)
    s.add_argument("--no-collect", dest="no_collect", action="store_true")
    s.add_argument("--poll-mode", dest="poll_mode", choices=["list", "retrieve"])
    s.add_argument("--interval", type=float)
    s.add_argument("--concurrency", type=int)

    s = sub.add_parser("collect", help="fetch output/error files for terminal jobs and build observations")
    s.add_argument("--phases", nargs="+", default=None)
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("recover", help="create labelled replacement batches for failed creation calls")
    s.add_argument("--execute", action="store_true")
    s.add_argument("--concurrency", type=int, default=10)
    s.add_argument("--max-cost-usd", dest="max_cost_usd", type=float)

    s = sub.add_parser("analyze", help="statistics, plots and reports/experiment_report.html")
    s.add_argument("--bootstrap", type=int, default=2000)
    s.add_argument("--out", default="reports/experiment_report.html")

    sub.add_parser("status", help="print experiment state")
    sub.add_parser("reconcile", help="adopt server-side batches missing locally (never creates batches)")
    return p


async def _run(args: argparse.Namespace) -> int:
    from .config import ExperimentConfig
    if args.command == "prepare":
        from .prepare import prepare
        out = await prepare(args)
        print(json.dumps(out, indent=2, default=str))
    elif args.command == "pilot":
        from .pilot import pilot
        out = await pilot(args)
        print(json.dumps(out, indent=2, default=str))
        if isinstance(out, dict) and "passed" in out and not out["passed"]:
            return 2
    elif args.command == "launch":
        from .launch import launch
        out = await launch(args)
        print(json.dumps(out, indent=2, default=str))
    elif args.command == "monitor":
        from .monitor import monitor
        from .runtime import Runtime
        cfg = ExperimentConfig.load(args.config)
        rt = Runtime.open(cfg)
        try:
            from .reconcile import reconcile
            await reconcile(rt)
            out = await monitor(rt, phases=args.phases, once=args.once, max_minutes=args.max_minutes,
                                collect_inline=not args.no_collect, poll_mode=args.poll_mode, interval=args.interval,
                                concurrency=args.concurrency)
        finally:
            await rt.aclose()
        print(json.dumps(out, indent=2, default=str))
    elif args.command == "collect":
        from .collect import collect_all
        from .observations import build_observations, write_observations
        from .runtime import Runtime
        cfg = ExperimentConfig.load(args.config)
        rt = Runtime.open(cfg)
        try:
            out = await collect_all(rt, phases=args.phases, force=args.force)
            df = build_observations(rt.store, cfg)
            paths = write_observations(df, cfg)
            out["observations_rows"] = int(len(df))
            out["paths"] = paths
        finally:
            await rt.aclose()
        print(json.dumps(out, indent=2, default=str))
    elif args.command == "analyze":
        from .analyze import analyze
        out = analyze(args)
        print(json.dumps(out, indent=2, default=str))
    elif args.command == "status":
        from .status import status
        await status(args)
    elif args.command == "reconcile":
        from .reconcile import reconcile
        from .runtime import Runtime
        cfg = ExperimentConfig.load(args.config)
        rt = Runtime.open(cfg)
        try:
            out = await reconcile(rt)
        finally:
            await rt.aclose()
        print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        from .redact import redact_text
        print(f"error: {type(e).__name__}: {redact_text(str(e))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
