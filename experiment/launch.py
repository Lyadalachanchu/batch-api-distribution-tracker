"""launch: the timed wave. Submits every pending production job in one compact, randomised wave."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
from typing import Any

from .config import ExperimentConfig
from .cost import project_cost, enforce_ceiling, actual_cost_usd
from .limits import seconds_until_allowed, recent_creations_from_api, CreationLimitExceeded, WINDOW_SECONDS
from .reconcile import reconcile
from .runtime import Runtime
from .timeutil import iso_now, epoch_now
from .wave import run_wave

log = logging.getLogger(__name__)


def spent_or_committed_usd(rt: Runtime) -> dict[str, float]:
    """Actual cost of collected jobs + worst-case cost of created-but-uncollected jobs (conservative)."""
    cfg = rt.cfg
    actual = 0.0
    committed = 0.0
    for j in rt.store.jobs_with_results():
        if j.get("batch_id") is None:
            continue
        if j.get("collected") and j.get("output_tokens") is not None:
            actual += actual_cost_usd(j.get("input_tokens"), j.get("cached_input_tokens"), j.get("output_tokens"), cfg.pricing) or 0.0
        elif j.get("collected"):
            continue  # terminal with no usage (failed/expired): nothing billed
        else:
            committed += (j["api_max_output_tokens"] * cfg.pricing.output_per_1m_usd
                          + cfg.estimated_input_tokens_per_request * cfg.pricing.input_per_1m_usd) / 1e6
    return {"actual_usd": round(actual, 6), "committed_worst_case_usd": round(committed, 6), "total_usd": round(actual + committed, 6)}


async def recent_creation_epochs(rt: Runtime, now: float, use_api: bool = True) -> tuple[list[float], dict[str, int]]:
    local = rt.store.recent_creation_epochs(now - WINDOW_SECONDS)
    api_epochs: list[float] = []
    if use_api and rt.api is not None:
        _, api_epochs = await recent_creations_from_api(rt.api, now - WINDOW_SECONDS)
    # take the larger count (the project may have other batches; local includes unknown outcomes)
    chosen = api_epochs if len(api_epochs) > len(local) else local
    return chosen, {"local": len(local), "api": len(api_epochs)}


def created_at_balance(rt: Runtime, phases: list[str]) -> dict[str, Any]:
    jobs = [j for j in rt.store.list_jobs(phases=phases) if j.get("created_at") is not None]
    launch_iso = rt.store.get_meta("launch_started_at")
    by_group: dict[int, list[float]] = {}
    for j in jobs:
        by_group.setdefault(j["requested_output_tokens"], []).append(float(j["created_at"]))
    out: dict[str, Any] = {"groups": {}}
    all_ca = [c for v in by_group.values() for c in v]
    if all_ca:
        out["created_at_min"] = min(all_ca)
        out["created_at_max"] = max(all_ca)
        out["created_at_span_seconds"] = max(all_ca) - min(all_ca)
    for n, vals in sorted(by_group.items()):
        vals = sorted(vals)
        q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [vals[0], vals[len(vals) // 2], vals[-1]]
        out["groups"][str(n)] = {"n": len(vals), "min": vals[0], "p25": q[0], "median": statistics.median(vals), "p75": q[-1],
                                 "max": vals[-1], "mean": statistics.fmean(vals)}
    if len(by_group) >= 2:
        medians = [g["median"] for g in out["groups"].values()]
        out["max_median_gap_seconds"] = max(medians) - min(medians)
    out["launch_started_at"] = launch_iso
    return out


async def launch(args: Any, api=None) -> dict[str, Any]:
    cfg = ExperimentConfig.load(getattr(args, "config", None) or "config/experiment.json")
    rt = Runtime.open(cfg, api=api)
    try:
        execute = bool(getattr(args, "execute", False))
        concurrency = int(getattr(args, "concurrency", None) or cfg.launch_concurrency)
        ceiling = float(getattr(args, "max_cost_usd", None) or cfg.max_cost_usd)
        phases = ["prod"]

        # gates
        pilot_status = rt.store.get_meta("pilot_status")
        if pilot_status != "passed" and not getattr(args, "skip_pilot_gate", False):
            raise RuntimeError(f"pilot_status={pilot_status!r}; run `python -m experiment pilot --execute` and pass it first "
                               "(or --skip-pilot-gate to override explicitly)")
        if rt.store.get_meta("launch_started_at") and not getattr(args, "resume", False):
            raise RuntimeError("launch_started_at already recorded: the timed wave has run. Use `recover` for failed creations, "
                               "or --resume to submit still-pending jobs (they will be labelled as a later wave).")
        rec = await reconcile(rt)
        if rec["duplicates"]:
            raise RuntimeError(f"duplicate batches exist for observations {rec['duplicates']}; resolve before launching")
        pending = rt.store.list_jobs(phases=phases, creation_state="pending")
        if not pending:
            log.info("nothing pending; launch is a no-op (idempotent)")
            return {"pending": 0}
        for j in pending:
            if not j.get("input_file_id"):
                raise RuntimeError(f"{j['observation_id']} has no input_file_id; run prepare (file_mode={cfg.file_mode})")

        # cost gate
        level_counts: dict[int, int] = {}
        for j in pending:
            level_counts[j["api_max_output_tokens"]] = level_counts.get(j["api_max_output_tokens"], 0) + 1
        spent = spent_or_committed_usd(rt)
        proj = project_cost(level_counts, cfg.estimated_input_tokens_per_request, cfg.pricing, ceiling, cfg.cost_safety_margin,
                            already_spent_usd=spent["total_usd"])
        enforce_ceiling(proj)

        # rolling-hour creation limit gate
        limit = cfg.max_creations_per_rolling_hour
        while True:
            now = epoch_now()
            epochs, counts = await recent_creation_epochs(rt, now)
            wait = seconds_until_allowed(epochs, len(pending), limit, now)
            if wait <= 0:
                break
            msg = (f"{len(epochs)} creations in the last hour (local={counts['local']}, api={counts['api']}) + {len(pending)} planned "
                   f"> {limit}; allowed in {wait/60:.1f} min")
            if not getattr(args, "wait", False) or not execute:
                raise CreationLimitExceeded(msg)
            log.info("waiting: %s", msg)
            await asyncio.sleep(min(wait + 1.0, 60.0))

        plan = {"pending": len(pending), "concurrency": concurrency, "cost_projection": proj.to_dict(), "spent_before": spent,
                "recent_creations": counts, "limit_per_hour": limit, "levels": level_counts, "execute": execute,
                "first_positions": [j["launch_position"] for j in pending[:10]]}
        if not execute:
            log.info("DRY RUN (no --execute): %s", json.dumps(plan))
            return plan

        wave_name = "launch" if not rt.store.get_meta("launch_started_at") else f"launch-resume-{iso_now()}"
        if wave_name == "launch":
            rt.store.set_meta("launch_started_at", iso_now())  # immediately before the first creation call
        stats = await run_wave(rt, pending, concurrency, wave_name)
        if wave_name == "launch":
            rt.store.set_meta("launch_finished_at", iso_now())
        balance = created_at_balance(rt, phases)
        summary = {"plan": plan, "wave": stats.to_dict(), "created_at_balance": balance,
                   "launch_started_at": rt.store.get_meta("launch_started_at"), "launch_finished_at": rt.store.get_meta("launch_finished_at")}
        os.makedirs(cfg.processed_dir, exist_ok=True)
        path = os.path.join(cfg.processed_dir, f"{wave_name.replace(':', '-')}_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("launch wave done: created=%d errors=%d unknown=%d skipped=%d in %.1fs; created_at span=%.0fs",
                 stats.created, stats.errors, stats.unknown, stats.skipped, stats.duration_seconds or 0,
                 balance.get("created_at_span_seconds", 0) or 0)
        return summary
    finally:
        await rt.aclose()
