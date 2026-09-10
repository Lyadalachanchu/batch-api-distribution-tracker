"""recover: create clearly-labelled replacement jobs for production jobs whose creation call failed."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from .config import ExperimentConfig
from .cost import project_cost, enforce_ceiling
from .launch import spent_or_committed_usd, recent_creation_epochs
from .limits import seconds_until_allowed, CreationLimitExceeded
from .manifest import observation_id as make_obs_id
from .reconcile import reconcile
from .runtime import Runtime
from .timeutil import epoch_now, iso_now
from .wave import run_wave

log = logging.getLogger(__name__)


def plan_replacements(rt: Runtime) -> list[dict[str, Any]]:
    """One replacement per failed creation that does not already have a live replacement."""
    jobs = rt.store.list_jobs(phases=["prod", "replacement"])
    children: dict[str, list[dict[str, Any]]] = {}
    for j in jobs:
        if j.get("parent_observation_id"):
            children.setdefault(j["parent_observation_id"], []).append(j)
    rows: list[dict[str, Any]] = []
    planned_roots: set[str] = set()
    for j in jobs:
        if j["creation_state"] != "error" or j.get("batch_id"):
            continue
        root = j.get("parent_observation_id") or j["observation_id"]
        if root in planned_roots:
            continue
        live = [c for c in children.get(root, []) if c["creation_state"] in ("pending", "in_flight", "created", "unknown")]
        if live:
            continue
        planned_roots.add(root)
        attempt = 1 + len(children.get(root, [])) + 1
        rows.append({
            "observation_id": make_obs_id("prod", j["requested_output_tokens"], int(root.split("-k")[-1].split("-")[0]), attempt),
            "phase": "replacement", "attempt_id": attempt, "parent_observation_id": root,
            "requested_output_tokens": j["requested_output_tokens"], "api_max_output_tokens": j["api_max_output_tokens"],
            "launch_position": j.get("launch_position"), "custom_id": j["custom_id"], "input_file_id": j.get("input_file_id"),
        })
    return rows


async def recover(args: Any, api=None) -> dict[str, Any]:
    cfg = ExperimentConfig.load(getattr(args, "config", None) or "config/experiment.json")
    rt = Runtime.open(cfg, api=api)
    try:
        await reconcile(rt)
        rows = plan_replacements(rt)
        already = rt.store.list_jobs(phases=["replacement"], creation_state="pending")
        rt.store.insert_jobs(rows, cfg.experiment_id)
        pending = rt.store.list_jobs(phases=["replacement"], creation_state="pending")
        plan = {"new_replacements": len(rows), "already_pending": len(already), "pending_total": len(pending)}
        if not pending:
            log.info("recover: nothing to replace")
            return plan
        ceiling = float(getattr(args, "max_cost_usd", None) or cfg.max_cost_usd)
        level_counts: dict[int, int] = {}
        for j in pending:
            level_counts[j["api_max_output_tokens"]] = level_counts.get(j["api_max_output_tokens"], 0) + 1
        spent = spent_or_committed_usd(rt)
        proj = project_cost(level_counts, cfg.estimated_input_tokens_per_request, cfg.pricing, ceiling, cfg.cost_safety_margin,
                            already_spent_usd=spent["total_usd"])
        enforce_ceiling(proj)
        now = epoch_now()
        epochs, counts = await recent_creation_epochs(rt, now)
        wait = seconds_until_allowed(epochs, len(pending), cfg.max_creations_per_rolling_hour, now)
        plan.update({"cost_projection": proj.to_dict(), "recent_creations": counts, "wait_seconds": wait})
        if wait > 0:
            raise CreationLimitExceeded(f"replacements would exceed the rolling-hour limit; allowed in {wait/60:.1f} min")
        if not getattr(args, "execute", False):
            log.info("DRY RUN recover: %s", json.dumps(plan))
            return plan
        stats = await run_wave(rt, pending, int(getattr(args, "concurrency", None) or 10), f"recover-{iso_now()}")
        plan["wave"] = stats.to_dict()
        os.makedirs(cfg.processed_dir, exist_ok=True)
        with open(os.path.join(cfg.processed_dir, "recover_summary.json"), "a", encoding="utf-8") as f:
            f.write(json.dumps(plan, default=str) + "\n")
        return plan
    finally:
        await rt.aclose()
