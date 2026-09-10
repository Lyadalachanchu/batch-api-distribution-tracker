"""Reconcile local state with the server so restarts never create duplicate Batch jobs.

If a creation call timed out (outcome=unknown) the server may still have created the batch. We page
/v1/batches, match on metadata.observation_id + experiment_id, and adopt any batch we did not record."""
from __future__ import annotations

import logging
from typing import Any

from .runtime import Runtime

log = logging.getLogger(__name__)


async def reconcile(rt: Runtime, max_pages: int = 300, since_epoch: float | None = None) -> dict[str, Any]:
    assert rt.api is not None
    adopted: list[str] = []
    duplicates: list[dict[str, Any]] = []
    seen_obs: dict[str, str] = {}
    after: str | None = None
    pages = 0
    while pages < max_pages:
        res = await rt.api.list_batches(after=after, limit=100)
        if not res.ok:
            raise RuntimeError(f"list_batches failed: {res.error_type} {res.error_message}")
        items = res.data["data"]
        if not items:
            break
        pages += 1
        for b in items:
            md = b.get("metadata") or {}
            if md.get("experiment_id") != rt.cfg.experiment_id:
                continue
            obs = md.get("observation_id")
            if not obs:
                continue
            if obs in seen_obs and seen_obs[obs] != b["id"]:
                duplicates.append({"observation_id": obs, "batch_ids": [seen_obs[obs], b["id"]]})
            seen_obs.setdefault(obs, b["id"])
            job = rt.store.get_job(obs)
            if job is None:
                continue
            if job.get("batch_id") is None:
                rt.store.adopt_batch(obs, b)
                adopted.append(obs)
                rt.creation_events.append({"kind": "batch_adopted", "observation_id": obs, "batch_id": b["id"],
                                           "created_at": b.get("created_at"), "status": b.get("status")})
        if since_epoch is not None and items[-1].get("created_at", 0) < since_epoch:
            break
        if not res.data.get("has_more"):
            break
        after = items[-1]["id"]
    unresolved = [j["observation_id"] for j in rt.store.list_jobs(creation_state="unknown")] + \
                 [j["observation_id"] for j in rt.store.list_jobs(creation_state="in_flight")]
    # anything still unknown after a full reconcile did not reach the server: treat as a creation error
    for obs in unresolved:
        job = rt.store.get_job(obs)
        if job and job.get("batch_id") is None:
            rt.store.conn.execute("UPDATE jobs SET creation_state='error' WHERE observation_id=? AND batch_id IS NULL", (obs,))
            rt.store.conn.execute("UPDATE creation_attempts SET outcome='error' WHERE observation_id=? AND outcome IN ('unknown','in_flight')", (obs,))
    out = {"pages": pages, "adopted": adopted, "duplicates": duplicates, "resolved_unknown_as_error": unresolved}
    if adopted or duplicates or unresolved:
        log.info("reconcile: adopted=%d duplicates=%d resolved_unknown=%d", len(adopted), len(duplicates), len(unresolved))
    return out
