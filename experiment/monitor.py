"""monitor: poll every active Batch every 5-10 s (bounded concurrency), append poll events, collect terminal jobs.

Resumable: state lives in SQLite; restarting never creates batches. Two poll modes:
  list      page /v1/batches (100 per call) — ~21 calls per cycle for 2,000 jobs — then retrieve any stragglers
  retrieve  one GET /v1/batches/{id} per active job per cycle
On a transition to a terminal status the definitive object is re-fetched with a direct retrieve and preserved."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .collect import collect_job
from .runtime import Runtime
from .timeutil import iso_now, epoch_now

log = logging.getLogger(__name__)

POLL_EVENT_FIELDS = ("status", "created_at", "in_progress_at", "finalizing_at", "completed_at", "failed_at", "expired_at",
                     "cancelling_at", "cancelled_at", "expires_at", "request_counts", "output_file_id", "error_file_id")


def poll_event(job: dict[str, Any], b: dict[str, Any], poll_iso: str, cycle: int, source: str) -> dict[str, Any]:
    ev = {"kind": "batch_poll", "batch_id": b.get("id"), "observation_id": job["observation_id"], "poll_timestamp_utc": poll_iso,
          "cycle": cycle, "source": source}
    for f in POLL_EVENT_FIELDS:
        ev[f] = b.get(f)
    return ev


async def _retrieve_many(rt: Runtime, batch_ids: list[str], concurrency: int) -> dict[str, dict[str, Any]]:
    assert rt.api is not None
    sem = asyncio.Semaphore(max(1, concurrency))
    out: dict[str, dict[str, Any]] = {}

    async def one(bid: str) -> None:
        async with sem:
            res = await rt.api.retrieve_batch(bid)
        if res.ok:
            out[bid] = res.data
        else:
            log.warning("retrieve %s failed: %s %s", bid, res.http_status, res.error_message)

    await asyncio.gather(*(one(b) for b in batch_ids))
    return out


async def _list_active(rt: Runtime, active: list[dict[str, Any]], max_pages: int = 400) -> dict[str, dict[str, Any]]:
    assert rt.api is not None
    wanted = {j["batch_id"] for j in active}
    oldest = min((j["created_at"] for j in active if j.get("created_at") is not None), default=None)
    found: dict[str, dict[str, Any]] = {}
    after: str | None = None
    for _ in range(max_pages):
        res = await rt.api.list_batches(after=after, limit=100)
        if not res.ok:
            log.warning("list_batches failed: %s %s", res.http_status, res.error_message)
            break
        items = res.data["data"]
        if not items:
            break
        for b in items:
            if b.get("id") in wanted:
                found[b["id"]] = b
        if len(found) >= len(wanted):
            break
        if oldest is not None and items[-1].get("created_at") is not None and items[-1]["created_at"] < oldest - 1:
            break
        if not res.data.get("has_more"):
            break
        after = items[-1]["id"]
    return found


async def monitor(rt: Runtime, phases: list[str] | None = None, once: bool = False, max_minutes: float | None = None,
                  collect_inline: bool = True, poll_mode: str | None = None, interval: float | None = None,
                  concurrency: int | None = None) -> dict[str, Any]:
    assert rt.api is not None
    cfg = rt.cfg
    mode = poll_mode or cfg.poll_mode
    interval = interval or cfg.poll_interval_seconds
    conc = concurrency or cfg.poll_concurrency
    cycle = int(rt.store.get_meta("monitor_cycles", 0))
    t_start = epoch_now()
    transitions = 0
    summary: dict[str, Any] = {}
    while True:
        active = rt.store.list_jobs(phases=phases, active_only=True)
        if not active:
            log.info("monitor: no active jobs remain")
            break
        cycle += 1
        cycle_t0 = epoch_now()
        poll_iso = iso_now()
        objs: dict[str, dict[str, Any]] = {}
        sources: dict[str, str] = {}
        if mode == "list":
            objs = await _list_active(rt, active)
            sources = {k: "list" for k in objs}
            missing = [j["batch_id"] for j in active if j["batch_id"] not in objs]
            if missing:
                extra = await _retrieve_many(rt, missing, conc)
                objs.update(extra)
                sources.update({k: "retrieve" for k in extra})
        else:
            objs = await _retrieve_many(rt, [j["batch_id"] for j in active], conc)
            sources = {k: "retrieve" for k in objs}

        status_counts: dict[str, int] = {}
        updates: list[tuple[str, dict[str, Any]]] = []
        by_obs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for job in active:
            b = objs.get(job["batch_id"])
            if b is None:
                continue
            status_counts[b.get("status") or "?"] = status_counts.get(b.get("status") or "?", 0) + 1
            rt.poll_events.append(poll_event(job, b, poll_iso, cycle, sources.get(job["batch_id"], "?")))
            updates.append((job["observation_id"], b))
            by_obs[job["observation_id"]] = (job, b)
        newly_terminal = rt.store.apply_batch_objects(updates, poll_iso)  # one transaction per cycle
        transitions += len(newly_terminal)

        sem = asyncio.Semaphore(max(1, conc))

        async def finalize(obs_id: str) -> None:
            job, b = by_obs[obs_id]
            final = b
            async with sem:
                if sources.get(job["batch_id"]) == "list":
                    res = await rt.api.retrieve_batch(job["batch_id"])
                    if res.ok:
                        final = res.data
                        rt.store.apply_batch_object(obs_id, final, iso_now())
                rt.batch_objects.append({"kind": "batch_final", "observation_id": obs_id, "seen_at": iso_now(), "batch": final})
                log.info("terminal: %s %s status=%s created=%s in_progress=%s completed=%s", obs_id, job["batch_id"],
                         final.get("status"), final.get("created_at"), final.get("in_progress_at"), final.get("completed_at"))
                if collect_inline:
                    refreshed = rt.store.get_job(obs_id) or job
                    try:
                        await collect_job(rt, refreshed)
                    except Exception as e:  # noqa: BLE001
                        log.exception("collect failed for %s: %s", obs_id, e)

        if newly_terminal:
            await asyncio.gather(*(finalize(o) for o in newly_terminal))
        rt.store.set_meta("monitor_cycles", cycle)
        rt.poll_events.checkpoint()
        elapsed = epoch_now() - cycle_t0
        summary = {"cycle": cycle, "active": len(active), "seen": len(objs), "status_counts": status_counts,
                   "cycle_seconds": round(elapsed, 2), "transitions_total": transitions}
        if cycle % 10 == 1 or transitions:
            log.info("poll cycle %d: active=%d seen=%d %s (%.1fs)", cycle, len(active), len(objs), status_counts, elapsed)
        if elapsed > 10.0:
            log.warning("poll cycle %d took %.1fs (> 10 s target); consider list mode or more concurrency", cycle, elapsed)
        if once:
            break
        if max_minutes is not None and (epoch_now() - t_start) > max_minutes * 60:
            log.warning("monitor: max_minutes=%.1f reached with %d active jobs", max_minutes, len(active))
            break
        await asyncio.sleep(max(0.0, interval - elapsed))
    summary["remaining_active"] = len(rt.store.list_jobs(phases=phases, active_only=True))
    return summary
