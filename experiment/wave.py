"""A creation wave: submit one Batch per job with bounded concurrency, no retries, full accounting."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from .runtime import Runtime
from .timeutil import iso_now, epoch_now

log = logging.getLogger(__name__)

BATCH_LIMIT_HINTS = ("batch", "per hour", "hour", "too many batches", "batch_limit")


@dataclass
class WaveStats:
    wave: str
    planned: int
    started_at: str | None = None
    finished_at: str | None = None
    created: int = 0
    errors: int = 0
    unknown: int = 0
    skipped: int = 0
    rate_limit_429s: int = 0
    stopped_early: bool = False
    stop_reason: str | None = None
    duration_seconds: float | None = None
    per_status: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata(cfg_experiment_id: str, job: dict[str, Any], attempt_no: int) -> dict[str, str]:
    return {
        "experiment_id": cfg_experiment_id,
        "observation_id": job["observation_id"],
        "attempt_no": str(attempt_no),
        "phase": job["phase"],
        "requested_output_tokens": str(job["requested_output_tokens"]),
        "launch_position": str(job.get("launch_position") if job.get("launch_position") is not None else -1),
    }


async def run_wave(rt: Runtime, jobs: list[dict[str, Any]], concurrency: int, wave: str,
                   max_consecutive_429: int = 5) -> WaveStats:
    """Create one Batch for every job, in the given order, with `concurrency` workers.

    Failed creation calls are NOT retried here; they are recorded (creation_state=error/unknown) and
    left for `recover`. On HTTP 429 the wave pauses for Retry-After (or 2 s); if the error text points
    at the batch-creation limit, or 429s persist, the wave stops so the hourly limit is never exceeded."""
    assert rt.api is not None
    stats = WaveStats(wave=wave, planned=len(jobs))
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stop = asyncio.Event()
    pause_until = {"t": 0.0}
    consecutive_429 = {"n": 0}
    lock = asyncio.Lock()

    async def worker(wid: int) -> None:
        while True:
            if stop.is_set():
                # drain the queue so the wave terminates
                try:
                    queue.get_nowait()
                    stats.skipped += 1
                    queue.task_done()
                    continue
                except asyncio.QueueEmpty:
                    return
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                now = epoch_now()
                if pause_until["t"] > now:
                    await asyncio.sleep(pause_until["t"] - now)
                if stop.is_set():
                    stats.skipped += 1
                    continue
                await _create_one(job)
            finally:
                queue.task_done()

    async def _create_one(job: dict[str, Any]) -> None:
        obs = job["observation_id"]
        started = iso_now()
        started_epoch = epoch_now()
        attempt_row_id = rt.store.begin_attempt(obs, wave, started, started_epoch)
        attempt_no = len(rt.store.attempts_for(obs))
        meta = _metadata(rt.cfg.experiment_id, job, attempt_no)
        res = await rt.api.create_batch(job["input_file_id"], rt.cfg.endpoint, rt.cfg.completion_window, meta)
        finished = iso_now()
        event: dict[str, Any] = {
            "kind": "batch_create",
            "wave": wave,
            "observation_id": obs,
            "attempt_no": attempt_no,
            "phase": job["phase"],
            "requested_output_tokens": job["requested_output_tokens"],
            "launch_position": job.get("launch_position"),
            "input_file_id": job["input_file_id"],
            "local_create_started_at": started,
            "local_create_finished_at": finished,
            "ok": res.ok,
            "http_status": res.http_status,
            "request_id": res.request_id,
            "ratelimit_headers": res.headers,
        }
        async with lock:
            if res.ok:
                b = res.data
                rt.store.finish_attempt_created(attempt_row_id, obs, b, finished, res.http_status, res.request_id)
                event.update({"batch_id": b.get("id"), "created_at": b.get("created_at"), "status": b.get("status"),
                              "expires_at": b.get("expires_at"), "metadata": b.get("metadata")})
                stats.created += 1
                consecutive_429["n"] = 0
                stats.per_status[str(res.http_status)] = stats.per_status.get(str(res.http_status), 0) + 1
            else:
                outcome = "unknown" if res.outcome_unknown else "error"
                rt.store.finish_attempt_failed(attempt_row_id, obs, outcome, finished, res.http_status, res.request_id,
                                               res.error_type, res.error_code, res.error_message)
                event.update({"outcome": outcome, "error_type": res.error_type, "error_code": res.error_code,
                              "error_message": res.error_message, "exception_kind": res.exception_kind})
                if outcome == "unknown":
                    stats.unknown += 1
                else:
                    stats.errors += 1
                key = str(res.http_status or res.exception_kind)
                stats.per_status[key] = stats.per_status.get(key, 0) + 1
                if res.http_status == 429:
                    stats.rate_limit_429s += 1
                    consecutive_429["n"] += 1
                    msg = (res.error_message or "").lower()
                    wait = res.retry_after if res.retry_after else 2.0
                    pause_until["t"] = max(pause_until["t"], epoch_now() + wait)
                    if any(h in msg for h in BATCH_LIMIT_HINTS) and "batch" in msg:
                        stop.set()
                        stats.stopped_early = True
                        stats.stop_reason = f"429 indicates batch-creation limit: {res.error_message}"
                    elif consecutive_429["n"] >= max_consecutive_429:
                        stop.set()
                        stats.stopped_early = True
                        stats.stop_reason = f"{consecutive_429['n']} consecutive 429s"
                    log.warning("429 on %s: %s (pausing %.1fs)", obs, res.error_message, wait)
                else:
                    log.warning("creation %s for %s: %s %s", outcome, obs, res.error_type, res.error_message)
        rt.creation_events.append(event)

    stats.started_at = iso_now()
    t0 = epoch_now()
    workers = [asyncio.create_task(worker(i)) for i in range(max(1, concurrency))]
    await queue.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    stats.finished_at = iso_now()
    stats.duration_seconds = round(epoch_now() - t0, 3)
    return stats
