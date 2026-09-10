"""Batch-creation rate limit enforcement (documented: 2,000 batch creations per hour)."""
from __future__ import annotations

from .api import BatchApi

WINDOW_SECONDS = 3600.0


class CreationLimitExceeded(RuntimeError):
    pass


def seconds_until_allowed(recent_epochs: list[float], planned: int, limit: int, now: float,
                          window: float = WINDOW_SECONDS) -> float:
    """Seconds to wait before `planned` creations fit in the rolling window given prior creation start times."""
    if planned > limit:
        raise CreationLimitExceeded(f"planned {planned} creations exceed the rolling-hour limit of {limit}")
    recent = sorted(e for e in recent_epochs if e > now - window)
    excess = len(recent) + planned - limit
    if excess <= 0:
        return 0.0
    # the `excess`-th oldest recent creation must age out of the window first
    return max(0.0, recent[excess - 1] + window - now)


def check_allowed(recent_epochs: list[float], planned: int, limit: int, now: float) -> None:
    wait = seconds_until_allowed(recent_epochs, planned, limit, now)
    if wait > 0:
        recent = [e for e in recent_epochs if e > now - WINDOW_SECONDS]
        raise CreationLimitExceeded(
            f"{len(recent)} creations in the last hour + {planned} planned > limit {limit}; allowed in {wait/60:.1f} min")


async def recent_creations_from_api(api: BatchApi, since_epoch: float, max_pages: int = 200) -> tuple[int, list[float]]:
    """Count batches the *project* created since `since_epoch` by paging /v1/batches (newest first)."""
    epochs: list[float] = []
    after: str | None = None
    for _ in range(max_pages):
        res = await api.list_batches(after=after, limit=100)
        if not res.ok:
            raise RuntimeError(f"list_batches failed: {res.error_type} {res.error_message}")
        items = res.data["data"]
        if not items:
            break
        stop = False
        for b in items:
            ca = b.get("created_at")
            if ca is None:
                continue
            if ca >= since_epoch:
                epochs.append(float(ca))
            else:
                stop = True
        if stop or not res.data.get("has_more"):
            break
        after = items[-1]["id"]
    return len(epochs), epochs
