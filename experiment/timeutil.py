from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    """ISO-8601 UTC timestamp with microseconds, e.g. 2026-09-10T07:00:00.123456+00:00."""
    return utc_now().isoformat(timespec="microseconds")


def epoch_now() -> float:
    return time.time()


def epoch_to_iso(epoch: float | int | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="seconds")


def iso_to_epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    return datetime.fromisoformat(iso).timestamp()
