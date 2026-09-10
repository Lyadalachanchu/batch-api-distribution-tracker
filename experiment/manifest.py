"""Randomised launch manifest: exactly runs_per_group copies of every level, shuffled with a recorded seed."""
from __future__ import annotations

import csv
import os
import random
from typing import Any

MANIFEST_COLUMNS = [
    "launch_position",
    "observation_id",
    "phase",
    "attempt_id",
    "requested_output_tokens",
    "api_max_output_tokens",
    "custom_id",
    "seed",
]


def observation_id(phase: str, tokens: int, k: int, attempt: int = 1) -> str:
    base = f"{phase}-t{int(tokens):05d}-k{int(k):04d}"
    return base if attempt == 1 else f"{base}-a{attempt}"


def build_manifest(levels: list[int], runs_per_group: int, seed: int, custom_ids: dict[int, str] | None = None,
                   phase: str = "prod") -> list[dict[str, Any]]:
    """Deterministic: same (levels, runs_per_group, seed) -> identical order."""
    rows: list[dict[str, Any]] = []
    for n in sorted(levels):
        for k in range(runs_per_group):
            rows.append({
                "observation_id": observation_id(phase, n, k),
                "phase": phase,
                "attempt_id": 1,
                "requested_output_tokens": int(n),
                "api_max_output_tokens": int(n),
                "custom_id": (custom_ids or {}).get(n, observation_id(phase, n, k)),
                "seed": int(seed),
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    for pos, row in enumerate(rows):
        row["launch_position"] = pos
    return [{c: r[c] for c in MANIFEST_COLUMNS} for r in rows]


def write_manifest_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_manifest_csv(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in ("launch_position", "attempt_id", "requested_output_tokens", "api_max_output_tokens", "seed"):
            r[c] = int(r[c])
    return rows


def check_manifest(rows: list[dict[str, Any]], levels: list[int], runs_per_group: int) -> list[str]:
    """Return a list of problems (empty == valid)."""
    problems: list[str] = []
    counts: dict[int, int] = {}
    for r in rows:
        counts[r["requested_output_tokens"]] = counts.get(r["requested_output_tokens"], 0) + 1
    for n in levels:
        if counts.get(n, 0) != runs_per_group:
            problems.append(f"level {n}: {counts.get(n, 0)} rows, expected {runs_per_group}")
    extra = set(counts) - set(levels)
    if extra:
        problems.append(f"unexpected levels in manifest: {sorted(extra)}")
    if len(rows) != len(levels) * runs_per_group:
        problems.append(f"total rows {len(rows)} != {len(levels) * runs_per_group}")
    ids = [r["observation_id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate observation_id values")
    positions = sorted(r["launch_position"] for r in rows)
    if positions != list(range(len(rows))):
        problems.append("launch_position is not a permutation of 0..N-1")
    return problems
