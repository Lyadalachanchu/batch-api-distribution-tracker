"""Append-only JSONL event writers. Every record is redacted before it touches disk."""
from __future__ import annotations

import gzip
import json
import os
from typing import Any, Iterator

from .redact import redact_obj


ROTATE_BYTES = 90_000_000  # keep every archive part under GitHub's 100 MB per-file limit


def part_paths(path: str) -> list[str]:
    """Rotated parts of a ".jsonl.gz" archive, in order: <base>.part01.jsonl.gz, .part02..."""
    if not path.endswith(".jsonl.gz"):
        return []
    base = path[: -len(".jsonl.gz")]
    d = os.path.dirname(path) or "."
    prefix = os.path.basename(base) + ".part"
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".jsonl.gz"))


class EventWriter:
    def __init__(self, path: str, rotate_bytes: int = ROTATE_BYTES):
        self.path = path
        self.rotate_bytes = rotate_bytes
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # open in append mode; never truncate. A ".gz" path appends a new gzip member per session
        # (multi-member gzip streams are valid; read_jsonl handles them), ~20x smaller for poll events.
        if path.endswith(".gz"):
            self._fh = gzip.open(path, "at", encoding="utf-8", compresslevel=6)
        else:
            self._fh = open(path, "a", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(redact_obj(event), separators=(",", ":"), ensure_ascii=False, sort_keys=True, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()

    def checkpoint(self) -> None:
        """Finish the current gzip member and start a new one so readers see everything written so far.
        When the archive exceeds rotate_bytes it is renamed to the next numbered part (append-only: parts
        are never rewritten) and a fresh archive is started."""
        if self.path.endswith(".gz"):
            self._fh.close()
            try:
                if self.path.endswith(".jsonl.gz") and os.path.getsize(self.path) >= self.rotate_bytes:
                    n = len(part_paths(self.path)) + 1
                    os.rename(self.path, f"{self.path[:-len('.jsonl.gz')]}.part{n:02d}.jsonl.gz")
            except OSError:
                pass
            self._fh = gzip.open(self.path, "at", encoding="utf-8", compresslevel=6)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Iterate records from a JSONL file, its rotated ".partNN.jsonl.gz" archives and its ".gz" sibling, in order."""
    gz = path if path.endswith(".gz") else path + ".gz"
    candidates = ([] if path.endswith(".gz") else [path]) + part_paths(gz) + [gz]
    for p in candidates:
        if not os.path.exists(p):
            continue
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            except EOFError:
                # a member still being written by a live monitor: everything up to the last checkpoint was yielded
                return
