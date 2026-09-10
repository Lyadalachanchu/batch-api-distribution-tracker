"""Append-only JSONL event writers. Every record is redacted before it touches disk."""
from __future__ import annotations

import gzip
import json
import os
from typing import Any, Iterator

from .redact import redact_obj


class EventWriter:
    def __init__(self, path: str):
        self.path = path
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
        """Finish the current gzip member and start a new one so readers see everything written so far."""
        if self.path.endswith(".gz"):
            self._fh.close()
            self._fh = gzip.open(self.path, "at", encoding="utf-8", compresslevel=6)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Iterate records from a JSONL file; also reads the ".gz" sibling if present (both are yielded)."""
    for p in (path, path + ".gz") if not path.endswith(".gz") else (path,):
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
