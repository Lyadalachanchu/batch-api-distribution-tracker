"""Append-only JSONL event writers. Every record is redacted before it touches disk."""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

from .redact import redact_obj


class EventWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # open in append mode; never truncate
        self._fh = open(path, "a", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(redact_obj(event), separators=(",", ":"), ensure_ascii=False, sort_keys=True, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
