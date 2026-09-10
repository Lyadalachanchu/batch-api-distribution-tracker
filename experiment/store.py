"""Transactional SQLite state store. All job state lives here so every command is idempotent/resumable."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable

from .config import TERMINAL_STATUSES
from .timeutil import iso_now, epoch_now

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
  file_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,                -- shared | individual
  tokens INTEGER NOT NULL,
  observation_id TEXT,               -- for individual files
  filename TEXT, bytes INTEGER, sha256 TEXT, uploaded_at TEXT, server_created_at INTEGER, purpose TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
  observation_id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL,
  phase TEXT NOT NULL,               -- pilot | prod | replacement
  attempt_id INTEGER NOT NULL DEFAULT 1,
  parent_observation_id TEXT,
  requested_output_tokens INTEGER NOT NULL,
  api_max_output_tokens INTEGER NOT NULL,
  launch_position INTEGER,
  custom_id TEXT NOT NULL,
  input_file_id TEXT,
  batch_id TEXT UNIQUE,
  creation_state TEXT NOT NULL DEFAULT 'pending',   -- pending | in_flight | created | error | unknown
  local_create_started_at TEXT, local_create_finished_at TEXT,
  create_http_status INTEGER, create_request_id TEXT, create_error_type TEXT, create_error_code TEXT, create_error_message TEXT,
  status TEXT, created_at INTEGER, in_progress_at INTEGER, finalizing_at INTEGER, completed_at INTEGER,
  failed_at INTEGER, expired_at INTEGER, cancelling_at INTEGER, cancelled_at INTEGER, expires_at INTEGER,
  request_counts TEXT, output_file_id TEXT, error_file_id TEXT, batch_errors TEXT, batch_usage TEXT, batch_model TEXT,
  terminal INTEGER NOT NULL DEFAULT 0, terminal_seen_at TEXT,
  collected INTEGER NOT NULL DEFAULT 0, collected_at TEXT,
  last_poll_at TEXT, poll_count INTEGER NOT NULL DEFAULT 0, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_phase ON jobs(phase);
CREATE INDEX IF NOT EXISTS jobs_batch ON jobs(batch_id);
CREATE TABLE IF NOT EXISTS creation_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  observation_id TEXT NOT NULL,
  attempt_no INTEGER NOT NULL,
  wave TEXT,
  started_at TEXT, started_epoch REAL, finished_at TEXT,
  outcome TEXT NOT NULL,             -- in_flight | created | error | unknown
  http_status INTEGER, request_id TEXT, batch_id TEXT, error_type TEXT, error_code TEXT, error_message TEXT
);
CREATE INDEX IF NOT EXISTS attempts_epoch ON creation_attempts(started_epoch);
CREATE TABLE IF NOT EXISTS results (
  observation_id TEXT PRIMARY KEY,
  batch_id TEXT, source_file_id TEXT, source_kind TEXT,
  response_status TEXT, incomplete_reason TEXT, http_status INTEGER, openai_request_id TEXT,
  input_tokens INTEGER, cached_input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, total_tokens INTEGER,
  error_code TEXT, error_type TEXT, error_message TEXT,
  output_text_chars INTEGER, output_word_count INTEGER,
  parse_error TEXT, line_sha256 TEXT, collected_at TEXT
);
"""

BATCH_TS_FIELDS = ("created_at", "in_progress_at", "finalizing_at", "completed_at", "failed_at",
                   "expired_at", "cancelling_at", "cancelled_at", "expires_at")

RESULT_FIELDS = ("batch_id", "source_file_id", "source_kind", "response_status", "incomplete_reason", "http_status",
                 "openai_request_id", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens",
                 "total_tokens", "error_code", "error_type", "error_message", "output_text_chars",
                 "output_word_count", "parse_error", "line_sha256", "collected_at")


def _j(v: Any) -> str | None:
    return None if v is None else json.dumps(v, sort_keys=True)


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------- meta ----------
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    # ---------- files ----------
    def add_file(self, file_id: str, kind: str, tokens: int, filename: str, nbytes: int, sha256: str,
                 server_created_at: int | None, observation_id: str | None = None, purpose: str = "batch") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO files(file_id, kind, tokens, observation_id, filename, bytes, sha256, uploaded_at, server_created_at, purpose) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (file_id, kind, tokens, observation_id, filename, nbytes, sha256, iso_now(), server_created_at, purpose))

    def get_shared_file(self, tokens: int, sha256: str | None = None) -> dict | None:
        q = "SELECT * FROM files WHERE kind='shared' AND tokens=?"
        args: list[Any] = [tokens]
        if sha256:
            q += " AND sha256=?"
            args.append(sha256)
        row = self.conn.execute(q + " ORDER BY uploaded_at DESC LIMIT 1", args).fetchone()
        return dict(row) if row else None

    def get_individual_file(self, observation_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM files WHERE kind='individual' AND observation_id=?", (observation_id,)).fetchone()
        return dict(row) if row else None

    def list_files(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM files ORDER BY uploaded_at")]

    # ---------- jobs ----------
    def insert_jobs(self, rows: Iterable[dict[str, Any]], experiment_id: str) -> int:
        """INSERT OR IGNORE: re-running prepare never duplicates or reorders jobs."""
        n = 0
        with self.conn:
            self.conn.execute("BEGIN")
            for r in rows:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO jobs(observation_id, experiment_id, phase, attempt_id, parent_observation_id, "
                    "requested_output_tokens, api_max_output_tokens, launch_position, custom_id, input_file_id, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (r["observation_id"], experiment_id, r["phase"], r.get("attempt_id", 1), r.get("parent_observation_id"),
                     r["requested_output_tokens"], r["api_max_output_tokens"], r.get("launch_position"), r["custom_id"],
                     r.get("input_file_id"), iso_now()))
                n += cur.rowcount
            self.conn.execute("COMMIT")
        return n

    def set_job_input_file(self, observation_id: str, file_id: str) -> None:
        self.conn.execute("UPDATE jobs SET input_file_id=?, updated_at=? WHERE observation_id=?", (file_id, iso_now(), observation_id))

    def get_job(self, observation_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE observation_id=?", (observation_id,)).fetchone()
        return dict(row) if row else None

    def get_job_by_batch(self, batch_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE batch_id=?", (batch_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, phases: Iterable[str] | None = None, creation_state: str | None = None,
                  active_only: bool = False, terminal_uncollected: bool = False) -> list[dict]:
        q = "SELECT * FROM jobs WHERE 1=1"
        args: list[Any] = []
        if phases:
            ph = list(phases)
            q += f" AND phase IN ({','.join('?' * len(ph))})"
            args += ph
        if creation_state:
            q += " AND creation_state=?"
            args.append(creation_state)
        if active_only:
            q += " AND batch_id IS NOT NULL AND terminal=0"
        if terminal_uncollected:
            q += " AND terminal=1 AND collected=0"
        q += " ORDER BY COALESCE(launch_position, 0), observation_id"
        return [dict(r) for r in self.conn.execute(q, args)]

    def count_jobs(self, phases: Iterable[str] | None = None) -> dict[str, int]:
        q = "SELECT creation_state, COALESCE(status,'-') AS status, terminal, collected, COUNT(*) AS n FROM jobs"
        args: list[Any] = []
        if phases:
            ph = list(phases)
            q += f" WHERE phase IN ({','.join('?' * len(ph))})"
            args += ph
        q += " GROUP BY creation_state, status, terminal, collected"
        out: dict[str, int] = {}
        for r in self.conn.execute(q, args):
            out[f"{r['creation_state']}/{r['status']}/terminal={r['terminal']}/collected={r['collected']}"] = r["n"]
        return out

    # ---------- creation attempts ----------
    def begin_attempt(self, observation_id: str, wave: str, started_at: str, started_epoch: float) -> int:
        with self.conn:
            self.conn.execute("BEGIN")
            n = self.conn.execute("SELECT COUNT(*) FROM creation_attempts WHERE observation_id=?", (observation_id,)).fetchone()[0]
            cur = self.conn.execute(
                "INSERT INTO creation_attempts(observation_id, attempt_no, wave, started_at, started_epoch, outcome) VALUES (?,?,?,?,?,'in_flight')",
                (observation_id, n + 1, wave, started_at, started_epoch))
            self.conn.execute("UPDATE jobs SET creation_state='in_flight', local_create_started_at=?, updated_at=? WHERE observation_id=?",
                              (started_at, iso_now(), observation_id))
            self.conn.execute("COMMIT")
            return int(cur.lastrowid)

    def finish_attempt_created(self, attempt_id: int, observation_id: str, batch: dict[str, Any], finished_at: str,
                               http_status: int | None, request_id: str | None) -> None:
        with self.conn:
            self.conn.execute("BEGIN")
            self.conn.execute(
                "UPDATE creation_attempts SET finished_at=?, outcome='created', http_status=?, request_id=?, batch_id=? WHERE id=?",
                (finished_at, http_status, request_id, batch.get("id"), attempt_id))
            self.conn.execute(
                "UPDATE jobs SET creation_state='created', batch_id=?, local_create_finished_at=?, create_http_status=?, create_request_id=?, "
                "updated_at=? WHERE observation_id=?",
                (batch.get("id"), finished_at, http_status, request_id, iso_now(), observation_id))
            self._apply_batch_fields(observation_id, batch, poll_iso=None)
            self.conn.execute("COMMIT")

    def finish_attempt_failed(self, attempt_id: int, observation_id: str, outcome: str, finished_at: str,
                              http_status: int | None, request_id: str | None, error_type: str | None,
                              error_code: str | None, error_message: str | None) -> None:
        assert outcome in ("error", "unknown")
        with self.conn:
            self.conn.execute("BEGIN")
            self.conn.execute(
                "UPDATE creation_attempts SET finished_at=?, outcome=?, http_status=?, request_id=?, error_type=?, error_code=?, error_message=? WHERE id=?",
                (finished_at, outcome, http_status, request_id, error_type, error_code, error_message, attempt_id))
            self.conn.execute(
                "UPDATE jobs SET creation_state=?, local_create_finished_at=?, create_http_status=?, create_request_id=?, "
                "create_error_type=?, create_error_code=?, create_error_message=?, updated_at=? WHERE observation_id=?",
                (outcome, finished_at, http_status, request_id, error_type, error_code, error_message, iso_now(), observation_id))
            self.conn.execute("COMMIT")

    def attempts_for(self, observation_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM creation_attempts WHERE observation_id=? ORDER BY id", (observation_id,))]

    def all_attempts(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM creation_attempts ORDER BY id")]

    def recent_creation_epochs(self, since_epoch: float) -> list[float]:
        """Start times of attempts that may have created a batch (created/unknown/in_flight) since the epoch."""
        return [r[0] for r in self.conn.execute(
            "SELECT started_epoch FROM creation_attempts WHERE started_epoch >= ? AND outcome IN ('created','unknown','in_flight') ORDER BY started_epoch",
            (since_epoch,))]

    def count_creations_since(self, since_epoch: float) -> int:
        return len(self.recent_creation_epochs(since_epoch))

    def adopt_batch(self, observation_id: str, batch: dict[str, Any]) -> None:
        """Reconcile: a batch exists server-side for this observation but the local write was lost."""
        with self.conn:
            self.conn.execute("BEGIN")
            self.conn.execute("UPDATE jobs SET creation_state='created', batch_id=?, updated_at=? WHERE observation_id=?",
                              (batch.get("id"), iso_now(), observation_id))
            self.conn.execute("UPDATE creation_attempts SET outcome='created', batch_id=? WHERE observation_id=? AND outcome IN ('unknown','in_flight')",
                              (batch.get("id"), observation_id))
            self._apply_batch_fields(observation_id, batch, poll_iso=None)
            self.conn.execute("COMMIT")

    # ---------- polling ----------
    def _apply_batch_fields(self, observation_id: str, b: dict[str, Any], poll_iso: str | None) -> bool:
        """Update server-side fields. Returns True if the job transitioned to terminal in this call."""
        status = b.get("status")
        was_terminal = self.conn.execute("SELECT terminal FROM jobs WHERE observation_id=?", (observation_id,)).fetchone()[0]
        now_terminal = 1 if status in TERMINAL_STATUSES else 0
        sets = ["status=?", "request_counts=?", "output_file_id=?", "error_file_id=?", "batch_errors=?", "batch_usage=?",
                "batch_model=?", "updated_at=?"]
        args: list[Any] = [status, _j(b.get("request_counts")), b.get("output_file_id"), b.get("error_file_id"),
                           _j(b.get("errors")), _j(b.get("usage")), b.get("model"), iso_now()]
        for f in BATCH_TS_FIELDS:
            sets.append(f"{f}=?")
            args.append(b.get(f))
        if poll_iso:
            sets.append("last_poll_at=?")
            args.append(poll_iso)
            sets.append("poll_count=poll_count+1")
        if now_terminal and not was_terminal:
            sets.append("terminal=1")
            sets.append("terminal_seen_at=?")
            args.append(poll_iso or iso_now())
        args.append(observation_id)
        self.conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE observation_id=?", args)
        return bool(now_terminal and not was_terminal)

    def apply_batch_object(self, observation_id: str, b: dict[str, Any], poll_iso: str) -> bool:
        with self.conn:
            self.conn.execute("BEGIN")
            transitioned = self._apply_batch_fields(observation_id, b, poll_iso)
            self.conn.execute("COMMIT")
        return transitioned

    def apply_batch_objects(self, updates: list[tuple[str, dict[str, Any]]], poll_iso: str) -> list[str]:
        """Apply a whole poll cycle in ONE transaction; returns the observation_ids that became terminal."""
        transitioned: list[str] = []
        with self.conn:
            self.conn.execute("BEGIN")
            for observation_id, b in updates:
                if self._apply_batch_fields(observation_id, b, poll_iso):
                    transitioned.append(observation_id)
            self.conn.execute("COMMIT")
        return transitioned

    # ---------- results ----------
    def upsert_result(self, observation_id: str, r: dict[str, Any]) -> None:
        cols = ["observation_id"] + list(RESULT_FIELDS)
        vals = [observation_id] + [r.get(k) for k in RESULT_FIELDS]
        placeholders = ",".join("?" * len(cols))
        updates = ",".join(f"{c}=excluded.{c}" for c in RESULT_FIELDS)
        self.conn.execute(f"INSERT INTO results({','.join(cols)}) VALUES ({placeholders}) ON CONFLICT(observation_id) DO UPDATE SET {updates}", vals)

    def mark_collected(self, observation_id: str) -> None:
        self.conn.execute("UPDATE jobs SET collected=1, collected_at=?, updated_at=? WHERE observation_id=?",
                          (iso_now(), iso_now(), observation_id))

    def get_result(self, observation_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM results WHERE observation_id=?", (observation_id,)).fetchone()
        return dict(row) if row else None

    def list_results(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM results")]

    def jobs_with_results(self, phases: Iterable[str] | None = None) -> list[dict]:
        q = ("SELECT j.*, r.response_status, r.incomplete_reason, r.http_status AS result_http_status, r.openai_request_id, "
             "r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens, r.total_tokens, "
             "r.error_code AS result_error_code, r.error_type AS result_error_type, r.error_message AS result_error_message, "
             "r.output_text_chars, r.output_word_count, r.parse_error, r.source_kind "
             "FROM jobs j LEFT JOIN results r ON r.observation_id=j.observation_id WHERE 1=1")
        args: list[Any] = []
        if phases:
            ph = list(phases)
            q += f" AND j.phase IN ({','.join('?' * len(ph))})"
            args += ph
        q += " ORDER BY COALESCE(j.launch_position, 0), j.observation_id"
        return [dict(r) for r in self.conn.execute(q, args)]
