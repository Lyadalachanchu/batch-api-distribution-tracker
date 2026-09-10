"""Unit tests for the deterministic building blocks.

Covers: exactly 400 jobs per group (1), exactly 2,000 production jobs (2), randomized and reproducible
launch order (3), JSONL validity (4), unique observation identifiers (5), timestamp calculations (7),
and CLI --help parsing for every subcommand."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from experiment.__main__ import build_parser
from experiment.config import DEFAULT_OUTPUT_TOKEN_LEVELS, ORIGINAL_REQUESTED_LEVELS, ExperimentConfig
from experiment.jsonl import (JsonlValidationError, build_request_body, build_request_line, serialize_lines,
                              shared_custom_id, validate_jsonl_bytes, write_jsonl_file)
from experiment.manifest import (MANIFEST_COLUMNS, build_manifest, check_manifest, observation_id, read_manifest_csv,
                                 write_manifest_csv)
from experiment.observations import build_observations
from experiment.pilot import pilot_job_rows
from experiment.prepare import prepare
from experiment.recover import plan_replacements
from experiment.store import Store
from experiment.timeutil import epoch_to_iso, iso_to_epoch
from tests.conftest import make_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIVE_LEVELS = [16, 100, 1000, 3000, 10000]
RUNS = 400
SEED = 20260910
SUBCOMMANDS = ["prepare", "pilot", "launch", "monitor", "collect", "recover", "analyze", "status", "reconcile"]


# ---------------------------------------------------------------------------------------------
# 1. exactly 400 jobs per group / 2. exactly 2,000 production jobs
# ---------------------------------------------------------------------------------------------

def test_exactly_400_jobs_per_group_and_exactly_2000_production_jobs_in_manifest():
    """Items 1 and 2: build_manifest(5 levels, 400, seed) is valid and has exact counts."""
    assert FIVE_LEVELS == DEFAULT_OUTPUT_TOKEN_LEVELS
    rows = build_manifest(FIVE_LEVELS, RUNS, SEED)
    assert check_manifest(rows, FIVE_LEVELS, RUNS) == []
    assert len(rows) == 2000
    assert Counter(r["requested_output_tokens"] for r in rows) == {n: 400 for n in FIVE_LEVELS}
    assert Counter(r["api_max_output_tokens"] for r in rows) == {n: 400 for n in FIVE_LEVELS}
    assert all(r["phase"] == "prod" and r["attempt_id"] == 1 and r["seed"] == SEED for r in rows)
    assert all(list(r) == MANIFEST_COLUMNS for r in rows)
    # without explicit custom ids the custom_id is the observation id
    assert all(r["custom_id"] == r["observation_id"] for r in rows)
    custom = {n: f"exp:t{n:05d}" for n in FIVE_LEVELS}
    rows2 = build_manifest(FIVE_LEVELS, RUNS, SEED, custom_ids=custom)
    assert all(r["custom_id"] == custom[r["requested_output_tokens"]] for r in rows2)


def test_check_manifest_detects_wrong_counts_duplicates_and_bad_positions():
    """Items 1 and 2: the validator flags every deviation from exact counts."""
    rows = build_manifest([16, 100], 3, 1)
    assert check_manifest(rows, [16, 100], 3) == []
    assert any("expected 3" in p for p in check_manifest(rows[:-1], [16, 100], 3))
    assert any("total rows" in p for p in check_manifest(rows[:-1], [16, 100], 3))
    assert any("expected 4" in p for p in check_manifest(rows, [16, 100], 4))
    dup = [dict(r) for r in rows]
    dup[0]["observation_id"] = dup[1]["observation_id"]
    assert any("duplicate observation_id" in p for p in check_manifest(dup, [16, 100], 3))
    bad_pos = [dict(r) for r in rows]
    bad_pos[0]["launch_position"] = 99
    assert any("permutation" in p for p in check_manifest(bad_pos, [16, 100], 3))
    assert any("unexpected levels" in p for p in check_manifest(rows, [16], 3))


def test_manifest_csv_roundtrip(tmp_path):
    rows = build_manifest([16, 100], 2, 3, custom_ids={16: "c16", 100: "c100"})
    path = str(tmp_path / "m" / "launch_manifest.csv")
    write_manifest_csv(rows, path)
    back = read_manifest_csv(path)
    assert back == rows
    assert check_manifest(back, [16, 100], 2) == []


async def test_prepare_inserts_exactly_2000_production_jobs_400_per_level_and_zero_creations(workdir, fake_api):
    """Items 1 and 2 end to end: prepare with the 5 levels x 400 runs (uploads via the fake API)."""
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=RUNS, seed=SEED, levels=FIVE_LEVELS,
                     experiment_id="exp-prod", data_dir=str(workdir / "data"), no_upload=False)
    summary = await prepare(args, api=fake_api)
    assert summary["total_production_jobs"] == 2000
    assert summary["jobs_inserted_now"] == 2000 and summary["jobs_in_db"] == 2000
    assert summary["runs_per_group"] == 400 and summary["levels"] == FIVE_LEVELS
    assert summary["batch_creations_made"] == 0 and fake_api.create_calls == 0 and len(fake_api.batches) == 0
    # one shared input per level actually used: 5 production levels + the pilot-only levels 1 and 10
    assert fake_api.upload_calls == len(set(ORIGINAL_REQUESTED_LEVELS) | set(FIVE_LEVELS)) == 7
    assert set(summary["shared_files"]) == {str(n) for n in sorted(set(ORIGINAL_REQUESTED_LEVELS) | set(FIVE_LEVELS))}

    cfg = ExperimentConfig.load(args.config)
    store = Store(cfg.db_path)
    try:
        jobs = store.list_jobs(phases=["prod"])
        assert len(jobs) == 2000
        assert store.conn.execute("SELECT COUNT(*) FROM jobs WHERE phase='prod'").fetchone()[0] == 2000
        assert store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2000  # nothing but production rows
        assert store.conn.execute("SELECT COUNT(*) FROM creation_attempts").fetchone()[0] == 0
        assert Counter(j["requested_output_tokens"] for j in jobs) == {n: 400 for n in FIVE_LEVELS}
        assert all(j["creation_state"] == "pending" and j["batch_id"] is None and j["attempt_id"] == 1 for j in jobs)
        assert sorted(j["launch_position"] for j in jobs) == list(range(2000))
        # every job of a level points at that level's shared file
        by_level = {}
        for j in jobs:
            by_level.setdefault(j["requested_output_tokens"], set()).add(j["input_file_id"])
        assert all(len(v) == 1 for v in by_level.values()) and len({next(iter(v)) for v in by_level.values()}) == 5
        for n, fids in by_level.items():
            assert fids == {summary["shared_files"][str(n)]}
        db_positions = {j["observation_id"]: j["launch_position"] for j in jobs}
    finally:
        store.close()

    manifest = read_manifest_csv(cfg.manifest_path)
    assert check_manifest(manifest, FIVE_LEVELS, RUNS) == []
    assert {r["observation_id"]: r["launch_position"] for r in manifest} == db_positions
    # the manifest is the same one build_manifest produces for this seed
    expected = build_manifest(FIVE_LEVELS, RUNS, SEED, {n: shared_custom_id(cfg, n) for n in FIVE_LEVELS})
    assert manifest == expected


# ---------------------------------------------------------------------------------------------
# 3. randomized and reproducible launch order
# ---------------------------------------------------------------------------------------------

def test_randomized_and_reproducible_launch_order():
    a = build_manifest(FIVE_LEVELS, RUNS, SEED)
    b = build_manifest(FIVE_LEVELS, RUNS, SEED)
    assert a == b, "same seed must give an identical order"
    order_a = [r["observation_id"] for r in a]
    c = build_manifest(FIVE_LEVELS, RUNS, SEED + 1)
    assert [r["observation_id"] for r in c] != order_a, "a different seed must give a different order"
    assert sorted(order_a) == sorted(r["observation_id"] for r in c)  # same population, different permutation
    # not sorted by tokens in either direction, and not grouped by level
    tokens = [r["requested_output_tokens"] for r in a]
    assert tokens != sorted(tokens) and tokens != sorted(tokens, reverse=True)
    assert len({tokens[i] for i in range(50)}) == 5, "the first 50 positions already mix every level"
    # positions are a permutation of 0..N-1 in list order
    positions = [r["launch_position"] for r in a]
    assert positions == list(range(2000))
    assert len(set(positions)) == 2000
    # rank correlation between token level and launch position is negligible
    rho, _ = spearmanr(positions, tokens)
    assert abs(rho) < 0.1, f"spearman rho {rho:.3f} suggests a token/position trend"
    # per-group mean position is close to the centre of the wave (uniform spread)
    centre = (2000 - 1) / 2
    for n in FIVE_LEVELS:
        pos = [r["launch_position"] for r in a if r["requested_output_tokens"] == n]
        assert len(pos) == 400
        assert abs(float(np.mean(pos)) - centre) <= 0.15 * centre, f"level {n} mean position {np.mean(pos):.1f}"
        # each level also appears in every fifth of the wave
        assert {p // 400 for p in pos} == {0, 1, 2, 3, 4}


def test_randomized_order_is_independent_of_level_input_order():
    """The shuffle acts on a canonical (sorted-levels) row list, so the level order given by the caller is irrelevant."""
    assert build_manifest([10000, 16, 3000, 100, 1000], 400, SEED) == build_manifest(FIVE_LEVELS, 400, SEED)


# ---------------------------------------------------------------------------------------------
# 4. JSONL validity
# ---------------------------------------------------------------------------------------------

def test_jsonl_validity_built_line_is_accepted_and_body_is_exact():
    cfg = ExperimentConfig(reasoning_effort=None)
    line = build_request_line(cfg, "exp:t00100", 100)
    assert line == {"custom_id": "exp:t00100", "method": "POST", "url": "/v1/responses",
                    "body": {"model": cfg.model, "input": cfg.prompt, "max_output_tokens": 100}}
    assert list(line["body"]) == ["model", "input", "max_output_tokens"], "body carries no reasoning field by default"
    assert isinstance(line["body"]["max_output_tokens"], int)
    data = serialize_lines([line])
    assert data.endswith(b"\n") and data.count(b"\n") == 1 and b"\r" not in data
    parsed = validate_jsonl_bytes(data, cfg.endpoint, expect_lines=1)
    assert parsed == [line]
    # several distinct lines validate too
    many = [build_request_line(cfg, f"id-{i}", 16 + i) for i in range(5)]
    assert validate_jsonl_bytes(serialize_lines(many), cfg.endpoint, expect_lines=5) == many


def test_jsonl_validity_body_includes_reasoning_only_when_effort_set():
    cfg = ExperimentConfig(reasoning_effort="low")
    body = build_request_body(cfg, 16)
    assert body == {"model": cfg.model, "input": cfg.prompt, "max_output_tokens": 16, "reasoning": {"effort": "low"}}
    validate_jsonl_bytes(serialize_lines([build_request_line(cfg, "x", 16)]), cfg.endpoint, expect_lines=1)
    for effort in (None, ""):
        cfg.reasoning_effort = effort
        assert "reasoning" not in build_request_body(cfg, 16)


_DEL = object()  # sentinel: remove this key from the request line


def _line(**over):
    base = {"custom_id": "a", "method": "POST", "url": "/v1/responses",
            "body": {"model": "gpt-5.6-luna", "input": "hi", "max_output_tokens": 16}}
    for k, v in over.items():
        target, key = (base["body"], k[5:]) if k.startswith("body.") else (base, k)
        if v is _DEL:
            target.pop(key)
        else:
            target[key] = v
    return base


def _bytes(*lines):
    return serialize_lines(list(lines))


@pytest.mark.parametrize("name,data,match", [
    ("empty file", b"", "empty file"),
    ("only blank lines", b"\n\n", "no request lines"),
    ("bad JSON", b'{"custom_id": "a", "method": "POST"', "invalid JSON"),
    ("not an object", b"[1, 2]\n", "not an object"),
    ("missing custom_id", _bytes(_line(custom_id=_DEL)), "missing 'custom_id'"),
    ("empty custom_id", _bytes(_line(custom_id="")), "non-empty string"),
    ("non-string custom_id", _bytes(_line(custom_id=5)), "non-empty string"),
    ("wrong url", _bytes(_line(url="/v1/chat/completions")), "url"),
    ("wrong method", _bytes(_line(method="GET")), "method must be POST"),
    ("missing body", _bytes(_line(body=_DEL)), "missing 'body'"),
    ("body without model", _bytes(_line(**{"body.model": _DEL})), "model"),
    ("body without input", _bytes(_line(**{"body.input": _DEL})), "body.input missing"),
    ("duplicate custom_id", _bytes(_line(), _line()), "duplicate custom_id"),
    ("string max_output_tokens", _bytes(_line(**{"body.max_output_tokens": "100"})), "positive integer"),
    ("float max_output_tokens", _bytes(_line(**{"body.max_output_tokens": 100.0})), "positive integer"),
    ("bool max_output_tokens", _bytes(_line(**{"body.max_output_tokens": True})), "positive integer"),
    ("zero max_output_tokens", _bytes(_line(**{"body.max_output_tokens": 0})), "positive integer"),
    ("missing max_output_tokens", _bytes(_line(**{"body.max_output_tokens": None})), "positive integer"),
])
def test_jsonl_validity_rejects_bad_input(name, data, match):
    with pytest.raises(JsonlValidationError, match=match):
        validate_jsonl_bytes(data, "/v1/responses")


def test_jsonl_validity_expect_lines_and_file_writer(tmp_path):
    cfg = ExperimentConfig()
    line = build_request_line(cfg, "x", 16)
    with pytest.raises(JsonlValidationError, match="expected 2 lines, found 1"):
        validate_jsonl_bytes(serialize_lines([line]), cfg.endpoint, expect_lines=2)
    path = str(tmp_path / "inputs" / "t00016.jsonl")
    data, sha = write_jsonl_file(path, [line])
    assert open(path, "rb").read() == data == serialize_lines([line])
    assert sha == hashlib.sha256(data).hexdigest()
    assert validate_jsonl_bytes(data, cfg.endpoint, expect_lines=1) == [line]
    assert shared_custom_id(cfg, 100) == f"{cfg.experiment_id}:t00100"


# ---------------------------------------------------------------------------------------------
# 5. unique observation identifiers
# ---------------------------------------------------------------------------------------------

def test_unique_observation_identifiers_across_prod_pilot_and_replacements():
    prod_ids = [r["observation_id"] for r in build_manifest(FIVE_LEVELS, RUNS, SEED)]
    assert len(set(prod_ids)) == 2000
    assert all(pid.startswith("prod-t") for pid in prod_ids)
    assert observation_id("prod", 100, 1) == "prod-t00100-k0001"
    assert observation_id("prod", 10000, 399) == "prod-t10000-k0399"
    assert observation_id("prod", 100, 1, attempt=2) == "prod-t00100-k0001-a2"
    assert observation_id("pilot", 10, 0) == "pilot-t00010-k0000"

    cfg = ExperimentConfig(output_token_levels=FIVE_LEVELS)
    waves = pilot_job_rows(cfg, {n: f"file-{n}" for n in cfg.pilot_levels})
    pilot_rows = [r for w in waves for r in w]
    pilot_ids = [r["observation_id"] for r in pilot_rows]
    assert len(waves) == 2 and len(waves[0]) == 1 and len(waves[1]) == len(cfg.pilot_levels) + 1
    assert len(set(pilot_ids)) == len(pilot_ids) == 1 + len(cfg.pilot_levels) + 1
    assert all(pid.startswith("pilot-t") and r["phase"] == "pilot" for pid, r in zip(pilot_ids, pilot_rows))
    assert not set(pilot_ids) & set(prod_ids)
    assert all(r["input_file_id"] == f"file-{r['requested_output_tokens']}" for r in pilot_rows)

    replacement_ids = [observation_id("prod", n, k, attempt) for n in FIVE_LEVELS for k in range(RUNS) for attempt in (2, 3)]
    assert len(set(replacement_ids)) == len(replacement_ids)
    assert not set(replacement_ids) & set(prod_ids)
    assert not set(replacement_ids) & set(pilot_ids)


def test_unique_observation_identifiers_replacement_planning(workdir):
    """plan_replacements derives non-colliding ids (-a2, then -a3) that point back at the original row."""
    store = Store(str(workdir / "data" / "state.sqlite"))
    try:
        rows = build_manifest([16, 100], 2, 1)
        store.insert_jobs(rows, "exp")
        obs = "prod-t00100-k0001"
        store.conn.execute("UPDATE jobs SET creation_state='error' WHERE observation_id=?", (obs,))
        plan = plan_replacements(SimpleNamespace(store=store))
        assert len(plan) == 1
        rep = plan[0]
        assert rep["observation_id"] == obs + "-a2" and rep["attempt_id"] == 2 and rep["phase"] == "replacement"
        assert rep["parent_observation_id"] == obs and rep["requested_output_tokens"] == 100
        assert rep["observation_id"] not in {r["observation_id"] for r in rows}
        # a live (pending/created) replacement suppresses further planning for that root
        store.insert_jobs(plan, "exp")
        assert plan_replacements(SimpleNamespace(store=store)) == []
        # a failed replacement leads to -a3, still rooted at the original id and colliding with nothing
        store.conn.execute("UPDATE jobs SET creation_state='error' WHERE observation_id=?", (rep["observation_id"],))
        plan2 = plan_replacements(SimpleNamespace(store=store))
        assert {p["observation_id"] for p in plan2} == {obs + "-a3"}
        assert all(p["attempt_id"] == 3 and p["parent_observation_id"] == obs for p in plan2)
        assert obs + "-a3" not in {j["observation_id"] for j in store.list_jobs()}
        store.insert_jobs(plan2, "exp")
        assert store.get_job(obs + "-a3")["attempt_id"] == 3
        assert plan_replacements(SimpleNamespace(store=store)) == []
        assert len({j["observation_id"] for j in store.list_jobs()}) == len(store.list_jobs()) == 6
    finally:
        store.close()


def test_unique_observation_identifiers_replacement_plan_has_no_duplicate_rows(workdir):
    store = Store(str(workdir / "data" / "state.sqlite"))
    try:
        obs = "prod-t00100-k0001"
        store.insert_jobs(build_manifest([16, 100], 2, 1), "exp")
        store.conn.execute("UPDATE jobs SET creation_state='error' WHERE observation_id=?", (obs,))
        store.insert_jobs(plan_replacements(SimpleNamespace(store=store)), "exp")
        store.conn.execute("UPDATE jobs SET creation_state='error' WHERE observation_id=?", (obs + "-a2",))
        plan2 = plan_replacements(SimpleNamespace(store=store))
        assert [p["observation_id"] for p in plan2] == [obs + "-a3"]
    finally:
        store.close()


# ---------------------------------------------------------------------------------------------
# 7. timestamp calculations
# ---------------------------------------------------------------------------------------------

def _job(obs, tokens, pos):
    return {"observation_id": obs, "phase": "prod", "attempt_id": 1, "requested_output_tokens": tokens,
            "api_max_output_tokens": tokens, "launch_position": pos, "custom_id": f"c{tokens}", "input_file_id": f"file-{tokens}"}


async def test_timestamp_calculations_from_batch_objects(workdir):
    """Item 7: turnaround/queue/active/offset/local duration arithmetic on hand-made batch objects."""
    cfg = ExperimentConfig(experiment_id="exp-ts", output_token_levels=[16, 100], runs_per_group=2,
                           data_dir=str(workdir / "data"), config_path=str(workdir / "config" / "experiment.json"))
    store = Store(cfg.db_path)
    try:
        launch_iso = "2026-09-10T12:00:00.000000+00:00"
        launch_epoch = iso_to_epoch(launch_iso)
        base = int(launch_epoch)
        store.set_meta("launch_started_at", launch_iso)
        OK, FAILED, EXPIRED, NEVER = "prod-t00100-k0000", "prod-t00016-k0000", "prod-t00016-k0001", "prod-t00100-k0001"
        store.insert_jobs([_job(OK, 100, 0), _job(FAILED, 16, 1), _job(EXPIRED, 16, 2), _job(NEVER, 100, 3)], cfg.experiment_id)
        poll_iso = "2026-09-10T12:05:00.000000+00:00"

        def create(obs, batch, started, finished):
            aid = store.begin_attempt(obs, "launch", started, iso_to_epoch(started))
            store.finish_attempt_created(aid, obs, batch, finished, 200, "req_" + obs)

        ok_ca = base + 5
        ok_batch = {"id": "batch_ok", "status": "validating", "created_at": ok_ca, "expires_at": ok_ca + 86400}
        create(OK, ok_batch, "2026-09-10T12:00:04.250000+00:00", "2026-09-10T12:00:04.750000+00:00")
        assert store.get_job(OK)["status"] == "validating" and store.get_job(OK)["terminal"] == 0
        final_ok = {**ok_batch, "status": "completed", "in_progress_at": ok_ca + 30, "finalizing_at": ok_ca + 50,
                    "completed_at": ok_ca + 51, "output_file_id": "file-out", "request_counts": {"total": 1, "completed": 1, "failed": 0}}
        assert store.apply_batch_object(OK, final_ok, poll_iso) is True
        assert store.apply_batch_object(OK, final_ok, poll_iso) is False  # already terminal: no second transition
        store.upsert_result(OK, {"response_status": "incomplete", "incomplete_reason": "max_output_tokens", "input_tokens": 45,
                                 "cached_input_tokens": 0, "output_tokens": 100, "reasoning_tokens": 45, "total_tokens": 145,
                                 "http_status": 200, "openai_request_id": "req_x"})
        store.mark_collected(OK)

        f_ca = base + 6
        create(FAILED, {"id": "batch_failed", "status": "validating", "created_at": f_ca},
               "2026-09-10T12:00:05.000000+00:00", "2026-09-10T12:00:05.100000+00:00")
        store.apply_batch_object(FAILED, {"id": "batch_failed", "status": "failed", "created_at": f_ca, "failed_at": f_ca + 3,
                                          "errors": {"object": "list", "data": [{"code": "invalid_json_line", "message": "bad"}]}}, poll_iso)

        e_ca = base + 7
        create(EXPIRED, {"id": "batch_expired", "status": "validating", "created_at": e_ca},
               "2026-09-10T12:00:06.000000+00:00", "2026-09-10T12:00:06.020000+00:00")
        store.apply_batch_object(EXPIRED, {"id": "batch_expired", "status": "expired", "created_at": e_ca, "in_progress_at": e_ca + 30,
                                           "expired_at": e_ca + 86400, "error_file_id": "file-err"}, poll_iso)

        aid = store.begin_attempt(NEVER, "launch", "2026-09-10T12:00:07.000000+00:00", iso_to_epoch("2026-09-10T12:00:07.000000+00:00"))
        store.finish_attempt_failed(aid, NEVER, "error", "2026-09-10T12:00:07.500000+00:00", 500, "req_n", "server_error", None, "boom")

        df = build_observations(store, cfg)
        assert len(df) == 4 and list(df.randomized_launch_position) == [0, 1, 2, 3]
        df = df.set_index("observation_id")

        ok = df.loc[OK]
        assert ok["created_at"] == ok_ca and ok["in_progress_at"] == ok_ca + 30 and ok["completed_at"] == ok_ca + 51
        assert ok["turnaround_seconds"] == 51 == ok["completed_at"] - ok["created_at"]
        assert ok["queue_seconds"] == 30 == ok["in_progress_at"] - ok["created_at"]
        assert ok["active_seconds"] == 21 == ok["completed_at"] - ok["in_progress_at"]
        assert ok["finalizing_seconds"] == 1
        assert ok["server_created_at_offset_seconds"] == 5 == ok_ca - launch_epoch
        assert ok["local_create_duration_ms"] == 500.0
        assert ok["local_create_started_at"] == "2026-09-10T12:00:04.250000+00:00"
        assert ok["created_at_iso"] == "2026-09-10T12:00:05+00:00" == epoch_to_iso(ok_ca)
        assert ok["in_progress_at_iso"] == epoch_to_iso(ok_ca + 30) and ok["completed_at_iso"] == epoch_to_iso(ok_ca + 51)
        assert ok["status"] == "completed" and ok["creation_state"] == "created" and ok["local_terminal_seen_at"] == poll_iso
        assert ok["valid_observation"] and ok["early_stop"] == False and ok["finish_or_incomplete_reason"] == "max_output_tokens"  # noqa: E712
        assert ok["http_status"] == 200 and ok["estimated_cost_usd"] == pytest.approx((45 * 0.10 + 100 * 0.60) / 1e6)

        f = df.loc[FAILED]
        assert f["status"] == "failed" and f["failed_at"] == f_ca + 3
        assert pd.isna(f["completed_at"]) and pd.isna(f["in_progress_at"])
        assert pd.isna(f["turnaround_seconds"]) and pd.isna(f["queue_seconds"]) and pd.isna(f["active_seconds"])
        assert f["server_created_at_offset_seconds"] == 6 and f["local_create_duration_ms"] == 100.0
        assert not f["valid_observation"]

        e = df.loc[EXPIRED]
        assert e["status"] == "expired" and e["expired_at"] == e_ca + 86400
        assert pd.isna(e["turnaround_seconds"]) and pd.isna(e["active_seconds"])
        assert e["queue_seconds"] == 30 and e["server_created_at_offset_seconds"] == 7
        assert e["local_create_duration_ms"] == 20.0 and not e["valid_observation"]

        n = df.loc[NEVER]
        assert n["status"] == "creation_error" and pd.isna(n["batch_id"]) and pd.isna(n["created_at"])
        assert pd.isna(n["turnaround_seconds"]) and pd.isna(n["server_created_at_offset_seconds"]) and pd.isna(n["created_at_iso"])
        assert n["local_create_duration_ms"] == 500.0 and n["http_status"] == 500 and n["create_error_message"] == "boom"
        assert n["error_type"] == "server_error" and not n["valid_observation"]

        # without a recorded launch start the server offset is undefined for every row
        store.conn.execute("DELETE FROM meta WHERE key='launch_started_at'")
        df2 = build_observations(store, cfg)
        assert df2.server_created_at_offset_seconds.isna().all()
        assert (df2.set_index("observation_id").loc[OK]["turnaround_seconds"]) == 51
    finally:
        store.close()


def test_timestamp_helpers_roundtrip():
    iso = "2026-09-10T12:00:04.250000+00:00"
    assert iso_to_epoch(iso) == pytest.approx(1789041604.25)
    assert epoch_to_iso(1789041604) == "2026-09-10T12:00:04+00:00"
    assert epoch_to_iso(None) is None and iso_to_epoch(None) is None and iso_to_epoch("") is None
    assert iso_to_epoch(epoch_to_iso(1800000000)) == 1800000000.0


# ---------------------------------------------------------------------------------------------
# CLI: --help for the program and every subcommand
# ---------------------------------------------------------------------------------------------

def test_cli_help_parses_for_program_and_every_subcommand(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as e:
        parser.parse_args(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert all(sub in out for sub in SUBCOMMANDS)
    for sub in SUBCOMMANDS:
        with pytest.raises(SystemExit) as e:
            parser.parse_args([sub, "--help"])
        assert e.value.code == 0, sub
        assert sub in capsys.readouterr().out
    # a subcommand is required
    with pytest.raises(SystemExit) as e:
        parser.parse_args([])
    assert e.value.code == 2
    # representative invocations parse into the expected namespace
    ns = parser.parse_args(["--config", "c.json", "launch", "--execute", "--concurrency", "5", "--wait", "--skip-pilot-gate"])
    assert (ns.command, ns.config, ns.execute, ns.concurrency, ns.wait, ns.skip_pilot_gate) == ("launch", "c.json", True, 5, True, True)
    ns = parser.parse_args(["prepare", "--levels", "16", "100", "--runs-per-group", "400", "--file-mode", "individual", "--no-upload"])
    assert ns.levels == [16, 100] and ns.runs_per_group == 400 and ns.file_mode == "individual" and ns.no_upload
    ns = parser.parse_args(["pilot", "--execute", "--max-cost-usd", "0.05", "--interval", "0.5"])
    assert ns.execute and ns.max_cost_usd == 0.05 and ns.interval == 0.5 and ns.again is False
    ns = parser.parse_args(["monitor", "--phases", "prod", "pilot", "--once"])
    assert ns.phases == ["prod", "pilot"] and ns.once
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare", "--file-mode", "bogus"])


def test_python_m_experiment_help_runs_as_a_process():
    proc = subprocess.run([sys.executable, "-m", "experiment", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "prepare" in proc.stdout and "launch" in proc.stdout and "usage:" in proc.stdout
    proc = subprocess.run([sys.executable, "-m", "experiment", "launch", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "--skip-pilot-gate" in proc.stdout
