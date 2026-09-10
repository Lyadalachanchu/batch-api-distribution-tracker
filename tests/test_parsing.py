"""Output parsing and terminal-state handling.

Covers: response-usage extraction (12), failed and expired jobs (13), malformed output files (14)."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import pandas as pd
import pytest

from experiment.collect import collect_all, collect_job
from experiment.config import ExperimentConfig
from experiment.events import read_jsonl
from experiment.launch import launch, spent_or_committed_usd
from experiment.monitor import monitor
from experiment.observations import build_observations, extract_output_text, parse_output_record
from experiment.prepare import prepare
from experiment.runtime import Runtime
from experiment.status import build_status
from tests.conftest import make_args
from tests.fake_api import FakeBatchApi


# ---------------------------------------------------------------------------------------------
# 12. response-usage extraction (record shapes copied from tests/fake_api.py)
# ---------------------------------------------------------------------------------------------

def _response_body(tokens=100, out_tokens=None, early=False, cached=3):
    out_tokens = tokens if out_tokens is None else out_tokens
    reasoning = min(out_tokens, 45)
    words = " ".join(f"w{i}" for i in range(max(0, out_tokens - reasoning)))
    return {"id": "resp_000001", "object": "response", "created_at": 1800000030,
            "status": "completed" if early else "incomplete",
            "incomplete_details": None if early else {"reason": "max_output_tokens"},
            "model": "gpt-5.6-luna", "max_output_tokens": tokens,
            "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                       {"type": "message", "id": "msg_1", "status": "completed" if early else "incomplete", "role": "assistant",
                        "content": [{"type": "output_text", "text": words, "annotations": []}]}],
            "usage": {"input_tokens": 45, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": out_tokens,
                      "output_tokens_details": {"reasoning_tokens": reasoning}, "total_tokens": 45 + out_tokens}}


def _output_line(body):
    return {"id": "batch_req_000002", "custom_id": "exp:t00100",
            "response": {"status_code": 200, "request_id": "req_000003", "body": body}, "error": None}


ERROR_400_LINE = {"id": "batch_req_000004", "custom_id": "exp:t00010",
                  "response": {"status_code": 400, "request_id": "req_000005",
                               "body": {"error": {"message": "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 10 instead.",
                                                  "type": "invalid_request_error", "param": "max_output_tokens", "code": "integer_below_min_value"}}},
                  "error": None}

EXPIRED_LINE = {"id": "batch_req_000006", "custom_id": "exp:t01000", "response": None,
                "error": {"code": "batch_expired", "message": "This request could not be executed before the completion window expired."}}


def test_response_usage_extraction_from_a_realistic_output_line():
    body = _response_body(tokens=100)
    line = _output_line(body)
    raw = json.dumps(line)
    r = parse_output_record(line, raw)
    text = body["output"][1]["content"][0]["text"]
    assert r["output_tokens"] == 100 and r["reasoning_tokens"] == 45 and r["input_tokens"] == 45
    assert r["cached_input_tokens"] == 3 and r["total_tokens"] == 145
    assert r["response_status"] == "incomplete" and r["incomplete_reason"] == "max_output_tokens"
    assert r["http_status"] == 200 and r["openai_request_id"] == "req_000003"
    assert r["output_word_count"] == 55 == len(text.split()) and r["output_text_chars"] == len(text)
    assert r["line_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert "parse_error" not in r and "error_code" not in r and "error_message" not in r
    # a completed (early-stopped) response
    r2 = parse_output_record(_output_line(_response_body(tokens=1000, out_tokens=500, early=True, cached=0)))
    assert r2["response_status"] == "completed" and r2["incomplete_reason"] is None
    assert r2["output_tokens"] == 500 and r2["reasoning_tokens"] == 45 and r2["cached_input_tokens"] == 0 and r2["line_sha256"] is None
    assert r2["output_word_count"] == 455


def test_response_usage_extraction_handles_missing_usage_details():
    body = _response_body()
    body["usage"] = {"input_tokens": 45, "output_tokens": 16, "total_tokens": 61}
    r = parse_output_record(_output_line(body))
    assert r["output_tokens"] == 16 and r["reasoning_tokens"] is None and r["cached_input_tokens"] is None
    body["usage"] = None
    r = parse_output_record(_output_line(body))
    assert r["response_status"] == "incomplete" and r["output_tokens"] is None and "parse_error" not in r
    assert r["output_word_count"] == 55


def test_response_usage_extraction_400_error_line():
    raw = json.dumps(ERROR_400_LINE)
    r = parse_output_record(ERROR_400_LINE, raw)
    assert r["http_status"] == 400 and r["openai_request_id"] == "req_000005"
    assert r["error_code"] == "integer_below_min_value" and r["error_type"] == "invalid_request_error"
    assert r["error_message"].startswith("Invalid 'max_output_tokens'")
    assert r.get("response_status") is None and r.get("output_tokens") is None and "parse_error" not in r
    assert r["line_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


def test_response_usage_extraction_batch_expired_line():
    r = parse_output_record(EXPIRED_LINE, json.dumps(EXPIRED_LINE))
    assert r["error_code"] == "batch_expired" and r["error_type"] == "batch_request_error"
    assert r["error_message"].startswith("This request could not be executed")
    assert r.get("http_status") is None and r.get("response_status") is None and "parse_error" not in r


@pytest.mark.parametrize("obj,match", [
    ("not a dict", "record is str, not an object"),
    (["x"], "record is list, not an object"),
    (None, "record is NoneType, not an object"),
    ({"id": "x", "custom_id": "c"}, "neither response nor error"),
    ({"response": None, "error": None}, "neither response nor error"),
    ({"response": "oops"}, "response is not an object"),
    ({"response": {"status_code": 200, "body": "text"}}, "response.body is not an object"),
])
def test_response_usage_extraction_never_raises_on_bad_records(obj, match):
    r = parse_output_record(obj, "raw")
    assert match in r["parse_error"] and r["line_sha256"] == hashlib.sha256(b"raw").hexdigest()


def test_extract_output_text_concatenates_message_parts_only():
    body = {"output": [{"type": "reasoning", "summary": [{"text": "hidden"}]},
                       {"type": "message", "content": [{"type": "output_text", "text": "a b"}, {"type": "refusal", "refusal": "no"},
                                                       {"type": "output_text", "text": " c"}]},
                       "garbage", {"type": "message", "content": None}]}
    assert extract_output_text(body) == "a b c"
    assert extract_output_text({}) == "" and extract_output_text({"output": None}) == ""


# ---------------------------------------------------------------------------------------------
# helpers for the fake pipeline
# ---------------------------------------------------------------------------------------------

async def _prepare_and_launch(workdir, api, levels=(16, 100), runs=1):
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=runs, seed=7, levels=list(levels),
                     experiment_id="exp-parse", data_dir=str(workdir / "data"))
    await prepare(args, api=api)
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, concurrency=2), api=api)
    assert out["wave"]["created"] == len(levels) * runs
    return args, ExperimentConfig.load(args.config)


# ---------------------------------------------------------------------------------------------
# 13. failed and expired jobs
# ---------------------------------------------------------------------------------------------

async def test_failed_and_expired_jobs_are_terminal_collected_and_kept_as_invalid_observations(workdir):
    EXPIRED, FAILED, OK = "prod-t00016-k0000", "prod-t00100-k0000", "prod-t01000-k0000"
    # the fake clock advances one hour per API call, so the 24 h completion window passes within a few dozen polls
    api = FakeBatchApi(auto_advance=3600.0, expire_for={EXPIRED}, fail_batch_for={FAILED})
    args, cfg = await _prepare_and_launch(workdir, api, levels=(16, 100, 1000))
    created_at = {b["metadata"]["observation_id"]: b["created_at"] for b in api.batches.values()}
    rt = Runtime.open(cfg, api=api)
    try:
        res = await monitor(rt, phases=["prod"], interval=0.01)
        assert res["remaining_active"] == 0 and res["transitions_total"] == 3
        assert api.create_calls == 3
        jobs = {j["observation_id"]: j for j in rt.store.jobs_with_results(phases=["prod"])}

        ex = jobs[EXPIRED]
        assert ex["status"] == "expired" and ex["terminal"] == 1 and ex["collected"] == 1
        assert ex["expired_at"] >= created_at[EXPIRED] + 86400 and ex["completed_at"] is None and ex["in_progress_at"] is not None
        assert ex["error_file_id"] and ex["output_file_id"] is None
        assert json.loads(ex["request_counts"]) == {"total": 1, "completed": 0, "failed": 1}
        assert ex["result_error_code"] == "batch_expired" and ex["result_error_type"] == "batch_request_error"
        assert "completion window expired" in ex["result_error_message"] and ex["source_kind"] == "error"
        assert ex["output_tokens"] is None and ex["parse_error"] is None

        fa = jobs[FAILED]
        assert fa["status"] == "failed" and fa["terminal"] == 1 and fa["collected"] == 1
        assert fa["failed_at"] is not None and fa["in_progress_at"] is None and fa["completed_at"] is None
        assert fa["output_file_id"] is None and fa["error_file_id"] is None
        assert json.loads(fa["batch_errors"])["data"][0]["code"] == "invalid_json_line"
        assert fa["result_error_type"] == "batch_validation_error" and fa["result_error_code"] == "invalid_json_line"
        assert fa["result_error_message"] == "This line is not parseable as valid JSON." and fa["source_kind"] == "batch.errors"
        assert fa["parse_error"] is None

        ok = jobs[OK]
        assert ok["status"] == "completed" and ok["output_tokens"] == 1000

        df = build_observations(rt.store, cfg)
        assert len(df) == 3 and Counter(df.status) == {"expired": 1, "failed": 1, "completed": 1}
        df = df.set_index("observation_id")
        assert not df.loc[EXPIRED]["valid_observation"] and not df.loc[FAILED]["valid_observation"] and df.loc[OK]["valid_observation"]
        assert df.loc[EXPIRED]["error_code"] == "batch_expired" and df.loc[FAILED]["error_code"] == "invalid_json_line"
        assert df.loc[EXPIRED]["expired_at"] == ex["expired_at"] and df.loc[FAILED]["failed_at"] == fa["failed_at"]
        assert pd.isna(df.loc[EXPIRED]["turnaround_seconds"]) and pd.isna(df.loc[FAILED]["turnaround_seconds"])
        assert df.loc[EXPIRED]["queue_seconds"] == ex["in_progress_at"] - ex["created_at"]
        assert pd.isna(df.loc[FAILED]["queue_seconds"]) and pd.isna(df.loc[EXPIRED]["active_seconds"])
        assert pd.isna(df.loc[EXPIRED]["actual_output_tokens"]) and pd.isna(df.loc[EXPIRED]["estimated_cost_usd"])
        assert df.loc[OK]["turnaround_seconds"] == df.loc[OK]["completed_at"] - df.loc[OK]["created_at"]
        # nothing is billed for the two jobs without usage
        spent = spent_or_committed_usd(rt)  # rounded to 6 decimals
        assert spent["committed_worst_case_usd"] == 0.0 and spent["actual_usd"] == pytest.approx((45 * 0.10 + 1000 * 0.60) / 1e6, abs=1e-6)
        st = build_status(rt)
        assert st["prod"]["by_status"] == {"expired": 1, "failed": 1, "completed": 1} and st["prod"]["terminal"] == 3
        # collect_all has nothing left, even with terminal jobs of every kind
        assert await collect_all(rt, phases=["prod"]) == {"collected": 0, "with_parse_error": 0, "total": 0}
    finally:
        await rt.aclose()
    # raw artifacts keep the definitive terminal objects and the expired error line
    finals = {f["observation_id"]: f["batch"] for f in read_jsonl(cfg.raw_path("batch_objects.jsonl"))}
    assert finals[EXPIRED]["status"] == "expired" and finals[FAILED]["status"] == "failed" and finals[FAILED]["errors"]["data"]
    errs = list(read_jsonl(cfg.raw_path("errors.jsonl")))
    assert [e["observation_id"] for e in errs] == [EXPIRED] and errs[0]["record"]["error"]["code"] == "batch_expired"
    polls = list(read_jsonl(cfg.raw_path("batch_poll_events.jsonl")))
    assert {p["status"] for p in polls if p["observation_id"] == EXPIRED} >= {"in_progress", "expired"}
    assert [p["status"] for p in polls if p["observation_id"] == FAILED] == ["failed"]


async def test_failed_and_expired_jobs_with_an_explicit_clock_jump(workdir):
    """Same outcome when the clock is advanced past created_at + 86400 s in one step."""
    EXPIRED = "prod-t00016-k0000"
    api = FakeBatchApi(auto_advance=15.0, expire_for={EXPIRED})
    args, cfg = await _prepare_and_launch(workdir, api, levels=(16,))
    api.advance(86400 + 60)
    rt = Runtime.open(cfg, api=api)
    try:
        res = await monitor(rt, phases=["prod"], interval=0.01)
        assert res["remaining_active"] == 0 and res["cycle"] == 1
        job = rt.store.get_job(EXPIRED)
        assert job["status"] == "expired" and job["terminal"] == 1 and job["collected"] == 1
        assert rt.store.get_result(EXPIRED)["error_code"] == "batch_expired"
        df = build_observations(rt.store, cfg)
        assert len(df) == 1 and not df.iloc[0]["valid_observation"] and df.iloc[0]["status"] == "expired"
    finally:
        await rt.aclose()


async def test_early_stop_is_flagged_but_remains_a_valid_observation(workdir):
    EARLY = "prod-t00100-k0000"
    api = FakeBatchApi(auto_advance=15.0, early_stop_for={EARLY})
    args, cfg = await _prepare_and_launch(workdir, api, levels=(100,))
    rt = Runtime.open(cfg, api=api)
    try:
        await monitor(rt, phases=["prod"], interval=0.01)
        row = build_observations(rt.store, cfg).iloc[0]
        assert row["response_status"] == "completed" and row["actual_output_tokens"] == 50 < row["requested_output_tokens"]
        assert row["early_stop"] == True and row["valid_observation"] and row["finish_or_incomplete_reason"] == "completed"  # noqa: E712
    finally:
        await rt.aclose()


# ---------------------------------------------------------------------------------------------
# 14. malformed output files
# ---------------------------------------------------------------------------------------------

async def test_malformed_output_file_does_not_raise_and_is_flagged_invalid(workdir):
    BAD, GOOD = "prod-t00016-k0000", "prod-t00100-k0000"
    api = FakeBatchApi(auto_advance=15.0, malformed_output_for={BAD})
    args, cfg = await _prepare_and_launch(workdir, api)
    rt = Runtime.open(cfg, api=api)
    try:
        res = await monitor(rt, phases=["prod"], interval=0.01)  # collects inline; must not raise
        assert res["remaining_active"] == 0
        job = rt.store.get_job(BAD)
        assert job["status"] == "completed" and job["collected"] == 1 and job["output_file_id"]
        r = rt.store.get_result(BAD)
        assert r["parse_error"] and "malformed JSON" in r["parse_error"] and "line 1" in r["parse_error"]
        assert r["output_tokens"] is None and r["response_status"] is None and r["batch_id"] == job["batch_id"]
        good = rt.store.get_result(GOOD)
        assert good["parse_error"] is None and good["output_tokens"] == 100
        # an explicit (forced) re-collect of the malformed job does not raise either and keeps the flag
        r2 = await collect_job(rt, rt.store.get_job(BAD), force=True)
        assert r2["parse_error"] and rt.store.get_result(BAD)["parse_error"]
        forced = await collect_all(rt, phases=["prod"], force=True)
        assert forced == {"collected": 1, "with_parse_error": 1, "total": 2}

        df = build_observations(rt.store, cfg).set_index("observation_id")
        assert not df.loc[BAD]["valid_observation"] and df.loc[BAD]["parse_error"] == r["parse_error"]
        assert df.loc[BAD]["status"] == "completed" and pd.isna(df.loc[BAD]["actual_output_tokens"])
        assert df.loc[GOOD]["valid_observation"]
    finally:
        await rt.aclose()
    # responses.jsonl holds a record for the malformed observation, marked as malformed
    recs = list(read_jsonl(cfg.raw_path("responses.jsonl")))
    bad = [x for x in recs if x["observation_id"] == BAD]
    assert bad and all(x["record"].get("__malformed__") is True and x["file_id"] == job["output_file_id"] for x in bad)
    assert [x for x in recs if x["observation_id"] == GOOD][0]["record"]["response"]["body"]["usage"]["output_tokens"] == 100
    assert any(x["recollect"] for x in bad) and not bad[0]["recollect"]


async def test_malformed_output_raw_line_is_preserved_in_responses_jsonl(workdir):
    BAD = "prod-t00016-k0000"
    api = FakeBatchApi(auto_advance=15.0, malformed_output_for={BAD})
    args, cfg = await _prepare_and_launch(workdir, api, levels=(16,))
    rt = Runtime.open(cfg, api=api)
    try:
        await monitor(rt, phases=["prod"], interval=0.01)
        raw_line = api.file_bytes[rt.store.get_job(BAD)["output_file_id"]].decode("utf-8").split("\n")[0]
    finally:
        await rt.aclose()
    text = open(cfg.raw_path("responses.jsonl"), encoding="utf-8").read()
    assert raw_line.strip() in text or json.dumps(raw_line.strip())[1:-1] in text
