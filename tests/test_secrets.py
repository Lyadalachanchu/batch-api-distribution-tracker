"""Absence of secrets from logs and artifacts (item 15)."""
from __future__ import annotations

import json
import logging
import os

import pytest

from experiment.collect import collect_all
from experiment.config import ConfigError, ExperimentConfig
from experiment.events import EventWriter, read_jsonl
from experiment.launch import launch
from experiment.logging_utils import RedactingFormatter, setup_logging
from experiment.monitor import monitor
from experiment.observations import build_observations, write_observations
from experiment.pilot import pilot, scan_for_secrets
from experiment.prepare import prepare
from experiment.recover import recover
from experiment.redact import REDACTED, contains_secret, redact_obj, redact_text
from experiment.runtime import Runtime
from tests.conftest import make_args
from tests.fake_api import FakeBatchApi

KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
KEY2 = "sk-svcacct-ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210"


# ---------------------------------------------------------------------------------------------
# redaction primitives
# ---------------------------------------------------------------------------------------------

def test_redact_text_masks_keys_and_bearer_tokens():
    assert redact_text(f"key={KEY}") == f"key={REDACTED}"
    assert KEY not in redact_text(f"a {KEY} b {KEY2} c") and redact_text(f"a {KEY} b {KEY2} c").count(REDACTED) == 2
    assert redact_text("Authorization: Bearer abcdefghijklmnop.123-456") == "Authorization: Bearer ***REDACTED***"
    assert redact_text("bearer abcdefghijklmnop") == "bearer ***REDACTED***"
    assert redact_text(f"Bearer {KEY}") == f"Bearer {REDACTED}"
    assert redact_text("") == "" and redact_text("no secrets here") == "no secrets here"
    # the marker itself is stable under repeated redaction
    assert redact_text(redact_text(KEY)) == REDACTED
    # short 'sk-' fragments (e.g. the word "task-list") are not keys
    assert redact_text("task-list sk-abc") == "task-list sk-abc"


def test_contains_secret_detects_keys_but_not_redacted_text():
    assert contains_secret(KEY) and contains_secret(f"x {KEY2} y") and contains_secret("Bearer abcdefghijklmnop")
    assert not contains_secret(redact_text(KEY))
    assert not contains_secret(REDACTED)
    assert not contains_secret("") and not contains_secret("sk-short") and not contains_secret("plain text")
    assert not contains_secret(json.dumps(redact_obj({"k": KEY, "b": f"Bearer {KEY}"})))


def test_redact_obj_walks_nested_structures():
    obj = {"a": KEY, "b": [KEY, {"c": KEY}], "d": (KEY, 1), "n": 5, "none": None, "f": 1.5, "t": True, "msg": f"Bearer {KEY}"}
    out = redact_obj(obj)
    assert out["a"] == REDACTED and out["b"] == [REDACTED, {"c": REDACTED}] and out["d"] == [REDACTED, 1]
    assert out["n"] == 5 and out["none"] is None and out["f"] == 1.5 and out["t"] is True
    assert out["msg"] == f"Bearer {REDACTED}"
    assert KEY not in json.dumps(out)
    assert redact_obj(KEY) == REDACTED and redact_obj(42) == 42 and redact_obj(None) is None
    # the input is not mutated
    assert obj["a"] == KEY


def test_redacting_formatter_redacts_log_records(tmp_path):
    fmt = RedactingFormatter("%(levelname)s %(name)s: %(message)s")
    rec = logging.LogRecord("experiment.wave", logging.WARNING, __file__, 1, "creation error for %s: %s", ("prod-x", f"bad key {KEY}"), None)
    s = fmt.format(rec)
    assert KEY not in s and REDACTED in s and s.startswith("WARNING experiment.wave: creation error for prod-x")
    rec2 = logging.LogRecord("x", logging.INFO, __file__, 1, "header Authorization: Bearer %s", (KEY,), None)
    assert KEY not in fmt.format(rec2)
    # the installed handlers (stream + file) both use the redacting formatter
    log_path = str(tmp_path / "logs" / "experiment.log")
    root = setup_logging(log_path)
    try:
        assert all(isinstance(h.formatter, RedactingFormatter) for h in root.handlers) and len(root.handlers) == 2
        logging.getLogger("experiment.test").warning("token %s and %s", KEY, f"Bearer {KEY2}")
        for h in root.handlers:
            h.flush()
        text = open(log_path, encoding="utf-8").read()
        assert KEY not in text and KEY2 not in text and REDACTED in text and "token" in text
        assert not contains_secret(text)
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()


def test_event_writer_redacts_values_and_appends(tmp_path):
    path = str(tmp_path / "raw" / "events.jsonl")
    w = EventWriter(path)
    w.append({"kind": "batch_create", "error_message": f"Incorrect API key provided: {KEY}", "metadata": {"note": f"Bearer {KEY2}"},
              "headers": [KEY], "n": 1})
    w.close()
    text = open(path, encoding="utf-8").read()
    assert KEY not in text and KEY2 not in text and not contains_secret(text)
    recs = list(read_jsonl(path))
    assert len(recs) == 1
    assert recs[0]["error_message"] == f"Incorrect API key provided: {REDACTED}"
    assert recs[0]["metadata"]["note"] == f"Bearer {REDACTED}" and recs[0]["headers"] == [REDACTED] and recs[0]["n"] == 1
    # append-only: reopening never truncates
    w2 = EventWriter(path)
    w2.append({"kind": "second", "v": KEY})
    w2.close()
    recs = list(read_jsonl(path))
    assert len(recs) == 2 and recs[0]["kind"] == "batch_create" and recs[1]["v"] == REDACTED
    assert list(read_jsonl(str(tmp_path / "missing.jsonl"))) == []
    # the gzip-appended variant (used for poll events) redacts the same way, across checkpoints and sessions
    gz = str(tmp_path / "raw" / "polls.jsonl.gz")
    w3 = EventWriter(gz)
    w3.append({"kind": "batch_poll", "status": KEY})
    w3.checkpoint()
    w3.append({"kind": "batch_poll", "status": f"Bearer {KEY2}"})
    w3.close()
    w4 = EventWriter(gz)
    w4.append({"kind": "batch_poll", "status": "ok"})
    w4.close()
    import gzip
    assert KEY.encode() not in gzip.open(gz, "rb").read() and not contains_secret(gzip.open(gz, "rt", encoding="utf-8").read())
    recs = list(read_jsonl(gz))
    assert [r["status"] for r in recs] == [REDACTED, f"Bearer {REDACTED}", "ok"]
    assert [r["status"] for r in read_jsonl(str(tmp_path / "raw" / "polls.jsonl"))] == [REDACTED, f"Bearer {REDACTED}", "ok"]


def test_config_save_refuses_secret_looking_values(workdir):
    cfg = ExperimentConfig(experiment_id=KEY)
    path = str(workdir / "config" / "bad.json")
    with pytest.raises(ConfigError, match="refusing to write a secret-looking value"):
        cfg.save(path)
    assert not os.path.exists(path)
    ok = ExperimentConfig(experiment_id="exp-ok")
    ok.save(str(workdir / "config" / "ok.json"))
    text = open(workdir / "config" / "ok.json", encoding="utf-8").read()
    assert "sk-" not in text and "OPENAI_API_KEY" not in text and "api_key" not in text


def test_scan_for_secrets_finds_planted_files_and_skips_binaries(workdir):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    with open("reports/clean.md", "w") as f:
        f.write("nothing here\n")
    with open("reports/leak.txt", "w") as f:
        f.write(f"token {KEY}\n")
    with open("data/logs/x.log", "w") as f:
        f.write(f"Authorization: Bearer {KEY2}\n")
    with open("data/state.sqlite", "wb") as f:
        f.write(KEY.encode())
    hits = scan_for_secrets(["data", "reports", "does-not-exist"])
    assert sorted(hits) == sorted([os.path.join("reports", "leak.txt"), os.path.join("data", "logs", "x.log")])
    assert scan_for_secrets(["reports/clean.md"]) == [] and scan_for_secrets(["reports/leak.txt"]) == ["reports/leak.txt"]


# ---------------------------------------------------------------------------------------------
# full fake pipeline with the key in the environment and echoed back by the "server"
# ---------------------------------------------------------------------------------------------

class LeakyApi(FakeBatchApi):
    """Echoes the API key back in a batch metadata value and in every creation error message."""

    def __init__(self, key: str, **kw):
        super().__init__(**kw)
        self.key = key

    def _make_batch(self, input_file_id, endpoint, completion_window, metadata):
        b = super()._make_batch(input_file_id, endpoint, completion_window, metadata)
        b["metadata"]["operator_note"] = f"created with Bearer {self.key}"
        return b

    async def create_batch(self, input_file_id, endpoint, completion_window, metadata):
        res = await super().create_batch(input_file_id, endpoint, completion_window, metadata)
        if not res.ok and res.error_message:
            res.error_message = f"{res.error_message} (Incorrect API key provided: {self.key})"
        return res


async def _run_pipeline(workdir, api, failed_obs="prod-t00100-k0001"):
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=2, seed=7, levels=[16, 100],
                     experiment_id="exp-secrets", data_dir=str(workdir / "data"))
    await prepare(args, api=api)
    ev = await pilot(make_args(config=args.config, execute=True, max_cost_usd=0.10, interval=0.01), api=api)
    assert ev["passed"] and ev["checks"]["no_secrets_in_artifacts"]["ok"]
    out = await launch(make_args(config=args.config, execute=True, concurrency=3), api=api)
    assert out["wave"]["created"] == 3 and out["wave"]["errors"] == 1
    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=api)
    try:
        await monitor(rt, interval=0.01)
        await collect_all(rt)
        assert rt.store.get_job(failed_obs)["create_error_message"].endswith(f"provided: {KEY})")  # the DB holds the raw message
    finally:
        await rt.aclose()
    plan = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan["wave"]["created"] == 1
    rt = Runtime.open(cfg, api=api)
    try:
        await monitor(rt, interval=0.01)
    finally:
        await rt.aclose()
    return args, cfg


async def test_absence_of_secrets_after_a_full_fake_pipeline_run(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    api = LeakyApi(KEY, auto_advance=15.0, fail_create_for={"prod-t00100-k0001"})
    args, cfg = await _run_pipeline(workdir, api)
    config_dir = os.path.dirname(args.config)

    assert scan_for_secrets([cfg.data_dir, config_dir, "reports"]) == []
    cfg_text = open(args.config, encoding="utf-8").read()
    assert "sk-" not in cfg_text and KEY not in cfg_text
    assert "sk-" not in open(os.path.join(config_dir, "experiment.example.json"), encoding="utf-8").read()
    assert KEY not in json.dumps(cfg.to_dict())

    # positive evidence that the injected secrets did flow through and were masked on the way to disk
    log_text = open(cfg.log_path, encoding="utf-8").read()
    assert KEY not in log_text and REDACTED in log_text and "creation error for prod-t00100-k0001" in log_text
    events = list(read_jsonl(cfg.raw_path("batch_creation_events.jsonl")))
    failed = [e for e in events if e["observation_id"] == "prod-t00100-k0001"]
    assert len(failed) == 1 and failed[0]["error_message"].endswith(f"provided: {REDACTED})")
    created = [e for e in events if e.get("ok")]
    assert created and all(e["metadata"]["operator_note"] == f"created with Bearer {REDACTED}" for e in created)
    finals = list(read_jsonl(cfg.raw_path("batch_objects.jsonl")))
    assert finals and all(f["batch"]["metadata"]["operator_note"] == f"created with Bearer {REDACTED}" for f in finals)
    # every raw event stream (plain or gzip-appended) is clean record by record
    for name in ("batch_creation_events.jsonl", "batch_poll_events.jsonl", "batch_objects.jsonl", "responses.jsonl", "errors.jsonl"):
        recs = list(read_jsonl(cfg.raw_path(name)))
        assert recs or name == "errors.jsonl", name
        assert not any(contains_secret(json.dumps(r)) for r in recs), name
    for fn in os.listdir(cfg.raw_dir):
        if fn.endswith(".jsonl"):
            assert not contains_secret(open(os.path.join(cfg.raw_dir, fn), encoding="utf-8").read()), fn
    for name in ("prepare_summary.json", "pilot_summary.json", "launch_summary.json", "recover_summary.json"):
        assert not contains_secret(open(os.path.join(cfg.processed_dir, name), encoding="utf-8").read())
    assert not contains_secret(open(os.path.join("reports", "pilot_report.md"), encoding="utf-8").read())
    assert not contains_secret(open(cfg.manifest_path, encoding="utf-8").read())
    for fn in os.listdir(cfg.inputs_dir):
        p = os.path.join(cfg.inputs_dir, fn)
        if os.path.isfile(p):
            assert not contains_secret(open(p, encoding="utf-8").read())


async def test_absence_of_secrets_in_observations_csv(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    api = LeakyApi(KEY, auto_advance=15.0, fail_create_for={"prod-t00100-k0001"})
    args, cfg = await _run_pipeline(workdir, api)
    rt = Runtime.open(cfg, api=api, need_api=False)
    try:
        df = build_observations(rt.store, cfg)
        write_observations(df, cfg)
    finally:
        await rt.aclose()
    assert scan_for_secrets([cfg.data_dir, os.path.dirname(args.config), "reports"]) == []


class ServerEchoApi(FakeBatchApi):
    """A per-request error whose message echoes the key (as an error file line, i.e. server-side content)."""

    def __init__(self, key: str, **kw):
        super().__init__(**kw)
        self.key = key

    def _error_file(self, b, expired):
        fid = super()._error_file(b, expired)
        text = self.file_bytes[fid].decode("utf-8")
        self.file_bytes[fid] = text.replace("instead.", f"instead. (api key {self.key})").encode("utf-8")
        return fid


async def test_absence_of_secrets_in_pilot_report_when_the_server_echoes_the_key(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    api = ServerEchoApi(KEY, auto_advance=15.0)
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=2, seed=7, levels=[16, 100],
                     experiment_id="exp-secrets", data_dir=str(workdir / "data"))
    await prepare(args, api=api)
    ev = await pilot(make_args(config=args.config, execute=True, max_cost_usd=0.10, interval=0.01), api=api)
    assert ev["passed"]
    cfg = ExperimentConfig.load(args.config)
    assert not contains_secret(open(cfg.raw_path("errors.jsonl"), encoding="utf-8").read())  # raw artifact is redacted
    assert scan_for_secrets([cfg.data_dir, os.path.dirname(args.config), "reports"]) == []
