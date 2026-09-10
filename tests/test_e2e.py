"""End-to-end pipeline against the in-memory fake API: prepare -> pilot -> launch -> monitor -> collect -> recover."""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from experiment.collect import collect_all
from experiment.config import ExperimentConfig
from experiment.launch import launch
from experiment.monitor import monitor
from experiment.observations import build_observations, write_observations
from experiment.pilot import pilot
from experiment.prepare import prepare
from experiment.recover import recover
from experiment.runtime import Runtime
from experiment.status import build_status
from tests.conftest import make_args
from tests.fake_api import FakeBatchApi


async def _run_pipeline(workdir, fake_api, levels=(16, 100, 1000), runs=3, **launch_kw):
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=runs, seed=7, levels=list(levels),
                     experiment_id="exp-e2e", data_dir=str(workdir / "data"))
    summary = await prepare(args, api=fake_api)
    assert summary["batch_creations_made"] == 0
    assert fake_api.create_calls == 0
    ev = await pilot(make_args(config=args.config, execute=True, max_cost_usd=0.10, timeout_minutes=5, interval=0.01), api=fake_api)
    assert ev["passed"], json.dumps(ev["checks"], indent=1, default=str)
    dry = await launch(make_args(config=args.config, execute=False), api=fake_api)
    assert dry["execute"] is False and dry["pending"] == len(levels) * runs
    creates_before = fake_api.create_calls
    out = await launch(make_args(config=args.config, execute=True, concurrency=5, **launch_kw), api=fake_api)
    assert fake_api.create_calls - creates_before == len(levels) * runs
    return args, out


async def test_full_pipeline(workdir, fake_api):
    args, out = await _run_pipeline(workdir, fake_api)
    assert out["wave"]["created"] == 9 and out["wave"]["errors"] == 0
    assert out["launch_started_at"] and out["launch_finished_at"]
    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=fake_api)
    try:
        res = await monitor(rt, phases=["prod"], interval=0.01)
        assert res["remaining_active"] == 0
        st = build_status(rt)
        assert st["prod"]["terminal"] == 9 and st["prod"]["collected"] == 9
        c = await collect_all(rt, phases=["prod"])
        assert c["total"] == 0  # everything already collected inline
        df = build_observations(rt.store, cfg)
        prod = df[df.phase == "prod"]
        assert len(prod) == 9 and prod.valid_observation.all()
        assert (prod.turnaround_seconds == prod.completed_at - prod.created_at).all()
        assert (prod.queue_seconds == prod.in_progress_at - prod.created_at).all()
        assert (prod.active_seconds == prod.completed_at - prod.in_progress_at).all()
        assert (prod.actual_output_tokens == prod.requested_output_tokens).all()
        assert prod.early_stop.eq(False).all()
        assert prod.estimated_cost_usd.gt(0).all()
        csv_path, pq_path = write_observations(df, cfg)
        back = pd.read_parquet(pq_path)
        assert len(back) == len(df)
        # raw artifacts exist and are append-only JSONL
        for name in ("batch_creation_events.jsonl", "batch_poll_events.jsonl", "batch_objects.jsonl", "responses.jsonl", "errors.jsonl"):
            assert os.path.exists(cfg.raw_path(name))
        polls = [json.loads(l) for l in open(cfg.raw_path("batch_poll_events.jsonl"))]
        assert {"batch_id", "poll_timestamp_utc", "status", "created_at", "in_progress_at", "completed_at", "failed_at",
                "expired_at", "cancelled_at", "request_counts", "output_file_id", "error_file_id"} <= set(polls[0])
        finals = [json.loads(l) for l in open(cfg.raw_path("batch_objects.jsonl"))]
        assert len([f for f in finals if f["observation_id"].startswith("prod-")]) == 9
        # a second monitor run is a no-op and creates nothing
        n = fake_api.create_calls
        res2 = await monitor(rt, phases=["prod"], interval=0.01)
        assert res2["remaining_active"] == 0 and fake_api.create_calls == n
    finally:
        await rt.aclose()


async def test_pilot_documents_api_floor_and_reuse(workdir, fake_api):
    args, _ = await _run_pipeline(workdir, fake_api)
    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=fake_api, need_api=False)
    try:
        rows = rt.store.jobs_with_results(phases=["pilot"])
        assert len(rows) == 1 + len(cfg.pilot_levels) + 1  # first 10-token job + one per pilot level + extra 16 reuse job
        infeasible = [r for r in rows if r["requested_output_tokens"] < 16]
        assert infeasible and all(r["result_http_status"] == 400 and r["result_error_code"] == "integer_below_min_value" for r in infeasible)
        files = {}
        for r in rows:
            files.setdefault(r["input_file_id"], []).append(r["batch_id"])
        assert any(len(v) >= 2 for v in files.values())
        assert rt.store.get_meta("pilot_status") == "passed"
    finally:
        await rt.aclose()
    assert os.path.exists("reports/pilot_report.md")


async def test_launch_gates(workdir, fake_api):
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=2, seed=1, levels=[16, 100],
                     experiment_id="exp-gate", data_dir=str(workdir / "data"))
    await prepare(args, api=fake_api)
    with pytest.raises(RuntimeError, match="pilot_status"):
        await launch(make_args(config=args.config, execute=True), api=fake_api)
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, concurrency=2), api=fake_api)
    assert out["wave"]["created"] == 4
    with pytest.raises(RuntimeError, match="already recorded"):
        await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True), api=fake_api)


async def test_recover_replaces_failed_creations(workdir):
    api = FakeBatchApi(auto_advance=15.0, fail_create_for={"prod-t00100-k0001"})
    args, out = await _run_pipeline(workdir, api, levels=(16, 100), runs=2)
    assert out["wave"]["created"] == 3 and out["wave"]["errors"] == 1
    cfg = ExperimentConfig.load(args.config)
    plan = await recover(make_args(config=args.config, execute=False), api=api)
    assert plan["new_replacements"] == 1 and plan["pending_total"] == 1
    plan2 = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan2["wave"]["created"] == 1
    rt = Runtime.open(cfg, api=api)
    try:
        await monitor(rt, interval=0.01)
        df = build_observations(rt.store, cfg)
        rep = df[df.phase == "replacement"]
        assert len(rep) == 1 and rep.iloc[0]["parent_observation_id"] == "prod-t00100-k0001" and rep.iloc[0]["attempt_id"] == 2
        failed = df[df.observation_id == "prod-t00100-k0001"].iloc[0]
        assert failed["status"] == "creation_error" and not failed["valid_observation"]
        # nothing left to replace
        plan3 = await recover(make_args(config=args.config, execute=True), api=api)
        assert plan3["pending_total"] == 0
    finally:
        await rt.aclose()
