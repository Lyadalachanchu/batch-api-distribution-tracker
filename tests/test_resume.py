"""Resumability against the in-memory fake API.

Covers: idempotent resume (11), retry accounting incl. reconcile adoption (10), and input-file reuse
or fallback to individual files (6)."""
from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager

import pytest

from experiment.api import ApiResult
from experiment.collect import collect_all
from experiment.config import ORIGINAL_REQUESTED_LEVELS, ExperimentConfig
from experiment.launch import launch
from experiment.monitor import monitor
from experiment.observations import build_observations
from experiment.pilot import pilot
from experiment.prepare import prepare
from experiment.reconcile import reconcile
from experiment.recover import recover
from experiment.runtime import Runtime
from experiment.store import Store
from tests.conftest import make_args
from tests.fake_api import FakeBatchApi


async def _prepare(workdir, api, levels=(16, 100), runs=2, experiment_id="exp-resume", seed=7, **kw):
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=runs, seed=seed, levels=list(levels),
                     experiment_id=experiment_id, data_dir=str(workdir / "data"), **kw)
    summary = await prepare(args, api=api)
    return args, summary


@contextmanager
def _store(args):
    cfg = ExperimentConfig.load(args.config)
    store = Store(cfg.db_path)
    try:
        yield store
    finally:
        store.close()


def _jobs(args, phases=None, **kw):
    with _store(args) as store:
        return store.list_jobs(phases=phases, **kw)


def _pilot_args(args):
    return make_args(config=args.config, execute=True, max_cost_usd=0.10, timeout_minutes=5, interval=0.01)


def _launch_args(args, **kw):
    base = dict(config=args.config, execute=True, skip_pilot_gate=True, concurrency=3)
    base.update(kw)
    return make_args(**base)


# ---------------------------------------------------------------------------------------------
# 11. idempotent resume
# ---------------------------------------------------------------------------------------------

async def test_idempotent_resume_prepare_twice_inserts_nothing_and_keeps_positions(workdir, fake_api):
    args, s1 = await _prepare(workdir, fake_api)
    n_levels = len(set(ORIGINAL_REQUESTED_LEVELS) | {16, 100})
    assert s1["jobs_inserted_now"] == 4 and s1["jobs_in_db"] == 4
    assert fake_api.upload_calls == n_levels
    before = _jobs(args, ["prod"])
    pos1 = {j["observation_id"]: (j["launch_position"], j["input_file_id"], j["updated_at"]) for j in before}
    manifest1 = open(ExperimentConfig.load(args.config).manifest_path).read()

    args2, s2 = await _prepare(workdir, fake_api)
    assert s2["jobs_inserted_now"] == 0 and s2["jobs_in_db"] == 4 and s2["total_production_jobs"] == 4
    assert fake_api.upload_calls == n_levels, "shared uploads are idempotent on sha256"
    assert fake_api.create_calls == 0
    after = _jobs(args2, ["prod"])
    assert {j["observation_id"]: (j["launch_position"], j["input_file_id"], j["updated_at"]) for j in after} == pos1
    assert open(ExperimentConfig.load(args.config).manifest_path).read() == manifest1
    assert s2["shared_files"] == s1["shared_files"]
    # the third run with no overrides at all still resolves the same experiment from the saved config
    s3 = await prepare(make_args(config=args.config), api=fake_api)
    assert s3["jobs_inserted_now"] == 0 and s3["experiment_id"] == "exp-resume" and s3["seed"] == 7


async def test_idempotent_resume_prepare_refuses_a_conflicting_seed_on_an_existing_db(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    with pytest.raises(RuntimeError, match="disagree with the existing DB"):
        await _prepare(workdir, fake_api, seed=8)
    # the original rows are untouched
    assert len(_jobs(args, ["prod"])) == 4 and fake_api.create_calls == 0


async def test_idempotent_resume_launch_again_after_a_full_wave_raises_already_recorded(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    out = await launch(_launch_args(args), api=fake_api)
    assert out["wave"]["created"] == 4 and out["wave"]["wave"] == "launch"
    n = fake_api.create_calls
    with pytest.raises(RuntimeError, match="already recorded"):
        await launch(_launch_args(args), api=fake_api)
    with pytest.raises(RuntimeError, match="already recorded"):
        await launch(_launch_args(args, execute=False), api=fake_api)
    assert fake_api.create_calls == n and len(fake_api.batches) == 4
    # --resume with nothing pending is an explicit no-op that still creates nothing
    out2 = await launch(_launch_args(args, resume=True), api=fake_api)
    assert out2 == {"pending": 0} and fake_api.create_calls == n
    with _store(args) as store:
        assert store.get_meta("launch_started_at") == out["launch_started_at"]
        assert len(store.all_attempts()) == 4


async def test_idempotent_resume_monitor_twice_creates_no_batches_and_db_survives_reopening(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    await launch(_launch_args(args), api=fake_api)
    cfg = ExperimentConfig.load(args.config)
    n_create = fake_api.create_calls
    rt = Runtime.open(cfg, api=fake_api)
    try:
        res1 = await monitor(rt, phases=["prod"], interval=0.01)
        assert res1["remaining_active"] == 0 and res1["transitions_total"] == 4
        cycles = rt.store.get_meta("monitor_cycles")
        assert cycles >= 1
    finally:
        await rt.aclose()
    assert fake_api.create_calls == n_create

    # reopen from disk: a brand-new Store sees everything the first process wrote
    store = Store(cfg.db_path)
    try:
        jobs = store.list_jobs(phases=["prod"])
        assert len(jobs) == 4 and all(j["terminal"] == 1 and j["collected"] == 1 and j["status"] == "completed" for j in jobs)
        assert store.get_meta("monitor_cycles") == cycles and store.get_meta("launch_started_at")
        assert len(store.list_results()) == 4 and len(store.all_attempts()) == 4
        assert store.get_meta("experiment_id") == "exp-resume"
    finally:
        store.close()

    rt2 = Runtime.open(cfg, api=fake_api)
    try:
        res2 = await monitor(rt2, phases=["prod"], interval=0.01)
        assert res2["remaining_active"] == 0
        assert rt2.store.get_meta("monitor_cycles") == cycles, "no poll cycle runs when nothing is active"
        # collect is idempotent as well: everything was collected inline already
        assert await collect_all(rt2, phases=["prod"]) == {"collected": 0, "with_parse_error": 0, "total": 0}
    finally:
        await rt2.aclose()
    assert fake_api.create_calls == n_create and len(fake_api.batches) == 4


async def test_idempotent_resume_pilot_without_again_creates_no_new_pilot_batches(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    ev1 = await pilot(_pilot_args(args), api=fake_api)
    assert ev1["passed"], json.dumps(ev1["checks"], default=str)
    n_calls, batches = fake_api.create_calls, set(fake_api.batches)
    pilot_jobs = _jobs(args, ["pilot"])
    assert len(pilot_jobs) == n_calls == 1 + len(ExperimentConfig.load(args.config).pilot_levels) + 1

    ev2 = await pilot(_pilot_args(args), api=fake_api)
    assert ev2["passed"]
    assert fake_api.create_calls == n_calls and set(fake_api.batches) == batches
    assert [j["observation_id"] for j in _jobs(args, ["pilot"])] == [j["observation_id"] for j in pilot_jobs]
    with _store(args) as store:
        assert store.get_meta("pilot_status") == "passed"
        assert all(len(store.attempts_for(j["observation_id"])) == 1 for j in pilot_jobs)
    # the dry run never creates anything either
    plan = await pilot(make_args(config=args.config, execute=False, max_cost_usd=0.10, interval=0.01), api=fake_api)
    assert plan.get("execute") is False and fake_api.create_calls == n_calls


@pytest.mark.xfail(strict=True, reason="BUG: pilot --again never creates new batches: pilot_job_rows() always yields the same "
                                       "fixed observation ids (k=0,1,2), insert_jobs is INSERT OR IGNORE and only "
                                       "creation_state='pending' rows are submitted, so a second pilot round is silently empty")
async def test_pilot_again_creates_a_fresh_set_of_pilot_batches(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    ev1 = await pilot(_pilot_args(args), api=fake_api)
    assert ev1["passed"]
    n_calls = fake_api.create_calls
    ev2 = await pilot(make_args(config=args.config, execute=True, max_cost_usd=0.10, interval=0.01, again=True), api=fake_api)
    assert ev2["passed"]
    assert fake_api.create_calls > n_calls, "--again is documented to create a fresh set of pilot batches"


# ---------------------------------------------------------------------------------------------
# 10. retry accounting
# ---------------------------------------------------------------------------------------------

async def test_retry_accounting_failed_creation_is_recorded_once_and_not_retried_in_the_wave(workdir):
    obs = "prod-t00100-k0001"
    api = FakeBatchApi(auto_advance=15.0, fail_create_for={obs})
    args, _ = await _prepare(workdir, api)
    out = await launch(_launch_args(args), api=api)
    w = out["wave"]
    assert (w["planned"], w["created"], w["errors"], w["unknown"], w["skipped"]) == (4, 3, 1, 0, 0)
    assert w["per_status"] == {"200": 3, "500": 1}
    assert api.create_calls == 4, "one call per job: the failed creation is not retried inside the wave"
    assert len(api.batches) == 3
    cfg = ExperimentConfig.load(args.config)
    with _store(args) as store:
        job = store.get_job(obs)
        assert job["creation_state"] == "error" and job["batch_id"] is None
        assert job["create_http_status"] == 500 and job["create_error_type"] == "server_error"
        assert job["local_create_started_at"] and job["local_create_finished_at"]
        attempts = store.attempts_for(obs)
        assert len(attempts) == 1
        a = attempts[0]
        assert (a["attempt_no"], a["outcome"], a["http_status"], a["wave"], a["batch_id"]) == (1, "error", 500, "launch", None)
        assert a["error_type"] == "server_error" and a["finished_at"]
        assert len(store.all_attempts()) == 4
        assert all(len(store.attempts_for(j["observation_id"])) == 1 for j in store.list_jobs(phases=["prod"]))
        assert Counter(j["creation_state"] for j in store.list_jobs(phases=["prod"])) == {"created": 3, "error": 1}
    events = [json.loads(l) for l in open(cfg.raw_path("batch_creation_events.jsonl"))]
    mine = [e for e in events if e["observation_id"] == obs]
    assert len(mine) == 1 and mine[0]["ok"] is False and mine[0]["outcome"] == "error" and mine[0]["attempt_no"] == 1
    assert mine[0]["http_status"] == 500 and mine[0]["wave"] == "launch"
    assert len(events) == 4


async def test_retry_accounting_recover_creates_a_labelled_replacement_with_attempt_id_2(workdir):
    obs = "prod-t00100-k0001"
    api = FakeBatchApi(auto_advance=15.0, fail_create_for={obs})
    args, _ = await _prepare(workdir, api)
    await launch(_launch_args(args), api=api)
    original = _jobs(args, ["prod"])
    orig = next(j for j in original if j["observation_id"] == obs)

    dry = await recover(make_args(config=args.config, execute=False), api=api)
    assert dry == {**dry, "new_replacements": 1, "already_pending": 0, "pending_total": 1}
    assert api.create_calls == 4  # dry run creates nothing
    plan = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan["new_replacements"] == 0 and plan["already_pending"] == 1 and plan["pending_total"] == 1
    assert plan["wave"]["created"] == 1 and plan["wave"]["errors"] == 0 and plan["wave"]["wave"].startswith("recover-")
    assert api.create_calls == 5 and len(api.batches) == 4

    with _store(args) as store:
        rep = store.get_job(obs + "-a2")
        assert rep is not None
        assert rep["phase"] == "replacement" and rep["attempt_id"] == 2 and rep["parent_observation_id"] == obs
        assert rep["creation_state"] == "created" and rep["batch_id"]
        assert rep["requested_output_tokens"] == 100 and rep["launch_position"] == orig["launch_position"]
        assert rep["input_file_id"] == orig["input_file_id"] and rep["custom_id"] == orig["custom_id"]
        rep_attempts = store.attempts_for(rep["observation_id"])
        assert len(rep_attempts) == 1 and rep_attempts[0]["outcome"] == "created" and rep_attempts[0]["attempt_no"] == 1
        # the original keeps exactly its one failed attempt and its error state
        again = store.get_job(obs)
        assert again["creation_state"] == "error" and again["batch_id"] is None and len(store.attempts_for(obs)) == 1
        assert len(store.all_attempts()) == 5
        assert len(store.list_jobs(phases=["prod"])) == 4 and len(store.list_jobs(phases=["replacement"])) == 1
    # the server-side batch carries the replacement's identity in its metadata
    b = next(b for b in api.batches.values() if b["metadata"]["observation_id"] == obs + "-a2")
    assert b["metadata"]["phase"] == "replacement" and b["metadata"]["attempt_no"] == "1"
    # a second recover has nothing to do and creates nothing
    plan2 = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan2 == {"new_replacements": 0, "already_pending": 0, "pending_total": 0} and api.create_calls == 5


async def test_retry_accounting_failed_replacement_gets_attempt_id_3(workdir):
    obs = "prod-t00016-k0000"
    api = FakeBatchApi(auto_advance=15.0, fail_create_for={obs, obs + "-a2"})
    args, _ = await _prepare(workdir, api)
    await launch(_launch_args(args), api=api)
    p1 = await recover(make_args(config=args.config, execute=True), api=api)
    assert p1["wave"]["errors"] == 1 and p1["wave"]["created"] == 0
    p2 = await recover(make_args(config=args.config, execute=True), api=api)
    assert p2["new_replacements"] == 1 and p2["wave"]["created"] == 1
    assert api.create_calls == 6  # 4 wave + a2 (failed) + a3 (created)
    with _store(args) as store:
        a2, a3 = store.get_job(obs + "-a2"), store.get_job(obs + "-a3")
        assert a2["creation_state"] == "error" and a2["attempt_id"] == 2 and a2["parent_observation_id"] == obs
        assert a3["creation_state"] == "created" and a3["attempt_id"] == 3 and a3["parent_observation_id"] == obs
        assert all(len(store.attempts_for(o)) == 1 for o in (obs, obs + "-a2", obs + "-a3"))
    # nothing more to replace
    p3 = await recover(make_args(config=args.config, execute=True), api=api)
    assert p3["pending_total"] == 0 and api.create_calls == 6
    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=api)
    try:
        await monitor(rt, interval=0.01)
        df = build_observations(rt.store, cfg).set_index("observation_id")
        assert df.loc[obs + "-a3"]["valid_observation"] and df.loc[obs + "-a3"]["is_replacement"]
        assert df.loc[obs]["status"] == "creation_error" and df.loc[obs + "-a2"]["status"] == "creation_error"
    finally:
        await rt.aclose()


async def test_retry_accounting_timeout_records_unknown_and_reconcile_adopts_the_server_batch(workdir):
    obs = "prod-t00016-k0000"
    api = FakeBatchApi(auto_advance=15.0, timeout_create_for={obs})
    args, _ = await _prepare(workdir, api)
    out = await launch(_launch_args(args), api=api)
    w = out["wave"]
    assert (w["created"], w["errors"], w["unknown"]) == (3, 0, 1) and w["per_status"] == {"200": 3, "timeout": 1}
    assert api.create_calls == 4 and len(api.batches) == 4, "the server did create the batch whose answer was lost"
    server_batch = next(b for b in api.batches.values() if b["metadata"]["observation_id"] == obs)

    with _store(args) as store:
        job = store.get_job(obs)
        assert job["creation_state"] == "unknown" and job["batch_id"] is None
        assert job["create_error_type"] == "APITimeoutError"
        attempts = store.attempts_for(obs)
        assert len(attempts) == 1 and attempts[0]["outcome"] == "unknown" and attempts[0]["batch_id"] is None
        assert attempts[0]["http_status"] is None and attempts[0]["error_type"] == "APITimeoutError"
        # the unknown attempt still counts against the rolling-hour budget
        assert store.count_creations_since(attempts[0]["started_epoch"] - 1) == 4

    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=api)
    try:
        rec = await reconcile(rt)
        assert rec["adopted"] == [obs] and rec["duplicates"] == [] and rec["resolved_unknown_as_error"] == []
        job = rt.store.get_job(obs)
        assert job["creation_state"] == "created" and job["batch_id"] == server_batch["id"]
        assert job["created_at"] == server_batch["created_at"] and job["status"] is not None
        attempts = rt.store.attempts_for(obs)
        assert len(attempts) == 1 and attempts[0]["outcome"] == "created" and attempts[0]["batch_id"] == server_batch["id"]
        assert api.create_calls == 4 and len(api.batches) == 4, "adoption never creates a duplicate batch"
        # reconcile is idempotent
        rec2 = await reconcile(rt)
        assert rec2["adopted"] == [] and rec2["duplicates"] == []
        assert Counter(j["creation_state"] for j in rt.store.list_jobs(phases=["prod"])) == {"created": 4}
    finally:
        await rt.aclose()
    events = [json.loads(l) for l in open(cfg.raw_path("batch_creation_events.jsonl"))]
    adopted = [e for e in events if e.get("kind") == "batch_adopted"]
    assert len(adopted) == 1 and adopted[0]["observation_id"] == obs and adopted[0]["batch_id"] == server_batch["id"]

    # recover has nothing to replace; the adopted batch then completes like any other
    plan = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan["new_replacements"] == 0 and plan["pending_total"] == 0 and api.create_calls == 4
    rt = Runtime.open(cfg, api=api)
    try:
        res = await monitor(rt, phases=["prod"], interval=0.01)
        assert res["remaining_active"] == 0
        df = build_observations(rt.store, cfg).set_index("observation_id")
        assert df.loc[obs]["valid_observation"] and df.loc[obs]["batch_id"] == server_batch["id"]
        assert df.loc[obs]["turnaround_seconds"] == df.loc[obs]["completed_at"] - df.loc[obs]["created_at"]
    finally:
        await rt.aclose()


class _LostTimeoutApi(FakeBatchApi):
    """A timeout where the request never reached the server (no batch exists to adopt)."""

    def __init__(self, lost: set[str], **kw):
        super().__init__(**kw)
        self.lost = set(lost)

    async def create_batch(self, input_file_id, endpoint, completion_window, metadata):
        obs = metadata.get("observation_id", "")
        if obs in self.lost:
            self.lost.discard(obs)
            self.create_calls += 1
            return ApiResult(ok=False, error_type="APIConnectionError", error_message="Connection error.", exception_kind="connection")
        return await super().create_batch(input_file_id, endpoint, completion_window, metadata)


async def test_retry_accounting_unknown_outcome_with_no_server_batch_becomes_error_then_replaced(workdir):
    obs = "prod-t00100-k0000"
    api = _LostTimeoutApi({obs}, auto_advance=15.0)
    args, _ = await _prepare(workdir, api)
    out = await launch(_launch_args(args), api=api)
    assert out["wave"]["unknown"] == 1 and out["wave"]["per_status"] == {"200": 3, "connection": 1}
    assert len(api.batches) == 3
    with _store(args) as store:
        assert store.get_job(obs)["creation_state"] == "unknown"
    # recover reconciles first: nothing on the server -> error -> replacement created
    plan = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan["new_replacements"] == 1 and plan["wave"]["created"] == 1
    with _store(args) as store:
        assert store.get_job(obs)["creation_state"] == "error"
        assert store.attempts_for(obs)[0]["outcome"] == "error"
        rep = store.get_job(obs + "-a2")
        assert rep["creation_state"] == "created" and rep["parent_observation_id"] == obs
    assert api.create_calls == 5 and len(api.batches) == 4


# ---------------------------------------------------------------------------------------------
# 6. input-file reuse or fallback
# ---------------------------------------------------------------------------------------------

async def test_input_file_reuse_rejected_makes_the_pilot_fail_with_creation_failures_listed(workdir):
    api = FakeBatchApi(auto_advance=15.0, reject_file_reuse=True)
    args, _ = await _prepare(workdir, api)
    ev = await pilot(_pilot_args(args), api=api)
    assert ev["passed"] is False
    chk = ev["checks"]["input_file_reuse_works"]
    assert chk["ok"] is False
    assert chk["detail"]["files_reused"] == {}
    failures = chk["detail"]["creation_failures"]
    assert "pilot-t00010-k0001" in failures, "second use of the 10-token file (first used in pilot wave 1) must be listed"
    assert any(f.startswith("pilot-t00016-k") for f in failures), "second use of the 16-token file in wave 2 must be listed"
    assert ev["checks"]["all_creations_succeeded"]["ok"] is False
    assert ev["checks"]["all_creations_succeeded"]["detail"]["created"] == ev["checks"]["all_creations_succeeded"]["detail"]["jobs"] - len(failures)
    with _store(args) as store:
        assert store.get_meta("pilot_status") == "failed"
        for f in failures:
            j = store.get_job(f)
            assert j["creation_state"] == "error" and j["create_http_status"] == 400 and j["create_error_code"] == "file_already_used"
            assert len(store.attempts_for(f)) == 1
        # every input file was accepted exactly once by the server
        assert all(n == 1 for n in api.file_uses.values())
    assert "FAILED" in open(os.path.join("reports", "pilot_report.md"), encoding="utf-8").read()
    # the failed pilot gates the launch
    with pytest.raises(RuntimeError, match="pilot_status='failed'"):
        await launch(make_args(config=args.config, execute=True), api=api)


async def test_input_file_fallback_individual_files_give_every_job_its_own_file_and_launch_succeeds(workdir):
    api = FakeBatchApi(auto_advance=15.0, reject_file_reuse=True)
    args, s1 = await _prepare(workdir, api, levels=(16, 100), runs=3)
    assert s1["file_mode"] == "shared"
    shared_ids = {j["input_file_id"] for j in _jobs(args, ["prod"])}
    assert len(shared_ids) == 2, "shared mode: one input file per level"
    uploads_before = api.upload_calls

    s2 = await prepare(make_args(config=args.config, file_mode="individual"), api=api)
    assert s2["file_mode"] == "individual" and s2["individual_files_uploaded_now"] == 6 and s2["jobs_inserted_now"] == 0
    assert api.upload_calls == uploads_before + 6 and api.create_calls == 0
    assert ExperimentConfig.load(args.config).file_mode == "individual"
    jobs = _jobs(args, ["prod"])
    ids = [j["input_file_id"] for j in jobs]
    assert len(ids) == 6 and len(set(ids)) == 6 and not set(ids) & shared_ids
    with _store(args) as store:
        for j in jobs:
            f = store.get_individual_file(j["observation_id"])
            assert f and f["file_id"] == j["input_file_id"] and f["kind"] == "individual" and f["tokens"] == j["api_max_output_tokens"]
            line = json.loads(api.file_bytes[j["input_file_id"]].decode("utf-8").strip())
            assert line["custom_id"] == j["observation_id"] and line["body"]["max_output_tokens"] == j["api_max_output_tokens"]
            assert os.path.exists(os.path.join(ExperimentConfig.load(args.config).inputs_dir, "individual", f"{j['observation_id']}.jsonl"))
    # re-running prepare in individual mode uploads nothing new
    s3 = await prepare(make_args(config=args.config, file_mode="individual"), api=api)
    assert s3["individual_files_uploaded_now"] == 0 and api.upload_calls == uploads_before + 6
    assert [j["input_file_id"] for j in _jobs(args, ["prod"])] == ids

    out = await launch(_launch_args(args, concurrency=5), api=api)
    assert out["wave"]["created"] == 6 and out["wave"]["errors"] == 0 and out["wave"]["unknown"] == 0
    assert all(api.file_uses[fid] == 1 for fid in ids)
    assert {b["input_file_id"] for b in api.batches.values()} == set(ids)
    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=api)
    try:
        res = await monitor(rt, phases=["prod"], interval=0.01)
        assert res["remaining_active"] == 0
        df = build_observations(rt.store, cfg)
        prod = df[df.phase == "prod"]
        assert len(prod) == 6 and prod.valid_observation.all()
        assert (prod.actual_output_tokens == prod.requested_output_tokens).all()
    finally:
        await rt.aclose()


async def test_input_file_fallback_uploads_one_file_per_production_job_for_5x400(workdir):
    """Individual mode on the full 5 x 400 design means exactly 2,000 uploads, one per production job."""
    api = FakeBatchApi(auto_advance=15.0, reject_file_reuse=True)
    args, s1 = await _prepare(workdir, api, levels=(16, 100, 1000, 3000, 10000), runs=400, experiment_id="exp-ind", file_mode="individual")
    assert s1["total_production_jobs"] == 2000 and s1["individual_files_uploaded_now"] == 2000
    assert api.upload_calls == 2000 + 7 and api.create_calls == 0
    jobs = _jobs(args, ["prod"])
    ids = [j["input_file_id"] for j in jobs]
    assert len(set(ids)) == 2000 and all(ids)
    with _store(args) as store:
        files = store.list_files()
        assert Counter(f["kind"] for f in files) == {"shared": 7, "individual": 2000}
        assert {f["observation_id"] for f in files if f["kind"] == "individual"} == {j["observation_id"] for j in jobs}
