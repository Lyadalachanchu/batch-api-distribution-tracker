"""Creation-limit enforcement (item 8): rolling-hour window arithmetic, the launch gate, --wait semantics
(without sleeping), and the wave stopping early on a batch-limit 429 without retrying."""
from __future__ import annotations

import json
import time
from collections import Counter
from types import SimpleNamespace

import pytest

import experiment.launch as launch_mod
from experiment.config import ExperimentConfig
from experiment.events import read_jsonl
from experiment.launch import launch
from experiment.limits import WINDOW_SECONDS, CreationLimitExceeded, check_allowed, recent_creations_from_api, seconds_until_allowed
from experiment.pilot import pilot
from experiment.prepare import prepare
from experiment.recover import recover
from experiment.store import Store
from tests.conftest import make_args
from tests.fake_api import FakeBatchApi

NOW = 1_800_000_000.0


def test_creation_limit_planned_above_limit_raises_regardless_of_history():
    with pytest.raises(CreationLimitExceeded, match="planned 2001 creations exceed the rolling-hour limit of 2000"):
        seconds_until_allowed([], 2001, 2000, NOW)
    with pytest.raises(CreationLimitExceeded):
        check_allowed([], 3, 2, NOW)


def test_creation_limit_exactly_at_limit_is_allowed():
    assert seconds_until_allowed([], 2000, 2000, NOW) == 0.0
    assert seconds_until_allowed([NOW - 10.0] * 1999, 1, 2000, NOW) == 0.0
    assert seconds_until_allowed([NOW - 10.0] * 1000, 1000, 2000, NOW) == 0.0
    check_allowed([NOW - 10.0] * 1999, 1, 2000, NOW)  # no exception
    # one more than the limit has to wait for the oldest creation to age out
    assert seconds_until_allowed([NOW - 10.0] * 2000, 1, 2000, NOW) == pytest.approx(WINDOW_SECONDS - 10.0)


def test_creation_limit_window_ageing_math():
    recent = [NOW - 3000.0, NOW - 2000.0, NOW - 1000.0]
    # excess 1: the oldest (3000 s old) must reach 3600 s -> 600 s
    assert seconds_until_allowed(recent, 1, 3, NOW) == pytest.approx(600.0)
    # excess 2: the second oldest must age out -> 1600 s
    assert seconds_until_allowed(recent, 2, 3, NOW) == pytest.approx(1600.0)
    # excess 3: the newest must age out -> 2600 s
    assert seconds_until_allowed(recent, 3, 3, NOW) == pytest.approx(2600.0)
    # creations older than the window do not count at all
    assert seconds_until_allowed([NOW - 3601.0, NOW - 7200.0], 3, 3, NOW) == 0.0
    assert seconds_until_allowed([NOW - 3601.0, NOW - 10.0], 2, 3, NOW) == 0.0
    assert seconds_until_allowed([NOW - 3601.0, NOW - 10.0], 3, 3, NOW) == pytest.approx(3590.0)
    # an epoch exactly at the window edge is treated as aged out (strict > comparison)
    assert seconds_until_allowed([NOW - WINDOW_SECONDS], 3, 3, NOW) == 0.0
    # unsorted input is handled
    assert seconds_until_allowed([NOW - 1000.0, NOW - 3000.0, NOW - 2000.0], 1, 3, NOW) == pytest.approx(600.0)
    # a custom window
    assert seconds_until_allowed([NOW - 50.0], 1, 1, NOW, window=60.0) == pytest.approx(10.0)
    assert seconds_until_allowed([NOW - 70.0], 1, 1, NOW, window=60.0) == 0.0


def test_creation_limit_check_allowed_message():
    with pytest.raises(CreationLimitExceeded, match=r"3 creations in the last hour \+ 1 planned > limit 3; allowed in 10.0 min"):
        check_allowed([NOW - 3000.0, NOW - 2000.0, NOW - 1000.0], 1, 3, NOW)
    check_allowed([NOW - 3000.0, NOW - 2000.0], 1, 3, NOW)


async def test_creation_limit_recent_creations_from_api_pages_and_filters_by_epoch():
    api = FakeBatchApi(list_page_limit=2)
    await api.upload_file("t.jsonl", b'{"custom_id":"a","method":"POST","url":"/v1/responses","body":{"model":"m","input":"x","max_output_tokens":16}}\n')
    fid = next(iter(api.files))
    for i in range(5):
        api.advance(10)
        res = await api.create_batch(fid, "/v1/responses", "24h", {"observation_id": f"o{i}"})
        assert res.ok
    n, epochs = await recent_creations_from_api(api, since_epoch=0)
    assert n == 5 and len(epochs) == 5 and epochs == sorted(epochs, reverse=True)
    assert api.list_calls == 3  # 5 batches, 2 per page
    n2, epochs2 = await recent_creations_from_api(api, since_epoch=epochs[1])
    assert n2 == 2 and epochs2 == epochs[:2]
    n3, _ = await recent_creations_from_api(api, since_epoch=api.now() + 10)
    assert n3 == 0


# ---------------------------------------------------------------------------------------------
# launch gate
# ---------------------------------------------------------------------------------------------

async def _prepare(workdir, api, runs=2, levels=(16, 100)):
    args = make_args(config=str(workdir / "config" / "experiment.json"), runs_per_group=runs, seed=7, levels=list(levels),
                     experiment_id="exp-limits", data_dir=str(workdir / "data"))
    await prepare(args, api=api)
    return args


def _set_limit(args, limit):
    cfg = ExperimentConfig.load(args.config)
    cfg.max_creations_per_rolling_hour = limit
    cfg.save(args.config)
    return cfg


async def test_creation_limit_launch_refuses_when_recent_plus_pending_exceeds_the_limit(workdir, fake_api):
    args = await _prepare(workdir, fake_api)  # 4 production jobs pending
    ev = await pilot(make_args(config=args.config, execute=True, max_cost_usd=0.10, interval=0.01), api=fake_api)
    assert ev["passed"]
    n_pilot = fake_api.create_calls  # creation attempts recorded in the store within the last hour
    assert n_pilot == len(fake_api.batches) == 8
    cfg = _set_limit(args, n_pilot + 4 - 1)
    store = Store(cfg.db_path)
    try:
        assert store.count_creations_since(time.time() - WINDOW_SECONDS) == n_pilot
    finally:
        store.close()

    for kw in (dict(execute=True), dict(execute=False), dict(execute=False, wait=True)):
        with pytest.raises(CreationLimitExceeded, match=f"{n_pilot} creations in the last hour .* \\+ 4 planned > {n_pilot + 3}; allowed in"):
            await launch(make_args(config=args.config, **kw), api=fake_api)
    assert fake_api.create_calls == n_pilot and len(fake_api.batches) == n_pilot
    store = Store(cfg.db_path)
    try:
        assert store.get_meta("launch_started_at") is None
        assert Counter(j["creation_state"] for j in store.list_jobs(phases=["prod"])) == {"pending": 4}
    finally:
        store.close()

    # exactly at the limit is allowed: recent + pending == limit
    _set_limit(args, n_pilot + 4)
    out = await launch(make_args(config=args.config, execute=True, concurrency=2), api=fake_api)
    assert out["wave"]["created"] == 4 and out["plan"]["recent_creations"]["local"] == n_pilot
    assert out["plan"]["limit_per_hour"] == n_pilot + 4 and fake_api.create_calls == n_pilot + 4


async def test_creation_limit_launch_refuses_when_pending_alone_exceeds_the_limit(workdir, fake_api):
    args = await _prepare(workdir, fake_api)
    _set_limit(args, 3)
    with pytest.raises(CreationLimitExceeded, match="planned 4 creations exceed the rolling-hour limit of 3"):
        await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, wait=True), api=fake_api)
    assert fake_api.create_calls == 0


async def test_creation_limit_launch_wait_sleeps_until_the_window_clears_without_real_sleeping(workdir, fake_api, monkeypatch):
    """--wait semantics with asyncio.sleep stubbed out: the gate re-evaluates after the sleep and then launches."""
    args = await _prepare(workdir, fake_api)
    _set_limit(args, 12)
    real_recent = launch_mod.recent_creation_epochs
    seen = []

    async def fake_recent(rt, now, use_api=True):
        seen.append(now)
        if len(seen) == 1:
            return [now - 10.0] * 10, {"local": 10, "api": 0}  # 10 recent + 4 pending > 12
        return await real_recent(rt, now, use_api)

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(launch_mod, "recent_creation_epochs", fake_recent)
    monkeypatch.setattr(launch_mod, "asyncio", SimpleNamespace(sleep=fake_sleep))
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, wait=True, concurrency=2), api=fake_api)
    # wait = 3590 s -> the loop sleeps min(wait + 1, 60) once, re-checks, then proceeds
    assert sleeps == [60.0] and len(seen) == 2
    assert out["wave"]["created"] == 4 and out["plan"]["recent_creations"] == {"local": 0, "api": 0}
    assert fake_api.create_calls == 4


# ---------------------------------------------------------------------------------------------
# wave stops early on a batch-limit 429
# ---------------------------------------------------------------------------------------------

async def test_creation_limit_wave_stops_early_on_batch_limit_429_and_does_not_retry(workdir):
    api = FakeBatchApi(auto_advance=15.0, rate_limit_after=3)
    args = await _prepare(workdir, api, runs=3)  # 6 production jobs
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, concurrency=5), api=api)
    w = out["wave"]
    assert (w["planned"], w["created"], w["errors"], w["unknown"], w["skipped"]) == (6, 3, 1, 0, 2)
    assert w["rate_limit_429s"] == 1 and w["stopped_early"] is True
    assert "batch-creation limit" in w["stop_reason"] and "2000 batches per hour" in w["stop_reason"]
    assert w["per_status"] == {"200": 3, "429": 1}
    assert api.create_calls == 4, "3 successes + the single 429; the skipped jobs are never attempted and nothing is retried"
    assert len(api.batches) == 3

    cfg = ExperimentConfig.load(args.config)
    store = Store(cfg.db_path)
    try:
        jobs = store.list_jobs(phases=["prod"])
        assert Counter(j["creation_state"] for j in jobs) == {"created": 3, "error": 1, "pending": 2}
        pending = [j for j in jobs if j["creation_state"] == "pending"]
        assert all(store.attempts_for(j["observation_id"]) == [] and j["batch_id"] is None and j["local_create_started_at"] is None for j in pending)
        limited = next(j for j in jobs if j["creation_state"] == "error")
        assert limited["create_http_status"] == 429 and limited["create_error_code"] == "rate_limit_exceeded"
        a = store.attempts_for(limited["observation_id"])
        assert len(a) == 1 and a[0]["outcome"] == "error" and a[0]["http_status"] == 429
        assert len(store.all_attempts()) == 4
        assert store.get_meta("launch_started_at") and store.get_meta("launch_finished_at")
        # the wave submitted jobs strictly in launch order and stopped at the fourth
        assert sorted(j["launch_position"] for j in jobs if j["creation_state"] != "pending") == [0, 1, 2, 3]
        assert sorted(j["launch_position"] for j in pending) == [4, 5]
    finally:
        store.close()
    events = list(read_jsonl(cfg.raw_path("batch_creation_events.jsonl")))
    assert len(events) == 4 and [e["http_status"] for e in events] == [200, 200, 200, 429]
    summary = json.load(open(cfg.processed_dir + "/launch_summary.json"))
    assert summary["wave"]["stopped_early"] is True

    # once the limit clears the still-pending jobs go out with --resume (as a later wave), still no automatic retries
    api.rate_limit_after = None
    out2 = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, resume=True, concurrency=5), api=api)
    assert out2["wave"]["created"] == 2 and out2["wave"]["planned"] == 2 and out2["wave"]["wave"].startswith("launch-resume-")
    assert api.create_calls == 6
    # the 429'd job is only ever re-submitted as a labelled replacement via recover
    plan = await recover(make_args(config=args.config, execute=True), api=api)
    assert plan["new_replacements"] == 1 and plan["wave"]["created"] == 1 and api.create_calls == 7
    store = Store(cfg.db_path)
    try:
        assert Counter(j["creation_state"] for j in store.list_jobs(phases=["prod"])) == {"created": 5, "error": 1}
        assert len(store.list_jobs(phases=["replacement"])) == 1
    finally:
        store.close()


async def test_creation_limit_wave_429_without_batch_hint_does_not_stop_immediately(workdir, monkeypatch):
    """A generic 429 pauses (Retry-After) but does not abort the wave on its own."""
    import experiment.wave as wave_mod
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(wave_mod, "asyncio", SimpleNamespace(**{k: getattr(wave_mod.asyncio, k) for k in ("Queue", "Event", "Lock", "QueueEmpty", "create_task", "gather")}, sleep=fake_sleep))

    class OneGeneric429(FakeBatchApi):
        async def create_batch(self, *a, **k):
            if self.create_calls == 1:  # second call only
                self.create_calls += 1
                from experiment.api import ApiResult
                return ApiResult(ok=False, http_status=429, error_type="rate_limit_error", error_code="rate_limit_exceeded",
                                 error_message="Rate limit reached for requests", exception_kind="status", retry_after=1.0)
            return await super().create_batch(*a, **k)

    api = OneGeneric429(auto_advance=15.0)
    args = await _prepare(workdir, api, runs=2)
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, concurrency=1), api=api)
    w = out["wave"]
    assert w["stopped_early"] is False and w["rate_limit_429s"] == 1 and (w["created"], w["errors"], w["skipped"]) == (3, 1, 0)
    assert api.create_calls == 4 and sleeps and all(0 < s <= 1.0 for s in sleeps)
