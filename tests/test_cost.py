"""Cost-ceiling enforcement (item 9): projection arithmetic, the ceiling gate in prepare/pilot/launch/recover,
and actual-usage costing including cached input tokens."""
from __future__ import annotations

import pytest

from experiment.config import DEFAULT_OUTPUT_TOKEN_LEVELS, ORIGINAL_REQUESTED_LEVELS, ExperimentConfig, Pricing
from experiment.cost import CostCeilingExceeded, CostProjection, actual_cost_usd, enforce_ceiling, project_cost
from experiment.launch import launch, spent_or_committed_usd
from experiment.monitor import monitor
from experiment.pilot import pilot
from experiment.prepare import prepare
from experiment.recover import recover
from experiment.runtime import Runtime
from experiment.store import Store
from tests.conftest import make_args
from tests.fake_api import FakeBatchApi

ORIGINAL_COUNTS = {n: 400 for n in ORIGINAL_REQUESTED_LEVELS}  # [1, 10, 100, 1000, 10000] x 400


def test_cost_ceiling_project_cost_math_for_the_original_brief():
    p = project_cost(ORIGINAL_COUNTS, input_tokens_per_request=60, pricing=Pricing(), ceiling_usd=4.0, safety_margin=0.10)
    assert isinstance(p, CostProjection)
    assert p.jobs == 2000
    assert p.max_output_tokens == 4_444_400 == 400 * (1 + 10 + 100 + 1000 + 10000)
    assert p.output_cost_usd == pytest.approx(2.66664, abs=1e-9)  # 4,444,400 tokens at $0.60 / 1M
    assert p.est_input_tokens == 120_000
    assert p.input_cost_usd == pytest.approx(0.012, abs=1e-9)  # 120,000 tokens at $0.10 / 1M
    assert p.total_cost_usd == pytest.approx(2.67864, abs=1e-9)
    assert p.safety_margin == 0.10
    assert p.total_with_margin_usd == pytest.approx(2.946504, abs=1e-9)
    assert p.ceiling_usd == 4.0
    enforce_ceiling(p)  # under the ceiling: no exception
    d = p.to_dict()
    assert d["max_output_tokens"] == 4_444_400 and d["output_cost_usd"] == p.output_cost_usd and set(d) >= {"jobs", "ceiling_usd"}


def test_cost_ceiling_output_only_projection_matches_the_brief_figure():
    """The $2.66664 figure is the output-token part alone (zero input tokens)."""
    p = project_cost(ORIGINAL_COUNTS, 0, Pricing(), 4.0, safety_margin=0.0)
    assert p.input_cost_usd == 0.0 and p.est_input_tokens == 0
    assert p.total_cost_usd == pytest.approx(2.66664) and p.total_with_margin_usd == pytest.approx(2.66664)


def test_cost_ceiling_production_levels_fit_under_the_default_ceiling():
    cfg = ExperimentConfig()
    p = project_cost({n: cfg.runs_per_group for n in DEFAULT_OUTPUT_TOKEN_LEVELS}, cfg.estimated_input_tokens_per_request,
                     cfg.pricing, cfg.max_cost_usd, cfg.cost_safety_margin)
    assert p.max_output_tokens == 5_646_400 and p.output_cost_usd == pytest.approx(3.38784)
    assert p.total_with_margin_usd == pytest.approx((3.38784 + 0.012) * 1.1) and p.total_with_margin_usd < 4.0
    enforce_ceiling(p)


def test_cost_ceiling_enforce_ceiling_raises_only_above_the_ceiling():
    p = project_cost(ORIGINAL_COUNTS, 60, Pricing(), ceiling_usd=2.9)
    with pytest.raises(CostCeilingExceeded, match=r"exceeds ceiling \$2.90"):
        enforce_ceiling(p)
    with pytest.raises(CostCeilingExceeded, match="incl. 10% margin"):
        enforce_ceiling(p)
    # exactly at the ceiling is allowed (strict > comparison)
    enforce_ceiling(project_cost(ORIGINAL_COUNTS, 60, Pricing(), ceiling_usd=p.total_with_margin_usd))
    with pytest.raises(CostCeilingExceeded):
        enforce_ceiling(project_cost(ORIGINAL_COUNTS, 60, Pricing(), ceiling_usd=p.total_with_margin_usd - 1e-6))
    # the margin is applied on top of the projection
    with pytest.raises(CostCeilingExceeded):
        enforce_ceiling(project_cost(ORIGINAL_COUNTS, 60, Pricing(), ceiling_usd=2.68, safety_margin=0.10))
    enforce_ceiling(project_cost(ORIGINAL_COUNTS, 60, Pricing(), ceiling_usd=2.68, safety_margin=0.0))


def test_cost_ceiling_already_spent_counts_towards_the_projection():
    base = project_cost(ORIGINAL_COUNTS, 60, Pricing(), 3.0)
    spent = project_cost(ORIGINAL_COUNTS, 60, Pricing(), 3.0, already_spent_usd=0.1)
    assert spent.total_cost_usd == pytest.approx(base.total_cost_usd + 0.1)
    assert spent.total_with_margin_usd == pytest.approx((base.total_cost_usd + 0.1) * 1.1)
    enforce_ceiling(base)
    with pytest.raises(CostCeilingExceeded):
        enforce_ceiling(spent)


def test_cost_ceiling_projection_uses_the_given_pricing():
    pr = Pricing(input_per_1m_usd=1.0, cached_input_per_1m_usd=0.5, output_per_1m_usd=2.0)
    p = project_cost({100: 10}, 50, pr, 10.0, safety_margin=0.0)
    assert p.jobs == 10 and p.max_output_tokens == 1000 and p.est_input_tokens == 500
    assert p.output_cost_usd == pytest.approx(0.002) and p.input_cost_usd == pytest.approx(0.0005)
    assert p.total_with_margin_usd == pytest.approx(0.0025)
    assert project_cost({}, 60, pr, 1.0).jobs == 0 and project_cost({}, 60, pr, 1.0).total_cost_usd == 0.0


def test_actual_cost_usd_arithmetic_including_cached_tokens():
    pr = Pricing()
    # 1000 input of which 400 cached, 2000 output: 600*0.10 + 400*0.01 + 2000*0.60 = 1264 micro-dollars
    assert actual_cost_usd(1000, 400, 2000, pr) == pytest.approx(0.001264, abs=1e-12)
    assert actual_cost_usd(45, 0, 100, pr) == pytest.approx((45 * 0.10 + 100 * 0.60) / 1e6, abs=1e-12)
    assert actual_cost_usd(45, None, 100, pr) == actual_cost_usd(45, 0, 100, pr)
    # cached tokens can never exceed input tokens
    assert actual_cost_usd(100, 500, 0, pr) == pytest.approx(100 * 0.01 / 1e6, abs=1e-12)
    # partial information
    assert actual_cost_usd(None, None, None, pr) is None
    assert actual_cost_usd(None, None, 50, pr) == pytest.approx(50 * 0.60 / 1e6, abs=1e-12)
    assert actual_cost_usd(50, None, None, pr) == pytest.approx(50 * 0.10 / 1e6, abs=1e-12)
    assert actual_cost_usd(0, 0, 0, pr) == 0.0
    # custom pricing and rounding to 8 decimals
    custom = Pricing(input_per_1m_usd=1.0, cached_input_per_1m_usd=0.5, output_per_1m_usd=2.0)
    assert actual_cost_usd(1000, 400, 2000, custom) == pytest.approx(0.0048, abs=1e-12)
    assert actual_cost_usd(1, 0, 0, pr) == round(0.10 / 1e6, 8)


# ---------------------------------------------------------------------------------------------
# gates in the commands
# ---------------------------------------------------------------------------------------------

async def _prepare(workdir, api, **kw):
    base = dict(config=str(workdir / "config" / "experiment.json"), runs_per_group=2, seed=7, levels=[16, 100],
                experiment_id="exp-cost", data_dir=str(workdir / "data"))
    base.update(kw)
    args = make_args(**base)
    return args, await prepare(args, api=api)


async def test_cost_ceiling_launch_refuses_before_any_creation(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    cfg = ExperimentConfig.load(args.config)
    for execute in (True, False):
        with pytest.raises(CostCeilingExceeded, match="exceeds ceiling"):
            await launch(make_args(config=args.config, execute=execute, skip_pilot_gate=True, max_cost_usd=0.00001), api=fake_api)
    assert fake_api.create_calls == 0 and len(fake_api.batches) == 0
    store = Store(cfg.db_path)
    try:
        assert store.get_meta("launch_started_at") is None
        assert all(j["creation_state"] == "pending" for j in store.list_jobs(phases=["prod"]))
        assert store.all_attempts() == []
    finally:
        store.close()
    # a config-level ceiling that is too low is refused as well, and an explicit --max-cost-usd overrides it
    cfg.max_cost_usd = 0.00001
    cfg.save(args.config)
    with pytest.raises(CostCeilingExceeded):
        await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True), api=fake_api)
    assert fake_api.create_calls == 0
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True, max_cost_usd=1.0), api=fake_api)
    assert out["wave"]["created"] == 4 and out["plan"]["cost_projection"]["ceiling_usd"] == 1.0
    assert fake_api.create_calls == 4


async def test_cost_ceiling_prepare_refuses_after_uploads_but_without_creations(workdir, fake_api):
    with pytest.raises(CostCeilingExceeded):
        await _prepare(workdir, fake_api, max_cost_usd=0.0001)
    assert fake_api.create_calls == 0
    # the ceiling itself must be positive
    from experiment.config import ConfigError
    with pytest.raises(ConfigError, match="max_cost_usd"):
        await _prepare(workdir, fake_api, max_cost_usd=0.0, reset_config=True)


async def test_cost_ceiling_pilot_refuses_with_a_tiny_ceiling(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    for execute in (True, False):
        with pytest.raises(CostCeilingExceeded, match="exceeds ceiling"):
            await pilot(make_args(config=args.config, execute=execute, max_cost_usd=0.000001, interval=0.01), api=fake_api)
    assert fake_api.create_calls == 0
    cfg = ExperimentConfig.load(args.config)
    store = Store(cfg.db_path)
    try:
        assert store.list_jobs(phases=["pilot"]) == [] and store.get_meta("pilot_status") is None
    finally:
        store.close()
    # the pilot's own projection is small: the default $0.10 pilot ceiling is enough
    ev = await pilot(make_args(config=args.config, execute=True, max_cost_usd=0.10, interval=0.01), api=fake_api)
    assert ev["passed"]


async def test_cost_ceiling_recover_refuses_with_a_tiny_ceiling(workdir):
    obs = "prod-t00100-k0001"
    api = FakeBatchApi(auto_advance=15.0, fail_create_for={obs})
    args, _ = await _prepare(workdir, api)
    await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True), api=api)
    n = api.create_calls
    with pytest.raises(CostCeilingExceeded):
        await recover(make_args(config=args.config, execute=True, max_cost_usd=0.000001), api=api)
    assert api.create_calls == n


async def test_cost_ceiling_spent_or_committed_tracks_worst_case_then_actual(workdir, fake_api):
    args, _ = await _prepare(workdir, fake_api)
    cfg = ExperimentConfig.load(args.config)
    rt = Runtime.open(cfg, api=fake_api)
    try:
        assert spent_or_committed_usd(rt) == {"actual_usd": 0.0, "committed_worst_case_usd": 0.0, "total_usd": 0.0}
    finally:
        await rt.aclose()
    out = await launch(make_args(config=args.config, execute=True, skip_pilot_gate=True), api=fake_api)
    assert out["plan"]["spent_before"]["total_usd"] == 0.0
    rt = Runtime.open(cfg, api=fake_api)
    try:
        spent = spent_or_committed_usd(rt)
        worst = sum((n * cfg.pricing.output_per_1m_usd + cfg.estimated_input_tokens_per_request * cfg.pricing.input_per_1m_usd) / 1e6
                    for n in (16, 16, 100, 100))
        assert spent["actual_usd"] == 0.0 and spent["committed_worst_case_usd"] == pytest.approx(worst, abs=1e-6)
        await monitor(rt, phases=["prod"], interval=0.01)
        spent = spent_or_committed_usd(rt)
        assert spent["committed_worst_case_usd"] == 0.0
        # the fake bills 45 input tokens and exactly max_output_tokens output per job
        expected = sum(actual_cost_usd(45, 0, n, cfg.pricing) for n in (16, 16, 100, 100))
        assert spent["actual_usd"] == pytest.approx(expected, abs=1e-6) and spent["total_usd"] == spent["actual_usd"]
        assert spent["actual_usd"] <= worst
    finally:
        await rt.aclose()
