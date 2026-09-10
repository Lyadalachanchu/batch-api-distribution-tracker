"""Tests for experiment.analyze on synthetic observation frames (no network, no store)."""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from experiment.analyze import FIGURE_ORDER, analyze_frame, render_report, write_outputs
from experiment.config import ExperimentConfig
from experiment.observations import OBSERVATION_COLUMNS
from experiment.redact import contains_secret
from experiment.timeutil import epoch_to_iso

LEVELS = [16, 100, 1000, 3000, 10000]
LAUNCH_EPOCH = 1_789_100_000  # 2026-09-10-ish, integer seconds like the server
EXPECTED_BIAS_CHECKS = {
    "submission_order_balance", "server_created_at_balance", "launch_window", "failure_rate_by_group",
    "early_stop_rate_by_group", "valid_observations_per_group", "actual_vs_requested_ratio",
}


def _row(**kw) -> dict:
    base = {c: None for c in OBSERVATION_COLUMNS}
    base.update(kw)
    return base


def _job(rng: np.random.Generator, obs_id: str, phase: str, level: int, position: int | None, outcome: str = "ok",
         error_message: str | None = None) -> dict:
    """One synthetic observation. outcome: ok | early | failed | creation_failed | infeasible."""
    created = LAUNCH_EPOCH + (int(position * 0.4) if position is not None else 5)
    queue = int(rng.integers(20, 200))
    active = int(30 + 0.04 * level + rng.exponential(60))
    reasoning = int(min(level, max(0, rng.normal(0.3 * level, 0.05 * level)))) if level >= 100 else 0
    row = _row(
        experiment_id="exp-test", observation_id=obs_id, attempt_id=1, phase=phase, is_replacement=(phase == "replacement"),
        requested_output_tokens=level, api_max_output_tokens=level, randomized_launch_position=position,
        local_create_started_at=epoch_to_iso(created - 0.5), local_create_finished_at=epoch_to_iso(created + 0.2),
        local_create_duration_ms=700.0, batch_id=f"batch_{obs_id}", input_file_id=f"file-{level}",
        created_at=created, created_at_iso=epoch_to_iso(created), expires_at=created + 86400,
        server_created_at_offset_seconds=float(created - LAUNCH_EPOCH), creation_state="created", status="completed",
        estimated_cost_usd=None, valid_observation=False,
    )
    if outcome == "creation_failed":
        row.update(batch_id=None, status="creation_error", creation_state="error", created_at=None, created_at_iso=None,
                   expires_at=None, server_created_at_offset_seconds=None, create_error_type="APIStatusError",
                   create_error_message=error_message or "boom", error_message=error_message or "boom", http_status=500)
        return row
    if outcome == "infeasible":
        row.update(response_status=None, http_status=400, error_code="integer_below_min_value",
                   error_message="Expected a value >= 16", finish_or_incomplete_reason=None)
        row.update(in_progress_at=created + queue, completed_at=created + queue + 30, in_progress_at_iso=epoch_to_iso(created + queue),
                   completed_at_iso=epoch_to_iso(created + queue + 30), turnaround_seconds=float(queue + 30),
                   queue_seconds=float(queue), active_seconds=30.0)
        return row
    if outcome == "failed":
        row.update(status="failed", failed_at=created + queue + 5, in_progress_at=created + queue, queue_seconds=float(queue),
                   error_type="batch_request_error", error_message=error_message or "server_error")
        return row
    actual = level if outcome == "ok" else max(1, level // 2)
    inp = 45
    cost = round((inp * 0.10 + actual * 0.60) / 1e6, 8)
    row.update(
        actual_output_tokens=actual, reasoning_tokens=min(reasoning, actual), actual_input_tokens=inp, cached_input_tokens=0,
        total_tokens=inp + actual, in_progress_at=created + queue, finalizing_at=created + queue + active - 2,
        completed_at=created + queue + active, in_progress_at_iso=epoch_to_iso(created + queue),
        completed_at_iso=epoch_to_iso(created + queue + active), turnaround_seconds=float(queue + active),
        queue_seconds=float(queue), active_seconds=float(active), finalizing_seconds=2.0,
        response_status="completed" if outcome == "early" else "incomplete",
        finish_or_incomplete_reason="completed" if outcome == "early" else "max_output_tokens",
        early_stop=(outcome == "early"), http_status=200, openai_request_id=f"req_{obs_id}", output_text_chars=actual * 5,
        output_word_count=actual, estimated_cost_usd=cost, valid_observation=True, error_message=error_message,
    )
    return row


def make_frame(seed: int = 7, error_message: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(LEVELS) * 40) + 1
    rows = []
    i = 0
    for lvl in LEVELS:
        for k in range(40):
            pos = int(positions[i]); i += 1
            if k == 0:
                outcome = "creation_failed"
            elif k == 1:
                outcome = "failed"
            elif k in (2, 3) and lvl >= 1000:
                outcome = "early"
            else:
                outcome = "ok"
            rows.append(_job(rng, f"prod-{lvl}-{k:03d}", "prod", lvl, pos, outcome, error_message=error_message))
    for j, lvl in enumerate(LEVELS):  # 5 replacements for the creation failures
        r = _job(rng, f"repl-{lvl}-000", "replacement", lvl, None, "ok")
        r.update(parent_observation_id=f"prod-{lvl}-000", attempt_id=2, created_at=LAUNCH_EPOCH + 900 + j,
                 server_created_at_offset_seconds=900.0 + j)
        rows.append(r)
    rows.append(_job(rng, "pilot-1-000", "pilot", 1, None, "infeasible"))
    rows.append(_job(rng, "pilot-10-000", "pilot", 10, None, "infeasible"))
    rows.append(_job(rng, "pilot-100-000", "pilot", 100, None, "ok"))
    df = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    assert list(df.columns) == OBSERVATION_COLUMNS
    return df


def make_meta() -> dict:
    return {
        "launch_started_at": epoch_to_iso(LAUNCH_EPOCH), "launch_finished_at": epoch_to_iso(LAUNCH_EPOCH + 95), "pilot_status": "passed",
        "launch_summary": {"wave": {"wave": "launch", "planned": 200, "created": 195, "errors": 5, "unknown": 0, "skipped": 0,
                                    "rate_limit_429s": 0, "duration_seconds": 95.0, "stopped_early": False, "stop_reason": None},
                           "plan": {"cost_projection": {"total_cost_usd": 3.39, "total_with_margin_usd": 3.73, "ceiling_usd": 4.0}}},
        "prepare_summary": {"cost_projection": {"jobs": 2000, "total_cost_usd": 3.4, "total_with_margin_usd": 3.74,
                                                "safety_margin": 0.1, "ceiling_usd": 4.0}},
        "pilot_summary": None,
    }


@pytest.fixture(scope="module")
def result():
    return analyze_frame(make_frame(), ExperimentConfig(), make_meta(), bootstrap=50)


def test_analyze_frame_groups(result):
    assert set(result) >= {"groups", "bias_flags", "model_fit", "cost", "datasets", "launch", "figures"}
    groups = result["groups"]
    assert [g["requested_output_tokens"] for g in groups] == LEVELS
    for g in groups:
        assert g["submitted"] == 40
        assert g["created"] == 39 and g["creation_failed"] == 1
        assert g["failed"] == 1 and g["completed"] == 38
        assert g["valid"] == 38 == g["turnaround_n"]
        assert g["n_replacement"] == 0
        expected_early = 2 if g["requested_output_tokens"] >= 1000 else 0
        assert g["early_stops"] == expected_early
        assert math.isclose(g["early_stop_rate"], expected_early / 38)
        assert g["turnaround_min"] <= g["turnaround_median"] <= g["turnaround_max"]
        assert g["turnaround_p10"] <= g["turnaround_p25"] <= g["turnaround_median"] <= g["turnaround_p75"] <= g["turnaround_p95"] <= g["turnaround_p99"]
        assert g["turnaround_median_ci_low"] <= g["turnaround_median_ci_high"]
        assert g["turnaround_p95_ci_low"] <= g["turnaround_p95_ci_high"]
        assert g["turnaround_median_ci_low"] <= g["turnaround_max"] and g["turnaround_median_ci_high"] >= g["turnaround_min"]
        assert g["turnaround_sd"] > 0
        for prefix in ("queue", "active"):
            assert g[f"{prefix}_n"] == 38
            assert g[f"{prefix}_min"] <= g[f"{prefix}_median"] <= g[f"{prefix}_max"]
            assert g[f"{prefix}_median_ci_low"] <= g[f"{prefix}_median_ci_high"]
        assert 0 < g["ratio_actual_requested_median"] <= 1.0
        assert g["actual_output_tokens_mean"] <= g["requested_output_tokens"]
        assert g["cost_usd"] > 0
    # datasets: main is prod only; replacement / with_replacements / pilot exist and are separate
    ds = result["datasets"]
    assert ds["main"]["phases"] == ["prod"] and ds["main"]["n_rows"] == 200
    assert ds["replacement"]["n_rows"] == 5
    assert ds["with_replacements"]["n_rows"] == 5 * 38 + 5 and ds["with_replacements"]["n_replacement"] == 5
    assert all(g["n_replacement"] == 1 and g["valid"] == 39 for g in ds["with_replacements"]["groups"])
    assert ds["pilot"]["n_rows"] == 3 and ds["pilot"]["levels"] == [1, 10, 100]
    assert result["warnings"] == []


def test_figures_and_bias_checks(result):
    figs = result["figures"]
    assert list(figs) == FIGURE_ORDER and len(figs) == 10
    assert all(isinstance(f, go.Figure) for f in figs.values())
    assert all(len(f.data) > 0 for f in figs.values())
    assert set(result["figure_captions"]) == set(FIGURE_ORDER)
    names = [c["name"] for c in result["bias_flags"]]
    assert set(names) == EXPECTED_BIAS_CHECKS
    for c in result["bias_flags"]:
        assert isinstance(c["flagged"], bool) and isinstance(c["detail"], str) and c["detail"]
    by = {c["name"]: c for c in result["bias_flags"]}
    assert by["valid_observations_per_group"]["flagged"] is True  # 38 < 400 runs_per_group
    assert by["launch_window"]["flagged"] is False
    assert "not testable" not in by["submission_order_balance"]["detail"]


def test_model_cost_launch(result):
    mf = result["model_fit"]
    assert "exploratory" in mf["note"].lower()
    for key in ("actual_tokens", "requested_factor"):
        m = mf[key]
        assert "error" not in m, m
        assert m["n"] == 190 and 0 <= m["r_squared"] <= 1
        terms = {c["term"] for c in m["coefficients"]}
        assert "randomized_launch_position" in terms and "server_created_at_offset_seconds" in terms
        for c in m["coefficients"]:
            assert c["ci_low"] <= c["estimate"] <= c["ci_high"]
            assert c["boot_ci_low"] <= c["boot_ci_high"]
    assert "requested = 10000 tokens" in {c["term"] for c in mf["requested_factor"]["coefficients"]}
    cost = result["cost"]
    assert cost["pricing"]["source"]
    assert set(cost["by_phase"]) == {"prod", "replacement", "pilot"}
    assert cost["production_usd"] > cost["pilot_usd"] > 0
    assert math.isclose(cost["total_usd"], sum(v["cost_usd"] for v in cost["by_phase"].values()))
    assert cost["projected_max"]["total_cost_usd"] == 3.4 and 0 < cost["production_share_of_projected_max"] < 1
    launch = result["launch"]
    assert math.isclose(launch["window_seconds"], 95.0)
    assert launch["n_submitted"] == 200 and launch["n_created"] == 195
    assert launch["reference_source"] == "launch_started_at"
    assert set(launch["created_at_offset_median_by_group"]) == {f"{l:,} tokens" for l in LEVELS}


def test_render_report(result, tmp_path):
    out = tmp_path / "reports" / "experiment_report.html"
    path = render_report(result, ExperimentConfig(), str(out))
    assert os.path.exists(path) and os.path.getsize(path) > 100_000
    html = out.read_text(encoding="utf-8")
    assert html.count("<script id=\"plotly-js\">") == 1
    without_bundle = re.sub(r'<script id="plotly-js">.*?</script>', "", html, count=1, flags=re.S)
    # self-contained: no external script or stylesheet references anywhere outside the embedded bundle
    assert re.search(r'<(script|link)[^>]+(src|href)="https?://', without_bundle) is None
    assert "Plotly.newPlot" in without_bundle
    assert not contains_secret(without_bundle)
    assert re.search(r"sk-[A-Za-z0-9]{20,}", html) is None
    for needle in ("Design deviation", "Bias checks", "Limitations", "Cost statement", "Pilot summary", "Exploratory model fit",
                   ExperimentConfig().experiment_id, "Generate a continuous sequence of unrelated lowercase words"):
        assert needle in html, needle
    for name in FIGURE_ORDER:
        assert f'id="fig-{name}"' in html
    assert "class=\"banner\"" not in html  # no warning banner when prod rows exist


def test_secret_in_error_message_is_redacted(tmp_path):
    key = "sk-proj-" + "A1b2C3d4" * 6
    df = make_frame(error_message=f"auth failed for {key}")
    res = analyze_frame(df, ExperimentConfig(), make_meta(), bootstrap=10)
    path = render_report(res, ExperimentConfig(), str(tmp_path / "r.html"))
    html = open(path, encoding="utf-8").read()
    assert key not in html


def test_single_tiny_group_does_not_raise(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame([_job(rng, "prod-100-000", "prod", 100, 1, "ok")], columns=OBSERVATION_COLUMNS)
    cfg = ExperimentConfig()
    res = analyze_frame(df, cfg, {}, bootstrap=20)
    g = {x["requested_output_tokens"]: x for x in res["groups"]}
    assert set(g) == set(cfg.output_token_levels)
    assert g[100]["valid"] == 1 and g[100]["turnaround_n"] == 1
    assert g[100]["turnaround_median"] == g[100]["turnaround_min"] == g[100]["turnaround_max"]
    assert math.isnan(g[100]["turnaround_sd"]) and math.isnan(g[100]["turnaround_median_ci_low"])
    assert g[16]["submitted"] == 0 and math.isnan(g[16]["turnaround_median"])
    assert len(res["figures"]) == 10
    assert "error" in res["model_fit"]["actual_tokens"]
    assert all(isinstance(c["flagged"], bool) for c in res["bias_flags"])
    assert res["launch"]["reference_source"] == "earliest created_at"
    path = render_report(res, cfg, str(tmp_path / "tiny.html"))
    assert os.path.getsize(path) > 100_000


def test_no_prod_rows_warns(tmp_path):
    df = make_frame()
    df = df[df["phase"] == "pilot"].reset_index(drop=True)
    res = analyze_frame(df, ExperimentConfig(), {}, bootstrap=10)
    assert res["warnings"] and "prod" in res["warnings"][0]
    assert res["datasets"]["main"]["phases"] == ["pilot"]
    assert 1 in res["levels"] and 10 in res["levels"]
    html = open(render_report(res, ExperimentConfig(), str(tmp_path / "pilot_only.html")), encoding="utf-8").read()
    assert 'class="banner"' in html


def test_empty_frame_does_not_raise():
    df = pd.DataFrame(columns=OBSERVATION_COLUMNS)
    res = analyze_frame(df, ExperimentConfig(), {}, bootstrap=10)
    assert res["warnings"] and len(res["figures"]) == 10
    assert all(g["submitted"] == 0 for g in res["groups"])


def test_write_outputs(result, tmp_path):
    cfg = ExperimentConfig(data_dir=str(tmp_path / "data"))
    paths = write_outputs(result, cfg)
    assert set(paths) == {"summary_stats_csv", "summary_stats_json", "bias_checks_json", "model_fit_json", "cost_statement_json"}
    for p in paths.values():
        assert os.path.exists(p)
    csv = pd.read_csv(paths["summary_stats_csv"])
    assert set(csv["dataset"]) == {"main", "replacement", "with_replacements", "pilot"}
    assert len(csv[csv["dataset"] == "main"]) == 5
    for key in ("summary_stats_json", "bias_checks_json", "model_fit_json", "cost_statement_json"):
        with open(paths[key], encoding="utf-8") as f:
            data = json.load(f)  # strict JSON: NaN must have become null
        assert isinstance(data, dict)
    with open(paths["bias_checks_json"], encoding="utf-8") as f:
        assert {c["name"] for c in json.load(f)["checks"]} == EXPECTED_BIAS_CHECKS
