"""analyze: descriptive statistics, bias checks, exploratory models, figures and a self-contained HTML report.

All computation lives in ``analyze_frame`` (pure, no I/O) so it can be exercised on synthetic frames;
``analyze`` wires it to the SQLite store and the file system, ``write_outputs`` writes the machine-readable
summaries and ``render_report`` produces the offline HTML page.

Datasets
--------
* ``main``               intent-to-measure: every ``phase == "prod"`` row (creation failures, failed and expired
                         jobs included, so per-group denominators equal ``runs_per_group``).
* ``replacement``        ``phase == "replacement"`` rows alone.
* ``with_replacements``  valid prod rows + valid replacement rows (reported separately, never merged into main).
* ``pilot``              ``phase == "pilot"`` rows, excluded from every production statistic.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline
from jinja2 import Environment, FileSystemLoader
from scipy import stats as sps

from .config import ExperimentConfig
from .observations import build_observations, write_observations
from .redact import redact_text
from .store import Store
from .timeutil import epoch_to_iso, iso_now, iso_to_epoch

# ------------------------------------------------------------------------------------------------------------
# Palette: the dataviz reference instance on the light surface. Token levels are *ordered* tiers, so the group
# colour is a one-hue ordinal ramp (blue steps 250..700, validated with validate_palette.js --ordinal). Nominal
# series (outcome kinds; the median/p95 pair) take the categorical slots in fixed order. Marker symbols are the
# secondary encoding for scatter charts where any two groups can be neighbours.
# ------------------------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
ORDINAL_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon", "pentagon", "triangle-down"]

FIGURE_ORDER = [
    "ecdf_turnaround", "violin_turnaround", "median_p95_vs_tokens", "queue_time", "active_time",
    "actual_tokens_vs_turnaround", "completion_curve", "created_at_by_group", "launch_position_vs_turnaround",
    "outcomes_by_group",
]
QUANTILES = {"p10": 10, "p25": 25, "median": 50, "p75": 75, "p90": 90, "p95": 95, "p99": 99}
LOCATION_KEYS = ["n", "mean", "sd", "min", *QUANTILES, "max", "median_ci_low", "median_ci_high", "p95_ci_low", "p95_ci_high"]
PLOTLYJS_PLACEHOLDER = "__PLOTLYJS_BUNDLE_PLACEHOLDER__"
TEMPLATE_NAME = "report_template.html.j2"
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NOTE = ("Exploratory association only. The launch is a single randomised wave on one day; the fit describes "
              "that wave and is not a causal estimate nor a guarantee of future Batch API performance.")


# ------------------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------------------
def _is_missing(v: Any) -> bool:
    if v is None or v is pd.NA:
        return True
    return isinstance(v, float) and math.isnan(v)


def _num(s: pd.Series) -> np.ndarray:
    """Float ndarray with NaN for missing, whatever the input dtype."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _bool_to_float(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for i, v in enumerate(s.tolist()):
        if _is_missing(v):
            continue
        out[i] = 1.0 if bool(v) else 0.0
    return out


def _nan_or(v: Any) -> float:
    return float("nan") if _is_missing(v) else float(v)


def _label(level: Any) -> str:
    return f"{int(level):,} tokens"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _group_colors(levels: list[int]) -> dict[int, str]:
    n = len(levels)
    preferred = {1: [4], 2: [2, 7], 3: [0, 4, 9], 4: [0, 3, 6, 9], 5: [0, 2, 4, 6, 9]}
    if n in preferred:
        idx = preferred[n]
    else:
        idx = [int(round(i)) for i in np.linspace(0, len(ORDINAL_RAMP) - 1, max(n, 1))]
    return {lvl: ORDINAL_RAMP[idx[i % len(idx)]] for i, lvl in enumerate(levels)}


def _group_symbols(levels: list[int]) -> dict[int, str]:
    return {lvl: SYMBOLS[i % len(SYMBOLS)] for i, lvl in enumerate(levels)}


def _clean(obj: Any) -> Any:
    """Make a nested structure JSON-serialisable (numpy scalars -> python, NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    if isinstance(obj, go.Figure):
        return None
    return obj


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------
def bootstrap_percentile_ci(x: np.ndarray, qs: list[float], n_boot: int, rng: np.random.Generator,
                            level: float = 0.95) -> dict[float, tuple[float, float]]:
    """Percentile-bootstrap CI of sample percentiles ``qs`` (vectorised: index matrix + np.percentile along axis)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2 or n_boot < 1:
        return {q: (float("nan"), float("nan")) for q in qs}
    chunk = max(1, min(n_boot, 4_000_000 // n))
    parts: list[np.ndarray] = []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        parts.append(np.percentile(x[idx], qs, axis=1))  # shape (len(qs), m)
        done += m
    stat = np.concatenate(parts, axis=1)
    lo, hi = np.percentile(stat, [(1 - level) / 2 * 100, (1 + level) / 2 * 100], axis=1)
    return {q: (float(lo[i]), float(hi[i])) for i, q in enumerate(qs)}


def _location_stats(x: np.ndarray, prefix: str, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    out: dict[str, float] = {f"{prefix}_{k}": float("nan") for k in LOCATION_KEYS}
    out[f"{prefix}_n"] = n
    if n == 0:
        return out
    out[f"{prefix}_mean"] = float(x.mean())
    out[f"{prefix}_sd"] = float(x.std(ddof=1)) if n >= 2 else float("nan")
    out[f"{prefix}_min"] = float(x.min())
    out[f"{prefix}_max"] = float(x.max())
    pct = np.percentile(x, list(QUANTILES.values()))
    for (name, _), val in zip(QUANTILES.items(), pct):
        out[f"{prefix}_{name}"] = float(val)
    ci = bootstrap_percentile_ci(x, [50, 95], n_boot, rng)
    out[f"{prefix}_median_ci_low"], out[f"{prefix}_median_ci_high"] = ci[50]
    out[f"{prefix}_p95_ci_low"], out[f"{prefix}_p95_ci_high"] = ci[95]
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the observation frame with typed working columns (prefixed ``_``)."""
    d = df.copy()
    if "phase" not in d.columns:
        d["phase"] = None
    d["_req"] = _num(d["requested_output_tokens"])
    d["_ta"] = _num(d["turnaround_seconds"])
    d["_queue"] = _num(d["queue_seconds"])
    d["_active"] = _num(d["active_seconds"])
    d["_valid"] = (_bool_to_float(d["valid_observation"]) == 1.0) & ~np.isnan(d["_ta"].to_numpy(dtype=float))
    d["_early"] = _bool_to_float(d["early_stop"])
    d["_actual"] = _num(d["actual_output_tokens"])
    d["_reasoning"] = _num(d["reasoning_tokens"])
    d["_created"] = _num(d["created_at"])
    d["_completed"] = _num(d["completed_at"])
    d["_pos"] = _num(d["randomized_launch_position"])
    d["_offset"] = _num(d["server_created_at_offset_seconds"])
    d["_cost"] = _num(d["estimated_cost_usd"])
    d["_has_batch"] = d["batch_id"].notna().to_numpy()
    d["_status"] = d["status"].astype(object).where(d["status"].notna(), None)
    d["_is_repl"] = _bool_to_float(d["is_replacement"]) == 1.0
    return d


def _group_stats(sub: pd.DataFrame, level: int, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    status = sub["_status"]
    valid = sub[sub["_valid"]]
    early = valid["_early"].to_numpy(dtype=float)
    early_known = early[~np.isnan(early)]
    actual = valid["_actual"].to_numpy(dtype=float)
    ratio = actual / float(level) if level else np.full(actual.shape, np.nan)
    reasoning = valid["_reasoning"].to_numpy(dtype=float)
    g: dict[str, Any] = {
        "requested_output_tokens": int(level),
        "label": _label(level),
        "submitted": int(len(sub)),
        "created": int(sub["_has_batch"].sum()),
        "creation_failed": int((~sub["_has_batch"]).sum()),
        "completed": int((status == "completed").sum()),
        "failed": int((status == "failed").sum()),
        "expired": int((status == "expired").sum()),
        "cancelled": int((status == "cancelled").sum()),
        "valid": int(len(valid)),
        "n_replacement": int(sub["_is_repl"].sum()),
        "early_stops": int(np.nansum(early)) if early.size else 0,
        "early_stop_known": int(early_known.size),
        "early_stop_rate": float(early_known.mean()) if early_known.size else float("nan"),
        "failure_rate": float((status != "completed").sum() / len(sub)) if len(sub) else float("nan"),
        "actual_output_tokens_mean": float(np.nanmean(actual)) if np.any(~np.isnan(actual)) else float("nan"),
        "actual_output_tokens_median": float(np.nanmedian(actual)) if np.any(~np.isnan(actual)) else float("nan"),
        "ratio_actual_requested_mean": float(np.nanmean(ratio)) if np.any(~np.isnan(ratio)) else float("nan"),
        "ratio_actual_requested_median": float(np.nanmedian(ratio)) if np.any(~np.isnan(ratio)) else float("nan"),
        "reasoning_tokens_mean": float(np.nanmean(reasoning)) if np.any(~np.isnan(reasoning)) else float("nan"),
        "cost_usd": float(np.nansum(sub["_cost"].to_numpy(dtype=float))),
    }
    g["other_status"] = int(g["submitted"] - g["creation_failed"] - g["completed"] - g["failed"] - g["expired"] - g["cancelled"])
    g.update(_location_stats(valid["_ta"].to_numpy(dtype=float), "turnaround", rng, n_boot))
    g.update(_location_stats(valid["_queue"].to_numpy(dtype=float), "queue", rng, n_boot))
    g.update(_location_stats(valid["_active"].to_numpy(dtype=float), "active", rng, n_boot))
    return g


def _dataset(frame: pd.DataFrame, name: str, title: str, phases: list[str], levels: list[int], description: str,
             n_boot: int, seed: int, ds_index: int) -> dict[str, Any]:
    groups = []
    for lvl in levels:
        rng = np.random.default_rng([int(seed), int(ds_index), int(lvl)])
        groups.append(_group_stats(frame[frame["_req"] == float(lvl)], int(lvl), rng, n_boot))
    return {
        "name": name, "title": title, "phases": phases, "description": description, "levels": [int(l) for l in levels],
        "n_rows": int(len(frame)), "n_valid": int(frame["_valid"].sum()), "n_replacement": int(frame["_is_repl"].sum()),
        "groups": groups,
    }


def _levels_in(frame: pd.DataFrame) -> set[int]:
    vals = frame["_req"].to_numpy(dtype=float)
    return {int(v) for v in vals[~np.isnan(vals)]}


# ------------------------------------------------------------------------------------------------------------
# launch summary & bias checks
# ------------------------------------------------------------------------------------------------------------
def _reference_epoch(meta: dict[str, Any], frame: pd.DataFrame) -> tuple[float | None, str | None]:
    started = meta.get("launch_started_at")
    if started:
        try:
            return float(iso_to_epoch(started)), "launch_started_at"
        except (TypeError, ValueError):
            pass
    created = frame["_created"].to_numpy(dtype=float)
    created = created[~np.isnan(created)]
    if created.size:
        return float(created.min()), "earliest created_at"
    return None, None


def _launch_summary(meta: dict[str, Any], frame: pd.DataFrame, levels: list[int], ref: float | None,
                    ref_source: str | None) -> dict[str, Any]:
    ls = meta.get("launch_summary") if isinstance(meta.get("launch_summary"), dict) else {}
    started, finished = meta.get("launch_started_at"), meta.get("launch_finished_at")
    window = None
    if started and finished:
        try:
            window = float(iso_to_epoch(finished) - iso_to_epoch(started))
        except (TypeError, ValueError):
            window = None
    created = frame["_created"].to_numpy(dtype=float)
    created = created[~np.isnan(created)]
    span = float(created.max() - created.min()) if created.size else None
    med_by_group: dict[str, float | None] = {}
    for lvl in levels:
        x = frame.loc[frame["_req"] == float(lvl), "_created_offset"].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        med_by_group[_label(lvl)] = float(np.median(x)) if x.size else None
    meds = [v for v in med_by_group.values() if v is not None]
    return {
        "launch_started_at": started, "launch_finished_at": finished, "window_seconds": window,
        "reference_epoch": ref, "reference_source": ref_source,
        "created_at_min_iso": epoch_to_iso(created.min()) if created.size else None,
        "created_at_max_iso": epoch_to_iso(created.max()) if created.size else None,
        "created_at_span_seconds": span,
        "created_at_offset_median_by_group": med_by_group,
        "max_median_gap_seconds": (max(meds) - min(meds)) if len(meds) >= 2 else None,
        "n_submitted": int(len(frame)), "n_created": int(frame["_has_batch"].sum()),
        "wave": ls.get("wave"), "created_at_balance_at_launch": ls.get("created_at_balance"),
        "pilot_status": meta.get("pilot_status"),
    }


def _kruskal(frame: pd.DataFrame, levels: list[int], col: str) -> tuple[float | None, float | None, dict[str, float], str | None]:
    samples, meds = [], {}
    for lvl in levels:
        x = frame.loc[frame["_req"] == float(lvl), col].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if x.size:
            samples.append(x)
            meds[_label(lvl)] = float(np.median(x))
    if len(samples) < 2:
        return None, None, meds, "not testable (fewer than two groups with data)"
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            h, p = sps.kruskal(*samples)
        if not (math.isfinite(float(h)) and math.isfinite(float(p))):
            return None, None, meds, "not testable (all values identical)"
        return float(h), float(p), meds, None
    except ValueError as e:
        return None, None, meds, f"not testable ({e})"


def _chi2(table: list[list[int]]) -> tuple[float | None, float | None, str | None]:
    arr = np.asarray(table, dtype=float)
    if arr.shape[0] < 2:
        return None, None, "not testable (fewer than two groups with data)"
    if arr.sum() == 0 or np.any(arr.sum(axis=0) == 0):
        return None, None, "not testable (an outcome column is empty in every group)"
    try:
        chi2, p, _, _ = sps.chi2_contingency(arr)
        return float(chi2), float(p), None
    except ValueError as e:
        return None, None, f"not testable ({e})"


def _fmt_rates(d: dict[str, float]) -> str:
    return ", ".join(f"{k}: {v:.1%}" for k, v in d.items())


def _bias_checks(frame: pd.DataFrame, groups: list[dict[str, Any]], levels: list[int], cfg: ExperimentConfig,
                 launch: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # (a) submission-order balance
    h, p, meds, note = _kruskal(frame, levels, "_pos")
    flagged = p is not None and p < 0.01
    detail = note or (f"Kruskal-Wallis H={h:.2f}, p={p:.3g}. Median launch position by group: "
                      + ", ".join(f"{k}: {v:,.0f}" for k, v in meds.items()))
    checks.append({"name": "submission_order_balance", "flagged": bool(flagged), "detail": detail,
                   "values": {"H": h, "p": p, "median_position_by_group": meds}})

    # (b) server created_at balance
    h, p, meds, note = _kruskal(frame, levels, "_created")
    gap = (max(meds.values()) - min(meds.values())) if len(meds) >= 2 else None
    flagged = (p is not None and p < 0.01) or (gap is not None and gap > 60)
    offs = launch.get("created_at_offset_median_by_group") or {}
    detail = note or (f"Kruskal-Wallis H={h:.2f}, p={p:.3g}; max gap between group median created_at = {gap:.1f} s. "
                      "Median offset from launch start by group: "
                      + ", ".join(f"{k}: {v:,.1f} s" for k, v in offs.items() if v is not None))
    checks.append({"name": "server_created_at_balance", "flagged": bool(flagged), "detail": detail,
                   "values": {"H": h, "p": p, "max_median_gap_seconds": gap, "median_offset_by_group": offs}})

    # (c) launch window
    window, span = launch.get("window_seconds"), launch.get("created_at_span_seconds")
    if window is None and span is None:
        checks.append({"name": "launch_window", "flagged": False, "detail": "not testable (no launch timestamps recorded)",
                       "values": {}})
    else:
        flagged = (window is not None and window > 600) or (span is not None and span > 600)
        parts = []
        parts.append(f"local launch window {window:,.1f} s" if window is not None else "local launch window unknown")
        parts.append(f"server created_at span {span:,.1f} s" if span is not None else "server created_at span unknown")
        checks.append({"name": "launch_window", "flagged": bool(flagged), "detail": "; ".join(parts) + " (flag threshold 600 s)",
                       "values": {"window_seconds": window, "created_at_span_seconds": span}})

    # (d) failure rate and early-stop rate by group
    rows, rates = [], {}
    for g in groups:
        if g["submitted"] > 0:
            fail = g["submitted"] - g["completed"]
            rows.append([fail, g["completed"]])
            rates[g["label"]] = fail / g["submitted"]
    chi2, p, note = _chi2(rows)
    spread = (max(rates.values()) - min(rates.values())) if rates else None
    flagged = (p is not None and p < 0.01) or (spread is not None and spread > 0.05)
    if note and not rates:
        detail = note
    else:
        detail = (f"chi-square={chi2:.2f}, p={p:.3g}; " if p is not None else f"{note}; ") + \
                 f"non-completed share by group: {_fmt_rates(rates)} (spread {spread:.1%})"
    checks.append({"name": "failure_rate_by_group", "flagged": bool(flagged), "detail": detail,
                   "values": {"chi2": chi2, "p": p, "rate_by_group": rates, "spread": spread}})

    rows, rates = [], {}
    for g in groups:
        if g["early_stop_known"] > 0:
            rows.append([g["early_stops"], g["early_stop_known"] - g["early_stops"]])
            rates[g["label"]] = g["early_stops"] / g["early_stop_known"]
    chi2, p, note = _chi2(rows)
    spread = (max(rates.values()) - min(rates.values())) if rates else None
    flagged = (p is not None and p < 0.01) or (spread is not None and spread > 0.05)
    if note and not rates:
        detail = note
    else:
        detail = (f"chi-square={chi2:.2f}, p={p:.3g}; " if p is not None else f"{note}; ") + \
                 f"early-stop rate by group: {_fmt_rates(rates)} (spread {spread:.1%})"
    checks.append({"name": "early_stop_rate_by_group", "flagged": bool(flagged), "detail": detail,
                   "values": {"chi2": chi2, "p": p, "rate_by_group": rates, "spread": spread}})

    # (e) valid observations per group
    short = {g["label"]: g["valid"] for g in groups if g["valid"] < cfg.runs_per_group}
    detail = (f"groups below runs_per_group={cfg.runs_per_group}: " + ", ".join(f"{k}: {v}" for k, v in short.items())
              if short else f"every group has at least runs_per_group={cfg.runs_per_group} valid observations")
    checks.append({"name": "valid_observations_per_group", "flagged": bool(short), "detail": detail,
                   "values": {"runs_per_group": cfg.runs_per_group, "valid_by_group": {g["label"]: g["valid"] for g in groups}}})

    # (f) actual vs requested length
    ratios = {g["label"]: g["ratio_actual_requested_median"] for g in groups if not _is_missing(g["ratio_actual_requested_median"])}
    if not ratios:
        checks.append({"name": "actual_vs_requested_ratio", "flagged": False, "detail": "not testable (no valid observations)",
                       "values": {}})
    else:
        bad = {k: v for k, v in ratios.items() if v < 0.9 or v > 1.0}
        detail = "median actual/requested by group: " + ", ".join(f"{k}: {v:.3f}" for k, v in ratios.items())
        if bad:
            detail += "; outside [0.9, 1.0]: " + ", ".join(bad)
        checks.append({"name": "actual_vs_requested_ratio", "flagged": bool(bad), "detail": detail,
                       "values": {"median_ratio_by_group": ratios}})
    return checks


# ------------------------------------------------------------------------------------------------------------
# exploratory models
# ------------------------------------------------------------------------------------------------------------
def _pretty_term(term: str) -> str:
    if term.startswith("C(requested_output_tokens)[T."):
        return "requested = " + term[len("C(requested_output_tokens)[T."):-1] + " tokens"
    return term


def _fit_one(d: pd.DataFrame, formula: str, display_formula: str, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    out: dict[str, Any] = {"formula": display_formula, "cov_type": "HC3", "n": int(len(d))}
    try:
        model = smf.ols(formula, data=d)
        X = np.asarray(model.exog, dtype=float)
        y = np.asarray(model.endog, dtype=float)
        n, k = X.shape
        if n < k + 5:
            out["error"] = f"too few rows ({n}) for {k} parameters"
            return out
        if np.linalg.matrix_rank(X) < k:
            out["error"] = "singular design matrix (a predictor is constant or collinear)"
            return out
        res = model.fit(cov_type="HC3")
        ci = res.conf_int(alpha=0.05)
        boots: list[np.ndarray] = []
        skipped = 0
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            Xb = X[idx]
            if np.linalg.matrix_rank(Xb) < k:
                skipped += 1
                continue
            boots.append(np.linalg.lstsq(Xb, y[idx], rcond=None)[0])
        blo = bhi = None
        if len(boots) >= 20:
            arr = np.vstack(boots)
            blo, bhi = np.percentile(arr, [2.5, 97.5], axis=0)
        coefs = []
        for i, term in enumerate(res.params.index):
            est = float(res.params.iloc[i])
            coefs.append({
                "term": _pretty_term(str(term)), "raw_term": str(term), "estimate": est,
                "robust_se": float(res.bse.iloc[i]), "ci_low": float(ci.iloc[i, 0]), "ci_high": float(ci.iloc[i, 1]),
                "p_value": float(res.pvalues.iloc[i]),
                "boot_ci_low": float(blo[i]) if blo is not None else float("nan"),
                "boot_ci_high": float(bhi[i]) if bhi is not None else float("nan"),
                "exp_estimate": float(math.exp(est)) if abs(est) < 700 else float("nan"),
            })
        out.update({
            "k": int(k), "r_squared": float(res.rsquared), "adj_r_squared": float(res.rsquared_adj),
            "bootstrap_resamples": int(len(boots)), "bootstrap_skipped": int(skipped), "coefficients": coefs,
        })
    except Exception as e:  # noqa: BLE001 - never let a degenerate fit abort the report
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _fit_models(valid: pd.DataFrame, n_boot: int, seed: int, offset_source: str | None) -> dict[str, Any]:
    d = pd.DataFrame({
        "turnaround_seconds": valid["_ta"].to_numpy(dtype=float),
        "actual_output_tokens": valid["_actual"].to_numpy(dtype=float),
        "randomized_launch_position": valid["_pos"].to_numpy(dtype=float),
        "server_created_at_offset_seconds": valid["_created_offset"].to_numpy(dtype=float),
        "requested_output_tokens": valid["_req"].to_numpy(dtype=float),
    })
    d = d.dropna()
    d = d[d["turnaround_seconds"] > 0]
    d["log_turnaround"] = np.log(d["turnaround_seconds"].to_numpy(dtype=float))
    d["log1p_actual_output_tokens"] = np.log1p(np.clip(d["actual_output_tokens"].to_numpy(dtype=float), 0, None))
    d["requested_output_tokens"] = d["requested_output_tokens"].astype(int)
    out: dict[str, Any] = {"note": MODEL_NOTE, "n": int(len(d)), "response": "log(turnaround_seconds)",
                           "bootstrap_resamples_requested": int(min(n_boot, 1000)),
                           "created_at_offset_source": offset_source or "server_created_at_offset_seconds"}
    nb = int(min(n_boot, 1000))
    if len(d) < 8:
        out["error"] = f"too few valid rows ({len(d)}) to fit the exploratory models"
        out["actual_tokens"] = {"error": out["error"]}
        out["requested_factor"] = {"error": out["error"]}
        return out
    out["actual_tokens"] = _fit_one(
        d, "log_turnaround ~ log1p_actual_output_tokens + randomized_launch_position + server_created_at_offset_seconds",
        "log(turnaround_seconds) ~ log1p(actual_output_tokens) + randomized_launch_position + server_created_at_offset_seconds",
        nb, np.random.default_rng([int(seed), 101]))
    out["requested_factor"] = _fit_one(
        d, "log_turnaround ~ C(requested_output_tokens) + randomized_launch_position + server_created_at_offset_seconds",
        "log(turnaround_seconds) ~ C(requested_output_tokens) + randomized_launch_position + server_created_at_offset_seconds",
        nb, np.random.default_rng([int(seed), 102]))
    return out


# ------------------------------------------------------------------------------------------------------------
# cost statement
# ------------------------------------------------------------------------------------------------------------
def _usage(sub: pd.DataFrame) -> dict[str, Any]:
    def s(col: str) -> int:
        return int(np.nansum(_num(sub[col]))) if len(sub) else 0
    return {
        "jobs": int(len(sub)), "jobs_with_usage": int(sub["_cost"].notna().sum()),
        "input_tokens": s("actual_input_tokens"), "cached_input_tokens": s("cached_input_tokens"),
        "output_tokens": s("actual_output_tokens"), "reasoning_tokens": s("reasoning_tokens"), "total_tokens": s("total_tokens"),
        "cost_usd": float(np.nansum(sub["_cost"].to_numpy(dtype=float))) if len(sub) else 0.0,
    }


def _cost_statement(df: pd.DataFrame, cfg: ExperimentConfig, meta: dict[str, Any]) -> dict[str, Any]:
    present = [str(p) for p in df["phase"].dropna().unique().tolist()]
    phases = [p for p in ("prod", "replacement", "pilot") if p in present] + sorted(p for p in present if p not in ("prod", "replacement", "pilot"))
    by_phase = {p: _usage(df[df["phase"] == p]) for p in phases}
    by_phase_group = []
    for p in phases:
        sub = df[df["phase"] == p]
        for lvl in sorted(_levels_in(sub)):
            by_phase_group.append({"phase": p, "requested_output_tokens": int(lvl), "label": _label(lvl),
                                   **_usage(sub[sub["_req"] == float(lvl)])})
    production = sum(by_phase[p]["cost_usd"] for p in ("prod", "replacement") if p in by_phase)
    pilot = by_phase["pilot"]["cost_usd"] if "pilot" in by_phase else 0.0
    prep = meta.get("prepare_summary") if isinstance(meta.get("prepare_summary"), dict) else {}
    projected = prep.get("cost_projection") if isinstance(prep.get("cost_projection"), dict) else None
    ls = meta.get("launch_summary") if isinstance(meta.get("launch_summary"), dict) else {}
    plan = ls.get("plan") if isinstance(ls.get("plan"), dict) else {}
    launch_projection = plan.get("cost_projection") if isinstance(plan.get("cost_projection"), dict) else None
    share = None
    if projected and projected.get("total_cost_usd"):
        share = production / float(projected["total_cost_usd"])
    return {
        "pricing": cfg.pricing.to_dict(), "by_phase": by_phase, "by_phase_group": by_phase_group,
        "production_usd": float(production), "pilot_usd": float(pilot), "total_usd": float(sum(v["cost_usd"] for v in by_phase.values())),
        "total_tokens": int(sum(v["total_tokens"] for v in by_phase.values())),
        "projected_max": projected, "launch_projection": launch_projection, "production_share_of_projected_max": share,
        "note": ("Actual usage priced at the documented Batch rates in the pricing table; the projected maximum assumes every "
                 "request emits its full max_output_tokens. Failed/expired jobs with no usage record are billed as zero here."),
    }


# ------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------
def _layout(title: str, xtitle: str, ytitle: str, legend_title: str | None = "Requested output tokens", height: int = 440) -> dict[str, Any]:
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS, showline=True,
                title_font=dict(size=12, color=INK_SECONDARY), tickfont=dict(size=11, color=INK_SECONDARY))
    return dict(
        template="none",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK),
        xaxis=dict(title=dict(text=xtitle), **axis), yaxis=dict(title=dict(text=ytitle), **axis),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=dict(text=legend_title) if legend_title else None,
                    font=dict(size=11, color=INK_SECONDARY)),
        margin=dict(l=64, r=24, t=96, b=64), height=height, hovermode="closest",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
    )


def _empty(fig: go.Figure, text: str = "No data available for this figure.") -> go.Figure:
    if not fig.data:
        fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=13, color=INK_MUTED))
    return fig


def _ecdf_figure(frame: pd.DataFrame, levels: list[int], col: str, colors: dict[int, str], symbols: dict[int, str],
                 title: str, xtitle: str) -> go.Figure:
    fig = go.Figure(layout=_layout(title, xtitle, "Fraction of valid jobs at or below x"))
    for lvl in levels:
        x = frame.loc[(frame["_req"] == float(lvl)) & frame["_valid"], col].to_numpy(dtype=float)
        x = np.sort(x[~np.isnan(x)])
        if not x.size:
            continue
        y = np.arange(1, x.size + 1) / x.size
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines" if x.size >= 3 else "lines+markers", name=_label(lvl),
            line=dict(color=colors[lvl], width=2, shape="hv"),
            marker=dict(size=8, symbol=symbols[lvl], color=colors[lvl], line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x:,.0f} s → %{y:.1%}<extra>" + _label(lvl) + "</extra>"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return _empty(fig)


def _figures(frame: pd.DataFrame, ds: dict[str, Any], levels: list[int], launch: dict[str, Any]) -> tuple[dict[str, go.Figure], dict[str, str]]:
    colors, symbols = _group_colors(levels), _group_symbols(levels)
    labels = [_label(l) for l in levels]
    groups = {g["requested_output_tokens"]: g for g in ds["groups"]}
    figs: dict[str, go.Figure] = {}
    caps: dict[str, str] = {}
    valid = frame[frame["_valid"]]

    # 1 ECDF of turnaround
    figs["ecdf_turnaround"] = _ecdf_figure(frame, levels, "_ta", colors, symbols,
                                           "Turnaround time by requested output tokens (ECDF)", "Turnaround: completed_at − created_at (s)")
    caps["ecdf_turnaround"] = ("Empirical cumulative distribution of batch turnaround for each requested-token level, valid jobs only. "
                               "A curve further to the right means slower jobs; the vertical spread at a given x is the difference in completion share.")

    # 2 violin + strip
    fig = go.Figure(layout=_layout("Turnaround distribution per group with individual jobs", "Requested output tokens", "Turnaround (s)", legend_title=None))
    for lvl in levels:
        y = valid.loc[valid["_req"] == float(lvl), "_ta"].to_numpy(dtype=float)
        y = y[~np.isnan(y)]
        if not y.size:
            continue
        fig.add_trace(go.Violin(
            y=y, x=[_label(lvl)] * y.size, name=_label(lvl), showlegend=False, points="all", jitter=0.35, pointpos=0,
            line=dict(color=colors[lvl], width=2), fillcolor=_rgba(colors[lvl], 0.15), spanmode="hard",
            marker=dict(size=5, color=colors[lvl], opacity=0.55, symbol=symbols[lvl], line=dict(color=SURFACE, width=1)),
            box=dict(visible=True, width=0.18, line=dict(color=INK_SECONDARY, width=1)), meanline=dict(visible=False),
            hoveron="points+kde", hovertemplate="%{y:,.0f} s<extra>" + _label(lvl) + "</extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    figs["violin_turnaround"] = _empty(fig)
    caps["violin_turnaround"] = ("Kernel-density outline (violin) with the inner box (median, quartiles) and every valid observation as a jittered point. "
                                 "Use it to see multimodality and outliers that summary quantiles hide.")

    # 3 median / p95 with CI vs tokens (log x)
    fig = go.Figure(layout=_layout("Median and p95 turnaround vs requested output tokens (95% bootstrap CI)", "Requested output tokens (log scale)", "Turnaround (s)", legend_title="Statistic"))
    for key, name, color, symbol in (("median", "Median", CATEGORICAL[0], "circle"), ("p95", "p95", CATEGORICAL[1], "square")):
        xs = [lvl for lvl in levels if groups[lvl]["turnaround_n"] > 0]
        ys = np.array([groups[l][f"turnaround_{key}"] for l in xs], dtype=float)
        lo = np.array([groups[l][f"turnaround_{key}_ci_low"] for l in xs], dtype=float)
        hi = np.array([groups[l][f"turnaround_{key}_ci_high"] for l in xs], dtype=float)
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=name, line=dict(color=color, width=2),
            marker=dict(size=9, symbol=symbol, color=color, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=hi - ys, arrayminus=ys - lo, color=color, thickness=1.5, width=6),
            customdata=np.column_stack([lo, hi]),
            hovertemplate="%{x:,} tokens: " + name + " %{y:,.0f} s (CI %{customdata[0]:,.0f}–%{customdata[1]:,.0f})<extra></extra>"))
    fig.update_xaxes(type="log", tickvals=levels, ticktext=[f"{l:,}" for l in levels])
    figs["median_p95_vs_tokens"] = _empty(fig)
    caps["median_p95_vs_tokens"] = ("Median and 95th percentile of turnaround per level with percentile-bootstrap 95% confidence intervals. "
                                    "The token axis is logarithmic; a flat line means the requested length does not move that quantile.")

    # 4 queue-time ECDF
    figs["queue_time"] = _ecdf_figure(frame, levels, "_queue", colors, symbols, "Queue time by group (ECDF)", "Queue: in_progress_at − created_at (s)")
    caps["queue_time"] = ("Time each batch waited in the validating/queued state before the server marked it in_progress. "
                          "Queueing is independent of the request payload, so the curves should coincide unless the scheduler treats levels differently.")

    # 5 active-time ECDF
    figs["active_time"] = _ecdf_figure(frame, levels, "_active", colors, symbols, "Active time by group (ECDF)", "Active: completed_at − in_progress_at (s)")
    caps["active_time"] = ("Time from in_progress to completion, which contains the model generation itself. This is where the requested output length "
                           "is expected to matter; compare with the queue-time panel to attribute the total turnaround.")

    # 6 actual tokens vs turnaround scatter
    fig = go.Figure(layout=_layout("Turnaround vs actual output tokens", "Actual output tokens (log scale)", "Turnaround (s)"))
    for lvl in levels:
        sub = valid[valid["_req"] == float(lvl)]
        if not len(sub):
            continue
        fig.add_trace(go.Scatter(
            x=sub["_actual"], y=sub["_ta"], mode="markers", name=_label(lvl),
            marker=dict(size=7, color=colors[lvl], opacity=0.7, symbol=symbols[lvl], line=dict(color=SURFACE, width=1)),
            hovertemplate="%{x:,} tokens, %{y:,.0f} s<extra>" + _label(lvl) + "</extra>"))
    fig.update_xaxes(type="log")
    figs["actual_tokens_vs_turnaround"] = _empty(fig)
    caps["actual_tokens_vs_turnaround"] = ("Each point is one valid job, placed by the output tokens the model actually produced (reasoning tokens included). "
                                           "Points to the left of their level's limit are early stops; the colour/symbol keeps the requested level visible.")

    # 7 completion-rate curve since launch
    ref = launch.get("reference_epoch")
    fig = go.Figure(layout=_layout("Share of submitted jobs completed vs time since launch", "Minutes since launch start", "Completed / submitted"))
    if ref is not None:
        def _curve(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            t = sub["_completed_offset"].to_numpy(dtype=float)
            t = np.sort(t[~np.isnan(t)]) / 60.0
            denom = max(len(sub), 1)
            return np.concatenate([[0.0], t]), np.concatenate([[0.0], np.arange(1, t.size + 1) / denom])
        for lvl in levels:
            sub = frame[frame["_req"] == float(lvl)]
            if not len(sub):
                continue
            x, y = _curve(sub)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=_label(lvl), line=dict(color=colors[lvl], width=2, shape="hv"),
                                     hovertemplate="%{x:,.1f} min → %{y:.1%}<extra>" + _label(lvl) + "</extra>"))
        if len(frame):
            x, y = _curve(frame)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="All groups", line=dict(color=INK_SECONDARY, width=2.5, shape="hv"),
                                     hovertemplate="%{x:,.1f} min → %{y:.1%}<extra>All groups</extra>"))
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    figs["completion_curve"] = _empty(fig, "No launch reference time or completed jobs; the completion curve cannot be drawn.")
    caps["completion_curve"] = ("Cumulative fraction of every submitted job (denominator includes creation failures and failed jobs) that had completed "
                                "by each minute after the launch wave started. Curves that never reach 100% show jobs that failed, expired or were never created.")

    # 8 created_at offsets by group
    fig = go.Figure(layout=_layout("Server created_at offset from launch start, by group", "Seconds after launch start", "Requested output tokens", legend_title=None, height=380))
    for lvl in levels:
        x = frame.loc[frame["_req"] == float(lvl), "_created_offset"].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if not x.size:
            continue
        fig.add_trace(go.Box(
            x=x, y=[_label(lvl)] * x.size, name=_label(lvl), orientation="h", showlegend=False, boxpoints="all", jitter=0.5, pointpos=0,
            marker=dict(size=5, color=colors[lvl], opacity=0.55, symbol=symbols[lvl], line=dict(color=SURFACE, width=1)),
            line=dict(color=colors[lvl], width=1.5), fillcolor=_rgba(colors[lvl], 0.15),
            hovertemplate="%{x:,.1f} s<extra>" + _label(lvl) + "</extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    figs["created_at_by_group"] = _empty(fig)
    caps["created_at_by_group"] = ("Server-side creation timestamps (1-second resolution) relative to the launch start, one box per level with each job as a point. "
                                   "Overlapping boxes confirm the randomised wave did not submit one level systematically earlier.")

    # 9 launch position vs turnaround
    fig = go.Figure(layout=_layout("Turnaround vs randomised launch position", "Randomised launch position (submission order)", "Turnaround (s)"))
    for lvl in levels:
        sub = valid[valid["_req"] == float(lvl)]
        if not len(sub):
            continue
        fig.add_trace(go.Scatter(
            x=sub["_pos"], y=sub["_ta"], mode="markers", name=_label(lvl),
            marker=dict(size=7, color=colors[lvl], opacity=0.7, symbol=symbols[lvl], line=dict(color=SURFACE, width=1)),
            hovertemplate="position %{x:,}, %{y:,.0f} s<extra>" + _label(lvl) + "</extra>"))
    figs["launch_position_vs_turnaround"] = _empty(fig)
    caps["launch_position_vs_turnaround"] = ("Turnaround against the position a job held in the shuffled submission order. A trend here would mean the wave's "
                                             "own duration or server load during the wave leaked into the measurement; the model reports the slope.")

    # 10 outcomes grouped bar
    fig = go.Figure(layout=_layout("Failures, expirations, creation failures and early stops by group", "Requested output tokens", "Jobs", legend_title="Outcome"))
    series = [("failed", "Failed"), ("expired", "Expired"), ("cancelled", "Cancelled"), ("creation_failed", "Creation failed"), ("early_stops", "Early stop")]
    any_bar = False
    for i, (key, name) in enumerate(series):
        counts = [groups[l][key] for l in levels]
        denom = [groups[l]["early_stop_known"] if key == "early_stops" else groups[l]["submitted"] for l in levels]
        rates = [c / d if d else float("nan") for c, d in zip(counts, denom)]
        if not any(counts):
            continue
        any_bar = True
        fig.add_trace(go.Bar(
            x=labels, y=counts, name=name, marker=dict(color=CATEGORICAL[i % len(CATEGORICAL)], line=dict(color=SURFACE, width=2)),
            customdata=rates, hovertemplate=name + ": %{y} (%{customdata:.1%})<extra>%{x}</extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    figs["outcomes_by_group"] = _empty(fig, "No failures, expirations, creation failures or early stops in any group.") if not any_bar else fig
    caps["outcomes_by_group"] = ("Counts of non-successful outcomes per level: batch failed/expired/cancelled, creation call failed (no batch_id), "
                                 "and jobs where the model stopped before its limit (early stop, out of valid jobs). Hover shows the share of the group.")
    return figs, caps


# ------------------------------------------------------------------------------------------------------------
# main entry points
# ------------------------------------------------------------------------------------------------------------
def analyze_frame(df: pd.DataFrame, cfg: ExperimentConfig, meta: dict[str, Any], bootstrap: int = 2000,
                  seed: int = 20260910) -> dict[str, Any]:
    """Pure analysis of an observation frame (no I/O). Returns groups, bias_flags, model_fit, cost, datasets, launch, figures."""
    meta = dict(meta or {})
    n_boot = max(0, int(bootstrap))
    d = _prepare(df)
    warnings: list[str] = []

    phase = d["phase"].astype(object)
    n_prod = int((phase == "prod").sum())
    if n_prod > 0:
        main_phases = ["prod"]
        main_mask = (phase == "prod").to_numpy()
    else:
        present = sorted(str(p) for p in d["phase"].dropna().unique().tolist())
        main_phases = present
        main_mask = np.ones(len(d), dtype=bool)
        if len(d):
            warnings.append(f"No production (phase='prod') observations found; statistics below describe phases {present} only. "
                            "This is not the intent-to-measure dataset.")
        else:
            warnings.append("The observation table is empty; nothing to analyse yet.")

    ref, ref_source = _reference_epoch(meta, d[main_mask])
    created = d["_created"].to_numpy(dtype=float)
    offset = d["_offset"].to_numpy(dtype=float)
    if ref is not None:
        d["_created_offset"] = np.where(np.isnan(offset), created - ref, offset)
        d["_completed_offset"] = d["_completed"].to_numpy(dtype=float) - ref
    else:
        d["_created_offset"] = offset
        d["_completed_offset"] = np.nan
    offset_source = "server_created_at_offset_seconds" if ref_source == "launch_started_at" else (f"created_at − {ref_source}" if ref_source else None)

    main = d[main_mask]
    repl = d[phase.to_numpy() == "replacement"]
    pilot = d[phase.to_numpy() == "pilot"]
    prod = d[phase.to_numpy() == "prod"]
    levels = sorted({int(l) for l in cfg.output_token_levels} | _levels_in(main))

    datasets: dict[str, Any] = {}
    datasets["main"] = _dataset(main, "main", "Intent-to-measure (all production jobs)", main_phases, levels,
                                "Every production job, including creation failures and failed/expired batches, so denominators equal the planned runs per group.",
                                n_boot, seed, 0)
    if len(repl):
        datasets["replacement"] = _dataset(repl, "replacement", "Replacement jobs only", ["replacement"], sorted(_levels_in(repl)),
                                           "Labelled replacement batches created by `recover` for production creation calls that failed.", n_boot, seed, 1)
        wr = pd.concat([prod[prod["_valid"]], repl[repl["_valid"]]], ignore_index=True)
        datasets["with_replacements"] = _dataset(wr, "with_replacements", "Production + replacements (valid rows only)", ["prod", "replacement"],
                                                 sorted({int(l) for l in cfg.output_token_levels} | _levels_in(wr)),
                                                 "Valid production rows plus valid replacement rows. Reported separately from the main dataset; never merged.",
                                                 n_boot, seed, 2)
    if len(pilot):
        datasets["pilot"] = _dataset(pilot, "pilot", "Pilot jobs (excluded from production statistics)", ["pilot"], sorted(_levels_in(pilot)),
                                     "Pilot batches used to validate the pipeline and document the API floor; excluded from every production statistic.",
                                     min(n_boot, 200), seed, 3)

    launch = _launch_summary(meta, main, levels, ref, ref_source)
    bias = _bias_checks(main, datasets["main"]["groups"], levels, cfg, launch)
    model = _fit_models(main[main["_valid"]], n_boot, seed, offset_source)
    cost = _cost_statement(d, cfg, meta)
    figs, caps = _figures(main, datasets["main"], levels, launch)
    if len(figs) != len(FIGURE_ORDER):
        raise RuntimeError("figure set incomplete")  # programming error guard

    experiment = {
        "experiment_id": cfg.experiment_id, "model": cfg.model, "endpoint": cfg.endpoint, "completion_window": cfg.completion_window,
        "sdk_version": cfg.sdk_version, "output_token_levels": list(cfg.output_token_levels),
        "original_requested_levels": list(cfg.original_requested_levels), "pilot_levels": list(cfg.pilot_levels),
        "api_min_max_output_tokens": cfg.api_min_max_output_tokens, "runs_per_group": cfg.runs_per_group, "seed": cfg.seed,
        "prompt": cfg.prompt, "reasoning_effort": cfg.reasoning_effort, "file_mode": cfg.file_mode, "notes": list(cfg.notes),
        "config_created_at": cfg.created_at, "bootstrap_resamples": n_boot, "bootstrap_seed": int(seed),
        "levels_dropped": [l for l in cfg.original_requested_levels if l not in cfg.output_token_levels],
        "levels_added": [l for l in cfg.output_token_levels if l not in cfg.original_requested_levels],
    }
    return {
        "generated_at": iso_now(), "experiment": experiment, "warnings": warnings, "levels": levels,
        "groups": datasets["main"]["groups"], "datasets": datasets, "bias_flags": bias, "model_fit": model, "cost": cost,
        "launch": launch, "figures": figs, "figure_captions": caps, "figure_order": list(FIGURE_ORDER),
        "n_rows": int(len(d)), "n_prod": n_prod,
    }


def write_outputs(result: dict[str, Any], cfg: ExperimentConfig) -> dict[str, str]:
    """Write summary_stats.csv/json, bias_checks.json, model_fit.json and cost_statement.json to cfg.processed_dir."""
    os.makedirs(cfg.processed_dir, exist_ok=True)
    paths: dict[str, str] = {}
    rows = []
    for ds_name, ds in result["datasets"].items():
        for g in ds["groups"]:
            rows.append({"dataset": ds_name, **g})
    csv_path = os.path.join(cfg.processed_dir, "summary_stats.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    paths["summary_stats_csv"] = csv_path

    payloads = {
        "summary_stats.json": {"generated_at": result["generated_at"], "experiment": result["experiment"], "warnings": result["warnings"],
                               "levels": result["levels"], "datasets": result["datasets"], "launch": result["launch"]},
        "bias_checks.json": {"generated_at": result["generated_at"], "checks": result["bias_flags"],
                             "n_flagged": sum(1 for c in result["bias_flags"] if c["flagged"])},
        "model_fit.json": result["model_fit"],
        "cost_statement.json": result["cost"],
    }
    for name, payload in payloads.items():
        path = os.path.join(cfg.processed_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_clean(payload), f, indent=2)
            f.write("\n")
        paths[name.replace(".", "_")] = path
    return paths


# ---- template filters ----
def _f_fmt(v: Any, nd: int = 1) -> str:
    if _is_missing(v):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return "∞"
        return f"{float(v):,.{int(nd)}f}"
    return str(v)


def _f_pct(v: Any, nd: int = 1) -> str:
    return "–" if _is_missing(v) else f"{100 * float(v):.{int(nd)}f}%"


def _f_usd(v: Any, nd: int = 4) -> str:
    return "–" if _is_missing(v) else f"${float(v):,.{int(nd)}f}"


def _f_ci(lo: Any, hi: Any, nd: int = 0) -> str:
    if _is_missing(lo) or _is_missing(hi):
        return "–"
    return f"[{float(lo):,.{int(nd)}f}, {float(hi):,.{int(nd)}f}]"


def _f_sci(v: Any) -> str:
    return "–" if _is_missing(v) else f"{float(v):.3g}"


def _f_duration(v: Any) -> str:
    if _is_missing(v):
        return "–"
    s = float(v)
    if s < 120:
        return f"{s:,.1f} s"
    if s < 7200:
        return f"{s / 60:,.1f} min"
    return f"{s / 3600:,.2f} h"


def render_report(result: dict[str, Any], cfg: ExperimentConfig, out_path: str) -> str:
    """Render the self-contained HTML report (plotly.js embedded once, no CDN) and return its path."""
    env = Environment(loader=FileSystemLoader(_PKG_DIR), autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters.update(fmt=_f_fmt, pct=_f_pct, usd=_f_usd, ci=_f_ci, sci=_f_sci, duration=_f_duration)
    tpl = env.get_template(TEMPLATE_NAME)
    figure_html = {}
    for name in result.get("figure_order", FIGURE_ORDER):
        fig = result["figures"].get(name)
        if fig is None:
            continue
        figure_html[name] = fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="440px",
                                        config={"responsive": True, "displaylogo": False})
    html = tpl.render(r=result, cfg=cfg.to_dict(), figure_html=figure_html, captions=result.get("figure_captions", {}),
                      figure_order=[n for n in result.get("figure_order", FIGURE_ORDER) if n in figure_html],
                      plotlyjs_placeholder=PLOTLYJS_PLACEHOLDER, generated_at=iso_now(),
                      n_flagged=sum(1 for c in result["bias_flags"] if c["flagged"]))
    html = redact_text(html)  # belt and braces: the bundle is injected afterwards so it is never altered
    html = html.replace(PLOTLYJS_PLACEHOLDER, plotly.offline.get_plotlyjs(), 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def analyze(args: Any) -> dict[str, Any]:
    """CLI entry: build observations from the store, analyse, write outputs and the HTML report."""
    cfg = ExperimentConfig.load(getattr(args, "config", None) or "config/experiment.json")
    bootstrap = int(getattr(args, "bootstrap", None) or 2000)
    out_path = getattr(args, "out", None) or "reports/experiment_report.html"
    store = Store(cfg.db_path)
    try:
        df = build_observations(store, cfg)
        obs_csv, obs_pq = write_observations(df, cfg)
        meta: dict[str, Any] = {k: store.get_meta(k) for k in ("launch_started_at", "launch_finished_at", "pilot_status")}
        for name in ("launch_summary", "prepare_summary", "pilot_summary"):
            meta[name] = _load_json(os.path.join(cfg.processed_dir, f"{name}.json"))
        result = analyze_frame(df, cfg, meta, bootstrap=bootstrap, seed=cfg.seed)
        paths = write_outputs(result, cfg)
        report = render_report(result, cfg, out_path)
    finally:
        store.close()
    return _clean({
        "report": report, "observations_csv": obs_csv, "observations_parquet": obs_pq, "outputs": paths,
        "rows": int(len(df)), "prod_rows": result["n_prod"], "warnings": result["warnings"],
        "groups": {g["label"]: {"submitted": g["submitted"], "valid": g["valid"], "median_turnaround_s": g["turnaround_median"]}
                   for g in result["groups"]},
        "flags_raised": [c["name"] for c in result["bias_flags"] if c["flagged"]],
        "model_errors": {k: v.get("error") for k, v in result["model_fit"].items() if isinstance(v, dict) and v.get("error")},
        "production_cost_usd": result["cost"]["production_usd"], "generated_at": result["generated_at"],
    })
