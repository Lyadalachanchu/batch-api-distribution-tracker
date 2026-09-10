"""Cost projection (worst case: every request emits its full max_output_tokens) and actual-usage costing."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .config import Pricing


class CostCeilingExceeded(RuntimeError):
    pass


@dataclass
class CostProjection:
    jobs: int
    max_output_tokens: int
    est_input_tokens: int
    output_cost_usd: float
    input_cost_usd: float
    total_cost_usd: float
    safety_margin: float
    total_with_margin_usd: float
    ceiling_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


def project_cost(level_counts: dict[int, int], input_tokens_per_request: int, pricing: Pricing,
                 ceiling_usd: float, safety_margin: float = 0.10, already_spent_usd: float = 0.0) -> CostProjection:
    jobs = sum(level_counts.values())
    max_out = sum(int(n) * int(c) for n, c in level_counts.items())
    est_in = jobs * int(input_tokens_per_request)
    out_cost = max_out / 1e6 * pricing.output_per_1m_usd
    in_cost = est_in / 1e6 * pricing.input_per_1m_usd
    total = out_cost + in_cost + already_spent_usd
    return CostProjection(
        jobs=jobs,
        max_output_tokens=max_out,
        est_input_tokens=est_in,
        output_cost_usd=round(out_cost, 6),
        input_cost_usd=round(in_cost, 6),
        total_cost_usd=round(total, 6),
        safety_margin=safety_margin,
        total_with_margin_usd=round(total * (1.0 + safety_margin), 6),
        ceiling_usd=ceiling_usd,
    )


def enforce_ceiling(projection: CostProjection) -> None:
    if projection.total_with_margin_usd > projection.ceiling_usd:
        raise CostCeilingExceeded(
            f"projected cost ${projection.total_with_margin_usd:.4f} (incl. {projection.safety_margin:.0%} margin) "
            f"exceeds ceiling ${projection.ceiling_usd:.2f}; pass --max-cost-usd to raise it explicitly"
        )


def actual_cost_usd(input_tokens: int | None, cached_input_tokens: int | None, output_tokens: int | None,
                    pricing: Pricing) -> float | None:
    if input_tokens is None and output_tokens is None:
        return None
    it = int(input_tokens or 0)
    ct = min(int(cached_input_tokens or 0), it)
    ot = int(output_tokens or 0)
    return round(((it - ct) * pricing.input_per_1m_usd + ct * pricing.cached_input_per_1m_usd + ot * pricing.output_per_1m_usd) / 1e6, 8)
