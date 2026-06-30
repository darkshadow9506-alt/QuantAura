"""Portfolio-level risk aggregation across a batch of signals.

This is the "position management / risk budget" layer from the
methodology: a single trade can be sized correctly yet a *basket* of
simultaneous signals can still over-concentrate risk. After a scan we
summarise total risk-at-stop, gross/net exposure, and flag concentration
so the user sees the whole book, not just one idea.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Side, Signal


@dataclass
class PortfolioSummary:
    n: int = 0
    longs: int = 0
    shorts: int = 0
    total_risk: float = 0.0          # account currency at risk if every stop hits
    total_risk_pct: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    by_class_risk: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    allocations: list = field(default_factory=list)   # risk-parity weights


@dataclass
class Allocation:
    """One position's risk-parity allocation within the book."""
    symbol: str
    strategy: str
    side: str
    weight_pct: float        # position notional as % of equity
    risk_pct: float          # risk-at-stop as % of equity
    units: float
    notional: float
    risk_amount: float
    capped: bool = False     # notional was clipped by the single-name cap


def optimize_risk_parity(
    signals: list[Signal], equity: float,
    risk_budget_pct: float = 6.0, max_name_pct: float = 25.0,
) -> list[Allocation]:
    """Allocate a *fixed total risk budget* equally across signals (risk parity).

    Each tradeable signal is given the SAME risk-at-stop = budget / N, so every
    position contributes equal risk to the book. Because units = target_risk /
    (entry-stop), a more volatile name (wider stop) automatically gets fewer
    units for the same dollar risk — i.e. inverse-volatility / equal-risk-
    contribution sizing. The total never exceeds the budget, however many
    signals fire. Forecast-only items (e.g. Iran shorts) are excluded. A single
    name's notional is clipped to `max_name_pct` of equity so one low-vol idea
    can't dominate the book.
    """
    tradeable = [s for s in signals
                 if not getattr(s, "forecast_only", False)
                 and s.risk_per_unit > 0 and s.entry > 0]
    if not tradeable or equity <= 0:
        return []
    budget = equity * (risk_budget_pct / 100.0)
    target_risk_each = budget / len(tradeable)
    cap_notional = equity * (max_name_pct / 100.0)

    allocations: list[Allocation] = []
    for s in tradeable:
        units = target_risk_each / s.risk_per_unit
        notional = units * s.entry
        risk_amount = target_risk_each
        capped = False
        if notional > cap_notional:
            # clip to the single-name cap (risk drops below the equal target)
            scale = cap_notional / notional
            units *= scale
            notional = cap_notional
            risk_amount *= scale
            capped = True
        allocations.append(Allocation(
            symbol=s.symbol, strategy=s.strategy, side=s.side.value,
            weight_pct=round(notional / equity * 100.0, 2),
            risk_pct=round(risk_amount / equity * 100.0, 2),
            units=round(units, 6), notional=round(notional, 2),
            risk_amount=round(risk_amount, 2), capped=capped))
    return allocations


def summarize(signals: list[Signal], equity: float,
              max_risk_pct: float = 6.0, max_per_class: int = 5,
              max_name_pct: float = 25.0) -> PortfolioSummary:
    s = PortfolioSummary()
    if not signals:
        return s
    # risk-parity allocation: split the total risk budget equally across signals
    s.allocations = optimize_risk_parity(signals, equity, max_risk_pct, max_name_pct)
    s.n = len(signals)
    long_notional = short_notional = 0.0
    class_count: dict[str, int] = {}
    for sig in signals:
        if sig.side is Side.LONG:
            s.longs += 1
            long_notional += sig.position_notional
        else:
            s.shorts += 1
            short_notional += sig.position_notional
        s.total_risk += sig.risk_amount
        cls = sig.asset_class.value
        s.by_class_risk[cls] = s.by_class_risk.get(cls, 0.0) + sig.risk_amount
        class_count[cls] = class_count.get(cls, 0) + 1

    s.gross_exposure = long_notional + short_notional
    s.net_exposure = long_notional - short_notional
    s.total_risk_pct = (s.total_risk / equity * 100.0) if equity > 0 else 0.0

    if s.total_risk_pct > max_risk_pct:
        s.warnings.append(
            f"Total risk-at-stop is {s.total_risk_pct:.1f}% of equity "
            f"(> {max_risk_pct:.0f}% budget) — consider taking fewer / smaller positions.")
    for cls, cnt in class_count.items():
        if cnt > max_per_class:
            s.warnings.append(
                f"{cnt} {cls} positions (> {max_per_class}) — concentrated in one asset class.")
    # directional skew
    if s.n >= 4 and (s.longs == 0 or s.shorts == 0):
        side = "long" if s.shorts == 0 else "short"
        s.warnings.append(f"All {s.n} signals are {side} — the book is one-directional.")
    return s


def format_summary(summary: PortfolioSummary, equity: float) -> str:
    if summary.n == 0:
        return ""
    lines = ["", "📁 *Portfolio risk (if you took all signals)*",
             f"Signals: {summary.n}  ({summary.longs} long / {summary.shorts} short)",
             f"Risk at stop: ${summary.total_risk:,.0f}  ({summary.total_risk_pct:.1f}% of equity)",
             f"Gross exposure: ${summary.gross_exposure:,.0f}  |  "
             f"Net: ${summary.net_exposure:,.0f}"]
    if summary.by_class_risk:
        parts = ", ".join(f"{k} ${v:,.0f}" for k, v in summary.by_class_risk.items())
        lines.append(f"Risk by class: {parts}")
    for w in summary.warnings:
        lines.append(f"⚠️ {w}")

    if summary.allocations:
        lines.append("")
        lines.append("⚖️ *Risk-parity allocation* (equal risk per position, "
                     "total within budget):")
        for a in summary.allocations:
            icon = "🟢" if a.side == "LONG" else "🔴"
            note = " (capped)" if a.capped else ""
            lines.append(
                f"{icon} *{a.symbol}* `{a.strategy}` — "
                f"*{a.weight_pct:.1f}% of wallet* (~${a.notional:,.0f}), "
                f"risk {a.risk_pct:.2f}%{note}")
        tot_w = sum(a.weight_pct for a in summary.allocations)
        tot_r = sum(a.risk_pct for a in summary.allocations)
        lines.append(f"Σ wallet deployed {tot_w:.1f}% | total risk {tot_r:.2f}%")
    return "\n".join(lines)
