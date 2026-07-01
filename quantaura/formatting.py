"""Human-readable rendering of Signal objects (Telegram / CLI)."""
from __future__ import annotations

from .data import IRAN_NAMES, is_global_tradeable
from .models import AssetClass, PairSignal, Side, Signal

_DISCLAIMER = (
    "⚠️ Educational signal from a backtested quant model. Markets are "
    "probabilistic — no signal is guaranteed. Risk only what you can lose."
)


def _fmt_price(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _bar(conf: float, width: int = 10) -> str:
    filled = int(round(conf * width))
    return "█" * filled + "░" * (width - filled)


def _label(sig: Signal) -> str:
    name = IRAN_NAMES.get(sig.symbol, sig.symbol)
    return f"{sig.symbol} — {name}" if name != sig.symbol else sig.symbol


def _class_label(sig: Signal) -> str:
    """Displayed asset class — global tgju symbols aren't 'iran'."""
    if sig.asset_class is AssetClass.IRAN and is_global_tradeable(sig.symbol):
        return "global"
    return sig.asset_class.value


def _format_forecast(sig: Signal, md: bool) -> str:
    """A directional forecast you can't trade (e.g. an Iranian gold/USD short)."""
    bt = sig.backtest
    lines = [
        f"📉 *Downside forecast — {_label(sig)}*  ({sig.asset_class.value})",
        f"Model: `{sig.strategy}`  |  Regime: {sig.regime}  |  TF: {sig.timeframe}",
        "",
        f"🔻 Expected pullback toward: `{_fmt_price(sig.target)}`",
        f"⛔ Forecast invalidated above: `{_fmt_price(sig.stop)}`",
        f"📍 Current level: `{_fmt_price(sig.entry)}`  |  ATR {_fmt_price(sig.atr)}",
        "",
        "💡 You likely can't short this market. Use it to:",
        f"   • wait and *buy lower* near `{_fmt_price(sig.target)}`, or",
        "   • take profit on holdings / hold off buying for now.",
        "",
        f"📊 Backtest of this downside setup: {bt.trades} trades | "
        f"hit {bt.win_rate*100:.0f}% | exp {bt.expectancy_R:+.2f}R",
    ]
    if sig.oos.trades > 0:
        lines.append(f"🧪 Out-of-sample: {sig.oos.trades} | hit {sig.oos.win_rate*100:.0f}%")
    lines.append(f"🎲 Reaches the level (model): {sig.montecarlo.win_prob*100:.0f}% "
                 f"| P(played out) {sig.montecarlo.prob_profitable*100:.0f}%")
    if sig.confluence > 1:
        lines.append(f"🔗 {sig.confluence} strategies agree on the downside")
    lines.append(f"🔆 Confidence: {_bar(sig.confidence)} {sig.confidence*100:.0f}%")
    lines.append("")
    lines.append("🇮🇷 Tehran free-market price (tgju.org). Heavily policy/news-driven — "
                 "a guide to direction, not a tradeable short.")
    lines.append("")
    lines.append(_DISCLAIMER)
    text = "\n".join(lines)
    return text if md else text.replace("*", "").replace("`", "")


def format_signal(sig: Signal, md: bool = True) -> str:
    if sig.forecast_only:
        return _format_forecast(sig, md)
    side_icon = "🟢 LONG" if sig.side is Side.LONG else "🔴 SHORT"
    bt = sig.backtest
    lines = []
    label = _label(sig)
    title = f"{side_icon}  *{label}*  ({_class_label(sig)})"
    lines.append(title)
    lines.append(f"Strategy: `{sig.strategy}`  |  Regime: {sig.regime}  |  TF: {sig.timeframe}")
    lines.append("")
    lines.append(f"🎯 Entry:  `{_fmt_price(sig.entry)}`")
    lines.append(f"🛑 Stop:   `{_fmt_price(sig.stop)}`")
    lines.append(f"✅ Target: `{_fmt_price(sig.target)}`")
    lines.append(f"⚖️ R:R = {sig.rr_ratio}  |  ATR = {_fmt_price(sig.atr)}")
    lines.append("")

    if isinstance(sig, PairSignal):
        leg_b = sig.leg_b_side or "?"
        lines.append(
            f"🔗 Pair leg: {leg_b} *{sig.pair_symbol}*  "
            f"(hedge β={sig.hedge_ratio}, spread z={sig.spread_z})"
        )
        lines.append("")

    # explicit, precise execution plan
    lines.append("📋 *Plan — exact*")
    if sig.equity_pct > 0:
        lines.append(
            f"• Enter *{sig.equity_pct:.1f}% of wallet* (~${sig.position_notional:,.0f}, "
            f"{_fmt_price(sig.position_units)} units) — risking {sig.risk_pct:.1f}% "
            f"(${sig.risk_amount:,.0f})"
        )
    elif sig.position_units > 0:
        lines.append(
            f"• Size: {_fmt_price(sig.position_units)} units "
            f"(~${sig.position_notional:,.0f}) | risk ${sig.risk_amount:,.0f}"
        )
    lines.append(f"• Enter at: `{_fmt_price(sig.entry)}`   Stop: `{_fmt_price(sig.stop)}`")
    if sig.risk_free_at:
        lines.append(
            f"• Risk-free at `{_fmt_price(sig.risk_free_at)}` (+1R): take ~half profit "
            f"& move stop to entry `{_fmt_price(sig.entry)}`"
        )
    lines.append(
        f"• Take profit: `{_fmt_price(sig.risk_free_at)}` (partial) → "
        f"`{_fmt_price(sig.target)}` (final)"
    )
    lines.append(
        f"📊 Backtest: {bt.trades} trades | win {bt.win_rate*100:.0f}% | "
        f"PF {bt.profit_factor:.2f} | exp {bt.expectancy_R:+.2f}R | "
        f"maxDD {bt.max_drawdown:.1f}R"
    )
    oos = sig.oos
    if oos.trades > 0:
        lines.append(
            f"🧪 Out-of-sample: {oos.trades} trades | win {oos.win_rate*100:.0f}% | "
            f"exp {oos.expectancy_R:+.2f}R"
        )
    mc = sig.montecarlo
    lines.append(
        f"🎲 Win prob (model): {mc.win_prob*100:.0f}% vs {mc.baseline_win_prob*100:.0f}% "
        f"baseline | P(profit) {mc.prob_profitable*100:.0f}% | ruin {mc.risk_of_ruin*100:.0f}%"
    )
    if sig.confluence > 1:
        lines.append(f"🔗 Confluence: {sig.confluence} strategies agree on {sig.side.value}")
    lines.append(f"🔆 Confidence: {_bar(sig.confidence)} {sig.confidence*100:.0f}%")
    if not sig.passed_gate:
        lines.append("❗ Below publish threshold — shown for inspection only.")
    lines.append("")
    # split out the structural (SMC) note so it stands on its own prominent line
    rationale, _, struct = sig.rationale.partition("🧱 Structure:")
    lines.append(f"💡 {rationale.strip()}")
    if struct.strip():
        lines.append(f"🧱 Structure: {struct.strip()}")
    else:
        # no qualifying support/resistance in range → the target is a deliberate
        # reward:risk floor, not a random number. Say so, so it's transparent.
        lines.append(f"📐 Target = {sig.rr_ratio:g}R reward:risk (no structural level "
                     f"in range — a mechanical floor; the stop stays structural).")
    if sig.management:
        lines.append(f"🧭 {sig.management}")
    if sig.asset_class is AssetClass.IRAN:
        if is_global_tradeable(sig.symbol):
            lines.append("🌍 Global USD spot price (via tgju.org) — a normal, "
                         "tradeable market you can go long or short.")
        else:
            lines.append("🇮🇷 Tehran free-market price (tgju.org). Heavily policy/news-driven "
                         "and hard to short — treat as educational only.")
    lines.append("")
    lines.append(_DISCLAIMER)

    text = "\n".join(lines)
    if not md:
        text = text.replace("*", "").replace("`", "")
    return text


def format_management(row: dict, rev) -> str:
    """Render a management review for one open position."""
    side = row.get("side", "")
    icon = "🟢" if side == "LONG" else "🔴"
    flags = []
    if rev.danger:
        flags.append("🚨 DANGER")
    if rev.near_target:
        flags.append("🎯 near target")
    if rev.at_breakeven or rev.trailed:
        flags.append("🔒 risk-free")
    head = f"{icon} *{row.get('symbol','')}* `{row.get('strategy','')}`  ({rev.R_now:+.2f}R)"
    if flags:
        head += "  " + " · ".join(flags)
    lines = [head,
             f"Entry {_fmt_price(float(row['entry']))} | "
             f"SL → *{_fmt_price(rev.recommended_sl)}* | "
             f"💰 Take profit → *{_fmt_price(rev.recommended_tp)}* | "
             f"final target {_fmt_price(float(row['target']))}"]
    for n in rev.notes:
        lines.append(f"• {n}")
    return "\n".join(lines)


def format_scan_summary(signals: list[Signal]) -> str:
    if not signals:
        return ("No signals passed the backtest gate this scan. "
                "That is normal — the model stays flat unless it has a measured edge.")
    head = f"📡 *{len(signals)} signal(s)* passed the gate (best first):\n"
    rows = []
    for s in signals:
        icon = "🟢" if s.side is Side.LONG else "🔴"
        rows.append(
            f"{icon} *{s.symbol}* `{s.strategy}` "
            f"entry {_fmt_price(s.entry)} → tgt {_fmt_price(s.target)} "
            f"(R:R {s.rr_ratio}, conf {s.confidence*100:.0f}%)"
        )
    return head + "\n".join(rows)
