"""Signal engine — turns market data into gated, sized Signal objects.

For each symbol:
  1. fetch OHLCV
  2. detect regime (ADX)
  3. run every enabled strategy whose preferred regime matches (or
     'neutral' -> run all); take the latest-bar TradePlan if any
  4. backtest THAT strategy on THAT symbol's history
  5. validate it out-of-sample (walk-forward) and with a Monte Carlo
     bootstrap; only publish if it clears the full signal_gate
  6. size the position (fixed-fractional capped, half-Kelly), attach the
     Monte Carlo win-probability, and blend everything into a confidence
  7. boost confidence when multiple strategies agree (confluence)

This is the mechanism that keeps signals honest: nothing is published
without a measured, out-of-sample, Monte-Carlo-robust edge on the very
instrument it targets.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from . import data as data_mod
from . import factor as factor_mod
from . import ml as ml_mod
from . import montecarlo as mc_mod
from . import pairs as pairs_mod
from . import risk as risk_mod
from .backtest import backtest_strategy, out_of_sample
from .config import Settings
from .indicators import atr as atr_ind
from .models import (
    AssetClass,
    BacktestStats,
    MonteCarloStats,
    PairSignal,
    Side,
    Signal,
)
from .strategies import build_strategies, detect_regime


# ---------------------------------------------------------------------
def _passes_gate(
    stats: BacktestStats, oos: BacktestStats, mc: MonteCarloStats, gate: dict
) -> bool:
    """Base in-sample edge + out-of-sample persistence + MC robustness."""
    base = (
        stats.trades >= int(gate.get("min_backtest_trades", 15))
        and stats.win_rate >= float(gate.get("min_win_rate", 0.40))
        and stats.profit_factor >= float(gate.get("min_profit_factor", 1.15))
        and stats.sharpe >= float(gate.get("min_sharpe", 0.3))
    )
    if not base:
        return False
    # Monte Carlo: the edge must be probably-profitable forward
    if mc.prob_profitable < float(gate.get("min_prob_profitable", 0.60)):
        return False
    # walk-forward: if we have enough out-of-sample trades, the edge must
    # have survived the holdout (positive expectancy). Too few -> skip check.
    if oos.trades >= int(gate.get("oos_min_trades", 4)):
        if oos.expectancy_R <= float(gate.get("oos_min_expectancy_R", 0.0)):
            return False
    return True


def _kelly_b(win_rate: float, profit_factor: float) -> float:
    """Recover the payoff ratio b = avg_win/avg_loss from W and PF."""
    if win_rate <= 0 or win_rate >= 1 or not math.isfinite(profit_factor):
        return 0.0
    return profit_factor * (1.0 - win_rate) / win_rate


def _recent_drift(df: Optional[pd.DataFrame], n: int = 14) -> float:
    """Average per-bar price change over the last n bars (momentum proxy)."""
    if df is None or "close" not in df or len(df) < n + 1:
        return 0.0
    diffs = df["close"].diff().dropna().tail(n)
    return float(diffs.mean()) if len(diffs) else 0.0


def _assess(
    side: Side, entry: float, stop: float, target: float, atr: float,
    stats: BacktestStats, drift: float, gate: dict,
):
    """Compute (oos, montecarlo) for a candidate signal."""
    oos = out_of_sample(stats, float(gate.get("oos_split", 0.7)))
    mc = mc_mod.assess(
        side=side, entry=entry, stop=stop, target=target,
        returns_R=stats.returns_R, atr=atr, drift=drift,
        ruin_R=float(gate.get("ruin_R", 10.0)),
        max_bars=int(gate.get("mc_max_bars", 60)),
    )
    return oos, mc


def _confidence(stats: BacktestStats, oos: BacktestStats, mc: MonteCarloStats,
                confluence: int = 1) -> float:
    bt = risk_mod.confidence_from_backtest(stats.win_rate, stats.profit_factor, stats.sharpe)
    return risk_mod.combined_confidence(
        backtest_conf=bt,
        prob_profitable=mc.prob_profitable,
        mc_win_prob=mc.win_prob,
        baseline_win_prob=mc.baseline_win_prob,
        oos_expectancy_R=oos.expectancy_R,
        confluence=confluence,
    )


_SizingZero = risk_mod.SizingResult(0.0, 0.0, 0.0, 0.0)


def _publish_filter(signals: list, settings: Settings, publish_only: bool) -> list:
    """Drop low-confidence signals when publishing.

    The signal_gate (win-rate / profit-factor / Monte-Carlo) decides whether a
    strategy has a *measured* edge; this final filter additionally requires the
    *blended* confidence (backtest + MC + out-of-sample + confluence) to clear
    ``min_confidence`` so only strong setups reach subscribers. Forecast-only
    items (e.g. Iran downside views) are informational and never filtered out.
    Call this AFTER confluence, since confluence raises confidence.
    """
    if not publish_only:
        return signals
    floor = float(settings.signal_gate.get("min_confidence", 0.0))
    if floor <= 0:
        return signals
    return [s for s in signals
            if getattr(s, "forecast_only", False) or s.confidence >= floor]


def _attach_plan(sig: Signal, settings: Settings) -> Signal:
    """Fill the explicit-plan fields: wallet %, risk %, risk-free (+1R) price."""
    eq = float(settings.account_equity or 0.0)
    if eq > 0:
        sig.equity_pct = round(sig.position_notional / eq * 100.0, 2)
        sig.risk_pct = round(sig.risk_amount / eq * 100.0, 2)
    be_R = float(settings.section("manage").get("breakeven_R", 1.0))
    one_R = sig.risk_per_unit
    if sig.side is Side.LONG:
        sig.risk_free_at = round(sig.entry + be_R * one_R, 6)
    else:
        sig.risk_free_at = round(sig.entry - be_R * one_R, 6)
    return sig


def _sized(settings: Settings, entry: float, risk_per_unit: float, stats: BacktestStats):
    risk_cfg = settings.risk
    b = _kelly_b(stats.win_rate, stats.profit_factor)
    return risk_mod.size_position(
        equity=settings.account_equity, entry=entry, risk_per_unit=risk_per_unit,
        win_rate=stats.win_rate, avg_win_R=b, avg_loss_R=1.0,
        risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 1.0)),
        kelly_fraction_mult=float(risk_cfg.get("kelly_fraction", 0.5)),
        max_kelly_pct=float(risk_cfg.get("max_kelly_pct", 5.0)),
    )


def _build_signal(
    *,
    symbol: str,
    asset_class: AssetClass,
    strategy_name: str,
    plan,
    stats: BacktestStats,
    regime: str,
    settings: Settings,
    passed_gate: bool,
    drift: float = 0.0,
    management: str = "",
    forecast_only: bool = False,
) -> Signal:
    gate = settings.signal_gate
    oos, mc = _assess(plan.side, plan.entry, plan.stop, plan.target, plan.atr,
                      stats, drift, gate)
    # a forecast is not a position you enter -> no size / risk
    sizing = (_SizingZero if forecast_only
              else _sized(settings, plan.entry, plan.risk_per_unit, stats))
    sig = Signal(
        symbol=symbol,
        asset_class=asset_class,
        strategy=strategy_name,
        side=plan.side,
        entry=round(plan.entry, 6),
        stop=round(plan.stop, 6),
        target=round(plan.target, 6),
        risk_per_unit=round(plan.risk_per_unit, 6),
        reward_per_unit=round(plan.reward_per_unit, 6),
        rr_ratio=round(plan.rr_ratio, 3),
        position_units=round(sizing.units, 6),
        position_notional=round(sizing.notional, 2),
        risk_amount=round(sizing.risk_amount, 2),
        atr=round(plan.atr, 6),
        regime=regime,
        rationale=plan.rationale,
        management=management,
        backtest=stats,
        oos=oos,
        montecarlo=mc,
        confidence=_confidence(stats, oos, mc),
        passed_gate=passed_gate,
        forecast_only=forecast_only,
        timeframe=settings.data.get("timeframe", "1d"),
        price_at_signal=round(plan.entry, 6),
    )
    return _attach_plan(sig, settings)


# ---------------------------------------------------------------------
def scan_symbol(
    symbol: str,
    asset_class: AssetClass,
    settings: Settings,
    publish_only: bool = True,
) -> list[Signal]:
    """Return signals for one symbol. If publish_only, only gated ones."""
    d = settings.data
    df = data_mod.get_ohlcv(
        symbol,
        asset_class,
        timeframe=d.get("timeframe", "1d"),
        lookback=int(d.get("lookback_bars", 400)),
        cache_minutes=int(d.get("cache_minutes", 30)),
        ccxt_exchange=settings.ccxt_exchange,
    )
    if df is None or len(df) < 60:
        return []

    regime = detect_regime(df, settings.regime)
    gate = settings.signal_gate
    drift = _recent_drift(df, int(settings.regime.get("adx_period", 14)))
    out: list[Signal] = []

    for strat in build_strategies(settings):
        # regime alignment: run a strategy when the market suits it, or
        # when the regime is neutral (let the backtest gate decide).
        if regime != "neutral" and strat.preferred_regime != regime:
            continue
        prepared = strat.prepare(df)
        plan = strat.evaluate(prepared, len(prepared) - 1)
        if plan is None or not plan.valid():
            continue
        # Iranian *rial-denominated local* shorts: retail usually can't short
        # them, so by default publish those as a non-tradeable "downside
        # forecast". Global USD markets that merely come from tgju (e.g. the
        # gold ounce = XAU/USD) are normally shortable and stay tradeable.
        forecast_only = False
        if (asset_class is AssetClass.IRAN and plan.side is Side.SHORT
                and not data_mod.is_global_tradeable(symbol)):
            mode = settings.section("iran").get("short_mode", "forecast")
            if mode == "off":
                continue
            forecast_only = (mode != "trade")
        # trailing (Chandelier) exit is used for trend/breakout strategies
        # when enabled, so the backtest reflects how the trade is managed.
        risk_cfg = settings.risk
        trail = 0.0
        management = ""
        if (risk_cfg.get("use_trailing_stop", False)
                and strat.preferred_regime == "trending"):
            trail = float(risk_cfg.get("trail_atr_mult", 3.0))
            management = (f"Trail the stop at (extreme − {trail}×ATR) after entry; "
                          f"the target is a minimum — let winners run on the trail.")
        stats, _ = backtest_strategy(strat, df, trail_atr_mult=trail)
        oos, mc = _assess(plan.side, plan.entry, plan.stop, plan.target, plan.atr,
                          stats, drift, gate)
        passed = _passes_gate(stats, oos, mc, gate)
        if publish_only and not passed:
            continue
        out.append(
            _build_signal(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=strat.name,
                plan=plan,
                stats=stats,
                regime=regime,
                settings=settings,
                passed_gate=passed,
                drift=drift,
                management=management,
                forecast_only=forecast_only,
            )
        )

    # confluence: when several strategies agree on the same direction,
    # raise confidence (independent confirmations of the same idea).
    _apply_confluence(out)
    return _publish_filter(out, settings, publish_only)


def _apply_confluence(signals: list[Signal]) -> None:
    by_side: dict[Side, list[Signal]] = {}
    for s in signals:
        by_side.setdefault(s.side, []).append(s)
    for side, group in by_side.items():
        n = len(group)
        if n <= 1:
            continue
        for s in group:
            s.confluence = n
            s.confidence = _confidence(s.backtest, s.oos, s.montecarlo, confluence=n)


# ---------------------------------------------------------------------
def scan_pairs(settings: Settings, publish_only: bool = True) -> list[PairSignal]:
    pcfg = settings.pairs
    if not pcfg.get("enabled", True):
        return []
    gate = settings.signal_gate
    d = settings.data
    out: list[PairSignal] = []

    # pairs candidates are equities here
    cache: dict[str, pd.DataFrame] = {}

    def _load(sym: str) -> Optional[pd.DataFrame]:
        if sym in cache:
            return cache[sym]
        try:
            df = data_mod.get_ohlcv(
                sym, AssetClass.STOCK,
                timeframe=d.get("timeframe", "1d"),
                lookback=int(d.get("lookback_bars", 400)),
                cache_minutes=int(d.get("cache_minutes", 30)),
            )
        except Exception:
            df = None
        cache[sym] = df
        return df

    for cand in pcfg.get("candidates", []):
        if not isinstance(cand, (list, tuple)) or len(cand) != 2:
            continue
        sym_a, sym_b = cand
        da, db = _load(sym_a), _load(sym_b)
        if da is None or db is None or len(da) < 60 or len(db) < 60:
            continue

        atr_a = float(atr_ind(da, 14).iloc[-1]) if len(da) > 20 else 0.0
        plan = pairs_mod.evaluate_pair(
            sym_a, sym_b, da["close"], db["close"], pcfg, atr_a=atr_a
        )
        if plan is None:
            continue
        stats = pairs_mod.backtest_pair(da["close"], db["close"], pcfg)
        oos = out_of_sample(stats, float(gate.get("oos_split", 0.7)))
        # pairs win-probability uses the spread mean-reversion (AR(1)) model,
        # not single-leg drift — the edge here is convergence of the spread.
        mc = mc_mod.assess_pairs(
            returns_R=stats.returns_R, z_now=plan.spread_z,
            z_exit=plan.z_exit, z_stop=plan.z_stop,
            phi=plan.ar1_phi, sigma_eps=plan.ar1_sigma,
            ruin_R=float(gate.get("ruin_R", 10.0)),
            max_bars=int(gate.get("mc_max_bars", 60)),
        )
        passed = _passes_gate(stats, oos, mc, gate)
        if publish_only and not passed:
            continue

        sizing = _sized(settings, plan.entry_a, plan.risk_per_unit, stats)
        leg_b_side = Side.SHORT if plan.side_a is Side.LONG else Side.LONG
        out.append(_attach_plan(
            PairSignal(
                symbol=plan.symbol_a,
                asset_class=AssetClass.STOCK,
                strategy="pairs_statarb",
                side=plan.side_a,
                entry=round(plan.entry_a, 6),
                stop=round(plan.stop_a, 6),
                target=round(plan.target_a, 6),
                risk_per_unit=round(plan.risk_per_unit, 6),
                reward_per_unit=round(plan.reward_per_unit, 6),
                rr_ratio=round(plan.rr_ratio, 3),
                position_units=round(sizing.units, 6),
                position_notional=round(sizing.notional, 2),
                risk_amount=round(sizing.risk_amount, 2),
                atr=round(plan.atr_a, 6),
                regime="stat-arb",
                rationale=plan.rationale,
                backtest=stats,
                oos=oos,
                montecarlo=mc,
                confidence=_confidence(stats, oos, mc),
                passed_gate=passed,
                timeframe=d.get("timeframe", "1d"),
                price_at_signal=round(plan.entry_a, 6),
                pair_symbol=plan.symbol_b,
                hedge_ratio=round(plan.hedge_ratio, 6),
                spread_z=round(plan.spread_z, 3),
                leg_b_side=leg_b_side.value,
            ), settings))
    return _publish_filter(out, settings, publish_only)


# ---------------------------------------------------------------------
def scan_factor(settings: Settings, publish_only: bool = True) -> list[Signal]:
    """Cross-sectional momentum factor, scanned per asset class."""
    fcfg = settings.section("factor_momentum")
    if not fcfg.get("enabled", True):
        return []
    d = settings.data
    class_map = {
        "stocks": AssetClass.STOCK,
        "forex": AssetClass.FOREX,
        "crypto": AssetClass.CRYPTO,
    }
    out: list[Signal] = []

    for cls, ac in class_map.items():
        symbols = settings.universe.get(cls, [])
        if len(symbols) < int(fcfg.get("top_n", 2)) + int(fcfg.get("bottom_n", 2)):
            continue
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                frames[sym] = data_mod.get_ohlcv(
                    sym, ac, timeframe=d.get("timeframe", "1d"),
                    lookback=int(d.get("lookback_bars", 1250)),
                    cache_minutes=int(d.get("cache_minutes", 30)),
                    ccxt_exchange=settings.ccxt_exchange,
                )
            except Exception:
                continue
        frames = {k: v for k, v in frames.items() if v is not None and len(v) > 60}
        if len(frames) < int(fcfg.get("top_n", 2)) + int(fcfg.get("bottom_n", 2)):
            continue

        panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
        stats = factor_mod.backtest_cross_sectional(panel, fcfg)
        passed = factor_mod.passes_factor_gate(stats, fcfg)
        if publish_only and not passed:
            continue

        legs = factor_mod.rank_live(frames, fcfg, settings.section("structure"))
        gate = settings.signal_gate
        oos = out_of_sample(stats, float(gate.get("oos_split", 0.7)))
        for leg in legs:
            sizing = _sized(settings, leg.entry, leg.risk_per_unit, stats)
            drift = _recent_drift(frames.get(leg.symbol), 14)
            mc = mc_mod.assess(
                side=leg.side, entry=leg.entry, stop=leg.stop, target=leg.target,
                returns_R=stats.returns_R, atr=leg.atr, drift=drift,
                ruin_R=float(gate.get("ruin_R", 10.0)),
                max_bars=int(gate.get("mc_max_bars", 60)),
            )
            rr = (abs(leg.target - leg.entry) / leg.risk_per_unit
                  if leg.risk_per_unit > 0 else 0.0)
            out.append(_attach_plan(Signal(
                symbol=leg.symbol, asset_class=ac, strategy="factor_momentum",
                side=leg.side, entry=round(leg.entry, 6), stop=round(leg.stop, 6),
                target=round(leg.target, 6), risk_per_unit=round(leg.risk_per_unit, 6),
                reward_per_unit=round(abs(leg.target - leg.entry), 6), rr_ratio=round(rr, 3),
                position_units=round(sizing.units, 6),
                position_notional=round(sizing.notional, 2),
                risk_amount=round(sizing.risk_amount, 2), atr=round(leg.atr, 6),
                regime="cross-sectional",
                rationale=(f"Ranked #{leg.rank} by 6-1 momentum "
                           f"({leg.score*100:+.1f}%) among {cls} — "
                           f"{'strongest→long' if leg.side is Side.LONG else 'weakest→short'}."
                           + (f" {leg.structure_note}" if leg.structure_note else "")),
                backtest=stats, oos=oos, montecarlo=mc,
                confidence=_confidence(stats, oos, mc), passed_gate=passed,
                timeframe=d.get("timeframe", "1d"), price_at_signal=round(leg.entry, 6),
            ), settings))
    return _publish_filter(out, settings, publish_only)


# ---------------------------------------------------------------------
def review_positions(settings: Settings, store) -> list:
    """For each OPEN journaled signal, compute live management advice.

    Returns list of (signal_row_dict, ManagementReview).
    """
    from datetime import datetime

    from . import journal as journal_mod
    from . import manage as manage_mod
    from .indicators import macd as _macd
    from .indicators import sma as _sma

    d = settings.data
    mcfg = settings.section("manage")
    out = []
    for row in store.open_signals():
        try:
            ac = AssetClass(row["asset_class"])
            created = datetime.fromisoformat(row["created_at"])
        except (ValueError, KeyError):
            continue
        try:
            df = data_mod.get_ohlcv(
                row["symbol"], ac, timeframe=d.get("timeframe", "1d"),
                lookback=int(d.get("lookback_bars", 1250)),
                cache_minutes=int(d.get("cache_minutes", 30)),
                ccxt_exchange=settings.ccxt_exchange)
        except Exception:
            continue
        if df is None or len(df) < 60:
            continue
        atr_v = float(atr_ind(df, 14).iloc[-1])
        ma_trend = float(_sma(df["close"], 200).iloc[-1])
        _, _, hist = _macd(df["close"])
        macd_hist = float(hist.iloc[-1])
        current = float(df["close"].iloc[-1])
        after = journal_mod.bars_after(df, created)
        hi = float(after["high"].max()) if not after.empty else float(df["high"].iloc[-1])
        lo = float(after["low"].min()) if not after.empty else float(df["low"].iloc[-1])
        try:
            side = Side(row["side"])
        except ValueError:
            continue

        # nearest structural level ahead of price in the trade's direction
        next_level = _next_level_ahead(df, side, current, settings.section("structure"))

        rev = manage_mod.review(
            side=side, entry=float(row["entry"]), stop=float(row["stop"]),
            target=float(row["target"]), current=current, atr=atr_v,
            ma_trend=ma_trend, macd_hist=macd_hist, hi_since=hi, lo_since=lo, cfg=mcfg,
            next_level=next_level)
        out.append((row, rev))
    return out


def _next_level_ahead(df, side: Side, current: float, scfg: dict):
    """Nearest support/resistance ahead of price in the trade direction."""
    from . import smc

    sw = int(scfg.get("swing_width", 3))
    lb = int(scfg.get("lookback", 120))
    sdf = df.copy()
    smc.add_levels(sdf, sw)
    n = len(sdf)
    hi = n - 1 - sw
    lo = max(0, n - 1 - lb)
    if hi <= lo:
        return None
    if side is Side.LONG:
        lv = [v for v in smc.collect_levels(sdf, lo, hi, smc.RES_COLS) if v > current]
        return min(lv) if lv else None
    lv = [v for v in smc.collect_levels(sdf, lo, hi, smc.SUP_COLS) if v < current]
    return max(lv) if lv else None


# ---------------------------------------------------------------------
def scan_ml_symbol(
    symbol: str, asset_class: AssetClass, settings: Settings, publish_only: bool = True
) -> list[Signal]:
    """Gradient-boosting (triple-barrier) signal for one symbol."""
    mlcfg = settings.section("ml")
    if not mlcfg.get("enabled", True):
        return []
    d = settings.data
    try:
        df = data_mod.get_ohlcv(
            symbol, asset_class, timeframe=d.get("timeframe", "1d"),
            lookback=int(d.get("lookback_bars", 1250)),
            cache_minutes=int(d.get("cache_minutes", 30)),
            ccxt_exchange=settings.ccxt_exchange,
        )
    except Exception:
        return []
    if df is None or len(df) < int(mlcfg.get("min_train", 250)) + 30:
        return []

    plan = ml_mod.latest_plan(df, mlcfg)
    if plan is None or not plan.valid():
        return []
    stats, _ = ml_mod.backtest_ml(df, mlcfg)
    drift = _recent_drift(df, int(settings.regime.get("adx_period", 14)))
    oos, mc = _assess(plan.side, plan.entry, plan.stop, plan.target, plan.atr,
                      stats, drift, settings.signal_gate)
    passed = _passes_gate(stats, oos, mc, settings.signal_gate)
    if publish_only and not passed:
        return []
    out = [_build_signal(
        symbol=symbol, asset_class=asset_class, strategy_name="ml_gboost",
        plan=plan, stats=stats, regime="ml", settings=settings,
        passed_gate=passed, drift=drift,
    )]
    return _publish_filter(out, settings, publish_only)


def scan_ml(settings: Settings, publish_only: bool = True) -> list[Signal]:
    """Run the ML model across the whole universe (heavier than /scan)."""
    uni = settings.universe
    class_map = {"stocks": AssetClass.STOCK, "forex": AssetClass.FOREX,
                 "crypto": AssetClass.CRYPTO}
    out: list[Signal] = []
    for cls, ac in class_map.items():
        for sym in uni.get(cls, []):
            try:
                out.extend(scan_ml_symbol(sym, ac, settings, publish_only))
            except Exception:
                continue
    out.sort(key=lambda s: s.confidence, reverse=True)
    return out


# ---------------------------------------------------------------------
def scan_universe(
    settings: Settings,
    classes: Optional[list[str]] = None,
    include_pairs: bool = True,
) -> list[Signal]:
    """Scan the whole configured universe. Returns gated signals only."""
    uni = settings.universe
    classes = classes or ["stocks", "forex", "crypto", "iran"]
    class_map = {
        "stocks": AssetClass.STOCK,
        "forex": AssetClass.FOREX,
        "crypto": AssetClass.CRYPTO,
        "iran": AssetClass.IRAN,
    }
    signals: list[Signal] = []
    for cls in classes:
        ac = class_map.get(cls)
        if ac is None:
            continue
        for sym in uni.get(cls, []):
            try:
                signals.extend(scan_symbol(sym, ac, settings, publish_only=True))
            except Exception:
                # one bad symbol must never break a full scan
                continue
    if include_pairs:
        try:
            signals.extend(scan_pairs(settings, publish_only=True))
        except Exception:
            pass
    if include_pairs:  # also run the cross-sectional factor on a full scan
        try:
            signals.extend(scan_factor(settings, publish_only=True))
        except Exception:
            pass

    # best signals first
    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals
