"""Single-asset strategies + regime detection.

Each strategy is a small class that:
  * `prepare(df)`  -> returns a copy of the OHLCV frame with the
                      indicator columns it needs, computed once.
  * `evaluate(df, i)` -> returns a `TradePlan` IF the strategy would
                      open a position at bar `i`, using ONLY data up to
                      and including bar `i` (no look-ahead), else None.

The same `evaluate` is used both live (i = last bar) and inside the
backtester (i swept across history). Entry/stop/target are concrete
prices derived from ATR and market structure — never guesses.

Strategy rationale (grounded in the methodology in README):
  * TrendBreakout   — time-series momentum / trend following. Donchian
                      breakout, filtered by the 200-period MA so we only
                      trade breakouts in the direction of the major trend.
  * MeanReversion   — short-horizon overreaction reversal. Bollinger /
                      z-score stretch + Connors RSI-2, taken only WITH
                      the long-term trend (buy dips up, fade rips down).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from . import smc
from .models import Side


# ---------------------------------------------------------------------
@dataclass
class TradePlan:
    """A strategy's concrete trade idea at one bar (pre-sizing)."""

    side: Side
    entry: float
    stop: float
    target: float
    atr: float
    rationale: str
    regime: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # surface the structural (SMC) levels the stop/target were anchored to
        # right in the rationale, so the published signal SHOWS it was used.
        note = self.meta.get("structure_note") if self.meta else None
        if note and note not in self.rationale:
            self.rationale = f"{self.rationale} {note}"

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.target - self.entry)

    @property
    def rr_ratio(self) -> float:
        r = self.risk_per_unit
        return self.reward_per_unit / r if r > 0 else 0.0

    def valid(self) -> bool:
        if not all(math.isfinite(x) for x in (self.entry, self.stop, self.target, self.atr)):
            return False
        if self.risk_per_unit <= 0 or self.reward_per_unit <= 0:
            return False
        if self.side is Side.LONG:
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop


def _finite(*vals: float) -> bool:
    return all(v is not None and math.isfinite(v) for v in vals)


def _struct_window(df: pd.DataFrame, i: int, scfg: dict):
    width = int(scfg.get("swing_width", 3))
    lookback = int(scfg.get("lookback", 120))
    hi = i - width                      # newest bar whose pivot is confirmed by i
    lo = max(0, i - lookback)
    return lo, hi


def _refine_target(df: pd.DataFrame, i: int, side: Side, entry: float,
                   target: float, atr: float, risk: float, scfg: dict) -> float:
    """Pull the target in to just before a significant structural level.

    If a confirmed support (for shorts) or resistance (for longs) — a swing
    pivot, FVG edge, or order block — sits between entry and the mechanical
    target, move the target to just *before* it so the order fills before
    price can reverse off that level. Falls back to the mechanical target if
    nothing qualifies or the refined reward:risk drops below `min_rr`.

    Returns (target, anchored_level_or_None).
    """
    if not scfg or not scfg.get("enabled", True) or atr <= 0 or risk <= 0:
        return target, None
    buf = float(scfg.get("buffer_atr", 0.25)) * atr
    min_rr = float(scfg.get("min_rr", 0.8))
    lo, hi = _struct_window(df, i, scfg)
    if hi <= lo:
        return target, None

    if side is Side.LONG:
        cand = [v for v in smc.collect_levels(df, lo, hi, smc.RES_COLS)
                if entry < v < target]
        if not cand:
            return target, None
        lvl = min(cand)                 # lowest resistance is hit first
        new_t = lvl - buf
        if new_t <= entry or (new_t - entry) / risk < min_rr:
            return target, None
        return new_t, lvl
    else:
        cand = [v for v in smc.collect_levels(df, lo, hi, smc.SUP_COLS)
                if target < v < entry]
        if not cand:
            return target, None
        lvl = max(cand)                 # highest support is hit first
        new_t = lvl + buf
        if new_t >= entry or (entry - new_t) / risk < min_rr:
            return target, None
        return new_t, lvl


def _refine_stop(df: pd.DataFrame, i: int, side: Side, entry: float,
                 atr: float, base_stop: float, scfg: dict) -> float:
    """Place the stop just beyond the nearest protective structural level.

    For a short the stop goes above the nearest resistance (swing high / FVG
    / order block) the trade is invalidated by; for a long, below the
    nearest support. The structural risk is capped to [stop_min_atr,
    stop_max_atr] × ATR — if the level is too far or too close, the blind
    ATR stop is kept. This avoids being wicked out just inside a level while
    never letting the risk blow out.

    Returns (stop, anchored_level_or_None).
    """
    if (not scfg or not scfg.get("enabled", True)
            or not scfg.get("structural_stop", True) or atr <= 0):
        return base_stop, None
    buf = float(scfg.get("buffer_atr", 0.25)) * atr
    max_risk = float(scfg.get("stop_max_atr", 4.0)) * atr
    min_risk = float(scfg.get("stop_min_atr", 0.5)) * atr
    lo, hi = _struct_window(df, i, scfg)
    if hi <= lo:
        return base_stop, None

    if side is Side.SHORT:
        cand = [v for v in smc.collect_levels(df, lo, hi, smc.RES_COLS) if v > entry]
        if not cand:
            return base_stop, None
        lvl = min(cand)                 # nearest resistance above
        new_stop = lvl + buf            # just above it
        risk = new_stop - entry
    else:
        cand = [v for v in smc.collect_levels(df, lo, hi, smc.SUP_COLS) if v < entry]
        if not cand:
            return base_stop, None
        lvl = max(cand)                 # nearest support below
        new_stop = lvl - buf            # just below it
        risk = entry - new_stop
    if not (min_risk <= risk <= max_risk):
        return base_stop, None
    return new_stop, lvl


def _stop_and_target(df: pd.DataFrame, i: int, side: Side, entry: float, atr: float,
                     atr_stop_mult: float, min_target_R: float, scfg: dict):
    """Structure-aware (stop, target, note): structural stop -> R-multiple
    target from that risk -> structural target. Falls back to ATR/R-multiple.

    `note` is a short human string naming the structural levels the stop and
    target were anchored to (empty when nothing structural applied), so the
    published signal can SHOW that SMC levels were used.
    """
    scfg = scfg or {}
    base_stop = entry - atr_stop_mult * atr if side is Side.LONG else entry + atr_stop_mult * atr
    stop, stop_lvl = _refine_stop(df, i, side, entry, atr, base_stop, scfg)
    risk = abs(entry - stop)
    if risk <= 0:
        stop, stop_lvl, risk = base_stop, None, abs(entry - base_stop)
    target = entry + min_target_R * risk if side is Side.LONG else entry - min_target_R * risk
    # HARD CAP on how far the mechanical target may sit from entry: on a daily
    # timeframe a wide ATR can make a 2R target absurdly distant (the "Long BTC
    # to 180k" problem). Cap it so every target is realistically reachable.
    max_t_atr = float(scfg.get("max_target_atr", 0) or 0) * atr
    if max_t_atr > 0:
        target = (min(target, entry + max_t_atr) if side is Side.LONG
                  else max(target, entry - max_t_atr))
    target, tgt_lvl = _refine_target(df, i, side, entry, target, atr, risk, scfg)
    note = _structure_note(side, stop_lvl, tgt_lvl)
    return stop, target, note


def _structure_note(side: Side, stop_lvl, tgt_lvl) -> str:
    """Human note describing which SMC levels anchored the stop / target."""
    if stop_lvl is None and tgt_lvl is None:
        return ""
    parts = []
    if stop_lvl is not None:
        where = "support" if side is Side.LONG else "resistance"
        parts.append(f"SL anchored beyond {where} {stop_lvl:.4f}")
    if tgt_lvl is not None:
        where = "resistance" if side is Side.LONG else "support"
        parts.append(f"TP set just before {where} {tgt_lvl:.4f}")
    return "🧱 Structure: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------
# Regime detection (ADX): trending vs ranging vs neutral
# ---------------------------------------------------------------------
def detect_regime(df: pd.DataFrame, cfg: dict) -> str:
    period = int(cfg.get("adx_period", 14))
    trend_th = float(cfg.get("adx_trend_threshold", 25))
    range_th = float(cfg.get("adx_range_threshold", 20))
    adx_series = ind.adx(df, period)
    if adx_series.dropna().empty:
        return "neutral"
    val = float(adx_series.iloc[-1])
    if not math.isfinite(val):
        return "neutral"
    if val >= trend_th:
        return "trending"
    if val < range_th:
        return "ranging"
    return "neutral"


# ---------------------------------------------------------------------
class TrendBreakout:
    name = "trend_breakout"
    preferred_regime = "trending"

    def __init__(self, cfg: dict, structure: dict | None = None):
        self.donchian_entry = int(cfg.get("donchian_entry", 20))
        self.donchian_exit = int(cfg.get("donchian_exit", 10))
        self.ma_fast = int(cfg.get("ma_fast", 50))
        self.ma_slow = int(cfg.get("ma_slow", 200))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 2.5))
        self.min_target_R = float(cfg.get("min_target_R", 2.0))
        self.structure = structure or {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_slow"] = ind.sma(d["close"], self.ma_slow)
        d["dc_up"], d["dc_low"] = ind.donchian(d, self.donchian_entry)
        d["dc_exit_up"], d["dc_exit_low"] = ind.donchian(d, self.donchian_exit)
        smc.add_levels(d, int(self.structure.get("swing_width", 3)))
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        row = df.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        atr_v = float(row["atr"])
        ma_slow = float(row["ma_slow"])
        dc_up, dc_low = float(row["dc_up"]), float(row["dc_low"])
        if not _finite(close, high, low, atr_v, ma_slow, dc_up, dc_low) or atr_v <= 0:
            return None

        # Long: break above prior N-bar high, in an uptrend
        if high > dc_up and close > ma_slow:
            entry = close
            stop, target, snote = _stop_and_target(df, i, Side.LONG, entry, atr_v,
                                            self.atr_stop_mult, self.min_target_R, self.structure)
            return TradePlan(
                Side.LONG, entry, stop, target, atr_v,
                rationale=(f"{self.donchian_entry}-bar breakout above {dc_up:.4f} "
                           f"with price over {self.ma_slow}MA (uptrend)."),
                meta={"dc_level": dc_up, "structure_note": snote},
            )

        # Short: break below prior N-bar low, in a downtrend
        if low < dc_low and close < ma_slow:
            entry = close
            stop, target, snote = _stop_and_target(df, i, Side.SHORT, entry, atr_v,
                                            self.atr_stop_mult, self.min_target_R, self.structure)
            return TradePlan(
                Side.SHORT, entry, stop, target, atr_v,
                rationale=(f"{self.donchian_entry}-bar breakdown below {dc_low:.4f} "
                           f"with price under {self.ma_slow}MA (downtrend)."),
                meta={"dc_level": dc_low, "structure_note": snote},
            )
        return None


# ---------------------------------------------------------------------
class MeanReversion:
    name = "mean_reversion"
    preferred_regime = "ranging"

    def __init__(self, cfg: dict):
        self.bb_period = int(cfg.get("bb_period", 20))
        self.bb_std = float(cfg.get("bb_std", 2.0))
        self.zscore_entry = float(cfg.get("zscore_entry", 2.0))
        self.rsi_period = int(cfg.get("rsi_period", 2))
        self.rsi_buy_below = float(cfg.get("rsi_buy_below", 10))
        self.rsi_sell_above = float(cfg.get("rsi_sell_above", 90))
        self.ma_trend = int(cfg.get("ma_trend", 200))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 3.0))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_trend"] = ind.sma(d["close"], self.ma_trend)
        mid, up, low = ind.bollinger(d["close"], self.bb_period, self.bb_std)
        d["bb_mid"], d["bb_up"], d["bb_low"] = mid, up, low
        d["zscore"] = ind.rolling_zscore(d["close"], self.bb_period)
        d["rsi"] = ind.rsi(d["close"], self.rsi_period)
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        row = df.iloc[i]
        close = float(row["close"])
        atr_v = float(row["atr"])
        ma_trend = float(row["ma_trend"])
        bb_mid = float(row["bb_mid"])
        bb_up, bb_low = float(row["bb_up"]), float(row["bb_low"])
        z = float(row["zscore"])
        rsi_v = float(row["rsi"])
        if not _finite(close, atr_v, ma_trend, bb_mid, bb_up, bb_low, z, rsi_v) or atr_v <= 0:
            return None

        risk = self.atr_stop_mult * atr_v

        # Long: oversold dip below lower band, but major trend still up
        if (close < bb_low and z <= -self.zscore_entry and rsi_v < self.rsi_buy_below
                and close > ma_trend):
            entry = close
            stop = entry - risk
            target = bb_mid  # revert to the mean
            plan = TradePlan(
                Side.LONG, entry, stop, target, atr_v,
                rationale=(f"Oversold: z={z:.2f}, RSI{self.rsi_period}={rsi_v:.1f} below "
                           f"lower band, price above {self.ma_trend}MA. Target = mean."),
                meta={"z": z, "rsi": rsi_v},
            )
            return plan if plan.valid() else None

        # Short: overbought spike above upper band, but major trend still down
        if (close > bb_up and z >= self.zscore_entry and rsi_v > self.rsi_sell_above
                and close < ma_trend):
            entry = close
            stop = entry + risk
            target = bb_mid
            plan = TradePlan(
                Side.SHORT, entry, stop, target, atr_v,
                rationale=(f"Overbought: z={z:.2f}, RSI{self.rsi_period}={rsi_v:.1f} above "
                           f"upper band, price below {self.ma_trend}MA. Target = mean."),
                meta={"z": z, "rsi": rsi_v},
            )
            return plan if plan.valid() else None
        return None


# ---------------------------------------------------------------------
class MacdTrend:
    """MACD signal-line crossover, filtered by the long-term trend.

    A genuine, widely-used momentum trigger: go long when the MACD line
    crosses ABOVE its signal line while price is above the 200-MA (mirror
    for shorts). Entry at the crossover close; ATR stop; 2R target.
    """

    name = "macd_trend"
    preferred_regime = "trending"

    def __init__(self, cfg: dict, structure: dict | None = None):
        self.fast = int(cfg.get("fast", 12))
        self.slow = int(cfg.get("slow", 26))
        self.signal = int(cfg.get("signal", 9))
        self.ma_slow = int(cfg.get("ma_slow", 200))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 2.5))
        self.min_target_R = float(cfg.get("min_target_R", 2.0))
        self.structure = structure or {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_slow"] = ind.sma(d["close"], self.ma_slow)
        d["macd"], d["macd_sig"], d["macd_hist"] = ind.macd(
            d["close"], self.fast, self.slow, self.signal
        )
        smc.add_levels(d, int(self.structure.get("swing_width", 3)))
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        if i < 1:
            return None
        row, prev = df.iloc[i], df.iloc[i - 1]
        close = float(row["close"])
        atr_v = float(row["atr"])
        ma_slow = float(row["ma_slow"])
        hist, hist_prev = float(row["macd_hist"]), float(prev["macd_hist"])
        if not _finite(close, atr_v, ma_slow, hist, hist_prev) or atr_v <= 0:
            return None

        # bullish crossover (hist turns positive) in an uptrend
        if hist > 0 and hist_prev <= 0 and close > ma_slow:
            entry = close
            stop, target, snote = _stop_and_target(df, i, Side.LONG, entry, atr_v,
                                            self.atr_stop_mult, self.min_target_R, self.structure)
            return TradePlan(
                Side.LONG, entry, stop, target, atr_v,
                rationale=(f"MACD bullish cross (hist {hist_prev:.4f}→{hist:.4f}) "
                           f"with price above {self.ma_slow}MA."),
                meta={"structure_note": snote},
            )
        # bearish crossover in a downtrend
        if hist < 0 and hist_prev >= 0 and close < ma_slow:
            entry = close
            stop, target, snote = _stop_and_target(df, i, Side.SHORT, entry, atr_v,
                                            self.atr_stop_mult, self.min_target_R, self.structure)
            return TradePlan(
                Side.SHORT, entry, stop, target, atr_v,
                rationale=(f"MACD bearish cross (hist {hist_prev:.4f}→{hist:.4f}) "
                           f"with price below {self.ma_slow}MA."),
                meta={"structure_note": snote},
            )
        return None


# ---------------------------------------------------------------------
class DualThrust:
    """Dual Thrust breakout (Michael Chalek).

    Range = max(HH-LC, HC-LL) over the prior N bars. BuyLine = Open +
    K1*Range, SellLine = Open - K2*Range. Break above BuyLine -> long;
    break below SellLine -> short. A 200-MA filter is applied to only
    take breakouts aligned with the major trend (reduces whipsaw).
    """

    name = "dual_thrust"
    preferred_regime = "trending"

    def __init__(self, cfg: dict, structure: dict | None = None):
        self.n = int(cfg.get("range_bars", 4))
        self.k1 = float(cfg.get("k1", 0.5))
        self.k2 = float(cfg.get("k2", 0.5))
        self.ma_slow = int(cfg.get("ma_slow", 200))
        self.use_trend_filter = bool(cfg.get("trend_filter", True))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 2.0))
        self.min_target_R = float(cfg.get("min_target_R", 2.0))
        self.structure = structure or {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_slow"] = ind.sma(d["close"], self.ma_slow)
        rng = ind.dual_thrust_range(d, self.n)
        d["buyline"] = d["open"] + self.k1 * rng
        d["sellline"] = d["open"] - self.k2 * rng
        smc.add_levels(d, int(self.structure.get("swing_width", 3)))
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        row = df.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        atr_v = float(row["atr"])
        ma_slow = float(row["ma_slow"])
        buyline, sellline = float(row["buyline"]), float(row["sellline"])
        if not _finite(close, high, low, atr_v, ma_slow, buyline, sellline) or atr_v <= 0:
            return None
        trend_ok_long = (close > ma_slow) if self.use_trend_filter else True
        trend_ok_short = (close < ma_slow) if self.use_trend_filter else True

        if high > buyline and trend_ok_long:
            entry = close
            stop, target, snote = _stop_and_target(df, i, Side.LONG, entry, atr_v,
                                            self.atr_stop_mult, self.min_target_R, self.structure)
            return TradePlan(
                Side.LONG, entry, stop, target, atr_v,
                rationale=(f"Dual Thrust break above BuyLine {buyline:.4f} "
                           f"(K1={self.k1}, {self.n}-bar range)."),
                meta={"buyline": buyline, "structure_note": snote},
            )
        if low < sellline and trend_ok_short:
            entry = close
            stop, target, snote = _stop_and_target(df, i, Side.SHORT, entry, atr_v,
                                            self.atr_stop_mult, self.min_target_R, self.structure)
            return TradePlan(
                Side.SHORT, entry, stop, target, atr_v,
                rationale=(f"Dual Thrust break below SellLine {sellline:.4f} "
                           f"(K2={self.k2}, {self.n}-bar range)."),
                meta={"sellline": sellline, "structure_note": snote},
            )
        return None


# ---------------------------------------------------------------------
class SqueezeBreakout:
    """TTM Squeeze breakout (John Carter).

    When Bollinger Bands contract INSIDE the Keltner channels, volatility
    is compressed (a 'squeeze'). When the squeeze releases, price tends to
    expand sharply. We enter in the direction of momentum on the release,
    with a tight ATR stop (vol was low) and a larger 3R target (vol
    expansion). A 200-MA filter keeps entries trend-aligned.
    """

    name = "squeeze_breakout"
    preferred_regime = "trending"

    def __init__(self, cfg: dict, structure: dict | None = None):
        self.bb_period = int(cfg.get("bb_period", 20))
        self.bb_std = float(cfg.get("bb_std", 2.0))
        self.kc_ema = int(cfg.get("kc_ema", 20))
        self.kc_atr = int(cfg.get("kc_atr", 10))
        self.kc_mult = float(cfg.get("kc_mult", 1.5))
        self.ma_slow = int(cfg.get("ma_slow", 200))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 1.5))
        self.min_target_R = float(cfg.get("min_target_R", 3.0))
        self.structure = structure or {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_slow"] = ind.sma(d["close"], self.ma_slow)
        mid, up, low = ind.bollinger(d["close"], self.bb_period, self.bb_std)
        kc_mid, kc_up, kc_low = ind.keltner(d, self.kc_ema, self.kc_atr, self.kc_mult)
        d["bb_up"], d["bb_low"], d["bb_mid"] = up, low, mid
        d["kc_up"], d["kc_low"] = kc_up, kc_low
        # squeeze ON when BB sits entirely inside KC (low volatility)
        d["squeeze_on"] = (up < kc_up) & (low > kc_low)
        smc.add_levels(d, int(self.structure.get("swing_width", 3)))
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        if i < 1:
            return None
        row, prev = df.iloc[i], df.iloc[i - 1]
        close = float(row["close"])
        atr_v = float(row["atr"])
        ma_slow = float(row["ma_slow"])
        bb_mid = float(row["bb_mid"])
        sq_now, sq_prev = bool(row["squeeze_on"]), bool(prev["squeeze_on"])
        if not _finite(close, atr_v, ma_slow, bb_mid) or atr_v <= 0:
            return None

        # fire only on the bar the squeeze RELEASES (on -> off)
        if sq_prev and not sq_now:
            if close > bb_mid and close > ma_slow:    # upward release in uptrend
                entry = close
                stop, target, snote = _stop_and_target(df, i, Side.LONG, entry, atr_v,
                                                self.atr_stop_mult, self.min_target_R, self.structure)
                return TradePlan(
                    Side.LONG, entry, stop, target, atr_v,
                    rationale="TTM squeeze released upward (vol expansion) above the mean and 200MA.",
                    meta={"structure_note": snote},
                )
            if close < bb_mid and close < ma_slow:    # downward release in downtrend
                entry = close
                stop, target, snote = _stop_and_target(df, i, Side.SHORT, entry, atr_v,
                                                self.atr_stop_mult, self.min_target_R, self.structure)
                return TradePlan(
                    Side.SHORT, entry, stop, target, atr_v,
                    rationale="TTM squeeze released downward (vol expansion) below the mean and 200MA.",
                    meta={"structure_note": snote},
                )
        return None


# ---------------------------------------------------------------------
_STRATEGY_REGISTRY = {
    "trend": ("trend", TrendBreakout),
    "mean_reversion": ("mean_reversion", MeanReversion),
    "macd": ("macd", MacdTrend),
    "dual_thrust": ("dual_thrust", DualThrust),
    "squeeze": ("squeeze", SqueezeBreakout),
}


def build_strategies(settings) -> list:
    """Instantiate every enabled single-asset strategy from settings."""
    scfg = settings.section("structure")
    # mean reversion targets the moving-average mean, so structure-refining
    # its target doesn't apply; the momentum/breakout strategies use it.
    structure_users = {"trend", "macd", "dual_thrust", "squeeze"}
    out = []
    for key, (section, cls) in _STRATEGY_REGISTRY.items():
        cfg = settings.section(section)
        if not cfg.get("enabled", True):
            continue
        if key in structure_users:
            out.append(cls(cfg, structure=scfg))
        else:
            out.append(cls(cfg))
    return out
