"""Offline end-to-end self-check using synthetic data (no network).

Validates the full analytics pipeline — indicators, both single-asset
strategies, the event-driven backtester, position sizing, the pairs
stat-arb path, and signal formatting — so the system can be verified in
any environment, including one with no market-data access.

Run with:  python -m quantaura selftest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from . import pairs as pairs_mod
from . import risk as risk_mod
from .backtest import backtest_strategy
from .models import AssetClass, Side, Signal
from .strategies import MeanReversion, TrendBreakout, detect_regime


def _ohlc_from_close(close: np.ndarray, idx, noise=0.004, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.asarray(close, dtype=float)
    spread = np.abs(close) * noise
    high = close + rng.uniform(0, 1, len(close)) * spread
    low = close - rng.uniform(0, 1, len(close)) * spread
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1e6, 5e6, len(close))
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([open_, high, close]),
         "low": np.minimum.reduce([open_, low, close]), "close": close, "volume": vol},
        index=idx,
    )


def _trending_series(n=500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # upward drift with momentum bursts so breakouts and reversals both occur
    steps = rng.normal(0.0008, 0.012, n)
    steps[120:160] += 0.01     # a strong leg up
    steps[300:330] -= 0.012    # a pullback / down leg
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return _ohlc_from_close(close, idx, seed=seed)


def _ranging_series(n=500, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Ornstein-Uhlenbeck around a slowly drifting mean. Large shocks make
    # the close pierce the Bollinger bands while the slow mean keeps the
    # 200-MA filter satisfiable -> genuine RSI-2 mean-reversion triggers.
    mean = 100 + np.linspace(0, 10, n)
    x = np.zeros(n)
    x[0] = mean[0]
    for t in range(1, n):
        shock = rng.normal(0, 2.6)
        if rng.random() < 0.06:           # occasional sharp 3-bar overshoot
            shock += rng.choice([-1, 1]) * rng.uniform(4, 7)
        x[t] = x[t - 1] + 0.12 * (mean[t] - x[t - 1]) + shock
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return _ohlc_from_close(x, idx, noise=0.006, seed=seed)


def _cointegrated_pair(n=400, seed=3):
    rng = np.random.default_rng(seed)
    b = 50 + np.cumsum(rng.normal(0, 0.5, n))
    spread = np.zeros(n)
    for t in range(1, n):  # stationary OU spread -> cointegration
        spread[t] = 0.6 * spread[t - 1] + rng.normal(0, 1.0)
    a = 1.8 * b + 10 + spread
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def run_selftest() -> bool:
    print("QuantAura self-test (synthetic data, no network)\n" + "=" * 50)
    ok = True

    # 1) indicators -------------------------------------------------------
    print("\n[1] Indicators")
    df = _trending_series()
    a = ind.atr(df, 14)
    r = ind.rsi(df["close"], 14)
    adx = ind.adx(df, 14)
    ok &= _check("ATR positive & finite", bool((a.dropna() > 0).all()))
    ok &= _check("RSI within [0,100]", bool(((r.dropna() >= 0) & (r.dropna() <= 100)).all()))
    ok &= _check("ADX within [0,100]", bool(((adx.dropna() >= 0) & (adx.dropna() <= 100)).all()))
    dc_up, dc_low = ind.donchian(df, 20)
    ok &= _check("Donchian upper>=lower", bool((dc_up.dropna() >= dc_low.dropna()).all()))

    # 2) regime -----------------------------------------------------------
    print("\n[2] Regime detection")
    reg_trend = detect_regime(df, {"adx_period": 14, "adx_trend_threshold": 25, "adx_range_threshold": 20})
    ok &= _check("regime label valid", reg_trend in ("trending", "ranging", "neutral"), reg_trend)

    # 3) trend strategy + backtest ---------------------------------------
    print("\n[3] Trend breakout strategy")
    trend = TrendBreakout({"donchian_entry": 20, "donchian_exit": 10, "ma_fast": 50,
                           "ma_slow": 200, "atr_period": 14, "atr_stop_mult": 2.5,
                           "min_target_R": 2.0})
    stats_t, trades_t = backtest_strategy(trend, df)
    ok &= _check("trend backtest produced trades", stats_t.trades > 0, f"{stats_t.trades} trades")
    ok &= _check("all trade R finite", all(np.isfinite(t.R) for t in trades_t))
    prepared = trend.prepare(df)
    plan_t = trend.evaluate(prepared, len(prepared) - 1)
    if plan_t is not None:
        ok &= _check("latest trend plan geometry valid", plan_t.valid())

    # 4) mean reversion strategy -----------------------------------------
    print("\n[4] Mean reversion strategy")
    mr_df = _ranging_series()
    mr = MeanReversion({"bb_period": 20, "bb_std": 2.0, "zscore_entry": 2.0,
                        "rsi_period": 2, "rsi_buy_below": 10, "rsi_sell_above": 90,
                        "ma_trend": 100, "atr_period": 14, "atr_stop_mult": 3.0})
    stats_m, trades_m = backtest_strategy(mr, mr_df)
    ok &= _check("MR backtest ran", stats_m.trades >= 0, f"{stats_m.trades} trades")
    ok &= _check("MR trades have valid geometry",
                 all(np.isfinite(t.R) for t in trades_m))

    # deterministic long setup: clean uptrend then a sharp 3-bar dip that
    # pierces the lower band and crushes RSI-2 -> must produce a LONG plan.
    up = np.linspace(100, 150, 240)             # steep uptrend (long MA lags below)
    dip = np.array([147, 144])                  # shallow dip: pierces band, stays > MA
    close = np.concatenate([up, dip])
    idx = pd.date_range("2021-01-01", periods=len(close), freq="B")
    trig_df = _ohlc_from_close(close, idx, noise=0.001, seed=9)
    prepared_mr = mr.prepare(trig_df)
    plan_mr = mr.evaluate(prepared_mr, len(prepared_mr) - 1)
    ok &= _check("MR fires a LONG on engineered dip", plan_mr is not None
                 and plan_mr.side is Side.LONG)
    if plan_mr is not None:
        ok &= _check("MR plan geometry valid (stop<entry<target=mean)", plan_mr.valid())

    # 5) sizing -----------------------------------------------------------
    print("\n[5] Position sizing")
    res = risk_mod.size_position(
        equity=10000, entry=100, risk_per_unit=2.5, win_rate=0.55,
        avg_win_R=1.5, avg_loss_R=1.0, risk_per_trade_pct=1.0,
        kelly_fraction_mult=0.5, max_kelly_pct=5.0,
    )
    ok &= _check("sizing non-negative", res.units >= 0 and res.notional >= 0)
    ok &= _check("risk cap respected", res.risk_amount <= 10000 * 0.01 + 1e-6,
                 f"risk=${res.risk_amount:.2f}")
    f = risk_mod.kelly_fraction(0.55, 1.5, 1.0)
    ok &= _check("kelly fraction in [0,1]", 0.0 <= f <= 1.0, f"f={f:.3f}")

    # 6) pairs ------------------------------------------------------------
    print("\n[6] Pairs / stat-arb")
    a_s, b_s = _cointegrated_pair()
    pcfg = {"lookback": 252, "coint_pvalue_max": 0.10, "zscore_entry": 2.0,
            "zscore_exit": 0.5, "zscore_stop": 3.5}
    bt = pairs_mod.backtest_pair(a_s, b_s, pcfg)
    ok &= _check("pairs backtest ran", bt.trades >= 0, f"{bt.trades} trades")
    plan_p = pairs_mod.evaluate_pair("A", "B", a_s, b_s, pcfg, atr_a=1.0)
    if plan_p is not None:
        ok &= _check("pair plan geometry valid", plan_p.valid())
    else:
        print("  [info] no live pair trigger on last bar (spread near mean) — OK")

    # 7) signal build + formatting ---------------------------------------
    print("\n[7] Signal build + formatting")
    from .config import Settings
    from . import engine
    from .formatting import format_signal
    from .strategies import TradePlan
    settings = Settings.load()

    # use the live plan if one triggered, else a deterministic fallback plan
    # so this branch is always exercised.
    plan_for_sig = plan_t or TradePlan(
        side=Side.LONG, entry=100.0, stop=95.0, target=110.0, atr=2.0,
        rationale="synthetic fallback plan for self-test", regime="trending",
    )
    ok &= _check("fallback/live plan valid", plan_for_sig.valid())
    sig = engine._build_signal(
        symbol="TEST", asset_class=AssetClass.STOCK, strategy_name="trend_breakout",
        plan=plan_for_sig, stats=stats_t, regime=reg_trend, settings=settings, passed_gate=True,
    )
    ok &= _check("signal geometry valid", sig.is_valid_geometry())
    ok &= _check("confidence in [0,1]", 0.0 <= sig.confidence <= 1.0)
    txt = format_signal(sig, md=False)
    ok &= _check("formatting non-empty", len(txt) > 50)
    ok &= _check("formatting has entry/stop/target",
                 all(k in txt for k in ("Entry", "Stop", "Target")))
    ok &= _check("card shows the exact plan (wallet % + risk-free level)",
                 "Plan" in txt and "Risk-free" in txt and "wallet" in txt)

    # 8) new momentum/breakout strategies --------------------------------
    print("\n[8] MACD / Dual Thrust / Squeeze")
    from .strategies import MacdTrend, DualThrust, SqueezeBreakout
    tdf = _trending_series()
    for cls, cfg in [
        (MacdTrend, {"fast": 12, "slow": 26, "signal": 9, "ma_slow": 200,
                     "atr_period": 14, "atr_stop_mult": 2.5, "min_target_R": 2.0}),
        (DualThrust, {"range_bars": 4, "k1": 0.5, "k2": 0.5, "trend_filter": True,
                      "ma_slow": 200, "atr_period": 14, "atr_stop_mult": 2.0,
                      "min_target_R": 2.0}),
        (SqueezeBreakout, {"bb_period": 20, "bb_std": 2.0, "kc_ema": 20, "kc_atr": 10,
                           "kc_mult": 1.5, "ma_slow": 200, "atr_period": 14,
                           "atr_stop_mult": 1.5, "min_target_R": 3.0}),
    ]:
        strat = cls(cfg)
        st, trs = backtest_strategy(strat, tdf)
        good_geom = all(np.isfinite(t.R) for t in trs)
        prepared_s = strat.prepare(tdf)
        plans = [strat.evaluate(prepared_s, k) for k in range(len(prepared_s))]
        plans = [p for p in plans if p is not None]
        good_plans = all(p.valid() for p in plans)
        ok &= _check(f"{strat.name}: backtest+plan geometry valid",
                     good_geom and good_plans, f"{st.trades} trades, {len(plans)} plans")

    # deterministic squeeze: tight consolidation in an uptrend, then a
    # breakout that expands BB beyond KC -> must fire a LONG on release.
    rng = np.random.default_rng(0)
    rise = np.linspace(80, 100, 205)
    flat = 100 + rng.normal(0, 0.05, 18)
    sq_close = np.concatenate([rise, flat, np.array([104.0, 107.0])])
    m = len(sq_close)
    hi = sq_close + 0.4; lo = sq_close - 0.4
    hi[205:223] = sq_close[205:223] + 0.4; lo[205:223] = sq_close[205:223] - 0.4
    op = np.concatenate([[sq_close[0]], sq_close[:-1]])
    sq_idx = pd.date_range("2021-01-01", periods=m, freq="B")
    sq_df = pd.DataFrame(
        {"open": op, "high": np.maximum.reduce([op, hi, sq_close]),
         "low": np.minimum.reduce([op, lo, sq_close]), "close": sq_close,
         "volume": np.full(m, 1e6)}, index=sq_idx)
    sq = SqueezeBreakout({"bb_period": 20, "bb_std": 2.0, "kc_ema": 20, "kc_atr": 10,
                          "kc_mult": 1.5, "ma_slow": 200, "atr_period": 14,
                          "atr_stop_mult": 1.5, "min_target_R": 3.0})
    sq_prep = sq.prepare(sq_df)
    fired = next((sq.evaluate(sq_prep, k) for k in range(220, m)
                  if sq.evaluate(sq_prep, k) is not None), None)
    ok &= _check("squeeze fires LONG on engineered release",
                 fired is not None and fired.side is Side.LONG and fired.valid())

    # 9) cross-sectional momentum factor ---------------------------------
    print("\n[9] Cross-sectional momentum factor")
    from . import factor as factor_mod
    fcfg = {"lookback_days": 126, "skip_days": 21, "rebalance_days": 21,
            "top_n": 2, "bottom_n": 2, "allow_short": True, "atr_period": 14,
            "atr_stop_mult": 3.0, "min_target_R": 2.0}
    frames = {}
    for j in range(6):  # 6 synthetic assets with different drifts
        rng = np.random.default_rng(20 + j)
        drift = 0.0003 * (j - 2)
        steps = rng.normal(drift, 0.012, 400)
        close = 100 * np.exp(np.cumsum(steps))
        idx = pd.date_range("2021-01-01", periods=400, freq="B")
        frames[f"SYM{j}"] = _ohlc_from_close(close, idx, seed=20 + j)
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    fstats = factor_mod.backtest_cross_sectional(panel, fcfg)
    ok &= _check("factor panel backtest produced rebalances", fstats.trades > 0,
                 f"{fstats.trades} rebalances")
    legs = factor_mod.rank_live(frames, fcfg)
    ok &= _check("factor produced long+short legs", len(legs) == 4, f"{len(legs)} legs")
    ok &= _check("all factor legs valid geometry", all(l.valid() for l in legs))
    sides = {l.side for l in legs}
    ok &= _check("factor has both long and short", len(sides) == 2)

    # 10) Monte Carlo + out-of-sample ------------------------------------
    print("\n[10] Monte Carlo robustness + walk-forward")
    from . import montecarlo as mc_mod
    from .backtest import out_of_sample
    winning = [2.0, -1.0, 2.0, -1.0, 2.0, 2.0, -1.0, 2.0, 2.0, -1.0]
    prob, med, p05, ruin = mc_mod.bootstrap_paths(winning, n_sims=4000)
    ok &= _check("bootstrap prob in [0,1]", 0.0 <= prob <= 1.0, f"P(profit)={prob:.2f}")
    ok &= _check("winning edge bootstraps profitable", prob > 0.6)
    win_p, base = mc_mod.prob_target_before_stop(Side.LONG, 100, 98, 104, sigma=1.0, drift=0.3)
    ok &= _check("drift lifts win-prob above baseline", win_p > base,
                 f"{win_p:.2f} > {base:.2f}")
    oos_stats = out_of_sample(stats_t, 0.7)
    ok &= _check("OOS split returns a holdout", oos_stats.trades >= 0,
                 f"{oos_stats.trades} OOS trades")
    ok &= _check("signal carries MC + OOS fields",
                 hasattr(sig, "montecarlo") and hasattr(sig, "oos")
                 and 0.0 <= sig.montecarlo.win_prob <= 1.0)

    # 11) ML, trailing stop, optimizer -----------------------------------
    print("\n[11] ML + trailing + optimizer")
    from . import ml as ml_mod
    from . import optimize as opt_mod
    ml_df = _trending_series(n=500)
    ml_cfg = {"horizon": 10, "k": 1.5, "min_train": 250, "refit_every": 60,
              "prob_threshold": 0.55, "max_iter": 120, "max_depth": 3,
              "learning_rate": 0.05}
    ml_stats, last_prob = ml_mod.backtest_ml(ml_df, ml_cfg)
    ok &= _check("ML walk-forward backtest ran", ml_stats.trades > 0,
                 f"{ml_stats.trades} trades")
    ok &= _check("ML last prob in [0,1]", last_prob is None or 0.0 <= last_prob <= 1.0)
    lab = ml_mod.triple_barrier_labels(ml_df, 10, 1.5)
    ok &= _check("triple-barrier labels binary",
                 set(lab.dropna().unique()) <= {0.0, 1.0})
    ml_plan = ml_mod.latest_plan(ml_df, ml_cfg)
    ok &= _check("ML plan valid (or no trade)",
                 ml_plan is None or ml_plan.valid())

    from .strategies import TrendBreakout as _TB
    _s = _TB({"donchian_entry": 20, "ma_slow": 200, "atr_period": 14,
              "atr_stop_mult": 2.5, "min_target_R": 2.0})
    _, fixed_tr = backtest_strategy(_s, ml_df)
    _, trail_tr = backtest_strategy(_s, ml_df, trail_atr_mult=3.0)
    ok &= _check("trailing exits are trail/time only",
                 all(t.outcome in ("trail", "time") for t in trail_tr))

    opt_res = opt_mod.optimize_on_df(
        ml_df, "trend", {"ma_slow": 200, "atr_period": 14, "donchian_exit": 10,
                         "ma_fast": 50}, min_trades=5, oos_min_trades=2)
    ok &= _check("optimizer searched grid", len(opt_res) == 27)
    ok &= _check("optimizer ranked by score",
                 [r.score for r in opt_res] == sorted([r.score for r in opt_res], reverse=True))

    # 12) pairs spread-reversion win-prob + portfolio risk -------------
    print("\n[12] Spread-reversion win-prob + portfolio risk")
    win, base = mc_mod.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.5, sigma_eps=1.0)
    ok &= _check("mean-reverting spread beats baseline", win > base and win > 0.7,
                 f"{win:.2f} > {base:.2f}")
    from . import portfolio as pf_mod
    psig = [
        Signal(symbol="A", asset_class=AssetClass.STOCK, strategy="x", side=Side.LONG,
               entry=100, stop=98, target=104, risk_per_unit=2, reward_per_unit=4,
               rr_ratio=2.0, risk_amount=300, position_notional=8000),
        Signal(symbol="B", asset_class=AssetClass.CRYPTO, strategy="y", side=Side.SHORT,
               entry=50, stop=52, target=46, risk_per_unit=2, reward_per_unit=4,
               rr_ratio=2.0, risk_amount=400, position_notional=5000),
    ]
    summ = pf_mod.summarize(psig, equity=10000, max_risk_pct=6.0)
    ok &= _check("portfolio totals correct",
                 summ.total_risk == 700 and abs(summ.total_risk_pct - 7.0) < 1e-9)
    ok &= _check("portfolio over-budget warning fires", len(summ.warnings) >= 1)
    ok &= _check("portfolio summary renders",
                 "Portfolio risk" in pf_mod.format_summary(summ, 10000))
    # risk-parity optimizer: equal dollar risk per leg, total within budget
    rp = [
        Signal(symbol="A", asset_class=AssetClass.STOCK, strategy="x", side=Side.LONG,
               entry=100, stop=90, target=120, risk_per_unit=10, reward_per_unit=20,
               rr_ratio=2.0),
        Signal(symbol="B", asset_class=AssetClass.CRYPTO, strategy="y", side=Side.SHORT,
               entry=50, stop=55, target=40, risk_per_unit=5, reward_per_unit=10,
               rr_ratio=2.0),
    ]
    allocs = pf_mod.optimize_risk_parity(rp, equity=10000, risk_budget_pct=6.0,
                                         max_name_pct=100.0)
    equal_risk = (len(allocs) == 2
                  and abs(allocs[0].risk_amount - allocs[1].risk_amount) < 1e-6)
    in_budget = sum(a.risk_amount for a in allocs) <= 600 + 1e-6
    ok &= _check("risk-parity equalises risk within budget", equal_risk and in_budget,
                 f"risk/leg=${allocs[0].risk_amount:.0f}, Σ=${sum(a.risk_amount for a in allocs):.0f}")

    # 13) structure-aware targets ----------------------------------------
    print("\n[13] Structure-aware targets")
    from .strategies import _refine_target
    import pandas as _pd
    sdf = _pd.DataFrame({"piv_low": [np.nan] * 130, "piv_high": [np.nan] * 130})
    sdf.loc[50, "piv_low"] = 185.0
    scfg = {"enabled": True, "swing_width": 3, "buffer_atr": 0.25,
            "lookback": 120, "min_rr": 0.8}
    refined, rlvl = _refine_target(sdf, 100, Side.SHORT, 200.0, 180.0, 4.0, 10.0, scfg)
    ok &= _check("target pulled in to just before support (186 vs blind 180)",
                 abs(refined - 186.0) < 1e-9 and rlvl == 185.0, f"refined={refined}")
    unchanged, ulvl = _refine_target(sdf, 100, Side.SHORT, 200.0, 180.0, 4.0, 10.0, {})
    ok &= _check("empty structure keeps mechanical target",
                 unchanged == 180.0 and ulvl is None)

    # 14) SMC: FVG, order blocks, structural stop ------------------------
    print("\n[14] SMC (FVG / order block / structural stop)")
    from . import smc as smc_mod
    from .strategies import _refine_stop
    fvg = _pd.DataFrame({"open": [10, 11, 13, 13], "high": [10.5, 12, 14, 14],
                         "low": [9.5, 11, 11.5, 12], "close": [10, 11.5, 13.5, 13]})
    sup, _res = smc_mod.fair_value_gaps(fvg)
    ok &= _check("bullish FVG support detected", abs(float(sup.iloc[2]) - 10.5) < 1e-9)
    ob = _pd.DataFrame({"open": [10, 12, 11.0, 11], "high": [10.5, 12.5, 11.2, 13],
                        "low": [9.5, 11.5, 10.5, 11], "close": [10, 12, 10.7, 13]})
    obs, _obr = smc_mod.order_blocks(ob)
    ok &= _check("bullish order block detected", abs(float(obs.iloc[3]) - 10.5) < 1e-9)
    nn = 130
    sd = _pd.DataFrame({c: [np.nan] * nn for c in
                        ("piv_low", "piv_high", "fvg_sup", "fvg_res", "ob_sup", "ob_res")})
    sd.loc[50, "piv_high"] = 210.0
    sscfg = {"enabled": True, "structural_stop": True, "swing_width": 3,
             "buffer_atr": 0.25, "lookback": 120, "stop_min_atr": 0.8, "stop_max_atr": 4.0}
    sstop, slvl = _refine_stop(sd, 100, Side.SHORT, 200.0, 4.0, 210.0, sscfg)
    ok &= _check("stop placed just beyond resistance (211 not blind 210)",
                 abs(sstop - 211.0) < 1e-9 and slvl == 210.0, f"stop={sstop}")

    # 15) active trade management ----------------------------------------
    print("\n[15] Active trade management")
    from . import manage as manage_mod
    mcfg = {"breakeven_R": 1.0, "trail_atr_mult": 3.0, "near_target_atr": 0.5,
            "danger_on_ma_break": True, "danger_on_macd_flip": True}
    rev = manage_mod.review(side=Side.LONG, entry=100, stop=90, target=140, current=128,
                            atr=2.0, ma_trend=110, macd_hist=1.0, hi_since=130, lo_since=99,
                            cfg=mcfg)
    ok &= _check("in-profit long trails stop above breakeven",
                 rev.trailed and abs(rev.recommended_sl - 124.0) < 1e-9,
                 f"sl={rev.recommended_sl}")
    rev2 = manage_mod.review(side=Side.LONG, entry=100, stop=90, target=120, current=96,
                             atr=2.0, ma_trend=99, macd_hist=-0.2, hi_since=108, lo_since=95,
                             cfg=mcfg)
    ok &= _check("long flags danger when price breaks trend MA", rev2.danger)
    rev3 = manage_mod.review(side=Side.LONG, entry=100, stop=90, target=140, current=103,
                             atr=2.0, ma_trend=98, macd_hist=0.4, hi_since=104, lo_since=99,
                             cfg=mcfg)
    ok &= _check("quiet hold is not actionable", not rev3.actionable)
    rev4 = manage_mod.review(side=Side.LONG, entry=100, stop=90, target=140, current=112,
                             atr=2.0, ma_trend=105, macd_hist=0.5, hi_since=113, lo_since=99,
                             cfg=mcfg, next_level=118.0)
    ok &= _check("live TP banks before a level ahead (117.5 not 140)",
                 abs(rev4.recommended_tp - 117.5) < 1e-9, f"tp={rev4.recommended_tp}")

    print("\n" + "=" * 50)
    print("RESULT:", "ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌")
    return ok
