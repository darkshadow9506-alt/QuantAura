import numpy as np

from quantaura import montecarlo as mc
from quantaura import portfolio as pf
from quantaura.models import AssetClass, Side, Signal


# ---------------- spread-reversion win probability ----------------
def test_spread_reversion_baseline():
    # driftless gambler's ruin: (z_stop-|z|)/(z_stop-z_exit)
    win, base = mc.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.0, sigma_eps=0.0)
    assert abs(base - (1.0 / 3.0)) < 1e-9
    assert win == base                      # no AR(1) dynamics -> baseline


def test_strong_mean_reversion_beats_baseline():
    # phi=0.5 -> fast reversion; a z=2.5 spread should very likely revert
    win, base = mc.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.5, sigma_eps=1.0,
                                         n_sims=8000)
    assert win > base
    assert win > 0.8                        # strong reversion -> high win prob


def test_no_reversion_when_phi_near_one():
    # phi≈1 is a random walk -> close to the driftless baseline
    win, base = mc.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.999, sigma_eps=1.0,
                                         n_sims=8000)
    assert abs(win - base) < 0.15


def test_assess_pairs_bundle():
    m = mc.assess_pairs(returns_R=[1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                        z_now=2.5, z_exit=0.5, z_stop=3.5, phi=0.5, sigma_eps=1.0)
    assert 0.0 <= m.win_prob <= 1.0
    assert 0.0 <= m.prob_profitable <= 1.0


# ---------------- portfolio summary ----------------
def _sig(symbol, side, ac, risk, notional):
    return Signal(symbol=symbol, asset_class=ac, strategy="x", side=side,
                  entry=100, stop=98, target=104, risk_per_unit=2,
                  reward_per_unit=4, rr_ratio=2.0,
                  risk_amount=risk, position_notional=notional)


def test_portfolio_totals_and_net():
    sigs = [_sig("A", Side.LONG, AssetClass.STOCK, 100, 5000),
            _sig("B", Side.LONG, AssetClass.STOCK, 150, 6000),
            _sig("C", Side.SHORT, AssetClass.CRYPTO, 120, 4000)]
    s = pf.summarize(sigs, equity=10000, max_risk_pct=6.0, max_per_class=5)
    assert s.n == 3 and s.longs == 2 and s.shorts == 1
    assert abs(s.total_risk - 370) < 1e-9
    assert abs(s.total_risk_pct - 3.7) < 1e-9
    assert abs(s.gross_exposure - 15000) < 1e-9
    assert abs(s.net_exposure - 7000) < 1e-9   # 11000 long - 4000 short
    assert s.by_class_risk["stock"] == 250


def test_portfolio_over_budget_warning():
    sigs = [_sig(f"S{i}", Side.LONG, AssetClass.STOCK, 200, 5000) for i in range(5)]
    s = pf.summarize(sigs, equity=10000, max_risk_pct=6.0, max_per_class=10)
    assert s.total_risk_pct == 10.0
    assert any("budget" in w for w in s.warnings)
    assert any("one-directional" in w for w in s.warnings)


def test_portfolio_concentration_warning():
    sigs = [_sig(f"S{i}", Side.LONG if i % 2 else Side.SHORT, AssetClass.STOCK, 50, 1000)
            for i in range(7)]
    s = pf.summarize(sigs, equity=100000, max_risk_pct=50.0, max_per_class=5)
    assert any("concentrated" in w for w in s.warnings)


def test_portfolio_format_and_empty():
    assert pf.format_summary(pf.summarize([], 10000), 10000) == ""
    sigs = [_sig("A", Side.LONG, AssetClass.STOCK, 100, 5000)]
    text = pf.format_summary(pf.summarize(sigs, 10000), 10000)
    assert "Portfolio risk" in text and "Risk at stop" in text


# ---------------- risk-parity optimizer ----------------
def _psig(symbol, entry, stop, side=Side.LONG, ac=AssetClass.STOCK, forecast=False):
    return Signal(symbol=symbol, asset_class=ac, strategy="x", side=side,
                  entry=entry, stop=stop, target=entry + 2 * (entry - stop),
                  risk_per_unit=abs(entry - stop), reward_per_unit=2 * abs(entry - stop),
                  rr_ratio=2.0, forecast_only=forecast)


def test_risk_parity_equalises_risk_and_respects_budget():
    # three different-volatility names; budget 6% of 10k = $600 -> $200 each
    sigs = [_psig("A", 100, 95),    # risk/unit 5
            _psig("B", 50, 49),     # risk/unit 1
            _psig("C", 200, 180)]   # risk/unit 20
    allocs = pf.optimize_risk_parity(sigs, equity=10000, risk_budget_pct=6.0,
                                     max_name_pct=100.0)
    assert len(allocs) == 3
    # equal dollar risk on every leg
    for a in allocs:
        assert abs(a.risk_amount - 200.0) < 1e-6
    # units scale inversely with volatility: A=40, B=200, C=10
    by = {a.symbol: a for a in allocs}
    assert abs(by["A"].units - 40.0) < 1e-6
    assert abs(by["B"].units - 200.0) < 1e-6
    assert abs(by["C"].units - 10.0) < 1e-6
    # total risk stays within the budget
    assert abs(sum(a.risk_amount for a in allocs) - 600.0) < 1e-6


def test_risk_parity_single_name_cap():
    # B is very low-vol -> would take 200*50=$10k notional (100% of wallet);
    # cap at 25% -> $2500, and its risk drops below the equal target.
    sigs = [_psig("A", 100, 95), _psig("B", 50, 49.5)]
    allocs = pf.optimize_risk_parity(sigs, equity=10000, risk_budget_pct=6.0,
                                     max_name_pct=25.0)
    b = next(a for a in allocs if a.symbol == "B")
    assert b.capped is True
    assert b.notional <= 2500 + 1e-6
    assert b.risk_amount < 300.0          # below the uncapped equal target


def test_risk_parity_excludes_forecasts():
    # high-vol name so the full $600 budget fits under the notional cap
    sigs = [_psig("A", 100, 80),
            _psig("IRGOLD", 100, 105, side=Side.SHORT,
                  ac=AssetClass.IRAN, forecast=True)]
    allocs = pf.optimize_risk_parity(sigs, equity=10000, risk_budget_pct=6.0,
                                     max_name_pct=100.0)
    assert [a.symbol for a in allocs] == ["A"]      # forecast excluded
    # the single tradeable name now gets the whole budget
    assert abs(allocs[0].risk_amount - 600.0) < 1e-6 and not allocs[0].capped


def test_risk_parity_zero_equity_safe():
    sigs = [_psig("A", 100, 95)]
    assert pf.optimize_risk_parity(sigs, equity=0) == []


def test_allocation_shows_in_summary_text():
    sigs = [_psig("A", 100, 95, Side.LONG), _psig("B", 50, 49, Side.SHORT)]
    text = pf.format_summary(pf.summarize(sigs, 10000), 10000)
    assert "Risk-parity allocation" in text and "% of wallet" in text
