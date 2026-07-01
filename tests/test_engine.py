"""End-to-end engine tests with injected (synthetic) data — no network.

These prove the full orchestration: data -> regime -> strategy ->
backtest gate -> sizing -> Signal, including failure robustness.
"""
import numpy as np
import pandas as pd
import pytest

from quantaura import engine
from quantaura.config import Settings
from quantaura.models import AssetClass, Side


def _rising(n=320, seed=5):
    rng = np.random.default_rng(seed)
    # steady uptrend with mild noise: makes fresh highs -> trend breakouts,
    # but with pullbacks so the backtest has a realistic mix of outcomes.
    steps = rng.normal(0.0012, 0.01, n)
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([open_, high, close]),
         "low": np.minimum.reduce([open_, low, close]), "close": close,
         "volume": rng.uniform(1e6, 5e6, n)}, index=idx)


@pytest.fixture
def settings():
    return Settings.load()


def test_scan_symbol_orchestration(monkeypatch, settings):
    df = _rising()
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: df)

    sigs = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=False)
    assert isinstance(sigs, list)
    for s in sigs:
        assert s.is_valid_geometry()
        assert 0.0 <= s.confidence <= 1.0
        assert s.position_units >= 0
        assert s.risk_per_unit > 0
        if s.side is Side.LONG:
            assert s.stop < s.entry < s.target
        # backtest stats attached
        assert s.backtest.trades >= 0
        # Monte Carlo + out-of-sample layers attached and sane
        assert 0.0 <= s.montecarlo.win_prob <= 1.0
        assert 0.0 <= s.montecarlo.prob_profitable <= 1.0
        assert 0.0 <= s.montecarlo.risk_of_ruin <= 1.0
        assert s.confluence >= 1
        # explicit plan fields
        assert s.equity_pct >= 0 and s.risk_pct >= 0
        # risk-free (+1R) sits one R beyond entry in the trade's direction
        one_R = s.risk_per_unit
        if s.side is Side.LONG:
            assert abs(s.risk_free_at - (s.entry + one_R)) < 1e-3
        else:
            assert abs(s.risk_free_at - (s.entry - one_R)) < 1e-3


def test_scan_symbol_robust_to_bad_data(monkeypatch, settings):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", boom)
    # scan_universe must swallow per-symbol errors and never crash
    out = engine.scan_universe(settings, classes=["stocks"], include_pairs=False)
    assert out == []


def test_scan_symbol_short_history_returns_empty(monkeypatch, settings):
    short = _rising(n=30)
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: short)
    out = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=False)
    assert out == []


def test_one_card_per_side(monkeypatch, settings):
    # a clean uptrend with a last-bar breakout fires several trending
    # strategies; the engine must collapse them to ONE long card.
    n = 320
    close = np.linspace(100, 175, n) + np.sin(np.linspace(0, 9, n)) * 1.2
    close[-1] = close[-2] + 3.5
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": np.concatenate([[close[0]], close[:-1]]),
                       "high": close + 0.8, "low": close - 0.8, "close": close,
                       "volume": np.full(n, 2e6)}, index=idx)
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: df)
    sigs = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=False)
    longs = [s for s in sigs if s.side is Side.LONG]
    assert len(longs) == 1                       # collapsed, not 2-3 cards
    s = longs[0]
    assert s.confluence >= 2                      # kept the confluence count
    assert "also confirmed by" in s.rationale     # names the agreeing strategies


def test_target_distance_capped(monkeypatch, settings):
    # verify the structure config carries the hard target cap and it binds
    from quantaura.strategies import _stop_and_target
    import pandas as _pd
    scfg = dict(settings.section("structure"))
    scfg["max_target_atr"] = 3.0                  # tight cap for the test
    nn = 130
    d = _pd.DataFrame({c: [np.nan] * nn for c in
                       ("piv_low", "piv_high", "fvg_sup", "fvg_res", "ob_sup", "ob_res")})
    atr, entry = 1000.0, 50000.0
    _, target, _ = _stop_and_target(d, 100, Side.LONG, entry, atr, 2.5, 5.0, scfg)
    # a 5R target on a 2.5-ATR stop would be 12.5 ATR away; the cap forbids it
    assert target - entry <= 3.0 * atr + 1e-6


def test_publish_filter_drops_low_confidence(monkeypatch, settings):
    df = _rising()
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: df)

    # With publish_only=False the filter is a no-op (every candidate returned).
    raw = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=False)
    # Force an impossible confidence floor: nothing should survive publishing.
    settings.signal_gate["min_confidence"] = 1.01
    published = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=True)
    assert published == []
    # A zero floor disables the filter -> publishing returns the gated subset.
    settings.signal_gate["min_confidence"] = 0.0
    everything = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=True)
    assert all(s.confidence >= 0.0 for s in everything)
    assert len(raw) >= len(everything)


def test_publish_filter_keeps_forecasts(settings):
    from quantaura.models import Signal

    settings.signal_gate["min_confidence"] = 0.99
    weak = Signal(symbol="X", asset_class=AssetClass.IRAN, strategy="trend_breakout",
                  side=Side.SHORT, entry=100, stop=102, target=94, risk_per_unit=2,
                  reward_per_unit=6, rr_ratio=3.0, confidence=0.10, forecast_only=True)
    strong_tradeable = Signal(symbol="Y", asset_class=AssetClass.STOCK,
                              strategy="trend_breakout", side=Side.LONG, entry=100,
                              stop=98, target=106, risk_per_unit=2, reward_per_unit=6,
                              rr_ratio=3.0, confidence=0.10, forecast_only=False)
    kept = engine._publish_filter([weak, strong_tradeable], settings, publish_only=True)
    # forecast survives despite low confidence; the tradeable low-conf one is cut
    assert weak in kept and strong_tradeable not in kept


def test_scan_factor_builds_signals(monkeypatch, settings):
    # distinct trend per symbol so ranking is non-degenerate
    def fake(symbol, asset_class, **k):
        h = (abs(hash(symbol)) % 7) - 3
        return _rising(n=400, seed=100 + h)
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", fake)
    sigs = engine.scan_factor(settings, publish_only=False)
    assert any(s.strategy == "factor_momentum" for s in sigs)
    for s in sigs:
        assert s.is_valid_geometry()
        assert s.risk_per_unit > 0
        assert 0.0 <= s.confidence <= 1.0
        assert s.regime == "cross-sectional"
