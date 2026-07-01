import numpy as np
import pandas as pd

from quantaura import data as data_mod
from quantaura.config import Settings
from quantaura.data import _parse_tgju, asset_class_of
from quantaura.models import AssetClass, Side, Signal


SAMPLE = {"total": 4, "data": [
    ["739,500", "738,000", "741,200", "740,800", "800", "0.11",
     "<span class=\"x\">2024/01/17</span>", "1402/10/27"],
    ["738,700", "737,000", "740,000", "739,500", "500", "0.07", "2024/01/16", "1402/10/26"],
    ["737,000", "735,500", "739,000", "738,700", "1200", "0.16", "2024/01/15", "1402/10/25"],
    ["bad", "row", "ignored", "x", "", "", "", ""],     # unparseable -> skipped
]}


def test_parse_tgju_basic():
    df = _parse_tgju(SAMPLE, 400)
    assert len(df) == 3                        # bad row skipped
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing    # sorted ascending
    assert df.index[0].year == 2024            # gregorian, not jalali (1402)
    assert (df["high"] >= df["low"]).all()
    assert df["open"].iloc[-1] == 739500.0


def test_parse_tgju_empty_raises():
    import pytest
    with pytest.raises(data_mod.DataError):
        _parse_tgju({"data": []}, 100)


def test_iran_routing():
    uni = {"iran": ["price_dollar_rl", "geram18"]}
    assert asset_class_of("price_dollar_rl", uni) is AssetClass.IRAN
    assert asset_class_of("geram18", {}) is AssetClass.IRAN        # via IRAN_NAMES
    assert asset_class_of("AAPL", uni) is AssetClass.STOCK


def _iran_df(n=320, seed=1):
    # a steadily devaluing rial -> strong uptrend in USD/gold (rial terms)
    rng = np.random.default_rng(seed)
    close = 500000 * np.exp(np.cumsum(rng.normal(0.0015, 0.01, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    hi = close * (1 + rng.uniform(0, 0.004, n))
    lo = close * (1 - rng.uniform(0, 0.004, n))
    op = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": op, "high": np.maximum.reduce([op, hi, close]),
         "low": np.minimum.reduce([op, lo, close]), "close": close,
         "volume": np.zeros(n)}, index=idx)


def test_iran_shorts_are_forecasts(monkeypatch):
    from quantaura import engine
    settings = Settings.load()
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: _iran_df())
    sigs = engine.scan_symbol("price_dollar_rl", AssetClass.IRAN, settings, publish_only=False)
    for s in sigs:
        if s.side is Side.SHORT:
            # published as a forecast: flagged, and with NO position to enter
            assert s.forecast_only
            assert s.position_units == 0 and s.position_notional == 0 and s.risk_amount == 0
        else:
            assert not s.forecast_only


def test_global_gold_short_is_tradeable(monkeypatch):
    # 'ons' is the GLOBAL gold ounce (USD) — shortable via normal brokers, so a
    # short must NOT be a forecast even though it is sourced from tgju.
    from quantaura import engine
    settings = Settings.load()

    def _down(*a, **k):
        n = 320
        close = np.linspace(2200, 1900, n)
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        return pd.DataFrame({"open": np.concatenate([[close[0]], close[:-1]]),
                             "high": close + 3, "low": close - 3, "close": close,
                             "volume": np.zeros(n)}, index=idx)
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", _down)
    sigs = engine.scan_symbol("ons", AssetClass.IRAN, settings, publish_only=False)
    shorts = [s for s in sigs if s.side is Side.SHORT]
    assert shorts, "expected a short setup on the falling series"
    for s in shorts:
        assert not s.forecast_only            # global gold is tradeable both ways


def test_global_gold_label_and_disclaimer():
    from quantaura.formatting import format_signal
    from quantaura.data import is_global_tradeable
    assert is_global_tradeable("ons") and not is_global_tradeable("sekee")
    s = Signal(symbol="ons", asset_class=AssetClass.IRAN, strategy="dual_thrust",
               side=Side.SHORT, entry=4000, stop=4100, target=3800,
               risk_per_unit=100, reward_per_unit=200, rr_ratio=2.0,
               forecast_only=False, confidence=0.7)
    txt = format_signal(s, md=False)
    assert "(global)" in txt                  # not "(iran)"
    assert "Global USD spot" in txt           # not the Tehran-only disclaimer
    assert "Tehran free-market" not in txt


def test_iran_short_mode_off(monkeypatch):
    from quantaura import engine
    settings = Settings.load()
    settings.cfg.setdefault("iran", {})["short_mode"] = "off"
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: _iran_df())
    sigs = engine.scan_symbol("price_dollar_rl", AssetClass.IRAN, settings, publish_only=False)
    assert all(s.side is Side.LONG for s in sigs)


def test_forecast_card_renders():
    from quantaura.formatting import format_signal
    s = Signal(symbol="geram18", asset_class=AssetClass.IRAN, strategy="trend_breakout",
               side=Side.SHORT, entry=800000, stop=830000, target=740000,
               risk_per_unit=30000, reward_per_unit=60000, rr_ratio=2.0,
               forecast_only=True, confidence=0.6)
    txt = format_signal(s, md=False)
    assert "Downside forecast" in txt and "buy lower" in txt
    assert "740" in txt          # the pullback target appears
