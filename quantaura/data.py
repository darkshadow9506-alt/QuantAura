"""Market data providers.

A single function -- `get_ohlcv(symbol, asset_class, ...)` -- returns a
normalized DataFrame with columns [open, high, low, close, volume] and a
tz-aware DatetimeIndex, regardless of the underlying source.

Sources:
  * stocks / forex / indices  -> yfinance
  * crypto                    -> ccxt (default exchange: binance)

Network libraries are imported lazily so the analytics/backtest code and
the unit tests run without them installed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import AssetClass

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

_COLUMNS = ["open", "high", "low", "close", "volume"]

# ccxt timeframe + yfinance interval mapping
_CCXT_TF = {"1d": "1d", "1h": "1h", "4h": "4h", "15m": "15m", "1w": "1w"}
_YF_INTERVAL = {"1d": "1d", "1h": "1h", "1wk": "1wk", "1w": "1wk"}


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("=", "_").replace(":", "_")
    # pickle keeps caching dependency-free (parquet would need pyarrow)
    return CACHE_DIR / f"{safe}.pkl"


def _read_cache(key: str, max_age_minutes: int) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age_min = (time.time() - path.stat().st_mtime) / 60.0
    if age_min > max_age_minutes:
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(_cache_path(key))
    except Exception:
        # caching is best-effort only; never let it break a fetch
        pass


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, keep OHLCV, drop incomplete rows, sort."""
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    keep = [c for c in _COLUMNS if c in df.columns]
    df = df[keep].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[c for c in ("open", "high", "low", "close") if c in df.columns])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ---------------------------------------------------------------------
# yfinance (stocks / forex)
# ---------------------------------------------------------------------
def _fetch_yfinance(symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    import yfinance as yf  # lazy

    interval = _YF_INTERVAL.get(timeframe, "1d")
    # request generous period; yfinance caps intraday history
    if interval.endswith(("m", "h")):
        period = "60d" if interval == "1h" else "7d"
    else:
        # ~ lookback trading days plus buffer, capped at "max"
        days = min(int(lookback * 1.6) + 30, 4000)
        period = f"{days}d"

    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned no data for {symbol}")
    # yfinance can return a MultiIndex column frame for single tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = _normalize(raw)
    return df.tail(lookback)


# ---------------------------------------------------------------------
# ccxt (crypto)
# ---------------------------------------------------------------------
def _fetch_ccxt(symbol: str, timeframe: str, lookback: int, exchange_id: str) -> pd.DataFrame:
    import ccxt  # lazy

    tf = _CCXT_TF.get(timeframe, "1d")
    ex_class = getattr(ccxt, exchange_id)
    exchange = ex_class({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=lookback)
    if not raw:
        raise DataError(f"ccxt returned no data for {symbol} on {exchange_id}")
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return _normalize(df).tail(lookback)


# ---------------------------------------------------------------------
# tgju.org (Iranian free-market gold & USD)
# ---------------------------------------------------------------------
# Friendly names for the tgju symbol codes (used for display).
IRAN_NAMES = {
    "price_dollar_rl": "USD/IRR (free market, rial)",
    "price_eur": "EUR/IRR (free market, rial)",
    "geram18": "Gold 18k / gram (rial)",
    "geram24": "Gold 24k / gram (rial)",
    "mesghal": "Gold mesghal (rial)",
    "sekee": "Gold coin — Emami (rial)",
    "nim": "Gold coin — half (rial)",
    "rob": "Gold coin — quarter (rial)",
    "ons": "Gold ounce — global (USD)",
}

# tgju symbols that are actually GLOBAL and USD-priced: retail CAN trade them
# long OR short via ordinary brokers (e.g. XAU/USD), unlike the rial-denominated
# local markets. These must NOT be treated as non-shortable "Iran forecasts".
IRAN_GLOBAL_TRADEABLE = {"ons"}


def is_global_tradeable(symbol: str) -> bool:
    return symbol in IRAN_GLOBAL_TRADEABLE

_TGJU_URL = "https://api.tgju.org/v1/market/indicator/summary-table-data/{symbol}"


def _strip_html(s: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", str(s)).strip()


def _parse_tgju(payload: dict, lookback: int) -> pd.DataFrame:
    """Parse a tgju summary-table-data JSON payload into OHLCV.

    Each row is [open, low, high, close, change, change%, gregorian_date,
    jalali_date]. We take open/close directly and derive high/low as the
    max/min of the four prices (robust to a low/high column swap). The
    Gregorian date (year >= 1990) is used; the Jalali date is ignored.
    """
    rows = payload.get("data") or []
    recs = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            o, a, b, c = (float(str(row[k]).replace(",", "").strip()) for k in range(4))
        except (ValueError, TypeError):
            continue
        # find the Gregorian date (a cell that parses to a sane year)
        dt = pd.NaT
        for k in range(4, len(row)):
            cand = pd.to_datetime(_strip_html(row[k]), errors="coerce")
            if not pd.isna(cand) and cand.year >= 1990:
                dt = cand
                break
        if pd.isna(dt):
            continue
        recs.append((dt, o, max(o, a, b, c), min(o, a, b, c), c))
    if not recs:
        raise DataError("tgju returned no parseable rows")
    df = pd.DataFrame(recs, columns=["date", "open", "high", "low", "close"])
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["volume"] = 0.0
    return df.tail(lookback)


def _fetch_tgju(symbol: str, lookback: int) -> pd.DataFrame:
    import requests  # lazy; provided transitively by yfinance/ccxt

    url = _TGJU_URL.format(symbol=symbol)
    resp = requests.get(url, timeout=20,
                        headers={"User-Agent": "Mozilla/5.0 QuantAura"})
    resp.raise_for_status()
    df = _parse_tgju(resp.json(), lookback)
    if df.empty:
        raise DataError(f"tgju returned no data for {symbol}")
    return df


class DataError(RuntimeError):
    """Raised when a provider cannot return usable data."""


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_ohlcv(
    symbol: str,
    asset_class: AssetClass,
    timeframe: str = "1d",
    lookback: int = 400,
    cache_minutes: int = 30,
    ccxt_exchange: str = "binance",
) -> pd.DataFrame:
    """Return normalized OHLCV for one symbol. Retries with backoff."""
    key = f"{asset_class.value}_{symbol}_{timeframe}"
    cached = _read_cache(key, cache_minutes)
    if cached is not None and not cached.empty:
        return cached

    last_err: Optional[Exception] = None
    attempts = 3
    for attempt in range(attempts):
        try:
            if asset_class is AssetClass.CRYPTO:
                df = _fetch_ccxt(symbol, timeframe, lookback, ccxt_exchange)
            elif asset_class is AssetClass.IRAN:
                df = _fetch_tgju(symbol, lookback)
            else:
                df = _fetch_yfinance(symbol, timeframe, lookback)
            if df.empty:
                raise DataError(f"empty frame for {symbol}")
            _write_cache(key, df)
            return df
        except Exception as exc:  # network / parsing
            last_err = exc
            # gentle, capped backoff; don't sleep after the final attempt
            if attempt < attempts - 1:
                time.sleep(min(1.5 * (attempt + 1), 4.0))   # 1.5s, 3s
    raise DataError(f"failed to fetch {symbol}: {last_err}")


def asset_class_of(symbol: str, universe: dict[str, list[str]]) -> AssetClass:
    """Resolve which asset class a symbol belongs to from the configured universe."""
    if symbol in universe.get("iran", []) or symbol in IRAN_NAMES:
        return AssetClass.IRAN
    if symbol in universe.get("crypto", []) or "/" in symbol:
        return AssetClass.CRYPTO
    if symbol in universe.get("forex", []) or symbol.endswith("=X"):
        return AssetClass.FOREX
    return AssetClass.STOCK
