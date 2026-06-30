# QuantAura — a backtest-validated quant trading signal bot

QuantAura turns real market data into **precise, fully-specified trade
signals** (exact entry, stop, target and position size) and delivers them
over **Telegram**. It is built on *published, mainstream* quantitative
strategies — trend / time-series momentum, short-horizon mean reversion,
and cointegration pairs (statistical arbitrage) — with strict risk
management on top.

> **Read this first.** QuantAura is an educational tool. Every signal is a
> *backtested probability*, not a promise. Real quant funds win on large
> samples of small edges, with leverage, infrastructure and risk controls.
> No honest system can give you "100% safe, guaranteed" entries — anyone
> claiming that is lying. What QuantAura *does* guarantee is that every
> number it shows you is computed from real data by a transparent, tested
> rule, and that it stays silent unless a setup has a **measured edge on
> that instrument's own history**.

---

## Why these strategies (and not "magic")

The strategy set is taken directly from how quantitative trading actually
works (see *What Is Quantitative Trading?* and the strategy taxonomy that
shipped with this project). The robust, well-documented retail-accessible
edges are:

| Strategy | Family | Edge source | Horizon |
|---|---|---|---|
| **Trend breakout (Donchian/Turtle)** | Time-series momentum / trend following | Persistence of price moves (the most robust cross-asset anomaly; the core of CTAs like Man AHL) | weeks–months |
| **MACD trend** | Momentum | Trend-aligned momentum acceleration | weeks |
| **Dual Thrust** | Range breakout | Volatility-scaled breakout of the prior range (Michael Chalek) | days |
| **TTM Squeeze** | Volatility breakout | Energy released after a volatility compression (John Carter) | days–weeks |
| **Mean reversion** | Short-horizon reversal | Temporary supply/demand imbalances revert to fair value | days |
| **Pairs / stat-arb** | Statistical arbitrage | Cointegrated assets converge after diverging | days–weeks |
| **Cross-sectional momentum** | Factor investing | Winners keep winning vs. losers (Jegadeesh–Titman / AQR) | weeks–months |
| **ML (gradient boosting)** | Supervised learning | Patterns across many features → P(win), triple-barrier labels | days–weeks |

Everything QuantAura does maps onto the standard quant pipeline:
**data → signal research → backtest → execution → risk management.** We
deliberately *do not* implement HFT or market making — those need
co-located servers, microsecond latency and FPGAs, and are impossible to
run honestly from a Telegram bot. Pretending otherwise would be the
"made-up" behaviour you explicitly asked to avoid.

### The three engines

**1. Trend breakout (momentum).** Donchian-channel breakout filtered by the
200-period moving average. Go long only when price breaks above the prior
20-bar high *and* trades above its 200-MA (and the mirror image for
shorts). Initial stop = `2.5 × ATR`; target = `2R`. This is textbook
trend-following.

**2. Mean reversion.** Bollinger / z-score stretch + Connors **RSI-2**,
taken *only with* the long-term trend (buy oversold dips inside an uptrend,
fade overbought spikes inside a downtrend). Stop = `3 × ATR`; target = the
moving-average mean (reversion to fair value).

**3. Pairs / statistical arbitrage.** Engle–Granger cointegration test +
OLS hedge ratio + z-score of the spread (Ornstein–Uhlenbeck-style
reversion). Only trades pairs that are statistically cointegrated
(p ≤ 0.05). Long the cheap leg, short the rich leg; exit toward the mean,
stop on a structural break (z ≥ 3.5). The win-probability for a pairs
trade is modelled with a **discrete Ornstein–Uhlenbeck / AR(1)** fit of
the spread's z-score (P that it reverts to the exit band before hitting
the stop band) — the correct model for stat-arb, since the edge is mean
reversion of the spread, not directional drift in either leg.

**4. MACD trend.** Go long when the MACD line crosses *above* its signal
line while price is above the 200-MA (mirror for shorts). ATR stop, 2R
target. A classic, widely-used momentum trigger.

**5. Dual Thrust (Michael Chalek).** `Range = max(HH−LC, HC−LL)` over the
prior N bars; `BuyLine = Open + K1·Range`, `SellLine = Open − K2·Range`.
Break above BuyLine → long, below SellLine → short, with a 200-MA
trend filter to avoid whipsaw. ([formula reference](https://www.quantconnect.com/research/15258/dual-thrust-trading-algorithm/),
[fmz writeup](https://github.com/fmzquant/strategies/blob/master/Dual-Thrust-Trading-Algorithm-ps4.md))

**6. TTM Squeeze (John Carter).** When Bollinger Bands contract *inside*
the Keltner channels, volatility is compressed (a "squeeze"). When it
releases, price tends to expand sharply — we enter in the direction of the
release with a tight 1.5×ATR stop and a wider 3R target.

**7. Cross-sectional momentum (the "momentum factor").** Within each asset
class, rank names by trailing ~6-month return (skipping the last month to
avoid short-term reversal); go **long the strongest, short the weakest**.
This is the Jegadeesh–Titman (1993) momentum factor used at scale by AQR
and others. It is validated by a **panel (long-short basket) backtest**,
not a per-symbol one.

**8. Machine learning (gradient boosting).** A `HistGradientBoosting`
classifier trained on ~18 engineered, look-ahead-free features (multi-
horizon returns, RSI, MACD, ADX, Bollinger %b, distance-from-MA, realised
vol, momentum, volume z-score) to predict the **triple-barrier label** of
López de Prado: *will price hit +k·ATR before −k·ATR within the horizon?*
The trade taken (TP = +k·ATR, SL = −k·ATR) matches the label, so the model
literally estimates P(win). The backtest uses **purged walk-forward**
training (the model predicting bar *t* is fit only on bars whose label
window closed at or before *t*), so there is no leakage. Because it is
heavier to compute, ML has its own `/ml` command and is not part of the
default `/scan`.

A **regime filter (ADX)** decides which engine is even allowed to fire on a
symbol: trending markets → the momentum/breakout engines (trend, MACD, dual
thrust, squeeze), ranging markets → mean reversion. This stops the engines
from fighting each other. Pairs and the momentum factor are relative-value
strategies and run independently of the single-symbol regime.

### How signals stay honest — the backtest gate

A raw strategy trigger is **not published** unless the *same strategy*,
backtested on the *same instrument's* history, clears all of:

- ≥ 15 trades (a meaningful sample, guards against curve-fits)
- win rate ≥ 40%
- profit factor ≥ 1.15
- significance score ("Sharpe") ≥ 0.3

The backtester is **event-driven and look-ahead-free**: entries use only
data up to the signal bar; exits walk forward bar-by-bar; if a bar's range
spans both stop and target we pessimistically assume the stop filled first.
Thresholds live in `config.yaml → signal_gate` and are fully tunable.

### The win-rate / robustness layer

Beyond the raw backtest, four extra checks raise the quality of what gets
published (all standard quant practice — your source doc lists "out-of-sample
testing, walk-forward analysis" and "Monte Carlo simulation of extreme
scenarios"):

- **Walk-forward / out-of-sample gate.** History is split (default
  70/30); the edge must *also* be positive on the held-out tail. This is
  the single most important defence against curve-fitting — a strategy
  that only worked in the first part of history is rejected.
- **Monte Carlo bootstrap.** The backtested trade outcomes are resampled
  thousands of times to estimate the **probability the edge is profitable
  forward**, the bad-case (5th-percentile) result, and the **risk of
  ruin** (probability of a 10R drawdown). A signal must clear
  `min_prob_profitable` (default 60%).
- **Per-signal win probability.** A Monte Carlo barrier simulation models
  price as a random walk with the symbol's *current* drift and volatility
  (ATR) to estimate **P(take-profit before stop)**, shown next to the
  driftless baseline `1/(1+RR)`. If the modelled probability beats the
  baseline, the measured momentum is genuinely tilting the odds.
- **Multi-strategy confluence.** When several independent strategies fire
  the same direction on the same symbol, confidence is boosted — these are
  independent confirmations of the same idea.
- **Structure-aware stops & targets (Smart-Money-Concepts).** Blind
  R-multiple levels ignore where price actually reacts. QuantAura detects
  market structure — **swing pivots, Fair Value Gaps, and order blocks**
  (all quantified and look-ahead-free) — and uses them for both ends of the
  trade:
  - **Targets** are pulled in to just *before* the nearest support
    (shorts) / resistance (longs), so they fill before a reversal.
  - **Stops** are placed just *beyond* the nearest protective level (the
    structure that invalidates the trade), instead of a blind ATR distance,
    with the structural risk capped to `[stop_min_atr, stop_max_atr] × ATR`
    so it never gets wicked out just inside a level nor blows the risk out.

  The backtest uses the same structural stops and targets, so the published
  stats stay consistent. (These are mechanical approximations of
  discretionary SMC/ICT ideas — they capture the quantifiable core, not a
  chartist's exact hand-drawn reading. Tunable in `config.yaml → structure`.)
  Applied to the trend/MACD/dual-thrust/squeeze strategies **and the
  cross-sectional momentum factor**. Pairs use their cointegration-spread
  levels and ML uses symmetric triple-barrier levels (both by design), so
  they keep their own methodology.

All of these feed a single blended **confidence** score (0–100%) shown on
every signal. None of them is a guarantee — they are honest probability
estimates that make weak setups visibly weak.

### Portfolio-level risk

A single trade can be sized correctly yet a *basket* of simultaneous
signals can still over-concentrate risk. After a scan, QuantAura appends a
**portfolio summary**: total risk-at-stop (% of equity), gross / net
exposure, risk by asset class, and warnings when the book exceeds the risk
budget, over-concentrates in one class, or becomes one-directional. This
is the "position management / risk budget" discipline from the doc.

It then runs a **risk-parity optimizer** (`portfolio.optimize_risk_parity`):
a fixed total risk budget (`portfolio_max_risk_pct`) is split *equally*
across the tradeable signals, so every position contributes the same
risk-at-stop. Because units = target-risk ÷ (entry − stop), a more volatile
name automatically gets fewer units for the same dollar risk — i.e.
inverse-volatility / equal-risk-contribution sizing — and the book never
exceeds the budget however many signals fire. Each leg is reported as an
explicit *“% of wallet”* allocation, and any single name is clipped to
`max_name_pct` of equity so one low-vol idea can't dominate. This is the
"portfolio optimizer" component a real quant stack is expected to have.

### Risk management

- **Fixed-fractional sizing:** never risk more than `risk_per_trade_pct`
  (default 1%) of equity per trade.
- **Fractional Kelly:** size is *also* scaled by the measured edge using
  **half-Kelly**, capped at `max_kelly_pct`. The final size is the more
  conservative of the two. Full Kelly is intentionally avoided — it is far
  too aggressive on noisy, non-stationary estimates.
- **Trailing (Chandelier) stop:** for trend/breakout strategies, an opt-in
  ATR trailing stop (`risk.use_trailing_stop`, `trail_atr_mult`) lets
  winners run past the fixed target. The backtest is run *with* the trail
  when enabled, so the published stats reflect how the trade is actually
  managed, and each signal carries a management note describing the trail.

### Tuning without curve-fitting — the optimizer

`python -m quantaura optimize SYMBOL --strategy trend` grid-searches a
strategy's key parameters and **scores every combination by its
out-of-sample expectancy**, never its in-sample fit. This is the
walk-forward discipline that separates a real parameter choice from an
over-fit one. It's a research tool — it reports the best held-out
parameters; it does not silently change your live config.

---

## Install

```bash
git clone <your-fork-url> QuantAura
cd QuantAura
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
# or, for an editable install that adds a `quantaura` command:
pip install -e .
```

Python 3.10+ required. A GitHub Actions workflow (`.github/workflows/ci.yml`)
runs the test suite and the offline self-test on every push.

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

1. **Create a Telegram bot:** open Telegram → talk to
   [@BotFather](https://t.me/BotFather) → `/newbot` → follow the prompts →
   copy the token (`123456789:ABC...`) into `TELEGRAM_BOT_TOKEN`.
2. *(optional)* Put your numeric Telegram id in `TELEGRAM_ALLOWED_USERS`
   (get it from [@userinfobot](https://t.me/userinfobot)) to lock the bot
   to just you.
3. *(optional)* Set `TELEGRAM_BROADCAST_CHAT_ID` to auto-push a scan every
   6 hours.
4. Set `ACCOUNT_EQUITY` so position-size suggestions match your account.

Tune the **universe, strategy parameters, risk and gate** in
`config.yaml` — every knob is documented inline.

## Run

```bash
# 1) Offline self-check (no network, proves the whole pipeline works)
python -m quantaura selftest

# 2) Terminal scans (need market-data access)
python -m quantaura scan --class stocks
python -m quantaura signal AAPL
python -m quantaura signal EURUSD=X
python -m quantaura signal BTC/USDT
python -m quantaura pairs
python -m quantaura factor
python -m quantaura ml AAPL                 # ML model for one symbol
python -m quantaura ml                      # ML across the universe (slower)
python -m quantaura optimize AAPL --strategy trend

# 3) Start the Telegram bot
python -m quantaura bot
```

### Run it 24/7 (always-on)

See **[DEPLOY.md](DEPLOY.md)** for keeping the bot running around the clock.
Quickest path on a server you already have (e.g. an existing v2ray VPS,
which is just a Linux box abroad — run it there for free):

```bash
git clone <your-repo-url> quantaura && cd quantaura
cp .env.example .env && nano .env      # add your bot token
sudo bash deploy/install.sh            # installs a systemd service
```

This auto-restarts on crash and on reboot. A `Dockerfile` +
`docker-compose.yml` are included as an alternative, plus notes on
genuinely-free hosts (e.g. Oracle Cloud Free Tier).

### Telegram commands

| Command | Action |
|---|---|
| `/start`, `/help` | usage |
| `/scan [stocks\|forex\|crypto\|all]` | scan the universe, push gated signals |
| `/signal SYMBOL` | analyse one symbol now (e.g. `/signal AAPL`) |
| `/pairs` | scan the cointegration pairs |
| `/factor` | scan the cross-sectional momentum factor |
| `/ml [SYMBOL]` | gradient-boosting model signal(s) |
| `/subscribe`, `/unsubscribe` | receive scheduled scans in this chat |
| `/performance` | live track record of all published signals |
| `/track` | resolve open signals against the latest prices |
| `/manage` | what to do now with each open position |
| `/status` | show the active configuration |

A signal card shows side, entry/stop/target, R:R, ATR, an **exact
execution plan** (what % of the wallet to enter with, the precise risk-free
/ +1R level where you take partial profit and move the stop to entry, and
the partial → final take-profit prices), the originating strategy's
backtest and out-of-sample record, the Monte Carlo win-probability /
probability-of-profit / risk-of-ruin, any multi-strategy confluence, and a
blended confidence bar.

### Live tracking, deduplication & subscriptions

Every published signal is journaled to a local SQLite database, which gives
the bot three live-trading qualities:

- **No spam (deduplication).** The same symbol+strategy+side is not
  re-published while it is still open within a cooldown window
  (`journal.cooldown_days`); repeats are counted and suppressed.
- **A real track record.** `/track` resolves every open signal against the
  latest price bars (did it hit the target or the stop?), and
  `/performance` reports the live win rate, average R and total R — the
  honest, forward equivalent of the backtest.
- **Subscriptions.** `/subscribe` registers a chat for the scheduled scan
  (every 6h), which first updates the journal, then broadcasts only the
  *new* gated signals plus the portfolio summary to all subscribers.
- **Active management.** `/manage` reviews every open position and tells you
  what to do *now*, based on the live state:
  - a **live take-profit price** — if a support/resistance level lies ahead
    before the target, it recommends banking profit just before it; if the
    path is clear, it says ride to target; if the thesis is breaking, take
    profit / close now;
  - at +1R, take partial profit and move the stop to breakeven;
  - once running, a concrete **trailed stop** price;
  - a **🚨 danger** alert to close/tighten when the thesis breaks (price back
    across the 200-MA or MACD flipping against you).

  The scheduled job (every `schedule.scan_interval_hours`, default 1h) also
  pushes danger alerts unprompted. It's advice only — the bot never places or
  closes orders; you act on Toobit yourself.

## Data sources

- **Stocks / ETFs / FX / indices →** [yfinance](https://github.com/ranaroussi/yfinance) (free, end-of-day + intraday).
- **Crypto →** [ccxt](https://github.com/ccxt/ccxt) (default exchange **Toobit**; change via `CCXT_EXCHANGE` to any ccxt-supported id, e.g. `bybit`, `bitget`, `kucoin`).
- **Iran free-market gold & USD →** [tgju.org](https://www.tgju.org) (USD/IRR,
  18k/24k gold per gram, mesghal, Emami coin, global ounce). These are the
  Tehran open-market prices. Retail can't easily short them, so SHORT
  signals are published as a **"downside forecast"** by default
  (`iran.short_mode: forecast`) — no position to enter, but it shows *how
  far price may fall* so you can time your buys or take profit on holdings
  (set `trade` to treat them as normal shorts, or `off` to drop them).
  Signals carry an extra disclaimer — these markets are heavily
  policy/news-driven, so treat them as educational only. Needs network
  access to tgju (works from Iran directly or through your proxy).

OHLCV is cached locally for `cache_minutes` to respect rate limits. The
default timeframe is **daily** — the horizon where retail quant edges are
real and execution speed doesn't dominate.

> If you run in a restricted/proxied network and data fetches return 403,
> the destination (e.g. Yahoo/Binance) is blocked by that network's egress
> policy, not by QuantAura. Run on an open network or a VPS.

## Tests

```bash
pytest -q            # 108 unit/integration tests (synthetic data, no network)
python -m quantaura selftest
```

## Project layout

```
quantaura/
  config.py        settings (YAML + .env)
  data.py          unified OHLCV providers (yfinance / ccxt / tgju) + cache
  indicators.py    ATR, RSI, Bollinger, z-score, ADX, Donchian, MACD,
                   Keltner, Dual-Thrust range (all look-ahead-free)
  strategies.py    TrendBreakout, MacdTrend, DualThrust, SqueezeBreakout,
                   MeanReversion, ADX regime filter
  pairs.py         Engle-Granger cointegration pairs (stat-arb)
  factor.py        cross-sectional momentum factor + panel backtest
  ml.py            gradient boosting + triple-barrier (purged walk-forward)
  montecarlo.py    bootstrap robustness + P(TP before SL) + spread-reversion
  optimize.py      walk-forward parameter search (scored out-of-sample)
  smc.py           swing pivots, Fair Value Gaps, order blocks (structure)
  portfolio.py     batch risk + risk-parity optimizer (equal-risk weights)
  storage.py       SQLite persistence (signals + subscribers), dedup
  journal.py       resolve open signals to TP/SL, live track record
  manage.py        active management advice for open positions
  risk.py          fixed-fractional + fractional-Kelly sizing
  backtest.py      event-driven, look-ahead-free backtester + R-metrics
  engine.py        orchestration: data -> signal, with the backtest gate
  formatting.py    Telegram/CLI signal rendering
  bot.py           Telegram front-end
  cli.py           command-line interface
  selftest.py      offline end-to-end validation
tests/             pytest suite
config.yaml        all strategy / risk / gate parameters
```

## Honest limitations

- Daily EOD signals; not for scalping or HFT.
- yfinance/ccxt are free data feeds — fine for research, not institutional
  quality.
- Backtests are realistic but still optimistic vs. live trading (slippage,
  fees, borrow, gaps and partial fills are only partially modelled).
- A measured historical edge can decay; markets are non-stationary.
- **Not financial advice.** You are responsible for every trade you take.

## License

MIT — see `LICENSE`.
