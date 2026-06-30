"""Telegram bot front-end for QuantAura.

Commands:
  /start, /help        — usage
  /scan [stocks|forex|crypto|all]  — scan the universe, push gated signals
  /signal SYMBOL       — analyse one symbol now (e.g. /signal AAPL,
                         /signal EURUSD=X, /signal BTC/USDT)
  /pairs               — scan the cointegration pairs
  /factor              — scan the cross-sectional momentum factor
  /ml [SYMBOL]         — gradient-boosting model signal(s)
  /status              — show the active configuration

Heavy work (network + backtests) runs in a worker thread via
asyncio.to_thread so the event loop never blocks. If a broadcast chat id
and the job-queue extra are configured, the bot also pushes a scheduled
scan automatically.

All Markdown sends fall back to plain text if Telegram rejects the
entities, so a stray character in a rationale can never break a command.
"""
from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from . import engine
from . import journal as journal_mod
from . import portfolio as portfolio_mod
from .config import Settings
from .data import asset_class_of
from .formatting import format_scan_summary, format_signal
from .storage import DEFAULT_DB, Store

log = logging.getLogger("quantaura.bot")

_MAX_DETAIL_SIGNALS = 8  # cap how many full cards we push per scan


# ---------------------------------------------------------------------
def _plain(text: str) -> str:
    return text.replace("*", "").replace("`", "")


async def _reply(message, text: str, md: bool = True) -> None:
    """Reply, preferring Markdown but falling back to plain text on error."""
    if md:
        try:
            await message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            return
        except BadRequest:
            log.warning("Markdown reply rejected; sending plain text.")
    await message.reply_text(_plain(text))


async def _broadcast(bot, chat_id: str, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except BadRequest:
        await bot.send_message(chat_id=chat_id, text=_plain(text))


def _authorized(settings: Settings, update: Update) -> bool:
    allowed = settings.telegram_allowed_users
    if not allowed:
        return True
    user = update.effective_user
    return bool(user and user.id in allowed)


async def _guard(settings: Settings, update: Update) -> bool:
    if not _authorized(settings, update):
        await update.message.reply_text("⛔ You are not authorized to use this bot.")
        return False
    return True


# ---------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    text = (
        "*QuantAura* — backtest-validated quant signals.\n\n"
        "Every signal carries a precise *entry / stop / target* and is only "
        "published if the strategy has a measured edge on that instrument's "
        "own history.\n\n"
        "*Commands*\n"
        "• `/scan [stocks|forex|crypto|all]` — scan & push gated signals\n"
        "• `/signal SYMBOL` — analyse one symbol now\n"
        "• `/pairs` — scan cointegration pairs\n"
        "• `/factor` — scan the cross-sectional momentum factor\n"
        "• `/ml [SYMBOL]` — gradient-boosting model signal(s)\n"
        "• `/subscribe` · `/unsubscribe` — scheduled scans in this chat\n"
        "• `/performance` — live track record of published signals\n"
        "• `/track` — resolve open signals against the latest prices\n"
        "• `/manage` — what to do now with each open position\n"
        "• `/reset` — clear the journal (start fresh)\n"
        "• `/status` — show configuration\n\n"
        "Examples: `/signal AAPL` · `/signal EURUSD=X` · `/signal BTC/USDT`"
    )
    await _reply(update.message, text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    uni = settings.universe
    g = settings.signal_gate
    text = (
        "*QuantAura status*\n"
        f"Timeframe: `{settings.data.get('timeframe')}`\n"
        f"Universe: {len(uni.get('stocks', []))} stocks, "
        f"{len(uni.get('forex', []))} FX, {len(uni.get('crypto', []))} crypto, "
        f"{len(uni.get('iran', []))} Iran (gold/USD)\n"
        "Strategies: `trend, macd, dual_thrust, squeeze, mean_reversion, "
        "pairs, factor_momentum, ml_gboost`\n"
        f"Trailing stop: {settings.risk.get('use_trailing_stop')} "
        f"({settings.risk.get('trail_atr_mult')}×ATR)\n"
        f"Gate: ≥{g.get('min_backtest_trades')} trades, "
        f"win≥{float(g.get('min_win_rate',0))*100:.0f}%, "
        f"PF≥{g.get('min_profit_factor')}, "
        f"P(profit)≥{float(g.get('min_prob_profitable',0))*100:.0f}%\n"
        f"Risk: {settings.risk.get('risk_per_trade_pct')}%/trade, "
        f"{settings.risk.get('kelly_fraction')}×Kelly\n"
        f"Account equity: ${settings.account_equity:,.0f}"
    )
    await _reply(update.message, text)


def _record_new(context, signals: list) -> tuple[list, int]:
    """Record gated signals to the journal; split into (new, repeat_count).

    A signal whose symbol+strategy+side is already open within the cooldown
    is treated as a repeat and suppressed (no spam on every scan).
    """
    settings: Settings = context.application.bot_data["settings"]
    store: Store | None = context.application.bot_data.get("store")
    jcfg = settings.section("journal")
    if store is None or not jcfg.get("enabled", True):
        return list(signals), 0
    cooldown = float(jcfg.get("cooldown_days", 3))
    new, repeats = [], 0
    for s in signals:
        # only journal genuinely published, tradeable positions
        if not s.passed_gate or s.forecast_only:
            new.append(s)
            continue
        if store.record_signal(s, cooldown):
            new.append(s)
        else:
            repeats += 1
    return new, repeats


async def _publish(target, context, signals, record: bool = True,
                   portfolio: bool = True):
    """Send a batch of signals: summary, detail cards, repeats note, portfolio."""
    new, repeats = _record_new(context, signals) if record else (list(signals), 0)
    await _reply(target, format_scan_summary(new))
    for sig in new[:_MAX_DETAIL_SIGNALS]:
        await _reply(target, format_signal(sig, md=True))
    if repeats:
        await _reply(target, f"🔁 {repeats} repeat signal(s) suppressed "
                             f"(already open within cooldown).")
    if portfolio and new:
        settings: Settings = context.application.bot_data["settings"]
        summary = portfolio_mod.summarize(
            new, settings.account_equity,
            max_risk_pct=float(settings.risk.get("portfolio_max_risk_pct", 6.0)),
            max_per_class=int(settings.risk.get("max_open_per_class", 5)),
            max_name_pct=float(settings.risk.get("max_name_pct", 25.0)))
        text = portfolio_mod.format_summary(summary, settings.account_equity)
        if text:
            await _reply(target, text)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    arg = (context.args[0].lower() if context.args else "all")
    classes = None if arg in ("all", "") else [arg]
    if classes and arg not in ("stocks", "forex", "crypto"):
        await update.message.reply_text(
            "Usage: /scan [stocks|forex|crypto|all]"
        )
        return

    await update.message.reply_text(
        f"🔎 Scanning {'all markets' if not classes else arg}… this can take a minute."
    )
    signals = await asyncio.to_thread(
        engine.scan_universe, settings, classes, True
    )
    await _publish(update.message, context, signals)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /signal SYMBOL  (e.g. /signal AAPL)")
        return
    symbol = context.args[0].strip().upper()
    ac = asset_class_of(symbol, settings.universe)
    await update.message.reply_text(f"🔎 Analysing {symbol} ({ac.value})…")
    try:
        # publish_only=False so the user sees the analysis even if it
        # doesn't clear the gate (clearly flagged as such).
        signals = await asyncio.to_thread(
            engine.scan_symbol, symbol, ac, settings, False
        )
    except Exception as exc:
        await update.message.reply_text(f"⚠️ Could not analyse {symbol}: {exc}")
        return
    if not signals:
        await update.message.reply_text(
            f"No active setup on {symbol} right now. No trigger from any strategy "
            f"on the latest bar — the model stays flat."
        )
        return
    # /signal is inspection (may be below gate) -> don't journal/dedup it
    await _publish(update.message, context, signals, record=False)


async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    await update.message.reply_text("🔎 Scanning cointegration pairs…")
    signals = await asyncio.to_thread(engine.scan_pairs, settings, True)
    await _publish(update.message, context, signals)


async def cmd_factor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    await update.message.reply_text("🔎 Scanning the cross-sectional momentum factor…")
    signals = await asyncio.to_thread(engine.scan_factor, settings, True)
    await _publish(update.message, context, signals)


async def cmd_ml(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    if context.args:
        symbol = context.args[0].strip().upper()
        ac = asset_class_of(symbol, settings.universe)
        await update.message.reply_text(f"🤖 Training the ML model on {symbol}…")
        signals = await asyncio.to_thread(engine.scan_ml_symbol, symbol, ac, settings, False)
        record = False                       # single-symbol /ml is inspection
    else:
        await update.message.reply_text("🤖 Running the ML model across the universe… (slow)")
        signals = await asyncio.to_thread(engine.scan_ml, settings, True)
        record = True
    await _publish(update.message, context, signals, record=record)


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    store: Store | None = context.application.bot_data.get("store")
    if store is None:
        await update.message.reply_text("Journaling is disabled, so subscriptions are off.")
        return
    store.add_subscriber(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Subscribed. You'll receive scheduled signal scans in this chat.")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    store: Store | None = context.application.bot_data.get("store")
    if store is None:
        return
    removed = store.remove_subscriber(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Unsubscribed." if removed else "You were not subscribed.")


async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    store: Store | None = context.application.bot_data.get("store")
    if store is None:
        await update.message.reply_text("Journaling is disabled.")
        return
    await _reply(update.message, journal_mod.format_performance(store.performance()))


async def cmd_track(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    store: Store | None = context.application.bot_data.get("store")
    if store is None:
        await update.message.reply_text("Journaling is disabled.")
        return
    await update.message.reply_text("⏱ Resolving open signals against the latest prices…")
    max_hold = int(settings.section("journal").get("max_hold_days", 30))
    s = await asyncio.to_thread(journal_mod.update_open_signals, store, settings, max_hold)
    if s["checked"] == 0:
        await update.message.reply_text(
            "No open signals to track yet. Run /scan first so positions get "
            "journaled, then /track resolves them against later prices.")
        return
    msg = (f"Checked {s['checked']} open signal(s): {s['tp']} hit target, "
           f"{s['sl']} stopped, {s['expired']} expired.")
    if s.get("no_data"):
        msg += (f"\n⚠️ Couldn't fetch prices for {s['no_data']} of them "
                f"(network/proxy?) — they were left open. Make sure v2rayN is "
                f"connected, then try again.")
    await update.message.reply_text(msg)
    await _reply(update.message, journal_mod.format_performance(store.performance()))


def _recipients(settings: Settings, store) -> set:
    r: set[str] = set()
    if settings.telegram_broadcast_chat_id:
        r.add(settings.telegram_broadcast_chat_id)
    if store is not None:
        r |= {str(c) for c in store.subscribers()}
    return r


async def cmd_manage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    store: Store | None = context.application.bot_data.get("store")
    if store is None:
        await update.message.reply_text("Journaling is disabled.")
        return
    await update.message.reply_text("⏱ Reviewing open positions…")
    max_hold = int(settings.section("journal").get("max_hold_days", 30))
    await asyncio.to_thread(journal_mod.update_open_signals, store, settings, max_hold)
    open_count = len(await asyncio.to_thread(store.open_signals))
    reviews = await asyncio.to_thread(engine.review_positions, settings, store)
    if not reviews:
        if open_count == 0:
            await update.message.reply_text(
                "No open positions to manage. Run /scan first so strong signals "
                "get journaled — then /manage gives live SL/TP advice on them.")
        else:
            await update.message.reply_text(
                f"You have {open_count} open position(s), but I couldn't fetch "
                f"live prices for them (network/proxy?). Check that v2rayN is "
                f"connected, then try /manage again.")
        return
    from .formatting import format_management
    for row, rev in reviews:
        await _reply(update.message, format_management(row, rev))


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    store: Store | None = context.application.bot_data.get("store")
    if store is None:
        await update.message.reply_text("Journaling is disabled.")
        return
    n = await asyncio.to_thread(store.clear_signals)
    await update.message.reply_text(
        f"🧹 Journal cleared — removed {n} signal(s). The next scan starts "
        f"fresh with the current strategy. (Subscriptions kept.)")


async def _scheduled_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    store: Store | None = context.application.bot_data.get("store")
    recipients = _recipients(settings, store)

    # 1) resolve open signals, then push DANGER management alerts
    if store is not None:
        try:
            max_hold = int(settings.section("journal").get("max_hold_days", 30))
            await asyncio.to_thread(journal_mod.update_open_signals, store, settings, max_hold)
            reviews = await asyncio.to_thread(engine.review_positions, settings, store)
            from .formatting import format_management
            for row, rev in reviews:
                if rev.danger and recipients:
                    for chat_id in recipients:
                        await _broadcast(context.bot, chat_id, format_management(row, rev))
        except Exception:
            log.warning("journal/manage update failed", exc_info=True)

    # 2) scan, record new (dedup), broadcast to env id + all subscribers
    signals = await asyncio.to_thread(engine.scan_universe, settings, None, True)
    new, _ = _record_new(context, signals) if store is not None else (signals, 0)
    if not new:
        return
    recipients = _recipients(settings, store)
    if not recipients:
        return

    summary_text = format_scan_summary(new)
    port = portfolio_mod.format_summary(
        portfolio_mod.summarize(
            new, settings.account_equity,
            max_risk_pct=float(settings.risk.get("portfolio_max_risk_pct", 6.0)),
            max_per_class=int(settings.risk.get("max_open_per_class", 5)),
            max_name_pct=float(settings.risk.get("max_name_pct", 25.0))),
        settings.account_equity)
    for chat_id in recipients:
        await _broadcast(context.bot, chat_id, summary_text)
        for sig in new[:_MAX_DETAIL_SIGNALS]:
            await _broadcast(context.bot, chat_id, format_signal(sig, md=True))
        if port:
            await _broadcast(context.bot, chat_id, port)


# ---------------------------------------------------------------------
def build_application(settings: Settings) -> Application:
    if not settings.telegram_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    # route outbound traffic through a proxy (e.g. a local v2ray SOCKS) if set
    from . import net
    proxy = net.apply_proxy(settings.proxy_url)
    builder = Application.builder().token(settings.telegram_token)
    if proxy:
        builder = builder.proxy(proxy).get_updates_proxy(proxy)
        log.info("Using proxy for Telegram: %s", proxy)
    app = builder.build()
    app.bot_data["settings"] = settings

    # local SQLite journal (dedup, live track record, subscribers)
    store = None
    jcfg = settings.section("journal")
    if jcfg.get("enabled", True):
        store = Store(settings.journal_db_path or str(DEFAULT_DB))
        log.info("Journal enabled at %s", store.path)
    app.bot_data["store"] = store

    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("pairs", cmd_pairs))
    app.add_handler(CommandHandler("factor", cmd_factor))
    app.add_handler(CommandHandler("ml", cmd_ml))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("track", cmd_track))
    app.add_handler(CommandHandler("manage", cmd_manage))
    app.add_handler(CommandHandler("reset", cmd_reset))

    # scheduled scan: runs if the job-queue extra is installed and there is
    # somewhere to send (a broadcast id or the subscriber journal).
    if app.job_queue is not None and (settings.telegram_broadcast_chat_id or store is not None):
        hours = float(settings.section("schedule").get("scan_interval_hours", 1) or 1)
        hours = max(0.1, hours)          # guard against an absurdly small interval
        app.job_queue.run_repeating(_scheduled_scan, interval=hours * 3600, first=30)
        log.info("Scheduled scan enabled (every %sh).", hours)

    return app


def run(settings: Settings | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = settings or Settings.load()
    app = build_application(settings)
    log.info("QuantAura bot starting (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
