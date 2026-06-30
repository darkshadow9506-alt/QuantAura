"""Command-line interface.

Usage:
  python -m quantaura bot                     # run the Telegram bot
  python -m quantaura scan [--class stocks]   # scan & print signals
  python -m quantaura signal AAPL             # analyse one symbol
  python -m quantaura pairs                    # scan cointegration pairs
  python -m quantaura selftest                # offline self-check (no network)
"""
from __future__ import annotations

import argparse
import sys

from .config import Settings
from .data import asset_class_of
from .formatting import format_scan_summary, format_signal


def _cmd_scan(args, settings: Settings) -> int:
    from . import engine

    classes = None if args.cls in (None, "all") else [args.cls]
    signals = engine.scan_universe(settings, classes, include_pairs=True)
    print(format_scan_summary(signals).replace("*", "").replace("`", ""))
    print()
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    from . import portfolio as portfolio_mod
    summary = portfolio_mod.summarize(
        signals, settings.account_equity,
        max_risk_pct=float(settings.risk.get("portfolio_max_risk_pct", 6.0)),
        max_per_class=int(settings.risk.get("max_open_per_class", 5)),
        max_name_pct=float(settings.risk.get("max_name_pct", 25.0)))
    text = portfolio_mod.format_summary(summary, settings.account_equity)
    if text:
        print(text.replace("*", ""))
    return 0


def _cmd_signal(args, settings: Settings) -> int:
    from . import engine

    sym = args.symbol.strip().upper()
    ac = asset_class_of(sym, settings.universe)
    try:
        signals = engine.scan_symbol(sym, ac, settings, publish_only=False)
    except Exception as exc:
        print(f"Could not analyse {sym}: {exc}")
        return 1
    if not signals:
        print(f"No active setup on {sym} right now.")
        return 0
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_pairs(args, settings: Settings) -> int:
    from . import engine

    signals = engine.scan_pairs(settings, publish_only=False)
    if not signals:
        print("No pair setups right now.")
        return 0
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_factor(args, settings: Settings) -> int:
    from . import engine

    signals = engine.scan_factor(settings, publish_only=False)
    if not signals:
        print("No factor signals right now.")
        return 0
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_ml(args, settings: Settings) -> int:
    from . import engine
    from .data import asset_class_of as _ac

    if getattr(args, "symbol", None):
        sym = args.symbol.strip().upper()
        signals = engine.scan_ml_symbol(sym, _ac(sym, settings.universe), settings,
                                        publish_only=False)
    else:
        signals = engine.scan_ml(settings, publish_only=False)
    if not signals:
        print("No ML signals right now.")
        return 0
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_optimize(args, settings: Settings) -> int:
    from .optimize import optimize_symbol, format_report

    print(format_report(optimize_symbol(args.symbol.strip().upper(),
                                        args.strategy, settings)))
    return 0


def _cmd_manage(args, settings: Settings) -> int:
    from . import engine, journal
    from .formatting import format_management
    from .storage import DEFAULT_DB, Store

    store = Store(settings.journal_db_path or str(DEFAULT_DB))
    journal.update_open_signals(
        store, settings, int(settings.section("journal").get("max_hold_days", 30)))
    reviews = engine.review_positions(settings, store)
    if not reviews:
        print("No open positions to manage.")
        return 0
    for row, rev in reviews:
        print(format_management(row, rev).replace("*", "").replace("`", ""))
        print("-" * 60)
    return 0


def _cmd_reset(args, settings: Settings) -> int:
    from .storage import DEFAULT_DB, Store

    store = Store(settings.journal_db_path or str(DEFAULT_DB))
    n = store.clear_signals()
    print(f"Cleared {n} signal(s) from the journal.")
    return 0


def _cmd_bot(args, settings: Settings) -> int:
    from .bot import run

    run(settings)
    return 0


def _cmd_selftest(args, settings: Settings) -> int:
    from .selftest import run_selftest

    ok = run_selftest()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantaura", description="QuantAura signal bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan the universe")
    p_scan.add_argument("--class", dest="cls", default="all",
                        choices=["all", "stocks", "forex", "crypto"])

    p_sig = sub.add_parser("signal", help="analyse one symbol")
    p_sig.add_argument("symbol")

    sub.add_parser("pairs", help="scan cointegration pairs")
    sub.add_parser("factor", help="scan the cross-sectional momentum factor")

    p_ml = sub.add_parser("ml", help="gradient-boosting model signal(s)")
    p_ml.add_argument("symbol", nargs="?", help="one symbol, or omit for the whole universe")

    p_opt = sub.add_parser("optimize", help="walk-forward parameter search for a strategy")
    p_opt.add_argument("symbol")
    p_opt.add_argument("--strategy", default="trend",
                       choices=["trend", "mean_reversion", "macd", "dual_thrust", "squeeze"])

    sub.add_parser("manage", help="management advice for open journaled positions")
    sub.add_parser("reset", help="clear the signal journal (start fresh)")
    sub.add_parser("bot", help="run the Telegram bot")
    sub.add_parser("selftest", help="offline self-check (no network)")

    args = parser.parse_args(argv)
    settings = Settings.load()

    # route data fetches through a proxy (e.g. a v2ray local SOCKS) if set
    from . import net
    net.apply_proxy(settings.proxy_url)

    dispatch = {
        "scan": _cmd_scan,
        "signal": _cmd_signal,
        "pairs": _cmd_pairs,
        "factor": _cmd_factor,
        "ml": _cmd_ml,
        "optimize": _cmd_optimize,
        "manage": _cmd_manage,
        "reset": _cmd_reset,
        "bot": _cmd_bot,
        "selftest": _cmd_selftest,
    }
    return dispatch[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
