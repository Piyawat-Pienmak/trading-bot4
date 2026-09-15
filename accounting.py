#!/usr/bin/env python3
"""Offline accounting maintenance, or explicit read-only exchange reconciliation."""

import argparse
import csv
import json
import os
from pathlib import Path

from trade_ledger import TradeLedger, account_scope, atomic_csv, parse_ms
from trade_reconciliation import Reconciler


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Expected a JSON array of exchange records")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="data/accounting/ledger.sqlite3")
    parser.add_argument("--scope", default="legacy-unassigned", help="Exact ledger namespace; use list-scopes to inspect")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-scopes")
    legacy = commands.add_parser("import-legacy", help="Preserve CSV evidence without claiming confirmed accounting")
    legacy.add_argument("--trades", default="reports/trade_log.csv")
    legacy.add_argument("--events", default="reports/event_log.csv")
    legacy.add_argument("--report", default="reports/accounting/legacy_review.csv")
    export = commands.add_parser("export")
    export.add_argument("--output-dir", default="reports/accounting")
    backup = commands.add_parser("backup")
    backup.add_argument("destination")
    for name in ("import-fills", "import-income"):
        command = commands.add_parser(name, help="Import Binance-format JSON or CSV, retaining original currencies and IDs")
        command.add_argument("path")
        command.add_argument("--complete-from", help="UTC ISO timestamp; only for an independently verified complete export")
        command.add_argument("--complete-through")
        command.add_argument("--symbol")
    link = commands.add_parser("link-order", help="Explicitly attribute an exchange order after verifying its ownership")
    link.add_argument("--trade-id", required=True)
    link.add_argument("--order-id", required=True)
    link.add_argument("--role", choices=["entry", "exit", "cleanup", "stop_loss", "take_profit"], required=True)
    adopt = commands.add_parser("adopt-entry", help="Register a verified historical entry ID; accounting remains pending")
    adopt.add_argument("--order-id", required=True)
    adopt.add_argument("--symbol", required=True)
    adopt.add_argument("--direction", choices=["long", "short"], required=True)
    adopt.add_argument("--time", required=True, help="UTC ISO timestamp")
    order_import = commands.add_parser("import-orders", help="Attach terminal order evidence to already linked regular orders")
    order_import.add_argument("path")
    sync = commands.add_parser("reconcile", help="Read-only Binance API calls; never submits or cancels orders")
    sync.add_argument("--symbol", default="DOGEUSDT")
    sync.add_argument("--account-id", default=os.getenv("BOT_ACCOUNT_ID", ""))
    sync.add_argument("--testnet", action="store_true")
    sync.add_argument("--from-time", help="UTC ISO timestamp; ordinary history is limited to recent months")
    args = parser.parse_args(argv)
    client = None
    scope = args.scope
    if args.command == "reconcile":
        from binance.um_futures import UMFutures
        from dotenv import load_dotenv
        load_dotenv()
        key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
        if not key or not secret:
            parser.error("BINANCE_API_KEY and BINANCE_API_SECRET are required for reconciliation")
        testnet = args.testnet or os.getenv("BINANCE_TESTNET") == "1"
        base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        account_id = args.account_id or os.getenv("BOT_ACCOUNT_ID", "")
        resolved_scope = account_scope(base_url, key, account_id)
        if args.scope != "legacy-unassigned" and args.scope != resolved_scope:
            parser.error("Scope does not match the selected exchange account/environment")
        scope = resolved_scope
        client = UMFutures(key=key, secret=secret, base_url=base_url, timeout=10)
    ledger = TradeLedger(args.ledger, scope)
    try:
        if args.command not in {"list-scopes", "export", "backup"}:
            ledger.acquire_writer_lock()
        if args.command == "list-scopes":
            scopes = set()
            for table in ("trades", "legacy", "settings", "fills", "income"):
                scopes.update(r[0] for r in ledger.db.execute(f"SELECT DISTINCT scope FROM {table}"))
            print("\n".join(sorted(scopes)))
        elif args.command == "import-legacy":
            for path, kind in ((args.trades, "trades"), (args.events, "events")):
                if not Path(path).is_file():
                    parser.error(f"Missing legacy source: {path}")
                ledger.import_legacy(path, kind)
            rows = ledger.legacy_report()
            atomic_csv(args.report, rows)
            counts = {status: sum(r["accounting_status"] == status for r in rows) for status in {r["accounting_status"] for r in rows}}
            print(json.dumps({"report": args.report, "scope": scope, "counts": counts}))
        elif args.command == "export":
            output = Path(args.output_dir)
            ledger.export(output / "trades.csv", output / "events.csv")
            orders = ledger.orders()
            atomic_csv(output / "orders.csv", orders)
            for table in ("fills", "income", "snapshots"):
                rows = ledger.raw_rows(table)
                for row in rows:
                    row["scope"] = scope
                    if table == "fills":
                        owner = next((o for o in orders if o["symbol"] == row["symbol"] and
                                      (o["exchange_id"] if o["kind"] == "regular" else o["actual_id"]) == str(row["orderId"])), None)
                        row["bot_trade_id"] = owner["trade_id"] if owner else ""
                        row["order_role"] = owner["role"] if owner else "unassigned"
                    for key, value in list(row.items()):
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                atomic_csv(output / f"{table}.csv", rows)
            atomic_csv(output / "legacy_review.csv", ledger.legacy_report())
            print(f"Exported ledger scope {scope} to {output}")
        elif args.command == "backup":
            ledger.backup(args.destination)
            print(f"Backup saved to {args.destination}")
        elif args.command in {"import-fills", "import-income"}:
            if any((args.complete_from, args.complete_through, args.symbol)) and not all((args.complete_from, args.complete_through, args.symbol)):
                parser.error("Coverage requires --complete-from, --complete-through and --symbol together")
            start = parse_ms(args.complete_from)
            end = parse_ms(args.complete_through)
            if args.complete_from and start > end:
                parser.error("Coverage start must not be after its end")
            (ledger.ingest_fills if args.command == "import-fills" else ledger.ingest_income)(load_records(args.path))
            if args.complete_from:
                ledger.mark_coverage("fills" if args.command == "import-fills" else "income", args.symbol.upper(), start, end)
        elif args.command == "link-order":
            ledger.link_order(args.trade_id, args.order_id, args.role)
        elif args.command == "adopt-entry":
            for order in ledger.orders():
                if order["symbol"] == args.symbol.upper() and order["exchange_id"] == args.order_id and order["kind"] == "regular":
                    parser.error("Order already linked; use its existing trade ID")
            timestamp = parse_ms(args.time)
            state = ledger.create_trade(args.symbol.upper(), args.direction, {"source": "historical_entry", "pnl_asset": "USDT"})
            with ledger.db:
                ledger.db.execute("UPDATE trades SET created_ms=? WHERE scope=? AND trade_id=?", (timestamp, scope, state["trade_id"]))
            ledger.link_order(state["trade_id"], args.order_id, "entry")
            print(state["trade_id"])
        elif args.command == "import-orders":
            for record in load_records(args.path):
                matches = [o for o in ledger.orders() if o["kind"] == "regular" and o["symbol"] == record["symbol"] and o["exchange_id"] == str(record["orderId"])]
                if len(matches) != 1:
                    raise ValueError("Link each imported order to a verified trade first")
                ledger.acknowledge(matches[0]["client_id"], record)
        elif args.command == "reconcile":
            errors = Reconciler(ledger, client, args.symbol.upper()).sync(start_ms=parse_ms(args.from_time) if args.from_time else None)
            print(json.dumps({"scope": scope, "errors": errors, "trades": len(ledger.trades())}))
            return 1 if errors else 0
        return 0
    finally:
        ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
