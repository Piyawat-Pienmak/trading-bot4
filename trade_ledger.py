"""Durable execution evidence and derived trade accounting (no exchange writes)."""

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

TERMINAL = {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"}
ZERO = Decimal("0")


def now_ms():
    return time.time_ns() // 1_000_000


def iso_ms(value):
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc).isoformat() if value else ""


def parse_ms(value):
    if not value:
        return 0
    return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)


def decimal(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite accounting amount")
    return result


def json_text(value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    return json.dumps(clean(value), sort_keys=True, default=str, allow_nan=False)


def account_scope(base_url, api_key, account_id=""):
    # The API key and secret are never stored. An explicit account label survives key rotation.
    identity = account_id or hashlib.sha256(api_key.encode()).hexdigest()[:24]
    return base_url.rstrip("/") + "|" + identity


def atomic_csv(path, rows, headers=None, limit=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = list(dict.fromkeys(k for row in rows for k in row)) or ["trade_id"]
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows[-limit:] if limit else rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TradeLedger:
    def __init__(self, path, scope):
        self.path = Path(path)
        self.scope = scope
        self._lock_file = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError(f"Unsupported ledger schema: {version}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                scope TEXT, trade_id TEXT, symbol TEXT NOT NULL, direction TEXT NOT NULL,
                created_ms INTEGER NOT NULL, state TEXT NOT NULL,
                PRIMARY KEY(scope, trade_id));
            CREATE TABLE IF NOT EXISTS orders (
                scope TEXT, client_id TEXT, trade_id TEXT NOT NULL, symbol TEXT NOT NULL,
                role TEXT NOT NULL, kind TEXT NOT NULL, exchange_id TEXT, actual_id TEXT,
                status TEXT NOT NULL, executed_qty TEXT, request TEXT NOT NULL,
                response TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(scope, client_id),
                UNIQUE(scope, symbol, kind, exchange_id),
                FOREIGN KEY(scope, trade_id) REFERENCES trades(scope, trade_id));
            CREATE TABLE IF NOT EXISTS fills (
                scope TEXT, symbol TEXT, fill_id TEXT, order_id TEXT NOT NULL,
                time_ms INTEGER NOT NULL, raw TEXT NOT NULL,
                PRIMARY KEY(scope, symbol, fill_id));
            CREATE TABLE IF NOT EXISTS income (
                scope TEXT, income_type TEXT, transaction_id TEXT, symbol TEXT,
                time_ms INTEGER NOT NULL, raw TEXT NOT NULL,
                PRIMARY KEY(scope, income_type, transaction_id));
            CREATE TABLE IF NOT EXISTS events (
                scope TEXT, event_id TEXT, time_ms INTEGER NOT NULL, raw TEXT NOT NULL,
                PRIMARY KEY(scope, event_id));
            CREATE TABLE IF NOT EXISTS coverage (
                scope TEXT, stream TEXT, symbol TEXT, start_ms INTEGER, end_ms INTEGER,
                PRIMARY KEY(scope, stream, symbol, start_ms, end_ms));
            CREATE TABLE IF NOT EXISTS settings (scope TEXT, key TEXT, value TEXT,
                PRIMARY KEY(scope, key));
            CREATE TABLE IF NOT EXISTS snapshots (
                scope TEXT, time_ms INTEGER, raw TEXT, PRIMARY KEY(scope, time_ms));
            CREATE TABLE IF NOT EXISTS legacy (
                scope TEXT, record_id TEXT, kind TEXT, source TEXT, raw TEXT,
                PRIMARY KEY(scope, record_id));
            CREATE INDEX IF NOT EXISTS fills_by_order ON fills(scope, symbol, order_id);
            PRAGMA user_version=1;
        """)

    def close(self):
        self.db.close()
        if self._lock_file:
            self._lock_file.close()

    def acquire_writer_lock(self):
        """One bot/maintenance process per account, including all its symbols."""
        digest = hashlib.sha256(self.scope.encode()).hexdigest()[:24]
        stream = (self.path.parent / (self.path.name + "." + digest + ".lock")).open("a")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise RuntimeError("This ledger account is already in use; stop its bot before maintenance") from None
        self._lock_file = stream

    def setting(self, key, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE scope=? AND key=?", (self.scope, key)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES(?,?,?)", (self.scope, key, json_text(value)))

    def create_trade(self, symbol, direction, state):
        trade_id = uuid.uuid4().hex
        state = {**state, "trade_id": trade_id, "symbol": symbol, "direction": direction}
        with self.db:
            self.db.execute("INSERT INTO trades VALUES(?,?,?,?,?,?)",
                            (self.scope, trade_id, symbol, direction, now_ms(), json_text(state)))
        return self.state(trade_id)

    def state(self, trade_id):
        row = self.db.execute("SELECT state FROM trades WHERE scope=? AND trade_id=?", (self.scope, trade_id)).fetchone()
        if not row:
            raise KeyError(trade_id)
        return json.loads(row[0])

    def save_state(self, state):
        with self.db:
            cursor = self.db.execute("UPDATE trades SET state=? WHERE scope=? AND trade_id=?",
                                     (json_text(state), self.scope, state["trade_id"]))
            if cursor.rowcount != 1:
                raise KeyError(state["trade_id"])

    def trades(self, symbol=None):
        sql = "SELECT * FROM trades WHERE scope=?"
        args = [self.scope]
        if symbol:
            sql += " AND symbol=?"
            args.append(symbol)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY created_ms,trade_id", args)]

    def intent(self, trade_id, role, kind, request):
        client_id = "tb4_" + uuid.uuid4().hex[:28]
        state = self.state(trade_id)
        with self.db:
            self.db.execute("INSERT INTO orders(scope,client_id,trade_id,symbol,role,kind,status,request) VALUES(?,?,?,?,?,?,?,?)",
                            (self.scope, client_id, trade_id, state["symbol"], role, kind, "SUBMITTING", json_text(request)))
        return client_id

    def orders(self, trade_id=None):
        sql = "SELECT * FROM orders WHERE scope=?"
        args = [self.scope]
        if trade_id:
            sql += " AND trade_id=?"
            args.append(trade_id)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY rowid", args)]

    def acknowledge(self, client_id, response, actual=None):
        order = next(r for r in self.orders() if r["client_id"] == client_id)
        regular = order["kind"] == "regular"
        if response.get("symbol") and response["symbol"] != order["symbol"]:
            raise ValueError("Order response symbol does not match the journal")
        exchange_id = response.get("orderId" if regular else "algoId") or order["exchange_id"]
        actual_id = response.get("actualOrderId") or order["actual_id"]
        fill_order_id = str(exchange_id) if regular and exchange_id else str(actual_id) if actual_id else None
        if fill_order_id:
            for other in self.orders():
                if other["client_id"] == client_id or other["symbol"] != order["symbol"]:
                    continue
                other_fill_id = other["exchange_id"] if other["kind"] == "regular" else other["actual_id"]
                if other_fill_id == fill_order_id:
                    raise ValueError("Execution order ID is already attributed to another journal order")
        status = response.get("status" if regular else "algoStatus", order["status"])
        executed = response.get("executedQty", order["executed_qty"]) if regular else order["executed_qty"]
        if actual:
            status = actual.get("status", status)
            executed = actual.get("executedQty", executed)
        elif not regular and not actual_id and status in {"CANCELED", "EXPIRED", "REJECTED"}:
            executed = "0"
        with self.db:
            self.db.execute("UPDATE orders SET exchange_id=?,actual_id=?,status=?,executed_qty=?,response=? WHERE scope=? AND client_id=?",
                            (str(exchange_id) if exchange_id else None, str(actual_id) if actual_id else None,
                             status, str(executed) if executed is not None else None,
                             json_text({**response, **({"actual_order": actual} if actual else {})}), self.scope, client_id))

    def order_error(self, client_id, status, message):
        with self.db:
            self.db.execute("UPDATE orders SET status=?,executed_qty=?,response=? WHERE scope=? AND client_id=?",
                            (status, "0" if status == "REJECTED" else None, json_text({"error": message}), self.scope, client_id))

    def link_order(self, trade_id, order_id, role):
        if role not in {"entry", "exit", "cleanup", "stop_loss", "take_profit"}:
            raise ValueError("Invalid order role")
        state = self.state(trade_id)
        existing = [o for o in self.orders() if o["symbol"] == state["symbol"] and
                    (o["exchange_id"] == str(order_id) and o["kind"] == "regular" or o["actual_id"] == str(order_id))]
        if existing:
            if existing[0]["trade_id"] != trade_id or existing[0]["role"] != role:
                raise ValueError("Order is already assigned to a different trade/role")
            return
        client_id = self.intent(trade_id, role, "regular", {"imported_order_id": str(order_id)})
        with self.db:
            self.db.execute("UPDATE orders SET exchange_id=?,status='UNKNOWN' WHERE scope=? AND client_id=?",
                            (str(order_id), self.scope, client_id))

    def ingest_fills(self, records):
        with self.db:
            for item in records:
                for key in ("symbol", "id", "orderId", "time", "qty", "price", "side"):
                    if item.get(key) in (None, ""):
                        raise ValueError(f"Fill missing {key}")
                if decimal(item["qty"]) <= 0 or decimal(item["price"]) <= 0:
                    raise ValueError("Fill quantity and price must be positive")
                for key in ("commission", "realizedPnl"):
                    if item.get(key) not in (None, ""):
                        decimal(item[key])
                previous = self.db.execute("SELECT raw FROM fills WHERE scope=? AND symbol=? AND fill_id=?",
                                           (self.scope, item["symbol"], str(item["id"]))).fetchone()
                if previous:
                    existing = json.loads(previous[0])
                    for key in ("orderId", "time", "side", "qty", "price", "commission", "commissionAsset", "realizedPnl", "marginAsset", "positionSide"):
                        old, new = existing.get(key), item.get(key)
                        if old not in (None, "") and new not in (None, ""):
                            same = decimal(old) == decimal(new) if key in {"qty", "price", "commission", "realizedPnl", "time"} else str(old) == str(new)
                            if not same:
                                raise ValueError(f"Conflicting {key} for fill {item['id']}; original evidence retained")
                    item = {**existing, **{k: v for k, v in item.items() if v not in (None, "")}}
                self.db.execute("INSERT INTO fills VALUES(?,?,?,?,?,?) ON CONFLICT(scope,symbol,fill_id) DO UPDATE SET raw=excluded.raw",
                                (self.scope, item["symbol"], str(item["id"]), str(item["orderId"]), int(item["time"]), json_text(item)))

    def ingest_income(self, records):
        with self.db:
            for item in records:
                for key in ("incomeType", "tranId", "time", "asset", "income"):
                    if item.get(key) in (None, ""):
                        raise ValueError(f"Income record missing {key}")
                decimal(item["income"])
                previous = self.db.execute("SELECT raw FROM income WHERE scope=? AND income_type=? AND transaction_id=?",
                                           (self.scope, item["incomeType"], str(item["tranId"]))).fetchone()
                if previous:
                    existing = json.loads(previous[0])
                    for key in ("income", "asset", "time", "symbol"):
                        old, new = existing.get(key, ""), item.get(key, "")
                        same = decimal(old) == decimal(new) if key in {"income", "time"} else str(old) == str(new)
                        if not same:
                            raise ValueError(f"Conflicting {key} for income {item['tranId']}; original evidence retained")
                self.db.execute("INSERT INTO income VALUES(?,?,?,?,?,?) ON CONFLICT(scope,income_type,transaction_id) DO UPDATE SET raw=excluded.raw",
                                (self.scope, item["incomeType"], str(item["tranId"]), item.get("symbol", ""), int(item["time"]), json_text(item)))

    def mark_coverage(self, stream, symbol, start, end):
        if end < start:
            return
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO coverage VALUES(?,?,?,?,?)", (self.scope, stream, symbol, start, end))

    def covered(self, stream, symbol, start, end):
        cursor = start
        for row in self.db.execute("SELECT start_ms,end_ms FROM coverage WHERE scope=? AND stream=? AND symbol=? ORDER BY start_ms",
                                   (self.scope, stream, symbol)):
            if row[0] > cursor:
                break
            cursor = max(cursor, row[1] + 1)
            if cursor > end:
                return True
        return False

    def event(self, row, event_id=None):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                            (self.scope, event_id or uuid.uuid4().hex, parse_ms(row.get("event_time")) or now_ms(), json_text(row)))

    def raw_rows(self, table):
        if table not in {"fills", "income", "events", "snapshots"}:
            raise ValueError("Unsupported evidence table")
        return [json.loads(r[0]) for r in self.db.execute(f"SELECT raw FROM {table} WHERE scope=? ORDER BY time_ms,rowid", (self.scope,))]

    def snapshot(self, balances, positions, timestamp=None):
        timestamp = timestamp or now_ms()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?)",
                            (self.scope, timestamp, json_text({"time": timestamp, "balances": balances, "positions": positions})))

    def order_fills(self, order):
        order_id = order["exchange_id"] if order["kind"] == "regular" else order["actual_id"]
        if not order_id:
            return []
        return [json.loads(r[0]) for r in self.db.execute("SELECT raw FROM fills WHERE scope=? AND symbol=? AND order_id=? ORDER BY time_ms,fill_id",
                                                         (self.scope, order["symbol"], order_id))]

    def exposure(self, trade_id):
        """Use confirmed order quantities when fill details lag; never requested size."""
        entry = exit_qty = ZERO
        uncertain = False
        for order in self.orders(trade_id):
            fills_qty = sum((decimal(f["qty"]) for f in self.order_fills(order)), ZERO)
            qty = max(decimal(order["executed_qty"] or 0), fills_qty)
            if order["role"] == "entry":
                entry += qty
                uncertain |= order["status"] not in TERMINAL
            else:
                exit_qty += qty
                uncertain |= order["status"] in {"UNKNOWN", "SUBMITTING", "TRIGGERING", "TRIGGERED", "FINISHED"}
                uncertain |= bool(order["actual_id"]) and order["status"] not in TERMINAL
                uncertain |= order["kind"] == "regular" and order["status"] not in TERMINAL
        return entry, exit_qty, uncertain

    def summary(self, trade_id):
        state = self.state(trade_id)
        orders = self.orders(trade_id)
        entries, exits = [], []
        reasons = []
        for order in orders:
            fills = self.order_fills(order)
            (entries if order["role"] == "entry" else exits).extend(fills)
            observed = sum((decimal(f["qty"]) for f in fills), ZERO)
            if order["executed_qty"] is not None and observed != decimal(order["executed_qty"]):
                reasons.append("fill_quantity_incomplete")
            if fills and order["executed_qty"] is None:
                reasons.append("order_quantity_unconfirmed")
        entered, exited, uncertain = self.exposure(trade_id)
        execution = "closed" if entered > 0 and entered == exited and not uncertain else "open" if entered > exited else "pending"
        if entered == 0 and orders and all(o["status"] in TERMINAL - {"FILLED"} and o["executed_qty"] is not None and decimal(o["executed_qty"]) == 0 for o in orders):
            execution = "rejected"
        if not orders and state.get("abandoned_before_submission"):
            execution = "rejected"
        if exited > entered:
            execution = "unresolved"
            reasons.append("exit_quantity_exceeds_entry")
        if state.get("execution_issue"):
            reasons.append(state["execution_issue"])
            if execution == "open":
                execution = "unresolved"
        if uncertain or (not orders and execution != "rejected"):
            reasons.append("order_status_unconfirmed")
        if execution not in {"closed", "rejected"}:
            reasons.append("position_not_closed")
        all_fills = entries + exits
        expected_side = "BUY" if state["direction"] == "long" else "SELL"
        if any(f["side"] != expected_side for f in entries) or any(f["side"] == expected_side for f in exits):
            reasons.append("fill_direction_mismatch")
        fee_totals = defaultdict(Decimal)
        fees_complete = bool(entries) and bool(exits) and "fill_quantity_incomplete" not in reasons
        for fill in all_fills:
            if fill.get("commission") in (None, "") or not fill.get("commissionAsset"):
                fees_complete = False
            else:
                fee_totals[fill["commissionAsset"]] += decimal(fill["commission"])
        if not fees_complete:
            reasons.append("commissions_pending")
        currency = state.get("pnl_asset", "USDT")
        if any(f.get("marginAsset") and f["marginAsset"] != currency for f in all_fills):
            reasons.append("realized_pnl_currency_conversion_pending")
        if any(asset != currency and amount != 0 for asset, amount in fee_totals.items()):
            reasons.append("commission_currency_conversion_pending")
        realized_complete = bool(exits) and all(f.get("realizedPnl") not in (None, "") for f in all_fills)
        gross = sum((decimal(f["realizedPnl"]) for f in all_fills), ZERO) if realized_complete else None
        if not realized_complete and execution != "rejected":
            reasons.append("realized_pnl_pending")
        first = min((int(f["time"]) for f in entries), default=0)
        last = max((int(f["time"]) for f in exits), default=0)
        funding = ZERO
        funding_complete = bool(first and last and execution == "closed" and self.covered("income", state["symbol"], first, last))
        if not funding_complete and execution != "rejected":
            reasons.append("funding_pending")
        if first and last:
            if not self.covered("fills", state["symbol"], first, last):
                reasons.append("trade_history_coverage_pending")
            # Only attribute account funding when this bot trade is the sole known exposure.
            all_orders = self.orders()
            owned_ids = {o["exchange_id"] if o["kind"] == "regular" else o["actual_id"] for o in orders}
            for fill in self.raw_rows("fills"):
                if fill["symbol"] == state["symbol"] and first <= int(fill["time"]) <= last and str(fill["orderId"]) not in owned_ids:
                    reasons.append("other_orders_during_trade")
            for other in self.trades(state["symbol"]):
                if other["trade_id"] == trade_id:
                    continue
                other_entries = [f for o in all_orders if o["trade_id"] == other["trade_id"] and o["role"] == "entry" for f in self.order_fills(o)]
                other_exits = [f for o in all_orders if o["trade_id"] == other["trade_id"] and o["role"] != "entry" for f in self.order_fills(o)]
                if other_entries:
                    other_start = min(int(f["time"]) for f in other_entries)
                    a, b, _ = self.exposure(other["trade_id"])
                    other_end = max((int(f["time"]) for f in other_exits), default=now_ms()) if a == b else now_ms()
                    if other_start < last and other_end > first:
                        reasons.append("overlapping_trades")
            for flow in self.raw_rows("income"):
                if flow["incomeType"] == "FUNDING_FEE" and flow.get("symbol") == state["symbol"] and first < int(flow["time"]) <= last:
                    if flow["asset"] != currency:
                        reasons.append("funding_currency_conversion_pending")
                    else:
                        funding += decimal(flow["income"])
        reasons = sorted(set(reasons))
        complete = not reasons and execution == "closed"
        def avg(fills):
            qty = sum((decimal(f["qty"]) for f in fills), ZERO)
            return str(sum((decimal(f["qty"]) * decimal(f["price"]) for f in fills), ZERO) / qty) if qty else ""
        entry_fee = sum((decimal(f["commission"]) for f in entries if f.get("commissionAsset") == currency and f.get("commission") not in (None, "")), ZERO)
        exit_fee = sum((decimal(f["commission"]) for f in exits if f.get("commissionAsset") == currency and f.get("commission") not in (None, "")), ZERO)
        net = gross - entry_fee - exit_fee + funding if complete else None
        risk = abs(decimal(avg(entries)) - decimal(state.get("sl_price") or avg(entries))) * entered if entries else ZERO
        exit_roles = [(int(f["time"]), o["role"]) for o in orders if o["role"] != "entry" for f in self.order_fills(o)]
        exit_reason = state.get("exit_reason") or (max(exit_roles)[1] if exit_roles else "")
        config = state.get("settings", {})
        entry_expected = state.get("entry_price_expected")
        exit_expected = state.get("exit_price_expected")
        if not exit_expected:
            exit_expected = state.get("sl_price") if exit_reason == "stop_loss" else state.get("tp_price") if exit_reason == "take_profit" else None
        return {**state, "scope": self.scope, "execution_status": execution, "exit_reason": exit_reason,
                "strategy_name": config.get("strategy_name", state.get("strategy_name", "")),
                "strategy_version": config.get("strategy_version", state.get("strategy_version", "")),
                "interval": config.get("interval", state.get("interval", "")),
                "stop_loss_price": state.get("sl_price", ""), "take_profit_price": state.get("tp_price", ""),
                "accounting_status": "complete" if complete else "not_applicable" if execution == "rejected" else "pending",
                "accounting_issues": "|".join(reasons), "entry_quantity": str(entered), "exit_quantity": str(exited),
                "position_size": str(entered), "entry_price_filled": avg(entries), "exit_price_filled": avg(exits),
                "slippage_entry": str(decimal(avg(entries)) - decimal(entry_expected)) if entries and entry_expected else "",
                "slippage_exit": str(decimal(avg(exits)) - decimal(exit_expected)) if exits and exit_expected else "",
                "entry_exec_time": iso_ms(first), "exit_exec_time": iso_ms(last),
                "gross_pnl": str(gross) if gross is not None else "", "fees_by_asset": json_text(dict(fee_totals)),
                "fee_entry": str(entry_fee) if fees_complete and len(set(fee_totals) - {currency}) == 0 else "",
                "fee_exit": str(exit_fee) if fees_complete and len(set(fee_totals) - {currency}) == 0 else "",
                "funding_pnl": str(funding) if funding_complete else "", "net_pnl": str(net) if net is not None else "",
                "r_multiple": str(net / risk) if net is not None and risk else ""}

    def export(self, trade_path, event_path=None, trade_limit=0, event_limit=0):
        rows = [self.summary(t["trade_id"]) for t in self.trades()]
        for row in rows:
            for key, value in list(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = json_text(value)
        headers = ["trade_id", "scope", "symbol", "direction", "execution_status", "accounting_status", "accounting_issues", "net_pnl"]
        headers += [k for row in rows for k in row if k not in headers]
        self._protect_existing_report(trade_path)
        if event_path:
            self._protect_existing_report(event_path)
        atomic_csv(trade_path, rows, list(dict.fromkeys(headers)), trade_limit)
        if event_path:
            atomic_csv(event_path, [{**r, "scope": self.scope} for r in self.raw_rows("events")],
                       ["event_time", "trade_id", "event_type", "symbol", "side", "price", "details", "scope"], event_limit)

    def _protect_existing_report(self, path):
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if "scope" in (reader.fieldnames or []):
                if any(row.get("scope") != self.scope for row in reader):
                    raise ValueError(f"Report belongs to another account scope: {path}; choose a separate report path")
                return
        original = path.read_bytes()
        archive = path.with_name(path.stem + ".legacy-" + hashlib.sha256(original).hexdigest()[:12] + ".csv")
        if not archive.exists():
            with archive.open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())

    def backup(self, path):
        path = Path(path)
        if path.resolve() == self.path.resolve():
            raise ValueError("Backup must use a different file")
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as destination:
            self.db.backup(destination)

    def import_legacy(self, path, kind):
        """Preserve original rows; never promote legacy aggregates to verified fills."""
        path = Path(path)
        with path.open(newline="") as stream, self.db:
            for row in csv.DictReader(stream):
                raw = json_text(row)
                key = hashlib.sha256((kind + raw).encode()).hexdigest()
                self.db.execute("INSERT OR IGNORE INTO legacy VALUES(?,?,?,?,?)",
                                (self.scope, key, kind, str(path.resolve()), raw))

    def legacy_report(self):
        records = [dict(r) for r in self.db.execute("SELECT * FROM legacy WHERE scope=? ORDER BY rowid", (self.scope,))]
        trades = [json.loads(r["raw"]) for r in records if r["kind"] == "trades"]
        events = [json.loads(r["raw"]) for r in records if r["kind"] == "events"]
        closed_ids = {r.get("trade_id") for r in trades}
        result = [{**r, "net_pnl": "", "legacy_reported_net_pnl": r.get("net_pnl", ""),
                   "accounting_status": "legacy_unverified", "accounting_issues": "exchange_fills_and_funding_not_verified"} for r in trades]
        seen = set()
        for event in events:
            trade_id = event.get("trade_id", "")
            if event.get("event_type") == "entry_exec" and trade_id not in closed_ids and trade_id not in seen:
                result.append({**event, "accounting_status": "unresolved", "accounting_issues": "entry_without_recorded_exit"})
                seen.add(trade_id)
            match = re.search(r"flatten_order_id=(\d+)", event.get("details", ""))
            if match:
                result.append({**event, "exit_order_id": match[1], "accounting_status": "unresolved",
                               "accounting_issues": "cleanup_pnl_and_fees_missing"})
        return result
