import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trade_ledger import TradeLedger, decimal, now_ms
from trade_reconciliation import OrderJournal, OrderOutcomeUnknown, Reconciler


class APIError(Exception):
    status_code = 400
    error_code = -2021


class FakeExchange:
    """An exchange that executes before returning; all tests stay offline."""
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.orders = {}
        self.algos = {}
        self.fills = []
        self.income = []
        self.qty = decimal(0)
        self.calls = []
        self.fail_protection = False
        self.crash_entry = False
        self.timeout_entry = False
        self.fail_history = False
        self.price = decimal(100)
        self.entry_price = decimal(100)
        self.next_id = 10

    def new_order(self, **payload):
        self.calls.append(("POST", dict(payload)))
        self.next_id += 1
        order_id = str(self.next_id)
        qty = decimal(payload["quantity"])
        pnl = (self.price - self.entry_price) * qty * (1 if self.qty > 0 else -1) if payload.get("reduceOnly") else decimal(0)
        if not payload.get("reduceOnly"):
            self.entry_price = self.price
        self.qty += qty if payload["side"] == "BUY" else -qty
        response = {"symbol": payload["symbol"], "orderId": order_id, "clientOrderId": payload["newClientOrderId"],
                    "status": "FILLED", "executedQty": str(qty), "avgPrice": str(self.price), "time": self.timestamp}
        self.orders[order_id] = response
        self.fills.append({"symbol": payload["symbol"], "id": self.next_id, "orderId": order_id,
                           "time": self.timestamp, "qty": str(qty), "price": str(self.price),
                           "side": payload["side"], "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": str(pnl)})
        if not payload.get("reduceOnly") and self.crash_entry:
            self.crash_entry = False
            raise SystemExit("process died after exchange fill")
        if not payload.get("reduceOnly") and self.timeout_entry:
            self.timeout_entry = False
            raise TimeoutError("response lost")
        return response

    def query_order(self, symbol, orderId=None, origClientOrderId=None):
        for order in self.orders.values():
            if str(order["orderId"]) == str(orderId) or order["clientOrderId"] == origClientOrderId:
                return dict(order)
        raise APIError("Order not found")

    def cancel_order(self, symbol, orderId=None, origClientOrderId=None):
        response = self.query_order(symbol, orderId, origClientOrderId)
        if response["status"] != "FILLED":
            response["status"] = "CANCELED"
            self.orders[str(response["orderId"])] = response
        return response

    def sign_request(self, method, path, payload):
        if method == "POST":
            self.calls.append(("POST_ALGO", dict(payload)))
            if self.fail_protection:
                raise APIError("Order would immediately trigger")
            self.next_id += 1
            response = {"symbol": payload["symbol"], "algoId": str(self.next_id),
                        "clientAlgoId": payload["clientAlgoId"], "algoStatus": "NEW"}
            self.algos[str(self.next_id)] = response
            return response
        for order in self.algos.values():
            if str(order["algoId"]) == str(payload.get("algoId")) or order["clientAlgoId"] == payload.get("clientAlgoId"):
                if method == "DELETE":
                    self.calls.append(("DELETE_ALGO", dict(payload)))
                    order["algoStatus"] = "CANCELED"
                return dict(order)
        raise APIError("Algo not found")

    def get_account_trades(self, symbol, **params):
        if self.fail_history:
            raise TimeoutError("history unavailable")
        rows = [f for f in self.fills if f["symbol"] == symbol]
        if "orderId" in params:
            rows = [f for f in rows if str(f["orderId"]) == str(params["orderId"])]
        if "startTime" in params:
            rows = [f for f in rows if params["startTime"] <= f["time"] <= params["endTime"]]
        if "fromId" in params:
            rows = [f for f in rows if int(f["id"]) >= int(params["fromId"])]
        return rows[:params.get("limit", 1000)]

    def get_income_history(self, **params):
        if self.fail_history:
            raise TimeoutError("income unavailable")
        rows = [f for f in self.income if params["startTime"] <= f["time"] <= params["endTime"]]
        start = (params.get("page", 1) - 1) * params.get("limit", 1000)
        return rows[start:start + params.get("limit", 1000)]

    def get_position_risk(self, symbol):
        return [{"symbol": symbol, "positionSide": "BOTH", "positionAmt": str(self.qty)}]

    def balance(self):
        return [{"asset": "USDT", "balance": "100"}]


class LedgerFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.ledger = TradeLedger(self.path, "test-account")
        self.timestamp = now_ms() - 600_000

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def trade(self):
        return self.ledger.create_trade("DOGEUSDT", "long", {"sl_price": "90", "pnl_asset": "USDT"})["trade_id"]

    def fill(self, id, order_id, side, qty="1", price="100", pnl="0", fee="0.1", asset="USDT", timestamp=None):
        return {"symbol": "DOGEUSDT", "id": id, "orderId": order_id, "side": side,
                "qty": qty, "price": price, "realizedPnl": pnl, "commission": fee,
                "commissionAsset": asset, "time": timestamp or self.timestamp}

    def order(self, trade_id, role, order_id, qty="1", status="FILLED"):
        client_id = self.ledger.intent(trade_id, role, "regular", {})
        self.ledger.acknowledge(client_id, {"orderId": order_id, "status": status, "executedQty": qty})
        return client_id

    def round_trip(self, exit_fee="0.1", exit_asset="USDT"):
        tid = self.trade()
        self.order(tid, "entry", "1")
        self.order(tid, "exit", "2")
        fills = [self.fill(1, "1", "BUY"), self.fill(2, "2", "SELL", price="110", pnl="10", fee=exit_fee, asset=exit_asset, timestamp=self.timestamp + 1000)]
        self.ledger.ingest_fills(fills)
        self.coverage()
        return tid, fills

    def coverage(self):
        for stream in ("income", "fills"):
            self.ledger.mark_coverage(stream, "DOGEUSDT", self.timestamp - 1, self.timestamp + 10_000)


class LedgerTests(LedgerFixture):
    def test_actual_pnl_fees_and_signed_funding_reconcile(self):
        tid, _ = self.round_trip()
        flow = {"incomeType": "FUNDING_FEE", "tranId": "1", "symbol": "DOGEUSDT", "time": self.timestamp + 500, "asset": "USDT", "income": "-0.5"}
        self.ledger.ingest_income([flow, flow])
        # Commission/realized-PnL income are retained as evidence but never counted twice.
        self.ledger.ingest_income([{**flow, "incomeType": "COMMISSION", "income": "-0.2"},
                                   {**flow, "incomeType": "TRANSFER", "income": "1000"}])
        result = self.ledger.summary(tid)
        self.assertEqual(result["accounting_status"], "complete")
        self.assertEqual(decimal(result["net_pnl"]), decimal("9.3"))
        self.assertEqual(len(self.ledger.raw_rows("income")), 3)

    def test_duplicate_fill_imports_survive_restart(self):
        tid, fills = self.round_trip()
        self.ledger.ingest_fills(fills)
        self.ledger.close()
        self.ledger = TradeLedger(self.path, "test-account")
        self.ledger.ingest_fills(fills)
        self.assertEqual(len(self.ledger.raw_rows("fills")), 2)
        self.assertEqual(decimal(self.ledger.summary(tid)["net_pnl"]), decimal("9.8"))

    def test_partial_exit_cannot_finalize_full_position(self):
        tid = self.trade()
        self.order(tid, "entry", "1", "2")
        exit_id = self.order(tid, "exit", "2", "1", "PARTIALLY_FILLED")
        self.ledger.ingest_fills([self.fill(1, "1", "BUY", "2"), self.fill(2, "2", "SELL", "1", pnl="1")])
        self.coverage()
        self.assertEqual(self.ledger.summary(tid)["execution_status"], "open")
        self.assertEqual(self.ledger.summary(tid)["net_pnl"], "")
        self.ledger.acknowledge(exit_id, {"orderId": "2", "status": "FILLED", "executedQty": "2"})
        self.assertIn("fill_quantity_incomplete", self.ledger.summary(tid)["accounting_issues"])
        self.ledger.ingest_fills([self.fill(3, "2", "SELL", "1", pnl="2")])
        self.assertEqual(self.ledger.summary(tid)["accounting_status"], "complete")

    def test_missing_commission_is_pending_not_zero(self):
        tid, fills = self.round_trip(exit_fee=None)
        result = self.ledger.summary(tid)
        self.assertEqual(result["execution_status"], "closed")
        self.assertEqual(result["net_pnl"], "")
        self.assertEqual(result["fee_exit"], "")
        self.assertIn("commissions_pending", result["accounting_issues"])
        fills[1]["commission"] = "0.1"
        self.ledger.ingest_fills(fills)
        self.assertEqual(self.ledger.summary(tid)["accounting_status"], "complete")

    def test_foreign_fee_currency_is_preserved_without_assumed_conversion(self):
        tid, fills = self.round_trip(exit_asset="BNB")
        result = self.ledger.summary(tid)
        self.assertEqual(result["net_pnl"], "")
        self.assertIn('"BNB"', result["fees_by_asset"])
        self.assertIn("commission_currency_conversion_pending", result["accounting_issues"])

    def test_conflicting_duplicate_does_not_overwrite_evidence(self):
        tid, fills = self.round_trip()
        fills[1]["qty"] = "2"
        with self.assertRaises(ValueError):
            self.ledger.ingest_fills(fills)
        self.assertEqual(decimal(self.ledger.summary(tid)["net_pnl"]), decimal("9.8"))

    def test_old_report_is_archived_and_other_account_report_is_protected(self):
        self.round_trip()
        report = Path(self.temp.name) / "trades.csv"
        original = b"trade_id,net_pnl\nold,1.25\n"
        report.write_bytes(original)
        self.ledger.export(report)
        self.assertEqual(next(report.parent.glob("trades.legacy-*.csv")).read_bytes(), original)
        other = TradeLedger(self.path, "other-account")
        try:
            saved = report.read_bytes()
            with self.assertRaises(ValueError):
                other.export(report)
            self.assertEqual(saved, report.read_bytes())
        finally:
            other.close()

    def test_duplicate_actual_order_link_is_rejected(self):
        tid = self.trade()
        self.order(tid, "exit", "30")
        algo = self.ledger.intent(tid, "take_profit", "algo", {})
        with self.assertRaises(ValueError):
            self.ledger.acknowledge(algo, {"algoId": "50", "actualOrderId": "30", "algoStatus": "FINISHED"})

    def test_funding_and_fill_coverage_must_be_complete(self):
        tid, _ = self.round_trip()
        with self.ledger.db:
            self.ledger.db.execute("DELETE FROM coverage")
        self.ledger.mark_coverage("income", "DOGEUSDT", self.timestamp, self.timestamp + 100)
        self.ledger.mark_coverage("income", "DOGEUSDT", self.timestamp + 102, self.timestamp + 1000)
        result = self.ledger.summary(tid)
        self.assertIn("funding_pending", result["accounting_issues"])
        self.assertIn("trade_history_coverage_pending", result["accounting_issues"])

    def test_unassigned_activity_does_not_get_attributed_to_bot(self):
        tid, _ = self.round_trip()
        self.ledger.ingest_fills([self.fill(90, "unrelated", "BUY", timestamp=self.timestamp + 500)])
        result = self.ledger.summary(tid)
        self.assertIn("other_orders_during_trade", result["accounting_issues"])
        self.assertEqual(result["net_pnl"], "")

    def test_account_scope_and_order_ownership(self):
        tid, fills = self.round_trip()
        other = TradeLedger(self.path, "other-account")
        try:
            other.ingest_fills(fills)
            self.assertEqual(other.trades(), [])
            self.assertEqual(len(other.raw_rows("fills")), 2)
        finally:
            other.close()
        another = self.trade()
        with self.assertRaises(ValueError):
            self.ledger.link_order(another, "1", "entry")

    def test_backup_and_report_limit_do_not_truncate_evidence(self):
        tid, _ = self.round_trip()
        self.ledger.event({"event_type": "a"})
        self.ledger.event({"event_type": "b"})
        self.ledger.export(Path(self.temp.name) / "trades.csv", Path(self.temp.name) / "events.csv", 1, 1)
        self.assertEqual(len(self.ledger.raw_rows("events")), 2)
        destination = Path(self.temp.name) / "backup.sqlite3"
        self.ledger.backup(destination)
        restored = TradeLedger(destination, "test-account")
        try:
            self.assertEqual(restored.summary(tid)["net_pnl"], self.ledger.summary(tid)["net_pnl"])
        finally:
            restored.close()

    def test_writer_lock_excludes_second_process(self):
        self.ledger.acquire_writer_lock()
        other = TradeLedger(self.path, "test-account")
        try:
            with self.assertRaises(RuntimeError):
                other.acquire_writer_lock()
        finally:
            other.close()

    def test_legacy_import_is_idempotent_and_unverified(self):
        trades = Path(self.temp.name) / "old_trades.csv"
        events = Path(self.temp.name) / "old_events.csv"
        trades.write_text("trade_id,net_pnl\n1,10\n")
        events.write_text("trade_id,event_type,details\n2,entry_exec,entry\n,exchange_error,flatten_order_id=3\n")
        for _ in range(2):
            self.ledger.import_legacy(trades, "trades")
            self.ledger.import_legacy(events, "events")
        result = self.ledger.legacy_report()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["legacy_reported_net_pnl"], "10")
        self.assertEqual(result[0]["net_pnl"], "")
        self.assertEqual(self.ledger.trades(), [])


class ReconciliationTests(LedgerFixture):
    def test_short_trade_uses_exchange_realized_pnl(self):
        exchange = FakeExchange(self.timestamp)
        tid = self.ledger.create_trade("DOGEUSDT", "short", {"sl_price": "102"})["trade_id"]
        journal = OrderJournal(self.ledger, exchange)
        journal.submit(tid, "entry", {"symbol": "DOGEUSDT", "side": "SELL", "type": "MARKET", "quantity": "1"})
        exchange.price = decimal(95)
        exchange.timestamp += 1000
        journal.submit(tid, "exit", {"symbol": "DOGEUSDT", "side": "BUY", "type": "MARKET", "quantity": "1", "reduceOnly": "true"})
        Reconciler(self.ledger, exchange, "DOGEUSDT").sync()
        self.assertEqual(decimal(self.ledger.summary(tid)["net_pnl"]), decimal("4.8"))

    def test_timeout_is_queried_by_client_id_without_resubmission(self):
        exchange = FakeExchange(self.timestamp)
        exchange.timeout_entry = True
        tid = self.trade()
        journal = OrderJournal(self.ledger, exchange)
        with self.assertRaises(OrderOutcomeUnknown):
            journal.submit(tid, "entry", {"symbol": "DOGEUSDT", "side": "BUY", "type": "MARKET", "quantity": "1"})
        self.assertEqual(self.ledger.orders(tid)[0]["status"], "UNKNOWN")
        reconciler = Reconciler(self.ledger, exchange, "DOGEUSDT")
        reconciler.sync()
        self.assertEqual(self.ledger.orders(tid)[0]["status"], "FILLED")
        self.assertEqual(self.ledger.summary(tid)["execution_status"], "open")
        self.assertEqual(len(exchange.calls), 1)

    def test_crash_after_exchange_fill_is_recoverable(self):
        exchange = FakeExchange(self.timestamp)
        exchange.crash_entry = True
        tid = self.trade()
        with self.assertRaises(SystemExit):
            OrderJournal(self.ledger, exchange).submit(tid, "entry", {"symbol": "DOGEUSDT", "side": "BUY", "type": "MARKET", "quantity": "1"})
        self.ledger.close()
        self.ledger = TradeLedger(self.path, "test-account")
        reconciler = Reconciler(self.ledger, exchange, "DOGEUSDT")
        reconciler.sync()
        self.assertEqual(reconciler.active_states()[0]["trade_id"], tid)
        self.assertEqual(len(exchange.calls), 1)

    def test_pending_history_keeps_fees_and_net_unknown(self):
        exchange = FakeExchange(self.timestamp)
        tid = self.trade()
        journal = OrderJournal(self.ledger, exchange)
        journal.submit(tid, "entry", {"symbol": "DOGEUSDT", "side": "BUY", "type": "MARKET", "quantity": "1"})
        journal.submit(tid, "exit", {"symbol": "DOGEUSDT", "side": "SELL", "type": "MARKET", "quantity": "1", "reduceOnly": "true"})
        exchange.fail_history = True
        Reconciler(self.ledger, exchange, "DOGEUSDT").sync()
        summary = self.ledger.summary(tid)
        self.assertEqual(summary["net_pnl"], "")
        self.assertEqual(summary["fee_entry"], "")
        self.assertEqual(summary["accounting_status"], "pending")

    def test_full_history_pagination_is_idempotent(self):
        exchange = FakeExchange(self.timestamp)
        exchange.fills = [self.fill(i, str(i), "BUY") for i in range(1, 2002)]
        exchange.income = [{"incomeType": "TRANSFER", "tranId": str(i), "symbol": "", "time": self.timestamp, "asset": "USDT", "income": "1"} for i in range(2002)]
        reconciler = Reconciler(self.ledger, exchange, "DOGEUSDT")
        for _ in range(2):
            self.assertEqual(reconciler.sync(), [])
        self.assertEqual(len(self.ledger.raw_rows("fills")), 2001)
        self.assertEqual(len(self.ledger.raw_rows("income")), 2002)

    def test_no_submission_intent_can_be_abandoned_safely(self):
        tid = self.trade()
        reconciler = Reconciler(self.ledger, FakeExchange(self.timestamp), "DOGEUSDT")
        reconciler.sync()
        self.assertEqual(self.ledger.summary(tid)["execution_status"], "rejected")
        self.assertEqual(reconciler.entry_block_reason(), "")

    def test_untracked_exchange_position_blocks_entry(self):
        exchange = FakeExchange(self.timestamp)
        exchange.qty = decimal("1")
        reconciler = Reconciler(self.ledger, exchange, "DOGEUSDT")
        reconciler.sync()
        self.assertIn("not assigned", reconciler.entry_block_reason())


if __name__ == "__main__":
    unittest.main()
