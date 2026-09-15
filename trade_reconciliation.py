"""Read-only exchange reconciliation plus a write-ahead journal for bot orders."""

import json
from decimal import Decimal

from trade_ledger import TERMINAL, ZERO, decimal, iso_ms, now_ms

DAY = 86_400_000


class OrderOutcomeUnknown(RuntimeError):
    pass


class OrderJournal:
    def __init__(self, ledger, client):
        self.ledger = ledger
        self.client = client

    def submit(self, trade_id, role, request, kind="regular"):
        client_id = self.ledger.intent(trade_id, role, kind, request)
        payload = dict(request)
        payload["newClientOrderId" if kind == "regular" else "clientAlgoId"] = client_id
        # Never retry a POST after an ambiguous response. Reconciliation queries this ID.
        try:
            if kind == "regular":
                response = self.client.new_order(**payload)
            else:
                response = self.client.sign_request("POST", "/fapi/v1/algoOrder", payload)
        except Exception as exc:
            code = getattr(exc, "error_code", None)
            status_code = getattr(exc, "status_code", 0) or 0
            rejected = 400 <= status_code < 500 and code not in {-1000, -1001, -1006, -1007, -1008}
            self.ledger.order_error(client_id, "REJECTED" if rejected else "UNKNOWN", str(exc))
            self.ledger.event({"event_time": iso_ms(now_ms()), "trade_id": trade_id,
                               "event_type": "order_rejected" if rejected else "order_unknown",
                               "symbol": request["symbol"], "details": f"{role} {client_id}: {exc}"})
            if rejected:
                raise
            raise OrderOutcomeUnknown(f"{role} outcome unknown; reconcile client ID {client_id}") from exc
        id_field = "orderId" if kind == "regular" else "algoId"
        if not response.get(id_field):
            self.ledger.order_error(client_id, "UNKNOWN", f"Response missing {id_field}")
            raise OrderOutcomeUnknown(f"{role} response missing {id_field}; reconcile {client_id}")
        self.ledger.acknowledge(client_id, response)
        self.ledger.event({"event_time": iso_ms(now_ms()), "trade_id": trade_id,
                           "event_type": "order_ack", "symbol": request["symbol"],
                           "details": f"{role} {id_field}={response[id_field]} client_id={client_id}"})
        return response


class Reconciler:
    def __init__(self, ledger, client, symbol, settlement_delay_seconds=60):
        self.ledger = ledger
        self.client = client
        self.symbol = symbol
        self.settlement_delay_ms = settlement_delay_seconds * 1000
        self.errors = []
        self.positions = []
        self.ready = False

    def _error(self, context, exc):
        message = f"{context}: {exc}"
        self.errors.append(message)
        self.ledger.event({"event_time": iso_ms(now_ms()), "event_type": "reconciliation_error",
                           "symbol": self.symbol, "details": message})

    def _query_order(self, order):
        if order["kind"] == "regular":
            key = {"orderId": order["exchange_id"]} if order["exchange_id"] else {"origClientOrderId": order["client_id"]}
            result = self.client.query_order(symbol=order["symbol"], **key)
            self.ledger.acknowledge(order["client_id"], result)
        else:
            key = {"algoId": order["exchange_id"]} if order["exchange_id"] else {"clientAlgoId": order["client_id"]}
            result = self.client.sign_request("GET", "/fapi/v1/algoOrder", key)
            # Persist the actual ID even if the next query fails.
            self.ledger.acknowledge(order["client_id"], result)
            actual_id = result.get("actualOrderId") or order["actual_id"]
            if actual_id:
                actual = self.client.query_order(symbol=order["symbol"], orderId=actual_id)
                self.ledger.acknowledge(order["client_id"], result, actual)

    def _fills_window(self, start, end):
        records = self.client.get_account_trades(symbol=self.symbol, startTime=start, endTime=end, limit=1000)
        seen = set()
        while True:
            in_window = [r for r in records if start <= int(r["time"]) <= end]
            self.ledger.ingest_fills(in_window)
            if len(records) < 1000 or any(int(r["time"]) > end for r in records):
                break
            cursor = max(int(r["id"]) for r in records) + 1
            if cursor in seen:
                raise ValueError("Trade-history pagination did not advance")
            seen.add(cursor)
            records = self.client.get_account_trades(symbol=self.symbol, fromId=cursor, limit=1000)
        self.ledger.mark_coverage("fills", self.symbol, start, end)

    def _income_window(self, start, end):
        page = 1
        seen = set()
        while True:
            records = self.client.get_income_history(startTime=start, endTime=end, page=page, limit=1000)
            signature = tuple((r["incomeType"], str(r["tranId"])) for r in records)
            if signature and signature in seen:
                raise ValueError("Income-history pagination did not advance")
            seen.add(signature)
            self.ledger.ingest_income(records)
            if len(records) < 1000:
                break
            page += 1
        self.ledger.mark_coverage("income", self.symbol, start, end)

    def sync(self, timestamp=None, start_ms=None):
        timestamp = timestamp or now_ms()
        self.errors = []
        self.ready = False
        for trade in self.ledger.trades(self.symbol):
            if not self.ledger.orders(trade["trade_id"]):
                state = self.ledger.state(trade["trade_id"])
                state["abandoned_before_submission"] = True
                self.ledger.save_state(state)
        for order in self.ledger.orders():
            if order["symbol"] != self.symbol:
                continue
            observed = sum((decimal(f["qty"]) for f in self.ledger.order_fills(order)), ZERO)
            # Finished orders need no retention-limited lookup once all their fills are saved.
            if order["status"] in TERMINAL and order["executed_qty"] is not None and observed == decimal(order["executed_qty"]):
                continue
            try:
                self._query_order(order)
            except Exception as exc:
                self._error(f"order {order['client_id']} remains unconfirmed", exc)

        cursor_key = "sync_cursor:" + self.symbol
        cursor = self.ledger.setting(cursor_key)
        earliest = min((r["created_ms"] for r in self.ledger.trades(self.symbol)), default=timestamp)
        requested_start = start_ms if start_ms is not None else cursor - DAY if cursor else min(earliest - 60_000, timestamp - DAY)
        if start_ms is None:
            repairable = {"fill_quantity_incomplete", "commissions_pending", "realized_pnl_pending",
                          "funding_pending", "trade_history_coverage_pending"}
            for trade in self.ledger.trades(self.symbol):
                summary = self.ledger.summary(trade["trade_id"])
                if summary["execution_status"] == "closed" and repairable.intersection(summary["accounting_issues"].split("|")):
                    requested_start = min(requested_start, trade["created_ms"] - 60_000)
        # Current ordinary endpoints retain three months. Never certify older empty responses.
        start = max(requested_start, timestamp - 89 * DAY)
        if requested_start < start and start_ms is not None:
            self._error("history coverage", "Older records need an exchange historical export")
        end = timestamp - self.settlement_delay_ms
        history_ok = True
        while start <= end:
            window_end = min(end, start + 7 * DAY - 1)
            for label, fetch in (("fills", self._fills_window), ("income", self._income_window)):
                try:
                    fetch(start, window_end)
                except Exception as exc:
                    history_ok = False
                    self._error(label, exc)
            if not history_ok:
                break
            start = window_end + 1
        if history_ok and end >= requested_start:
            self.ledger.set_setting(cursor_key, end)

        # Fetch fresh fills for active orders without certifying funding as settled yet.
        for order in self.ledger.orders():
            if order["symbol"] != self.symbol:
                continue
            order_id = order["exchange_id"] if order["kind"] == "regular" else order["actual_id"]
            observed = sum((decimal(f["qty"]) for f in self.ledger.order_fills(order)), ZERO)
            if order_id and (order["executed_qty"] is None or observed != decimal(order["executed_qty"]) or order["status"] not in TERMINAL):
                try:
                    records = self.client.get_account_trades(symbol=self.symbol, orderId=order_id, limit=1000)
                    self.ledger.ingest_fills(records)
                    # The window scan above handles pagination; no coverage inferred here.
                except Exception as exc:
                    self._error(f"fills for order {order_id}", exc)
        try:
            self.positions = self.client.get_position_risk(symbol=self.symbol)
            balances = self.client.balance()
            self.ledger.snapshot(balances, self.positions, timestamp)
            self.ready = True
            expected = ZERO
            active = []
            for trade in self.ledger.trades(self.symbol):
                entered, exited, _ = self.ledger.exposure(trade["trade_id"])
                state = self.ledger.state(trade["trade_id"])
                if entered > exited:
                    expected += (entered - exited) * (1 if state["direction"] == "long" else -1)
                    active.append(state)
                elif entered == exited and state.get("execution_issue") == "exchange_position_mismatch":
                    state.pop("execution_issue")
                    self.ledger.save_state(state)
            mismatch = self.position_amount() != expected
            for state in active:
                if mismatch:
                    state["execution_issue"] = "exchange_position_mismatch"
                else:
                    state.pop("execution_issue", None)
                self.ledger.save_state(state)
            if mismatch:
                self._error("position reconciliation", f"tracked={expected}, exchange={self.position_amount()}; order ownership requires review")
        except Exception as exc:
            self._error("account snapshot", exc)
        self.ledger.set_setting("last_reconciliation:" + self.symbol,
                                {"time": timestamp, "errors": self.errors, "snapshot_available": self.ready})
        return self.errors

    def active_states(self):
        result = []
        for trade in self.ledger.trades(self.symbol):
            summary = self.ledger.summary(trade["trade_id"])
            if summary["execution_status"] not in {"closed", "rejected"}:
                state = self.ledger.state(trade["trade_id"])
                orders = self.ledger.orders(trade["trade_id"])
                for order in orders:
                    prefix = {"entry": "entry", "stop_loss": "stop", "take_profit": "tp"}.get(order["role"])
                    if prefix and order["exchange_id"]:
                        state[prefix + "_order_id"] = order["exchange_id"]
                        state[prefix + "_order_kind"] = order["kind"]
                    if order["role"] == "entry":
                        response = json.loads(order["response"])
                        price = response.get("avgPrice")
                        if price and decimal(price) > 0:
                            state["entry_price"] = float(price)
                if summary["entry_price_filled"]:
                    state["entry_price"] = float(summary["entry_price_filled"])
                state["size"] = float(decimal(summary["entry_quantity"]) - decimal(summary["exit_quantity"]))
                self.ledger.save_state(state)
                result.append(state)
        return result

    def position_amount(self):
        return sum((decimal(p.get("positionAmt", 0)) for p in self.positions if p.get("symbol") == self.symbol), ZERO)

    def entry_block_reason(self):
        if not self.ready:
            return "exchange position has not been reconciled"
        if any(p.get("positionSide", "BOTH") != "BOTH" for p in self.positions if p.get("symbol") == self.symbol):
            return "accounting currently supports one-way positions only"
        if any(self.ledger.summary(t["trade_id"])["execution_status"] not in {"closed", "rejected"} for t in self.ledger.trades(self.symbol)):
            return "a tracked trade is open or its execution is unresolved"
        if any(o["symbol"] == self.symbol and o["status"] not in TERMINAL for o in self.ledger.orders()):
            return "an earlier order still needs cancellation or reconciliation"
        if self.position_amount() != 0:
            return "exchange position is not assigned to a tracked trade"
        return ""
