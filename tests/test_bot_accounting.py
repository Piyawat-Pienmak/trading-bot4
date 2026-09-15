import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bot import FuturesBot, OrderPlacementError, Settings
from test_accounting import FakeExchange, LedgerFixture
from trade_ledger import decimal
from trade_reconciliation import OrderJournal, Reconciler


class BotAccountingTests(LedgerFixture):
    def make_bot(self, exchange):
        bot = FuturesBot.__new__(FuturesBot)
        bot.settings = Settings(symbol="DOGEUSDT", live=True, ledger_path=str(self.path),
                                trade_log=str(Path(self.temp.name) / "trades.csv"),
                                event_log=str(Path(self.temp.name) / "events.csv"))
        bot.client = exchange
        bot.ledger = self.ledger
        bot.journal = OrderJournal(self.ledger, exchange)
        bot.reconciler = Reconciler(self.ledger, exchange, "DOGEUSDT")
        bot._live_state = None
        bot._cooldown_left = 0
        return bot

    def test_failed_protection_records_entry_and_emergency_exit(self):
        exchange = FakeExchange(self.timestamp)
        exchange.fail_protection = True
        bot = self.make_bot(exchange)
        with self.assertRaises(OrderPlacementError):
            bot._place_orders(1, 1, 98, 103, 100, {"entry_signal_open_time": self.timestamp})
        tid = bot._live_state["trade_id"]
        self.assertEqual({o["role"] for o in self.ledger.orders(tid)}, {"entry", "stop_loss", "cleanup"})
        self.assertEqual(exchange.qty, 0)
        bot._process_live_exit()
        result = self.ledger.summary(tid)
        self.assertEqual(result["execution_status"], "closed")
        self.assertEqual(result["accounting_status"], "complete")
        self.assertEqual(decimal(result["net_pnl"]), decimal("-0.2"))
        self.assertEqual(result["exit_reason"], "entry_cleanup")

    def test_restart_after_entry_before_protection_recovers_and_closes(self):
        exchange = FakeExchange(self.timestamp)
        exchange.crash_entry = True
        bot = self.make_bot(exchange)
        with self.assertRaises(SystemExit):
            bot._place_orders(1, 1, 98, 103, 100)
        tid = bot._live_state["trade_id"]
        restarted = self.make_bot(exchange)
        restarted._process_live_exit()
        self.assertEqual(exchange.qty, 0)
        restarted._process_live_exit()
        self.assertEqual(self.ledger.summary(tid)["accounting_status"], "complete")
        self.assertEqual(len([c for c in exchange.calls if c[0] == "POST"]), 2)

    def test_strategy_exit_uses_same_fill_accounting(self):
        exchange = FakeExchange(self.timestamp)
        bot = self.make_bot(exchange)
        state = bot._place_orders(1, 1, 98, 103, 100)
        exchange.price = decimal(102)
        exchange.timestamp += 1000
        self.assertTrue(bot._market_exit_live_position("midline_exit", 102))
        bot._process_live_exit()
        result = self.ledger.summary(state["trade_id"])
        self.assertEqual(result["accounting_status"], "complete")
        self.assertEqual(decimal(result["net_pnl"]), decimal("1.8"))
        self.assertEqual(result["exit_reason"], "midline_exit")

    def test_algo_exit_while_offline_is_recovered_once(self):
        exchange = FakeExchange(self.timestamp)
        bot = self.make_bot(exchange)
        state = bot._place_orders(1, 1, 98, 103, 100)
        exchange.price = decimal(103)
        exchange.timestamp += 1000
        fill = exchange.new_order(symbol="DOGEUSDT", side="SELL", type="MARKET", quantity="1",
                                  reduceOnly="true", newClientOrderId="exchange-trigger")
        exchange.algos[state["tp_order_id"]].update({"algoStatus": "FINISHED", "actualOrderId": fill["orderId"]})
        restarted = self.make_bot(exchange)
        restarted._process_live_exit()
        restarted._process_live_exit()
        result = self.ledger.summary(state["trade_id"])
        self.assertEqual(result["accounting_status"], "complete")
        self.assertEqual(decimal(result["net_pnl"]), decimal("2.8"))
        exits = [r for r in self.ledger.raw_rows("events") if r["event_type"] == "exit_exec"]
        self.assertEqual(len(exits), 1)
        self.assertIsNone(restarted._live_state)

    def test_partial_entry_does_not_get_reported_as_closed(self):
        exchange = FakeExchange(self.timestamp)
        bot = self.make_bot(exchange)
        state = bot._place_orders(1, 2, 98, 103, 100)
        entry = exchange.orders[state["entry_order_id"]]
        entry.update({"status": "PARTIALLY_FILLED", "executedQty": "1"})
        exchange.fills[0]["qty"] = "1"
        exchange.qty = decimal(1)
        # Correct exchange evidence before the first reconciliation.
        order = self.ledger.orders(state["trade_id"])[0]
        self.ledger.acknowledge(order["client_id"], entry)
        bot._process_live_exit()
        self.assertFalse(bot._market_exit_live_position("midline_exit", 102))
        self.assertEqual(self.ledger.summary(state["trade_id"])["net_pnl"], "")

    def test_exit_timeout_prevents_duplicate_exit(self):
        exchange = FakeExchange(self.timestamp)
        bot = self.make_bot(exchange)
        state = bot._place_orders(1, 1, 98, 103, 100)
        original = exchange.new_order
        def lose_response(**kwargs):
            original(**kwargs)
            raise TimeoutError("exit response lost")
        with patch.object(exchange, "new_order", side_effect=lose_response):
            self.assertFalse(bot._market_exit_live_position("midline_exit", 102))
            self.assertFalse(bot._market_exit_live_position("midline_exit", 102))
        self.assertEqual(len([c for c in exchange.calls if c[0] == "POST"]), 2)
        bot._process_live_exit()
        self.assertEqual(self.ledger.summary(state["trade_id"])["execution_status"], "closed")

    def test_manual_close_is_unresolved_until_explicitly_linked(self):
        exchange = FakeExchange(self.timestamp)
        bot = self.make_bot(exchange)
        state = bot._place_orders(1, 1, 98, 103, 100)
        exchange.timestamp += 1000
        manual = exchange.new_order(symbol="DOGEUSDT", side="SELL", type="MARKET", quantity="1", reduceOnly="true", newClientOrderId="manual")
        bot._process_live_exit()
        self.assertEqual(self.ledger.summary(state["trade_id"])["execution_status"], "unresolved")
        self.assertTrue(bot.reconciler.entry_block_reason())
        self.ledger.link_order(state["trade_id"], manual["orderId"], "exit")
        bot._process_live_exit()
        result = self.ledger.summary(state["trade_id"])
        self.assertEqual(result["execution_status"], "closed")
        self.assertEqual(result["net_pnl"], "-0.2")

    def test_dry_run_does_not_create_ledger_or_submit_orders(self):
        exchange = FakeExchange(self.timestamp)
        with patch("bot.UMFutures", return_value=exchange), \
             patch.object(FuturesBot, "_run_auth_preflight", return_value=True), \
             patch.object(FuturesBot, "_fetch_filters", return_value={}):
            bot = FuturesBot(Settings(symbol="DOGEUSDT", live=False, ledger_path=str(Path(self.temp.name) / "dry.sqlite3"),
                                      trade_log=str(Path(self.temp.name) / "dry_trades.csv"), event_log=str(Path(self.temp.name) / "dry_events.csv"),
                                      signal_diagnostics_log=str(Path(self.temp.name) / "signals.csv")), "test-key", "test-secret")
        self.assertIsNone(bot.ledger)
        self.assertFalse((Path(self.temp.name) / "dry.sqlite3").exists())
        self.assertEqual(exchange.calls, [])

    def test_failed_entry_is_not_repeated_on_the_same_candle(self):
        exchange = FakeExchange(self.timestamp)
        exchange.fail_protection = True
        bot = self.make_bot(exchange)
        data = pd.DataFrame([{"open_time": self.timestamp - 3600_000, "close_time": self.timestamp - 1,
                              "open": 100, "high": 101, "low": 99, "close": 100, "atr": 1,
                              "signal": 1, "rolling_mean": 101, "upper_band": 103, "lower_band": 99,
                              "zscore": -1, "regime": "sideways"}])
        with patch.object(bot, "fetch_klines", return_value=data), \
             patch.object(bot, "_compute_indicators", side_effect=lambda frame: frame), \
             patch.object(bot, "_run_auth_preflight", return_value=True), \
             patch.object(bot, "_build_orders", return_value=(1, 98, 103)), \
             patch.object(bot, "_ensure_leverage"), \
             patch.object(bot, "_log_signal_diagnostic"), contextlib.redirect_stdout(io.StringIO()):
            bot.run_once(1, "test")
            bot.run_once(2, "test")
        self.assertEqual(len(self.ledger.trades()), 1)
        self.assertEqual(len([call for call in exchange.calls if call[0] == "POST"]), 2)

    def test_hedge_mode_exit_does_not_submit_an_order(self):
        exchange = FakeExchange(self.timestamp)
        bot = self.make_bot(exchange)
        bot._place_orders(1, 1, 98, 103, 100)
        before = len(exchange.calls)
        with patch.object(exchange, "get_position_risk", return_value=[{"symbol": "DOGEUSDT", "positionSide": "LONG", "positionAmt": "1"}]):
            self.assertFalse(bot._market_exit_live_position("midline_exit", 102))
        self.assertEqual(len(exchange.calls), before)


if __name__ == "__main__":
    unittest.main()
