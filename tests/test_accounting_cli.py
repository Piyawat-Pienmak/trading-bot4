import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from accounting import main


class AccountingCLITests(unittest.TestCase):
    def test_verified_historical_exports_can_complete_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = ["--ledger", str(root / "ledger.sqlite3"), "--scope", "verified-test-account"]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(base + ["adopt-entry", "--order-id", "10", "--symbol", "DOGEUSDT", "--direction", "long", "--time", "2025-01-01T00:00:00+00:00"])
            trade_id = stdout.getvalue().strip()
            main(base + ["link-order", "--trade-id", trade_id, "--order-id", "20", "--role", "exit"])
            orders = root / "orders.json"
            orders.write_text(json.dumps([{"symbol": "DOGEUSDT", "orderId": n, "status": "FILLED", "executedQty": "1"} for n in (10, 20)]))
            main(base + ["import-orders", str(orders)])
            fills = root / "fills.json"
            fills.write_text(json.dumps([
                {"symbol": "DOGEUSDT", "id": "1", "orderId": "10", "time": 1735689600000, "side": "BUY", "qty": "1", "price": "100", "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "0"},
                {"symbol": "DOGEUSDT", "id": "2", "orderId": "20", "time": 1735693200000, "side": "SELL", "qty": "1", "price": "101", "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "1"},
            ]))
            income = root / "income.json"
            income.write_text("[]")
            coverage = ["--complete-from", "2025-01-01T00:00:00+00:00", "--complete-through", "2025-01-01T01:00:00+00:00", "--symbol", "DOGEUSDT"]
            main(base + ["import-fills", str(fills)] + coverage)
            main(base + ["import-income", str(income)] + coverage)
            with contextlib.redirect_stdout(io.StringIO()):
                main(base + ["export", "--output-dir", str(root / "reports")])
            with (root / "reports" / "trades.csv").open() as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["accounting_status"], "complete")
            self.assertEqual(result["net_pnl"], "0.8")
            with (root / "reports" / "fills.csv").open() as stream:
                self.assertEqual({r["bot_trade_id"] for r in csv.DictReader(stream)}, {trade_id})


if __name__ == "__main__":
    unittest.main()
