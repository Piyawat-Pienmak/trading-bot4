import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import requests
import urllib3
from binance.error import ClientError
from binance.um_futures import UMFutures
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    symbol: str
    interval: str = "1h"
    lookback: int = 300
    atr_period: int = 14
    ema_period: int = 200
    breakout_lookback: int = 24
    stop_atr_mult: float = 2.0
    tp_rr: float = 3.0
    max_notional: float | None = None
    leverage: int = 3
    risk_pct: float = 0.01
    initial_equity: float = 25.0
    max_slippage_pct: float = 0.0015
    testnet: bool = False
    live: bool = False
    base_url: str | None = "https://fapi.binance.com"
    loop: bool = False
    poll_seconds: int = 300
    live_log: str = "reports/live_trades.csv"
    api_retries: int = 3
    api_retry_backoff: float = 2.0
    api_timeout: float = 10.0


class FuturesBot:
    def __init__(self, settings: Settings, api_key: str, api_secret: str) -> None:
        self.settings = settings
        base_url = settings.base_url
        if settings.testnet and not base_url:
            base_url = "https://testnet.binancefuture.com"
        if not base_url:
            base_url = "https://fapi.binance.com"
        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url=base_url,
            timeout=settings.api_timeout,
        )
        self.filters = self._fetch_filters(settings.symbol)
        self._live_state: dict | None = None
        self._equity_cache: float | None = None
        os.makedirs(os.path.dirname(self.settings.live_log) or ".", exist_ok=True)

    def _call_with_retries(self, func, *args, **kwargs):
        attempts = max(1, int(self.settings.api_retries))
        backoff = max(0.0, float(self.settings.api_retry_backoff))
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, urllib3.exceptions.ProtocolError) as exc:
                if attempt >= attempts:
                    raise
                delay = backoff ** (attempt - 1)
                delay = min(delay, 30.0)
                print(
                    f"[warn] API call failed ({exc}); retry {attempt}/{attempts} in {delay:.1f}s"
                )
                time.sleep(delay)

    def _fetch_filters(self, symbol: str) -> Dict[str, float]:
        info = self._call_with_retries(self.client.exchange_info)
        match = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
        if not match:
            raise ValueError(f"Symbol {symbol} not found on exchange")
        lot = next(f for f in match["filters"] if f["filterType"] == "LOT_SIZE")
        price = next(f for f in match["filters"] if f["filterType"] == "PRICE_FILTER")
        notional = next(
            f for f in match["filters"] if f["filterType"] == "MIN_NOTIONAL"
        )
        return {
            "step_size": float(lot["stepSize"]),
            "min_qty": float(lot["minQty"]),
            "tick_size": float(price["tickSize"]),
            "min_notional": float(notional["notional"]),
        }

    def fetch_klines(self) -> pd.DataFrame:
        """Paginate Binance klines to collect up to lookback candles (1500 max per call)."""
        target = max(1, int(self.settings.lookback))
        per_call = 1500
        rows: List[list] = []
        end_time = None

        while len(rows) < target:
            limit = min(per_call, target - len(rows))
            params = {
                "symbol": self.settings.symbol,
                "interval": self.settings.interval,
                "limit": limit,
            }
            if end_time is not None:
                params["endTime"] = end_time
            try:
                batch = self._call_with_retries(self.client.klines, **params)
            except Exception as exc:
                print(f"[error] failed to fetch klines after retries: {exc}")
                break
            if not batch:
                break
            rows.extend(batch)
            # Walk backward in time using earliest candle's open_time
            end_time = batch[0][0] - 1
            if len(batch) < limit:
                break

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "trades",
                "taker_base_vol",
                "taker_quote_vol",
                "ignore",
            ],
        )
        if df.empty:
            return df
        # chronological order
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        # drop last incomplete candle (close_time in future)
        now_ms = pd.Timestamp.utcnow().timestamp() * 1000
        if now_ms < float(df.iloc[-1]["close_time"]):
            df = df.iloc[:-1]
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema200"] = df["close"].ewm(span=self.settings.ema_period, adjust=False).mean()
        df["ema200_10"] = df["ema200"].shift(10)
        df["high_low"] = df["high"] - df["low"]
        df["prev_close"] = df["close"].shift(1)
        df["high_close"] = (df["high"] - df["prev_close"]).abs()
        df["low_close"] = (df["low"] - df["prev_close"]).abs()
        df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=self.settings.atr_period, min_periods=1).mean()
        df["hh24_prev"] = (
            df["high"]
            .rolling(window=self.settings.breakout_lookback, min_periods=self.settings.breakout_lookback)
            .max()
            .shift(1)
        )
        df["ll24_prev"] = (
            df["low"]
            .rolling(window=self.settings.breakout_lookback, min_periods=self.settings.breakout_lookback)
            .min()
            .shift(1)
        )
        return df

    def _signal_direction(self, row: pd.Series) -> int:
        direction, _ = self._signal_decision(row, explain=False)
        return direction

    def _signal_detail(self, row: pd.Series) -> Tuple[int, str]:
        direction, reason = self._signal_decision(row, explain=True)
        return direction, reason or ""

    def _signal_decision(self, row: pd.Series, explain: bool) -> Tuple[int, str | None]:
        price = float(row["close"])
        ema200 = float(row["ema200"])
        ema200_10 = float(row.get("ema200_10", float("nan")))
        atr = float(row["atr"])
        hh24_prev = float(row.get("hh24_prev", float("nan")))
        ll24_prev = float(row.get("ll24_prev", float("nan")))
        reason = None
        if (
            math.isnan(ema200)
            or math.isnan(ema200_10)
            or math.isnan(atr)
            or math.isnan(hh24_prev)
            or math.isnan(ll24_prev)
        ):
            return 0, "missing indicators" if explain else None
        long_ok = price > ema200 and ema200 > ema200_10 and price > hh24_prev
        short_ok = price < ema200 and ema200 < ema200_10 and price < ll24_prev
        bullish = long_ok
        bearish = short_ok
        if bullish:
            if explain:
                reason = (
                    f"LONG: close {price:.6f}>ema200 {ema200:.6f}, "
                    f"ema200>ema200_10 {ema200_10:.6f}, "
                    f"close {price:.6f}>hh24_prev {hh24_prev:.6f}"
                )
            return 1, reason
        if bearish:
            if explain:
                reason = (
                    f"SHORT: close {price:.6f}<ema200 {ema200:.6f}, "
                    f"ema200<ema200_10 {ema200_10:.6f}, "
                    f"close {price:.6f}<ll24_prev {ll24_prev:.6f}"
                )
            return -1, reason
        return 0, "conditions not aligned" if explain else None

    def _latest_signal(self, df: pd.DataFrame) -> Tuple[int, float, float]:
        last = df.iloc[-1]
        atr = float(last["atr"])
        price = float(last["close"])
        direction = self._signal_direction(last)
        return direction, price, atr

    def _order_trades(self, order_id: int) -> List[dict]:
        try:
            trades = self.client.get_account_trades(symbol=self.settings.symbol, orderId=order_id)
        except ClientError:
            return []
        return trades or []

    def _trades_avg_price_and_fee(self, trades: List[dict]) -> Tuple[float, float]:
        if not trades:
            return 0.0, 0.0
        total_qty = 0.0
        total_quote = 0.0
        total_fee = 0.0
        for t in trades:
            qty = float(t.get("qty", 0) or 0.0)
            price = float(t.get("price", 0) or 0.0)
            fee = float(t.get("commission", 0) or 0.0)
            total_qty += qty
            total_quote += qty * price
            total_fee += fee
        avg_price = (total_quote / total_qty) if total_qty > 0 else 0.0
        return avg_price, total_fee

    def _process_live_exit(self) -> None:
        if not self.settings.live or not self._live_state:
            return
        stop_id = self._live_state.get("stop_order_id")
        tp_id = self._live_state.get("tp_order_id")
        filled_id = None
        reason = None
        def _order_status(oid: int) -> str:
            try:
                res = self.client.query_order(symbol=self.settings.symbol, orderId=oid)
                return res.get("status", "")
            except ClientError:
                return ""
        if stop_id:
            status = _order_status(stop_id)
            if status == "FILLED":
                filled_id = stop_id
                reason = "hit_sl"
        if tp_id and filled_id is None:
            status = _order_status(tp_id)
            if status == "FILLED":
                filled_id = tp_id
                reason = "hit_tp"
        if filled_id is None:
            return
        trades = self._order_trades(filled_id)
        exit_price, exit_fee = self._trades_avg_price_and_fee(trades)
        direction = self._live_state.get("direction", "")
        entry_price = float(self._live_state.get("entry_price", 0.0) or 0.0)
        entry_fee = float(self._live_state.get("entry_fee", 0.0) or 0.0)
        size = float(self._live_state.get("size", 0.0) or 0.0)
        if exit_price <= 0:
            exit_price = entry_price
        gross_pnl = (exit_price - entry_price) * size if direction == "long" else (entry_price - exit_price) * size
        fees = entry_fee + exit_fee
        net_pnl = gross_pnl - fees
        equity_after = self._wallet_equity()
        self._equity_cache = equity_after
        self._append_log_row(
            {
                "timestamp_entry": self._live_state.get("timestamp_entry", ""),
                "timestamp_exit": pd.Timestamp.utcnow().isoformat(),
                "direction": direction.upper(),
                "entry_price": round(entry_price, 8),
                "exit_price": round(exit_price, 8),
                "size": size,
                "sl_price": self._live_state.get("sl_price", ""),
                "tp_price": self._live_state.get("tp_price", ""),
                "gross_pnl": round(gross_pnl, 8),
                "fees": round(fees, 8),
                "net_pnl": round(net_pnl, 8),
                "equity_after_trade": round(equity_after, 8),
                "reason_exit": reason or "",
            }
        )
        print(f"[live] exit detected ({reason}); logged trade with net PnL {net_pnl:.6f}")
        self._live_state = None

    def _wallet_equity(self) -> float:
        balances = self.client.balance()
        usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
        if not usdt:
            return 0.0
        return float(usdt.get("balance", 0.0) or 0.0)

    def _append_log_row(self, row: Dict[str, object]) -> None:
        path = self.settings.live_log
        header_needed = not os.path.exists(path)
        with open(path, "a", encoding="ascii") as f:
            if header_needed:
                f.write(
                    ",".join(
                        [
                            "timestamp_entry",
                            "timestamp_exit",
                            "direction",
                            "entry_price",
                            "exit_price",
                            "size",
                            "sl_price",
                            "tp_price",
                            "gross_pnl",
                            "fees",
                            "net_pnl",
                            "equity_after_trade",
                            "reason_exit",
                        ]
                    )
                    + "\n"
                )
            f.write(
                ",".join(
                    [
                        str(row.get("timestamp_entry", "")),
                        str(row.get("timestamp_exit", "")),
                        str(row.get("direction", "")),
                        f"{row.get('entry_price', '')}",
                        f"{row.get('exit_price', '')}",
                        f"{row.get('size', '')}",
                        f"{row.get('sl_price', '')}",
                        f"{row.get('tp_price', '')}",
                        f"{row.get('gross_pnl', '')}",
                        f"{row.get('fees', '')}",
                        f"{row.get('net_pnl', '')}",
                        f"{row.get('equity_after_trade', '')}",
                        str(row.get("reason_exit", "")),
                    ]
                )
                + "\n"
            )

    def _round_to(self, value: float, step: float) -> float:
        precision = max(int(round(-math.log(step, 10))) if step < 1 else 0, 0)
        return round(math.floor(value / step) * step, precision)

    def _round_up_to(self, value: float, step: float) -> float:
        precision = max(int(round(-math.log(step, 10))) if step < 1 else 0, 0)
        return round(math.ceil(value / step) * step, precision)

    def _account_equity(self) -> float:
        balances = self.client.balance()
        usdt = next((b for b in balances if b["asset"] == "USDT"), None)
        if not usdt:
            return self.settings.initial_equity
        # use wallet balance to include unrealized PnL, fallback to available
        wallet = float(usdt.get("balance", usdt.get("walletBalance", 0)))
        return max(wallet, self.settings.initial_equity)

    def _ensure_leverage(self) -> None:
        try:
            self.client.change_leverage(
                symbol=self.settings.symbol, leverage=self.settings.leverage
            )
        except ClientError as exc:
            print(f"[warn] leverage change failed: {exc}")

    def _size_position(
        self,
        direction: int,
        entry: float,
        stop_price: float,
        equity_override: float | None = None,
    ) -> Tuple[float, float]:
        risk_distance = abs(entry - stop_price)
        if risk_distance <= 0:
            raise ValueError("Stop equals entry; risk distance invalid")
        equity = equity_override if equity_override is not None else self._account_equity()
        risk_amount = equity * self.settings.risk_pct
        raw_qty = risk_amount / risk_distance
        qty = max(raw_qty, self.filters["min_qty"])
        qty = max(self._round_to(qty, self.filters["step_size"]), self.filters["min_qty"])
        notional = qty * entry
        if notional < self.filters["min_notional"]:
            min_qty = self.filters["min_notional"] / entry
            qty = self._round_up_to(max(min_qty, self.filters["min_qty"]), self.filters["step_size"])
            notional = qty * entry
        if self.settings.max_notional and notional > self.settings.max_notional:
            capped_qty = self._round_to(self.settings.max_notional / entry, self.filters["step_size"])
            qty = max(capped_qty, self.filters["min_qty"])
            notional = qty * entry
            if notional < self.filters["min_notional"]:
                raise ValueError("max_notional is below exchange minimum notional; increase it")
        if qty <= 0:
            raise ValueError("Quantity computed as zero; check filters and price")
        return qty, notional

    def _has_open_position(self) -> bool:
        positions = self.client.get_position_risk(symbol=self.settings.symbol)
        for pos in positions:
            pos_amt = float(pos.get("positionAmt", 0))
            if abs(pos_amt) > 0:
                return True
        return False

    def _place_orders(
        self,
        direction: int,
        qty: float,
        stop_price: float,
        tp_price: float,
        price: float,
    ) -> dict:
        side = "BUY" if direction > 0 else "SELL"
        exit_side = "SELL" if direction > 0 else "BUY"
        print(
            f"[live] submitting market {side} {qty} {self.settings.symbol} @ ~{price}, "
            f"stop {stop_price}, tp {tp_price}"
        )
        order = self.client.new_order(
            symbol=self.settings.symbol,
            side=side,
            type="MARKET",
            quantity=str(qty),
        )
        print(f"[live] entry order id: {order.get('orderId')}")
        entry_order_id = int(order.get("orderId"))
        try:
            stop_order = self.client.new_order(
                symbol=self.settings.symbol,
                side=exit_side,
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                quantity=str(qty),
                reduceOnly="true",
            )
        except ClientError as exc:
            if "algo order" in exc.error_message.lower() or "order type not supported" in exc.error_message.lower():
                print(
                    f"[warn] STOP_MARKET rejected ({exc.error_message}); retrying as STOP (stop-limit)"
                )
                stop_order = self.client.new_order(
                    symbol=self.settings.symbol,
                    side=exit_side,
                    type="STOP",
                    timeInForce="GTC",
                    stopPrice=str(stop_price),
                    price=str(stop_price),
                    quantity=str(qty),
                    reduceOnly="true",
                )
            else:
                raise
        print(f"[live] stop set id: {stop_order.get('orderId')}")
        effective_entry = float(order.get("avgPrice") or price)
        try:
            tp_order = self.client.new_order(
                symbol=self.settings.symbol,
                side=exit_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(tp_price),
                quantity=str(qty),
                reduceOnly="true",
            )
        except ClientError as exc:
            if "algo order" in exc.error_message.lower() or "order type not supported" in exc.error_message.lower():
                print(
                    f"[warn] TAKE_PROFIT_MARKET rejected ({exc.error_message}); retrying as TAKE_PROFIT (stop-limit)"
                )
                tp_order = self.client.new_order(
                    symbol=self.settings.symbol,
                    side=exit_side,
                    type="TAKE_PROFIT",
                    timeInForce="GTC",
                    stopPrice=str(tp_price),
                    price=str(tp_price),
                    quantity=str(qty),
                    reduceOnly="true",
                )
            else:
                raise
        print(f"[live] take-profit set id: {tp_order.get('orderId')}")
        stop_order_id = int(stop_order.get("orderId"))
        tp_order_id = int(tp_order.get("orderId"))
        trades = self._order_trades(entry_order_id)
        entry_price, entry_fee = self._trades_avg_price_and_fee(trades)
        if entry_price <= 0:
            entry_price = float(order.get("avgPrice") or price)
        return {
            "direction": "long" if direction > 0 else "short",
            "entry_order_id": entry_order_id,
            "stop_order_id": stop_order_id,
            "tp_order_id": tp_order_id,
            "entry_price": entry_price,
            "entry_fee": entry_fee,
            "size": qty,
            "sl_price": stop_price,
            "tp_price": tp_price,
            "timestamp_entry": pd.Timestamp.utcnow().isoformat(),
        }

    def _build_orders(
        self, direction: int, price: float, atr: float
    ) -> Tuple[float, float, float]:
        tick = self.filters["tick_size"]
        stop_distance = self.settings.stop_atr_mult * atr
        raw_stop = price - stop_distance if direction > 0 else price + stop_distance
        stop_price = self._round_to(max(raw_stop, 0.0001), tick)
        if direction > 0 and stop_price >= price:
            stop_price = self._round_to(max(price - tick, 0.0001), tick)
        elif direction < 0 and stop_price <= price:
            stop_price = self._round_to(price + tick, tick)
        risk_dist = abs(price - stop_price)
        tp_price = price + direction * (self.settings.tp_rr * risk_dist)
        tp_price = self._round_to(max(tp_price, 0.0001), tick)
        qty, _ = self._size_position(direction, price, stop_price)
        return qty, stop_price, tp_price

    def run_once(self) -> None:
        self._process_live_exit()
        klines = self.fetch_klines()
        if klines.empty:
            print("[error] no klines fetched; skipping cycle")
            return
        with_indicators = self._compute_indicators(klines)
        direction, price, atr = self._latest_signal(with_indicators)
        print(
            f"[info] signal: {'LONG' if direction>0 else 'SHORT' if direction<0 else 'FLAT'} "
            f"price={price:.4f} atr={atr:.4f}"
        )
        if direction == 0:
            print("[info] no trade signal; exiting")
            return
        if self._has_open_position():
            print("[info] open position detected; skipping new entry")
            return
        qty, stop_price, tp_price = self._build_orders(direction, price, atr)
        slippage_price = price * (1 + self.settings.max_slippage_pct * direction)
        plan = (
            f"[plan] {self.settings.symbol} {('LONG' if direction>0 else 'SHORT')}, "
            f"qty={qty}, entry~{price:.4f}, stop={stop_price:.4f}, tp={tp_price:.4f}, "
            f"est.notional=${qty*price:.2f}"
        )
        print(plan)
        if not self.settings.live:
            print("[dry-run] pass --live to send orders")
            return
        self._ensure_leverage()
        try:
            state = self._place_orders(direction, qty, stop_price, tp_price, slippage_price)
            self._live_state = state
            print(f"[live] entry logged at {state['entry_price']:.6f}, waiting for exit")
        except ClientError as exc:
            print(f"[error] order failed: {exc.error_message}")


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser(description="Binance Futures bot (single cycle)")
    parser.add_argument("--symbol", default=os.getenv("BOT_SYMBOL", "DOGEUSDT"))
    parser.add_argument("--interval", default=os.getenv("BOT_INTERVAL", "1h"))
    parser.add_argument(
        "--lookback",
        type=int,
        default=int(os.getenv("BOT_LOOKBACK", "300")),
        help="number of candles to fetch (pagination; Binance limit 1500 per call)",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=int(os.getenv("BOT_SLEEP", "300")),
        help="seconds to sleep between cycles when --loop is enabled",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="keep polling continuously instead of exiting after one check",
    )
    parser.add_argument("--live", action="store_true", help="execute live orders")
    parser.add_argument("--testnet", action="store_true", help="use Binance Futures testnet")
    parser.add_argument(
        "--live-log",
        default=os.getenv("BOT_LIVE_LOG", "reports/live_trades.csv"),
        help="CSV path to append live trade logs",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=float(os.getenv("BOT_API_TIMEOUT", "10")),
        help="HTTP timeout (seconds) for Binance requests",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=int(os.getenv("BOT_API_RETRIES", "3")),
        help="Retries for recoverable Binance HTTP errors",
    )
    parser.add_argument(
        "--api-retry-backoff",
        type=float,
        default=float(os.getenv("BOT_API_RETRY_BACKOFF", "2.0")),
        help="Exponential backoff base (seconds) between retries",
    )
    args = parser.parse_args()
    testnet_env = os.getenv("BINANCE_TESTNET", "0") == "1"
    return Settings(
        symbol=args.symbol.upper(),
        interval=args.interval,
        lookback=args.lookback,
        testnet=args.testnet or testnet_env,
        live=args.live,
        loop=args.loop,
        poll_seconds=max(1, args.sleep),
        live_log=args.live_log,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        api_retry_backoff=args.api_retry_backoff,
    )


def main() -> None:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_API_KEY/SECRET required in .env")
    settings = parse_settings()
    bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"\n[cycle] {cycle} @ {pd.Timestamp.utcnow().isoformat()}Z")
            bot.run_once()
            if not settings.loop:
                break
            sleep_for = max(1, settings.poll_seconds)
            print(f"[sleep] waiting {sleep_for}s before next check")
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[info] interrupted by user; exiting")


if __name__ == "__main__":
    main()
