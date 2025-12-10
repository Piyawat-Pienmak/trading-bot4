import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
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


class FuturesBot:
    def __init__(self, settings: Settings, api_key: str, api_secret: str) -> None:
        self.settings = settings
        base_url = settings.base_url
        if settings.testnet and not base_url:
            base_url = "https://testnet.binancefuture.com"
        if not base_url:
            base_url = "https://fapi.binance.com"
        self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        self.filters = self._fetch_filters(settings.symbol)

    def _fetch_filters(self, symbol: str) -> Dict[str, float]:
        info = self.client.exchange_info()
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
            batch = self.client.klines(**params)
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

    def _round_to(self, value: float, step: float) -> float:
        precision = max(int(round(-math.log(step, 10))) if step < 1 else 0, 0)
        return round(math.floor(value / step) * step, precision)

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
            qty = self._round_to(max(min_qty, self.filters["min_qty"]), self.filters["step_size"])
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
        positions = self.client.position_information(symbol=self.settings.symbol)
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
    ) -> None:
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
        stop_order = self.client.new_order(
            symbol=self.settings.symbol,
            side=exit_side,
            type="STOP_MARKET",
            stopPrice=str(stop_price),
            closePosition="true",
            workingType="MARK_PRICE",
        )
        print(f"[live] stop set id: {stop_order.get('orderId')}")
        effective_entry = float(order.get("avgPrice") or price)
        tp_order = self.client.new_order(
            symbol=self.settings.symbol,
            side=exit_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=str(tp_price),
            closePosition="true",
            workingType="MARK_PRICE",
        )
        print(f"[live] take-profit set id: {tp_order.get('orderId')}")

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
        klines = self.fetch_klines()
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
            self._place_orders(direction, qty, stop_price, tp_price, slippage_price)
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
    parser.add_argument("--live", action="store_true", help="execute live orders")
    parser.add_argument("--testnet", action="store_true", help="use Binance Futures testnet")
    args = parser.parse_args()
    testnet_env = os.getenv("BINANCE_TESTNET", "0") == "1"
    return Settings(
        symbol=args.symbol.upper(),
        interval=args.interval,
        lookback=args.lookback,
        testnet=args.testnet or testnet_env,
        live=args.live,
    )


def main() -> None:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_API_KEY/SECRET required in .env")
    settings = parse_settings()
    bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
    bot.run_once()


if __name__ == "__main__":
    main()
