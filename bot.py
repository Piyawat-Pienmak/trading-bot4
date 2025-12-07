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
    interval: str = "15m"
    lookback: int = 200
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    rsi_long: float = 55.0
    rsi_short: float = 45.0
    atr_vol_min_pct: float = 0.002
    trend_lookback: int = 3
    atr_period: int = 14
    atr_multiplier: float = 2.5
    leverage: int = 5
    risk_pct: float = 0.01  # 1% of equity per trade
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
        klines = self.client.klines(
            symbol=self.settings.symbol,
            interval=self.settings.interval,
            limit=self.settings.lookback,
        )
        df = pd.DataFrame(
            klines,
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
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)
        return df

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.settings.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.settings.rsi_period).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.settings.fast_ema, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.settings.slow_ema, adjust=False).mean()
        df["prev_close"] = df["close"].shift(1)
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = (df["high"] - df["prev_close"]).abs()
        df["low_close"] = (df["low"] - df["prev_close"]).abs()
        df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["atr"] = (
            df["tr"].rolling(window=self.settings.atr_period, min_periods=1).mean()
        )
        df["rsi"] = self._compute_rsi(df["close"])
        df["atr_pct"] = df["atr"] / df["close"].replace(0, pd.NA)
        df["slow_slope"] = df["ema_slow"] - df["ema_slow"].shift(self.settings.trend_lookback)
        return df

    def _signal_direction(self, row: pd.Series) -> int:
        price = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        rsi = float(row["rsi"])
        atr_pct = float(row["atr_pct"])
        slope = float(row["slow_slope"])
        if (
            math.isnan(ema_fast)
            or math.isnan(ema_slow)
            or math.isnan(rsi)
            or math.isnan(atr_pct)
            or math.isnan(slope)
        ):
            return 0
        if atr_pct < self.settings.atr_vol_min_pct:
            return 0
        bullish = (
            ema_fast > ema_slow
            and price > ema_slow
            and rsi >= self.settings.rsi_long
            and slope > 0
        )
        bearish = (
            ema_fast < ema_slow
            and price < ema_slow
            and rsi <= self.settings.rsi_short
            and slope < 0
        )
        if bullish:
            return 1
        if bearish:
            return -1
        return 0

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
        self, direction: int, price: float, atr: float
    ) -> Tuple[float, float]:
        stop_distance = self.settings.atr_multiplier * atr
        if stop_distance <= 0:
            raise ValueError("ATR stop distance invalid")
        equity = self._account_equity()
        risk_amount = equity * self.settings.risk_pct
        raw_qty = risk_amount / stop_distance
        qty = max(raw_qty, self.filters["min_qty"])
        qty = max(self._round_to(qty, self.filters["step_size"]), self.filters["min_qty"])
        notional = qty * price
        if notional < self.filters["min_notional"]:
            min_qty = self.filters["min_notional"] / price
            qty = self._round_to(max(min_qty, self.filters["min_qty"]), self.filters["step_size"])
            notional = qty * price
        if qty <= 0:
            raise ValueError("Quantity computed as zero; check filters and price")
        stop_price = price - stop_distance if direction > 0 else price + stop_distance
        stop_price = max(stop_price, 0.0001)
        stop_price = self._round_to(stop_price, self.filters["tick_size"])
        return qty, stop_price

    def _has_open_position(self) -> bool:
        positions = self.client.position_information(symbol=self.settings.symbol)
        for pos in positions:
            pos_amt = float(pos.get("positionAmt", 0))
            if abs(pos_amt) > 0:
                return True
        return False

    def _place_orders(
        self, direction: int, qty: float, stop_price: float, price: float
    ) -> None:
        side = "BUY" if direction > 0 else "SELL"
        exit_side = "SELL" if direction > 0 else "BUY"
        print(
            f"[live] submitting market {side} {qty} {self.settings.symbol} @ ~{price}, stop {stop_price}"
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
        qty, stop_price = self._size_position(direction, price, atr)
        slippage_price = price * (1 + self.settings.max_slippage_pct * direction)
        plan = (
            f"[plan] {self.settings.symbol} {('LONG' if direction>0 else 'SHORT')}, "
            f"qty={qty}, entry~{price:.4f}, stop={stop_price:.4f}, "
            f"est.notional=${qty*price:.2f}"
        )
        print(plan)
        if not self.settings.live:
            print("[dry-run] pass --live to send orders")
            return
        self._ensure_leverage()
        try:
            self._place_orders(direction, qty, stop_price, slippage_price)
        except ClientError as exc:
            print(f"[error] order failed: {exc.error_message}")


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser(description="Binance Futures bot (single cycle)")
    parser.add_argument("--symbol", default=os.getenv("BOT_SYMBOL", "DOGEUSDT"))
    parser.add_argument("--interval", default=os.getenv("BOT_INTERVAL", "5m"))
    parser.add_argument("--live", action="store_true", help="execute live orders")
    parser.add_argument("--testnet", action="store_true", help="use Binance Futures testnet")
    args = parser.parse_args()
    testnet_env = os.getenv("BINANCE_TESTNET", "0") == "1"
    return Settings(
        symbol=args.symbol.upper(),
        interval=args.interval,
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
