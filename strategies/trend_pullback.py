import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr, rsi


class TrendPullback(BaseStrategy):
    key = "trend_pullback"
    name = "Trend Pullback"
    family = "1"
    version = "1.0"
    description = "Buy pullbacks in an uptrend / sell bounces in a downtrend using EMA slope + RSI confirmation."

    def __init__(
        self,
        slow: int = 200,
        fast: int = 50,
        rsi_len: int = 14,
        stop_mult: float = 1.5,
        tp_rr: float = 2.5,
    ) -> None:
        self.slow = slow
        self.fast = fast
        self.rsi_len = rsi_len
        self.stop_mult = stop_mult
        self.tp_rr = tp_rr

    def warmup(self) -> int:
        return max(self.slow, self.fast) + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema_slow"] = out["close"].ewm(span=self.slow, adjust=False).mean()
        out["ema_slow_prev"] = out["ema_slow"].shift(5)
        out["ema_fast"] = out["close"].ewm(span=self.fast, adjust=False).mean()
        out["rsi"] = rsi(out["close"], self.rsi_len)
        out["atr14"] = atr(out, 14)
        return out

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        close = float(row["close"])
        ema_slow = float(row["ema_slow"])
        ema_slow_prev = float(row["ema_slow_prev"])
        ema_fast = float(row["ema_fast"])
        atr_val = float(row["atr14"])
        rsi_val = float(row["rsi"])
        if any(pd.isna(x) for x in [ema_slow, ema_slow_prev, ema_fast, atr_val, rsi_val]):
            return None
        in_pullback_long = close <= ema_fast * 1.005 and close >= ema_fast * 0.985 and rsi_val < 45
        in_pullback_short = close >= ema_fast * 0.995 and close <= ema_fast * 1.015 and rsi_val > 55
        trend_up = ema_slow > ema_slow_prev and close > ema_slow
        trend_down = ema_slow < ema_slow_prev and close < ema_slow
        if trend_up and in_pullback_long:
            stop = close - self.stop_mult * atr_val
            risk = close - stop
            if risk <= 0:
                return None
            tp = close + self.tp_rr * risk
            return EntrySignal("long", close, stop, tp, "Uptrend pullback to EMAfast + RSI<45")
        if trend_down and in_pullback_short:
            stop = close + self.stop_mult * atr_val
            risk = stop - close
            if risk <= 0:
                return None
            tp = close - self.tp_rr * risk
            return EntrySignal("short", close, stop, tp, "Downtrend pullback to EMAfast + RSI>55")
        return None

    def check_exit(
        self,
        i: int,
        df: pd.DataFrame,
        position: str,
        entry_price: float,
        stop_price: float,
        take_profit: float,
    ):
        row = df.iloc[i]
        close = float(row["close"])
        ema_fast = float(row["ema_fast"])
        rsi_val = float(row["rsi"])
        if pd.isna(ema_fast) or pd.isna(rsi_val):
            return None
        if position == "long":
            if close < ema_fast:
                return ExitSignal(close, "fast_ema_break")
            if rsi_val > 65:
                return ExitSignal(close, "rsi_relief")
        if position == "short":
            if close > ema_fast:
                return ExitSignal(close, "fast_ema_break")
            if rsi_val < 35:
                return ExitSignal(close, "rsi_relief")
        return None
