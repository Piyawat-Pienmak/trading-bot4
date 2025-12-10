import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr


class TrendBreakoutV1(BaseStrategy):
    key = "trend_breakout_v1"
    name = "Trend Breakout v1 (archived reference)"
    family = "0"
    version = "1.0"
    description = "EMA200 trend filter + 24-bar breakout with ATR(14) 2R/3R stop/target."

    def warmup(self) -> int:
        return 200 + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
        out["ema200_10"] = out["ema200"].shift(10)
        out["atr14"] = atr(out, 14)
        out["hh24_prev"] = out["high"].rolling(window=24, min_periods=24).max().shift(1)
        out["ll24_prev"] = out["low"].rolling(window=24, min_periods=24).min().shift(1)
        return out

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        close = float(row["close"])
        ema = float(row["ema200"])
        ema_10 = float(row["ema200_10"])
        atr_val = float(row["atr14"])
        hh_prev = float(row["hh24_prev"])
        ll_prev = float(row["ll24_prev"])
        if any(pd.isna(x) for x in [ema, ema_10, atr_val, hh_prev, ll_prev]):
            return None
        if close > ema and ema > ema_10 and close > hh_prev:
            stop = close - 2.0 * atr_val
            risk = close - stop
            if risk <= 0:
                return None
            tp = close + 3.0 * risk
            return EntrySignal("long", close, stop, tp, "EMA up + HH24 breakout")
        if close < ema and ema < ema_10 and close < ll_prev:
            stop = close + 2.0 * atr_val
            risk = stop - close
            if risk <= 0:
                return None
            tp = close - 3.0 * risk
            return EntrySignal("short", close, stop, tp, "EMA down + LL24 breakdown")
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
        ema = float(row["ema200"])
        close = float(row["close"])
        if position == "long" and close < ema:
            return ExitSignal(close, "EMA_break")
        if position == "short" and close > ema:
            return ExitSignal(close, "EMA_break")
        return None
