import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr, bollinger_bands


class VolatilitySqueezeBreakout(BaseStrategy):
    key = "vol_squeeze_breakout"
    name = "Volatility Squeeze Breakout"
    family = "3"
    version = "1.0"
    description = "Detect low BB width then trade breakout beyond bands with ATR-based stops."

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        squeeze_mult: float = 0.8,
        stop_mult: float = 1.0,
        tp_rr: float = 2.0,
    ) -> None:
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.squeeze_mult = squeeze_mult
        self.stop_mult = stop_mult
        self.tp_rr = tp_rr

    def warmup(self) -> int:
        return self.bb_window + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        bb = bollinger_bands(out["close"], window=self.bb_window, num_std=self.bb_std)
        out = pd.concat([out, bb], axis=1)
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
        out["bb_width_sma"] = out["bb_width"].rolling(window=20, min_periods=20).mean()
        out["atr14"] = atr(out, 14)
        return out

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        close = float(row["close"])
        upper = float(row["bb_upper"])
        lower = float(row["bb_lower"])
        mid = float(row["bb_mid"])
        width = float(row["bb_width"])
        width_sma = float(row["bb_width_sma"])
        atr_val = float(row["atr14"])
        if any(pd.isna(x) for x in [upper, lower, mid, width, width_sma, atr_val]):
            return None
        squeeze = width < width_sma * self.squeeze_mult
        if not squeeze:
            return None
        if close > upper:
            stop = max(mid - self.stop_mult * atr_val, 0.0001)
            risk = close - stop
            if risk <= 0:
                return None
            tp = close + self.tp_rr * risk
            return EntrySignal("long", close, stop, tp, "Breakout up after squeeze")
        if close < lower:
            stop = max(mid + self.stop_mult * atr_val, 0.0001)
            risk = stop - close
            if risk <= 0:
                return None
            tp = close - self.tp_rr * risk
            return EntrySignal("short", close, stop, tp, "Breakdown after squeeze")
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
        mid = float(row["bb_mid"])
        close = float(row["close"])
        if pd.isna(mid):
            return None
        if position == "long" and close < mid:
            return ExitSignal(close, "gave_back_to_mid")
        if position == "short" and close > mid:
            return ExitSignal(close, "gave_back_to_mid")
        return None
