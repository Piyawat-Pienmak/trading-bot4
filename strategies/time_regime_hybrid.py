import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr, bollinger_bands, rsi


class TimeRegimeHybrid(BaseStrategy):
    key = "time_regime_hybrid"
    name = "Time/Regime Hybrid"
    family = "6"
    version = "1.0"
    description = "Session-filtered range fade during low-vol regime (example hybrid)."

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 1.5,
        stop_mult: float = 1.0,
    ) -> None:
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.stop_mult = stop_mult

    def warmup(self) -> int:
        return self.bb_window + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        bb = bollinger_bands(out["close"], window=self.bb_window, num_std=self.bb_std)
        out = pd.concat([out, bb], axis=1)
        out["atr14"] = atr(out, 14)
        out["atr_slow"] = out["atr14"].rolling(window=50, min_periods=50).mean()
        out["rsi"] = rsi(out["close"], 14)
        out["hour_utc"] = out["time"].dt.hour
        return out

    def _session_ok(self, hour_utc: int) -> bool:
        return 0 <= hour_utc <= 7

    def _vol_low(self, row: pd.Series) -> bool:
        atr_now = float(row["atr14"])
        atr_slow = float(row["atr_slow"])
        if any(pd.isna(x) for x in [atr_now, atr_slow]):
            return False
        return atr_now < atr_slow

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        hour = int(row["hour_utc"])
        close = float(row["close"])
        upper = float(row["bb_upper"])
        lower = float(row["bb_lower"])
        mid = float(row["bb_mid"])
        atr_val = float(row["atr14"])
        rsi_val = float(row["rsi"])
        if any(pd.isna(x) for x in [upper, lower, mid, atr_val, rsi_val]):
            return None
        if not self._session_ok(hour) or not self._vol_low(row):
            return None
        if close < lower and rsi_val < 40:
            stop = close - self.stop_mult * atr_val
            risk = close - stop
            if risk <= 0:
                return None
            tp = mid if not pd.isna(mid) else close + 2 * risk
            return EntrySignal("long", close, stop, tp, "Asia low-vol fade lower band")
        if close > upper and rsi_val > 60:
            stop = close + self.stop_mult * atr_val
            risk = stop - close
            if risk <= 0:
                return None
            tp = mid if not pd.isna(mid) else close - 2 * risk
            return EntrySignal("short", close, stop, tp, "Asia low-vol fade upper band")
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
        hour = int(row["hour_utc"])
        close = float(row["close"])
        if pd.isna(mid):
            return None
        if position == "long":
            if close >= mid:
                return ExitSignal(mid, "mid_band")
            if not self._session_ok(hour):
                return ExitSignal(close, "session_end_flatten")
        if position == "short":
            if close <= mid:
                return ExitSignal(mid, "mid_band")
            if not self._session_ok(hour):
                return ExitSignal(close, "session_end_flatten")
        return None
