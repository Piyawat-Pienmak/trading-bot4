import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr, bollinger_bands, rsi


class RangeMeanReversion(BaseStrategy):
    key = "range_mean_reversion"
    name = "Range Mean Reversion"
    family = "2"
    version = "1.0"
    description = "Fade extremes to the mid-band when trend is flat using Bollinger Bands + RSI."

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        ema_flat: int = 50,
        stop_mult: float = 1.2,
    ) -> None:
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.ema_flat = ema_flat
        self.stop_mult = stop_mult

    def warmup(self) -> int:
        return max(self.bb_window, self.ema_flat) + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        bb = bollinger_bands(out["close"], window=self.bb_window, num_std=self.bb_std)
        out = pd.concat([out, bb], axis=1)
        out["ema_flat"] = out["close"].ewm(span=self.ema_flat, adjust=False).mean()
        out["ema_flat_prev"] = out["ema_flat"].shift(10)
        out["rsi"] = rsi(out["close"], 14)
        out["atr14"] = atr(out, 14)
        return out

    def _is_flat(self, row: pd.Series) -> bool:
        ema = float(row["ema_flat"])
        prev = float(row["ema_flat_prev"])
        if any(pd.isna(x) for x in [ema, prev]):
            return False
        slope = abs(ema - prev) / ema if ema else 0.0
        return slope < 0.01

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        close = float(row["close"])
        bb_upper = float(row["bb_upper"])
        bb_lower = float(row["bb_lower"])
        bb_mid = float(row["bb_mid"])
        rsi_val = float(row["rsi"])
        atr_val = float(row["atr14"])
        if any(pd.isna(x) for x in [bb_upper, bb_lower, bb_mid, rsi_val, atr_val]):
            return None
        flat = self._is_flat(row)
        if not flat:
            return None
        if close < bb_lower and rsi_val < 35:
            stop = close - self.stop_mult * atr_val
            risk = close - stop
            if risk <= 0:
                return None
            tp = bb_mid
            if pd.isna(tp) or tp <= 0:
                tp = close + risk * 2
            return EntrySignal("long", close, stop, tp, "Range fade lower band + RSI<35")
        if close > bb_upper and rsi_val > 65:
            stop = close + self.stop_mult * atr_val
            risk = stop - close
            if risk <= 0:
                return None
            tp = bb_mid
            if pd.isna(tp) or tp <= 0:
                tp = close - risk * 2
            return EntrySignal("short", close, stop, tp, "Range fade upper band + RSI>65")
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
        bb_mid = float(row["bb_mid"])
        rsi_val = float(row["rsi"])
        close = float(row["close"])
        if pd.isna(bb_mid) or pd.isna(rsi_val):
            return None
        if position == "long":
            if close >= bb_mid:
                return ExitSignal(bb_mid, "hit_mid_band")
            if rsi_val > 55:
                return ExitSignal(close, "rsi_mean")
        if position == "short":
            if close <= bb_mid:
                return ExitSignal(bb_mid, "hit_mid_band")
            if rsi_val < 45:
                return ExitSignal(close, "rsi_mean")
        return None
