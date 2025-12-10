import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr, rsi


class MultiTimeframeTrend(BaseStrategy):
    key = "multi_timeframe_trend"
    name = "Multi-Timeframe Trend + Trigger"
    family = "5"
    version = "1.0"
    description = "HTF EMA trend proxy with LTF pullback trigger; approximate HTF using longer EMA on same feed."

    def __init__(
        self,
        htf_span: int = 400,
        ltf_span: int = 50,
        rsi_len: int = 14,
        stop_mult: float = 1.3,
        tp_rr: float = 2.0,
    ) -> None:
        self.htf_span = htf_span
        self.ltf_span = ltf_span
        self.rsi_len = rsi_len
        self.stop_mult = stop_mult
        self.tp_rr = tp_rr

    def warmup(self) -> int:
        return self.htf_span + 20

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema_htf"] = out["close"].ewm(span=self.htf_span, adjust=False).mean()
        out["ema_htf_prev"] = out["ema_htf"].shift(24)
        out["ema_ltf"] = out["close"].ewm(span=self.ltf_span, adjust=False).mean()
        out["rsi"] = rsi(out["close"], self.rsi_len)
        out["atr14"] = atr(out, 14)
        return out

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        close = float(row["close"])
        ema_htf = float(row["ema_htf"])
        ema_htf_prev = float(row["ema_htf_prev"])
        ema_ltf = float(row["ema_ltf"])
        rsi_val = float(row["rsi"])
        atr_val = float(row["atr14"])
        if any(pd.isna(x) for x in [ema_htf, ema_htf_prev, ema_ltf, rsi_val, atr_val]):
            return None
        htf_up = ema_htf > ema_htf_prev
        htf_down = ema_htf < ema_htf_prev
        pullback_long = close >= ema_ltf * 0.99 and close <= ema_ltf * 1.01 and rsi_val > 55
        pullback_short = close >= ema_ltf * 0.99 and close <= ema_ltf * 1.01 and rsi_val < 45
        if htf_up and close > ema_htf and pullback_long:
            stop = close - self.stop_mult * atr_val
            risk = close - stop
            if risk <= 0:
                return None
            tp = close + self.tp_rr * risk
            return EntrySignal("long", close, stop, tp, "HTF up + pullback trigger")
        if htf_down and close < ema_htf and pullback_short:
            stop = close + self.stop_mult * atr_val
            risk = stop - close
            if risk <= 0:
                return None
            tp = close - self.tp_rr * risk
            return EntrySignal("short", close, stop, tp, "HTF down + pullback trigger")
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
        ema_ltf = float(row["ema_ltf"])
        ema_htf = float(row["ema_htf"])
        close = float(row["close"])
        if pd.isna(ema_ltf) or pd.isna(ema_htf):
            return None
        if position == "long" and (close < ema_ltf or close < ema_htf):
            return ExitSignal(close, "lt_break")
        if position == "short" and (close > ema_ltf or close > ema_htf):
            return ExitSignal(close, "lt_break")
        return None
