import pandas as pd

from .base import BaseStrategy, EntrySignal, ExitSignal
from .indicators import atr, rsi


class MomentumReversal(BaseStrategy):
    key = "momentum_reversal"
    name = "Momentum Reversal / Exhaustion"
    family = "4"
    version = "1.0"
    description = "RSI2 exhaustion with short-horizon momentum stretch and quick mean reversion targets."

    def __init__(
        self,
        rsi_len: int = 2,
        ema_anchor: int = 50,
        stop_mult: float = 1.0,
        tp_rr: float = 1.5,
        shock_lookback: int = 3,
        shock_move: float = 0.02,
    ) -> None:
        self.rsi_len = rsi_len
        self.ema_anchor = ema_anchor
        self.stop_mult = stop_mult
        self.tp_rr = tp_rr
        self.shock_lookback = shock_lookback
        self.shock_move = shock_move

    def warmup(self) -> int:
        return max(self.ema_anchor, self.shock_lookback + 1) + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["rsi_fast"] = rsi(out["close"], self.rsi_len)
        out["ema_anchor"] = out["close"].ewm(span=self.ema_anchor, adjust=False).mean()
        out["atr14"] = atr(out, 14)
        out["chg_shock"] = out["close"].pct_change(self.shock_lookback)
        return out

    def check_entry(self, i: int, df: pd.DataFrame, position: str):
        row = df.iloc[i]
        close = float(row["close"])
        ema = float(row["ema_anchor"])
        rsi_fast = float(row["rsi_fast"])
        atr_val = float(row["atr14"])
        shock = float(row["chg_shock"])
        if any(pd.isna(x) for x in [ema, rsi_fast, atr_val, shock]):
            return None
        if shock < -self.shock_move and rsi_fast < 10 and close > ema * 0.9:
            stop = close - self.stop_mult * atr_val
            risk = close - stop
            if risk <= 0:
                return None
            tp = close + self.tp_rr * risk
            return EntrySignal("long", close, stop, tp, "Fast selloff + RSI2 oversold")
        if shock > self.shock_move and rsi_fast > 90 and close < ema * 1.1:
            stop = close + self.stop_mult * atr_val
            risk = stop - close
            if risk <= 0:
                return None
            tp = close - self.tp_rr * risk
            return EntrySignal("short", close, stop, tp, "Fast spike + RSI2 overbought")
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
        ema = float(row["ema_anchor"])
        close = float(row["close"])
        if pd.isna(ema):
            return None
        if position == "long" and close < ema:
            return ExitSignal(close, "back_below_anchor")
        if position == "short" and close > ema:
            return ExitSignal(close, "back_above_anchor")
        return None
