from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class EntrySignal:
    direction: str  # "long" or "short"
    entry_price: float
    stop_price: float
    take_profit: float
    reason: str = ""


@dataclass
class ExitSignal:
    exit_price: float
    reason: str


class BaseStrategy:
    """Interface for all strategy families."""

    key: str = "base"
    name: str = "Base Strategy"
    family: str = "N/A"
    version: str = "0.0"
    description: str = ""

    def warmup(self) -> int:
        """Number of bars to skip before signals are valid."""
        return 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators and return a new dataframe."""
        raise NotImplementedError

    def check_entry(
        self,
        i: int,
        df: pd.DataFrame,
        position: str,
    ) -> Optional[EntrySignal]:
        raise NotImplementedError

    def check_exit(
        self,
        i: int,
        df: pd.DataFrame,
        position: str,
        entry_price: float,
        stop_price: float,
        take_profit: float,
    ) -> Optional[ExitSignal]:
        return None
