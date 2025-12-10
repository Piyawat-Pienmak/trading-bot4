from typing import Dict, List

from .base import BaseStrategy
from .momentum_reversal import MomentumReversal
from .multi_timeframe_trend import MultiTimeframeTrend
from .range_mean_reversion import RangeMeanReversion
from .time_regime_hybrid import TimeRegimeHybrid
from .trend_breakout_v1 import TrendBreakoutV1
from .trend_pullback import TrendPullback
from .vol_squeeze_breakout import VolatilitySqueezeBreakout

_REGISTRY: Dict[str, type[BaseStrategy]] = {
    TrendBreakoutV1.key: TrendBreakoutV1,
    TrendPullback.key: TrendPullback,
    RangeMeanReversion.key: RangeMeanReversion,
    VolatilitySqueezeBreakout.key: VolatilitySqueezeBreakout,
    MomentumReversal.key: MomentumReversal,
    MultiTimeframeTrend.key: MultiTimeframeTrend,
    TimeRegimeHybrid.key: TimeRegimeHybrid,
}


def get_strategy_keys() -> List[str]:
    return list(_REGISTRY.keys())


def build_strategy(key: str) -> BaseStrategy:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown strategy key '{key}'")
    return _REGISTRY[key]()  # type: ignore


def all_strategies() -> List[BaseStrategy]:
    return [cls() for cls in _REGISTRY.values()]  # type: ignore
