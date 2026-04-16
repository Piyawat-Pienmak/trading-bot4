#!/usr/bin/env python3

import argparse
import itertools
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

OPT_LOOKBACKS = [10, 20, 30, 50]
OPT_ZSCORES = [1.0, 1.5, 2.0, 2.5]
OPT_STOP_ATR_MULTS = [1.0, 1.5, 2.0, 2.5]
OPT_TAKE_ATR_MULTS = [1.0, 1.5, 2.0, 3.0]
OPT_REGIME_THRESHOLD_BPS = [10.0, 15.0, 20.0, 30.0]
OPT_POSITION_MODES = ["both", "long", "short"]
OPT_EXIT_ON_MIDLINE = [True, False]
OPT_EXIT_ON_OPPOSITE_SIGNAL = [False, True]
TIMESTAMP_COLUMN_ALIASES = {
    "open_time": ["open_time", "timestamp", "datetime", "date", "time"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mean reversion long/short backtest using range re-entry, z-score, and ATR exits."
        )
    )
    parser.add_argument("csv", type=Path, help="Path to CSV file with OHLCV columns.")
    parser.add_argument("--timestamp-col", default="open_time", help="Timestamp column name. Default: open_time")
    parser.add_argument("--open-col", default="open", help="Open column name. Default: open")
    parser.add_argument("--high-col", default="high", help="High column name. Default: high")
    parser.add_argument("--low-col", default="low", help="Low column name. Default: low")
    parser.add_argument("--close-col", default="close", help="Close column name. Default: close")
    parser.add_argument("--volume-col", default="volume", help="Volume column name. Default: volume")
    parser.add_argument("--initial-capital", type=float, default=25.0, help="Starting capital. Default: 25")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Fee per trade side as decimal. Default: 0.001")
    parser.add_argument("--lookback", type=int, default=20, help="Rolling range / mean window. Default: 20")
    parser.add_argument("--zscore-threshold", type=float, default=1.0, help="Minimum absolute z-score before re-entry trigger. Default: 1.0")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period. Default: 14")
    parser.add_argument("--stop-atr-mult", type=float, default=2.0, help="Stop-loss in ATR multiples. Default: 2.0")
    parser.add_argument("--take-atr-mult", type=float, default=3.0, help="Take-profit in ATR multiples. Default: 3.0")
    parser.add_argument("--trail-atr-mult", type=float, default=0.0, help="Trailing stop in ATR multiples. Set 0 to disable. Default: 0")
    parser.add_argument("--position-mode", choices=["both", "long", "short"], default="both", help="Trade direction mode. Default: both")
    parser.add_argument("--trend-ema-length", type=int, default=100, help="Optional EMA trend filter. 0 disables it. Default: 100")
    parser.add_argument("--trend-slope-lookback", type=int, default=5, help="Bars used to smooth EMA slope for regime detection. Default: 5")
    parser.add_argument("--regime-threshold-bps", type=float, default=0.0, help="Treat bars as trending when average EMA slope magnitude is at least this many bps per bar. 0 disables the slope regime filter. Default: 0")
    parser.add_argument("--trend-adx-period", type=int, default=14, help="ADX period used for regime detection. Default: 14")
    parser.add_argument("--trend-adx-threshold", type=float, default=0.0, help="Treat bars as trending when ADX is at least this value. 0 disables the ADX regime filter. Default: 0")
    parser.add_argument("--trend-extension-atr", type=float, default=1.0, help="Treat bars as trending when price is this many ATRs away from trend EMA. 0 disables the extension filter. Default: 2.5")
    parser.add_argument(
        "--max-efficiency-ratio",
        type=float,
        default=0.0,
        help=(
            "Only enter when rolling directional efficiency ratio is at or below this value. "
            "Lower values favor oscillating/choppy price action. 0 disables the filter. Default: 0"
        ),
    )
    parser.add_argument(
        "--min-mean-crosses",
        type=int,
        default=0,
        help=(
            "Only enter when price has crossed the rolling mean at least this many times within the lookback window. "
            "0 disables the filter. Default: 0"
        ),
    )
    parser.add_argument("--volume-ma-period", type=int, default=20, help="Volume MA period. Default: 20")
    parser.add_argument("--volume-max-mult", type=float, default=0.0, help="Only enter when volume <= volume_ma * this value. 0 disables the filter. Default: 0")
    parser.add_argument("--reentry-buffer-bps", type=float, default=5.0, help="Require close to come back inside the range by this many bps. Default: 5")
    parser.add_argument("--min-hold-bars", type=int, default=1, help="Minimum bars to hold before mean-exit/opposite exit. Default: 1")
    parser.add_argument("--cooldown-bars", type=int, default=1, help="Bars to wait after exit before new entry. Default: 1")
    parser.add_argument(
        "--exit-on-midline",
        dest="exit_on_midline",
        action="store_true",
        help="Exit when price reverts to rolling mean. Default: enabled",
    )
    parser.add_argument(
        "--no-exit-on-midline",
        dest="exit_on_midline",
        action="store_false",
        help="Disable exit on rolling mean reversion.",
    )
    parser.add_argument("--exit-on-opposite-signal", action="store_true", help="Exit when opposite mean-reversion signal appears.")
    parser.add_argument("--optimize", action="store_true", help="Grid-search core parameters and use the best result.")
    parser.add_argument("--min-trades", type=int, default=4, help="Minimum closed trades required for optimization candidate. Default: 4")
    parser.add_argument("--plot", action="store_true", help="Display chart with entries/exits and equity.")
    parser.add_argument("--plot-path", type=Path, help="Save chart to a file.")
    parser.add_argument("--plot-data-csv-path", type=Path, help="Optional output path for exported plot-ready CSV.")
    parser.add_argument(
        "--save-plot",
        dest="save_plot",
        action="store_true",
        help="Save a timestamped plot into reports/. Default: enabled",
    )
    parser.add_argument(
        "--no-save-plot",
        dest="save_plot",
        action="store_false",
        help="Disable saving a timestamped plot into reports/.",
    )
    parser.add_argument("--trades-csv-path", type=Path, help="Optional output path for exported trade results CSV.")
    parser.set_defaults(exit_on_midline=False, exit_on_opposite_signal=False, save_plot=True)
    args = parser.parse_args()

    if args.lookback < 2:
        raise SystemExit("--lookback must be at least 2.")
    if args.zscore_threshold <= 0:
        raise SystemExit("--zscore-threshold must be > 0.")
    if args.atr_period < 1:
        raise SystemExit("--atr-period must be at least 1.")
    if args.stop_atr_mult < 0 or args.take_atr_mult < 0 or args.trail_atr_mult < 0:
        raise SystemExit("ATR multipliers must be >= 0.")
    if args.trend_ema_length < 0:
        raise SystemExit("--trend-ema-length must be >= 0.")
    if args.trend_slope_lookback < 1:
        raise SystemExit("--trend-slope-lookback must be at least 1.")
    if args.regime_threshold_bps < 0:
        raise SystemExit("--regime-threshold-bps must be >= 0.")
    if args.trend_adx_period < 1:
        raise SystemExit("--trend-adx-period must be at least 1.")
    if args.trend_adx_threshold < 0:
        raise SystemExit("--trend-adx-threshold must be >= 0.")
    if args.trend_extension_atr < 0:
        raise SystemExit("--trend-extension-atr must be >= 0.")
    if args.max_efficiency_ratio < 0:
        raise SystemExit("--max-efficiency-ratio must be >= 0.")
    if args.min_mean_crosses < 0:
        raise SystemExit("--min-mean-crosses must be >= 0.")
    if args.volume_ma_period < 1:
        raise SystemExit("--volume-ma-period must be at least 1.")
    if args.volume_max_mult < 0:
        raise SystemExit("--volume-max-mult must be >= 0.")
    if args.reentry_buffer_bps < 0:
        raise SystemExit("--reentry-buffer-bps must be >= 0.")
    if args.min_hold_bars < 0 or args.cooldown_bars < 0:
        raise SystemExit("Hold/cooldown bars must be >= 0.")
    if args.min_trades < 0:
        raise SystemExit("--min-trades must be >= 0.")
    return args


def load_data(csv_path: Path, required_columns: list[str], min_rows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    df = df[required_columns].copy()
    ts_col = required_columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    for col in required_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values(ts_col).drop_duplicates(subset=[ts_col]).reset_index(drop=True)
    if len(df) < min_rows:
        raise ValueError(f"Not enough rows. Need at least {min_rows}.")
    return df


def resolve_column_name(requested: str, available: list[str]) -> str:
    if requested in available:
        return requested

    for alias in TIMESTAMP_COLUMN_ALIASES.get(requested, []):
        if alias in available:
            return alias

    return requested


def _true_range(data: pd.DataFrame, high_col: str, low_col: str, close_col: str) -> pd.Series:
    prev_close = data[close_col].shift(1)
    tr1 = data[high_col] - data[low_col]
    tr2 = (data[high_col] - prev_close).abs()
    tr3 = (data[low_col] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _adx(data: pd.DataFrame, high_col: str, low_col: str, close_col: str, period: int) -> pd.Series:
    up_move = data[high_col].diff()
    down_move = -data[low_col].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(data, high_col, low_col, close_col)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, pd.NA)
    plus_di = (100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr).fillna(0.0)
    minus_di = (100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr).fillna(0.0)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def _efficiency_ratio(series: pd.Series, lookback: int) -> pd.Series:
    net_move = (series - series.shift(lookback)).abs()
    path_length = series.diff().abs().rolling(lookback).sum()
    return (net_move / path_length.replace(0, pd.NA)).fillna(0.0)


def _rolling_mean_crosses(close: pd.Series, rolling_mean: pd.Series, lookback: int) -> pd.Series:
    deviation = close - rolling_mean
    prev_deviation = deviation.shift(1)
    crossed = (
        deviation.notna()
        & prev_deviation.notna()
        & (deviation != 0)
        & (prev_deviation != 0)
        & ((deviation > 0) != (prev_deviation > 0))
    )
    return crossed.rolling(lookback).sum().fillna(0.0)


def run_backtest(
    df: pd.DataFrame,
    timestamp_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    lookback: int,
    zscore_threshold: float,
    atr_period: int,
    stop_atr_mult: float,
    take_atr_mult: float,
    trail_atr_mult: float,
    trend_ema_length: int,
    trend_slope_lookback: int,
    regime_threshold_bps: float,
    trend_adx_period: int,
    trend_adx_threshold: float,
    trend_extension_atr: float,
    max_efficiency_ratio: float,
    min_mean_crosses: int,
    volume_ma_period: int,
    volume_max_mult: float,
    reentry_buffer_bps: float,
    min_hold_bars: int,
    cooldown_bars: int,
    exit_on_midline: bool,
    exit_on_opposite_signal: bool,
    position_mode: str,
    initial_capital: float,
    fee_rate: float,
) -> tuple[pd.DataFrame, list[dict]]:
    data = df.copy()
    data["rolling_mean"] = data[close_col].rolling(lookback).mean()
    data["rolling_std"] = data[close_col].rolling(lookback).std(ddof=0)
    data["upper_band"] = data[high_col].rolling(lookback).max().shift(1)
    data["lower_band"] = data[low_col].rolling(lookback).min().shift(1)
    data["zscore"] = (data[close_col] - data["rolling_mean"]) / data["rolling_std"].replace(0, pd.NA)
    data["zscore"] = data["zscore"].fillna(0.0)

    tr = _true_range(data, high_col, low_col, close_col)
    data["atr"] = tr.rolling(atr_period).mean()
    data["atr"] = data["atr"].bfill()
    data["adx"] = _adx(data, high_col, low_col, close_col, trend_adx_period)
    data["efficiency_ratio"] = _efficiency_ratio(data[close_col], lookback)
    data["mean_cross_count"] = _rolling_mean_crosses(data[close_col], data["rolling_mean"], lookback)

    data["volume_ma"] = data[volume_col].rolling(volume_ma_period).mean().bfill()
    if trend_ema_length > 0:
        data["trend_ema"] = data[close_col].ewm(span=trend_ema_length, adjust=False).mean()
        data["trend_ema_prev"] = data["trend_ema"].shift(trend_slope_lookback)
        data["trend_slope_bps"] = (
            (((data["trend_ema"] / data["trend_ema_prev"]) - 1.0) / trend_slope_lookback).fillna(0.0) * 10000.0
        )
        data["trend_extension_atr"] = ((data[close_col] - data["trend_ema"]).abs() / data["atr"].replace(0, pd.NA)).fillna(0.0)
    else:
        data["trend_ema"] = pd.NA
        data["trend_ema_prev"] = pd.NA
        data["trend_slope_bps"] = 0.0
        data["trend_extension_atr"] = 0.0

    buffer = reentry_buffer_bps / 10000.0

    prev_high_break = data[close_col].shift(1) > data["upper_band"].shift(1)
    prev_low_break = data[close_col].shift(1) < data["lower_band"].shift(1)
    short_reentry = prev_high_break & (data[close_col] <= data["upper_band"] * (1.0 - buffer)) & (data["zscore"] >= zscore_threshold)
    long_reentry = prev_low_break & (data[close_col] >= data["lower_band"] * (1.0 + buffer)) & (data["zscore"] <= -zscore_threshold)

    if trend_ema_length > 0:
        short_reentry &= data[close_col] >= data["trend_ema"]
        long_reentry &= data[close_col] <= data["trend_ema"]

    data["pre_regime_signal"] = 0
    if position_mode in {"both", "short"}:
        data.loc[short_reentry, "pre_regime_signal"] = -1
    if position_mode in {"both", "long"}:
        data.loc[long_reentry, "pre_regime_signal"] = 1

    slope_trend = data["trend_slope_bps"].abs() >= regime_threshold_bps if regime_threshold_bps > 0 else pd.Series(False, index=data.index)
    adx_trend = data["adx"] >= trend_adx_threshold if trend_adx_threshold > 0 else pd.Series(False, index=data.index)
    extension_trend = data["trend_extension_atr"] >= trend_extension_atr if trend_extension_atr > 0 else pd.Series(False, index=data.index)
    efficient_trend = data["efficiency_ratio"] > max_efficiency_ratio if max_efficiency_ratio > 0 else pd.Series(False, index=data.index)
    insufficient_crosses = data["mean_cross_count"] < float(min_mean_crosses) if min_mean_crosses > 0 else pd.Series(False, index=data.index)
    data["rejected_by_regime_threshold_bps"] = False
    data["rejected_by_trend_adx_threshold"] = False
    data["rejected_by_trend_extension_atr"] = False
    data["rejected_by_efficiency_ratio"] = False
    data["rejected_by_mean_cross_count"] = False
    data["regime_reject_reasons"] = ""
    data["regime_rejected_signal"] = 0

    if trend_ema_length > 0 and (
        regime_threshold_bps > 0
        or trend_adx_threshold > 0
        or trend_extension_atr > 0
        or max_efficiency_ratio > 0
        or min_mean_crosses > 0
    ):
        sideways_regime = ~(slope_trend | adx_trend | extension_trend | efficient_trend | insufficient_crosses)
        rejected_by_regime = (short_reentry | long_reentry) & ~sideways_regime
        data.loc[rejected_by_regime, "rejected_by_regime_threshold_bps"] = slope_trend[rejected_by_regime]
        data.loc[rejected_by_regime, "rejected_by_trend_adx_threshold"] = adx_trend[rejected_by_regime]
        data.loc[rejected_by_regime, "rejected_by_trend_extension_atr"] = extension_trend[rejected_by_regime]
        data.loc[rejected_by_regime, "rejected_by_efficiency_ratio"] = efficient_trend[rejected_by_regime]
        data.loc[rejected_by_regime, "rejected_by_mean_cross_count"] = insufficient_crosses[rejected_by_regime]
        data.loc[rejected_by_regime, "regime_rejected_signal"] = data.loc[rejected_by_regime, "pre_regime_signal"]
        reject_reason_columns = [
            ("rejected_by_regime_threshold_bps", "regime_threshold_bps"),
            ("rejected_by_trend_adx_threshold", "trend_adx_threshold"),
            ("rejected_by_trend_extension_atr", "trend_extension_atr"),
            ("rejected_by_efficiency_ratio", "max_efficiency_ratio"),
            ("rejected_by_mean_cross_count", "min_mean_crosses"),
        ]
        for column_name, label in reject_reason_columns:
            matches = rejected_by_regime & data[column_name]
            if matches.any():
                current = data.loc[matches, "regime_reject_reasons"]
                data.loc[matches, "regime_reject_reasons"] = current.where(current == "", current + "|") + label
        short_reentry &= sideways_regime
        long_reentry &= sideways_regime
        data["regime"] = "sideways"
        data.loc[~sideways_regime, "regime"] = "trend"
    else:
        data["regime"] = "sideways"

    if volume_max_mult > 0:
        quiet_enough = data[volume_col] <= data["volume_ma"] * volume_max_mult
        short_reentry &= quiet_enough
        long_reentry &= quiet_enough

    data["signal"] = 0
    if position_mode in {"both", "short"}:
        data.loc[short_reentry, "signal"] = -1
    if position_mode in {"both", "long"}:
        data.loc[long_reentry, "signal"] = 1

    equity = initial_capital
    position = 0
    entry_price: float | None = None
    entry_atr: float | None = None
    entry_time = None
    hold_bars = 0
    cooldown_left = 0
    peak_price = 0.0
    trough_price = 0.0
    stop_price: float | None = None
    take_price: float | None = None
    equity_curve: list[float] = []
    trades: list[dict] = []
    prev_close: float | None = None

    for row in data.itertuples(index=False):
        timestamp = getattr(row, timestamp_col)
        close = float(getattr(row, close_col))
        high = float(getattr(row, high_col))
        low = float(getattr(row, low_col))
        atr = float(getattr(row, "atr"))
        rolling_mean = float(getattr(row, "rolling_mean")) if pd.notna(getattr(row, "rolling_mean")) else close
        signal = int(getattr(row, "signal"))
        regime = str(getattr(row, "regime"))
        trend_ema_value = getattr(row, "trend_ema")
        if pd.isna(trend_ema_value):
            trend_signal = "disabled"
        elif close > float(trend_ema_value):
            trend_signal = "bullish"
        elif close < float(trend_ema_value):
            trend_signal = "bearish"
        else:
            trend_signal = "flat"

        if prev_close is not None and position != 0:
            bar_return = position * ((close / prev_close) - 1.0)
            equity *= 1.0 + bar_return
            hold_bars += 1

        if position != 0 and entry_price is not None and entry_atr is not None:
            if position == 1:
                peak_price = max(peak_price, high)
                dynamic_stop = peak_price - (trail_atr_mult * atr) if trail_atr_mult > 0 else stop_price
                stop_hit = stop_price is not None and low <= stop_price
                trail_hit = trail_atr_mult > 0 and dynamic_stop is not None and low <= dynamic_stop
                target_hit = take_price is not None and high >= take_price
                mean_hit = exit_on_midline and hold_bars >= min_hold_bars and close >= rolling_mean
                opposite_hit = exit_on_opposite_signal and hold_bars >= min_hold_bars and signal == -1
                if stop_hit or trail_hit or target_hit or mean_hit or opposite_hit:
                    exit_price = close
                    if stop_hit and stop_price is not None:
                        exit_reason = "stop_loss"
                        exit_price = stop_price
                    elif trail_hit and dynamic_stop is not None:
                        exit_reason = "trailing_stop"
                        exit_price = dynamic_stop
                    elif target_hit and take_price is not None:
                        exit_reason = "take_profit"
                        exit_price = take_price
                    elif mean_hit:
                        exit_reason = "midline_exit"
                    else:
                        exit_reason = "opposite_signal"
                    trade_return_pct = ((exit_price * (1 - fee_rate)) / (entry_price * (1 + fee_rate)) - 1.0) * 100
                    equity *= 1.0 - fee_rate
                    trades[-1].update({
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "return_pct": trade_return_pct,
                        "exit_reason": exit_reason,
                    })
                    position = 0
                    entry_price = None
                    entry_atr = None
                    stop_price = None
                    take_price = None
                    hold_bars = 0
                    cooldown_left = cooldown_bars
            else:
                trough_price = min(trough_price, low)
                dynamic_stop = trough_price + (trail_atr_mult * atr) if trail_atr_mult > 0 else stop_price
                stop_hit = stop_price is not None and high >= stop_price
                trail_hit = trail_atr_mult > 0 and dynamic_stop is not None and high >= dynamic_stop
                target_hit = take_price is not None and low <= take_price
                mean_hit = exit_on_midline and hold_bars >= min_hold_bars and close <= rolling_mean
                opposite_hit = exit_on_opposite_signal and hold_bars >= min_hold_bars and signal == 1
                if stop_hit or trail_hit or target_hit or mean_hit or opposite_hit:
                    exit_price = close
                    if stop_hit and stop_price is not None:
                        exit_reason = "stop_loss"
                        exit_price = stop_price
                    elif trail_hit and dynamic_stop is not None:
                        exit_reason = "trailing_stop"
                        exit_price = dynamic_stop
                    elif target_hit and take_price is not None:
                        exit_reason = "take_profit"
                        exit_price = take_price
                    elif mean_hit:
                        exit_reason = "midline_exit"
                    else:
                        exit_reason = "opposite_signal"
                    trade_return_pct = ((entry_price * (1 - fee_rate)) / (exit_price * (1 + fee_rate)) - 1.0) * 100
                    equity *= 1.0 - fee_rate
                    trades[-1].update({
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "return_pct": trade_return_pct,
                        "exit_reason": exit_reason,
                    })
                    position = 0
                    entry_price = None
                    entry_atr = None
                    stop_price = None
                    take_price = None
                    hold_bars = 0
                    cooldown_left = cooldown_bars

        if position == 0 and cooldown_left > 0:
            cooldown_left -= 1

        if position == 0 and cooldown_left == 0 and signal != 0 and atr > 0:
            equity *= 1.0 - fee_rate
            position = signal
            entry_price = close
            entry_atr = atr
            entry_time = timestamp
            hold_bars = 0
            peak_price = high
            trough_price = low
            if position == 1:
                stop_price = close - (stop_atr_mult * atr) if stop_atr_mult > 0 else None
                take_price = close + (take_atr_mult * atr) if take_atr_mult > 0 else None
                side = "long"
            else:
                stop_price = close + (stop_atr_mult * atr) if stop_atr_mult > 0 else None
                take_price = close - (take_atr_mult * atr) if take_atr_mult > 0 else None
                side = "short"
            trades.append({
                "trade_number": len(trades) + 1,
                "side": side,
                "entry_time": entry_time,
                "entry_price": close,
                "entry_signal": signal,
                "entry_trend_signal": trend_signal,
                "entry_regime": regime,
                "exit_time": None,
                "exit_price": None,
                "return_pct": None,
                "exit_reason": None,
            })

        equity_curve.append(equity)
        prev_close = close

    if position != 0 and entry_price is not None:
        last_close = float(data.iloc[-1][close_col])
        last_time = data.iloc[-1][timestamp_col]
        if position == 1:
            trade_return_pct = ((last_close * (1 - fee_rate)) / (entry_price * (1 + fee_rate)) - 1.0) * 100
        else:
            trade_return_pct = ((entry_price * (1 - fee_rate)) / (last_close * (1 + fee_rate)) - 1.0) * 100
        equity *= 1.0 - fee_rate
        trades[-1].update({
            "exit_time": last_time,
            "exit_price": last_close,
            "return_pct": trade_return_pct,
            "exit_reason": "end_of_data",
        })
        if equity_curve:
            equity_curve[-1] = equity

    data["equity"] = equity_curve
    data["entry_price_marker"] = pd.NA
    data["exit_price_marker"] = pd.NA
    data["entry_side_marker"] = pd.NA
    data["exit_side_marker"] = pd.NA
    data["entry_trade_number_marker"] = pd.NA
    data["exit_trade_number_marker"] = pd.NA

    for trade in trades:
        entry_mask = data[timestamp_col] == trade["entry_time"]
        data.loc[entry_mask, "entry_price_marker"] = trade["entry_price"]
        data.loc[entry_mask, "entry_side_marker"] = trade["side"]
        data.loc[entry_mask, "entry_trade_number_marker"] = trade["trade_number"]
        if trade["exit_time"] is not None and trade["exit_price"] is not None:
            exit_mask = data[timestamp_col] == trade["exit_time"]
            data.loc[exit_mask, "exit_price_marker"] = trade["exit_price"]
            data.loc[exit_mask, "exit_side_marker"] = trade["side"]
            data.loc[exit_mask, "exit_trade_number_marker"] = trade["trade_number"]

    data.attrs.update({
        "lookback": lookback,
        "zscore_threshold": zscore_threshold,
        "atr_period": atr_period,
        "stop_atr_mult": stop_atr_mult,
        "take_atr_mult": take_atr_mult,
        "trail_atr_mult": trail_atr_mult,
        "trend_ema_length": trend_ema_length,
        "trend_slope_lookback": trend_slope_lookback,
        "regime_threshold_bps": regime_threshold_bps,
        "trend_adx_period": trend_adx_period,
        "trend_adx_threshold": trend_adx_threshold,
        "trend_extension_atr": trend_extension_atr,
        "max_efficiency_ratio": max_efficiency_ratio,
        "min_mean_crosses": min_mean_crosses,
        "volume_ma_period": volume_ma_period,
        "volume_max_mult": volume_max_mult,
        "reentry_buffer_bps": reentry_buffer_bps,
        "exit_on_midline": exit_on_midline,
        "exit_on_opposite_signal": exit_on_opposite_signal,
        "position_mode": position_mode,
        "min_hold_bars": min_hold_bars,
        "cooldown_bars": cooldown_bars,
    })
    return data, trades


def optimize_parameters(
    df: pd.DataFrame,
    timestamp_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    atr_period: int,
    trail_atr_mult: float,
    trend_ema_length: int,
    trend_slope_lookback: int,
    regime_threshold_bps: float,
    trend_adx_period: int,
    trend_adx_threshold: float,
    trend_extension_atr: float,
    max_efficiency_ratio: float,
    min_mean_crosses: int,
    volume_ma_period: int,
    volume_max_mult: float,
    reentry_buffer_bps: float,
    min_hold_bars: int,
    cooldown_bars: int,
    initial_capital: float,
    fee_rate: float,
    min_trades: int,
    allowed_position_modes: list[str],
) -> dict[str, float | int | str]:
    best: dict[str, float | int | str] | None = None
    regime_candidates = [regime_threshold_bps] if regime_threshold_bps > 0 else [0.0]
    if trend_ema_length > 0 and regime_threshold_bps > 0:
        regime_candidates = OPT_REGIME_THRESHOLD_BPS

    for lookback, zscore, stop_mult, take_mult, regime_bps, mode, exit_on_midline, exit_on_opposite_signal in itertools.product(
        OPT_LOOKBACKS,
        OPT_ZSCORES,
        OPT_STOP_ATR_MULTS,
        OPT_TAKE_ATR_MULTS,
        regime_candidates,
        allowed_position_modes,
        OPT_EXIT_ON_MIDLINE,
        OPT_EXIT_ON_OPPOSITE_SIGNAL,
    ):
        result, trades = run_backtest(
            df=df,
            timestamp_col=timestamp_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            lookback=lookback,
            zscore_threshold=zscore,
            atr_period=atr_period,
            stop_atr_mult=stop_mult,
            take_atr_mult=take_mult,
            trail_atr_mult=trail_atr_mult,
            trend_ema_length=trend_ema_length,
            trend_slope_lookback=trend_slope_lookback,
            regime_threshold_bps=regime_bps,
            trend_adx_period=trend_adx_period,
            trend_adx_threshold=trend_adx_threshold,
            trend_extension_atr=trend_extension_atr,
            max_efficiency_ratio=max_efficiency_ratio,
            min_mean_crosses=min_mean_crosses,
            volume_ma_period=volume_ma_period,
            volume_max_mult=volume_max_mult,
            reentry_buffer_bps=reentry_buffer_bps,
            min_hold_bars=min_hold_bars,
            cooldown_bars=cooldown_bars,
            exit_on_midline=exit_on_midline,
            exit_on_opposite_signal=exit_on_opposite_signal,
            position_mode=mode,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
        )
        closed = [t for t in trades if t["return_pct"] is not None]
        if len(closed) < min_trades:
            continue
        final_equity = float(result["equity"].iloc[-1])
        strategy_return = ((final_equity / initial_capital) - 1.0) * 100
        drawdown = ((result["equity"] - result["equity"].cummax()) / result["equity"].cummax() * 100).min()
        profits = [float(t["return_pct"]) for t in closed if float(t["return_pct"]) > 0]
        losses = [abs(float(t["return_pct"])) for t in closed if float(t["return_pct"]) <= 0]
        profit_factor = (sum(profits) / sum(losses)) if losses else float("inf") if profits else 0.0
        score = strategy_return + (0.2 * drawdown) + (0.5 * min(profit_factor, 3.0))
        if best is None or score > float(best["score"]):
            best = {
                "score": float(score),
                "strategy_return": float(strategy_return),
                "max_drawdown": float(drawdown),
                "profit_factor": float(profit_factor),
                "trades": int(len(closed)),
                "lookback": int(lookback),
                "zscore_threshold": float(zscore),
                "stop_atr_mult": float(stop_mult),
                "take_atr_mult": float(take_mult),
                "regime_threshold_bps": float(regime_bps),
                "position_mode": str(mode),
                "exit_on_midline": bool(exit_on_midline),
                "exit_on_opposite_signal": bool(exit_on_opposite_signal),
            }
    if best is None:
        raise ValueError("No optimization candidate met constraints. Try lowering --min-trades.")
    return best


def build_summary_text(data: pd.DataFrame, trades: list[dict], initial_capital: float, input_csv: Path) -> str:
    final_equity = float(data["equity"].iloc[-1])
    strategy_return = ((final_equity / initial_capital) - 1.0) * 100
    buy_hold = ((data["close"].iloc[-1] / data["close"].iloc[0]) - 1.0) * 100
    drawdown = ((data["equity"] - data["equity"].cummax()) / data["equity"].cummax() * 100).min()
    closed = [t for t in trades if t["return_pct"] is not None]
    long_trades = [t for t in closed if t["side"] == "long"]
    short_trades = [t for t in closed if t["side"] == "short"]
    wins = [t for t in closed if float(t["return_pct"]) > 0]
    losses = [abs(float(t["return_pct"])) for t in closed if float(t["return_pct"]) <= 0]
    profits = [float(t["return_pct"]) for t in closed if float(t["return_pct"]) > 0]
    win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
    avg_trade = (sum(float(t["return_pct"]) for t in closed) / len(closed)) if closed else 0.0
    profit_factor = (sum(profits) / sum(losses)) if losses else float("inf") if profits else 0.0

    lines = [
        "Mean Reversion Long/Short Backtest",
        f"Input CSV: {input_csv}",
        f"Rows: {len(data)}",
        f"Trades: {len(closed)}",
        f"Position mode: {data.attrs.get('position_mode', 'both')}",
        (
            "Config: "
            f"lookback={int(data.attrs.get('lookback', 0))}, "
            f"zscore_threshold={float(data.attrs.get('zscore_threshold', 0.0)):.2f}, "
            f"atr_period={int(data.attrs.get('atr_period', 0))}, "
            f"stop_atr_mult={float(data.attrs.get('stop_atr_mult', 0.0)):.2f}, "
            f"take_atr_mult={float(data.attrs.get('take_atr_mult', 0.0)):.2f}, "
            f"trail_atr_mult={float(data.attrs.get('trail_atr_mult', 0.0)):.2f}, "
            f"trend_ema_length={int(data.attrs.get('trend_ema_length', 0))}, "
            f"trend_slope_lookback={int(data.attrs.get('trend_slope_lookback', 0))}, "
            f"regime_threshold_bps={float(data.attrs.get('regime_threshold_bps', 0.0)):.2f}, "
            f"trend_adx_period={int(data.attrs.get('trend_adx_period', 0))}, "
            f"trend_adx_threshold={float(data.attrs.get('trend_adx_threshold', 0.0)):.2f}, "
            f"trend_extension_atr={float(data.attrs.get('trend_extension_atr', 0.0)):.2f}, "
            f"max_efficiency_ratio={float(data.attrs.get('max_efficiency_ratio', 0.0)):.2f}, "
            f"min_mean_crosses={int(data.attrs.get('min_mean_crosses', 0))}, "
            f"volume_ma_period={int(data.attrs.get('volume_ma_period', 0))}, "
            f"volume_max_mult={float(data.attrs.get('volume_max_mult', 0.0)):.2f}, "
            f"reentry_buffer_bps={float(data.attrs.get('reentry_buffer_bps', 0.0)):.0f}, "
            f"exit_on_midline={bool(data.attrs.get('exit_on_midline', False))}, "
            f"exit_on_opposite_signal={bool(data.attrs.get('exit_on_opposite_signal', False))}, "
            f"min_hold_bars={int(data.attrs.get('min_hold_bars', 0))}, "
            f"cooldown_bars={int(data.attrs.get('cooldown_bars', 0))}"
        ),
        f"Long trades: {len(long_trades)}",
        f"Short trades: {len(short_trades)}",
        f"Win rate: {win_rate:.2f}%",
        f"Profit factor: {profit_factor:.2f}",
        f"Average trade return: {avg_trade:.2f}%",
        f"Initial capital: {initial_capital:,.2f}",
        f"Final equity: {final_equity:,.2f}",
        f"Strategy return: {strategy_return:.2f}%",
        f"Buy and hold return: {buy_hold:.2f}%",
        f"Max drawdown: {drawdown:.2f}%",
    ]

    if closed:
        lines.append("")
        lines.append("Recent trades:")
        for trade in closed[-5:]:
            lines.append(
                f"{trade['side']} | {trade['entry_time']} -> {trade['exit_time']} | "
                f"entry={trade['entry_price']:.4f} exit={trade['exit_price']:.4f} "
                f"return={trade['return_pct']:.2f}% reason={trade['exit_reason']}"
            )

    return "\n".join(lines)


def print_summary(data: pd.DataFrame, trades: list[dict], initial_capital: float, input_csv: Path) -> str:
    summary = build_summary_text(data, trades, initial_capital, input_csv)
    print(summary)
    return summary


def build_output_dir(strategy_name: str, timestamp: str, base_dir: Path = Path("reports")) -> Path:
    strategy_slug = re.sub(r"[^a-z0-9]+", "_", strategy_name.strip().lower()).strip("_")
    output_dir = base_dir / f"{strategy_slug}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_report(summary_text: str, report_dir: Path, timestamp: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"backtest_report_{timestamp}.txt"
    report_path.write_text(summary_text + "\n", encoding="utf-8")
    return report_path


def save_trades_csv(
    trades: list[dict],
    report_dir: Path,
    timestamp: str,
    output_path: Path | None = None,
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "side",
                "entry_time",
                "entry_price",
                "entry_signal",
                "entry_trend_signal",
                "entry_regime",
                "exit_time",
                "exit_price",
                "return_pct",
                "exit_reason",
            ]
        )

    if "entry_time" in trades_df.columns:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], errors="coerce")
    if "exit_time" in trades_df.columns:
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")

    default_path = report_dir / f"trade_results_{timestamp}.csv"
    output_paths = [default_path]
    if output_path is not None and output_path != default_path:
        output_paths.append(output_path)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(path, index=False)

    return output_paths


def save_plot_data_csv(
    data: pd.DataFrame,
    timestamp_col: str,
    close_col: str,
    report_dir: Path,
    timestamp: str,
    output_path: Path | None = None,
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_columns = [
        timestamp_col,
        close_col,
        "rolling_mean",
        "upper_band",
        "lower_band",
        "trend_ema",
        "equity",
        "entry_price_marker",
        "entry_side_marker",
        "entry_trade_number_marker",
        "exit_price_marker",
        "exit_side_marker",
        "exit_trade_number_marker",
    ]
    missing = [column for column in plot_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing plot data column(s): {', '.join(missing)}")

    plot_data = data.loc[:, plot_columns].copy()
    plot_data.rename(columns={timestamp_col: "timestamp", close_col: "close"}, inplace=True)
    plot_data["timestamp"] = pd.to_datetime(plot_data["timestamp"], errors="coerce")

    default_path = report_dir / f"plot_data_{timestamp}.csv"
    output_paths = [default_path]
    if output_path is not None and output_path != default_path:
        output_paths.append(output_path)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_data.to_csv(path, index=False)

    return output_paths


def save_signal_diagnostics_csv(
    data: pd.DataFrame,
    timestamp_col: str,
    close_col: str,
    report_dir: Path,
    timestamp: str,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_mask = (data["pre_regime_signal"] != 0) | (data["signal"] != 0) | (data["regime_rejected_signal"] != 0)
    diagnostics_df = data.loc[
        diagnostic_mask,
        [
            timestamp_col,
            close_col,
            "pre_regime_signal",
            "signal",
            "regime_rejected_signal",
            "regime_reject_reasons",
            "rejected_by_regime_threshold_bps",
            "rejected_by_trend_adx_threshold",
            "rejected_by_trend_extension_atr",
            "rejected_by_efficiency_ratio",
            "rejected_by_mean_cross_count",
            "trend_slope_bps",
            "adx",
            "trend_extension_atr",
            "efficiency_ratio",
            "mean_cross_count",
            "regime",
            "trend_ema",
        ],
    ].copy()
    diagnostics_df.rename(
        columns={
            timestamp_col: "timestamp",
            close_col: "close",
            "signal": "final_signal",
        },
        inplace=True,
    )
    diagnostics_df["candidate_side"] = diagnostics_df["pre_regime_signal"].map({1: "long", -1: "short"}).fillna("")
    diagnostics_df["final_side"] = diagnostics_df["final_signal"].map({1: "long", -1: "short"}).fillna("")
    diagnostics_df["rejected_side"] = diagnostics_df["regime_rejected_signal"].map({1: "long", -1: "short"}).fillna("")

    path = report_dir / f"signal_diagnostics_{timestamp}.csv"
    diagnostics_df.to_csv(path, index=False)
    return path


def plot_backtest(
    data: pd.DataFrame,
    timestamp_col: str,
    close_col: str,
    plot_path: Path | None = None,
    extra_plot_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skipping plot export: matplotlib is not installed.")
        return

    fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    timestamps = data[timestamp_col]
    ax_price.plot(timestamps, data[close_col], label="Close", linewidth=1.2)
    ax_price.plot(timestamps, data["rolling_mean"], label="Rolling Mean", linewidth=1.0)
    ax_price.plot(timestamps, data["upper_band"], label="Upper Range", linewidth=0.9)
    ax_price.plot(timestamps, data["lower_band"], label="Lower Range", linewidth=0.9)
    if "trend_ema" in data.columns and data["trend_ema"].notna().any():
        ax_price.plot(timestamps, data["trend_ema"], label="Trend EMA", linewidth=1.0)

    entry_points = data.dropna(subset=["entry_price_marker"])
    exit_points = data.dropna(subset=["exit_price_marker"])
    long_entries = entry_points[entry_points["entry_side_marker"] == "long"]
    short_entries = entry_points[entry_points["entry_side_marker"] == "short"]
    long_exits = exit_points[exit_points["exit_side_marker"] == "long"]
    short_exits = exit_points[exit_points["exit_side_marker"] == "short"]

    if not long_entries.empty:
        ax_price.scatter(long_entries[timestamp_col], long_entries["entry_price_marker"], marker="^", s=70, label="Long Entry")
    if not short_entries.empty:
        ax_price.scatter(short_entries[timestamp_col], short_entries["entry_price_marker"], marker="v", s=70, label="Short Entry")
    if not long_exits.empty:
        ax_price.scatter(long_exits[timestamp_col], long_exits["exit_price_marker"], marker="v", s=60, label="Long Exit")
    if not short_exits.empty:
        ax_price.scatter(short_exits[timestamp_col], short_exits["exit_price_marker"], marker="^", s=60, label="Short Exit")

    for points, price_col, number_col, y_offset in [
        (entry_points, "entry_price_marker", "entry_trade_number_marker", 8),
        (exit_points, "exit_price_marker", "exit_trade_number_marker", -12),
    ]:
        for point in points.itertuples(index=False):
            trade_number = getattr(point, number_col)
            if pd.isna(trade_number):
                continue
            ax_price.annotate(
                str(int(trade_number)),
                (getattr(point, timestamp_col), getattr(point, price_col)),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    ax_price.set_title("Mean Reversion Entries and Exits")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend()

    ax_equity.plot(timestamps, data["equity"], label="Equity", linewidth=1.3)
    ax_equity.set_title("Equity Curve")
    ax_equity.set_ylabel("Equity")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, alpha=0.25)
    ax_equity.legend()

    fig.tight_layout()
    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {plot_path}")
    if extra_plot_path is not None and extra_plot_path != plot_path:
        extra_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {extra_plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_columns = pd.read_csv(args.csv, nrows=0).columns.tolist()
    args.timestamp_col = resolve_column_name(args.timestamp_col, csv_columns)
    export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = build_output_dir(export_timestamp,"mean_reversion_with_filter")

    selected_lookback = args.lookback
    selected_zscore = args.zscore_threshold
    selected_stop = args.stop_atr_mult
    selected_take = args.take_atr_mult
    selected_regime_threshold_bps = args.regime_threshold_bps
    selected_mode = args.position_mode
    selected_exit_on_midline = args.exit_on_midline
    selected_exit_on_opposite_signal = args.exit_on_opposite_signal

    required_columns = [
        args.timestamp_col,
        args.open_col,
        args.high_col,
        args.low_col,
        args.close_col,
        args.volume_col,
    ]
    required_rows = max(
        args.lookback + 2,
        args.atr_period + 2,
        args.trend_ema_length + args.trend_slope_lookback + 2 if args.trend_ema_length > 0 else 2,
        args.trend_adx_period + 2,
    )
    if args.optimize:
        required_rows = max(required_rows, max(OPT_LOOKBACKS) + 2)
    data = load_data(args.csv, required_columns, required_rows)

    optimization_modes = OPT_POSITION_MODES if args.position_mode == "both" else [args.position_mode]
    if args.optimize:
        best = optimize_parameters(
            df=data,
            timestamp_col=args.timestamp_col,
            open_col=args.open_col,
            high_col=args.high_col,
            low_col=args.low_col,
            close_col=args.close_col,
            volume_col=args.volume_col,
            atr_period=args.atr_period,
            trail_atr_mult=args.trail_atr_mult,
            trend_ema_length=args.trend_ema_length,
            trend_slope_lookback=args.trend_slope_lookback,
            regime_threshold_bps=args.regime_threshold_bps,
            trend_adx_period=args.trend_adx_period,
            trend_adx_threshold=args.trend_adx_threshold,
            trend_extension_atr=args.trend_extension_atr,
            max_efficiency_ratio=args.max_efficiency_ratio,
            min_mean_crosses=args.min_mean_crosses,
            volume_ma_period=args.volume_ma_period,
            volume_max_mult=args.volume_max_mult,
            reentry_buffer_bps=args.reentry_buffer_bps,
            min_hold_bars=args.min_hold_bars,
            cooldown_bars=args.cooldown_bars,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            min_trades=args.min_trades,
            allowed_position_modes=optimization_modes,
        )
        selected_lookback = int(best["lookback"])
        selected_zscore = float(best["zscore_threshold"])
        selected_stop = float(best["stop_atr_mult"])
        selected_take = float(best["take_atr_mult"])
        selected_regime_threshold_bps = float(best["regime_threshold_bps"])
        selected_mode = str(best["position_mode"])
        selected_exit_on_midline = bool(best["exit_on_midline"])
        selected_exit_on_opposite_signal = bool(best["exit_on_opposite_signal"])
        print(
            "Optimization selected "
            f"lookback={selected_lookback} zscore={selected_zscore:.2f} "
            f"stop_atr_mult={selected_stop:.2f} take_atr_mult={selected_take:.2f} "
            f"regime_threshold_bps={float(best['regime_threshold_bps']):.2f} "
            f"position_mode={selected_mode} "
            f"exit_on_midline={selected_exit_on_midline} "
            f"exit_on_opposite_signal={selected_exit_on_opposite_signal} "
            f"(return={float(best['strategy_return']):.2f}% drawdown={float(best['max_drawdown']):.2f}% "
            f"pf={float(best['profit_factor']):.2f} trades={int(best['trades'])})"
        )

    result, trades = run_backtest(
        df=data,
        timestamp_col=args.timestamp_col,
        open_col=args.open_col,
        high_col=args.high_col,
        low_col=args.low_col,
        close_col=args.close_col,
        volume_col=args.volume_col,
        lookback=selected_lookback,
        zscore_threshold=selected_zscore,
        atr_period=args.atr_period,
        stop_atr_mult=selected_stop,
        take_atr_mult=selected_take,
        trail_atr_mult=args.trail_atr_mult,
        trend_ema_length=args.trend_ema_length,
        trend_slope_lookback=args.trend_slope_lookback,
        regime_threshold_bps=selected_regime_threshold_bps,
        trend_adx_period=args.trend_adx_period,
        trend_adx_threshold=args.trend_adx_threshold,
        trend_extension_atr=args.trend_extension_atr,
        max_efficiency_ratio=args.max_efficiency_ratio,
        min_mean_crosses=args.min_mean_crosses,
        volume_ma_period=args.volume_ma_period,
        volume_max_mult=args.volume_max_mult,
        reentry_buffer_bps=args.reentry_buffer_bps,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        exit_on_midline=selected_exit_on_midline,
        exit_on_opposite_signal=selected_exit_on_opposite_signal,
        position_mode=selected_mode,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
    )

    summary_text = print_summary(
        result.rename(columns={args.close_col: "close"}),
        trades,
        args.initial_capital,
        args.csv,
    )
    report_path = save_report(summary_text, output_dir, export_timestamp)
    print(f"Saved report to: {report_path}")
    for trade_csv_path in save_trades_csv(trades, output_dir, export_timestamp, args.trades_csv_path):
        print(f"Saved trade results CSV to: {trade_csv_path}")
    for plot_data_csv_path in save_plot_data_csv(
        result,
        args.timestamp_col,
        args.close_col,
        output_dir,
        export_timestamp,
        args.plot_data_csv_path,
    ):
        print(f"Saved plot data CSV to: {plot_data_csv_path}")
    signal_diagnostics_path = save_signal_diagnostics_csv(
        result,
        args.timestamp_col,
        args.close_col,
        output_dir,
        export_timestamp,
    )
    print(f"Saved signal diagnostics CSV to: {signal_diagnostics_path}")

    should_render_plot = args.plot or args.save_plot or args.plot_path is not None
    if should_render_plot:
        plot_path = args.plot_path if args.plot_path is not None else output_dir / f"backtest_plot_{export_timestamp}.png"
        plot_backtest(
            data=result,
            timestamp_col=args.timestamp_col,
            close_col=args.close_col,
            plot_path=plot_path if (args.save_plot or args.plot_path is not None) else None,
            show_plot=args.plot,
        )


if __name__ == "__main__":
    main()
