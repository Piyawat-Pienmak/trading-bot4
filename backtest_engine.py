from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    df: pd.DataFrame
    equity: pd.Series
    trades: List[dict]
    events: List[dict]
    summary: dict


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "time" not in out.columns:
        if "open_time" not in out.columns:
            raise ValueError("Dataframe missing 'time' or 'open_time' column for timestamp conversion")
        out["time"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    else:
        out["time"] = pd.to_datetime(out["time"], utc=True)
    out["time"] = out["time"].dt.tz_convert("Asia/Bangkok")
    return out.sort_values("time").reset_index(drop=True)


def calc_position_size(equity: float, risk_pct: float, entry_price: float, sl_price: float) -> float:
    risk_usdt = equity * risk_pct
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= 0:
        return 0.0
    return risk_usdt / risk_per_unit


def calc_max_drawdown(equity_series: List[float]) -> float:
    eq = np.array(equity_series, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = eq[0]
    max_dd = 0.0
    for x in eq:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def format_summary_lines(summary: dict, trades: List[dict]) -> List[str]:
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    losses = sum(1 for t in trades if t["net_pnl"] <= 0)
    initial = summary.get("initial_equity", 0.0) or 0.0
    final = summary.get("final_equity", 0.0) or 0.0
    pct_pnl = (summary.get("total_net_pnl", 0.0) / initial * 100) if initial else 0.0
    max_base = max(initial, final)
    pct_dd = (summary.get("max_drawdown", 0.0) / max_base * 100) if max_base else 0.0
    return [
        f"initial_equity: {summary.get('initial_equity', 0):.4f}",
        f"final_equity:   {summary.get('final_equity', 0):.4f}",
        f"total_net_pnl:  {summary.get('total_net_pnl', 0):.4f}   ({pct_pnl:+.1f}%)",
        f"num_trades:     {summary.get('num_trades', 0)}",
        f"win_rate:       {summary.get('win_rate', 0):.4f}   ({wins} wins, {losses} losses)",
        f"avg_win:        {summary.get('avg_win', 0):.4f}",
        f"avg_loss:       {summary.get('avg_loss', 0):.4f}",
        f"profit_factor:  {summary.get('profit_factor', 0):.4f}",
        f"max_drawdown:   {summary.get('max_drawdown', 0):.4f}   ({pct_dd:+.1f}% from peak)",
    ]


def _simulate(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    initial_equity: float,
    risk_pct: float,
    fee_rate: float,
) -> Tuple[pd.Series, List[dict], List[dict], dict]:
    equity = initial_equity
    equity_curve: List[float] = []
    equity_times: List[pd.Timestamp] = []
    trades: List[dict] = []
    events: List[dict] = []
    position = "flat"
    entry_price = 0.0
    stop_price = 0.0
    take_profit = 0.0
    position_size = 0.0
    entry_time = None
    last_exit_index = -9999
    warmup = strategy.warmup()

    for i in range(len(df)):
        row = df.iloc[i]
        time = row["time"]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if i < warmup:
            equity_times.append(time)
            equity_curve.append(equity)
            continue

        exit_price = None
        exit_reason = None

        if position != "flat":
            hit_sl = low <= stop_price if position == "long" else high >= stop_price
            hit_tp = high >= take_profit if position == "long" else low <= take_profit
            if hit_sl and not hit_tp:
                exit_price = stop_price
                exit_reason = "SL"
            elif hit_tp and not hit_sl:
                exit_price = take_profit
                exit_reason = "TP"
            elif hit_sl and hit_tp:
                exit_price = stop_price
                exit_reason = "SL"
            if exit_price is None:
                extra_exit = strategy.check_exit(i, df, position, entry_price, stop_price, take_profit)
                if extra_exit:
                    exit_price = extra_exit.exit_price
                    exit_reason = extra_exit.reason

            if exit_price is not None:
                notional_entry = entry_price * position_size
                notional_exit = exit_price * position_size
                fee_entry = notional_entry * fee_rate
                fee_exit = notional_exit * fee_rate
                gross_pnl = (
                    (exit_price - entry_price) * position_size
                    if position == "long"
                    else (entry_price - exit_price) * position_size
                )
                net_pnl = gross_pnl - (fee_entry + fee_exit)
                equity += net_pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": time,
                        "direction": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": position_size,
                        "sl_price": stop_price,
                        "tp_price": take_profit,
                        "gross_pnl": gross_pnl,
                        "fees": fee_entry + fee_exit,
                        "net_pnl": net_pnl,
                        "exit_reason": exit_reason,
                        "equity_after_trade": equity,
                    }
                )
                events.append(
                    {
                        "event": "exit",
                        "time": time,
                        "dir": position.upper(),
                        "price": exit_price,
                        "qty": position_size,
                        "pnl": net_pnl,
                        "reason": exit_reason,
                        "equity": equity,
                    }
                )
                position = "flat"
                entry_price = 0.0
                stop_price = 0.0
                take_profit = 0.0
                position_size = 0.0
                entry_time = None
                last_exit_index = i

        if position == "flat" and i != last_exit_index:
            entry_sig = strategy.check_entry(i, df, position)
            if entry_sig:
                risk_per_unit = abs(entry_sig.entry_price - entry_sig.stop_price)
                if risk_per_unit > 0:
                    size = calc_position_size(equity, risk_pct, entry_sig.entry_price, entry_sig.stop_price)
                    if size > 0:
                        position = entry_sig.direction
                        entry_price = entry_sig.entry_price
                        stop_price = entry_sig.stop_price
                        take_profit = entry_sig.take_profit
                        position_size = size
                        entry_time = time
                        events.append(
                            {
                                "event": "entry",
                                "time": time,
                                "dir": position.upper(),
                                "price": entry_price,
                                "qty": size,
                                "stop": stop_price,
                                "tp": take_profit,
                                "condition": entry_sig.reason,
                                "equity": equity,
                            }
                        )

        equity_times.append(time)
        equity_curve.append(equity)

    final_equity = equity
    total_net_pnl = final_equity - initial_equity
    num_trades = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    win_rate = len(wins) / num_trades if num_trades else 0.0
    avg_win = float(np.mean([t["net_pnl"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([-t["net_pnl"] for t in losses])) if losses else 0.0
    gross_profit = sum(max(t["net_pnl"], 0.0) for t in trades)
    gross_loss = sum(-min(t["net_pnl"], 0.0) for t in trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_dd = calc_max_drawdown(equity_curve)

    summary = {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_net_pnl": total_net_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }

    eq_series = pd.Series(equity_curve, index=equity_times)
    return eq_series, trades, events, summary


def run_backtest(
    raw_df: pd.DataFrame,
    strategy: BaseStrategy,
    initial_equity: float,
    risk_pct: float,
    fee_rate: float,
) -> BacktestResult:
    base_df = normalize_ohlcv(raw_df)
    prepared = strategy.prepare(base_df)
    prepared = prepared.sort_values("time").reset_index(drop=True)
    equity, trades, events, summary = _simulate(prepared, strategy, initial_equity, risk_pct, fee_rate)
    return BacktestResult(prepared, equity, trades, events, summary)


def run_split_backtest(
    raw_df: pd.DataFrame,
    strategy: BaseStrategy,
    split_ratio: float,
    initial_equity: float,
    risk_pct: float,
    fee_rate: float,
) -> Tuple[BacktestResult, BacktestResult]:
    if split_ratio <= 0 or split_ratio >= 1:
        raise ValueError("split_ratio must be between 0 and 1 (e.g. 0.7 for 70/30)")
    base_df = normalize_ohlcv(raw_df)
    total = len(base_df)
    if total < 2:
        raise ValueError("Not enough rows to split; need at least 2 candles")
    split_idx = int(total * split_ratio)
    split_idx = min(max(split_idx, 1), total - 1)
    df_train = base_df.iloc[:split_idx].copy()
    df_test = base_df.iloc[split_idx:].copy()
    print(f"Total candles: {total}, Train: {len(df_train)}, Test: {len(df_test)}")
    train_res = run_backtest(df_train, strategy, initial_equity, risk_pct, fee_rate)
    test_res = run_backtest(df_test, strategy, initial_equity, risk_pct, fee_rate)
    return train_res, test_res
