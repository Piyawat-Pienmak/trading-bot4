import argparse
import json
import os
import webbrowser
from typing import List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from bot import FuturesBot, Settings


load_dotenv(".env")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA200, EMA200 shifted 10 bars, ATR(14), and prior 24-bar highs/lows."""
    df = df.copy()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema200_10"] = df["ema200"].shift(10)

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14, min_periods=14).mean()

    df["hh24_prev"] = df["high"].rolling(window=24, min_periods=24).max().shift(1)
    df["ll24_prev"] = df["low"].rolling(window=24, min_periods=24).min().shift(1)

    return df


def calc_position_size(equity: float, risk_pct: float, entry_price: float, sl_price: float) -> float:
    """Simple risk-based position size."""
    risk_usdt = equity * risk_pct
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= 0:
        return 0.0
    return risk_usdt / risk_per_unit


def calc_max_drawdown(equity_series: List[float]) -> float:
    """Return max drawdown (absolute) from an equity curve list."""
    eq = np.array(equity_series, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = eq[0]
    max_dd = 0.0
    for x in eq:
        peak = max(peak, x)
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def simulate_trades(
    df: pd.DataFrame,
    initial_equity: float,
    risk_pct: float,
    fee_rate: float,
) -> Tuple[pd.Series, List[dict], List[dict], dict]:
    equity = initial_equity
    equity_curve: List[float] = []
    equity_times: List[pd.Timestamp] = []
    trades: List[dict] = []
    event_log: List[dict] = []
    position = "flat"
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    position_size = 0.0
    entry_time = None
    last_exit_bar_index = -9999

    min_index = max(200, 14, 24) + 10

    for i, row in df.iterrows():
        time = row["time"]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        ema = float(row["ema200"])
        ema_10 = float(row["ema200_10"])
        atr = float(row["atr14"])
        hh_prev = float(row["hh24_prev"]) if not np.isnan(row["hh24_prev"]) else np.nan
        ll_prev = float(row["ll24_prev"]) if not np.isnan(row["ll24_prev"]) else np.nan

        if i < min_index or any(np.isnan(x) for x in [ema, ema_10, atr, hh_prev, ll_prev]):
            equity_times.append(time)
            equity_curve.append(equity)
            continue

        exit_price = None
        exit_reason = None

        if position != "flat":
            if position == "long":
                hit_sl = low <= sl_price
                hit_tp = high >= tp_price
                if hit_sl and not hit_tp:
                    exit_price = sl_price
                    exit_reason = "SL"
                elif hit_tp and not hit_sl:
                    exit_price = tp_price
                    exit_reason = "TP"
                elif hit_sl and hit_tp:
                    exit_price = sl_price
                    exit_reason = "SL"
                if exit_price is None and close < ema:
                    exit_price = close
                    exit_reason = "EMA_break"
            else:
                hit_sl = high >= sl_price
                hit_tp = low <= tp_price
                if hit_sl and not hit_tp:
                    exit_price = sl_price
                    exit_reason = "SL"
                elif hit_tp and not hit_sl:
                    exit_price = tp_price
                    exit_reason = "TP"
                elif hit_sl and hit_tp:
                    exit_price = sl_price
                    exit_reason = "SL"
                if exit_price is None and close > ema:
                    exit_price = close
                    exit_reason = "EMA_break"

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
                        "gross_pnl": gross_pnl,
                        "fees": fee_entry + fee_exit,
                        "net_pnl": net_pnl,
                        "exit_reason": exit_reason,
                    }
                )
                event_log.append(
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
                sl_price = 0.0
                tp_price = 0.0
                position_size = 0.0
                entry_time = None
                last_exit_bar_index = i

        if position == "flat" and i != last_exit_bar_index:
            long_cond = (close > ema) and (ema > ema_10) and (close > hh_prev)
            short_cond = (close < ema) and (ema < ema_10) and (close < ll_prev)
            if long_cond:
                sl = close - 2.0 * atr
                risk_per_unit = close - sl
                if risk_per_unit > 0:
                    tp = close + 3.0 * risk_per_unit
                    size = calc_position_size(equity, risk_pct, close, sl)
                    if size > 0:
                        position = "long"
                        entry_price = close
                        sl_price = sl
                        tp_price = tp
                        position_size = size
                        entry_time = time
                        event_log.append(
                            {
                                "event": "entry",
                                "time": time,
                                "dir": "LONG",
                                "price": entry_price,
                                "qty": size,
                                "stop": sl,
                                "tp": tp,
                                "condition": "EMA up + HH24 breakout",
                                "equity": equity,
                            }
                        )
            elif short_cond:
                sl = close + 2.0 * atr
                risk_per_unit = sl - close
                if risk_per_unit > 0:
                    tp = close - 3.0 * risk_per_unit
                    size = calc_position_size(equity, risk_pct, close, sl)
                    if size > 0:
                        position = "short"
                        entry_price = close
                        sl_price = sl
                        tp_price = tp
                        position_size = size
                        entry_time = time
                        event_log.append(
                            {
                                "event": "entry",
                                "time": time,
                                "dir": "SHORT",
                                "price": entry_price,
                                "qty": size,
                                "stop": sl,
                                "tp": tp,
                                "condition": "EMA down + LL24 breakdown",
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
    return eq_series, trades, event_log, summary


def prepare_backtest_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLC data has indicators, signals, and a unified time column."""
    df = raw_df.copy()
    float_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = compute_indicators(df)
    if "time" not in df:
        if "open_time" not in df:
            raise ValueError("Dataframe missing 'time' or 'open_time' column for timestamp conversion")
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time"] = df["time"].dt.tz_convert("Asia/Bangkok")
    df["signal"] = df.apply(
        lambda r: 1
        if (r["close"] > r["ema200"] and r["ema200"] > r["ema200_10"] and r["close"] > r["hh24_prev"])
        else (-1 if (r["close"] < r["ema200"] and r["ema200"] < r["ema200_10"] and r["close"] < r["ll24_prev"]) else 0),
        axis=1,
    )
    df["signal_prev"] = df["signal"].shift(1).fillna(0)
    df["long_entry"] = (df["signal"] == 1) & (df["signal_prev"] != 1)
    df["short_entry"] = (df["signal"] == -1) & (df["signal_prev"] != -1)
    df["long_exit"] = (df["signal_prev"] == 1) & (df["signal"] != 1)
    df["short_exit"] = (df["signal_prev"] == -1) & (df["signal"] != -1)
    return df.sort_values("time").reset_index(drop=True)


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


def backtest(
    symbol: str,
    interval: str,
    lookback: int,
    initial: float,
    risk_pct: float = 0.03,
    fee_rate: float = 0.0004,
    leverage: int = 2,
) -> Tuple[pd.DataFrame, pd.Series, List[dict], List[dict], Settings, dict]:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    settings = Settings(symbol=symbol, interval=interval, lookback=lookback, live=False, leverage=leverage)
    bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
    df = prepare_backtest_df(bot.fetch_klines())
    eq, trades, event_log, summary = simulate_trades(df, initial, risk_pct, fee_rate)
    return df, eq, trades, event_log, settings, summary


def run_split_backtest(
    df: pd.DataFrame,
    fee_rate: float = 0.0004,
    initial_equity: float = 25.0,
    risk_pct: float = 0.03,
    leverage: int = 2,
    split_ratio: float = 0.7,
) -> Tuple[
    Tuple[pd.DataFrame, pd.Series, List[dict], List[dict], dict],
    Tuple[pd.DataFrame, pd.Series, List[dict], List[dict], dict],
]:
    """Split a dataframe into train/test segments and run the simulator on each."""
    _ = leverage  # present for interface parity with live settings
    if split_ratio <= 0 or split_ratio >= 1:
        raise ValueError("split_ratio must be between 0 and 1 (e.g. 0.7 for 70/30)")
    prepared = prepare_backtest_df(df)
    total = len(prepared)
    if total < 2:
        raise ValueError("Not enough rows to split; need at least 2 candles")
    split_idx = int(total * split_ratio)
    split_idx = min(max(split_idx, 1), total - 1)
    df_train = prepared.iloc[:split_idx].copy()
    df_test = prepared.iloc[split_idx:].copy()
    print(f"Total candles: {total}, Train: {len(df_train)}, Test: {len(df_test)}")

    eq_tr, trades_tr, log_tr, summary_tr = simulate_trades(
        df_train, initial_equity, risk_pct, fee_rate
    )
    print("\n=== TRAIN (old data) ===")
    print("\n".join(format_summary_lines(summary_tr, trades_tr)))

    eq_te, trades_te, log_te, summary_te = simulate_trades(
        df_test, initial_equity, risk_pct, fee_rate
    )
    print("\n=== TEST (new data) ===")
    print("\n".join(format_summary_lines(summary_te, trades_te)))

    return (
        df_train,
        eq_tr,
        trades_tr,
        log_tr,
        summary_tr,
    ), (
        df_test,
        eq_te,
        trades_te,
        log_te,
        summary_te,
    )


def plot_results_html(
    df: pd.DataFrame,
    equity: pd.Series,
    symbol: str,
    output_path: str,
    open_report: bool = True,
    theme: str = "dark",
    event_log: List[dict] | None = None,
) -> None:
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    is_dark = theme.lower() == "dark"
    bg_color = "#0b1224" if is_dark else "#ffffff"
    panel_color = "#0f172a" if is_dark else "#ffffff"
    grid_color = "#1f2937" if is_dark else "#e5e5e5"
    text_color = "#e5e7eb" if is_dark else "#1a1a1a"
    spike_color = "#475569" if is_dark else "#aaaaaa"
    candle_up = "#22c55e" if is_dark else "#2d8a5f"
    candle_down = "#f87171" if is_dark else "#e15759"
    candle_up_fill = "rgba(34,197,94,0.35)" if is_dark else "#6fcf97"
    candle_down_fill = "rgba(248,113,113,0.35)" if is_dark else "#f4a6a8"
    ema_fast_color = "#60a5fa" if is_dark else "#6fa8dc"
    ema_slow_color = "#fbbf24" if is_dark else "#f6b26b"
    rsi_color = "#a855f7" if is_dark else "#8b5cf6"
    equity_color = "#22d3ee" if is_dark else "#00695c"
    template = "plotly_dark" if is_dark else "plotly_white"
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"secondary_y": False}], [{}], [{}]],  # type: ignore
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"{symbol} price with EMA200 and 24-bar breakout",
            "ATR(14)",
            "Equity",
        ),
    )
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles",
            increasing_line_color=candle_up,
            decreasing_line_color=candle_down,
            increasing_fillcolor=candle_up_fill,
            decreasing_fillcolor=candle_down_fill,
            hovertemplate="Time: %{x}<br>O: %{open}<br>H: %{high}<br>L: %{low}<br>C: %{close}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema200"],
            name="EMA200",
            line=dict(color=ema_fast_color, width=1.4),
            hovertemplate="EMA200: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["hh24_prev"],
            name="HH24 prev",
            line=dict(color=ema_slow_color, width=1, dash="dash"),
            hovertemplate="HH24: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ll24_prev"],
            name="LL24 prev",
            line=dict(color="#e11d48", width=1, dash="dash"),
            hovertemplate="LL24: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["atr14"],
            name="ATR(14)",
            line=dict(color=rsi_color, width=1.2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=equity,
            name="Equity",
            line=dict(color=equity_color, width=1.2),
        ),
        row=3,
        col=1,
    )
    # mark entries/exits using event_log when available to keep alignment with equity/trades
    log_df = pd.DataFrame(event_log) if event_log else None
    long_df = log_df[(log_df["event"] == "entry") & (log_df["dir"] == "LONG")] if log_df is not None else df[df["long_entry"]]
    short_df = log_df[(log_df["event"] == "entry") & (log_df["dir"] == "SHORT")] if log_df is not None else df[df["short_entry"]]
    long_exit_df = log_df[(log_df["event"] == "exit") & (log_df["dir"] == "LONG")] if log_df is not None else df[df["long_exit"]]
    short_exit_df = log_df[(log_df["event"] == "exit") & (log_df["dir"] == "SHORT")] if log_df is not None else df[df["short_exit"]]
    entry_stops = log_df[log_df["event"] == "entry"] if log_df is not None else None
    if long_df is not None and not long_df.empty:
        fig.add_trace(
            go.Scatter(
                x=long_df["time"],
                y=long_df["price"] if "price" in long_df.columns else long_df["close"],
                mode="markers",
                name="Long Entry",
                marker=dict(
                    color=candle_up,
                    size=9,
                    symbol="triangle-up",
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="Long Entry<br>%{x}<br>Price: %{y:.6f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    if entry_stops is not None and not entry_stops.empty:
        long_stops = entry_stops[entry_stops["dir"] == "LONG"]
        short_stops = entry_stops[entry_stops["dir"] == "SHORT"]
        if not long_stops.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_stops["time"],
                    y=long_stops["stop"],
                    mode="markers",
                    name="Long Stop",
                    marker=dict(
                        color="#ef4444",
                        size=14,
                        symbol="line-ew",
                        line=dict(color="#ef4444", width=3),
                    ),
                    hovertemplate="Long Stop<br>%{x}<br>Stop: %{y:.6f}<extra></extra>",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if not short_stops.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_stops["time"],
                    y=short_stops["stop"],
                    mode="markers",
                    name="Short Stop",
                    marker=dict(
                        color="#ef4444",
                        size=14,
                        symbol="line-ew",
                        line=dict(color="#ef4444", width=3),
                    ),
                    hovertemplate="Short Stop<br>%{x}<br>Stop: %{y:.6f}<extra></extra>",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    if short_df is not None and not short_df.empty:
        fig.add_trace(
            go.Scatter(
                x=short_df["time"],
                y=short_df["price"] if "price" in short_df.columns else short_df["close"],
                mode="markers",
                name="Short Entry",
                marker=dict(
                    color=candle_down,
                    size=9,
                    symbol="triangle-down",
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="Short Entry<br>%{x}<br>Price: %{y:.6f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    if long_exit_df is not None and not long_exit_df.empty:
        fig.add_trace(
            go.Scatter(
                x=long_exit_df["time"],
                y=long_exit_df["price"] if "price" in long_exit_df.columns else long_exit_df["close"],
                mode="markers",
                name="Long Exit",
                marker=dict(
                    color=candle_up,
                    size=9,
                    symbol="x",
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="Long Exit<br>%{x}<br>Price: %{y:.6f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    if short_exit_df is not None and not short_exit_df.empty:
        fig.add_trace(
            go.Scatter(
                x=short_exit_df["time"],
                y=short_exit_df["price"] if "price" in short_exit_df.columns else short_exit_df["close"],
                mode="markers",
                name="Short Exit",
                marker=dict(
                    color=candle_down,
                    size=9,
                    symbol="x",
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="Short Exit<br>%{x}<br>Price: %{y:.6f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        hovermode="x",
        hoverdistance=10,
        template=template,
        showlegend=True,
        height=900,
        dragmode="pan",
        plot_bgcolor=panel_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, 'Helvetica Neue', Arial"),
        hoverlabel=dict(bgcolor=panel_color, font=dict(color=text_color)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=60, b=60),
        uirevision="keep",
        transition=dict(duration=0),
    )
    for row in range(1, 4):
        fig.update_xaxes(
            row=row,
            col=1,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
            spikecolor=spike_color,
            gridcolor=grid_color,
            zeroline=False,
        )
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="solid",
        spikecolor=spike_color,
        gridcolor=grid_color,
        hoverformat=None,
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=3, col=1)
    html_str = fig.to_html(
        include_plotlyjs="inline",  # embed script to avoid CDN/load blockers
        full_html=True,
        div_id="backtest-chart",
        config={
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "modeBarButtonsToRemove": ["autoScale2d"],
        },
    )
    theme_payload = {
        "dark": {
            "bodyBg": "#0b1224",
            "text": "#e5e7eb",
            "panelBg": "#0f172a",
            "plotly": {
                "template": "plotly_dark",
                "paper_bgcolor": "#0b1224",
                "plot_bgcolor": "#0f172a",
                "font.color": "#e5e7eb",
                "hoverlabel.bgcolor": "#0f172a",
                "hoverlabel.font.color": "#e5e7eb",
                "legend.font.color": "#e5e7eb",
                "gridcolor": grid_color,
                "spikecolor": spike_color,
            },
        },
        "light": {
            "bodyBg": "#ffffff",
            "text": "#1a1a1a",
            "panelBg": "#ffffff",
            "plotly": {
                "template": "plotly_white",
                "paper_bgcolor": "#ffffff",
                "plot_bgcolor": "#ffffff",
                "font.color": "#1a1a1a",
                "hoverlabel.bgcolor": "#ffffff",
                "hoverlabel.font.color": "#1a1a1a",
                "legend.font.color": "#1a1a1a",
                "gridcolor": grid_color,
                "spikecolor": spike_color,
            },
        },
    }
    style_block = (
        "<style>"
        "body{margin:0;background:%(bg)s;color:%(text)s;font-family:Inter, 'Helvetica Neue', Arial, sans-serif;}"
        "#controls{position:sticky;top:0;z-index:10;padding:12px 16px;background:rgba(0,0,0,0.08);backdrop-filter:blur(6px);display:flex;justify-content:flex-end;}"
        "#theme-toggle{border:none;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer;background:%(text)s;color:%(bg)s;box-shadow:0 2px 6px rgba(0,0,0,0.2);}"
        "#theme-toggle:hover{opacity:0.9;}"
        "#backtest-chart{background:%(panel)s;}"
        "</style>"
        % {"bg": bg_color, "text": text_color, "panel": panel_color}
    )
    toggle_block = (
        '<div id="controls"><button id="theme-toggle">'
        + ("Light mode" if is_dark else "Dark mode")
        + "</button></div>"
    )
    script_block = f"""
<script>
const THEMES = {json.dumps(theme_payload)};
let currentTheme = "{'dark' if is_dark else 'light'}";
function applyTheme() {{
  const t = THEMES[currentTheme];
  document.body.style.background = t.bodyBg;
  document.body.style.color = t.text;
  const chart = document.getElementById("backtest-chart");
  if (chart && window.Plotly) {{
    const axes = Object.keys(chart.layout).filter(k => k.startsWith("xaxis") || k.startsWith("yaxis"));
    const updates = {{
      template: t.plotly.template,
      paper_bgcolor: t.plotly.paper_bgcolor,
      plot_bgcolor: t.plotly.plot_bgcolor,
      "font.color": t.plotly["font.color"],
      "hoverlabel.bgcolor": t.plotly["hoverlabel.bgcolor"],
      "hoverlabel.font.color": t.plotly["hoverlabel.font.color"],
      "legend.font.color": t.plotly["legend.font.color"],
    }};
    axes.forEach(ax => {{
      updates[`${{ax}}.gridcolor`] = t.plotly.gridcolor;
      updates[`${{ax}}.spikecolor`] = t.plotly.spikecolor;
    }});
    Plotly.relayout(chart, updates);
  }}
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = currentTheme === "dark" ? "Light mode" : "Dark mode";
}}
document.addEventListener("DOMContentLoaded", () => {{
  const btn = document.getElementById("theme-toggle");
  if (btn) {{
    btn.addEventListener("click", () => {{
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme();
    }});
  }}
  applyTheme();
}});
</script>"""
    if "</head>" in html_str:
        html_str = html_str.replace("</head>", f"{style_block}</head>")
    if "<body>" in html_str:
        html_str = html_str.replace("<body>", f"<body>{toggle_block}")
    if "</body>" in html_str:
        html_str = html_str.replace("</body>", f"{script_block}</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    if open_report:
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f"file://{abs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA200 1H breakout backtest and plot")
    parser.add_argument("--symbol", default=os.getenv("BOT_SYMBOL", "DOGEUSDT"))
    parser.add_argument("--interval", default=os.getenv("BOT_INTERVAL", "1h"))
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--initial", type=float, default=25.0, help="starting equity")
    parser.add_argument("--risk-pct", type=float, default=0.03, help="fraction of equity to risk per trade")
    parser.add_argument("--fee-rate", type=float, default=0.0004, help="taker fee rate per side (e.g. 0.0004 for 0.04 pct)")
    parser.add_argument("--leverage", type=int, default=2, help="leverage used for sizing context")
    parser.add_argument("--html", default="reports/backtest_report.html", help="output HTML report path (set empty to skip)")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark", help="color theme for the HTML report")
    parser.add_argument("--no-open", action="store_true", help="skip opening HTML automatically")
    parser.add_argument("--log", default="reports/trade_log.csv", help="CSV path to export trade log")
    parser.add_argument("--split", type=float, default=None, help="optional train/test split ratio (e.g. 0.7)")
    args = parser.parse_args()

    df, eq, trades, event_log, settings, summary = backtest(
        args.symbol.upper(),
        args.interval,
        args.lookback,
        args.initial,
        risk_pct=args.risk_pct,
        fee_rate=args.fee_rate,
        leverage=args.leverage,
    )
    print("Summary:")
    print("\n".join(format_summary_lines(summary, trades)))
    # export logs instead of printing all trades
    if args.log:
        log_dir = os.path.dirname(args.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_df = pd.DataFrame(event_log)
        log_df.to_csv(args.log, index=False)
        trades_df = pd.DataFrame(trades)
        closed_path = args.log.replace(".csv", "_closed.csv")
        trades_df.to_csv(closed_path, index=False)
        equity_path = args.log.replace(".csv", "_equity.csv")
        eq.to_csv(equity_path, header=["equity"], index_label="time")
        print(f"Trade event log written to {args.log}")
        print(f"Closed trades written to {closed_path}")
        print(f"Equity curve written to {equity_path}")
    if args.html:
        plot_results_html(
            df,
            eq,
            args.symbol.upper(),
            args.html,
            open_report=not args.no_open,
            theme=args.theme,
            event_log=event_log,
        )
        print(f"HTML report written to {args.html}")
        if args.no_open:
            print("Open the report in your browser to view the interactive crosshairs.")
    if args.split is not None:
        train_res, test_res = run_split_backtest(
            df,
            fee_rate=args.fee_rate,
            initial_equity=args.initial,
            risk_pct=args.risk_pct,
            leverage=args.leverage,
            split_ratio=args.split,
        )
        (df_train, eq_train, trades_train, log_train, summary_train) = train_res
        (df_test, eq_test, trades_test, log_test, summary_test) = test_res
        if args.log:
            base, ext = os.path.splitext(args.log)
            log_dir = os.path.dirname(args.log)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            pd.DataFrame(log_train).to_csv(f"{base}_train{ext or '.csv'}", index=False)
            pd.DataFrame(trades_train).to_csv(f"{base}_train_closed{ext or '.csv'}", index=False)
            eq_train.to_csv(f"{base}_train_equity{ext or '.csv'}", header=["equity"], index_label="time")
            pd.DataFrame(log_test).to_csv(f"{base}_test{ext or '.csv'}", index=False)
            pd.DataFrame(trades_test).to_csv(f"{base}_test_closed{ext or '.csv'}", index=False)
            eq_test.to_csv(f"{base}_test_equity{ext or '.csv'}", header=["equity"], index_label="time")
            print(f"Split train/test logs written to {os.path.dirname(args.log) or '.'}")


if __name__ == "__main__":
    main()
