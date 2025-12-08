import argparse
import os
import webbrowser
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from bot import FuturesBot, Settings


load_dotenv(".env")


def equity_curve(prices: pd.Series, signals: pd.Series, initial: float) -> pd.Series:
    returns = prices.pct_change().fillna(0)
    strat_returns = returns * signals.shift(1).fillna(0)
    equity = (1 + strat_returns).cumprod() * initial
    return equity


def simulate_trades(df: pd.DataFrame, bot: FuturesBot, initial: float) -> Tuple[pd.Series, List[dict], List[dict]]:
    equity = initial
    equity_curve = []
    trades: List[dict] = []
    event_log: List[dict] = []
    position: dict | None = None
    settings = bot.settings

    for _, row in df.iterrows():
        time = row["time"]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])
        signal = int(row["signal"])

        # manage open position first
        if position:
            direction = position["direction"]
            stop = position["stop"]
            tp = position["tp"]
            entry = position["entry"]
            stop_dist = position["stop_dist"]
            peak = position["peak"]
            trough = position["trough"]
            qty = position["qty"]
            partial_taken = position.get("partial_taken", False)

            # update peaks/troughs
            if direction > 0:
                peak = max(peak, high)
            else:
                trough = min(trough, low)

            # activate break-even stop
            if settings.break_even_rr > 0:
                be_level = entry + direction * stop_dist * settings.break_even_rr
                if (direction > 0 and high >= be_level) or (direction < 0 and low <= be_level):
                    stop = max(stop, entry) if direction > 0 else min(stop, entry)

            # activate/update trailing
            trail_stop = None
            if settings.trailing_start_rr > 0 and settings.trailing_callback_pct > 0:
                activate = entry + direction * stop_dist * settings.trailing_start_rr
                callback = settings.trailing_callback_pct / 100.0
                if direction > 0 and high >= activate:
                    trail_stop = peak * (1 - callback)
                elif direction < 0 and low <= activate:
                    trail_stop = trough * (1 + callback)

            # check exits (conservative: stop/trail before TP if both touched same bar)
            exit_reason = None
            exit_price = None

            # price ladder handling: stop/trail first, then partial TP, then final TP
            if direction > 0:
                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif trail_stop is not None and low <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "trail"
                elif (
                    settings.partial_tp_enabled
                    and not partial_taken
                    and settings.partial_tp_ratio > 0
                    and settings.partial_tp_ratio < 1
                ):
                    ptp_price = entry + direction * stop_dist * settings.partial_tp_rr
                    if high >= ptp_price:
                        realized = direction * (ptp_price - entry) * qty * settings.partial_tp_ratio
                        equity += realized
                        qty *= 1 - settings.partial_tp_ratio
                        partial_taken = True
                        position["qty"] = qty
                        position["partial_taken"] = True
                if exit_price is None and high >= tp:
                    exit_price = tp
                    exit_reason = "tp"
            else:
                if high >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif trail_stop is not None and high >= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "trail"
                elif (
                    settings.partial_tp_enabled
                    and not partial_taken
                    and settings.partial_tp_ratio > 0
                    and settings.partial_tp_ratio < 1
                ):
                    ptp_price = entry + direction * stop_dist * settings.partial_tp_rr
                    if low <= ptp_price:
                        realized = direction * (ptp_price - entry) * qty * settings.partial_tp_ratio
                        equity += realized
                        qty *= 1 - settings.partial_tp_ratio
                        partial_taken = True
                        position["qty"] = qty
                        position["partial_taken"] = True
                if exit_price is None and low <= tp:
                    exit_price = tp
                    exit_reason = "tp"

            if exit_price is not None:
                pnl = direction * (exit_price - entry) * qty
                equity += pnl
                exit_detail = {
                    "stop": "Stop-loss hit",
                    "trail": "Trailing stop activated",
                    "tp": "Take-profit hit",
                }.get(exit_reason or "", exit_reason or "")
                exit_text = f"{exit_detail} @ {exit_price:.6f}".strip()
                trades.append(
                    {
                        "time": time,
                        "dir": "LONG" if direction > 0 else "SHORT",
                        "entry": entry,
                        "exit": exit_price,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "exit_condition": exit_text,
                        "equity_after": equity,
                    }
                )
                event_log.append(
                    {
                        "event": "exit",
                        "time": time,
                        "dir": "LONG" if direction > 0 else "SHORT",
                        "price": exit_price,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "condition": exit_text,
                        "equity": equity,
                    }
                )
                position = None
            else:
                # exit when signal flips/flat even if no stop/TP hit
                if signal == 0 or signal != direction:
                    exit_price = price
                    pnl = direction * (exit_price - entry) * qty
                    equity += pnl
                    exit_text = f"Signal exit @ {exit_price:.6f}"
                    trades.append(
                        {
                            "time": time,
                            "dir": "LONG" if direction > 0 else "SHORT",
                            "entry": entry,
                            "exit": exit_price,
                            "pnl": pnl,
                            "reason": "signal_exit",
                            "exit_condition": exit_text,
                            "equity_after": equity,
                        }
                    )
                    event_log.append(
                        {
                            "event": "exit",
                            "time": time,
                            "dir": "LONG" if direction > 0 else "SHORT",
                            "price": exit_price,
                            "qty": qty,
                            "pnl": pnl,
                            "reason": "signal_exit",
                            "condition": exit_text,
                            "equity": equity,
                        }
                    )
                    position = None
                else:
                    position["peak"] = peak
                    position["trough"] = trough

        # consider new entry only if flat
        if not position and signal != 0:
            _, entry_reason = bot._signal_detail(row)
            try:
                qty, stop_price, tp_price = bot._size_position(signal, price, atr, equity_override=equity)
            except Exception:
                equity_curve.append(equity)
                continue
            entry_price = price * (1 + settings.max_slippage_pct * signal)
            stop_distance = abs(entry_price - stop_price)
            position = {
                "direction": signal,
                "qty": qty,
                "entry": entry_price,
                "stop": stop_price,
                "tp": tp_price,
                "stop_dist": stop_distance,
                "peak": high if signal > 0 else entry_price,
                "trough": low if signal < 0 else entry_price,
                "entry_reason": entry_reason,
            }
            event_log.append(
                {
                    "event": "entry",
                    "time": time,
                    "dir": "LONG" if signal > 0 else "SHORT",
                    "price": entry_price,
                    "qty": qty,
                    "stop": stop_price,
                    "tp": tp_price,
                    "condition": entry_reason,
                    "equity": equity,
                }
            )

        equity_curve.append(equity)

    # if backtest ends with open position, close at last price for logging
    if position:
        last_row = df.iloc[-1]
        exit_price = float(last_row["close"])
        direction = position["direction"]
        qty = position["qty"]
        entry = position["entry"]
        pnl = direction * (exit_price - entry) * qty
        equity += pnl
        if equity_curve:
            equity_curve[-1] = equity
        exit_text = f"Forced close at end of data @ {exit_price:.6f}"
        trades.append(
            {
                "time": last_row["time"],
                "dir": "LONG" if direction > 0 else "SHORT",
                "entry": entry,
                "exit": exit_price,
                "pnl": pnl,
                "reason": "end_of_data",
                "exit_condition": exit_text,
                "equity_after": equity,
            }
        )
        event_log.append(
            {
                "event": "exit",
                "time": last_row["time"],
                "dir": "LONG" if direction > 0 else "SHORT",
                "price": exit_price,
                "qty": qty,
                "pnl": pnl,
                "reason": "end_of_data",
                "condition": exit_text,
                "equity": equity,
            }
        )

    return pd.Series(equity_curve, index=df.index), trades, event_log


def backtest(symbol: str, interval: str, lookback: int, initial: float) -> Tuple[pd.DataFrame, pd.Series, List[dict], List[dict]]:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    settings = Settings(symbol=symbol, interval=interval, lookback=lookback, live=False)
    bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
    if settings.htf_interval:
        bot.htf_trend = bot._fetch_htf_trend()
    df = bot.fetch_klines()
    df = bot._compute_indicators(df)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(
        "Asia/Bangkok"
    )
    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["vol_color"] = df.apply(
        lambda r: "rgba(0,200,140,0.35)" if r["close"] >= r["open"] else "rgba(255,90,90,0.35)",
        axis=1,
    )
    df["signal"] = df.apply(bot._signal_direction, axis=1)
    df["signal_prev"] = df["signal"].shift(1).fillna(0)
    df["long_entry"] = (df["signal"] == 1) & (df["signal_prev"] != 1)
    df["short_entry"] = (df["signal"] == -1) & (df["signal_prev"] != -1)
    df["long_exit"] = (df["signal_prev"] == 1) & (df["signal"] != 1)
    df["short_exit"] = (df["signal_prev"] == -1) & (df["signal"] != -1)
    eq, trades, event_log = simulate_trades(df, bot, initial)
    return df, eq, trades, event_log


def plot_results_html(df: pd.DataFrame, equity: pd.Series, symbol: str, output_path: str, open_report: bool = True) -> None:
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"secondary_y": False}], [{}], [{}]],  # type: ignore
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"{symbol} price with EMA crossover",
            "RSI",
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
            increasing_line_color="#2d8a5f",
            decreasing_line_color="#e15759",
            increasing_fillcolor="#6fcf97",
            decreasing_fillcolor="#f4a6a8",
            hovertemplate="Time: %{x}<br>O: %{open}<br>H: %{high}<br>L: %{low}<br>C: %{close}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema_fast"],
            name="EMA Fast",
            line=dict(color="#6fa8dc", width=1.2),
            hovertemplate="EMA Fast: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema_slow"],
            name="EMA Slow",
            line=dict(color="#f6b26b", width=1.2),
            hovertemplate="EMA Slow: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["rsi"],
            name="RSI",
            line=dict(color="#8b5cf6", width=1.2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line=dict(color="#f4b6c2", dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#cce2cb", dash="dash"), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=equity,
            name="Equity",
            line=dict(color="#00695c", width=1.2),
        ),
        row=3,
        col=1,
    )
    # mark long/short entries
    long_df = df[df["long_entry"]]
    short_df = df[df["short_entry"]]
    long_exit_df = df[df["long_exit"]]
    short_exit_df = df[df["short_exit"]]
    if not long_df.empty:
        fig.add_trace(
            go.Scatter(
                x=long_df["time"],
                y=long_df["close"],
                mode="markers",
                name="Long Entry",
                marker=dict(
                    color="#2d8a5f",
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
    if not short_df.empty:
        fig.add_trace(
            go.Scatter(
                x=short_df["time"],
                y=short_df["close"],
                mode="markers",
                name="Short Entry",
                marker=dict(
                    color="#e15759",
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
    if not long_exit_df.empty:
        fig.add_trace(
            go.Scatter(
                x=long_exit_df["time"],
                y=long_exit_df["close"],
                mode="markers",
                name="Long Exit",
                marker=dict(
                    color="#2d8a5f",
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
    if not short_exit_df.empty:
        fig.add_trace(
            go.Scatter(
                x=short_exit_df["time"],
                y=short_exit_df["close"],
                mode="markers",
                name="Short Exit",
                marker=dict(
                    color="#e15759",
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
        template="plotly_white",
        showlegend=True,
        height=900,
        dragmode="pan",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1a1a1a", family="Inter, 'Helvetica Neue', Arial"),
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
            spikecolor="#aaa",
            gridcolor="#e5e5e5",
            zeroline=False,
        )
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="solid",
        spikecolor="#aaa",
        gridcolor="#e5e5e5",
        hoverformat=None,
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=3, col=1)
    html_str = fig.to_html(
        include_plotlyjs="cdn",
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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    if open_report:
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f"file://{abs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple EMA/RSI backtest and plot")
    parser.add_argument("--symbol", default=os.getenv("BOT_SYMBOL", "DOGEUSDT"))
    parser.add_argument("--interval", default=os.getenv("BOT_INTERVAL", "5m"))
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--initial", type=float, default=25.0, help="starting equity")
    parser.add_argument("--html", default="reports/backtest_report.html", help="output HTML report path (set empty to skip)")
    parser.add_argument("--no-open", action="store_true", help="skip opening HTML automatically")
    parser.add_argument("--log", default="reports/trade_log.csv", help="CSV path to export trade log")
    args = parser.parse_args()

    df, eq, trades, event_log = backtest(args.symbol.upper(), args.interval, args.lookback, args.initial)
    final = eq.iloc[-1]
    print(f"Final equity: ${final:.2f} ({(final-args.initial)/args.initial*100:.2f}%)")
    print(f"Trades: {len(trades)}")
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    print(f"Wins: {wins}, Losses: {losses}, Win rate: {win_rate:.2f}%")
    # export logs instead of printing all trades
    if args.log:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        log_df = pd.DataFrame(event_log)
        log_df.to_csv(args.log, index=False)
        trades_df = pd.DataFrame(trades)
        closed_path = args.log.replace(".csv", "_closed.csv")
        trades_df.to_csv(closed_path, index=False)
        equity_path = args.log.replace(".csv", "_equity.csv")
        eq.to_csv(equity_path, header=["equity"])
        print(f"Trade event log written to {args.log}")
        print(f"Closed trades written to {closed_path}")
        print(f"Equity curve written to {equity_path}")
    if args.html:
        plot_results_html(df, eq, args.symbol.upper(), args.html, open_report=not args.no_open)
        print(f"HTML report written to {args.html}")
        if args.no_open:
            print("Open the report in your browser to view the interactive crosshairs.")


if __name__ == "__main__":
    main()
