import argparse
import os
import webbrowser
from typing import Tuple

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


def backtest(symbol: str, interval: str, lookback: int, initial: float) -> Tuple[pd.DataFrame, pd.Series]:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    settings = Settings(symbol=symbol, interval=interval, lookback=lookback, live=False)
    bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
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
    eq = equity_curve(df["close"], df["signal"], initial)
    return df, eq


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
    args = parser.parse_args()

    df, eq = backtest(args.symbol.upper(), args.interval, args.lookback, args.initial)
    print(f"Final equity: ${eq.iloc[-1]:.2f}")
    if args.html:
        plot_results_html(df, eq, args.symbol.upper(), args.html, open_report=not args.no_open)
        print(f"HTML report written to {args.html}")
        if args.no_open:
            print("Open the report in your browser to view the interactive crosshairs.")


if __name__ == "__main__":
    main()
