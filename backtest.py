import argparse
import os
import webbrowser
from typing import List

import pandas as pd
from dotenv import load_dotenv
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from backtest_engine import BacktestResult, format_summary_lines, run_backtest, run_split_backtest
from bot import FuturesBot, Settings
from strategies import all_strategies, build_strategy, get_strategy_keys


load_dotenv(".env")


def plot_results_html(
    result: BacktestResult,
    symbol: str,
    strategy_name: str,
    output_path: str,
    open_report: bool = True,
    theme: str = "dark",
) -> None:
    df = result.df
    equity = result.equity
    event_log = result.events
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
    overlay_colors = {
        "ema200": "#60a5fa",
        "ema_slow": "#fbbf24",
        "ema_fast": "#6b7280",
        "bb_upper": "#a855f7",
        "bb_lower": "#a855f7",
        "bb_mid": "#a855f7",
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"secondary_y": False}], [{}]],  # type: ignore
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol} price ({strategy_name})", "Equity"),
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
    for col, color in overlay_colors.items():
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[col],
                    name=col,
                    line=dict(color=color, width=1.2, dash="dash" if "bb" in col else "solid"),
                    hovertemplate=f"{col}: "+"%{y:.6f}<extra></extra>",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Equity",
            line=dict(color="#22d3ee" if is_dark else "#00695c", width=1.2),
        ),
        row=2,
        col=1,
    )
    log_df = pd.DataFrame(event_log) if event_log else None
    long_df = log_df[(log_df["event"] == "entry") & (log_df["dir"] == "LONG")] if log_df is not None else None
    short_df = log_df[(log_df["event"] == "entry") & (log_df["dir"] == "SHORT")] if log_df is not None else None
    long_exit_df = log_df[(log_df["event"] == "exit") & (log_df["dir"] == "LONG")] if log_df is not None else None
    short_exit_df = log_df[(log_df["event"] == "exit") & (log_df["dir"] == "SHORT")] if log_df is not None else None
    if long_df is not None and not long_df.empty:
        fig.add_trace(
            go.Scatter(
                x=long_df["time"],
                y=long_df["price"],
                mode="markers",
                name="Long Entry",
                marker=dict(color=candle_up, size=9, symbol="triangle-up", line=dict(color="#ffffff", width=1)),
                hovertemplate="Long Entry<br>%{x}<br>Price: %{y:.6f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    if short_df is not None and not short_df.empty:
        fig.add_trace(
            go.Scatter(
                x=short_df["time"],
                y=short_df["price"],
                mode="markers",
                name="Short Entry",
                marker=dict(color=candle_down, size=9, symbol="triangle-down", line=dict(color="#ffffff", width=1)),
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
                y=long_exit_df["price"],
                mode="markers",
                name="Long Exit",
                marker=dict(color=candle_up, size=9, symbol="x", line=dict(color="#ffffff", width=1)),
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
                y=short_exit_df["price"],
                mode="markers",
                name="Short Exit",
                marker=dict(color=candle_down, size=9, symbol="x", line=dict(color="#ffffff", width=1)),
                hovertemplate="Short Exit<br>%{x}<br>Price: %{y:.6f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        hovermode="x",
        hoverdistance=10,
        template="plotly_dark" if is_dark else "plotly_white",
        showlegend=True,
        height=750,
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
    for row in range(1, 3):
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
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.write_html(
        output_path,
        include_plotlyjs="inline",
        full_html=True,
        config={
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "modeBarButtonsToRemove": ["autoScale2d"],
        },
    )
    if open_report:
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f"file://{abs_path}")


def save_logs(base_path: str, result: BacktestResult, suffix: str | None = None) -> None:
    suffix = f"_{suffix}" if suffix else ""
    base, ext = os.path.splitext(base_path)
    out_base = f"{base}{suffix}{ext or '.csv'}"
    log_dir = os.path.dirname(out_base)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(result.events).to_csv(out_base, index=False)
    pd.DataFrame(result.trades).to_csv(f"{base}{suffix}_closed{ext or '.csv'}", index=False)
    result.equity.to_csv(f"{base}{suffix}_equity{ext or '.csv'}", header=["equity"], index_label="time")
    print(f"Logs written to {os.path.dirname(out_base) or '.'} for {suffix or 'strategy'}")


def fetch_ohlcv(symbol: str, interval: str, lookback: int, leverage: int) -> pd.DataFrame:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    settings = Settings(symbol=symbol, interval=interval, lookback=lookback, live=False, leverage=leverage)
    bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
    return bot.fetch_klines()


def run_single_strategy(args: argparse.Namespace) -> None:
    strat = build_strategy(args.strategy)
    raw_df = fetch_ohlcv(args.symbol.upper(), args.interval, args.lookback, args.leverage)
    result = run_backtest(raw_df, strat, args.initial, args.risk_pct, args.fee_rate)
    print(f"\n=== {strat.name} ({strat.key}) ===")
    print("\n".join(format_summary_lines(result.summary, result.trades)))
    if args.log:
        save_logs(args.log, result)
    if args.html:
        plot_results_html(
            result,
            args.symbol.upper(),
            strat.name,
            args.html,
            open_report=not args.no_open,
            theme=args.theme,
        )
        print(f"HTML report written to {args.html}")
        if args.no_open:
            print("Open the report in your browser to view the interactive crosshairs.")
    if args.split is not None:
        train_res, test_res = run_split_backtest(
            raw_df,
            strat,
            args.split,
            args.initial,
            args.risk_pct,
            args.fee_rate,
        )
        print("\n=== TRAIN (old data) ===")
        print("\n".join(format_summary_lines(train_res.summary, train_res.trades)))
        print("\n=== TEST (recent data) ===")
        print("\n".join(format_summary_lines(test_res.summary, test_res.trades)))
        if args.log:
            base, ext = os.path.splitext(args.log)
            save_logs(f"{base}_train{ext or '.csv'}", train_res)
            save_logs(f"{base}_test{ext or '.csv'}", test_res)


def run_all(args: argparse.Namespace) -> None:
    strategies = all_strategies()
    raw_df = fetch_ohlcv(args.symbol.upper(), args.interval, args.lookback, args.leverage)
    summaries: List[tuple[str, BacktestResult]] = []
    for strat in strategies:
        print(f"\n=== Running {strat.name} ({strat.key}) ===")
        res = run_backtest(raw_df, strat, args.initial, args.risk_pct, args.fee_rate)
        summaries.append((strat.key, res))
        print("\n".join(format_summary_lines(res.summary, res.trades)))
        if args.log:
            base, ext = os.path.splitext(args.log)
            save_logs(f"{base}_{strat.key}{ext or '.csv'}", res, suffix=None)
        if args.split is not None:
            train_res, test_res = run_split_backtest(
                raw_df,
                strat,
                args.split,
                args.initial,
                args.risk_pct,
                args.fee_rate,
            )
            print("  TRAIN:", " | ".join(format_summary_lines(train_res.summary, train_res.trades)))
            print("  TEST :", " | ".join(format_summary_lines(test_res.summary, test_res.trades)))
    print("\n=== Aggregate Summary ===")
    for key, res in summaries:
        print(f"{key}: PF {res.summary['profit_factor']:.2f}, trades {res.summary['num_trades']}, "
              f"PnL {res.summary['total_net_pnl']:.2f}, DD {res.summary['max_drawdown']:.2f}")
    if args.html:
        print("HTML plotting is skipped in --strategy all mode; rerun single strategy to plot.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-strategy backtester")
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
    parser.add_argument("--strategy", default="trend_breakout_v1", choices=get_strategy_keys() + ["all"], help="strategy key or 'all'")
    args = parser.parse_args()

    if args.strategy == "all":
        run_all(args)
    else:
        run_single_strategy(args)


if __name__ == "__main__":
    main()
