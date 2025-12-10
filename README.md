# Binance Futures Bot (starter)

Python starter for a risk-conscious Binance USDⓈ-M futures bot. It pulls market data, sizes positions off a $25 starting balance, and can run in either dry-run (default) or live mode.

## Quick start

- Python 3.10+ recommended.
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```
- Copy `.env` (do **not** commit real keys) and fill in:
  ```
  BINANCE_API_KEY=your_key
  BINANCE_API_SECRET=your_secret
  BINANCE_TESTNET=1            # keep enabled while testing
  BOT_SYMBOL=DOGEUSDT
  BOT_INTERVAL=5m
  ```
- Run a single cycle in dry-run (no orders):
  ```bash
  python bot.py
  ```
- Live mode (places orders): add `--live`. Keep `BINANCE_TESTNET=1` until you are confident.
- Backtest to HTML:
  ```bash
  python backtest.py --symbol DOGEUSDT --interval 5m --lookback 500 --initial 25
  # HTML auto-opens and is saved to reports/backtest_report.html (candlestick + EMA + volume, RSI, equity with synced crosshair)
  # Add --no-open to skip auto-opening the browser
  ```

## What it does

- On each 1H close, trades a simple EMA200 + 24-bar breakout: longs when price > EMA200, EMA200 rising (vs 10 bars ago), and price breaks the prior 24-bar high; shorts are the mirror.
- Uses ATR(14) for risk: stop at 2× ATR from entry, TP at 3R; 1% of equity risked per trade sized via entry/stop distance and exchange filters.
- $25 starting equity by default; adapts to wallet balance if higher. Leverage defaults to 3x isolated.
- Places market entry, stop-market, and take-profit-market orders; exits immediately if price closes back across EMA200 against the position.
- Prints a dry-run plan before sending anything live.

## Safety

- Credentials are loaded from `.env`; never commit them.
- Default is dry-run; pass `--live` to trade.
- Testnet support via `BINANCE_TESTNET=1`. Stay on testnet until you have validated behavior.
- This is a template, not financial advice. Extend with your own risk checks, logging, and monitoring.
