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

- Pulls recent klines and computes EMA cross with RSI, slow-EMA slope, and volatility filters to avoid chop.
- Uses $25 starting equity with 1% risk per trade; adapts to live wallet balance if higher.
- Sizes quantity based on ATR stop distance and exchange lot/tick filters.
- Sets leverage (default 5x), enters with a market order, and places a stop-market exit.
- Prints a dry-run plan before sending anything live.

## Safety

- Credentials are loaded from `.env`; never commit them.
- Default is dry-run; pass `--live` to trade.
- Testnet support via `BINANCE_TESTNET=1`. Stay on testnet until you have validated behavior.
- This is a template, not financial advice. Extend with your own risk checks, logging, and monitoring.
