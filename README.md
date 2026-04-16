# Binance Futures Mean-Reversion Bot

Python bot for Binance USD-M futures built around a range mean-reversion strategy. The current `bot.py` defaults to dry-run, can run once or in a loop, and only places orders when you pass `--live`.

It still talks to Binance in dry-run. Exchange info and market data are always fetched, and account data is used when sizing a signal, so `BINANCE_API_KEY` and `BINANCE_API_SECRET` must be present before running it.

## Quick start

- Python 3.10+ recommended.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Create a `.env` file and keep real keys out of git:
  ```dotenv
  BINANCE_API_KEY=your_key
  BINANCE_API_SECRET=your_secret
  BINANCE_TESTNET=1

  BOT_SYMBOL=DOGEUSDT
  BOT_INTERVAL=1h
  BOT_LOOKBACK=300
  BOT_RANGE_LOOKBACK=20
  BOT_ZSCORE_THRESHOLD=1.0
  ```
- Dry-run a single cycle:
  ```bash
  python bot.py --testnet
  ```
- Dry-run continuously:
  ```bash
  python bot.py --testnet --loop --sleep 300 --log-flat
  ```
- Live on testnet:
  ```bash
  python bot.py --testnet --live
  ```

Keep `BINANCE_TESTNET=1` or pass `--testnet` until the behavior, sizing, and logs look correct.

## What `bot.py` does

- Fetches up to `--lookback` candles, sorts them chronologically, and drops the last candle if it is still open.
- Computes rolling mean and standard deviation, prior range high and low, z-score, ATR, ADX, efficiency ratio, rolling mean-cross count, volume moving average, and an optional trend EMA.
- Long signal: the previous candle broke below the prior range low, the latest close re-entered back above the lower band by `--reentry-buffer-bps`, and z-score is at or below the negative threshold set by `--zscore-threshold`.
- Short signal: the mirror setup above the prior range high, with z-score at or above `--zscore-threshold`.
- If `--trend-ema-length` is enabled, longs are only allowed at or below the EMA and shorts only at or above it.
- Optional regime filters reject entries when price action looks too directional, based on EMA slope, ADX, ATR extension from the EMA, efficiency ratio, and minimum mean-cross count.
- Optional volume filter rejects entries when current volume is above `volume_ma * --volume-max-mult`.
- Position size is based on 1% of `max(wallet_balance, 25 USDT)` divided by stop distance, then rounded to Binance quantity and notional filters. Current code also hardcodes 3x leverage and a 0.15% slippage allowance for live entries.
- Live exits use ATR-based stop-loss and take-profit orders. Optional soft exits can also close the trade on trailing stop, midline reversion, or an opposite signal.
- After a logged live exit, the bot waits `--cooldown-bars` bars before opening a new position.

`risk_pct`, `leverage`, `initial_equity`, `max_notional`, and `max_slippage_pct` exist in the `Settings` dataclass, but they are not currently exposed as CLI flags.

## Runtime behavior

- Default mode is dry-run: the bot prints the signal and order plan, but does not submit orders.
- `--live` switches to real order placement.
- `--loop` keeps polling instead of exiting after one check.
- `--sleep` sets the delay between loop iterations.
- `--log-flat` prints one-line status output even when there is no signal.
- Completed live trades are written to `reports/trade_log.csv`.
- Entry, exit, and error events are written to `reports/event_log.csv`.
- Log files are capped at 1000 rows by default; set `--trade-log-limit 0` or `--event-log-limit 0` to keep all rows.

In live mode the bot submits a market entry plus reduce-only stop and take-profit orders. If Binance rejects `STOP_MARKET` or `TAKE_PROFIT_MARKET` for the symbol/account, it falls back to `STOP` and `TAKE_PROFIT`.

## Main options

Most settings can be passed as CLI flags or by matching `BOT_*` environment variables. Boolean env vars use truthy values such as `1`, `true`, `yes`, or `on`.

- Market/data: `--symbol` (`DOGEUSDT`), `--interval` (`1h`), `--lookback` (`300`)
- Signal: `--range-lookback` (`20`), `--zscore-threshold` (`1.0`), `--position-mode` (`both`), `--reentry-buffer-bps` (`5.0`)
- Exits: `--atr-period` (`14`), `--stop-atr-mult` (`2.0`), `--take-atr-mult` (`3.0`), `--trail-atr-mult` (`0.0`), `--min-hold-bars` (`1`), `--cooldown-bars` (`1`), `--exit-on-midline`, `--exit-on-opposite-signal`
- Regime/volume filters: `--trend-ema-length` (`100`), `--trend-slope-lookback` (`5`), `--regime-threshold-bps` (`0.0`), `--trend-adx-period` (`14`), `--trend-adx-threshold` (`0.0`), `--trend-extension-atr` (`1.0`), `--max-efficiency-ratio` (`0.0`), `--min-mean-crosses` (`0`), `--volume-ma-period` (`20`), `--volume-max-mult` (`0.0`)
- Runtime/logging: `--loop`, `--sleep` (`300`), `--live`, `--testnet`, `--trade-log`, `--event-log`, `--trade-log-limit` (`1000`), `--event-log-limit` (`1000`), `--api-timeout` (`10`), `--api-retries` (`3`), `--api-retry-backoff` (`2.0`), `--log-flat`

Examples:

```bash
python bot.py --testnet --position-mode long --exit-on-midline
python bot.py --testnet --trend-adx-threshold 25 --volume-max-mult 1.5
python bot.py --testnet --loop --sleep 60 --trade-log reports/my_trades.csv
```

## Logs

- `reports/trade_log.csv` stores completed live trades with signal timestamps, expected vs filled prices, stop and take-profit levels, fees, slippage, MFE/MAE, PnL, and diagnostic flags.
- `reports/event_log.csv` stores timestamped entry, exit, and error events.

## Backtesting

Use `backtest_mean_reversion.py` for offline testing on a local OHLCV CSV file. By default it expects columns named `open_time`, `open`, `high`, `low`, `close`, and `volume`.

Example:

```bash
python backtest_mean_reversion.py data/ohlcv.csv \
  --lookback 20 \
  --zscore-threshold 1.0 \
  --position-mode both \
  --plot
```

What it does:

- Runs the same range mean-reversion logic on local CSV data.
- Can grid-search a subset of core parameters with `--optimize`.
- Writes a timestamped report folder under `reports/` with a text summary, trade CSV, plot data CSV, signal diagnostics CSV, and a PNG chart unless you pass `--no-save-plot`.

## IP change alerts

Run `python ip_monitor.py` to email yourself when the machine's public IP changes. This is useful if your Binance API key is IP-whitelisted.

Configure SMTP and alert settings in `.env`:

```dotenv
IP_ALERT_EMAIL=you@example.com
IP_SMTP_USER=you@example.com
IP_SMTP_PASSWORD=your_smtp_app_password
IP_SMTP_HOST=smtp.gmail.com
IP_SMTP_PORT=465
IP_WATCH_STATE_FILE=data/state/ip_watch.json
IP_WATCH_ENDPOINT=https://api.ipify.org
```

Flags:

- `--interval N` keeps polling every `N` seconds. `0` runs once.
- `--force` sends an email even if the IP has not changed.

## Safety

- Never commit real API keys.
- Dry-run is the default. `--live` places orders.
- Stay on Binance Futures testnet until you have verified the strategy, sizing, and logs.
- This repository is a starter template, not financial advice.
