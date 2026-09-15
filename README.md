# Binance Futures Mean-Reversion Bot

Python bot for Binance USD-M futures built around a range mean-reversion strategy. The current `bot.py` defaults to dry-run, can run once or in a loop, and only places orders when you pass `--live`.

It still talks to Binance in dry-run. Exchange info and market data are always fetched, and account data is used when sizing a signal, so `BINANCE_API_KEY` and `BINANCE_API_SECRET` must be present before running it.

## Quick start

- Python 3.10+ recommended.
- Install dependencies:
  ```bash
  pip install binance-futures-connector pandas python-dotenv requests matplotlib
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
  python bot.py --testnet --no-loop
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
- Live execution evidence is committed to `data/accounting/ledger.sqlite3` before order submission and after acknowledgements.
- Trade summaries are exported to `reports/ledger_trades.csv`. Both open trades and closed trades with pending accounting are included.
- Entry, exit, and error events are exported to `reports/ledger_events.csv`.
- CSV reports are capped at 1000 rows by default; set `--trade-log-limit 0` or `--event-log-limit 0` to export all rows. The ledger retains all records regardless of report limits.

In live mode the bot submits a market entry plus reduce-only conditional stop and take-profit orders through Binance's algo-order API. If protection cannot be confirmed, the bot attempts to close its verified exposure and journals that emergency exit. An uncertain submission is queried by its saved client order ID; it is never blindly resubmitted. A candle can generate at most one entry attempt, including across restarts.

## Main options

Most settings can be passed as CLI flags or by matching `BOT_*` environment variables. Boolean env vars use truthy values such as `1`, `true`, `yes`, or `on`.

- Market/data: `--symbol` (`DOGEUSDT`), `--interval` (`1h`), `--lookback` (`300`)
- Signal: `--range-lookback` (`20`), `--zscore-threshold` (`1.0`), `--position-mode` (`both`), `--reentry-buffer-bps` (`5.0`)
- Exits: `--atr-period` (`14`), `--stop-atr-mult` (`2.0`), `--take-atr-mult` (`3.0`), `--trail-atr-mult` (`0.0`), `--min-hold-bars` (`1`), `--cooldown-bars` (`1`), `--exit-on-midline`, `--exit-on-opposite-signal`
- Regime/volume filters: `--trend-ema-length` (`100`), `--trend-slope-lookback` (`5`), `--regime-threshold-bps` (`0.0`), `--trend-adx-period` (`14`), `--trend-adx-threshold` (`0.0`), `--trend-extension-atr` (`1.0`), `--max-efficiency-ratio` (`0.0`), `--min-mean-crosses` (`0`), `--volume-ma-period` (`20`), `--volume-max-mult` (`0.0`)
- Runtime/logging: `--loop`, `--sleep` (`300`), `--live`, `--testnet`, `--trade-log`, `--event-log`, `--signal-diagnostics-log`, `--trade-log-limit` (`1000`), `--event-log-limit` (`1000`), `--signal-diagnostics-limit` (`1000`), `--api-timeout` (`10`), `--api-retries` (`3`), `--api-retry-backoff` (`2.0`), `--log-flat`

Examples:

```bash
python bot.py --testnet --position-mode long --exit-on-midline
python bot.py --testnet --trend-adx-threshold 25 --volume-max-mult 1.5
python bot.py --testnet --loop --sleep 60 --trade-log reports/my_trades.csv
```

## Trade accounting

`trade_ledger.py` stores trades, order intents, raw fills, income, events, account snapshots, and history coverage in SQLite. `trade_reconciliation.py` checks those records against Binance using read-only requests. `bot.py` uses the order journal for every entry, protective order, strategy exit, and emergency closure. `accounting.py` provides maintenance commands without running the trading loop.

The bot reconciles at startup and on every cycle. It restores position metadata, trailing-stop state and cooldown state. If a process dies after an order executes, reconciliation queries the persisted client ID and retrieves the actual fills. If an entry is recovered without confirmed protection, the live bot attempts an emergency closure of its verified quantity. Cancellation targets only the tracked trade's protective orders. `accounting.py reconcile` itself never submits or cancels orders.

The current execution model requires one-way mode and exclusive ownership of the traded symbol's position. A pre-existing or manually changed position is reported as unresolved and blocks new entries. An operator can explicitly link a verified manual closing order to its trade. The bot does not guess ownership from matching price or time alone.

### Confirmed versus pending results

Each trade has separate `execution_status` and `accounting_status` fields. An executed closure can be `closed` while its accounting is still `pending`. A blank P&L or fee means unavailable, not zero. Only `accounting_status=complete` rows are ready to include in a confirmed performance total; always report the pending/unresolved count alongside that total.

Confirmed net P&L is the sum of Binance fill `realizedPnl`, minus entry/exit commissions, plus signed funding payments. Commission and realized-PnL records from income history are retained for audit but are not counted again. Transfers and other account income remain separate. Partial fills are matched by quantity, and duplicate fill IDs cannot create additional profit or fees. Conflicting duplicate evidence is rejected rather than overwritten.

Accounting remains pending when fills, fees, order outcomes, or history coverage are incomplete. Funding coverage is checked with a 60-second settlement delay and a rolling overlap to retrieve delayed records. Unrelated orders during the trade or overlapping trades prevent automatic funding attribution. Fees are retained by their original currency; non-USDT commissions requiring conversion leave net P&L pending. No exchange rate is invented. MFE/MAE remain candle-based observations, not exchange-certified tick history.

### Storage and account identity

- Default ledger: `data/accounting/ledger.sqlite3`; override with `--ledger-path` or `BOT_LEDGER_PATH`.
- Automatic daily SQLite backups: `data/accounting/backups/`. Backups also work while WAL mode is active.
- Account namespaces include the exchange base URL and a hash of the API key. No API secret is stored. Set `BOT_ACCOUNT_ID` / `--account-id` to a stable, unique account label before first use if you need continuity across API-key rotations. Reuse it only for that same account. Mainnet and testnet remain separate.
- One bot or maintenance writer may use an account namespace at a time. Stop the bot before imports, explicit order linking, or standalone reconciliation. Use separate report paths if you operate multiple account namespaces.
- CSV exports use temporary files and atomic replacement. If an overridden report path contains an older CSV, a copy with a `.legacy-<hash>.csv` suffix is preserved before replacement. A report containing another account scope is not overwritten. Their row limits do not remove ledger evidence. Dry-run does not create or reconcile a live ledger.
- `reports/signal_diagnostics.csv` remains a separate rolling signal diagnostic log; it is not a record of executed trades.

### Preserve and investigate older CSV history

The previous `reports/trade_log.csv` and `reports/event_log.csv` are not overwritten by the new defaults. Import them into an unassigned namespace:

```bash
python accounting.py import-legacy
python accounting.py backup data/accounting/backups/manual-backup.sqlite3
python accounting.py list-scopes
```

The import preserves original rows and writes `reports/accounting/legacy_review.csv`. Previously reported totals appear as `legacy_reported_net_pnl`; confirmed `net_pnl` stays blank. Missing exits and emergency closures are listed separately. Repeating an import does not duplicate records. Old records have no account/testnet identity, so they are not automatically assigned to the current API key.

Ordinary Binance trade and income history endpoints currently document a three-month lookback. Older gaps require historical exports and may remain unresolved if evidence is unavailable. Use Binance-format JSON arrays or CSVs whose field names match the API (`id`, `orderId`, `symbol`, `qty`, `price`, `side`, `time`, `commission`, `commissionAsset`, `realizedPnl`; income uses `incomeType`, `tranId`, `asset`, `income`, `time`, `symbol`). Convert differently formatted exchange downloads to those named fields without replacing missing values with zero.

### Read-only reconciliation and exports

These commands require the bot to be stopped. Reconciliation uses `.env` credentials and the same account label/environment as the bot, and prints the exact scope for subsequent commands:

```bash
python accounting.py reconcile --symbol DOGEUSDT --testnet
python accounting.py --scope 'EXACT_SCOPE_FROM_RECONCILE' export --output-dir reports/accounting/testnet
```

Exports contain trades, order-to-trade mappings, attributed/unassigned fills, income, account snapshots, events, and legacy review rows. Never combine mainnet, testnet, or different accounts into one performance total.

To repair history after verifying account and order ownership, register a known entry and explicitly attach its exits. Replace placeholders with verified values; each command below is a separate step:

```bash
python accounting.py --scope 'EXACT_SCOPE' adopt-entry --order-id ENTRY_ID --symbol DOGEUSDT --direction long --time '2026-08-01T12:00:00+00:00'
python accounting.py --scope 'EXACT_SCOPE' link-order --trade-id RETURNED_TRADE_ID --order-id EXIT_ID --role exit
python accounting.py --scope 'EXACT_SCOPE' import-orders verified_orders.json
python accounting.py --scope 'EXACT_SCOPE' import-fills verified_fills.json
python accounting.py --scope 'EXACT_SCOPE' import-income verified_income.json
```

`adopt-entry` only records the relationship; it does not place an order. Imported order evidence must include `symbol`, `orderId`, `status`, and `executedQty`. Full historical coverage can be explicitly certified on both `import-fills` and `import-income` with `--complete-from ISO_TIME --complete-through ISO_TIME --symbol DOGEUSDT`, but only after independently verifying each export covers the entire interval. Ordinary coverage is established by reconciliation. Imports and account snapshots do not by themselves prove a complete history.

API references: [account trades and order lookup](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade), [income and historical exports](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account).

### Offline tests

```bash
python -m unittest discover -s tests -v
```

Tests use a fake exchange and temporary databases. They cover execution before a crash/timeout, restart recovery, emergency closure, manual exit attribution, partial fills, delayed fees/funding, duplicate imports, pagination, account isolation, backups, and dry-run behavior. They never connect to Binance or place real orders.

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
