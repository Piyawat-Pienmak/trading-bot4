import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import requests
import urllib3
from binance.error import ClientError
from binance.um_futures import UMFutures
from dotenv import load_dotenv

load_dotenv()


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


TRADE_LOG_HEADERS = [
    "trade_id",
    "strategy_name",
    "strategy_version",
    "symbol",
    "interval",
    "side",
    "entry_signal_time",
    "entry_exec_time",
    "exit_signal_time",
    "exit_exec_time",
    "entry_reason",
    "exit_reason",
    "market_regime",
    "signal_open",
    "signal_high",
    "signal_low",
    "signal_close",
    "entry_price_expected",
    "entry_price_filled",
    "exit_price_expected",
    "exit_price_filled",
    "position_size",
    "notional_usdt",
    "leverage",
    "stop_loss_price",
    "take_profit_price",
    "trailing_stop_price",
    "ema_value",
    "atr_value",
    "adx_value",
    "volume",
    "volume_ma",
    "breakout_level",
    "zscore",
    "fee_entry",
    "fee_exit",
    "slippage_entry",
    "slippage_exit",
    "max_unrealized_profit",
    "max_unrealized_loss",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "gross_pnl",
    "net_pnl",
    "r_multiple",
    "bars_held",
    "signal_valid",
    "missed_trade",
    "wrong_trade",
    "exchange_error",
    "data_error",
    "server_error",
    "manual_intervention",
    "notes",
]

EVENT_LOG_HEADERS = [
    "event_time",
    "trade_id",
    "event_type",
    "symbol",
    "side",
    "price",
    "details",
]

SIGNAL_DIAGNOSTICS_HEADERS = [
    "timestamp",
    "close_time",
    "symbol",
    "interval",
    "close",
    "pre_regime_signal",
    "final_signal",
    "regime_rejected_signal",
    "candidate_side",
    "final_side",
    "rejected_side",
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
]

DEFAULT_STRATEGY_NAME = "range_mean_reversion"
DEFAULT_STRATEGY_VERSION = "1"
PUBLIC_IP_ENDPOINTS = (
    ("https://api.ipify.org", "api.ipify.org"),
    ("https://ifconfig.me/ip", "ifconfig.me"),
    ("https://icanhazip.com", "icanhazip.com"),
)


class AuthPreflightError(RuntimeError):
    pass


def _true_range(data: pd.DataFrame) -> pd.Series:
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _adx(data: pd.DataFrame, period: int) -> pd.Series:
    up_move = data["high"].diff()
    down_move = -data["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(data)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr = atr.where(atr != 0.0)
    plus_di = (100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr).fillna(0.0)
    minus_di = (100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr).fillna(0.0)
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0.0)).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def _efficiency_ratio(series: pd.Series, lookback: int) -> pd.Series:
    net_move = (series - series.shift(lookback)).abs()
    path_length = series.diff().abs().rolling(lookback).sum()
    return (net_move / path_length.where(path_length != 0.0)).fillna(0.0)


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    symbol: str
    interval: str = "1h"
    lookback: int = 300
    range_lookback: int = 20
    zscore_threshold: float = 1.0
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    take_atr_mult: float = 3.0
    trail_atr_mult: float = 0.0
    position_mode: str = "both"
    trend_ema_length: int = 100
    trend_slope_lookback: int = 5
    regime_threshold_bps: float = 0.0
    trend_adx_period: int = 14
    trend_adx_threshold: float = 0.0
    trend_extension_atr: float = 1.0
    max_efficiency_ratio: float = 0.0
    min_mean_crosses: int = 0
    volume_ma_period: int = 20
    volume_max_mult: float = 0.0
    reentry_buffer_bps: float = 5.0
    min_hold_bars: int = 1
    cooldown_bars: int = 1
    exit_on_midline: bool = False
    exit_on_opposite_signal: bool = False
    max_notional: float | None = None
    leverage: int = 3
    risk_pct: float = 0.01
    initial_equity: float = 25.0
    max_slippage_pct: float = 0.0015
    testnet: bool = False
    live: bool = False
    base_url: str | None = None
    loop: bool = True
    poll_seconds: int = 300
    trade_log: str = "reports/trade_log.csv"
    event_log: str = "reports/event_log.csv"
    signal_diagnostics_log: str = "reports/signal_diagnostics.csv"
    trade_log_limit: int = 1000
    event_log_limit: int = 1000
    signal_diagnostics_limit: int = 1000
    strategy_name: str = DEFAULT_STRATEGY_NAME
    strategy_version: str = DEFAULT_STRATEGY_VERSION
    api_retries: int = 3
    api_retry_backoff: float = 2.0
    api_timeout: float = 10.0
    log_flat: bool = False


class FuturesBot:
    def __init__(self, settings: Settings, api_key: str, api_secret: str) -> None:
        self.settings = settings
        base_url = settings.base_url
        if settings.testnet and not base_url:
            base_url = "https://testnet.binancefuture.com"
        if not base_url:
            base_url = "https://fapi.binance.com"
        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url=base_url,
            timeout=settings.api_timeout,
        )
        self.base_url = base_url
        self._live_state: dict | None = None
        self._equity_cache: float | None = None
        self._cooldown_left = 0
        self._public_ip: str | None = None
        self._last_auth_check_at: str = ""
        self._run_auth_preflight("startup", fail_fast=True)
        self.filters = self._fetch_filters(settings.symbol)
        for path in (
            self.settings.trade_log,
            self.settings.event_log,
            self.settings.signal_diagnostics_log,
        ):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _resolve_public_ip(self) -> Tuple[str | None, str]:
        timeout = min(max(float(self.settings.api_timeout), 1.0), 5.0)
        errors: List[str] = []
        for url, label in PUBLIC_IP_ENDPOINTS:
            try:
                response = requests.get(
                    url,
                    headers={"Accept": "text/plain"},
                    timeout=timeout,
                )
                response.raise_for_status()
                ip = response.text.strip()
                if ip:
                    return ip, label
                errors.append(f"{label}: empty response")
            except requests.exceptions.RequestException as exc:
                errors.append(f"{label}: {exc}")
        return None, "; ".join(errors) if errors else "no public IP endpoint configured"

    def _run_auth_preflight(self, context: str, fail_fast: bool = False) -> bool:
        environment = "testnet" if self.settings.testnet else "mainnet"
        live_mode = "live" if self.settings.live else "dry-run"
        print(
            f"[auth] {context} check env={environment} mode={live_mode} base_url={self.base_url}"
        )

        public_ip, ip_note = self._resolve_public_ip()
        if public_ip:
            if self._public_ip and self._public_ip != public_ip:
                print(f"[auth] outbound public IP changed: {self._public_ip} -> {public_ip}")
            else:
                print(f"[auth] outbound public IP: {public_ip}")
            self._public_ip = public_ip
        else:
            print(f"[auth] outbound public IP unavailable: {ip_note}")

        try:
            balances = self._call_with_retries(self.client.balance)
        except ClientError as exc:
            print(
                f"[auth] Binance signed auth failed: code={exc.error_code} "
                f"msg={exc.error_message}"
            )
            if fail_fast:
                raise AuthPreflightError(
                    "Binance auth preflight failed; see the auth logs above."
                ) from exc
            return False
        except (requests.exceptions.RequestException, urllib3.exceptions.ProtocolError) as exc:
            print(f"[auth] Binance signed auth failed: network error: {exc}")
            if fail_fast:
                raise AuthPreflightError(
                    "Binance auth preflight failed because the signed account check did not reach Binance."
                ) from exc
            return False
        except Exception as exc:
            print(f"[auth] Binance signed auth failed: {exc}")
            if fail_fast:
                raise AuthPreflightError(
                    "Binance auth preflight failed; see the auth logs above."
                ) from exc
            return False

        usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
        wallet_balance = ""
        if usdt is not None:
            balance_value = usdt.get("balance", usdt.get("walletBalance", ""))
            if balance_value not in ("", None):
                wallet_balance = f" usdt_wallet={balance_value}"
        self._last_auth_check_at = self._utc_now_iso()
        print(
            f"[auth] Binance signed auth OK at {self._last_auth_check_at}; "
            f"USER_DATA access confirmed{wallet_balance}"
        )
        return True

    def _call_with_retries(self, func, *args, **kwargs):
        attempts = max(1, int(self.settings.api_retries))
        backoff = max(0.0, float(self.settings.api_retry_backoff))
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, urllib3.exceptions.ProtocolError) as exc:
                if self._live_state:
                    note = f"{getattr(func, '__name__', 'api_call')} failed: {exc}"
                    self._mark_live_flag("server_error", note)
                    self._log_event("server_error", note)
                if attempt >= attempts:
                    raise
                delay = backoff ** (attempt - 1)
                delay = min(delay, 30.0)
                print(
                    f"[warn] API call failed ({exc}); retry {attempt}/{attempts} in {delay:.1f}s"
                )
                time.sleep(delay)

    def _fetch_filters(self, symbol: str) -> Dict[str, float]:
        info = self._call_with_retries(self.client.exchange_info)
        match = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
        if not match:
            raise ValueError(f"Symbol {symbol} not found on exchange")
        lot = next(f for f in match["filters"] if f["filterType"] == "LOT_SIZE")
        price = next(f for f in match["filters"] if f["filterType"] == "PRICE_FILTER")
        notional = next(
            f for f in match["filters"] if f["filterType"] == "MIN_NOTIONAL"
        )
        return {
            "step_size": float(lot["stepSize"]),
            "min_qty": float(lot["minQty"]),
            "tick_size": float(price["tickSize"]),
            "min_notional": float(notional["notional"]),
        }

    def fetch_klines(self) -> pd.DataFrame:
        """Paginate Binance klines to collect up to lookback candles (1500 max per call)."""
        target = max(1, int(self.settings.lookback))
        per_call = 1500
        rows: List[list] = []
        end_time = None

        while len(rows) < target:
            limit = min(per_call, target - len(rows))
            params = {
                "symbol": self.settings.symbol,
                "interval": self.settings.interval,
                "limit": limit,
            }
            if end_time is not None:
                params["endTime"] = end_time
            try:
                batch = self._call_with_retries(self.client.klines, **params)
            except Exception as exc:
                if self._live_state:
                    note = f"failed to fetch klines after retries: {exc}"
                    self._mark_live_flag("data_error", note)
                    self._log_event("data_error", note)
                print(f"[error] failed to fetch klines after retries: {exc}")
                break
            if not batch:
                break
            rows.extend(batch)
            # Walk backward in time using earliest candle's open_time
            end_time = batch[0][0] - 1
            if len(batch) < limit:
                break

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "trades",
                "taker_base_vol",
                "taker_quote_vol",
                "ignore",
            ],
        )
        if df.empty:
            return df
        # chronological order
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        # drop last incomplete candle (close_time in future)
        now_ms = _utc_now().timestamp() * 1000
        if now_ms < float(df.iloc[-1]["close_time"]):
            df = df.iloc[:-1]
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lookback = self.settings.range_lookback
        df["rolling_mean"] = df["close"].rolling(lookback).mean()
        df["rolling_std"] = df["close"].rolling(lookback).std(ddof=0)
        df["upper_band"] = df["high"].rolling(lookback).max().shift(1)
        df["lower_band"] = df["low"].rolling(lookback).min().shift(1)
        df["zscore"] = (
            (df["close"] - df["rolling_mean"])
            / df["rolling_std"].where(df["rolling_std"] != 0.0)
        )
        df["zscore"] = df["zscore"].fillna(0.0)

        tr = _true_range(df)
        df["atr"] = tr.rolling(self.settings.atr_period).mean().bfill()
        df["adx"] = _adx(df, self.settings.trend_adx_period)
        df["efficiency_ratio"] = _efficiency_ratio(df["close"], lookback)
        df["mean_cross_count"] = _rolling_mean_crosses(
            df["close"], df["rolling_mean"], lookback
        )

        df["volume_ma"] = df["volume"].rolling(self.settings.volume_ma_period).mean().bfill()
        if self.settings.trend_ema_length > 0:
            df["trend_ema"] = df["close"].ewm(
                span=self.settings.trend_ema_length, adjust=False
            ).mean()
            df["trend_ema_prev"] = df["trend_ema"].shift(
                self.settings.trend_slope_lookback
            )
            df["trend_slope_bps"] = (
                (
                    ((df["trend_ema"] / df["trend_ema_prev"]) - 1.0)
                    / self.settings.trend_slope_lookback
                ).fillna(0.0)
                * 10000.0
            )
            df["trend_extension_atr"] = (
                (df["close"] - df["trend_ema"]).abs()
                / df["atr"].where(df["atr"] != 0.0)
            ).fillna(0.0)
        else:
            df["trend_ema"] = pd.NA
            df["trend_ema_prev"] = pd.NA
            df["trend_slope_bps"] = 0.0
            df["trend_extension_atr"] = 0.0

        buffer = self.settings.reentry_buffer_bps / 10000.0
        prev_high_break = df["close"].shift(1) > df["upper_band"].shift(1)
        prev_low_break = df["close"].shift(1) < df["lower_band"].shift(1)
        short_reentry = (
            prev_high_break
            & (df["close"] <= df["upper_band"] * (1.0 - buffer))
            & (df["zscore"] >= self.settings.zscore_threshold)
        )
        long_reentry = (
            prev_low_break
            & (df["close"] >= df["lower_band"] * (1.0 + buffer))
            & (df["zscore"] <= -self.settings.zscore_threshold)
        )

        if self.settings.trend_ema_length > 0:
            short_reentry &= df["close"] >= df["trend_ema"]
            long_reentry &= df["close"] <= df["trend_ema"]

        df["pre_regime_signal"] = 0
        if self.settings.position_mode in {"both", "short"}:
            df.loc[short_reentry, "pre_regime_signal"] = -1
        if self.settings.position_mode in {"both", "long"}:
            df.loc[long_reentry, "pre_regime_signal"] = 1

        slope_trend = (
            df["trend_slope_bps"].abs() >= self.settings.regime_threshold_bps
            if self.settings.regime_threshold_bps > 0
            else pd.Series(False, index=df.index)
        )
        adx_trend = (
            df["adx"] >= self.settings.trend_adx_threshold
            if self.settings.trend_adx_threshold > 0
            else pd.Series(False, index=df.index)
        )
        extension_trend = (
            df["trend_extension_atr"] >= self.settings.trend_extension_atr
            if self.settings.trend_extension_atr > 0
            else pd.Series(False, index=df.index)
        )
        efficient_trend = (
            df["efficiency_ratio"] > self.settings.max_efficiency_ratio
            if self.settings.max_efficiency_ratio > 0
            else pd.Series(False, index=df.index)
        )
        insufficient_crosses = (
            df["mean_cross_count"] < float(self.settings.min_mean_crosses)
            if self.settings.min_mean_crosses > 0
            else pd.Series(False, index=df.index)
        )

        df["rejected_by_regime_threshold_bps"] = False
        df["rejected_by_trend_adx_threshold"] = False
        df["rejected_by_trend_extension_atr"] = False
        df["rejected_by_efficiency_ratio"] = False
        df["rejected_by_mean_cross_count"] = False
        df["regime_reject_reasons"] = ""
        df["regime_rejected_signal"] = 0
        if self.settings.trend_ema_length > 0 and (
            self.settings.regime_threshold_bps > 0
            or self.settings.trend_adx_threshold > 0
            or self.settings.trend_extension_atr > 0
            or self.settings.max_efficiency_ratio > 0
            or self.settings.min_mean_crosses > 0
        ):
            sideways_regime = ~(
                slope_trend
                | adx_trend
                | extension_trend
                | efficient_trend
                | insufficient_crosses
            )
            rejected = (short_reentry | long_reentry) & ~sideways_regime
            df.loc[rejected, "rejected_by_regime_threshold_bps"] = slope_trend[rejected]
            df.loc[rejected, "rejected_by_trend_adx_threshold"] = adx_trend[rejected]
            df.loc[rejected, "rejected_by_trend_extension_atr"] = extension_trend[rejected]
            df.loc[rejected, "rejected_by_efficiency_ratio"] = efficient_trend[rejected]
            df.loc[rejected, "rejected_by_mean_cross_count"] = insufficient_crosses[rejected]
            df.loc[rejected, "regime_rejected_signal"] = df.loc[
                rejected, "pre_regime_signal"
            ]
            reject_checks = [
                ("rejected_by_regime_threshold_bps", "regime_threshold_bps"),
                ("rejected_by_trend_adx_threshold", "trend_adx_threshold"),
                ("rejected_by_trend_extension_atr", "trend_extension_atr"),
                ("rejected_by_efficiency_ratio", "max_efficiency_ratio"),
                ("rejected_by_mean_cross_count", "min_mean_crosses"),
            ]
            for column_name, label in reject_checks:
                matches = rejected & df[column_name]
                if matches.any():
                    current = df.loc[matches, "regime_reject_reasons"]
                    df.loc[matches, "regime_reject_reasons"] = (
                        current.where(current == "", current + "|") + label
                    )
            short_reentry &= sideways_regime
            long_reentry &= sideways_regime
            df["regime"] = "sideways"
            df.loc[~sideways_regime, "regime"] = "trend"
        else:
            df["regime"] = "sideways"

        if self.settings.volume_max_mult > 0:
            quiet_enough = df["volume"] <= df["volume_ma"] * self.settings.volume_max_mult
            short_reentry &= quiet_enough
            long_reentry &= quiet_enough

        df["signal"] = 0
        if self.settings.position_mode in {"both", "short"}:
            df.loc[short_reentry, "signal"] = -1
        if self.settings.position_mode in {"both", "long"}:
            df.loc[long_reentry, "signal"] = 1
        return df

    def _signal_direction(self, row: pd.Series) -> int:
        direction, _ = self._signal_decision(row, explain=False)
        return direction

    def _signal_detail(self, row: pd.Series) -> Tuple[int, str]:
        direction, reason = self._signal_decision(row, explain=True)
        return direction, reason or ""

    def _signal_decision(self, row: pd.Series, explain: bool) -> Tuple[int, str | None]:
        price = float(row["close"])
        rolling_mean = float(row.get("rolling_mean", float("nan")))
        upper_band = float(row.get("upper_band", float("nan")))
        lower_band = float(row.get("lower_band", float("nan")))
        zscore = float(row.get("zscore", float("nan")))
        atr = float(row["atr"])
        signal = int(row.get("signal", 0))
        reason = None
        if (
            math.isnan(rolling_mean)
            or math.isnan(upper_band)
            or math.isnan(lower_band)
            or math.isnan(zscore)
            or math.isnan(atr)
        ):
            return 0, "missing indicators" if explain else None
        if signal > 0:
            if explain:
                reason = (
                    f"LONG re-entry: close {price:.6f}, lower_band {lower_band:.6f}, "
                    f"mean {rolling_mean:.6f}, zscore {zscore:.2f}, atr {atr:.6f}"
                )
            return 1, reason
        if signal < 0:
            if explain:
                reason = (
                    f"SHORT re-entry: close {price:.6f}, upper_band {upper_band:.6f}, "
                    f"mean {rolling_mean:.6f}, zscore {zscore:.2f}, atr {atr:.6f}"
                )
            return -1, reason
        rejected_signal = int(row.get("regime_rejected_signal", 0) or 0)
        reject_reasons = str(row.get("regime_reject_reasons", "") or "")
        if rejected_signal != 0 and explain:
            side = "LONG" if rejected_signal > 0 else "SHORT"
            return 0, f"{side} candidate rejected by regime filter: {reject_reasons}"
        return 0, "conditions not aligned" if explain else None

    @staticmethod
    def _utc_now_iso() -> str:
        return _utc_now().isoformat()

    @staticmethod
    def _iso_from_ms(value: float | int | None) -> str:
        if value in (None, ""):
            return ""
        try:
            return pd.to_datetime(float(value), unit="ms", utc=True).isoformat()
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _csv_value(value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        return value

    def _append_capped_csv(
        self,
        path: str,
        fieldnames: List[str],
        row: Dict[str, object],
        limit: int,
        prepend_newest: bool = False,
        dedupe_keys: Tuple[str, ...] = (),
    ) -> None:
        rows: List[Dict[str, object]] = []
        if os.path.exists(path):
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for existing in reader:
                        rows.append({name: existing.get(name, "") for name in fieldnames})
        new_row = {name: self._csv_value(row.get(name, "")) for name in fieldnames}
        if dedupe_keys:
            rows = [
                existing
                for existing in rows
                if any(str(existing.get(key, "")) != str(new_row.get(key, "")) for key in dedupe_keys)
            ]
        if prepend_newest:
            rows = [new_row] + rows
        else:
            rows.append(new_row)
        if limit > 0:
            rows = rows[:limit] if prepend_newest else rows[-limit:]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _append_trade_note(self, note: str) -> None:
        if not note or not self._live_state:
            return
        current = str(self._live_state.get("notes", "") or "")
        self._live_state["notes"] = note if not current else f"{current} | {note}"

    def _mark_live_flag(self, field_name: str, note: str = "") -> None:
        if not self._live_state:
            return
        self._live_state[field_name] = 1
        self._append_trade_note(note)

    def _log_event(
        self,
        event_type: str,
        details: str,
        trade_id: str = "",
        side: str = "",
        price: float | None = None,
        event_time: str = "",
    ) -> None:
        state = self._live_state or {}
        resolved_side = side or str(state.get("side") or state.get("direction") or "")
        resolved_side = resolved_side.upper()
        price_value: object = ""
        if price is not None and not math.isnan(price) and not math.isinf(price):
            price_value = round(float(price), 8)
        self._append_capped_csv(
            self.settings.event_log,
            EVENT_LOG_HEADERS,
            {
                "event_time": event_time or self._utc_now_iso(),
                "trade_id": trade_id or str(state.get("trade_id", "") or ""),
                "event_type": event_type,
                "symbol": self.settings.symbol,
                "side": resolved_side,
                "price": price_value,
                "details": details,
            },
            self.settings.event_log_limit,
        )

    def _log_signal_diagnostic(self, row: pd.Series) -> None:
        pre_regime_signal = int(row.get("pre_regime_signal", 0) or 0)
        final_signal = int(row.get("signal", 0) or 0)
        rejected_signal = int(row.get("regime_rejected_signal", 0) or 0)
        if pre_regime_signal == 0 and final_signal == 0 and rejected_signal == 0:
            return
        timestamp = self._iso_from_ms(row.get("open_time", 0) or 0)
        close_time = self._iso_from_ms(row.get("close_time", 0) or 0)
        side_labels = {1: "long", -1: "short"}
        self._append_capped_csv(
            self.settings.signal_diagnostics_log,
            SIGNAL_DIAGNOSTICS_HEADERS,
            {
                "timestamp": timestamp,
                "close_time": close_time,
                "symbol": self.settings.symbol,
                "interval": self.settings.interval,
                "close": row.get("close", ""),
                "pre_regime_signal": pre_regime_signal,
                "final_signal": final_signal,
                "regime_rejected_signal": rejected_signal,
                "candidate_side": side_labels.get(pre_regime_signal, ""),
                "final_side": side_labels.get(final_signal, ""),
                "rejected_side": side_labels.get(rejected_signal, ""),
                "regime_reject_reasons": row.get("regime_reject_reasons", ""),
                "rejected_by_regime_threshold_bps": row.get("rejected_by_regime_threshold_bps", False),
                "rejected_by_trend_adx_threshold": row.get("rejected_by_trend_adx_threshold", False),
                "rejected_by_trend_extension_atr": row.get("rejected_by_trend_extension_atr", False),
                "rejected_by_efficiency_ratio": row.get("rejected_by_efficiency_ratio", False),
                "rejected_by_mean_cross_count": row.get("rejected_by_mean_cross_count", False),
                "trend_slope_bps": row.get("trend_slope_bps", ""),
                "adx": row.get("adx", ""),
                "trend_extension_atr": row.get("trend_extension_atr", ""),
                "efficiency_ratio": row.get("efficiency_ratio", ""),
                "mean_cross_count": row.get("mean_cross_count", ""),
                "regime": row.get("regime", ""),
                "trend_ema": row.get("trend_ema", ""),
            },
            self.settings.signal_diagnostics_limit,
            prepend_newest=True,
            dedupe_keys=("symbol", "interval", "timestamp"),
        )

    def _update_live_trade_extremes(self, row: pd.Series) -> None:
        if not self._live_state:
            return
        entry_price = float(self._live_state.get("entry_price", 0.0) or 0.0)
        size = float(self._live_state.get("size", 0.0) or 0.0)
        if entry_price <= 0 or size <= 0:
            return
        high = float(row.get("high", entry_price) or entry_price)
        low = float(row.get("low", entry_price) or entry_price)
        if math.isnan(high) or math.isnan(low):
            return
        direction = self._live_state.get("direction", "")
        if direction == "long":
            favorable_price = max(0.0, high - entry_price)
            adverse_price = min(0.0, low - entry_price)
        else:
            favorable_price = max(0.0, entry_price - low)
            adverse_price = min(0.0, entry_price - high)
        favorable_pnl = favorable_price * size
        adverse_pnl = adverse_price * size
        self._live_state["max_favorable_excursion"] = max(
            float(self._live_state.get("max_favorable_excursion", 0.0) or 0.0),
            favorable_price,
        )
        self._live_state["max_adverse_excursion"] = min(
            float(self._live_state.get("max_adverse_excursion", 0.0) or 0.0),
            adverse_price,
        )
        self._live_state["max_unrealized_profit"] = max(
            float(self._live_state.get("max_unrealized_profit", 0.0) or 0.0),
            favorable_pnl,
        )
        self._live_state["max_unrealized_loss"] = min(
            float(self._live_state.get("max_unrealized_loss", 0.0) or 0.0),
            adverse_pnl,
        )

    def _latest_signal(self, df: pd.DataFrame) -> Tuple[int, float, float]:
        last = df.iloc[-1]
        atr = float(last["atr"])
        price = float(last["close"])
        direction = self._signal_direction(last)
        return direction, price, atr

    def _order_trades(self, order_id: int) -> List[dict]:
        try:
            trades = self.client.get_account_trades(symbol=self.settings.symbol, orderId=order_id)
        except ClientError:
            return []
        return trades or []

    def _trades_summary(self, trades: List[dict]) -> Tuple[float, float, str]:
        if not trades:
            return 0.0, 0.0, ""
        total_qty = 0.0
        total_quote = 0.0
        total_fee = 0.0
        latest_trade_time = 0.0
        for t in trades:
            qty = float(t.get("qty", 0) or 0.0)
            price = float(t.get("price", 0) or 0.0)
            fee = float(t.get("commission", 0) or 0.0)
            total_qty += qty
            total_quote += qty * price
            total_fee += fee
            latest_trade_time = max(latest_trade_time, float(t.get("time", 0) or 0.0))
        avg_price = (total_quote / total_qty) if total_qty > 0 else 0.0
        return avg_price, total_fee, self._iso_from_ms(latest_trade_time)

    def _process_live_exit(self) -> None:
        if not self.settings.live or not self._live_state:
            return
        stop_id = self._live_state.get("stop_order_id")
        tp_id = self._live_state.get("tp_order_id")
        filled_id = None
        reason = None

        def _order_status(oid: int) -> str:
            try:
                res = self.client.query_order(symbol=self.settings.symbol, orderId=oid)
                return res.get("status", "")
            except ClientError as exc:
                msg = getattr(exc, "error_message", str(exc))
                note = f"query_order failed for {oid}: {msg}"
                self._mark_live_flag("exchange_error", note)
                self._log_event("exchange_error", note)
                return ""

        if stop_id:
            status = _order_status(stop_id)
            if status == "FILLED":
                filled_id = stop_id
                reason = "hit_sl"
        if tp_id and filled_id is None:
            status = _order_status(tp_id)
            if status == "FILLED":
                filled_id = tp_id
                reason = "hit_tp"
        if filled_id is None:
            return
        try:
            self.client.cancel_open_orders(symbol=self.settings.symbol)
        except (ClientError, AttributeError) as exc:
            msg = getattr(exc, "error_message", str(exc))
            note = f"cancel_open_orders failed after exit: {msg}"
            self._mark_live_flag("exchange_error", note)
            self._log_event("exchange_error", note)
            print(f"[warn] cancel_open_orders failed after exit: {msg}")
        trades = self._order_trades(filled_id)
        exit_price, exit_fee, exit_exec_time = self._trades_summary(trades)
        expected_price = None
        if reason == "hit_sl":
            expected_price = float(self._live_state.get("sl_price", 0.0) or 0.0)
        elif reason == "hit_tp":
            expected_price = float(self._live_state.get("tp_price", 0.0) or 0.0)
        self._log_live_exit(
            exit_price,
            exit_fee,
            reason or "",
            exit_exec_time=exit_exec_time,
            exit_signal_time=exit_exec_time,
            exit_expected_price=expected_price if expected_price and expected_price > 0 else None,
        )

    def _log_live_exit(
        self,
        exit_price: float,
        exit_fee: float,
        reason: str,
        exit_exec_time: str = "",
        exit_signal_time: str = "",
        exit_expected_price: float | None = None,
    ) -> None:
        if not self._live_state:
            return
        state = self._live_state
        direction = state.get("direction", "")
        entry_price = float(state.get("entry_price", 0.0) or 0.0)
        entry_fee = float(state.get("entry_fee", 0.0) or 0.0)
        size = float(state.get("size", 0.0) or 0.0)
        if exit_price <= 0:
            exit_price = entry_price
        entry_expected_price = float(state.get("entry_price_expected", entry_price) or entry_price)
        if exit_expected_price is None or exit_expected_price <= 0:
            exit_expected_price = exit_price
        gross_pnl = (exit_price - entry_price) * size if direction == "long" else (entry_price - exit_price) * size
        fees = entry_fee + exit_fee
        net_pnl = gross_pnl - fees
        risk_amount = abs(entry_price - float(state.get("sl_price", entry_price) or entry_price)) * size
        entry_exec_ts = str(state.get("entry_exec_time", state.get("timestamp_entry", "")) or "")
        exit_signal_ts = exit_signal_time or exit_exec_time or self._utc_now_iso()
        exit_exec_ts = exit_exec_time or exit_signal_ts
        equity_after = self._wallet_equity()
        self._equity_cache = equity_after
        trade_row = {
            "trade_id": state.get("trade_id", state.get("entry_order_id", "")),
            "strategy_name": self.settings.strategy_name,
            "strategy_version": self.settings.strategy_version,
            "symbol": self.settings.symbol,
            "interval": self.settings.interval,
            "side": state.get("side", direction.upper()),
            "entry_signal_time": state.get("entry_signal_time", ""),
            "entry_exec_time": entry_exec_ts,
            "exit_signal_time": exit_signal_ts,
            "exit_exec_time": exit_exec_ts,
            "entry_reason": state.get("entry_reason", ""),
            "exit_reason": reason or "",
            "market_regime": state.get("market_regime", ""),
            "signal_open": state.get("signal_open", ""),
            "signal_high": state.get("signal_high", ""),
            "signal_low": state.get("signal_low", ""),
            "signal_close": state.get("signal_close", ""),
            "entry_price_expected": round(entry_expected_price, 8) if entry_expected_price > 0 else "",
            "entry_price_filled": round(entry_price, 8) if entry_price > 0 else "",
            "exit_price_expected": round(exit_expected_price, 8) if exit_expected_price > 0 else "",
            "exit_price_filled": round(exit_price, 8) if exit_price > 0 else "",
            "position_size": size,
            "notional_usdt": state.get("notional_usdt", round(entry_price * size, 8)),
            "leverage": state.get("leverage", self.settings.leverage),
            "stop_loss_price": state.get("sl_price", ""),
            "take_profit_price": state.get("tp_price", ""),
            "trailing_stop_price": state.get("trailing_stop_price", ""),
            "ema_value": state.get("ema_value", ""),
            "atr_value": state.get("atr_value", ""),
            "adx_value": state.get("adx_value", ""),
            "volume": state.get("volume", ""),
            "volume_ma": state.get("volume_ma", ""),
            "breakout_level": state.get("breakout_level", ""),
            "zscore": state.get("zscore", ""),
            "fee_entry": round(entry_fee, 8),
            "fee_exit": round(exit_fee, 8),
            "slippage_entry": round(entry_price - entry_expected_price, 8) if entry_expected_price > 0 else "",
            "slippage_exit": round(exit_price - exit_expected_price, 8) if exit_expected_price > 0 else "",
            "max_unrealized_profit": state.get("max_unrealized_profit", 0.0),
            "max_unrealized_loss": state.get("max_unrealized_loss", 0.0),
            "max_favorable_excursion": state.get("max_favorable_excursion", 0.0),
            "max_adverse_excursion": state.get("max_adverse_excursion", 0.0),
            "gross_pnl": round(gross_pnl, 8),
            "net_pnl": round(net_pnl, 8),
            "r_multiple": round(net_pnl / risk_amount, 8) if risk_amount > 0 else "",
            "bars_held": int(state.get("bars_held", 0) or 0),
            "signal_valid": int(state.get("signal_valid", 1) or 0),
            "missed_trade": int(state.get("missed_trade", 0) or 0),
            "wrong_trade": int(state.get("wrong_trade", 0) or 0),
            "exchange_error": int(state.get("exchange_error", 0) or 0),
            "data_error": int(state.get("data_error", 0) or 0),
            "server_error": int(state.get("server_error", 0) or 0),
            "manual_intervention": int(state.get("manual_intervention", 0) or 0),
            "notes": state.get("notes", ""),
        }
        self._append_capped_csv(
            self.settings.trade_log,
            TRADE_LOG_HEADERS,
            trade_row,
            self.settings.trade_log_limit,
        )
        self._log_event(
            "exit_exec",
            f"{reason or 'exit'} net_pnl={net_pnl:.8f}",
            trade_id=str(trade_row["trade_id"]),
            side=str(trade_row["side"]),
            price=exit_price,
            event_time=exit_exec_ts,
        )
        print(f"[live] exit detected ({reason}); logged trade with net PnL {net_pnl:.6f}")
        self._live_state = None
        self._cooldown_left = max(0, int(self.settings.cooldown_bars))

    def _market_exit_live_position(
        self,
        reason: str,
        fallback_price: float,
        expected_price: float | None = None,
    ) -> bool:
        if not self.settings.live or not self._live_state:
            return False
        direction = self._live_state.get("direction", "")
        qty = abs(self._position_amt())
        if qty <= 0:
            return False
        signal_time = self._utc_now_iso()
        signal_price = expected_price if expected_price is not None else fallback_price
        self._log_event("exit_signal", reason, price=signal_price, event_time=signal_time)
        try:
            self.client.cancel_open_orders(symbol=self.settings.symbol)
        except (ClientError, AttributeError) as exc:
            msg = getattr(exc, "error_message", str(exc))
            note = f"cancel_open_orders failed: {msg}"
            self._mark_live_flag("exchange_error", note)
            self._log_event("exchange_error", note)
            print(f"[warn] cancel_open_orders failed: {msg}")
        exit_side = "SELL" if direction == "long" else "BUY"
        try:
            order = self.client.new_order(
                symbol=self.settings.symbol,
                side=exit_side,
                type="MARKET",
                quantity=str(qty),
                reduceOnly="true",
            )
        except ClientError as exc:
            note = f"strategy exit order failed: {exc.error_message}"
            self._mark_live_flag("exchange_error", note)
            self._log_event("exchange_error", note, price=fallback_price, event_time=signal_time)
            print(f"[error] strategy exit order failed: {exc.error_message}")
            return False
        exit_order_id = int(order.get("orderId"))
        trades = self._order_trades(exit_order_id)
        exit_price, exit_fee, exit_exec_time = self._trades_summary(trades)
        if exit_price <= 0:
            exit_price = float(order.get("avgPrice") or fallback_price)
        if not exit_exec_time:
            exit_exec_time = self._iso_from_ms(order.get("updateTime") or order.get("time"))
        self._log_live_exit(
            exit_price,
            exit_fee,
            reason,
            exit_exec_time=exit_exec_time,
            exit_signal_time=signal_time,
            exit_expected_price=signal_price,
        )
        return True

    def _bars_held(self, df: pd.DataFrame) -> int:
        if not self._live_state:
            return 0
        entry_open_time = self._live_state.get("entry_signal_open_time")
        if entry_open_time is None or "open_time" not in df.columns:
            return int(self._live_state.get("bars_held", 0) or 0)
        return int((df["open_time"].astype(float) > float(entry_open_time)).sum())

    def _maybe_exit_on_strategy_signal(self, df: pd.DataFrame) -> bool:
        if not self.settings.live or not self._live_state:
            return False
        last = df.iloc[-1]
        close = float(last.get("close", float("nan")))
        high = float(last.get("high", float("nan")))
        low = float(last.get("low", float("nan")))
        atr = float(last.get("atr", float("nan")))
        rolling_mean = float(last.get("rolling_mean", float("nan")))
        signal = int(last.get("signal", 0) or 0)
        if math.isnan(close) or math.isnan(atr):
            return False

        direction = self._live_state.get("direction", "")
        bars_held = self._bars_held(df)
        self._live_state["bars_held"] = bars_held
        self._update_live_trade_extremes(last)

        latest_open_time = float(last.get("open_time", 0) or 0)
        entry_open_time = float(self._live_state.get("entry_signal_open_time", 0) or 0)
        if latest_open_time > entry_open_time:
            if direction == "long":
                peak = float(self._live_state.get("peak_price", high) or high)
                self._live_state["peak_price"] = max(peak, high)
            elif direction == "short":
                trough = float(self._live_state.get("trough_price", low) or low)
                self._live_state["trough_price"] = min(trough, low)

        if self.settings.trail_atr_mult > 0 and latest_open_time > entry_open_time:
            if direction == "long":
                peak = float(self._live_state.get("peak_price", high) or high)
                dynamic_stop = peak - (self.settings.trail_atr_mult * atr)
                self._live_state["trailing_stop_price"] = round(dynamic_stop, 8)
                if low <= dynamic_stop:
                    return self._market_exit_live_position("trailing_stop", close, dynamic_stop)
            elif direction == "short":
                trough = float(self._live_state.get("trough_price", low) or low)
                dynamic_stop = trough + (self.settings.trail_atr_mult * atr)
                self._live_state["trailing_stop_price"] = round(dynamic_stop, 8)
                if high >= dynamic_stop:
                    return self._market_exit_live_position("trailing_stop", close, dynamic_stop)

        if bars_held < self.settings.min_hold_bars or math.isnan(rolling_mean):
            return False
        mean_hit = (
            self.settings.exit_on_midline
            and (
                (direction == "long" and close >= rolling_mean)
                or (direction == "short" and close <= rolling_mean)
            )
        )
        opposite_hit = (
            self.settings.exit_on_opposite_signal
            and (
                (direction == "long" and signal == -1)
                or (direction == "short" and signal == 1)
            )
        )
        if mean_hit:
            return self._market_exit_live_position("midline_exit", close, close)
        if opposite_hit:
            return self._market_exit_live_position("opposite_signal", close, close)
        return False

    def _wallet_equity(self) -> float:
        balances = self.client.balance()
        usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
        if not usdt:
            return 0.0
        return float(usdt.get("balance", 0.0) or 0.0)


    def _round_to(self, value: float, step: float) -> float:
        precision = max(int(round(-math.log(step, 10))) if step < 1 else 0, 0)
        return round(math.floor(value / step) * step, precision)

    def _round_up_to(self, value: float, step: float) -> float:
        precision = max(int(round(-math.log(step, 10))) if step < 1 else 0, 0)
        return round(math.ceil(value / step) * step, precision)

    def _account_equity(self) -> float:
        balances = self.client.balance()
        usdt = next((b for b in balances if b["asset"] == "USDT"), None)
        if not usdt:
            return self.settings.initial_equity
        # use wallet balance to include unrealized PnL, fallback to available
        wallet = float(usdt.get("balance", usdt.get("walletBalance", 0)))
        return max(wallet, self.settings.initial_equity)

    def _ensure_leverage(self) -> None:
        try:
            self.client.change_leverage(
                symbol=self.settings.symbol, leverage=self.settings.leverage
            )
        except ClientError as exc:
            print(f"[warn] leverage change failed: {exc}")

    def _size_position(
        self,
        direction: int,
        entry: float,
        stop_price: float,
        equity_override: float | None = None,
    ) -> Tuple[float, float]:
        risk_distance = abs(entry - stop_price)
        if risk_distance <= 0:
            raise ValueError("Stop equals entry; risk distance invalid")
        equity = equity_override if equity_override is not None else self._account_equity()
        risk_amount = equity * self.settings.risk_pct
        raw_qty = risk_amount / risk_distance
        qty = max(raw_qty, self.filters["min_qty"])
        qty = max(self._round_to(qty, self.filters["step_size"]), self.filters["min_qty"])
        notional = qty * entry
        if notional < self.filters["min_notional"]:
            min_qty = self.filters["min_notional"] / entry
            qty = self._round_up_to(max(min_qty, self.filters["min_qty"]), self.filters["step_size"])
            notional = qty * entry
        if self.settings.max_notional and notional > self.settings.max_notional:
            capped_qty = self._round_to(self.settings.max_notional / entry, self.filters["step_size"])
            qty = max(capped_qty, self.filters["min_qty"])
            notional = qty * entry
            if notional < self.filters["min_notional"]:
                raise ValueError("max_notional is below exchange minimum notional; increase it")
        if qty <= 0:
            raise ValueError("Quantity computed as zero; check filters and price")
        return qty, notional

    def _has_open_position(self) -> bool:
        return abs(self._position_amt()) > 0

    def _position_amt(self) -> float:
        positions = self.client.get_position_risk(symbol=self.settings.symbol)
        for pos in positions:
            pos_amt = float(pos.get("positionAmt", 0))
            if abs(pos_amt) > 0:
                return pos_amt
        return 0.0

    def _place_orders(
        self,
        direction: int,
        qty: float,
        stop_price: float,
        tp_price: float,
        price: float,
    ) -> dict:
        side = "BUY" if direction > 0 else "SELL"
        exit_side = "SELL" if direction > 0 else "BUY"
        order = self.client.new_order(
            symbol=self.settings.symbol,
            side=side,
            type="MARKET",
            quantity=str(qty),
        )
        entry_order_id = int(order.get("orderId"))
        try:
            stop_order = self.client.new_order(
                symbol=self.settings.symbol,
                side=exit_side,
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                quantity=str(qty),
                reduceOnly="true",
            )
        except ClientError as exc:
            if "algo order" in exc.error_message.lower() or "order type not supported" in exc.error_message.lower():
                print(
                    f"[warn] STOP_MARKET rejected ({exc.error_message}); retrying as STOP (stop-limit)"
                )
                stop_order = self.client.new_order(
                    symbol=self.settings.symbol,
                    side=exit_side,
                    type="STOP",
                    timeInForce="GTC",
                    stopPrice=str(stop_price),
                    price=str(stop_price),
                    quantity=str(qty),
                    reduceOnly="true",
                )
            else:
                raise
        try:
            tp_order = self.client.new_order(
                symbol=self.settings.symbol,
                side=exit_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(tp_price),
                quantity=str(qty),
                reduceOnly="true",
            )
        except ClientError as exc:
            if "algo order" in exc.error_message.lower() or "order type not supported" in exc.error_message.lower():
                print(
                    f"[warn] TAKE_PROFIT_MARKET rejected ({exc.error_message}); retrying as TAKE_PROFIT (stop-limit)"
                )
                tp_order = self.client.new_order(
                    symbol=self.settings.symbol,
                    side=exit_side,
                    type="TAKE_PROFIT",
                    timeInForce="GTC",
                    stopPrice=str(tp_price),
                    price=str(tp_price),
                    quantity=str(qty),
                    reduceOnly="true",
                )
            else:
                raise
        stop_order_id = int(stop_order.get("orderId"))
        tp_order_id = int(tp_order.get("orderId"))
        trades = self._order_trades(entry_order_id)
        entry_price, entry_fee, entry_exec_time = self._trades_summary(trades)
        if entry_price <= 0:
            entry_price = float(order.get("avgPrice") or price)
        if not entry_exec_time:
            entry_exec_time = self._iso_from_ms(order.get("updateTime") or order.get("time"))
        return {
            "direction": "long" if direction > 0 else "short",
            "entry_order_id": entry_order_id,
            "stop_order_id": stop_order_id,
            "tp_order_id": tp_order_id,
            "entry_price": entry_price,
            "entry_fee": entry_fee,
            "entry_exec_time": entry_exec_time or self._utc_now_iso(),
            "size": qty,
            "sl_price": stop_price,
            "tp_price": tp_price,
            "timestamp_entry": _utc_now().isoformat(),
        }

    def _build_orders(
        self, direction: int, price: float, atr: float
    ) -> Tuple[float, float, float]:
        if math.isnan(atr) or atr <= 0:
            raise ValueError("ATR is unavailable; cannot build risk orders")
        tick = self.filters["tick_size"]
        stop_distance = self.settings.stop_atr_mult * atr
        raw_stop = price - stop_distance if direction > 0 else price + stop_distance
        stop_price = self._round_to(max(raw_stop, 0.0001), tick)
        if direction > 0 and stop_price >= price:
            stop_price = self._round_to(max(price - tick, 0.0001), tick)
        elif direction < 0 and stop_price <= price:
            stop_price = self._round_to(price + tick, tick)
        tp_price = price + direction * (self.settings.take_atr_mult * atr)
        tp_price = self._round_to(max(tp_price, 0.0001), tick)
        qty, _ = self._size_position(direction, price, stop_price)
        return qty, stop_price, tp_price

    def run_once(self, cycle: int, timestamp: str, sleep_for: int | None = None) -> bool:
        self._process_live_exit()
        klines = self.fetch_klines()
        if klines.empty:
            print(f"[cycle] {cycle} @ {timestamp} [error] no klines fetched; skipping cycle")
            return False
        with_indicators = self._compute_indicators(klines)
        direction, price, atr = self._latest_signal(with_indicators)
        last = with_indicators.iloc[-1]
        rolling_mean = float(last.get("rolling_mean", float("nan")))
        upper_band = float(last.get("upper_band", float("nan")))
        lower_band = float(last.get("lower_band", float("nan")))
        zscore = float(last.get("zscore", float("nan")))
        regime = str(last.get("regime", ""))
        _, signal_reason = self._signal_detail(last)
        self._log_signal_diagnostic(last)
        if self._maybe_exit_on_strategy_signal(with_indicators):
            print(f"[cycle] {cycle} @ {timestamp}")
            return False
        stop_price = float("nan")
        tp_price = float("nan")
        qty = 0.0
        has_position = False
        signal_label = "LONG" if direction > 0 else "SHORT" if direction < 0 else "FLAT"
        if direction != 0:
            auth_context = "pre-trade" if self.settings.live else "signal"
            if not self._run_auth_preflight(auth_context):
                print("[error] auth preflight failed; skipping this signal")
                return False
            has_position = self._has_open_position()
            if not has_position:
                if self._cooldown_left > 0:
                    self._cooldown_left -= 1
                if self._cooldown_left > 0:
                    print(
                        f"[cycle] {cycle} @ {timestamp} signal={signal_label} "
                        f"cooldown_bars_left={self._cooldown_left}; skipping new entry"
                    )
                    return False
                qty, stop_price, tp_price = self._build_orders(direction, price, atr)
        if direction == 0:
            if self.settings.log_flat:
                line = (
                    f"[cycle] {cycle} @ {timestamp} "
                    f"signal={signal_label} price={price:.4f} atr={atr:.4f} "
                    f"mean={rolling_mean:.4f} zscore={zscore:.2f} "
                    f"range=({lower_band:.4f},{upper_band:.4f}) regime={regime} "
                    f"reason={signal_reason}"
                )
                if sleep_for is not None:
                    line += f" sleep={sleep_for}s"
                print(line)
                return True
            return False
        print(f"\n[cycle] {cycle} @ {timestamp}")
        if not self.settings.live:
            print(
                f"[signal] signal={signal_label} price={price:.4f} atr={atr:.4f} "
                f"mean={rolling_mean:.4f} zscore={zscore:.2f} "
                f"range=({lower_band:.4f},{upper_band:.4f}) regime={regime} "
                f"sl={stop_price:.4f} tp={tp_price:.4f} reason={signal_reason}"
            )
        if has_position:
            print("[info] open position detected; skipping new entry")
            return False
        slippage_price = price * (1 + self.settings.max_slippage_pct * direction)
        plan = (
            f"[plan] {self.settings.symbol} {('LONG' if direction>0 else 'SHORT')}, "
            f"qty={qty}, entry~{price:.4f}, stop={stop_price:.4f}, tp={tp_price:.4f}, "
            f"est.notional=${qty*price:.2f}"
        )
        if not self.settings.live:
            print(plan)
            print("[dry-run] pass --live to send orders")
            return False
        self._ensure_leverage()
        try:
            state = self._place_orders(direction, qty, stop_price, tp_price, slippage_price)
            state.update(
                {
                    "trade_id": str(state["entry_order_id"]),
                    "side": signal_label,
                    "entry_signal_time": self._iso_from_ms(last.get("close_time", 0) or 0) or timestamp,
                    "entry_signal_open_time": int(last.get("open_time", 0) or 0),
                    "entry_reason": signal_reason,
                    "market_regime": regime,
                    "signal_open": float(last.get("open", float("nan"))),
                    "signal_high": float(last.get("high", float("nan"))),
                    "signal_low": float(last.get("low", float("nan"))),
                    "signal_close": float(last.get("close", float("nan"))),
                    "entry_price_expected": price,
                    "notional_usdt": round(state["entry_price"] * qty, 8),
                    "leverage": self.settings.leverage,
                    "trailing_stop_price": "",
                    "ema_value": float(last.get("trend_ema", float("nan"))),
                    "atr_value": atr,
                    "adx_value": float(last.get("adx", float("nan"))),
                    "volume": float(last.get("volume", float("nan"))),
                    "volume_ma": float(last.get("volume_ma", float("nan"))),
                    "breakout_level": lower_band if direction > 0 else upper_band,
                    "zscore": zscore,
                    "max_unrealized_profit": 0.0,
                    "max_unrealized_loss": 0.0,
                    "max_favorable_excursion": 0.0,
                    "max_adverse_excursion": 0.0,
                    "signal_valid": 1,
                    "missed_trade": 0,
                    "wrong_trade": 0,
                    "exchange_error": 0,
                    "data_error": 0,
                    "server_error": 0,
                    "manual_intervention": 0,
                    "notes": "",
                    "bars_held": 0,
                    "peak_price": float(last.get("high", price) or price),
                    "trough_price": float(last.get("low", price) or price),
                }
            )
            self._live_state = state
            self._log_event(
                "entry_signal",
                signal_reason,
                trade_id=state["trade_id"],
                side=signal_label,
                price=price,
                event_time=state["entry_signal_time"],
            )
            self._log_event(
                "entry_exec",
                f"entry_order_id={state['entry_order_id']}",
                trade_id=state["trade_id"],
                side=signal_label,
                price=state["entry_price"],
                event_time=state["entry_exec_time"],
            )
            print(
                f"[cycle] {cycle} @ {timestamp} "
                f"signal={signal_label} price={price:.4f} atr={atr:.4f} "
                f"mean={rolling_mean:.4f} zscore={zscore:.2f} regime={regime} "
                f"sl={stop_price:.4f} tp={tp_price:.4f} qty={qty} "
                f"entry_id={state['entry_order_id']} stop_id={state['stop_order_id']} "
                f"tp_id={state['tp_order_id']} entry={state['entry_price']:.6f}"
            )
        except ClientError as exc:
            self._log_event(
                "exchange_error",
                f"entry order failed: {exc.error_message}",
                side=signal_label,
                price=price,
            )
            print(f"[error] order failed: {exc.error_message}")
        return False


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser(
        description="Binance Futures mean-reversion bot (single cycle)"
    )
    parser.add_argument("--symbol", default=os.getenv("BOT_SYMBOL", "DOGEUSDT"))
    parser.add_argument("--interval", default=os.getenv("BOT_INTERVAL", "1h"))
    parser.add_argument(
        "--lookback",
        type=int,
        default=int(os.getenv("BOT_LOOKBACK", "300")),
        help="number of candles to fetch (pagination; Binance limit 1500 per call)",
    )
    parser.add_argument(
        "--range-lookback",
        type=int,
        default=int(os.getenv("BOT_RANGE_LOOKBACK", "20")),
        help="rolling range and mean window used by the mean-reversion signal",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=float(os.getenv("BOT_ZSCORE_THRESHOLD", "1.0")),
        help="minimum absolute z-score required for range re-entry",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=int(os.getenv("BOT_ATR_PERIOD", "14")),
        help="ATR period used for stop and take-profit distances",
    )
    parser.add_argument(
        "--stop-atr-mult",
        type=float,
        default=float(os.getenv("BOT_STOP_ATR_MULT", "2.0")),
        help="stop-loss distance in ATR multiples",
    )
    parser.add_argument(
        "--take-atr-mult",
        type=float,
        default=float(os.getenv("BOT_TAKE_ATR_MULT", "3.0")),
        help="take-profit distance in ATR multiples",
    )
    parser.add_argument(
        "--trail-atr-mult",
        type=float,
        default=float(os.getenv("BOT_TRAIL_ATR_MULT", "0.0")),
        help="optional soft trailing exit in ATR multiples; 0 disables it",
    )
    parser.add_argument(
        "--position-mode",
        choices=["both", "long", "short"],
        default=os.getenv("BOT_POSITION_MODE", "both"),
        help="trade direction mode",
    )
    parser.add_argument(
        "--trend-ema-length",
        type=int,
        default=int(os.getenv("BOT_TREND_EMA_LENGTH", "100")),
        help="EMA length for trend/regime filtering; 0 disables it",
    )
    parser.add_argument(
        "--trend-slope-lookback",
        type=int,
        default=int(os.getenv("BOT_TREND_SLOPE_LOOKBACK", "5")),
        help="bars used to smooth EMA slope for regime detection",
    )
    parser.add_argument(
        "--regime-threshold-bps",
        type=float,
        default=float(os.getenv("BOT_REGIME_THRESHOLD_BPS", "0.0")),
        help="reject entries when EMA slope magnitude exceeds this bps-per-bar threshold",
    )
    parser.add_argument(
        "--trend-adx-period",
        type=int,
        default=int(os.getenv("BOT_TREND_ADX_PERIOD", "14")),
        help="ADX period used for regime detection",
    )
    parser.add_argument(
        "--trend-adx-threshold",
        type=float,
        default=float(os.getenv("BOT_TREND_ADX_THRESHOLD", "0.0")),
        help="reject entries when ADX is at or above this value; 0 disables it",
    )
    parser.add_argument(
        "--trend-extension-atr",
        type=float,
        default=float(os.getenv("BOT_TREND_EXTENSION_ATR", "0.0")),
        help="reject entries when price is this many ATRs from the trend EMA; 0 disables it",
    )
    parser.add_argument(
        "--max-efficiency-ratio",
        type=float,
        default=float(os.getenv("BOT_MAX_EFFICIENCY_RATIO", "0.0")),
        help="reject entries when rolling efficiency ratio is above this value; 0 disables it",
    )
    parser.add_argument(
        "--min-mean-crosses",
        type=int,
        default=int(os.getenv("BOT_MIN_MEAN_CROSSES", "0")),
        help="minimum mean crosses in the range window; 0 disables it",
    )
    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=int(os.getenv("BOT_VOLUME_MA_PERIOD", "20")),
        help="volume moving-average period",
    )
    parser.add_argument(
        "--volume-max-mult",
        type=float,
        default=float(os.getenv("BOT_VOLUME_MAX_MULT", "0.0")),
        help="only enter when volume is at or below volume_ma times this value; 0 disables it",
    )
    parser.add_argument(
        "--reentry-buffer-bps",
        type=float,
        default=float(os.getenv("BOT_REENTRY_BUFFER_BPS", "5.0")),
        help="required move back inside the range in basis points",
    )
    parser.add_argument(
        "--min-hold-bars",
        type=int,
        default=int(os.getenv("BOT_MIN_HOLD_BARS", "1")),
        help="minimum bars to hold before soft midline/opposite exits",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=int(os.getenv("BOT_COOLDOWN_BARS", "1")),
        help="bars to wait after a logged exit before a new entry",
    )
    parser.add_argument(
        "--exit-on-midline",
        dest="exit_on_midline",
        action="store_true",
        default=_env_bool("BOT_EXIT_ON_MIDLINE", False),
        help="soft-exit when close reverts to the rolling mean",
    )
    parser.add_argument(
        "--no-exit-on-midline",
        dest="exit_on_midline",
        action="store_false",
        help="disable soft midline exits",
    )
    parser.add_argument(
        "--exit-on-opposite-signal",
        dest="exit_on_opposite_signal",
        action="store_true",
        default=_env_bool("BOT_EXIT_ON_OPPOSITE_SIGNAL", False),
        help="soft-exit when the opposite mean-reversion signal appears",
    )
    parser.add_argument(
        "--no-exit-on-opposite-signal",
        dest="exit_on_opposite_signal",
        action="store_false",
        help="disable soft opposite-signal exits",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=int(os.getenv("BOT_SLEEP", "300")),
        help="seconds to sleep between cycles when --loop is enabled",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="keep polling continuously instead of exiting after one check",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="run one check and exit",
    )
    parser.add_argument("--live", action="store_true", help="execute live orders")
    parser.add_argument("--testnet", action="store_true", help="use Binance Futures testnet")
    parser.add_argument(
        "--trade-log",
        dest="trade_log",
        default=os.getenv("BOT_TRADE_LOG", os.getenv("BOT_LIVE_LOG", "reports/trade_log.csv")),
        help="CSV path for completed trade logs",
    )
    parser.add_argument(
        "--live-log",
        dest="trade_log",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--event-log",
        default=os.getenv("BOT_EVENT_LOG", "reports/event_log.csv"),
        help="CSV path for entry/exit/error event logs",
    )
    parser.add_argument(
        "--signal-diagnostics-log",
        default=os.getenv("BOT_SIGNAL_DIAGNOSTICS_LOG", "reports/signal_diagnostics.csv"),
        help="CSV path for rolling signal diagnostics",
    )
    parser.add_argument(
        "--trade-log-limit",
        type=int,
        default=int(os.getenv("BOT_TRADE_LOG_LIMIT", "1000")),
        help="maximum number of rows to keep in trade_log.csv; 0 keeps all rows",
    )
    parser.add_argument(
        "--event-log-limit",
        type=int,
        default=int(os.getenv("BOT_EVENT_LOG_LIMIT", "1000")),
        help="maximum number of rows to keep in event_log.csv; 0 keeps all rows",
    )
    parser.add_argument(
        "--signal-diagnostics-limit",
        type=int,
        default=int(os.getenv("BOT_SIGNAL_DIAGNOSTICS_LIMIT", "1000")),
        help="maximum number of rows to keep in signal_diagnostics.csv; 0 keeps all rows",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=float(os.getenv("BOT_API_TIMEOUT", "10")),
        help="HTTP timeout (seconds) for Binance requests",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=int(os.getenv("BOT_API_RETRIES", "3")),
        help="Retries for recoverable Binance HTTP errors",
    )
    parser.add_argument(
        "--api-retry-backoff",
        type=float,
        default=float(os.getenv("BOT_API_RETRY_BACKOFF", "2.0")),
        help="Exponential backoff base (seconds) between retries",
    )
    parser.add_argument(
        "--log-flat",
        action="store_true",
        help="log one-line output when signal is FLAT",
    )
    args = parser.parse_args()
    if args.range_lookback < 2:
        raise SystemExit("--range-lookback must be at least 2.")
    if args.zscore_threshold <= 0:
        raise SystemExit("--zscore-threshold must be > 0.")
    if args.atr_period < 1:
        raise SystemExit("--atr-period must be at least 1.")
    if args.stop_atr_mult <= 0 or args.take_atr_mult <= 0:
        raise SystemExit("--stop-atr-mult and --take-atr-mult must be > 0.")
    if args.trail_atr_mult < 0:
        raise SystemExit("--trail-atr-mult must be >= 0.")
    if args.position_mode not in {"both", "long", "short"}:
        raise SystemExit("--position-mode must be one of: both, long, short.")
    if args.trend_ema_length < 0:
        raise SystemExit("--trend-ema-length must be >= 0.")
    if args.trend_slope_lookback < 1:
        raise SystemExit("--trend-slope-lookback must be at least 1.")
    if args.regime_threshold_bps < 0:
        raise SystemExit("--regime-threshold-bps must be >= 0.")
    if args.trend_adx_period < 1:
        raise SystemExit("--trend-adx-period must be at least 1.")
    if args.trend_adx_threshold < 0 or args.trend_extension_atr < 0:
        raise SystemExit("Trend filter thresholds must be >= 0.")
    if args.max_efficiency_ratio < 0 or args.min_mean_crosses < 0:
        raise SystemExit("Efficiency and mean-cross filters must be >= 0.")
    if args.volume_ma_period < 1 or args.volume_max_mult < 0:
        raise SystemExit("Volume filter settings are invalid.")
    if args.reentry_buffer_bps < 0:
        raise SystemExit("--reentry-buffer-bps must be >= 0.")
    if args.min_hold_bars < 0 or args.cooldown_bars < 0:
        raise SystemExit("Hold/cooldown bars must be >= 0.")
    if (
        args.trade_log_limit < 0
        or args.event_log_limit < 0
        or args.signal_diagnostics_limit < 0
    ):
        raise SystemExit("Log row limits must be >= 0.")
    required_lookback = max(
        args.range_lookback + 2,
        args.atr_period + 2,
        args.trend_ema_length + args.trend_slope_lookback + 2
        if args.trend_ema_length > 0
        else 2,
        args.trend_adx_period + 2,
        args.volume_ma_period + 2,
    )
    if args.lookback < required_lookback:
        raise SystemExit(
            f"--lookback must be at least {required_lookback} for the selected strategy settings."
        )
    testnet_env = os.getenv("BINANCE_TESTNET", "0") == "1"
    return Settings(
        symbol=args.symbol.upper(),
        interval=args.interval,
        lookback=args.lookback,
        range_lookback=args.range_lookback,
        zscore_threshold=args.zscore_threshold,
        atr_period=args.atr_period,
        stop_atr_mult=args.stop_atr_mult,
        take_atr_mult=args.take_atr_mult,
        trail_atr_mult=args.trail_atr_mult,
        position_mode=args.position_mode,
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
        exit_on_midline=args.exit_on_midline,
        exit_on_opposite_signal=args.exit_on_opposite_signal,
        testnet=args.testnet or testnet_env,
        live=args.live,
        loop=args.loop,
        poll_seconds=max(1, args.sleep),
        trade_log=args.trade_log,
        event_log=args.event_log,
        signal_diagnostics_log=args.signal_diagnostics_log,
        trade_log_limit=args.trade_log_limit,
        event_log_limit=args.event_log_limit,
        signal_diagnostics_limit=args.signal_diagnostics_limit,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        api_retry_backoff=args.api_retry_backoff,
        log_flat=args.log_flat,
    )


def main() -> None:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_API_KEY/SECRET required in .env")
    settings = parse_settings()
    try:
        bot = FuturesBot(settings=settings, api_key=api_key, api_secret=api_secret)
    except AuthPreflightError as exc:
        print(f"[error] {exc}")
        return
    cycle = 0
    try:
        while True:
            cycle += 1
            timestamp = _utc_now().isoformat()
            sleep_for = max(1, settings.poll_seconds) if settings.loop else None
            flat_logged = bot.run_once(cycle, timestamp, sleep_for)
            if not settings.loop:
                break
            if settings.log_flat and not flat_logged and sleep_for is not None:
                print(f"[sleep] waiting {sleep_for}s before next check")
            if sleep_for is not None:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[info] interrupted by user; exiting")


if __name__ == "__main__":
    main()
