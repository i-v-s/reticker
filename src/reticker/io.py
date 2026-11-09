"""Artifact IO and read-only market data sources."""

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from .config import RetickerError, Selection, utc

CANDLE_COLUMNS = [
    "time",
    "category",
    "symbol",
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
]
TICKER_COLUMNS = [
    "time",
    "category",
    "symbol",
    "bid1_price",
    "ask1_price",
    "bid1_size",
    "ask1_size",
    "last_price",
    "mark_price",
    "index_price",
]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        raise RetickerError(f"Invalid JSON document: {path}") from None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class DataSource(Protocol):
    def candles(self, selection: Selection, start, end) -> pd.DataFrame: ...
    def tickers(self, selection: Selection, start, end) -> pd.DataFrame: ...


class FrameSource:
    """In-memory source for notebooks and reproducible experiments. Times must denote UTC."""

    def __init__(self, candles, tickers=None):
        self._candles = candles.copy()
        self._tickers = (
            tickers.copy() if tickers is not None else pd.DataFrame(columns=TICKER_COLUMNS)
        )

    @staticmethod
    def _select(frame, selection, start, end):
        frame = frame.copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        mask = (frame.time >= utc(start)) & (frame.time < utc(end))
        mask &= frame.symbol.isin(selection.symbols) & frame.category.isin(selection.markets)
        return frame.loc[mask].reset_index(drop=True)

    def candles(self, selection, start, end):
        return self._select(self._candles, selection, start, end)

    def tickers(self, selection, start, end):
        return self._select(self._tickers, selection, start, end)


class ClickHouseSource:
    """Native ClickHouse reader; credentials are never included in artifacts or errors."""

    def __init__(self, url):
        from clickhouse_driver import Client

        try:
            parsed = urlsplit(url if "://" in url else "clickhouse://" + url)
            if parsed.scheme not in {"clickhouse", "clickhouses"} or not parsed.hostname:
                raise ValueError
            self.client = Client(
                host=parsed.hostname,
                port=parsed.port or (9440 if parsed.scheme == "clickhouses" else 9000),
                user=unquote(parsed.username or "default"),
                password=unquote(parsed.password or ""),
                database=unquote(parsed.path.lstrip("/") or "default"),
                secure=parsed.scheme == "clickhouses",
                connect_timeout=10,
                send_receive_timeout=120,
                settings={"readonly": 2, "max_execution_time": 90, "use_client_time_zone": False},
            )
        except (ValueError, TypeError):
            raise RetickerError(
                "Invalid CH_URL; expected [clickhouse://]user:password@host:port/database"
            ) from None

    @classmethod
    def from_env(cls, env_file=".env"):
        value = os.environ.get("CH_URL") or dotenv_values(env_file).get("CH_URL")
        if not value:
            raise RetickerError("Set CH_URL in the environment or .env")
        return cls(value)

    def _read(self, sql, params=None):
        try:
            rows, columns = self.client.execute(sql, params or {}, with_column_types=True)
        except Exception as exc:
            raise RetickerError(
                f"ClickHouse read failed ({type(exc).__name__}); check connectivity, schema, and read access"
            ) from None
        frame = pd.DataFrame(rows, columns=[c[0] for c in columns])
        for column in ("time", "first_time", "last_time"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], unit="s", utc=True)
        return frame

    @staticmethod
    def _params(selection, start, end):
        return {
            "symbols": selection.symbols,
            "markets": selection.markets,
            "start": int(utc(start).timestamp()),
            "end": int(utc(end).timestamp()),
        }

    def candles(self, selection, start, end):
        return self._read(
            """
            SELECT toUnixTimestamp(time) AS time, category, symbol, `interval`,
                   open, high, low, close, volume, turnover
            FROM kline FINAL
            WHERE category IN %(markets)s AND symbol IN %(symbols)s AND `interval` = '1'
              AND time >= toDateTime(%(start)s, 'UTC') AND time < toDateTime(%(end)s, 'UTC')
            ORDER BY category, symbol, time
        """,
            self._params(selection, start, end),
        )

    def tickers(self, selection, start, end):
        frames = []
        for market in selection.markets:
            table = "spot_ticker" if market == "spot" else "future_ticker"
            extra = (
                "0.0 AS mark_price, 0.0 AS index_price"
                if market == "spot"
                else "mark_price, index_price"
            )
            category = "'spot' AS category" if market == "spot" else "category"
            frames.append(
                self._read(
                    f"""
                SELECT toUnixTimestamp(time) AS time, {category}, symbol, bid1_price, ask1_price,
                       bid1_size, ask1_size, last_price, {extra}
                FROM {table} FINAL
                WHERE symbol IN %(symbols)s AND time >= toDateTime(%(start)s, 'UTC')
                  AND time < toDateTime(%(end)s, 'UTC')
                  {"AND category = 'linear'" if market == "linear" else ""}
                ORDER BY symbol, time
            """,
                    self._params(selection, start, end),
                )
            )
        return pd.concat(frames, ignore_index=True)

    def coverage(self, selection):
        params = {"symbols": selection.symbols, "markets": selection.markets}
        frames = [
            self._read(
                """
            SELECT 'kline' AS source, category, symbol, count() AS rows,
                   toUnixTimestamp(min(time)) AS first_time, toUnixTimestamp(max(time)) AS last_time
            FROM kline FINAL WHERE category IN %(markets)s AND symbol IN %(symbols)s AND `interval` = '1'
            GROUP BY category, symbol
        """,
                params,
            )
        ]
        for market in selection.markets:
            table = "spot_ticker" if market == "spot" else "future_ticker"
            category = "'spot' AS category" if market == "spot" else "category"
            frames.append(
                self._read(
                    f"""
                SELECT '{table}' AS source, {category}, symbol, count() AS rows,
                       toUnixTimestamp(min(time)) AS first_time, toUnixTimestamp(max(time)) AS last_time
                FROM {table} FINAL WHERE symbol IN %(symbols)s
                  {"AND category = 'linear'" if market == "linear" else ""}
                GROUP BY category, symbol
            """,
                    params,
                )
            )
        return pd.concat(frames, ignore_index=True)
