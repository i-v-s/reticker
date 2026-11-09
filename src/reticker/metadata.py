"""Frozen instrument rules with optional historical overrides."""

import json
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import RetickerError, utc
from .io import read_json


class MetadataCatalog:
    """Mapping market:symbol -> rules, optionally bounded by valid_from / valid_to."""

    def __init__(self, rules):
        if not isinstance(rules, dict):
            raise RetickerError("Metadata must map market:symbol keys to lists of rules")
        self.rules = rules
        for key, entries in rules.items():
            if (
                not isinstance(entries, list)
                or not entries
                or not all(isinstance(e, dict) for e in entries)
            ):
                raise RetickerError(f"No instrument rules for {key}")
            bounded = []
            for entry in entries:
                try:
                    for name in ("tick_size", "quantity_step"):
                        value = Decimal(str(entry[name]))
                        if not value.is_finite() or value <= 0 or not np.isfinite(float(value)):
                            raise ValueError
                except (KeyError, InvalidOperation, ValueError):
                    raise RetickerError(f"Invalid tick_size or quantity_step for {key}") from None
                start = utc(entry["valid_from"]) if entry.get("valid_from") else None
                end = utc(entry["valid_to"]) if entry.get("valid_to") else None
                if start is not None and end is not None and start >= end:
                    raise RetickerError(f"Invalid rule interval for {key}")
                if start is not None or end is not None:
                    bounded.append((start, end))
            bounded.sort(
                key=lambda p: p[0] if p[0] is not None else pd.Timestamp.min.tz_localize("UTC")
            )
            for previous, current in zip(bounded, bounded[1:]):
                if previous[1] is None or current[0] is None or previous[1] > current[0]:
                    raise RetickerError(f"Overlapping historical rules for {key}")
            if sum(not e.get("valid_from") and not e.get("valid_to") for e in entries) > 1:
                raise RetickerError(f"Multiple fallback rules for {key}")

    def values(self, market, symbol, times):
        key = f"{market}:{symbol}"
        if key not in self.rules:
            raise RetickerError(f"Missing instrument metadata: {key}")
        times = pd.DatetimeIndex(times)
        tick = np.full(len(times), np.nan)
        qty = np.full(len(times), np.nan)
        current = np.zeros(len(times), dtype=bool)
        entries = sorted(
            self.rules[key], key=lambda e: bool(e.get("valid_from") or e.get("valid_to"))
        )
        for entry in entries:
            mask = np.ones(len(times), dtype=bool)
            if entry.get("valid_from"):
                mask &= times >= utc(entry["valid_from"])
            if entry.get("valid_to"):
                mask &= times < utc(entry["valid_to"])
            tick[mask], qty[mask] = float(entry["tick_size"]), float(entry["quantity_step"])
            current[mask] = entry.get("source") == "bybit_current"
        if np.isnan(tick).any():
            raise RetickerError(f"Instrument metadata does not cover requested history: {key}")
        return tick, qty, current

    @classmethod
    def fetch(cls, selection, overrides=None):
        rules = read_json(overrides) if overrides else {}
        for market, symbol in selection.keys:
            key = f"{market}:{symbol}"
            entries = rules.setdefault(key, [])
            if any(not e.get("valid_from") and not e.get("valid_to") for e in entries):
                continue
            try:
                query = urlencode({"category": market, "symbol": symbol})
                with urlopen(
                    f"https://api.bybit.com/v5/market/instruments-info?{query}", timeout=20
                ) as response:
                    payload = json.load(response)
                if payload.get("retCode") != 0:
                    raise ValueError
                candidates = payload["result"]["list"]
                instrument = next(i for i in candidates if i["symbol"] == symbol)
                if market == "linear" and instrument.get("contractType") != "LinearPerpetual":
                    raise RetickerError(f"Only linear perpetuals are supported: {symbol}")
                if instrument.get("quoteCoin") != "USDT":
                    raise RetickerError(f"Only USDT instruments are supported: {symbol}")
                lot = instrument["lotSizeFilter"]
                entries.insert(
                    0,
                    {
                        "tick_size": instrument["priceFilter"]["tickSize"],
                        "quantity_step": lot["basePrecision"]
                        if market == "spot"
                        else lot["qtyStep"],
                        "source": "bybit_current",
                        "fetched_at": str(pd.Timestamp.now(tz="UTC")),
                    },
                )
            except RetickerError:
                raise
            except Exception:
                raise RetickerError(
                    f"Cannot fetch Bybit metadata for {key}; supply a metadata JSON fallback"
                ) from None
        return cls(rules)
