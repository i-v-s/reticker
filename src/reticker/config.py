"""Validated, serializable public configuration."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class RetickerError(ValueError):
    """An actionable data, configuration, or artifact error."""


def utc(value) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        raise RetickerError("Invalid timestamp; use an ISO 8601 date/time") from None
    if pd.isna(result):
        raise RetickerError("A timestamp is required")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def minute_range(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = utc(start), utc(end)
    if start >= end or start != start.floor("min") or end != end.floor("min"):
        raise RetickerError("Use start < end, both on UTC minute boundaries; end is exclusive")
    return start, end


@dataclass(frozen=True)
class Selection:
    symbols: tuple[str, ...]
    markets: tuple[str, ...] = ("spot", "linear")

    def __post_init__(self):
        symbols = tuple(sorted(set(self.symbols)))
        markets = tuple(sorted(set(self.markets)))
        if not symbols or any(not re.fullmatch(r"[A-Z0-9]+USDT", s) for s in symbols):
            raise RetickerError("Select at least one uppercase USDT symbol")
        if not markets or not set(markets) <= {"spot", "linear"}:
            raise RetickerError("Supported markets: spot, linear (USDT perpetuals)")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "markets", markets)

    @property
    def keys(self):
        return tuple((m, s) for m in self.markets for s in self.symbols)


@dataclass(frozen=True)
class PreparationConfig:
    selection: Selection
    start: str
    end: str
    tolerance_seconds: int = 5

    def __post_init__(self):
        minute_range(self.start, self.end)
        if not 0 <= self.tolerance_seconds < 30:
            raise RetickerError("Snapshot tolerance must be between 0 and 29 seconds")


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05
    seed: int = 42
    threads: int = -1
    block_lengths: tuple[int, ...] = (1, 5, 15, 60)
    tuning_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self):
        if self.iterations < 1 or not 1 <= self.depth <= 12 or self.learning_rate <= 0:
            raise RetickerError("Invalid CatBoost training parameters")
        if not self.block_lengths or any(n < 1 for n in self.block_lengths):
            raise RetickerError("Block lengths must be positive")
        if (
            not self.tuning_seeds
            or any(seed < 0 for seed in self.tuning_seeds)
            or self.seed < 0
            or self.threads == 0
        ):
            raise RetickerError("Use nonnegative seeds, nonempty tuning seeds, and nonzero threads")


@dataclass(frozen=True)
class GenerationConfig:
    start: str
    end: str
    symbols: tuple[str, ...] | None = None
    seed: int = 42
    chunk_days: int = 1

    def __post_init__(self):
        minute_range(self.start, self.end)
        if self.seed < 0 or self.chunk_days < 1:
            raise RetickerError("Use a nonnegative seed and a positive chunk_days")


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(20)))

    def __post_init__(self):
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise RetickerError("Evaluation requires nonnegative seeds")


def new_directory(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RetickerError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
