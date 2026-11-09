"""Causal minute features, approximate boundary labels, and chronological splits."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PreparationConfig, RetickerError, Selection, minute_range, new_directory, utc
from .io import digest, read_json, write_json
from .metadata import MetadataCatalog

TARGETS = {
    "spot": ("mid_offset", "log_spread", "log_bid_size", "log_ask_size"),
    "linear": (
        "mid_offset",
        "log_spread",
        "log_bid_size",
        "log_ask_size",
        "log_mark_ratio",
        "log_index_ratio",
    ),
}
PRICE_COLUMNS = ("open", "high", "low", "close")
DERIVED = (
    "last_price",
    "prev_price24h",
    "price24h_pcnt",
    "high_price24h",
    "low_price24h",
    "volume24h",
    "turnover24h",
    "prev_price1h",
)


def deduplicate(frame, keys):
    frame = frame.copy()
    if frame.empty:
        return frame, 0
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        distinct = frame.loc[duplicates].drop_duplicates()
        if distinct.duplicated(keys).any():
            raise RetickerError(
                "Conflicting duplicate source keys; resolve them before preparing data"
            )
    count = int(frame.duplicated(keys).sum())
    return frame.drop_duplicates(keys).reset_index(drop=True), count


def candle_features(candles, selection, metadata, start, end):
    """Compute each boundary solely from candles closing at or before it."""
    start, end = minute_range(start, end)
    candles = candles.copy()
    required = {"time", "category", "symbol", "interval", *PRICE_COLUMNS, "volume", "turnover"}
    if not required <= set(candles):
        raise RetickerError(f"Missing candle columns: {sorted(required - set(candles))}")
    candles["time"] = pd.to_datetime(candles.time, utc=True)
    candles = candles.loc[candles["interval"].astype(str) == "1"].copy()
    candles, duplicates = deduplicate(candles, ["category", "symbol", "time"])
    frames = []
    stats = {
        "duplicate_candles": duplicates,
        "invalid_candles": 0,
        "missing_streams": [],
        "streams": {},
    }
    for market, symbol in selection.keys:
        part = candles.loc[(candles.category == market) & (candles.symbol == symbol)].copy()
        if part.empty:
            stats["missing_streams"].append(f"{market}:{symbol}")
            continue
        numeric = part[[*PRICE_COLUMNS, "volume", "turnover"]].to_numpy(dtype=float)
        valid = np.isfinite(numeric).all(axis=1)
        valid &= (part[list(PRICE_COLUMNS)] > 0).all(axis=1).to_numpy()
        valid &= (part.low <= part[["open", "close"]].min(axis=1)).to_numpy()
        valid &= (part.high >= part[["open", "close"]].max(axis=1)).to_numpy()
        valid &= ((part.volume >= 0) & (part.turnover >= 0)).to_numpy()
        valid &= (part.time == part.time.dt.floor("min")).to_numpy()
        stats["invalid_candles"] += int((~valid).sum())
        part = part.loc[valid].set_index("time").sort_index()
        if part.empty:
            stats["missing_streams"].append(f"{market}:{symbol}")
            continue
        # Reindexing prevents a rolling N-row window from silently spanning missing minutes.
        grid = pd.date_range(
            start - pd.Timedelta(minutes=1441), end - pd.Timedelta(minutes=1), freq="min"
        )
        part = part.reindex(grid)
        close = part.close
        f = pd.DataFrame(index=grid + pd.Timedelta(minutes=1))
        f.index.name = "time"
        f["category"], f["symbol"] = market, symbol
        for name in PRICE_COLUMNS:
            f[name] = part[name].to_numpy()
        f["last_price"] = close.to_numpy()
        f["prev_price24h"] = close.shift(1440).to_numpy()
        f["price24h_pcnt"] = (close / close.shift(1440) - 1).to_numpy()
        f["prev_price1h"] = close.shift(60).to_numpy()
        for output, column, operation in (
            ("high_price24h", "high", "max"),
            ("low_price24h", "low", "min"),
            ("volume24h", "volume", "sum"),
            ("turnover24h", "turnover", "sum"),
        ):
            f[output] = getattr(
                part[column].rolling(1440, min_periods=1440), operation
            )().to_numpy()
        f["size_reference"] = (
            part.volume.rolling(60, min_periods=60).mean().clip(lower=1e-12).to_numpy()
        )
        for horizon in (1, 5, 15, 60, 1440):
            f[f"f_return_{horizon}"] = (close / close.shift(horizon) - 1).to_numpy()
        returns = close.pct_change(fill_method=None)
        for horizon in (5, 15, 60):
            f[f"f_vol_{horizon}"] = returns.rolling(horizon, min_periods=horizon).std().to_numpy()
        f["f_range"] = ((part.high - part.low) / close).to_numpy()
        f["f_body"] = ((close - part.open) / close).to_numpy()
        width = (part.high - part.low).replace(0, np.nan)
        f["f_close_position"] = ((close - part.low) / width).fillna(0.5).to_numpy()
        for name in ("volume", "turnover"):
            reference = part[name].rolling(60, min_periods=60).mean().clip(lower=1e-12)
            f[f"f_log_{name}_ratio"] = np.log1p(part[name] / reference).to_numpy()
        tick, quantity, current = metadata.values(market, symbol, f.index)
        f["tick_size"], f["quantity_step"], f["current_metadata"] = tick, quantity, current
        f["f_relative_tick"] = tick / close.to_numpy()
        contiguous = part.close.notna().rolling(1441, min_periods=1441).sum().eq(1441)
        f["context_valid"] = contiguous.to_numpy()
        f = f.loc[(f.index >= start) & (f.index < end) & (f.index <= pd.Timestamp.now(tz="UTC"))]
        stats["streams"][f"{market}:{symbol}"] = {
            "requested_minutes": len(f),
            "complete_context_minutes": int(f.context_valid.sum()),
        }
        frames.append(f.reset_index())
    if stats["missing_streams"]:
        raise RetickerError(
            "Missing candle streams: "
            + ", ".join(stats["missing_streams"])
            + "; load candles externally"
        )
    if not frames:
        raise RetickerError("No candle data")
    features = pd.concat(frames, ignore_index=True)
    features["index_reference"] = features.close
    if len(selection.markets) == 2:
        other_names = [
            "f_return_1",
            "f_return_5",
            "f_return_60",
            "f_vol_60",
            "f_range",
            "f_log_volume_ratio",
        ]
        peer = features[
            ["time", "category", "symbol", "close", "context_valid", *other_names]
        ].copy()
        peer["category"] = peer.category.map({"spot": "linear", "linear": "spot"})
        peer = peer.rename(
            columns={
                "close": "peer_close",
                "context_valid": "peer_context_valid",
                **{name: "f_peer_" + name[2:] for name in other_names},
            }
        )
        features = features.merge(
            peer, on=["time", "category", "symbol"], how="left", validate="one_to_one"
        )
        linear = features.category == "linear"
        features["f_basis"] = np.where(
            linear,
            features.close / features.peer_close - 1,
            features.peer_close / features.close - 1,
        )
        features.loc[linear, "index_reference"] = features.loc[linear, "peer_close"]
        features["context_valid"] &= features.peer_context_valid.fillna(False).astype(bool)
    names = sorted(c for c in features if c.startswith("f_"))
    features["context_valid"] &= np.isfinite(features[names].to_numpy()).all(axis=1)
    return features.sort_values(["time", "category", "symbol"]).reset_index(drop=True), stats


def targets(frame):
    mid = (frame.bid1_price + frame.ask1_price) / 2
    output = pd.DataFrame(index=frame.index)
    output["mid_offset"] = mid / frame.close - 1
    output["log_spread"] = np.log((frame.ask1_price - frame.bid1_price) / frame.close)
    output["log_bid_size"] = np.log(frame.bid1_size / frame.size_reference)
    output["log_ask_size"] = np.log(frame.ask1_size / frame.size_reference)
    if "mark_price" in frame:
        linear = frame.category == "linear"
        output["log_mark_ratio"] = np.nan
        output["log_index_ratio"] = np.nan
        output.loc[linear, "log_mark_ratio"] = np.log(
            frame.loc[linear, "mark_price"] / frame.loc[linear, "close"]
        )
        output.loc[linear, "log_index_ratio"] = np.log(
            frame.loc[linear, "index_price"] / frame.loc[linear, "index_reference"]
        )
    return output


def align_labels(features, tickers, selection, tolerance):
    tickers = tickers.copy()
    required = {
        "time",
        "category",
        "symbol",
        "bid1_price",
        "ask1_price",
        "bid1_size",
        "ask1_size",
        "last_price",
    }
    if "linear" in selection.markets:
        required |= {"mark_price", "index_price"}
    if not required <= set(tickers):
        raise RetickerError(f"Missing ticker columns: {sorted(required - set(tickers))}")
    tickers["time"] = pd.to_datetime(tickers.time, utc=True)
    tickers, duplicates = deduplicate(tickers, ["category", "symbol", "time"])
    positive = ["bid1_price", "ask1_price", "bid1_size", "ask1_size", "last_price"]
    valid = np.isfinite(tickers[positive].to_numpy(dtype=float)).all(axis=1)
    valid &= (tickers[positive] > 0).all(axis=1).to_numpy()
    valid &= (tickers.ask1_price > tickers.bid1_price).to_numpy()
    linear = tickers.category == "linear"
    for name in ("mark_price", "index_price"):
        if name in tickers:
            valid &= (~linear | (np.isfinite(tickers[name]) & (tickers[name] > 0))).to_numpy()
    stats = {"duplicate_tickers": duplicates, "invalid_tickers": int((~valid).sum())}
    tickers = tickers.loc[valid]
    parts = []
    for market, symbol in selection.keys:
        left = features.loc[
            (features.category == market) & (features.symbol == symbol) & features.context_valid
        ].copy()
        right = tickers.loc[(tickers.category == market) & (tickers.symbol == symbol)].copy()
        # Derived ticker values come from candles; last_price here remains a source diagnostic.
        keep = ["time", "bid1_price", "ask1_price", "bid1_size", "ask1_size", "last_price"]
        if market == "linear":
            keep += ["mark_price", "index_price"]
        right = right[keep].rename(
            columns={"time": "observed_time", "last_price": "observed_last_price"}
        )
        if right.empty or left.empty:
            continue
        left = left.sort_values("time")
        right = right.sort_values("observed_time")
        matched = pd.merge_asof(
            left,
            right,
            left_on="time",
            right_on="observed_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance),
        )
        matched = matched.loc[matched.observed_time.notna()].copy()
        matched["label_offset_seconds"] = (matched.observed_time - matched.time).dt.total_seconds()
        parts.append(matched)
    if not parts:
        raise RetickerError("No usable labels within the snapshot tolerance and candle context")
    data = pd.concat(parts, ignore_index=True)
    complete_times = data.groupby("time").size().eq(len(selection.keys))
    data = data.loc[data.time.isin(complete_times.index[complete_times])].copy()
    if data.empty:
        raise RetickerError("No complete joint panel across selected symbols and markets")
    for name, values in targets(data).items():
        data["y_" + name] = values
    stats.update(
        {
            "matched_rows_before_joint_filter": sum(len(p) for p in parts),
            "joint_rows": len(data),
            "joint_minutes": int(data.time.nunique()),
            "offset_seconds_quantiles": data.label_offset_seconds.quantile(
                [0, 0.05, 0.5, 0.95, 1]
            ).to_dict(),
            "last_close_error_bps_quantiles": ((data.observed_last_price / data.close - 1) * 1e4)
            .quantile([0.05, 0.5, 0.95])
            .to_dict(),
        }
    )
    return data.sort_values(["time", "category", "symbol"]).reset_index(drop=True), stats


def chronological_splits(data):
    times = pd.DatetimeIndex(sorted(data.time.unique()))
    if len(times) < 480:
        raise RetickerError(
            "At least 480 complete joint minutes are required for a preliminary experiment"
        )
    start, end = times[0], times[-1] + pd.Timedelta(minutes=1)
    weekly = end - start >= pd.Timedelta(days=28)
    if weekly:
        boundaries = [
            start,
            end - pd.Timedelta(days=21),
            end - pd.Timedelta(days=14),
            end - pd.Timedelta(days=7),
            end,
        ]
    else:
        boundaries = [
            start,
            times[int(len(times) * 0.5)],
            times[int(len(times) * 0.7)],
            times[int(len(times) * 0.85)],
            end,
        ]
    splits = {}
    for name, a, b in zip(
        ("train", "validation", "calibration", "test"), boundaries, boundaries[1:]
    ):
        count = int(((times >= a) & (times < b)).sum())
        if count < (120 if name == "train" else 30):
            raise RetickerError(f"Too few complete joint minutes in {name}: {count}")
        splits[name] = {"start": str(a), "end": str(b), "minutes": count}
    return splits, "weekly" if weekly else "preliminary"


@dataclass
class PreparedDataset:
    path: Path
    manifest: dict

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = read_json(path / "manifest.json")
        if manifest.get("format_version") != 1 or manifest.get("status") != "complete":
            raise RetickerError("Unsupported or incomplete prepared dataset")
        if digest(path / "panel.parquet") != manifest["data_sha256"]:
            raise RetickerError("Prepared dataset checksum mismatch")
        return cls(path, manifest)

    @property
    def selection(self):
        return Selection(**self.manifest["selection"])

    def frame(self, split=None):
        data = pd.read_parquet(self.path / "panel.parquet")
        if split:
            bounds = self.manifest["splits"][split]
            data = data.loc[(data.time >= utc(bounds["start"])) & (data.time < utc(bounds["end"]))]
        return data.reset_index(drop=True)


def prepare(source, config: PreparationConfig, output_dir, metadata=None):
    start, end = minute_range(config.start, config.end)
    if end > pd.Timestamp.now(tz="UTC").floor("min"):
        raise RetickerError("Preparation end must not include unclosed candles")
    tickers = source.tickers(
        config.selection,
        start - pd.Timedelta(seconds=config.tolerance_seconds),
        end + pd.Timedelta(seconds=config.tolerance_seconds),
    )
    if not {"time", "category", "symbol"} <= set(tickers):
        raise RetickerError("Ticker source must provide time, category, and symbol")
    tickers = tickers.copy()
    tickers["time"] = pd.to_datetime(tickers.time, utc=True)
    # Avoid loading years of candles when labels only cover a short overlap.
    tolerance = pd.Timedelta(seconds=config.tolerance_seconds)
    for market, symbol in config.selection.keys:
        observed = tickers.loc[(tickers.category == market) & (tickers.symbol == symbol), "time"]
        if observed.empty:
            raise RetickerError(f"No ticker labels for {market}:{symbol} in the requested period")
        start = max(start, (observed.min() - tolerance).ceil("min"))
        end = min(end, (observed.max() + tolerance).floor("min") + pd.Timedelta(minutes=1))
    if start >= end:
        raise RetickerError("Selected ticker streams have no common time overlap")
    metadata = metadata or MetadataCatalog.fetch(config.selection)
    candles = source.candles(config.selection, start - pd.Timedelta(minutes=1441), end)
    features, candle_stats = candle_features(candles, config.selection, metadata, start, end)
    panel, label_stats = align_labels(features, tickers, config.selection, config.tolerance_seconds)
    splits, status = chronological_splits(panel)
    path = new_directory(output_dir)
    panel.to_parquet(path / "panel.parquet", index=False, compression="zstd")
    manifest = {
        "format_version": 1,
        "status": "complete",
        "assessment": status,
        "created_at": str(pd.Timestamp.now(tz="UTC")),
        "config": asdict(config),
        "effective_period": {"start": str(start), "end": str(end)},
        "selection": asdict(config.selection),
        "splits": splits,
        "features": sorted(c for c in panel if c.startswith("f_")),
        "metadata": metadata.rules,
        "data_sha256": digest(path / "panel.parquet"),
        "candle_quality": candle_stats,
        "label_quality": label_stats,
    }
    write_json(path / "manifest.json", manifest)
    return PreparedDataset(path, manifest)
