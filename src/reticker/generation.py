"""Stateful joint generation; chunk boundaries do not reset bootstrap or RNG state."""

import hashlib
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import GenerationConfig, RetickerError, Selection, minute_range, new_directory
from .dataset import DERIVED, candle_features, targets
from .io import write_json
from .model import ModelBundle, predict
from .statistics import context_matrix, panel_matrix


def reconstruct(frame):
    frame = frame.copy()
    corrections = np.zeros(len(frame), dtype=bool)
    for market in sorted(frame.category.unique()):
        mask = frame.category == market
        positions = np.flatnonzero(mask)
        part = frame.loc[mask]
        close = part.close.to_numpy()
        tick = part.tick_size.to_numpy()
        qty = part.quantity_step.to_numpy()
        raw_mid = close * (1 + part.y_mid_offset.to_numpy())
        mid = np.maximum(raw_mid, tick * 1.5)
        log_spread = part.y_log_spread.to_numpy()
        raw_width = close * np.exp(np.clip(log_spread, -50, 50))
        width = np.minimum(np.maximum(raw_width, tick), 2 * (mid - tick))
        corrections[positions] |= (
            (raw_mid != mid) | (raw_width != width) | (np.abs(log_spread) > 50)
        )
        # Quantize width and center together: rounding two continuous edges outward
        # independently almost always doubles an otherwise one-tick spread.
        width_units = np.maximum(1, np.floor(np.round(width / tick, 10) + 0.5))
        bid_units = np.maximum(1, np.floor(np.round(mid / tick - width_units / 2, 10) + 0.5))
        ask_units = bid_units + width_units
        frame.loc[mask, "bid1_price"] = bid_units * tick
        frame.loc[mask, "ask1_price"] = ask_units * tick
        for side in ("bid", "ask"):
            value = part["y_log_" + side + "_size"].to_numpy()
            raw = np.exp(np.clip(value, -50, 50)) * part.size_reference.to_numpy()
            frame.loc[mask, side + "1_size"] = np.maximum(1, np.rint(raw / qty)) * qty
            corrections[positions] |= np.abs(value) > 50
        if market == "linear":
            for name, reference in (("mark", "close"), ("index", "index_reference")):
                value = part["y_log_" + name + "_ratio"].to_numpy()
                frame.loc[mask, name + "_price"] = (
                    np.exp(np.clip(value, -50, 50)) * part[reference].to_numpy()
                )
                corrections[positions] |= np.abs(value) > 50
    return frame, int(corrections.sum())


class Simulation:
    """Internal engine shared by validation, evaluation and public streaming generation."""

    def __init__(self, bundle, selection, seed, method="joint", block_length=None):
        if method not in {"joint", "independent", "median"}:
            raise RetickerError("Unknown simulation method")
        if set(selection.markets) != set(bundle.selection.markets) or not set(
            selection.symbols
        ) <= set(bundle.selection.symbols):
            raise RetickerError(
                "Generation needs the artifact's markets and a trained subset of symbols"
            )
        if bundle.bank is None:
            raise RetickerError("Model has no residual bank")
        self.bundle, self.selection, self.method = bundle, selection, method
        self.length = block_length or bundle.manifest.get("block_length", 15)
        self.rng = np.random.default_rng(seed)
        self.position, self.remaining, self.last_time = None, 0, None

    def run(self, frame):
        frame = frame.sort_values(["time", "category", "symbol"]).reset_index(drop=True)
        predicted = predict(self.bundle, frame)
        times, names, medians = panel_matrix(predicted, self.selection, "p_")
        _, _, scales = panel_matrix(predicted, self.selection, "s_")
        _, context_names, contexts = context_matrix(predicted, self.selection)
        bank = self.bundle.bank
        columns = [bank.target_names.index(name) for name in names]
        noise = np.zeros_like(medians)
        seams = []
        for i, timestamp in enumerate(times):
            second = int(timestamp.timestamp())
            if self.last_time is not None and second <= self.last_time:
                raise RetickerError("Simulation chunks must be strictly increasing in time")
            if self.last_time is not None and second - self.last_time != 60:
                self.remaining = 0
            if self.method == "joint":
                if not self.remaining:
                    neighbors, weights = bank.neighbors(contexts[i], context_names, self.length)
                    self.position = int(self.rng.choice(neighbors, p=weights))
                    self.remaining = self.length
                    seams.append(str(timestamp))
                noise[i] = bank.residuals[self.position, columns]
                self.position += 1
                self.remaining -= 1
            elif self.method == "independent":
                neighbors, weights = bank.neighbors(contexts[i], context_names, 1)
                choices = self.rng.choice(neighbors, size=len(columns), p=weights)
                noise[i] = bank.residuals[choices, columns]
            self.last_time = second
        generated = medians + scales * noise
        working = frame.copy()
        for j, name in enumerate(names):
            market, symbol, target = name.split("|")
            mask = (working.category == market) & (working.symbol == symbol)
            working.loc[mask, "y_" + target] = generated[:, j]
        working, repairs = reconstruct(working)
        for name, values in targets(working).items():
            working["y_" + name] = values
        _, _, actual = panel_matrix(working, self.selection)
        out_of_range = np.zeros(len(working), dtype=bool)
        for market, symbol in self.selection.keys:
            mask = (working.category == market) & (working.symbol == symbol)
            limits = self.bundle.manifest["feature_ranges"][f"{market}:{symbol}"]
            for feature, (low, high) in limits.items():
                out_of_range |= (
                    mask & ((working[feature] < low) | (working[feature] > high))
                ).to_numpy()
        columns = [
            "time",
            "category",
            "symbol",
            *DERIVED,
            "bid1_price",
            "ask1_price",
            "bid1_size",
            "ask1_size",
        ]
        if "linear" in self.selection.markets:
            columns += ["mark_price", "index_price"]
        output = working[columns].copy()
        # prev_price1h is a futures-only field, rather than a claim about spot API compatibility.
        output.loc[output.category == "spot", "prev_price1h"] = np.nan
        output.attrs["target_panel"] = actual
        output.attrs["diagnostics"] = {
            "rows": len(output),
            "constraint_repairs": repairs,
            "quantile_crossings": predicted.attrs["quantile_crossings"],
            "rows_with_out_of_range_features": int(out_of_range.sum()),
            "current_metadata_rows": int(working.current_metadata.sum()),
            "block_starts": seams,
        }
        return output


def _combine_chunks(frames):
    if len(frames) == 1:
        return frames[0]
    attributes = [frame.attrs.copy() for frame in frames]
    for frame in frames:
        frame.attrs = {}
    result = pd.concat(frames, ignore_index=True)
    diagnostics = {
        key: sum(a["diagnostics"][key] for a in attributes)
        for key in attributes[0]["diagnostics"]
        if key != "block_starts"
    }
    diagnostics["block_starts"] = [t for a in attributes for t in a["diagnostics"]["block_starts"]]
    result.attrs = {
        "diagnostics": diagnostics,
        "source_sha256": hashlib.sha256(
            "".join(a["source_sha256"] for a in attributes).encode()
        ).hexdigest(),
        "candle_quality": {"segments": [a["candle_quality"] for a in attributes]},
    }
    return result


def generate(source, bundle, config: GenerationConfig):
    bundle = bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle)
    selection = Selection(config.symbols or bundle.selection.symbols, bundle.selection.markets)
    engine = Simulation(bundle, selection, config.seed)
    start, end = minute_range(config.start, config.end)
    if end > pd.Timestamp.now(tz="UTC").floor("min"):
        raise RetickerError("Generation end must not include unclosed candles")
    cursor, pending = start, []
    while cursor < end:
        # Fixed computation windows keep rolling numerical accumulators independent
        # of the caller's requested output chunk size.
        stop = min(cursor + pd.Timedelta(days=1), end)
        candles = source.candles(selection, cursor - pd.Timedelta(minutes=1441), stop)
        features, quality = candle_features(candles, selection, bundle.metadata, cursor, stop)
        expected = int((stop - cursor).total_seconds() / 60) * len(selection.keys)
        if len(features) != expected or not features.context_valid.all():
            invalid = features.loc[~features.context_valid, ["time", "category", "symbol"]]
            detail = invalid.iloc[0].to_dict() if len(invalid) else {"start": str(cursor)}
            raise RetickerError(
                f"Incomplete candle context in requested generation range: {detail}"
            )
        result = engine.run(features)
        result.attrs.pop("target_panel", None)
        result.attrs["source_sha256"] = hashlib.sha256(
            pd.util.hash_pandas_object(candles, index=False).to_numpy().tobytes()
        ).hexdigest()
        result.attrs["candle_quality"] = quality
        pending.append(result)
        cursor = stop
        if len(pending) == config.chunk_days or cursor == end:
            yield _combine_chunks(pending)
            pending = []


def write_generation(source, bundle, config, output_dir, *, layout="single"):
    """Stream Parquet to one file or market/date partitions, with a run manifest."""
    if layout not in {"single", "partitioned"}:
        raise RetickerError("Unknown generation layout; use single or partitioned")
    bundle = bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle)
    path = new_directory(output_dir)
    manifest = {
        "format_version": 1,
        "status": "running",
        "layout": layout,
        "model_id": bundle.manifest["model_id"],
        "config": asdict(config),
        "metadata": bundle.manifest["metadata"],
        "chunks": [],
        "assessment": bundle.manifest["assessment"],
        "rows": 0,
    }
    write_json(path / "manifest.json", manifest)
    try:
        with ExitStack() as stack:
            writer = None
            for index, frame in enumerate(generate(source, bundle, config)):
                files = []
                if layout == "single":
                    part = frame.copy(deep=False)
                    part.attrs = {}
                    table = pa.Table.from_pandas(
                        part,
                        schema=writer.schema if writer is not None else None,
                        preserve_index=False,
                    )
                    if writer is None:
                        writer = stack.enter_context(
                            pq.ParquetWriter(
                                path / "tickers.parquet", table.schema, compression="zstd"
                            )
                        )
                    writer.write_table(table)
                    files.append("tickers.parquet")
                else:
                    for (market, date), part in frame.groupby(
                        ["category", frame.time.dt.strftime("%Y-%m-%d")]
                    ):
                        directory = path / f"market={market}" / f"date={date}"
                        directory.mkdir(parents=True, exist_ok=True)
                        file = directory / f"part-{index:06d}.parquet"
                        # ndarray attrs are for in-process evaluation, not Parquet metadata.
                        part = part.copy()
                        part.attrs = {}
                        part.to_parquet(file, index=False, compression="zstd")
                        files.append(str(file.relative_to(path)))
                manifest["rows"] += len(frame)
                manifest["chunks"].append(
                    {
                        "files": files,
                        "source_sha256": frame.attrs["source_sha256"],
                        "diagnostics": frame.attrs["diagnostics"],
                        "candle_quality": frame.attrs["candle_quality"],
                    }
                )
                write_json(path / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed",
            error=str(exc) if isinstance(exc, RetickerError) else type(exc).__name__,
        )
        write_json(path / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    write_json(path / "manifest.json", manifest)
    return manifest
