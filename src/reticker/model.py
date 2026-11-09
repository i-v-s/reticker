"""Quantile models and a serializable joint calibration residual bank."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import RetickerError, Selection, TrainingConfig, new_directory, utc
from .dataset import TARGETS, PreparedDataset
from .io import digest, json_safe, read_json, write_json
from .metadata import MetadataCatalog
from .statistics import context_matrix, distribution_comparison, panel_matrix


@dataclass
class ResidualBank:
    times: np.ndarray
    residuals: np.ndarray
    contexts: np.ndarray
    target_names: list[str]
    context_names: list[str]
    center: np.ndarray
    scale: np.ndarray

    def candidates(self, length):
        if length > len(self.times):
            return np.array([], dtype=int)
        bad = np.r_[0, np.cumsum(np.diff(self.times) != 60)]
        starts = np.arange(len(self.times) - length + 1)
        return starts[bad[starts + length - 1] == bad[starts]]

    def neighbors(self, context, names, length):
        columns = [self.context_names.index(name) for name in names]
        candidates = self.candidates(length)
        if not len(candidates):
            raise RetickerError(f"No continuous calibration block of {length} minutes")
        normalized = (context - self.center[columns]) / self.scale[columns]
        historic = (self.contexts[candidates][:, columns] - self.center[columns]) / self.scale[
            columns
        ]
        distance = np.sqrt(np.mean((historic - normalized) ** 2, axis=1))
        # Stable order makes fixed seeds reproducible even at equal distances.
        nearest = np.argsort(distance, kind="stable")[:20]
        weights = 1 / (1 + distance[nearest])
        return candidates[nearest], weights / weights.sum()

    def save(self, path):
        np.savez_compressed(
            path,
            times=self.times,
            residuals=self.residuals,
            contexts=self.contexts,
            target_names=np.array(self.target_names),
            context_names=np.array(self.context_names),
            center=self.center,
            scale=self.scale,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["times"],
                data["residuals"],
                data["contexts"],
                data["target_names"].tolist(),
                data["context_names"].tolist(),
                data["center"],
                data["scale"],
            )


@dataclass
class ModelBundle:
    path: Path | None
    manifest: dict
    models: dict
    bank: ResidualBank | None

    @property
    def selection(self):
        return Selection(**self.manifest["selection"])

    @property
    def metadata(self):
        return MetadataCatalog(self.manifest["metadata"])

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = read_json(path / "manifest.json")
        if (
            manifest.get("format_version") != 1
            or manifest.get("status") != "complete"
            or manifest.get("decoder") != "tick_width_mid_v1"
        ):
            raise RetickerError("Unsupported or incomplete model artifact")
        for filename, expected in manifest["checksums"].items():
            if digest(path / filename) != expected:
                raise RetickerError(f"Model artifact checksum mismatch: {filename}")
        models = {}
        for key, filename in manifest["models"].items():
            models[key] = CatBoostRegressor().load_model(str(path / filename))
        return cls(path, manifest, models, ResidualBank.load(path / "residuals.npz"))


def predict(bundle, frame):
    output = frame.copy()
    crossings = 0
    for market in bundle.selection.markets:
        mask = output.category == market
        if not mask.any():
            continue
        x = output.loc[mask, ["symbol", *bundle.manifest["features"]]]
        for target in TARGETS[market]:
            key = f"{market}:{target}"
            raw = np.asarray(bundle.models[key].predict(x))
            crossings += int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
            ordered = np.sort(raw, axis=1)
            output.loc[mask, "p_" + target] = ordered[:, 1]
            output.loc[mask, "s_" + target] = np.maximum(
                ordered[:, 2] - ordered[:, 0], bundle.manifest["scale_floors"][key]
            )
            output.loc[mask, "q05_" + target] = ordered[:, 0]
            output.loc[mask, "q95_" + target] = ordered[:, 2]
    output.attrs["quantile_crossings"] = crossings
    return output


def make_bank(bundle, frame, train_contexts=None):
    predicted = predict(bundle, frame)
    for market in bundle.selection.markets:
        mask = predicted.category == market
        for target in TARGETS[market]:
            predicted.loc[mask, "r_" + target] = (
                predicted.loc[mask, "y_" + target] - predicted.loc[mask, "p_" + target]
            ) / predicted.loc[mask, "s_" + target]
    times, target_names, residuals = panel_matrix(predicted, bundle.selection, "r_")
    _, context_names, contexts = context_matrix(predicted, bundle.selection)
    reference = train_contexts if train_contexts is not None else contexts
    center = np.median(reference, axis=0)
    scale = np.maximum(
        np.quantile(reference, 0.75, axis=0) - np.quantile(reference, 0.25, axis=0), 1e-12
    )
    return ResidualBank(
        times.as_unit("s").asi8, residuals, contexts, target_names, context_names, center, scale
    )


def train(dataset, output_dir, config=None):
    from .generation import Simulation

    config = config or TrainingConfig()
    dataset = dataset if isinstance(dataset, PreparedDataset) else PreparedDataset.load(dataset)
    if not isinstance(dataset, PreparedDataset):
        raise RetickerError("Expected a prepared dataset")
    path = new_directory(output_dir)
    train_frame, validation = dataset.frame("train"), dataset.frame("validation")
    manifest = {
        "format_version": 1,
        "status": "training",
        "selection": asdict(dataset.selection),
        "features": dataset.manifest["features"],
        "metadata": dataset.manifest["metadata"],
        "dataset_sha256": dataset.manifest["data_sha256"],
        "splits": dataset.manifest["splits"],
        "assessment": dataset.manifest["assessment"],
        "training": asdict(config),
        "quantiles": [0.05, 0.5, 0.95],
        "decoder": "tick_width_mid_v1",
        "scale_floors": {},
        "feature_ranges": {},
        "models": {},
        "versions": {
            name: version(name)
            for name in ("reticker", "catboost", "numpy", "pandas", "pyarrow", "clickhouse-driver")
        },
        "python": platform.python_version(),
        "created_at": str(pd.Timestamp.now(tz="UTC")),
    }
    write_json(path / "manifest.json", manifest)
    bundle = ModelBundle(path, manifest, {}, None)
    for market in dataset.selection.markets:
        a = train_frame.loc[train_frame.category == market]
        b = validation.loc[validation.category == market]
        features = ["symbol", *manifest["features"]]
        for target in TARGETS[market]:
            key = f"{market}:{target}"
            model = CatBoostRegressor(
                loss_function="MultiQuantile:alpha=0.05,0.5,0.95",
                iterations=config.iterations,
                depth=config.depth,
                learning_rate=config.learning_rate,
                random_seed=config.seed,
                thread_count=config.threads,
                verbose=False,
                allow_writing_files=False,
                task_type="CPU",
            )
            model.fit(
                a[features],
                a["y_" + target],
                cat_features=["symbol"],
                eval_set=(b[features], b["y_" + target]),
                early_stopping_rounds=50,
            )
            filename = f"{market}_{target}.cbm"
            model.save_model(str(path / filename))
            bundle.models[key] = model
            manifest["models"][key] = filename
            values = a["y_" + target].to_numpy()
            manifest["scale_floors"][key] = max(
                1e-8, float(np.subtract(*np.quantile(values, [0.75, 0.25]))) * 1e-3
            )
        for symbol, part in a.groupby("symbol"):
            manifest["feature_ranges"][f"{market}:{symbol}"] = {
                name: [float(part[name].min()), float(part[name].max())]
                for name in manifest["features"]
            }
    _, _, train_contexts = context_matrix(train_frame, dataset.selection)
    validation_times = sorted(validation.time.unique())
    midpoint = validation_times[len(validation_times) // 2]
    provisional = validation.loc[validation.time < midpoint]
    control = validation.loc[validation.time >= midpoint].reset_index(drop=True)
    bundle.bank = make_bank(bundle, provisional, train_contexts)
    times, _, real = panel_matrix(control, dataset.selection)
    tuning = []
    for length in sorted(set(config.block_lengths)):
        if not len(bundle.bank.candidates(length)):
            continue
        scores = []
        for seed in config.tuning_seeds:
            simulated = Simulation(bundle, dataset.selection, seed, "joint", length).run(control)
            generated_targets = simulated.attrs["target_panel"]
            scores.append(distribution_comparison(real, generated_targets, times)["score"])
        tuning.append({"length": length, "score": float(np.mean(scores))})
    if not tuning:
        raise RetickerError("Validation has no continuous blocks for the requested lengths")
    chosen = min(tuning, key=lambda item: (item["score"], item["length"]))["length"]
    bundle.bank = make_bank(bundle, dataset.frame("calibration"), train_contexts)
    if not len(bundle.bank.candidates(chosen)):
        raise RetickerError(
            f"Calibration has no continuous {chosen}-minute block; change the split or block candidates"
        )
    manifest["block_length"], manifest["block_tuning"] = chosen, tuning
    manifest["bank"] = {
        "minutes": len(bundle.bank.times),
        "first_time": str(utc(pd.Timestamp(bundle.bank.times[0], unit="s"))),
        "last_time": str(utc(pd.Timestamp(bundle.bank.times[-1], unit="s"))),
    }
    bundle.bank.save(path / "residuals.npz")
    manifest["checksums"] = {
        name: digest(path / name) for name in [*manifest["models"].values(), "residuals.npz"]
    }
    identity = {
        key: manifest[key]
        for key in (
            "checksums",
            "dataset_sha256",
            "metadata",
            "training",
            "block_length",
            "decoder",
        )
    }
    manifest["model_id"] = hashlib.sha256(
        json.dumps(json_safe(identity), sort_keys=True).encode()
    ).hexdigest()[:24]
    manifest["status"] = "complete"
    write_json(path / "manifest.json", manifest)
    return bundle
