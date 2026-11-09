"""Held-out statistical evaluation, baselines, and portable reports."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EvaluationConfig, RetickerError, new_directory, utc
from .dataset import TARGETS, PreparedDataset
from .generation import Simulation
from .io import write_json
from .model import ModelBundle, predict
from .statistics import LAGS, QUANTILES, distribution_comparison, panel_matrix, run_lengths


@dataclass
class EvaluationReport:
    path: Path
    metrics: dict


def _mean_summaries(summaries):
    return {key: np.mean([s[key] for s in summaries], axis=0) for key in summaries[0]}


def _episodes(real, synthetic, names, times):
    result = {}
    for j, name in enumerate(names):
        if not name.endswith("|log_spread"):
            continue
        threshold = np.quantile(real[:, j], 0.95)
        entry = {"threshold_log_relative_spread": threshold}
        for label, values in (("real", real), ("generated", synthetic)):
            runs = run_lengths(values[:, j] > threshold, times)
            entry[label] = {
                "count": len(runs),
                "median_minutes": float(np.median(runs)) if len(runs) else 0,
                "p95_minutes": float(np.quantile(runs, 0.95)) if len(runs) else 0,
            }
        result[name] = entry
    return result


def _jumps(real, generated, times, seams):
    seconds = times.as_unit("s").asi8
    contiguous = np.diff(seconds) == 60
    scale = np.maximum(np.quantile(real, 0.75, axis=0) - np.quantile(real, 0.25, axis=0), 1e-8)
    actual = np.mean(np.abs(np.diff(generated, axis=0)) / scale, axis=1)
    reference = np.mean(np.abs(np.diff(real, axis=0)) / scale, axis=1)
    seam_mask = times[1:].isin(pd.DatetimeIndex([utc(t) for t in seams])) & contiguous
    return {
        "real_mean": float(reference[contiguous].mean()) if contiguous.any() else None,
        "generated_mean": float(actual[contiguous].mean()) if contiguous.any() else None,
        "seam_mean": float(actual[seam_mask].mean()) if seam_mask.any() else None,
        "seam_p95": float(np.quantile(actual[seam_mask], 0.95)) if seam_mask.any() else None,
        "seam_count": int(seam_mask.sum()),
    }


def _point_errors(test, median):
    merged = test.merge(
        median,
        on=["time", "category", "symbol"],
        suffixes=("_real", "_generated"),
        validate="one_to_one",
    )
    output = {}
    for (market, symbol), part in merged.groupby(["category", "symbol"]):
        close, tick = part.close.to_numpy(), part.tick_size.to_numpy()
        actual_mid = (part.bid1_price_real + part.ask1_price_real).to_numpy() / 2
        generated_mid = (part.bid1_price_generated + part.ask1_price_generated).to_numpy() / 2
        fields = {
            "mid": (actual_mid, generated_mid),
            "spread": (
                (part.ask1_price_real - part.bid1_price_real).to_numpy(),
                (part.ask1_price_generated - part.bid1_price_generated).to_numpy(),
            ),
        }
        if market == "linear":
            fields.update(
                {
                    name: (part[name + "_real"].to_numpy(), part[name + "_generated"].to_numpy())
                    for name in ("mark_price", "index_price")
                }
            )
        output[f"{market}:{symbol}"] = {
            name: {
                "mae_bps": float(np.mean(np.abs(a - b) / close * 1e4)),
                "mae_ticks": float(np.mean(np.abs(a - b) / tick)),
            }
            for name, (a, b) in fields.items()
        }
    return output


def _plots(path, metrics):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    methods = metrics["methods"]
    labels = list(methods)
    axes[0].bar(labels, [methods[m]["score"] for m in labels])
    axes[0].set(
        title="Held-out distribution score (lower is better)",
        ylabel="Equal-weight normalized error",
    )
    real = np.asarray(methods["joint"]["real_acf"])
    generated = np.asarray(methods["joint"]["generated_acf"])
    axes[1].plot(LAGS, real.mean(axis=1), "o-", label="Observed")
    axes[1].plot(LAGS, generated.mean(axis=1), "o-", label="Joint generator")
    axes[1].set(title="Mean target autocorrelation", xlabel="Lag, minutes")
    axes[1].legend()
    delta = np.asarray(methods["joint"]["generated_correlations"]) - np.asarray(
        methods["joint"]["real_correlations"]
    )
    im = axes[2].imshow(delta, vmin=-1, vmax=1, cmap="coolwarm")
    axes[2].set(title="Joint correlation error", xlabel="Target channel", ylabel="Target channel")
    fig.colorbar(im, ax=axes[2])
    fig.suptitle(f"Reticker — {metrics['assessment']} assessment")
    fig.savefig(path / "diagnostics.png", dpi=160)
    plt.close(fig)


def evaluate(dataset, bundle, output_dir, config=None):
    config = config or EvaluationConfig()
    dataset = dataset if isinstance(dataset, PreparedDataset) else PreparedDataset.load(dataset)
    bundle = bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle)
    if dataset.manifest["data_sha256"] != bundle.manifest["dataset_sha256"]:
        raise RetickerError("Evaluation dataset does not match the model's training lineage")
    if (
        dataset.manifest["splits"] != bundle.manifest["splits"]
        or dataset.selection != bundle.selection
    ):
        raise RetickerError("Evaluation selection or time splits do not match the model")
    calibration = bundle.manifest["splits"]["calibration"]
    lower, upper = (
        int(utc(calibration["start"]).timestamp()),
        int(utc(calibration["end"]).timestamp()),
    )
    if not np.all((bundle.bank.times >= lower) & (bundle.bank.times < upper)):
        raise RetickerError("Residual bank contains observations outside the calibration split")
    test = dataset.frame("test")
    times, names, real = panel_matrix(test, dataset.selection)
    path = new_directory(output_dir)
    metrics = {
        "format_version": 1,
        "assessment": dataset.manifest["assessment"],
        "model_id": bundle.manifest["model_id"],
        "config": asdict(config),
        "dataset_sha256": dataset.manifest["data_sha256"],
        "test": dataset.manifest["splits"]["test"],
        "target_names": names,
        "quantiles": QUANTILES,
        "acf_lags": LAGS,
        "label_quality": dataset.manifest["label_quality"],
        "candle_quality": dataset.manifest["candle_quality"],
        "methods": {},
        "score_definition": "Mean of IQR-normalized quantile error, absolute ACF error, and off-diagonal correlation error",
    }
    ensemble = []
    for method in ("median", "independent", "joint"):
        summaries, seed_metrics = [], []
        seeds = (config.seeds[0],) if method == "median" else config.seeds
        for seed in seeds:
            synthetic = Simulation(bundle, dataset.selection, seed, method).run(test)
            values = synthetic.attrs["target_panel"]
            summary = distribution_comparison(real, values, times)
            summaries.append(summary)
            seed_metrics.append(
                {
                    "seed": seed,
                    "score": summary["score"],
                    "diagnostics": synthetic.attrs["diagnostics"],
                    "jumps": _jumps(
                        real, values, times, synthetic.attrs["diagnostics"]["block_starts"]
                    ),
                    "wide_spread_episodes": _episodes(real, values, names, times),
                }
            )
            if method == "median":
                metrics["point_errors"] = _point_errors(test, synthetic)
            elif method == "joint":
                ensemble.append(values)
        metrics["methods"][method] = {**_mean_summaries(summaries), "seeds": seed_metrics}
    quantile_prediction = predict(bundle, test)
    coverage = {}
    for market in dataset.selection.markets:
        part = quantile_prediction.loc[quantile_prediction.category == market]
        for target in TARGETS[market]:
            covered = (part["y_" + target] >= part["q05_" + target]) & (
                part["y_" + target] <= part["q95_" + target]
            )
            coverage[f"{market}:{target}"] = float(covered.mean())
    ensemble_low, ensemble_high = np.quantile(np.stack(ensemble), [0.05, 0.95], axis=0)
    metrics["quantile_90_percent_coverage"] = coverage
    metrics["ensemble_90_percent_coverage"] = dict(
        zip(names, np.mean((real >= ensemble_low) & (real <= ensemble_high), axis=0))
    )
    metrics["ensemble_interval_samples"] = len(ensemble)
    metrics["limitations"] = [
        "Labels approximate a candle boundary using the nearest snapshot within the configured tolerance.",
        "Ensemble interval estimates are Monte Carlo estimates with the reported number of seeds.",
        "Current metadata fallback is not a historical instrument specification.",
        "Statistical diagnostics do not establish strategy profitability or historical transferability.",
    ]
    if metrics["assessment"] == "preliminary":
        metrics["limitations"].append(
            "Less than four weeks of joint data: repeat evaluation on held-out weeks after collection."
        )
    write_json(path / "report.json", metrics)
    _plots(path, metrics)
    lines = [
        "# Reticker evaluation",
        "",
        f"Assessment: **{metrics['assessment']}**",
        "",
        f"Model: `{metrics['model_id']}`",
        "",
        f"Held-out joint minutes: {len(times)}",
        "",
        "| Method | Distribution score ↓ | Quantile error ↓ | ACF error ↓ | Correlation error ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, summary in metrics["methods"].items():
        lines.append(
            f"| {method} | {summary['score']:.4g} | {summary['normalized_quantile_error']:.4g} | {summary['acf_error']:.4g} | {summary['correlation_error']:.4g} |"
        )
    lines += ["", "![Diagnostics](diagnostics.png)", "", "## Interpretation", ""]
    lines += [f"- {message}" for message in metrics["limitations"]]
    lines += [
        "",
        "Detailed per-field errors, quantiles, correlations, coverage, episodes and boundary diagnostics: [report.json](report.json).",
        "",
    ]
    (path / "report.md").write_text("\n".join(lines))
    return EvaluationReport(path, metrics)
