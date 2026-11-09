"""Distributional diagnostics used for validation and held-out evaluation."""

import numpy as np
import pandas as pd

from .config import RetickerError
from .dataset import TARGETS

QUANTILES = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
LAGS = (1, 5, 15, 60)


def panel_matrix(frame, selection, prefix="y_"):
    times = pd.DatetimeIndex(sorted(frame.time.unique()))
    arrays, names = [], []
    for market, symbol in selection.keys:
        part = (
            frame.loc[(frame.category == market) & (frame.symbol == symbol)]
            .set_index("time")
            .reindex(times)
        )
        for target in TARGETS[market]:
            arrays.append(part[prefix + target].to_numpy(dtype=float))
            names.append(f"{market}|{symbol}|{target}")
    matrix = np.column_stack(arrays)
    if not np.isfinite(matrix).all():
        raise RetickerError("Joint panel has missing or non-finite target values")
    return times, names, matrix


def context_matrix(frame, selection):
    times = pd.DatetimeIndex(sorted(frame.time.unique()))
    arrays, names = [], []
    for market, symbol in selection.keys:
        part = (
            frame.loc[(frame.category == market) & (frame.symbol == symbol)]
            .set_index("time")
            .reindex(times)
        )
        fields = ["f_vol_60", "f_log_volume_ratio"]
        if market == "linear" and len(selection.markets) == 2:
            fields.append("f_basis")
        for name in fields:
            arrays.append(part[name].to_numpy(dtype=float))
            names.append(f"{market}|{symbol}|{name}")
    values = np.column_stack(arrays)
    if not np.isfinite(values).all():
        raise RetickerError("Incomplete joint candle context")
    return times, names, values


def correlations(values):
    centered = values - values.mean(axis=0)
    norms = np.sqrt(np.sum(centered**2, axis=0))
    standardized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-14)
    return standardized.T @ standardized


def acf(values, times):
    result = []
    seconds = pd.DatetimeIndex(times).as_unit("s").asi8
    for lag in LAGS:
        if len(values) <= lag:
            result.append(np.zeros(values.shape[1]))
            continue
        valid = seconds[lag:] - seconds[:-lag] == lag * 60
        a, b = values[:-lag][valid], values[lag:][valid]
        if len(a) < 2:
            result.append(np.zeros(values.shape[1]))
            continue
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
        result.append(
            np.divide(
                (a * b).sum(axis=0), denom, out=np.zeros(values.shape[1]), where=denom > 1e-20
            )
        )
    return np.asarray(result)


def distribution_comparison(real, generated, times):
    reference = np.quantile(real, QUANTILES, axis=0)
    actual = np.quantile(generated, QUANTILES, axis=0)
    scale = np.maximum(np.quantile(real, 0.75, axis=0) - np.quantile(real, 0.25, axis=0), 1e-8)
    qerror = np.abs(actual - reference) / scale
    acf_real, acf_generated = acf(real, times), acf(generated, times)
    corr_real, corr_generated = correlations(real), correlations(generated)
    off_diagonal = ~np.eye(real.shape[1], dtype=bool)
    components = [
        float(qerror.mean()),
        float(np.abs(acf_generated - acf_real).mean()),
        float(np.abs(corr_generated - corr_real)[off_diagonal].mean()),
    ]
    return {
        "score": float(np.mean(components)),
        "normalized_quantile_error": components[0],
        "acf_error": components[1],
        "correlation_error": components[2],
        "real_quantiles": reference,
        "generated_quantiles": actual,
        "real_acf": acf_real,
        "generated_acf": acf_generated,
        "real_correlations": corr_real,
        "generated_correlations": corr_generated,
    }


def run_lengths(mask, times):
    seconds = pd.DatetimeIndex(times).as_unit("s").asi8
    runs, current = [], 0
    for i, value in enumerate(mask):
        if i and seconds[i] - seconds[i - 1] != 60 and current:
            runs.append(current)
            current = 0
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return np.asarray(runs, dtype=float)
