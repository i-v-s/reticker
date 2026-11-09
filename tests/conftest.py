import numpy as np
import pandas as pd
import pytest

from reticker import (
    FrameSource,
    MetadataCatalog,
    PreparationConfig,
    Selection,
    TrainingConfig,
    prepare,
    train,
)


@pytest.fixture(scope="session")
def market_data():
    """Two related markets and assets with a persistent common liquidity shock."""
    rng = np.random.default_rng(1807)
    n = 1441 + 3100
    times = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    shock = np.zeros(n)
    innovations = rng.normal(size=n)
    for i in range(1, n):
        shock[i] = 0.94 * shock[i - 1] + 0.34 * innovations[i]
    common_returns = rng.normal(0, 0.0003, n)
    selection = Selection(("AAAUSDT", "BBBUSDT"))
    candles, tickers, rules = [], [], {}
    spot_close = {}
    for j, symbol in enumerate(selection.symbols):
        tick = 0.01 if j == 0 else 0.001
        base = (100 if j == 0 else 10) * np.exp(
            np.cumsum(common_returns + rng.normal(0, 0.0001, n))
        )
        spot_close[symbol] = np.round(base / tick) * tick
        for market in selection.markets:
            close = spot_close[symbol].copy()
            if market == "linear":
                close = (
                    np.round(close * (1 + 0.001 + 0.0002 * np.sin(np.arange(n) / 60)) / tick) * tick
                )
            opened = np.r_[close[0], close[:-1]]
            volume = np.exp(rng.normal(2, 0.5, n))
            frame = pd.DataFrame(
                {
                    "time": times,
                    "category": market,
                    "symbol": symbol,
                    "interval": "1",
                    "open": opened,
                    "high": np.maximum(opened, close) + tick * 3,
                    "low": np.minimum(opened, close) - tick * 3,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * close,
                }
            )
            candles.append(frame)
            spread_ticks = 1 + np.floor(np.exp(0.5 + 0.6 * shock)).astype(int)
            mid = close * (1 + 0.0001 * shock + rng.normal(0, 0.00001, n))
            bid = np.floor((mid - spread_ticks * tick / 2) / tick) * tick
            size_reference = pd.Series(volume).rolling(60, min_periods=1).mean().to_numpy()
            bid_size = np.maximum(
                0.001, np.round(size_reference * np.exp(-0.6 * shock + rng.normal(0, 0.03, n)), 3)
            )
            ask_size = np.maximum(
                0.001, np.round(size_reference * np.exp(-0.5 * shock + rng.normal(0, 0.03, n)), 3)
            )
            tickers.append(
                pd.DataFrame(
                    {
                        "time": times + pd.Timedelta(seconds=62),
                        "category": market,
                        "symbol": symbol,
                        "bid1_price": bid,
                        "ask1_price": bid + spread_ticks * tick,
                        "bid1_size": bid_size,
                        "ask1_size": ask_size,
                        "last_price": close,
                        "mark_price": close * np.exp(0.00005 * shock),
                        "index_price": spot_close[symbol] * np.exp(0.00004 * shock),
                    }
                )
            )
            rules[f"{market}:{symbol}"] = [
                {"tick_size": str(tick), "quantity_step": ".001", "source": "test"}
            ]
    source = FrameSource(
        pd.concat(candles, ignore_index=True), pd.concat(tickers, ignore_index=True)
    )
    start = times[1441]
    return source, selection, MetadataCatalog(rules), start


@pytest.fixture(scope="session")
def fitted(tmp_path_factory, market_data):
    source, selection, metadata, start = market_data
    root = tmp_path_factory.mktemp("pipeline")
    dataset = prepare(
        source,
        PreparationConfig(selection, str(start), str(start + pd.Timedelta(minutes=800))),
        root / "dataset",
        metadata,
    )
    model = train(
        dataset,
        root / "model",
        TrainingConfig(iterations=15, depth=3, threads=2, tuning_seeds=(0,)),
    )
    return source, dataset, model
