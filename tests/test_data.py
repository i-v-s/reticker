import numpy as np
import pandas as pd
import pytest

from reticker import MetadataCatalog, PreparationConfig, RetickerError, Selection
from reticker.dataset import align_labels, candle_features, chronological_splits


def test_features_use_close_boundaries_and_full_day(market_data):
    source, selection, metadata, start = market_data
    end = start + pd.Timedelta(minutes=5)
    candles = source.candles(selection, start - pd.Timedelta(minutes=1441), end)
    features, _ = candle_features(candles, selection, metadata, start, end)
    row = features.loc[(features.category == "spot") & (features.symbol == "AAAUSDT")].iloc[0]
    history = candles.loc[(candles.category == "spot") & (candles.symbol == "AAAUSDT")].set_index(
        "time"
    )
    assert row.last_price == history.loc[start - pd.Timedelta(minutes=1), "close"]
    assert row.prev_price24h == history.loc[start - pd.Timedelta(minutes=1441), "close"]
    day = history.loc[(history.index >= start - pd.Timedelta(days=1)) & (history.index < start)]
    assert row.volume24h == pytest.approx(day.volume.sum())
    assert row.high_price24h == day.high.max()
    # A later candle must not change an earlier feature or derived ticker field.
    changed = candles.copy()
    changed.loc[changed.time >= start, ["open", "high", "low", "close", "volume", "turnover"]] *= 2
    later, _ = candle_features(changed, selection, metadata, start, end)
    pd.testing.assert_frame_equal(
        features.loc[features.time == start], later.loc[later.time == start]
    )


def test_gaps_invalidate_context_but_zero_volume_does_not(market_data):
    source, selection, metadata, start = market_data
    end = start + pd.Timedelta(minutes=5)
    candles = source.candles(selection, start - pd.Timedelta(minutes=1441), end)
    mask = (candles.category == "spot") & (candles.symbol == "AAAUSDT")
    zero = candles.copy()
    zero.loc[mask, ["volume", "turnover"]] = 0
    features, _ = candle_features(zero, selection, metadata, start, end)
    assert features.context_valid.all()
    broken = candles.loc[~(mask & (candles.time == start - pd.Timedelta(minutes=10)))]
    features, _ = candle_features(broken, selection, metadata, start, end)
    assert not features.loc[features.symbol == "AAAUSDT", "context_valid"].any()
    assert features.loc[features.symbol == "BBBUSDT", "context_valid"].all()


def test_alignment_prefers_earlier_tie_and_excludes_six_seconds(market_data):
    source, _, metadata, start = market_data
    selection = Selection(("AAAUSDT",), ("spot",))
    end = start + pd.Timedelta(minutes=3)
    candles = source.candles(selection, start - pd.Timedelta(minutes=1441), end)
    features, _ = candle_features(candles, selection, metadata, start, end)
    original = source.tickers(selection, start, end).iloc[0]
    records = []
    for offset in (-2, 2, 66, 122):
        row = original.copy()
        row["time"] = start + pd.Timedelta(seconds=offset)
        records.append(row)
    data, _ = align_labels(features, pd.DataFrame(records), selection, 5)
    assert list(data.label_offset_seconds) == [-2, 2]
    assert start + pd.Timedelta(minutes=1) not in set(data.time)


def test_metadata_history_and_overlap():
    catalog = MetadataCatalog(
        {
            "spot:AAAUSDT": [
                {"tick_size": ".1", "quantity_step": ".01", "source": "bybit_current"},
                {
                    "tick_size": ".01",
                    "quantity_step": ".001",
                    "valid_from": "2020-01-01",
                    "valid_to": "2025-01-01",
                },
            ]
        }
    )
    tick, qty, current = catalog.values(
        "spot", "AAAUSDT", pd.to_datetime(["2024-12-31", "2025-01-01"], utc=True)
    )
    np.testing.assert_equal(tick, [0.01, 0.1])
    np.testing.assert_equal(qty, [0.001, 0.01])
    np.testing.assert_equal(current, [False, True])
    with pytest.raises(RetickerError, match="Overlapping"):
        MetadataCatalog(
            {
                "spot:AAAUSDT": [
                    {
                        "tick_size": ".1",
                        "quantity_step": ".01",
                        "valid_from": "2020-01-01",
                        "valid_to": "2023-01-01",
                    },
                    {
                        "tick_size": ".1",
                        "quantity_step": ".01",
                        "valid_from": "2022-01-01",
                        "valid_to": "2024-01-01",
                    },
                ]
            }
        )


def test_weekly_splits_and_config_validation():
    data = pd.DataFrame(
        {"time": pd.date_range("2025-01-01", periods=35 * 1440, freq="min", tz="UTC")}
    )
    splits, status = chronological_splits(data)
    assert status == "weekly"
    assert splits["test"]["minutes"] == 7 * 1440
    assert splits["train"]["end"] == splits["validation"]["start"]
    with pytest.raises(RetickerError):
        PreparationConfig(Selection(("AAAUSDT",)), "2025-01-01T00:00:01Z", "2025-01-02")
    with pytest.raises(RetickerError):
        Selection(("BTCUSDT'; DROP TABLE kline",))


def test_missing_future_stream_has_actionable_error(market_data):
    source, selection, metadata, start = market_data
    end = start + pd.Timedelta(minutes=3)
    candles = source.candles(
        Selection(selection.symbols, ("spot",)), start - pd.Timedelta(minutes=1441), end
    )
    with pytest.raises(RetickerError, match="load candles externally"):
        candle_features(candles, selection, metadata, start, end)
