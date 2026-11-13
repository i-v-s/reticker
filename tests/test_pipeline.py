import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from reticker import (
    EvaluationConfig,
    GenerationConfig,
    ModelBundle,
    PreparedDataset,
    RetickerError,
    evaluate,
    generate,
    write_generation,
)
from reticker.cli import main
from reticker.config import utc
from reticker.generation import Simulation, reconstruct
from reticker.io import json_safe
from reticker.model import ResidualBank
from reticker.statistics import acf, correlations, panel_matrix


def test_training_bank_is_held_out_and_artifact_roundtrip(fitted):
    _, dataset, bundle = fitted
    loaded = ModelBundle.load(bundle.path)
    data = dataset.frame("test")
    a = Simulation(bundle, dataset.selection, 9).run(data)
    b = Simulation(loaded, dataset.selection, 9).run(data)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    bounds = dataset.manifest["splits"]["calibration"]
    assert loaded.bank.times.min() >= utc(bounds["start"]).timestamp()
    assert loaded.bank.times.max() < utc(bounds["end"]).timestamp()
    assert loaded.bank.times.max() < utc(dataset.manifest["splits"]["test"]["start"]).timestamp()
    assert loaded.manifest["assessment"] == "preliminary"


def test_reproducibility_and_chunk_invariance(fitted, market_data):
    source, _, bundle = fitted
    start = market_data[3]
    end = start + pd.Timedelta(minutes=2940)

    def run(days, seed):
        frames = list(
            generate(
                source, bundle, GenerationConfig(str(start), str(end), seed=seed, chunk_days=days)
            )
        )
        return pd.concat(frames, ignore_index=True)

    a = run(1, 15)
    b = run(3, 15)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    c = run(3, 16)
    assert not np.array_equal(a.bid1_size, c.bid1_size)
    assert (a.bid1_price > 0).all() and (a.ask1_price > a.bid1_price).all()
    assert (a[["bid1_size", "ask1_size"]] > 0).all().all()
    for symbol, part in a.groupby("symbol"):
        tick = 0.01 if symbol == "AAAUSDT" else 0.001
        np.testing.assert_allclose(
            part.bid1_price / tick, np.round(part.bid1_price / tick), atol=1e-8
        )
        np.testing.assert_allclose(
            part.ask1_price / tick, np.round(part.ask1_price / tick), atol=1e-8
        )


def test_joint_blocks_preserve_persistence_and_cross_asset_dependence(fitted):
    _, dataset, bundle = fitted
    # A fixed validation-independent length isolates the bootstrap's dependence behavior.
    frame = dataset.frame("test")
    joint = Simulation(bundle, dataset.selection, 273, "joint", 15).run(frame)
    independent = Simulation(bundle, dataset.selection, 273, "independent").run(frame)
    times, names, _ = panel_matrix(frame, dataset.selection)
    sizes = [i for i, name in enumerate(names) if name.endswith("log_bid_size")]
    j = joint.attrs["target_panel"][:, sizes]
    i = independent.attrs["target_panel"][:, sizes]
    assert acf(j, times)[0].mean() > acf(i, times)[0].mean() + 0.25
    off_diagonal = ~np.eye(len(sizes), dtype=bool)
    assert correlations(j)[off_diagonal].mean() > correlations(i)[off_diagonal].mean() + 0.25


@pytest.mark.parametrize("layout", [None, "single", "partitioned"])
def test_subset_and_parquet_output(fitted, market_data, tmp_path, layout):
    source, _, bundle = fitted
    start = market_data[3]
    config = GenerationConfig(str(start), str(start + pd.Timedelta(minutes=7)), ("AAAUSDT",), 5)
    options = {} if layout is None else {"layout": layout}
    result = write_generation(source, bundle, config, tmp_path / "output", **options)
    assert result["status"] == "complete" and result["rows"] == 14
    assert result["layout"] == (layout or "single")
    files = list((tmp_path / "output").rglob("*.parquet"))
    expected_files = (
        {f"market={market}/date=2025-01-02/part-000000.parquet" for market in ("spot", "linear")}
        if layout == "partitioned"
        else {"tickers.parquet"}
    )
    assert {str(path.relative_to(tmp_path / "output")) for path in files} == expected_files
    assert set(result["chunks"][0]["files"]) == expected_files
    data = pd.concat([pd.read_parquet(path) for path in files])
    assert set(data.symbol) == {"AAAUSDT"}
    assert data.time.dt.tz is not None
    assert data.attrs == {}
    assert data.loc[data.category == "spot", "prev_price1h"].isna().all()
    assert data.loc[data.category == "spot", ["mark_price", "index_price"]].isna().all().all()
    assert json.loads((tmp_path / "output/manifest.json").read_text())["config"]["seed"] == 5
    with pytest.raises(RetickerError, match="trained subset"):
        list(
            generate(
                source,
                bundle,
                GenerationConfig(
                    str(start), str(start + pd.Timedelta(minutes=1)), ("UNKNOWNUSDT",)
                ),
            )
        )


@pytest.mark.parametrize("layout", ["single", "partitioned"])
def test_parquet_output_spans_chunks_and_dates(fitted, market_data, tmp_path, layout):
    source, _, bundle = fitted
    start = market_data[3]
    config = GenerationConfig(str(start), str(start + pd.Timedelta(minutes=1450)), ("AAAUSDT",), 5)
    frames = list(generate(source, bundle, config))
    expected = pd.concat(frames, ignore_index=True)
    expected.attrs = {}
    output = tmp_path / "output"
    result = write_generation(source, bundle, config, output, layout=layout)
    assert result["status"] == "complete" and result["rows"] == 2900
    assert len(result["chunks"]) == 2
    for chunk, frame in zip(result["chunks"], frames, strict=True):
        for key in ("source_sha256", "diagnostics", "candle_quality"):
            assert chunk[key] == frame.attrs[key]
    files = list(output.rglob("*.parquet"))
    if layout == "single":
        assert files == [output / "tickers.parquet"]
        assert all(chunk["files"] == ["tickers.parquet"] for chunk in result["chunks"])
        parquet = pq.ParquetFile(files[0])
        assert parquet.metadata.num_row_groups == 2
        assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"
    else:
        assert len(files) == 6
        assert (output / "market=spot/date=2025-01-03/part-000000.parquet").is_file()
        assert (output / "market=spot/date=2025-01-03/part-000001.parquet").is_file()
    # Read file contents without adding Hive partition columns from their parent paths.
    actual = pd.concat(
        [pq.ParquetFile(path).read().to_pandas() for path in files], ignore_index=True
    )
    actual.attrs = {}
    keys = ["time", "category", "symbol"]
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    assert json.loads((output / "manifest.json").read_text()) == json_safe(result)


@pytest.mark.parametrize("layout", ["single", "partitioned"])
def test_generation_failure_keeps_written_chunk(fitted, market_data, tmp_path, monkeypatch, layout):
    source, _, bundle = fitted
    start = market_data[3]
    config = GenerationConfig(str(start), str(start + pd.Timedelta(minutes=7)), ("AAAUSDT",), 5)
    frame = next(generate(source, bundle, config))
    output = tmp_path / "output"

    def interrupted(*args):
        yield frame
        running = json.loads((output / "manifest.json").read_text())
        assert running["status"] == "running" and running["rows"] == len(frame)
        assert list(output.rglob("*.parquet"))
        raise RetickerError("Generation interrupted")

    monkeypatch.setattr("reticker.generation.generate", interrupted)
    with pytest.raises(RetickerError, match="Generation interrupted"):
        write_generation(source, bundle, config, output, layout=layout)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"] == "Generation interrupted"
    assert manifest["rows"] == len(frame) and len(manifest["chunks"]) == 1
    data = pd.concat(
        [pq.ParquetFile(path).read().to_pandas() for path in output.rglob("*.parquet")]
    )
    pd.testing.assert_frame_equal(
        data.sort_values(["time", "category", "symbol"]).reset_index(drop=True),
        frame.reset_index(drop=True),
        check_exact=True,
    )


def test_parquet_close_failure_marks_run_failed(fitted, market_data, tmp_path, monkeypatch):
    source, _, bundle = fitted
    start = market_data[3]
    config = GenerationConfig(str(start), str(start + pd.Timedelta(minutes=7)))
    output = tmp_path / "output"
    close = pq.ParquetWriter.close

    def failed_close(writer):
        assert json.loads((output / "manifest.json").read_text())["status"] == "running"
        close(writer)
        raise OSError("Cannot finalize Parquet")

    monkeypatch.setattr(pq.ParquetWriter, "close", failed_close)
    with pytest.raises(OSError, match="Cannot finalize Parquet"):
        write_generation(source, bundle, config, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"] == "OSError"


def test_invalid_generation_layout_does_not_create_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(RetickerError, match="layout"):
        write_generation(
            None, None, GenerationConfig("2025-01-01", "2025-01-02"), output, layout="daily"
        )
    assert not output.exists()


@pytest.mark.parametrize("layout", [None, "partitioned"])
def test_cli_generation_layout(fitted, market_data, tmp_path, monkeypatch, capsys, layout):
    source, _, bundle = fitted
    start = market_data[3]
    output = tmp_path / "output"
    monkeypatch.setattr("reticker.cli.ClickHouseSource.from_env", lambda env_file: source)
    args = [
        "generate",
        "--model",
        str(bundle.path),
        "--start",
        str(start),
        "--end",
        str(start + pd.Timedelta(minutes=7)),
        "--out",
        str(output),
    ]
    if layout is not None:
        args += ["--layout", layout]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["layout"] == (layout or "single")
    assert len(list(output.rglob("*.parquet"))) == (2 if layout == "partitioned" else 1)


def test_cli_rejects_unknown_generation_layout(tmp_path, capsys):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--model",
                "unused",
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-02",
                "--out",
                str(output),
                "--layout",
                "daily",
            ]
        )
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not output.exists()


def test_evaluation_reports_and_intervals(fitted, tmp_path):
    _, dataset, model = fitted
    report = evaluate(dataset, model, tmp_path / "report", EvaluationConfig((1, 2, 3)))
    assert report.metrics["assessment"] == "preliminary"
    assert report.metrics["ensemble_interval_samples"] == 3
    assert set(report.metrics["methods"]) == {"median", "independent", "joint"}
    assert all(0 <= value <= 1 for value in report.metrics["quantile_90_percent_coverage"].values())
    assert (report.path / "diagnostics.png").stat().st_size > 1000
    stored = json.loads((report.path / "report.json").read_text())
    assert len(stored["methods"]["joint"]["seeds"]) == 3
    assert "preliminary" in (report.path / "report.md").read_text()


def test_tampered_dataset_and_leaking_bank_are_rejected(fitted, tmp_path):
    _, dataset, model = fitted
    shutil.copytree(dataset.path, tmp_path / "dataset")
    with (tmp_path / "dataset/panel.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RetickerError, match="checksum mismatch"):
        PreparedDataset.load(tmp_path / "dataset")
    old = model.bank.times.copy()
    try:
        model.bank.times[0] = int(utc(dataset.manifest["splits"]["test"]["start"]).timestamp())
        with pytest.raises(RetickerError, match="outside the calibration"):
            evaluate(dataset, model, tmp_path / "invalid")
    finally:
        model.bank.times[:] = old


def test_bank_blocks_cannot_cross_gaps():
    bank = ResidualBank(
        np.array([0, 60, 120, 240, 300, 360]),
        np.ones((6, 1)),
        np.ones((6, 1)),
        ["target"],
        ["context"],
        np.zeros(1),
        np.ones(1),
    )
    np.testing.assert_equal(bank.candidates(3), [0, 3])
    assert not len(bank.candidates(4))


def test_one_tick_spread_is_not_doubled_by_rounding():
    frame = pd.DataFrame(
        {
            "category": ["spot"] * 3,
            "close": [100.0] * 3,
            "tick_size": [0.01] * 3,
            "quantity_step": [0.001] * 3,
            "size_reference": [10.0] * 3,
            "y_mid_offset": [0, 0.00001, -0.00001],
            "y_log_spread": [np.log(0.01 / 100)] * 3,
            "y_log_bid_size": [0.0] * 3,
            "y_log_ask_size": [0.0] * 3,
        }
    )
    result, _ = reconstruct(frame)
    np.testing.assert_allclose(result.ask1_price - result.bid1_price, 0.01, atol=1e-12)
    desired_mid = frame.close * (1 + frame.y_mid_offset)
    assert (
        np.abs((result.bid1_price + result.ask1_price) / 2 - desired_mid) <= 0.005 + 1e-12
    ).all()


def test_cli_validation(capsys):
    assert main(["inspect", "--symbols", "INVALID"]) == 2
    assert "USDT" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
