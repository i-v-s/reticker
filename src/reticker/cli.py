"""Command-line entry points. All commands call the same public library API."""

import argparse
import json
import sys

from .config import (
    EvaluationConfig,
    GenerationConfig,
    PreparationConfig,
    RetickerError,
    Selection,
    TrainingConfig,
)
from .dataset import prepare
from .evaluation import evaluate
from .generation import write_generation
from .io import ClickHouseSource, json_safe
from .metadata import MetadataCatalog
from .model import ModelBundle, train


def parser():
    root = argparse.ArgumentParser(
        prog="reticker", description="Joint stochastic minute tickers from candles"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--symbols", nargs="+", required=True)
        command.add_argument("--markets", nargs="+", default=["spot", "linear"])
        command.add_argument("--env-file", default=".env")
        if name == "prepare":
            command.add_argument("--start", required=True)
            command.add_argument("--end", required=True)
            command.add_argument(
                "--metadata", help="JSON instrument rules; overrides current Bybit metadata"
            )
            command.add_argument("--tolerance-seconds", type=int, default=5)
            command.add_argument("--out", required=True)
    command = commands.add_parser("train")
    command.add_argument("--dataset", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--iterations", type=int, default=500)
    command.add_argument("--depth", type=int, default=6)
    command.add_argument("--learning-rate", type=float, default=0.05)
    command.add_argument("--threads", type=int, default=-1)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--block-lengths", nargs="+", type=int, default=[1, 5, 15, 60])
    command = commands.add_parser("evaluate")
    command.add_argument("--dataset", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    command = commands.add_parser("generate")
    command.add_argument("--model", required=True)
    command.add_argument("--symbols", nargs="+")
    command.add_argument("--markets", nargs="+", help="Must match the model's trained markets")
    command.add_argument("--env-file", default=".env")
    command.add_argument("--start", required=True)
    command.add_argument("--end", required=True)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--chunk-days", type=int, default=1)
    command.add_argument(
        "--layout",
        choices=("single", "partitioned"),
        default="single",
        help="Parquet layout: one file (default) or market/date partitions",
    )
    command.add_argument("--out", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in {"inspect", "prepare"}:
            selection = Selection(tuple(args.symbols), tuple(args.markets))
            source = ClickHouseSource.from_env(args.env_file)
            if args.command == "inspect":
                result = source.coverage(selection)
                print(result.to_string(index=False))
                missing = [
                    f"{m}:{s}"
                    for m, s in selection.keys
                    if not (
                        (result.source == "kline") & (result.category == m) & (result.symbol == s)
                    ).any()
                ]
                if missing:
                    print("Missing candle streams (load externally): " + ", ".join(missing))
            else:
                config = PreparationConfig(selection, args.start, args.end, args.tolerance_seconds)
                metadata = MetadataCatalog.fetch(selection, args.metadata)
                result = prepare(source, config, args.out, metadata)
                print(
                    json.dumps(
                        json_safe(
                            {
                                "dataset": str(result.path),
                                "assessment": result.manifest["assessment"],
                                "quality": result.manifest["label_quality"],
                            }
                        ),
                        indent=2,
                    )
                )
        elif args.command == "train":
            config = TrainingConfig(
                iterations=args.iterations,
                depth=args.depth,
                learning_rate=args.learning_rate,
                threads=args.threads,
                seed=args.seed,
                block_lengths=tuple(args.block_lengths),
            )
            result = train(args.dataset, args.out, config)
            print(
                json.dumps(
                    {
                        "model": str(result.path),
                        "model_id": result.manifest["model_id"],
                        "block_length": result.manifest["block_length"],
                    },
                    indent=2,
                )
            )
        elif args.command == "evaluate":
            result = evaluate(
                args.dataset, args.model, args.out, EvaluationConfig(tuple(args.seeds))
            )
            print(
                json.dumps(
                    json_safe(
                        {
                            "report": str(result.path / "report.md"),
                            "assessment": result.metrics["assessment"],
                            "scores": {m: r["score"] for m, r in result.metrics["methods"].items()},
                        }
                    ),
                    indent=2,
                )
            )
        elif args.command == "generate":
            bundle = ModelBundle.load(args.model)
            if args.markets and set(args.markets) != set(bundle.selection.markets):
                raise RetickerError(
                    "--markets must match the model; train a separate artifact for another market set"
                )
            config = GenerationConfig(
                args.start,
                args.end,
                tuple(args.symbols) if args.symbols else None,
                args.seed,
                args.chunk_days,
            )
            result = write_generation(
                ClickHouseSource.from_env(args.env_file),
                bundle,
                config,
                args.out,
                layout=args.layout,
            )
            print(
                json.dumps(
                    {"output": args.out, "rows": result["rows"], "status": result["status"]},
                    indent=2,
                )
            )
    except (RetickerError, FileNotFoundError) as exc:
        print(f"reticker: {exc}", file=sys.stderr)
        return 2
    return 0
