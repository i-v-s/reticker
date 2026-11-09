"""Minute ticker reconstruction with joint conditional residual bootstrap."""

__version__ = "0.1.0"

from .config import (
    EvaluationConfig,
    GenerationConfig,
    PreparationConfig,
    RetickerError,
    Selection,
    TrainingConfig,
)
from .dataset import PreparedDataset, prepare
from .evaluation import EvaluationReport, evaluate
from .generation import generate, write_generation
from .io import ClickHouseSource, FrameSource
from .metadata import MetadataCatalog
from .model import ModelBundle, train

__all__ = [
    "ClickHouseSource",
    "EvaluationConfig",
    "EvaluationReport",
    "FrameSource",
    "GenerationConfig",
    "MetadataCatalog",
    "ModelBundle",
    "PreparationConfig",
    "PreparedDataset",
    "RetickerError",
    "Selection",
    "TrainingConfig",
    "evaluate",
    "generate",
    "prepare",
    "train",
    "write_generation",
]


def main():
    from .cli import main as cli_main

    return cli_main()
