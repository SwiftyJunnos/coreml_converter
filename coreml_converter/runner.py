import dataclasses
import logging
import os

import torch
import torch.nn as nn

from .const import RESULT_DIR
from .converter import Converter
from .loader import load_weight
from .models.models import Models

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class RuntimeConfig:
    device: torch.device
    weight_dir: str
    weight_ext: str = ".bin"


def _weight_path(runtime_config: RuntimeConfig, filename: str) -> str:
    return os.path.join(runtime_config.weight_dir, filename) + runtime_config.weight_ext


def _input_size(model: nn.Module) -> torch.Size:
    param = next(model.parameters())
    return torch.Size((1, param.size(-1)))


def run(runtime_config: RuntimeConfig) -> int:
    """Run Converter."""
    _LOGGER.info("Start running converter...")

    converter = Converter()

    for model in Models:
        _LOGGER.info(f"Start converting with model {model.value}")
        weight_path = _weight_path(runtime_config, model.value)
        weighted_model = load_weight(
            model=model.make_model(),
            weight_path=weight_path,
            device=runtime_config.device,
        )
        if weighted_model is None:
            _LOGGER.error("Failed to load weight")
            continue

        # Prepare addtional args for convert.
        input_size = _input_size(weighted_model)
        example_inputs = model.example_inputs(input_size)

        # Convert weight-summed model to MLProgram type.
        ml_model, ext = converter.convert(weighted_model, **example_inputs)
        if ml_model is None:
            _LOGGER.error(
                f"Failed to convert {model.value} to CoreML compatible model."
            )
            continue

        # Optimize for mobile environment.

        # Save converted model as ".mlpackage" file.
        result_path = os.path.join(RESULT_DIR, model.value.lower()) + ext
        _LOGGER.info(f"Saving result to {result_path}")
        ml_model.save(result_path)  # type: ignore

    return 1
