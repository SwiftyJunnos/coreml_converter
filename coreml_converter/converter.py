import logging
from typing import Optional

import coremltools as ct
import torch
import torch.jit as jit
import torch.nn as nn
from coremltools.converters.mil import Program
from coremltools.models.model import MLModel

from .const import _MLPACKAGE_EXTENSION

_LOGGER = logging.getLogger(__name__)


class Converter:
    def __init__(self):
        pass

    def convert(
        self, model: nn.Module, **example_inputs: torch.Tensor
    ) -> tuple[Optional[Program], str]:
        model.eval()

        example_values = tuple(example_inputs.values())
        _LOGGER.info(f"Tracing: {example_values}")
        traced_model = jit.trace(model, example_values)
        _LOGGER.info("Tracing complete!")

        _LOGGER.info("Start converting...")
        converted_model = ct.convert(
            model=traced_model,
            inputs=[ct.TensorType(shape=m.shape) for m in example_values],
            minimum_deployment_target=ct.target.iOS15,
        )

        if isinstance(converted_model, Program) or isinstance(converted_model, MLModel):
            _LOGGER.info('Success to convert as ".mlpackage".')
            return (converted_model, _MLPACKAGE_EXTENSION)
        else:
            return (None, "")
