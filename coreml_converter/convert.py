import logging
from typing import Optional

import coremltools as ct
import torch
import torch.jit as jit
import torch.nn as nn
from coremltools.converters.mil import Program

_LOGGER = logging.getLogger(__name__)


class Converter:
    def __init__(self):
        pass

    def convert(
        self, model: nn.Module, **example_inputs: torch.Tensor
    ) -> Optional[Program]:
        model.eval()

        example_values = tuple(example_inputs.values())
        _LOGGER.info(f"Tracing: {example_values}")
        traced_model = jit.trace(model, example_values)
        _LOGGER.info("Tracing complete!")

        _LOGGER.info("Start converting...")
        converted_model = ct.convert(
            model=traced_model,
            source="pytorch",
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_values[0].shape)],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS15,
        )

        if converted_model is Program:
            _LOGGER.info("Success to convert model.")
            return converted_model
        else:
            _LOGGER.error("Failed to convert model.")
            return None
