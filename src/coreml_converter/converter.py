import logging
import numpy as np

import coremltools as ct
import torch
import torch.jit as jit
import torch.nn as nn
from coremltools.models.model import MLModel

from .const import _MLPACKAGE_EXTENSION

_LOGGER = logging.getLogger(__name__)


class Converter:
    def __init__(self):
        pass

    def convert(
        self, model: nn.Module, **example_inputs: torch.Tensor
    ) -> tuple[MLModel, str]:
        model.eval()

        _example_values = example_inputs.values()
        traced_model = jit.trace(model, list(_example_values))

        _LOGGER.info("Start converting...")

        converted_model = ct.convert(
            model=traced_model,
            inputs=[
                ct.TensorType(name=m[0], shape=m[1].shape, dtype=np.int32)
                for m in example_inputs.items()
            ],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.iOS15,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
        )

        if isinstance(converted_model, MLModel):
            return (converted_model, _MLPACKAGE_EXTENSION)
        else:
            raise ValueError("Format of converting output is not a `MLModel`.")
