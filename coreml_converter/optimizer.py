import coremltools.optimize.coreml as cto
import torch.nn as nn


def quantize(model: nn.Module):
    quantization_model = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=512
    )
    config = cto.OptimizationConfig(global_config=config)
