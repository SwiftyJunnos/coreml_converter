import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Program
from coremltools.models.model import MLModel


def quantize(model: Program | MLModel) -> Program | MLModel:
    op_config = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=512
    )
    config = cto.OptimizationConfig(global_config=op_config)

    quantized_model = cto.linear_quantize_weights(model, config=config)
    if quantized_model is None:
        raise ValueError("Failed to optimize by quantizing.")
    return quantized_model
