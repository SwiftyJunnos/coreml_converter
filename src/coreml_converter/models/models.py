import os
from enum import StrEnum

import coremltools as ct
import torch
import torch.nn as nn
from coremltools.models import MLModel
from transformers import RobertaConfig, RobertaTokenizerFast

from .const import (
    G_STEP_NUM_LABEL,
    KEYWORD_NUM_LABEL,
    INTENT_NUM_LABEL,
    MODEL_URL,
    RESULT_DIR,
    _MLPACKAGE_EXTENSION,
)
from .available_models import IntentModel, KeywordModel, GStepModel


class Models(StrEnum):
    INTENT = "Intent"
    KEYWORD = "Keyword"
    GSTEP = "GStep"

    def config(self, num_labels: int) -> RobertaConfig:
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_URL)
        return RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=512,
            hidden_dropout_prob=0.1,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512 * 4,
            max_position_embeddings=514,
            type_vocab_size=1,
            num_labels=num_labels,
        )

    @property
    def num_labels(self) -> int:
        if self == Models.INTENT:
            return INTENT_NUM_LABEL
        elif self == Models.KEYWORD:
            return KEYWORD_NUM_LABEL
        elif self == Models.GSTEP:
            return G_STEP_NUM_LABEL
        else:
            raise ValueError("Invalid case for enum `Models`.")

    def example_inputs(self, input_size: torch.Size) -> dict[str, torch.Tensor]:
        # input_size: (batch_size, 512)
        if self is Models.INTENT:
            _input_size = (input_size[0], input_size[1] * 3)
            return {
                "input_ids": torch.ones(_input_size).to(torch.int32),
                "attention_mask": torch.zeros(_input_size).to(torch.int32),
            }
        elif self == Models.KEYWORD:
            return {
                "input_ids": torch.ones(input_size).to(torch.int32),
                "attention_mask": torch.zeros(input_size).to(torch.int32),
            }
        elif self == Models.GSTEP:
            return {
                "doc_input_ids": torch.ones(input_size).to(torch.int32),
                "doc_attention_mask": torch.zeros(input_size).to(torch.int32),
                "intent_input_ids": torch.ones(input_size).to(torch.int32),
                "intent_attention_mask": torch.zeros(input_size).to(torch.int32),
                "keyword_input_ids": torch.ones(input_size).to(torch.int32),
                "keyword_attention_mask": torch.zeros(input_size).to(torch.int32),
            }
        else:
            raise ValueError("Invalid case for enum `Models`.")

    def make_model(self) -> nn.Module:
        num_labels = self.num_labels
        config = self.config(num_labels)
        if self == Models.INTENT:
            return IntentModel(config, num_labels)
        elif self == Models.KEYWORD:
            return KeywordModel(config, num_labels)
        elif self == Models.GSTEP:
            return GStepModel(config, num_labels)
        else:
            raise ValueError("Invalid case for enum `Models`.")

    def load_coreml_package(self) -> MLModel:
        model_path = os.path.join(
            RESULT_DIR, "opt_" + self.value + _MLPACKAGE_EXTENSION
        )
        return ct.models.MLModel(model_path)
