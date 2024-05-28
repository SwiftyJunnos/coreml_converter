from enum import StrEnum

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizerFast

from .constants import G_STEP_NUM_LABEL, KEYWORD_NUM_LABEL, INTENT_NUM_LABEL, MODEL_URL
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
        # input_size: (32000, 512)
        if self == Models.INTENT:
            _input_size = (input_size[0], input_size[-1] * 3)
            return {
                "input_ids": torch.rand(_input_size).to(torch.int64),
                "attention_mask": torch.zeros(_input_size).to(torch.int64)
            }
        elif self == Models.KEYWORD:
            return {
                "input_ids": torch.rand(input_size),
                "attention_mask": torch.Tensor(input_size[-1])
            }
        elif self == Models.GSTEP:
            return {
                "doc_input_ids": torch.rand(input_size),
                "doc_attention_mask": torch.Tensor(input_size[-1]),
                "intent_input_ids": torch.rand(input_size),
                "intent_attention_mask": torch.Tensor(input_size[-1]),
                "keyword_input_ids": torch.rand(input_size),
                "keyword_attention_mask": torch.Tensor(input_size[-1])
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
