import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class KeywordModel(nn.Module):
    def __init__(self, config, num_labels):
        super(KeywordModel, self).__init__()
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = self.dropout(sequence_output)

        logits = self.classifier(pooled_output)

        # Softmax 함수를 적용하여 확률 값으로 변환
        probabilities = F.softmax(logits, dim=-1)

        max_indices = torch.max(probabilities, dim=-1)[1]

        return max_indices
