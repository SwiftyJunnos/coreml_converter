import torch
import torch.nn as nn
from transformers import RobertaModel


class GStepModel(nn.Module):
    def __init__(self, config, num_labels):
        super(GStepModel, self).__init__()
        self.roberta_doc = RobertaModel(config)
        self.roberta_intent = RobertaModel(config)
        self.roberta_keyword = RobertaModel(config)

        self.dropout = nn.Dropout(self.roberta_doc.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.roberta_doc.config.hidden_size * 3, num_labels)

    def forward(
        self,
        doc_input_ids,
        doc_attention_mask,
        intent_input_ids,
        intent_attention_mask,
        keyword_input_ids,
        keyword_attention_mask,
    ):
        doc_output = self.roberta_doc(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        )
        doc_cls_token = doc_output.last_hidden_state[:, 0, :]

        intent_output = self.roberta_intent(
            input_ids=intent_input_ids, attention_mask=intent_attention_mask
        )
        intent_cls_token = intent_output.last_hidden_state[:, 0, :]

        keyword_output = self.roberta_keyword(
            input_ids=keyword_input_ids, attention_mask=keyword_attention_mask
        )
        keyword_cls_token = keyword_output.last_hidden_state[:, 0, :]

        cls_tokens = torch.cat(
            (doc_cls_token, intent_cls_token, keyword_cls_token), dim=1
        )

        pooled_output = self.dropout(cls_tokens)

        logits = self.classifier(pooled_output)

        return logits
