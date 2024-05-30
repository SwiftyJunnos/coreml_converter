import torch.nn as nn
from transformers import RobertaModel


class IntentModel(nn.Module):
    def __init__(self, config, num_labels):
        super(IntentModel, self).__init__()
        self.roberta = RobertaModel(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(3 * config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # input_ids.shape: (batch size, 1536)
        batch_size = input_ids.size(0)

        input_ids = input_ids.reshape(batch_size * 3, -1)
        attention_mask = attention_mask.reshape(batch_size * 3, -1)

        # Pass through RoBERTa model
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # Get CLS token

        # Reshape CLS tokens to (batch_size, 3, hidden_size)
        cls_tokens = cls_tokens.reshape(batch_size, 3, -1)

        # Flatten the tokens to (batch_size, 3 * hidden_size)
        flattened_cls_tokens = cls_tokens.reshape(batch_size, -1)

        # Apply dropout
        pooled_output = self.dropout(flattened_cls_tokens)

        # Apply each classifier
        logits = self.classifier(pooled_output)

        # Stack and apply sigmoid
        output = self.sigmoid(logits)

        return output
