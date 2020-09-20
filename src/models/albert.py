

import torch.nn as nn
from transformers import AlbertModel, AlbertPreTrainedModel

# ----------------------------------------------
class AlbertForSequenceRanking(AlbertPreTrainedModel):

    # ------------------------------------------
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.init_weights()
    # ------------------------------------------

    # ------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        outputs = pooled_output
        return outputs
    # ------------------------------------------
# ----------------------------------------------
