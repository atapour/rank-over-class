

import torch.nn as nn
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel

# ROBERTA:
# ----------------------------------------------
class RobertaForSequenceRanking(BertPreTrainedModel):

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # ------------------------------------------
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
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

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[1]
        outputs = sequence_output

        return outputs
    # ------------------------------------------
# ----------------------------------------------
