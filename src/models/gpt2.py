import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_utils import SequenceSummary

# ----------------------------------------------
class GPT2ForSequenceRanking(GPT2PreTrainedModel):

    # ------------------------------------------
    def __init__(self, config):
        super(GPT2ForSequenceRanking, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        config.summary_type='mean'
        self.good_head = SequenceSummary(config)

        self.size = config.n_embd

        self.init_weights()
    # ------------------------------------------

    # ------------------------------------------
    def get_output_embeddings(self):
        return self.lm_head
    # ------------------------------------------

    # ------------------------------------------
    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]
        outputs = self.good_head(hidden_states).squeeze(-1)

        return outputs
    # ------------------------------------------
# ----------------------------------------------
