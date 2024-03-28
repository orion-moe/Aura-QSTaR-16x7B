# pylint: disable=too-many-instance-attributes,import-error,no-name-in-module,relative-beyond-top-level
""" AuraMLP """
from torch import nn
from transformers.activations import ACT2FN

from .aura_gate_adapter import AuraGateAdapter

class AuraMLP(nn.Module):
    """ AuraMLP """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.moe_adapter = AuraGateAdapter(config)

    def forward(self, x):
        router_hidden_states = x
        up_proj = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(up_proj)
        down_proj, router_logits = self.moe_adapter(
            down_proj, down_proj, router_hidden_states
        )

        return down_proj, router_logits
