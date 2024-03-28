# pylint: disable=too-many-instance-attributes,import-error,no-name-in-module,import-error,no-name-in-module,relative-beyond-top-level
""" AuraGateAdapter """
import torch
from torch import nn
import torch.nn.functional as F

from .configuration_aura import AuraConfig
from .aura_parallel_adapter_mlp import AuraParallelAdapterMLP

class AuraGateAdapter(nn.Module):
    def __init__(self, config: AuraConfig):
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        # Passo 1: Roteamento
        self.num_experts = config.num_experts
        self.topk = config.topk
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.dtype = getattr(torch, config.moe_dtype)

        # Passo 2: Experts
        self.expert_indicies = list(range(self.num_experts))
        self.experts = nn.ModuleList(
            [
                AuraParallelAdapterMLP(config, config.adapter_dim, config.moe_scaling)
                for _ in self.expert_indicies
            ]
        )

    def forward(self, input_hidden_states, output_hidden_states, router_hidden_states):
        input_hidden_states = input_hidden_states.to(self.dtype)
        output_hidden_states = output_hidden_states.to(self.dtype)
        router_hidden_states = router_hidden_states.to(self.dtype)

        orig_shape = output_hidden_states.shape
        input_hidden_states = input_hidden_states.view(
            -1, input_hidden_states.shape[-1]
        )
        output_hidden_states = output_hidden_states.view(
            -1, output_hidden_states.shape[-1]
        )
        router_hidden_states = router_hidden_states.view(
            -1, router_hidden_states.shape[-1]
        )

        router_logits = self.router(router_hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = selected_experts == expert_idx
            expert_weights = (routing_weights * expert_mask).sum(dim=-1, keepdim=True)

            current_hidden_states = (
                expert_layer(input_hidden_states)
                .add_(output_hidden_states)
                .mul_(expert_weights)
            )
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return final_hidden_states.view(*orig_shape), router_logits
