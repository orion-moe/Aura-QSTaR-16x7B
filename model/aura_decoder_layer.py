# pylint: disable=too-many-arguments,too-many-branches,too-many-statements,too-many-locals,import-error,no-name-in-module,relative-beyond-top-level
""" AuraDecoderLayer """
import warnings

from typing import Optional, Tuple

import torch
from torch import nn

from .attention_aura import AuraAttention, AuraFlashAttention2, AuraSdpaAttention
from .configuration_aura import AuraConfig
from .aura_mlp import AuraMLP
from .aura_rms_norm import AuraRMSNorm

AURA_ATTENTION_CLASSES = {
    "eager": AuraAttention,
    "flash_attention_2": AuraFlashAttention2,
    "sdpa": AuraSdpaAttention,
}

class AuraDecoderLayer(nn.Module):
    """ AuraDecoderLayer Class """
    def __init__(self, config: AuraConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = AURA_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        self.mlp = AuraMLP(config)
        self.input_layernorm = AuraRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = AuraRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "A passagem de `padding_mask` está depreciada e será removida na versão 4.37."
                " Certifique-se de usar `attention_mask`.`"
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # hidden_states = residual + hidden_states
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Totalmente Conectado
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
