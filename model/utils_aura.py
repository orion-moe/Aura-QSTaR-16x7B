# pylint: disable=broad-exception-caught,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,unused-argument,line-too-long,logging-fstring-interpolation,no-member,protected-access,not-callable
""" Funções utilitárias da Aura """
import torch
from torch import nn

def nonzero_mean(x, axis=None):
    """ Calcula a média dos valores diferentes de 0 """
    if axis is not None:
        return x.sum(axis) / (x != 0).sum(axis)
    return x.sum() / (x != 0).sum()

def loss_mean(x):
    """ Calcula a média da função de perda """
    return x.sum() / (x != 0).sum()

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
        Isso é equivalente a torch.repeat_interleave(x, dim=1, repete=n_rep). Os estados ocultos vão de (lote,
        num_key_value_heads, seqlen, head_dim) para (lote, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.size()

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )

    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Aura Mixture of Experts
def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
) -> float:
    if gate_logits is None:
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        gate_logits = torch.cat(
            [gate.to(compute_device) for gate in gate_logits], dim=0
        )

    routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1)

    # Transforma os índices dos especialistas em int64, caso contrário a codificação one-hot falhará
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Para um determinado token, determine se ele foi roteado para um determinado especialista.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # convertido para float32 caso contrário significa que falhará
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)
    ) * (num_experts**2)

def rotate_half(x):
    """ Gira metade dos dims ocultos da entrada. """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_old(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """ Aplica incorporação de posição rotativa à consulta e aos tensores principais. """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

import torch

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Aplica incorporação de posição rotativa à consulta e aos tensores principais."""
    # Assegurar que position_ids está dentro dos limites de cos e sin
    max_pos_id = cos.size(0) - 1  # Obter o índice máximo permitido
    clipped_position_ids = position_ids.clamp(min=0, max=max_pos_id)  # Restringir position_ids aos limites

    cos = cos[clipped_position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[clipped_position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class AuraRotaryEmbedding(nn.Module):
    """ AuraRotaryEmbedding """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

		# fazer `torch.jit.trace` funcionar aqui
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """ AuraRotaryEmbedding Forward """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
