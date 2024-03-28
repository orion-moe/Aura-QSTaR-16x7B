import math
import time
from typing import List, Optional, Tuple, Union
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)

from .reporting_aura import save_tokens_with_rewards_to_pdf
from .utils_aura import load_balancing_loss_func, nonzero_mean, loss_mean
from .configuration_aura import AuraConfig
from .attention_aura import AuraAttention, AuraFlashAttention2, AuraSdpaAttention

from .aura_rms_norm import AuraRMSNorm
from .aura_moe_outputs import AuraMoEModelOutputWithPast, AuraMoECausalLMOutputWithPast
from .aura_decoder_layer import AuraDecoderLayer

logger = logging.get_logger(__name__)

AURA_ATTENTION_CLASSES = {
    "eager": AuraAttention,
    "flash_attention_2": AuraFlashAttention2,
    "sdpa": AuraSdpaAttention,
}

num_token_gen = 0

class AuraPreTrainedModel(PreTrainedModel):
    config_class = AuraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AuraDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class AuraModel(AuraPreTrainedModel):
    def __init__(self, config: AuraConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                AuraDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = AuraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AuraMoEModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # recupera input_ids e inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "Você não pode especificar decoder_input_ids e decoder_inputs_embeds ao mesmo tempo"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "Você deve especificar decoder_input_ids ou decoder_inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` é incompatível com checkpoint de gradiente. Definir `use_cache=False`..."
                )
                use_cache = False

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
					"Você está tentando realizar geração em lote com padding_side='right'"
                    " isso pode levar a um comportamento inesperado para a versão Flash Attention do Aura. Certifique-se de "
                    " chame `tokenizer.padding_side = 'left'` antes de tokenizar a entrada. "
                )

        if self._attn_implementation == "flash_attention_2":
            # A máscara 2d é passada pelas camadas
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
			# output_attentions=True não pode ser suportado ao usar SDPA, e recorremos
            # a implementação manual que requer uma máscara causal 4D em todos os casos.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # A máscara 4d é passada pelas camadas
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # camadas de decodificação
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None pra past_key_value
                        return module(
                            *inputs, output_attentions, output_router_logits, None
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # Adiciona estados ocultos da última camada do decodificador
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return AuraMoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

class AuraForCausalLM(AuraPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = AuraModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_thoughts = config.max_thoughts
        self.merged_lm_and_talk_heads = config.merged_lm_and_talk_heads
        self.use_concat_talk_head = config.use_concat_talk_head
        self.use_shallow_talk = config.use_shallow_talk
        self.use_complex_talk_head = config.use_complex_talk_head
        self.use_weighted_talk_head = config.use_weighted_talk_head
        # O weighted head produzirá um único valor, portanto não pode ser
        # passado para o lm head
        assert not (self.use_weighted_talk_head and self.use_shallow_talk)

        self.n_ahead = 1
        self.n_ahead_talk = 1
        self.n_passes = 1
        self.n_tokens_print = 1
        self.gradient_accumulation_steps = 1
        self.training_steps = 0
        self.start_token_id = None
        self.end_token_id = None
        self.rm_initialized = False
        self.residual_talk_head = True
        self.thought_init_std_scale = 1e-2

        self.final_only_mode = False
        self.first_and_last_mode = True
        self.first_only = False
        self.original_loss_weight = 0.5

        self.cumulative_residual = False
        self.clever_residual = False
        self.skip_residual = False
        self.no_residual = True

        self.optimize_lm_head_only_at_start = False
        self.optimize_model_only_at_start = False

        if self.optimize_model_only_at_start:
            raise NotImplementedError
        self.train_only_thinking_embedding = False
        self.weighted_embeddings = False
        self.use_start_thought_token = True
        self.use_end_thought_token = True
        self.initialize_thought_embedding_to_normal = False
        self.initial_start_token = "<|im_start|>"
        self.initial_end_token = "<|im_end|>"
        self.output_logits_at_the_end = True
        self.tokenizer_has_start_thought_token = False
        self.tokenizer_has_end_thought_token = False

        self.wandb_enabled = False
        self.gumbel_temperature = 0.001

        self.use_policy_loss = True
        self.include_policy_loss = True
        self.trice_mode = True
        self.remove_negative_rewards = True
        self.use_policy_loss_for_end_thought = True

        self.base_original_mode = False
        self.original_mode = False

        self.thought_prefix = "(Vamos pensar passo-a-passo"
        self.tokenized_thought_prefix = None
        self.log_dict = defaultdict(int)
        self.eval_log_dict = defaultdict(int)
        self.print_final_only = True
        self.loss_mean = loss_mean
        self.all_rewards = []
        self.all_unreduced_losses = []
        # self.kill_after = 100

        self.start_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))
        self.end_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))

        self.policy_loss_beta = 1e6
        self.embedding_scale = 1e2
        self.reinforce_temperature = 3
        self.base_loss_beta = 1.0

        # Não foi utilizado no artigo:
        self.use_thought_prefix = False
        self.use_reparam_for_thought_embeddings = False
        self.use_upper_triangular = False
        self.subtract_mean_reward = False
        self.comparison_mode = False
        self.gumbel_detach = True

        # Para visualização
        self.eval_mode = True

        num_talk = 1
        talk_input_dim = config.hidden_size if not self.use_concat_talk_head else config.hidden_size * 2
        if self.use_weighted_talk_head:
            talk_output_dim = 1
        else:
            talk_output_dim = config.hidden_size if self.use_shallow_talk else config.vocab_size

        if not self.merged_lm_and_talk_heads:
            if self.use_complex_talk_head:
                self.talk_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(talk_input_dim, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, talk_output_dim, bias=False)
                )])
            else:
                self.talk_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                )])

        # Inicialize os pesos e aplique o processamento final
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def apply_head(self, head, states, detach=False):
        if detach:
            head_weight = head.weight.detach()
        else:
            head_weight = head.weight
        head_weight = head_weight.to(states.device)
        return (head_weight @ states.transpose(-1, -2)).transpose(-1, -2).contiguous()

    def idx_if_sequential(self, head, idx=0):
        if isinstance(head, nn.Sequential) or isinstance(head, nn.ModuleList):
            return idx_if_sequential(head[idx], idx=idx)
        return head

    def none_repeat_interleave(self, x, n):
        if x is None:
            return x
        return x.repeat_interleave(n, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0].to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            # Muda para que os tokens < n prevejam n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Achata os tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Habilita paralelismo do modelo
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # As saídas de decodificação consistem em (dec_features, layer_state, dec_hidden, dec_attn)
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.config.num_experts,
                self.config.topk,
            )
            if labels is not None:
                aux_loss = aux_loss.to(loss.device)
                loss += self.config.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        log_dict = self.log_dict if self.training else self.eval_log_dict

        if not self.training:
            n_ahead_talk_to_restore = self.n_ahead_talk
            n_passes_to_restore = self.n_passes
            self.n_ahead_talk = 1
            self.n_passes = 1

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert self.cumulative_residual or self.clever_residual or self.skip_residual or self.no_residual
        assert not (self.skip_residual and self.use_policy_loss)

        if self.tokenized_thought_prefix is None and self.use_thought_prefix:
            self.tokenized_thought_prefix = self.tokenizer(self.thought_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]

        if self.n_passes > 1:
            input_ids = self.none_repeat_interleave(input_ids, self.n_passes)
            attention_mask = self.none_repeat_interleave(attention_mask, self.n_passes)
            position_ids = self.none_repeat_interleave(position_ids, self.n_passes)
            inputs_embeds = self.none_repeat_interleave(inputs_embeds, self.n_passes)
            labels = self.none_repeat_interleave(labels, self.n_passes)
            if past_key_values is not None:
                past_key_values = [self.none_repeat_interleave(p, self.n_passes) for p in past_key_values]
        cur_token_indices = torch.arange(input_ids.shape[1], device=input_ids.device)

        self.tokenizer_has_start_thought_token = True
        self.tokenizer_has_end_thought_token = True
        if self.start_token_id is None:
            self.start_token_id = self.tokenizer.convert_tokens_to_ids("<|th_start|>")
            if self.start_token_id == 0:
                self.start_token_id = self.tokenizer.bos_token_id
                self.tokenizer_has_start_thought_token = False
            elif self.use_start_thought_token:
                # base_start_id = self.tokenizer.convert_tokens_to_ids(self.initial_start_token)
                base_start_id = self.tokenizer.encode(self.initial_start_token, add_special_tokens=False)[0]
                if self.initialize_thought_embedding_to_normal:
                    self.start_embedding.data = torch.zeros_like(self.start_embedding.data)
                else:
                    self.start_embedding.data[0] = self.model.embed_tokens.weight.data[base_start_id].clone().detach() / self.embedding_scale
                self.start_embedding.data[1] = torch.log(self.model.embed_tokens.weight.data.std(dim=0) * self.thought_init_std_scale / self.embedding_scale)

        if self.end_token_id is None:
            self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|th_end|>")
            if self.end_token_id == 0:
                self.end_token_id = self.tokenizer.eos_token_id
                self.tokenizer_has_end_thought_token = False
            elif self.use_end_thought_token:
                # base_end_id = self.tokenizer.convert_tokens_to_ids(self.initial_end_token)
                base_end_id = self.tokenizer.encode(self.initial_end_token, add_special_tokens=False)[0]
                if self.initialize_thought_embedding_to_normal:
                    self.end_embedding.data = torch.zeros_like(self.end_embedding.data)
                else:
                    self.end_embedding.data[0] = self.model.embed_tokens.weight.data[base_end_id].clone().detach() / self.embedding_scale
                self.end_embedding.data[1] = torch.log(self.model.embed_tokens.weight.data.std(dim=0) * self.thought_init_std_scale / self.embedding_scale)

        if not self.rm_initialized and (self.n_ahead > 1 or not self.base_original_mode):
            self.rm_initialized = True
            if not self.use_shallow_talk:
                head = self.talk_head[0]
                cur_head = head[-1] if isinstance(head, nn.Sequential) else head
                talk_input_dim = cur_head.weight.data.shape[1]
                talk_output_dim = 1 if self.use_weighted_talk_head else self.lm_head.weight.data.shape[0]
                cur_head.weight.data = torch.zeros(talk_output_dim, talk_input_dim, device=cur_head.weight.device, dtype=cur_head.weight.dtype)
            else:
                # converter para identity transform
                def lambda_transform(cur_head):
                    if cur_head.weight.data.shape[0] != cur_head.weight.data.shape[1]:
                        return torch.cat([
                        torch.eye(
                            cur_head.weight.data.shape[0],
                            device=cur_head.weight.device,
                            dtype=cur_head.weight.dtype
                        ),
                        torch.zeros(
                            cur_head.weight.data.shape[0],
                            cur_head.weight.data.shape[1] - cur_head.weight.data.shape[0],
                            device=cur_head.weight.device,
                            dtype=cur_head.weight.dtype
                        )], dim=1)
                    return torch.eye(
                        cur_head.weight.data.shape[0],
                        device=cur_head.weight.device,
                        dtype=cur_head.weight.dtype
                    )
                if isinstance(self.talk_head[0], nn.Sequential):
                    for cur_head in self.talk_head[0]:
                        # Se tiver pesos
                        if hasattr(cur_head, "weight"):
                            cur_head.weight.data = lambda_transform(cur_head)
                else:
                    self.talk_head[-1].weight.data = lambda_transform(self.talk_head[0])

        prev_rm_tokens = None
        cur_rm_tokens = None
        prev_rm_logits = None
        prev_sample_probs = None
        did_skip_sampling = None
        skip_sampling = None
        sample_probs = None
        hidden_states = None
        talk_kl_penalty = None
        rm_logits = None
        residual_logits = None
        probabilities_2d = None
        prev_probabilities_2d = None
        policy_reward = None
        logits_to_output = None
        batch_size, seq_len = input_ids.shape
        base_input_ids = input_ids.clone()
        loss_list = []
        dqn_loss_list = []
        sampled_token_history = []
        sample_probs_history = []
        action_loglikelihoods_list = []

        if self.use_end_thought_token or self.use_start_thought_token:
            if not self.use_reparam_for_thought_embeddings:
                start_embedding = self.start_embedding[0].unsqueeze(0) * self.embedding_scale
                end_embedding = self.end_embedding[0].unsqueeze(0) * self.embedding_scale
            else:
                start_embedding = self.start_embedding * self.embedding_scale
                end_embedding = self.end_embedding * self.embedding_scale

            base_embeddings = self.model.embed_tokens.weight
            if self.train_only_thinking_embedding:
                base_embeddings = base_embeddings.detach()

        # A saída do decoder consiste em (dec_features, layer_state, dec_hidden, dec_attn)
        fwd_iters = 1 if self.original_mode else self.n_ahead + self.n_ahead_talk - 1
        for ahead_idx in range(fwd_iters):
            past_key_values_length = 0
            if past_key_values is not None:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_len)

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
            else:
                position_ids = position_ids.view(-1, seq_len).long()

            if inputs_embeds is None:
                contains_start = self.use_start_thought_token and (input_ids == self.start_token_id).any()
                contains_end = self.use_end_thought_token and (input_ids == self.end_token_id).any()
                contains_thought = contains_start or contains_end

                if contains_thought:
                    thought_id = self.start_token_id if contains_start else self.end_token_id
                    cur_thought_embedding = start_embedding if contains_start else end_embedding
                    if self.use_reparam_for_thought_embeddings:
                        inputs_embeds = torch.randn(batch_size, seq_len, self.model.config.hidden_size, device=input_ids.device, dtype=cur_thought_embedding.dtype)
                        inputs_embeds = inputs_embeds.detach() * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
                        if contains_end:
                            sampled_end = inputs_embeds.clone().detach()
                    else:
                        inputs_embeds = cur_thought_embedding.unsqueeze(0).repeat(batch_size, seq_len, 1)
                else:
                    with torch.set_grad_enabled(not self.train_only_thinking_embedding):
                        inputs_embeds = self.model.embed_tokens(input_ids)

            if self.n_ahead != 1 or self.n_ahead_talk != 1 or self.comparison_mode:
                if attention_mask is None:
                    base_attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(input_ids.device)
                    base_attention_mask = base_attention_mask.view(1, 1, seq_len, seq_len)
                    base_attention_mask = base_attention_mask.repeat(input_ids.shape[0], 1, 1, 1)
                    attention_mask = base_attention_mask
                    breakpoint()
                elif attention_mask.dim() == 2:
                    if seq_len + past_key_values_length != attention_mask.shape[-1]:
                        breakpoint()
                        attention_mask = torch.cat(
                            [torch.ones((attention_mask.shape[0], past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask],
                            dim=-1
                        )

                    attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask,
                        (batch_size, seq_len),
                        inputs_embeds,
                        past_key_values_length,
                        sliding_window=self.config.sliding_window,
                    )

            outputs = self.model(
                #input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=return_dict,
            )

            prev_hidden_states = hidden_states
            hidden_states = outputs[0].to(self.lm_head.weight.dtype)
            prev_rm_logits = rm_logits  # Para a política de gradient
            prev_rm_tokens = cur_rm_tokens  # Para a política de gradient

            if ahead_idx == 0:
                hidden_states_lm = hidden_states
                logits = self.lm_head(hidden_states_lm)
                base_hidden_states = hidden_states.clone()
                initial_loss_logits = logits.clone()
                if self.optimize_lm_head_only_at_start or self.optimize_model_only_at_start:
                    logits = logits.detach()
                    base_hidden_states = base_hidden_states.detach()
                if self.optimize_model_only_at_start:
                    hidden_states = hidden_states.detach()
                base_logits = logits.clone()
            else:
                talk_hidden_states = hidden_states
                if self.merged_lm_and_talk_heads:
                    assert self.no_residual
                    residual_logits = self.lm_head(hidden_states)
                    talk_hidden_states = hidden_states
                else:
                    if ahead_idx > self.n_ahead - 1:
                        cur_base_hidden = torch.cat([
                            base_hidden_states[..., ahead_idx - self.n_ahead + 1:, :],
                            base_hidden_states[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                    else:
                        cur_base_hidden = base_hidden_states

                    if self.use_concat_talk_head:
                        # Concatena os estados ocultos com o estado oculto original
                        head_input_hidden_states = torch.cat([cur_base_hidden, talk_hidden_states], dim=-1)
                    else:
                        head_input_hidden_states = talk_hidden_states

                    residual_logits = self.talk_head[0](head_input_hidden_states)
                    if self.use_shallow_talk:
                        residual_logits = self.apply_head(
                            self.lm_head,
                            residual_logits,
                            detach=self.optimize_lm_head_only_at_start
                        )
                    residual_logits = residual_logits.to(logits.device)
                    if self.use_weighted_talk_head:
                        # Combina cur_base_hidden com talk_hidden_states de acordo com o head ponderado
                        residual_logits = cur_base_hidden * (1 - residual_logits) + talk_hidden_states * residual_logits
                        residual_logits = self.apply_head(self.lm_head, residual_logits, detach=self.optimize_lm_head_only_at_start)

                assert sum([self.cumulative_residual, self.clever_residual, self.skip_residual, self.no_residual]) == 1
                if self.clever_residual:
                    if ahead_idx >= self.n_ahead - 1:
                        # Muda os logits de acordo com a conversa atual
                        cur_base_logits = torch.cat([
                            base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                            base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                        if self.optimize_lm_head_only_at_start:
                            cur_base_logits = cur_base_logits.detach()
                        logits = cur_base_logits + residual_logits
                    else:
                        logits += residual_logits / self.n_ahead
                elif self.cumulative_residual:
                    if self.residual_talk_head:
                        if ahead_idx < self.n_ahead:
                            logits += residual_logits
                        else:
                            # Muda os logits de acordo com a conversa atual
                            cur_base_logits = torch.cat([
                                base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                                base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                            ], dim=-2)
                            if self.optimize_lm_head_only_at_start:
                                cur_base_logits = cur_base_logits.detach()
                            logits = cur_base_logits + residual_logits
                    else:
                        if ahead_idx < self.n_ahead:
                            logits += residual_logits
                        else:
                            logits = residual_logits
                elif self.skip_residual:
                    if ahead_idx >= self.n_ahead:
                        # Muda os logits de acordo com a conversa atual
                        cur_base_logits = torch.cat([
                            base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                            base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                        if self.optimize_lm_head_only_at_start:
                            cur_base_logits = cur_base_logits.detach()
                        logits = cur_base_logits
                elif self.no_residual:
                    logits = residual_logits
                else:
                    logits = base_logits + residual_logits

            attempted = False
            talk_loss_list = []
            if self.original_mode or (self.n_ahead == 1) or (self.comparison_mode and ahead_idx == 0) or (self.optimize_lm_head_only_at_start and ahead_idx == 0):
                # As saídas de decodificação consistem em (dec_features, layer_state, dec_hidden, dec_attn)
                attempted = True

                if labels is not None:
                    for shift_amount in range(self.n_ahead_talk):
                        if ahead_idx == 0 and self.optimize_lm_head_only_at_start:
                            loss_logits = initial_loss_logits
                        else:
                            loss_logits = logits
                        shift_logits = loss_logits[..., shift_amount:-1, :].contiguous()
                        shift_labels = labels[..., 1 + shift_amount:].contiguous()

                        # Achata os tokens
                        loss_fct = CrossEntropyLoss(reduction="none")
                        shift_logits = shift_logits.view(-1, self.config.vocab_size)
                        shift_labels = shift_labels.view(-1).clone()
                        # Habilita o paralelismo do modelo
                        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
                        shift_labels = shift_labels.to(shift_logits.device)
                        loss = loss_fct(shift_logits, shift_labels)
                        if not self.comparison_mode and not (self.optimize_lm_head_only_at_start and (self.n_ahead + self.n_ahead_talk > 2)) or self.original_mode:
                            loss_list.append(loss)
                        talk_loss_list.append(nonzero_mean(loss).detach())

            if not attempted or self.comparison_mode:
                rm_hidden_states = hidden_states
                rm_logits = self.apply_head(self.lm_head, rm_hidden_states, detach=self.optimize_lm_head_only_at_start)

                # Não permite que token de pensamento seja previsto
                if self.tokenizer_has_start_thought_token:
                    rm_logits[..., self.start_token_id] = -1e10 if self.training else torch.finfo(rm_logits.dtype).min
                if self.tokenizer_has_end_thought_token:
                    rm_logits[..., self.end_token_id] = -1e10 if self.training else torch.finfo(rm_logits.dtype).min
                probabilities = rm_logits
                if probabilities_2d is not None:
                    prev_probabilities_2d = probabilities_2d.clone()
                probabilities_2d = probabilities.view(-1, probabilities.size(-1))

                did_skip_sampling = skip_sampling
                skip_sampling = False

                if ahead_idx == 0 and self.use_start_thought_token:
                    override_token = self.start_token_id
                elif self.use_thought_prefix and ahead_idx < self.tokenized_thought_prefix.shape[-1]:
                    override_token = self.tokenized_thought_prefix[..., ahead_idx]
                elif ahead_idx == self.n_ahead - 2 and self.use_end_thought_token:
                    override_token = self.end_token_id
                else:
                    override_token = None
                if override_token is not None and self.n_ahead > 1:
                    # Sempre começa com o token inicial
                    probabilities_2d = torch.zeros_like(probabilities_2d)
                    probabilities_2d[:, override_token] = 1.0
                    skip_sampling = True
                elif ahead_idx >= self.n_ahead - 1:
                    if labels is not None:  # estamos na fase de conversa
                        cur_talk_n = ahead_idx - (self.n_ahead - 1) + 1
                        # print("Definindo rm para labels", cur_talk_n, "durante", ahead_idx)
                        shift_labels = labels[..., cur_talk_n:].contiguous().to(probabilities_2d.device)
                        padding = torch.full_like(
                            labels[..., :cur_talk_n],
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=shift_labels.device
                        )
                        new_rm_tokens = torch.cat(
                            [shift_labels, padding],
                            dim=-1
                        )
                        # Converte rm tokens para one-hot
                        probabilities_2d = F.one_hot(new_rm_tokens, num_classes=self.vocab_size).reshape(-1, self.vocab_size).to(probabilities_2d.dtype)
                        skip_sampling = True
                    else:
                        continue

                temperature = self.gumbel_temperature if self.training else 0.001
                prev_sample_probs = sample_probs
                sample_probs = probabilities_2d
                if ahead_idx < self.n_ahead - 1 and not skip_sampling:
                    probabilities_2d = F.gumbel_softmax(sample_probs, tau=temperature, hard=True, dim=-1)
                    if self.gumbel_detach:
                        probabilities_2d = probabilities_2d.detach()
                sampled_token_history.append(probabilities_2d.argmax(dim=-1).detach().cpu())
                # Converte rm logits diretamente em embeddings
                contains_start = self.use_start_thought_token and (probabilities_2d[..., self.start_token_id].sum() > 0)
                contains_end = self.use_end_thought_token and (probabilities_2d[..., self.end_token_id].sum() > 0)
                contains_thought = contains_start or contains_end

                if not contains_thought:
                    with torch.set_grad_enabled(not self.train_only_thinking_embedding):
                        inputs_embeds = probabilities_2d @ (self.model.embed_tokens.weight.to(probabilities.device).to(probabilities.dtype))
                else:
                    thought_id = self.start_token_id if contains_start else self.end_token_id
                    cur_thought_embedding = start_embedding if contains_start else end_embedding
                    if self.use_reparam_for_thought_embeddings:
                        inputs_embeds = torch.randn(batch_size, seq_len, self.model.config.hidden_size, device=input_ids.device, dtype=cur_thought_embedding.dtype)
                        inputs_embeds = inputs_embeds * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
                        if contains_start:
                            sampled_start = inputs_embeds.clone().detach()
                        else:
                            sampled_end = inputs_embeds.clone().detach()
                    else:
                        inputs_embeds = cur_thought_embedding.unsqueeze(0).repeat(batch_size, seq_len, 1)
                        inputs_embeds = inputs_embeds.view(probabilities.size(0), probabilities.size(1), -1).to(self.model.embed_tokens.weight.dtype)
                inputs_embeds = inputs_embeds.view(probabilities.size(0), probabilities.size(1), -1).to(self.model.embed_tokens.weight.dtype)

                if len(attention_mask.shape) == 2:
                    breakpoint()
                else:
                    original_attention = attention_mask[..., :attention_mask.shape[-2]]
                    if self.use_upper_triangular:
                        new_attention = original_attention
                    else:
                        original_attention = original_attention == attention_mask.max()
                        # Porque eye não está implementado para BF16, precisamos cuidar do caso
                        if not attention_mask.dtype == torch.bfloat16:
                            new_attention = torch.eye(
                                seq_len, dtype=attention_mask.dtype, device=attention_mask.device
                            )
                        else:
                            new_attention = torch.eye(
                                seq_len, dtype=torch.float32, device=attention_mask.device
                            ).to(attention_mask.dtype)

                        new_attention = new_attention.view(1, 1, seq_len, seq_len).repeat(input_ids.shape[0], 1, 1, 1)
                        new_attention = new_attention * original_attention

                        new_attention[new_attention == 0] = attention_mask.min()
                        new_attention[new_attention == 1] = attention_mask.max()

                    # if self.training:
                    attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
                past_key_values = outputs.past_key_values
                position_ids = position_ids + 1

                if labels is not None and (self.n_ahead > 1 or not self.base_original_mode):
                    # Muda para que os tokens < n prevejam n
                    # logits: abcdef -> bcdef? -> cdef??
                    # labels: abcdef -> ?bcdef -> ??cdef
                    if ahead_idx == 0 and self.optimize_lm_head_only_at_start:
                        loss_logits = initial_loss_logits
                    else:
                        loss_logits = logits

                    shift_idx = 1 + max(0, ahead_idx - (self.n_ahead - 1))
                    shift_logits = loss_logits[..., :-shift_idx, :].contiguous()
                    shift_labels = labels[..., shift_idx:].contiguous()

                    # Achata os tokens
                    loss_fct = CrossEntropyLoss(reduction="none")
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)

                    # Habilita o paralelismo do modelo
                    shift_labels = shift_labels.to(shift_logits.device)
                    # if shift_labels.min() == self.tokenizer.pad_token_id:
                    shift_labels = torch.where(shift_labels == self.tokenizer.pad_token_id, -100, shift_labels)
                    unreduced_loss = loss_fct(shift_logits, shift_labels)
                    if torch.any(unreduced_loss != unreduced_loss):
                        raise ValueError("NaN loss")
                    unreduced_loss = unreduced_loss.reshape(logits.shape[0], -1)
                    loss_list.append(unreduced_loss)

                    if self.use_policy_loss and ahead_idx > 0 and (ahead_idx > 1 or not self.use_start_thought_token):
                        # Tratamos a mudança na perda como a recompensa
                        previous_loss = loss_list[-2]
                        if ahead_idx < self.n_ahead - 1:
                            shift_amount = 0
                            original_dqn_reward = (previous_loss - unreduced_loss).detach()
                            if self.first_and_last_mode:
                                original_dqn_reward = original_dqn_reward * 0.0
                        else:
                            shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                            cur_policy_shift_logits = initial_loss_logits[..., shift_amount:-1, :].contiguous().detach()
                            cur_policy_shift_labels = labels[..., 1 + shift_amount:].contiguous()
                            # Achata os tokens
                            cur_policy_loss_fct = CrossEntropyLoss(reduction="none")
                            cur_policy_shift_logits = cur_policy_shift_logits.view(-1, self.config.vocab_size)
                            cur_policy_shift_labels = cur_policy_shift_labels.view(-1).clone()
                            # Habilita o paralelismo do modelo
                            cur_policy_shift_labels[cur_policy_shift_labels == self.tokenizer.pad_token_id] = -100
                            cur_policy_shift_labels = cur_policy_shift_labels.to(cur_policy_shift_labels.device)
                            cur_policy_reward_base_loss = loss_fct(
                                cur_policy_shift_logits, cur_policy_shift_labels.to(cur_policy_shift_logits.device)
                            ).reshape(logits.shape[0], -1)
                            original_dqn_reward = cur_policy_reward_base_loss.detach() - unreduced_loss

                        if not did_skip_sampling:
                            nonzero_indices = prev_probabilities_2d.nonzero()
                            action_loglikelihoods = F.log_softmax(prev_sample_probs / self.reinforce_temperature, dim=-1)[nonzero_indices[:, 0], nonzero_indices[:, 1]]
                            action_loglikelihoods_2d = action_loglikelihoods.reshape(batch_size, -1)[:, :-1 - shift_amount]
                            action_loglikelihoods_list.append(action_loglikelihoods_2d)
                        if policy_reward is None:
                            policy_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                        else:
                            if self.n_ahead_talk > shift_amount:
                                added_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                            else:
                                added_reward = original_dqn_reward
                            policy_reward += added_reward

                    if self.use_policy_loss and ahead_idx == self.n_ahead + self.n_ahead_talk - 2:
                        # Calcula apenas durante a fase de pensamento
                        if self.use_reparam_for_thought_embeddings and (self.use_start_thought_token or self.use_end_thought_token):
                            sampled_start, sampled_end = inputs_embeds.clone().detach()
                            # Calcula a probabilidade logarítmica dos embeddings iniciais e finais amostrados a partir de
                            # uma distribuição normal multivariada com média start_embedding[0] e desvio padrão
                            # start_embedding[1]

                            if self.use_start_thought_token:
                                exp_start_std = torch.exp(start_embedding[1])
                                start_loglikelihood = -0.5 * (sampled_start.detach() - start_embedding[0]) ** 2 / exp_start_std ** 2 - start_embedding[1] - 0.5 * math.log(2 * math.pi)
                                start_loglikelihood = start_loglikelihood.mean(dim=-1)
                            if self.use_end_thought_token:
                                exp_end_std = torch.exp(end_embedding[1])
                                end_loglikelihood = -0.5 * (sampled_end.detach() - end_embedding[0]) ** 2 / exp_end_std ** 2 - end_embedding[1] - 0.5 * math.log(2 * math.pi)
                                end_loglikelihood = end_loglikelihood.mean(dim=-1)

                            # Usamos a média em vez da soma para evitar a dependência da dimensionalidade dos embeddings
                            if self.use_end_thought_token and self.use_policy_loss_for_end_thought:
                                action_loglikelihoods_list.append(end_loglikelihood)
                            if self.use_start_thought_token:
                                action_loglikelihoods_list.append(start_loglikelihood)

                        if ahead_idx == self.n_ahead + self.n_ahead_talk - 2 and self.eval_mode:
                            with torch.no_grad():
                                # Calcula o quantile 0,75 das recompensas
                                filtered_tokens = input_ids[:, :policy_reward.shape[-1]].cpu().detach().numpy().flatten()
                                filtered_tokens_mask = filtered_tokens != self.tokenizer.pad_token_id
                                filtered_tokens = filtered_tokens[filtered_tokens_mask]
                                filtered_rewards = policy_reward.float().cpu().detach().numpy()[:, :seq_len - self.n_ahead_talk].flatten()
                                filtered_rewards = filtered_rewards[filtered_tokens_mask]

                                abs_reward_list = np.abs(policy_reward.float().cpu().detach().numpy()[:, :seq_len - self.n_ahead_talk].flatten())
                                abs_reward_list = abs_reward_list[filtered_tokens_mask]
                                medium_quantile = np.quantile(abs_reward_list, 0.5)
                                upper_quantile = np.quantile(abs_reward_list, 0.95)

                                save_tokens_with_rewards_to_pdf(
                                    filtered_tokens,
                                    [0] + filtered_rewards.tolist(),
                                    self.tokenizer,
                                    output_file=f"rewards_talk_{self.n_ahead_talk}_{self.training_steps}.pdf",
                                    eps=medium_quantile,
                                    eps2=upper_quantile,
                                )

                                def plot_kde(data, losses):
                                    sns.set(style="whitegrid")
                                    # Cria o plot
                                    sns.kdeplot(data, fill=True)
                                    plt.title("KDE Plot")
                                    plt.xlabel("Value")
                                    plt.ylabel("Density")
                                    # Salva e fecha
                                    plt.savefig(f"kde_talk_{self.n_ahead_talk}_{self.training_steps}.pdf")
                                    plt.close()

                                    base_colors = sns.color_palette("light:#5A9", n_colors=256)
                                    base_cmap = LinearSegmentedColormap.from_list("log_light", base_colors)
                                    log_norm = LogNorm(vmin=1e-3, vmax=10)

                                    sns.kdeplot(x=data, y=losses, fill=True, levels=20, norm=log_norm, cut=0, linewidths=0)

                                    # limit y entre 0 e 25 e x entre -1 e 1
                                    plt.xlim(-1, 1)
                                    plt.ylim(0, 25)
                                    plt.savefig(f"jointer_talk_{self.n_ahead_talk}_{self.training_steps}.pdf")
                                    plt.close()

                                self.all_rewards.extend(filtered_rewards)
                                self.all_unreduced_losses.extend(unreduced_loss[:, :-1].flatten()[filtered_tokens_mask].float().flatten().cpu().detach().numpy())
                                plot_kde(self.all_rewards, self.all_unreduced_losses)

                        for action_loglikelihoods_2d in action_loglikelihoods_list:
                            train_policy_reward = policy_reward

                            # Descarta recompensas abaixo da média
                            if self.trice_mode and self.n_passes > 1:
                                batched_policy_reward = train_policy_reward.reshape(-1, self.n_passes, train_policy_reward.shape[-1])
                                # média dos passes
                                train_policy_reward = batched_policy_reward - batched_policy_reward.mean(dim=1, keepdim=True)
                                train_policy_reward = train_policy_reward.reshape(-1, train_policy_reward.shape[-1])

                            if self.subtract_mean_reward:
                                train_policy_reward = train_policy_reward - train_policy_reward.mean()
                            if self.remove_negative_rewards:
                                fixed_policy_reward = train_policy_reward.detach().clamp(min=0)
                            else:
                                fixed_policy_reward = train_policy_reward.detach()
                            actor_loss = -fixed_policy_reward * action_loglikelihoods_2d[:, :policy_reward.shape[-1]].to(policy_reward.device)
                            if action_loglikelihoods_2d.mean() < -1e4 and not self.use_policy_loss_just_for_thoughts:
                                # Isso só acontecerá quando forçarmos o próximo token a ser o token do fim do pensamento
                                break
                            dqn_loss_list.append(actor_loss.mean())

        if loss_list:
            if self.first_and_last_mode:
                loss = sum(
                    self.loss_mean(loss_list[-(i + 1)]) for i in range(self.n_ahead_talk)
                ) * (1 - self.original_loss_weight) / self.n_ahead_talk
                loss = loss + self.loss_mean(loss_list[0]) * self.original_loss_weight
                # Filtrando NaN dos demais
                # por exemplo. se n_ahead_talk = 2 e a lista tiver 5, queremos NaN 1, 2, mas manter 0, 3, 4
                for i in range(1, len(loss_list) - self.n_ahead_talk):
                    loss_list[i] = loss_list[i] * math.nan
            elif self.first_only:
                loss = self.loss_mean(loss_list[0])
            elif self.final_only_mode:
                loss = sum(
                    self.loss_mean(loss_list[-i]) for i in range(1, self.n_ahead_talk + 1)
                ) / self.n_ahead_talk
            else:
                loss = None
                for cur_loss_item in loss_list:
                    cur_loss = self.loss_mean(cur_loss_item)
                    if loss is not None:
                        loss = loss + cur_loss.to(loss.device)
                    else:
                        loss = cur_loss
                loss = loss / len(loss_list)

            loss = loss * self.base_loss_beta

        if dqn_loss_list:
            dqn_loss = sum(dqn_loss_list) / len(dqn_loss_list)
            if self.include_policy_loss:
                if loss is not None:
                    loss += dqn_loss * self.policy_loss_beta
                else:
                    loss = dqn_loss * self.policy_loss_beta

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        base_log_dict = {
            f"loss_{i}": nonzero_mean(loss_list[i]) for i in range(len(loss_list))
        }

        if loss is not None:
            base_log_dict["loss_train"] = loss.item()

        for loss_key, loss_val in base_log_dict.items():
            log_dict[loss_key] += loss_val / self.n_tokens_print

        if self.use_policy_loss and policy_reward is not None:
            log_dict["policy_loss"] += dqn_loss / self.n_tokens_print
            log_dict["policy_reward"] += policy_reward.mean() / self.n_tokens_print

        if not loss_list:
            if loss is not None:
                log_dict["loss_0"] += loss / self.n_tokens_print
        else:
            log_dict["loss_final"] += nonzero_mean(loss_list[-1]) / self.n_tokens_print
            log_dict["loss_talk"] += sum(nonzero_mean(cur_loss_item) for cur_loss_item in loss_list[-self.n_ahead_talk:]) / self.n_ahead_talk / self.n_tokens_print

        # também registra perdas relativas em loss_0
        if loss_list:
            for i, loss_item in enumerate(loss_list):
                talk_idx = min(max(i - (self.n_ahead - 1), 0), len(talk_loss_list) - 1)
                if not talk_loss_list:
                    cur_talk_loss = nonzero_mean(loss_list[0])
                else:
                    cur_talk_loss = talk_loss_list[talk_idx]
                log_dict[f"rel_loss_{i}"] += (nonzero_mean(loss_item) - cur_talk_loss) / self.n_tokens_print
        if self.training:
            self.training_steps += 1

        try:
            if self.training_steps % (self.n_tokens_print) == 0 or not self.training:
                if not self.training:
                    new_log_dict = {}
                    for key in list(log_dict.keys()):
                        new_log_dict["eval_" + key] = log_dict[key]
                    log_dict = new_log_dict

                log_dict["training_steps"] = self.training_steps
                log_dict["batch_size"] = batch_size
                log_dict["example_steps"] = self.training_steps * batch_size * self.gradient_accumulation_steps

                if self.n_ahead > 1:
                    log_dict["compute_steps"] = self.training_steps * batch_size * (self.n_ahead + self.n_ahead_talk - 1) * self.gradient_accumulation_steps
                else: # Não há sobrecarga para tokens de conversação se não houver pensamento
                    log_dict["compute_steps"] = self.training_steps * batch_size * self.gradient_accumulation_steps
                # Remove todos os NaN
                for key in list(log_dict.keys()):
                    if log_dict[key] != log_dict[key]:
                        del log_dict[key]
                if self.training:
                    # Criando e exibindo o relatório
                    # report = "\nQuiet-STaR Training\n"
                    # for key, value in log_dict.items():
                    #     report += f"{key}: {value}\n"
                    print(log_dict['loss_0'])
                    self.log_dict = defaultdict(int)
                else:
                    self.eval_log_dict = defaultdict(int)
        except Exception as e:
            pass

        if not self.training:
            self.n_ahead_talk = n_ahead_talk_to_restore
            self.n_passes = n_passes_to_restore

        return AuraMoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=output_attentions,
            router_logits=output_router_logits,
        )
