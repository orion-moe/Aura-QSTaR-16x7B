# pylint: disable=too-many-arguments,too-many-branches,too-many-statements,too-many-locals,fixme,too-many-instance-attributes,protected-access,not-callable,too-many-lines,line-too-long,abstract-method,unused-variable,missing-class-docstring,arguments-differ,missing-function-docstring,forgotten-debug-statement,broad-exception-caught,import-error,no-name-in-module,relative-beyond-top-level,too-many-nested-blocks,invalid-unary-operand-type
"""
    Aura Model com:
    - Quiet-STaR (https://arxiv.org/abs/2403.09629)
    - Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts (https://arxiv.org/abs/2401.02731)
"""
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

class AuraPreTrainedModel(PreTrainedModel):
    """ AuraPreTrainedModel """
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

num_token_gen = 0
class AuraModel(AuraPreTrainedModel):
    """
        Decodificador de Transformer que consiste em camadas *config.num_hidden_layers*.
        Cada camada é um [`AuraDecoderLayer`]
    """
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
        # Inicializa os pesos e aplica o processamento final
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
        """ AuraModel Forward """
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

        if input_ids is not None:
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
                    "`use_cache=True` é incompatível com checkpoint de gradiente. Redefinindo `use_cache=False`..."
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
                    " isso pode levar a um comportamento inesperado para a versão FlashAttention da Aura. Certifique-se de "
                    " chamar `tokenizer.padding_side = 'left'` antes de tokenizar a entrada."
                )

        if self._attn_implementation == "flash_attention_2":
            # A máscara 2d é passada pelas camadas
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        # Desativado para implementar Quiet-STaR
        elif self._attn_implementation == "sdpa" and not output_attentions and self.training:
        # força um False aqui.
        # elif self._attn_implementation == "sdpa" and not output_attentions and attention_mask.dim() == 2 and False:
			# output_attentions=True não pode ser suportado ao usar SDPA, e recorremos
            # a implementação manual que requer uma máscara causal 4D em todos os casos.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        elif attention_mask is None or attention_mask.dim() == 2 or self.training:
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

                # Desativa para testar Quiet-STaAR
                # layer_outputs = torch.utils.checkpoint.checkpoint(
                #     create_custom_forward(decoder_layer),
                #     hidden_states,
                #     attention_mask,
                #     position_ids,
                #     None,
                # )
                layer_outputs = self._gradient_checkpointing_func(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
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

        global num_token_gen
        num_token_gen += 1

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
    """ AuraForCausalLM Class """
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

        tokenizer = AutoTokenizer.from_pretrained("/ors/models/Aura-16x7B-QuietSTaR")
        tokenizer.padding_side = "right"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        special_tokens_to_add = ["<|im_start|>", "<|im_end|>"]
        special_tokens_to_add.append("<|th_start|>")
        special_tokens_to_add.append("<|th_end|>")
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        self.resize_token_embeddings(len(tokenizer))

        self.tokenizer = tokenizer
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
        self.kill_after = 100

        self.start_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))
        self.end_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))

        self.policy_loss_beta = 1e6
        self.embedding_scale = 1e2
        self.reinforce_temperature = 3
        self.base_loss_beta = 1

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

    @torch.no_grad()
    def infer(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        batch_size, seq_len = input_ids.shape

        # Salva os input_ids e attention_mask originais para uso posterior
        original_input_ids = input_ids.clone()
        original_attention_mask = attention_mask.clone() if attention_mask is not None else None

        # Anexa o token de pensamento inicial à sequência de entrada
        start_thought_token_id = self.tokenizer.convert_tokens_to_ids("<|th_start|>")
        input_ids = torch.cat([input_ids, torch.tensor([[start_thought_token_id]] * batch_size).to(input_ids.device)], dim=-1)
        seq_len += 1

        # Atualiza attention_mask
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1)).to(attention_mask.device)], dim=-1)

        # Gera a continuação
        continuation_length = self.n_ahead - 2
        new_key_values = past_key_values

        start_time = time.time()
        for continuation_idx in range(continuation_length):
            outputs = self.model(
                input_ids=input_ids if continuation_idx == 0 else next_token_id.unsqueeze(-1).to(input_ids.device),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=new_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            new_key_values = outputs.past_key_values

            hidden_states = outputs[0]

            logits = self.lm_head(hidden_states)
            logits = logits[:, -1, :]  # Considera apenas o último token

            # Aplica Gumbel-Softmax aos logits
            next_token_logits = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=True, dim=-1)
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # Anexa o token gerado à sequência de entrada
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1).to(input_ids.device)], dim=-1)
            seq_len += 1

            # Atualiza attention_mask
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1)).to(attention_mask.device)], dim=-1)

        # Anexa o token de pensamento final à sequência de entrada
        end_thought_token_id = self.tokenizer.convert_tokens_to_ids("<|th_end|>")
        input_ids = torch.cat([input_ids, torch.tensor([[end_thought_token_id]] * batch_size).to(input_ids.device)], dim=-1)
        seq_len += 1

        # Atualize a attention_mask
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1)).to(attention_mask.device)], dim=-1)

        # Pega os estados ocultos antes e depois do pensamento
        outputs_before = self.model(
            input_ids=original_input_ids,
            attention_mask=original_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states_before = outputs_before[0][:, -1:, :]

        # Dois novos tokens: último token de continuação e token de pensamento final
        outputs_after = self.model(
            input_ids=torch.cat([next_token_id.unsqueeze(-1).to(input_ids.device), torch.tensor(end_thought_token_id).unsqueeze(-1).unsqueeze(-1).to(input_ids.device)], dim=-1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=new_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states_after = outputs_after[0][:, -1:, :]

        # Aplica o talk_head para obter o peso da mistura
        mixing_weight = self.talk_head[0](torch.cat([hidden_states_before, hidden_states_after], dim=-1))

        # Aplica o peso de mistura aos estados ocultos
        mixed_hidden_states = (1 - mixing_weight) * hidden_states_before + mixing_weight * hidden_states_after

        # Aplica o lm_head de linguagem para obter os logits finais
        logits = self.lm_head(mixed_hidden_states)
        return logits

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
        log_dict = self.log_dict if self.training else self.eval_log_dict

        #if self.training and self.kill_after is not None and self.training_steps // self.gradient_accumulation_steps > self.kill_after:
            #raise ValueError("Killed after")

        if not self.training:
            n_ahead_talk_to_restore = self.n_ahead_talk
            n_passes_to_restore = self.n_passes
            self.n_ahead_talk = 1
            self.n_passes = 1

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

        assert self.cumulative_residual or self.clever_residual or self.skip_residual or self.no_residual
        assert not (self.skip_residual and self.use_policy_loss)

        if self.tokenized_thought_prefix is None and self.use_thought_prefix:
            self.tokenized_thought_prefix = self.tokenizer(
                self.thought_prefix,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"]

        def apply_head(head, states, detach=False):
            if detach:
                head_weight = head.weight.detach()
            else:
                head_weight = head.weight
            head_weight = head_weight.to(states.device)
            return (head_weight @ states.transpose(-1, -2)).transpose(-1, -2).contiguous()

        def idx_if_sequential(head, idx=0):
            if isinstance(head, nn.Sequential) or isinstance(head, nn.ModuleList):
                return idx_if_sequential(head[idx], idx=idx)
            return head

        def none_repeat_interleave(x, n):
            if x is None:
                return x
            return x.repeat_interleave(n, dim=0)

        if self.n_passes > 1:
            input_ids = none_repeat_interleave(input_ids, self.n_passes)
            attention_mask = none_repeat_interleave(attention_mask, self.n_passes)
            position_ids = none_repeat_interleave(position_ids, self.n_passes)
            inputs_embeds = none_repeat_interleave(inputs_embeds, self.n_passes)
            labels = none_repeat_interleave(labels, self.n_passes)
            if past_key_values is not None:
                past_key_values = [none_repeat_interleave(p, self.n_passes) for p in past_key_values]
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

        loss = None
        prev_rm_tokens = None
        cur_rm_tokens = None
        prev_rm_logits = None
        prev_sample_probs = None
        did_skip_sampling = None
        skip_sampling = None
        sample_probs = None
        hidden_states = None
        logits = None
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

            hidden_states = outputs[0].to(self.lm_head.weight.dtype)
            prev_hidden_states = hidden_states
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
                        residual_logits = apply_head(
                            self.lm_head,
                            residual_logits,
                            detach=self.optimize_lm_head_only_at_start
                        )
                    residual_logits = residual_logits.to(logits.device)
                    if self.use_weighted_talk_head:
                        # Combina cur_base_hidden com talk_hidden_states de acordo com o head ponderado
                        residual_logits = cur_base_hidden * (1 - residual_logits) + talk_hidden_states * residual_logits
                        residual_logits = apply_head(self.lm_head, residual_logits, detach=self.optimize_lm_head_only_at_start)

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

                #aux_loss = None
                #if output_router_logits:
                #    aux_loss = load_balancing_loss_func(
                #        outputs.router_logits if return_dict else outputs[-1],
                #        self.config.num_experts,
                #        self.config.topk,
                #    )
                #    if labels is not None:
                #        # Verifica o dispositivo de 'loss' e move 'aux_loss' para o mesmo dispositivo
                #        device = loss.device
                #        aux_loss = aux_loss.to(device)
                #        loss += self.config.router_aux_loss_coef * aux_loss

            if not attempted or self.comparison_mode:
                rm_hidden_states = hidden_states
                rm_logits = apply_head(self.lm_head, rm_hidden_states, detach=self.optimize_lm_head_only_at_start)

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
                    thought_token = self.tokenizer.convert_ids_to_tokens(thought_id)
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

                    if self.training:
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
                    print("unreduced_loss", unreduced_loss)
                    if torch.any(unreduced_loss != unreduced_loss):
                        raise ValueError("NaN loss")
                    unreduced_loss = unreduced_loss.reshape(logits.shape[0], -1)
                    loss_list.append(unreduced_loss)

                    if self.use_policy_loss and ahead_idx > 0 and (ahead_idx > 1 or not self.use_start_thought_token):
                        # Tratamos a mudança na perda como a recompensa
                        previous_loss = loss_list[-2]
                        # por exemplo, suponha que n_ahead = 3 e n_ahead_talk = 2
                        # observe que terminamos em self.n_ahead + self.n_ahead_talk - 2
                        # neste caso, 5 - 2 = 3, então terminamos em ahead_idx = 3
                        # também prevemos o próximo token em ahead_idx = 2
                        # quando chegamos em ahead_idx = 2, prevemos adiante
                        # então mudamos em 1
                        # isso é ahead_idx = n_ahead - 1
                        # quando chegamos em ahead_idx = 3, prevemos adiante
                        # então mudamos por 2
                        # e isso é ahead_idx = n_ahead
                        if ahead_idx < self.n_ahead - 1:
                            shift_amount = 0
                            original_dqn_reward = (previous_loss - unreduced_loss).detach()
                            if self.first_and_last_mode:
                                original_dqn_reward = original_dqn_reward * 0.0
                        else:
                            # logits x cur_policy_shift_logits
                            # vamos dar uma olhada em rm_logits e prev_rm_logits
                            shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                            # digamos shift_amount = 2
                            # abcdefg -> bcdefg? -> cdefg??
                            # logits = [a b]c d e f[g]
                            # labels = [a b c]d e f g
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
            # if self.training_steps % (self.gradient_accumulation_steps * 256) == 0:
            # if self.training_steps % (self.n_tokens_print) == 0 or not self.training and "0" in str(loss.device):
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
                    report = "Quiet-STaR Training\n---------------\n"
                    for key, value in log_dict.items():
                        report += f"{key}: {value}\n"
                    print(report)
                    self.log_dict = defaultdict(int)
                else:
                    self.eval_log_dict = defaultdict(int)
        except Exception as e:
            print("🚨 EXCEPTION! 🚨", e)

        if not self.training:
            self.n_ahead_talk = n_ahead_talk_to_restore
            self.n_passes = n_passes_to_restore

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            print('---------- 🚨 not return_dict 🚨 ----------')
            return (loss,) + output if loss is not None else output

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.config.num_experts,
                self.config.topk,
            )
            if labels is not None:
                # Verifica o dispositivo de 'loss' e move 'aux_loss' para o mesmo dispositivo
                device = loss.device
                aux_loss = aux_loss.to(device)
                loss += self.config.router_aux_loss_coef * aux_loss

        return AuraMoECausalLMOutputWithPast(
            loss=loss if loss is not None else None,
            logits=(rm_logits if self.n_ahead > 1 else logits) if not self.output_logits_at_the_end else logits,
            aux_loss=aux_loss if aux_loss is not None else 0.0,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Omite tokens cobertos por past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

			# Mantém apenas os tokens não processados:
            # 1 - Se o comprimento da máscara de atenção exceder o comprimento de input_ids, então estamos em um cenário
            # onde algumas das entradas são passadas exclusivamente como parte do cache (por exemplo, ao passar
            # input_embeds como entrada)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - Se past_length for menor que input_ids', então input_ids contém todos os tokens de entrada.
            # Podemos descartar input_ids baseado em past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Caso contrário (past_length >= input_ids.shape[1]), vamos assumir que input_ids possui apenas tokens
            # não processados.

            # Se estivermos prestes a ultrapassar o comprimento máximo do cache e max_cache_length for especificado,
            # precisaremos cortar a máscara de atenção de entrada.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        # if attention_mask is not None and position_ids is None:
        #     # Cria position_ids dinamicamente para geração em lote
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # Se `inputs_embeds` forem passados, usaremos apenas na etapa de primeira geração
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

class AuraForSequenceClassification(AuraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = AuraModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Não é possível manipular tamanhos de lote > 1 se nenhum token de preenchimento estiver definido.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # Se nenhum PAD Token for encontrado, usar o módulo ao invés de fazer uma
                # indexação reversa para compatibilidade de ONNX
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
