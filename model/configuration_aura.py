# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
""" Aura model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AuraConfig(PretrainedConfig):
    """ AuraConfig """
    model_type = "aura"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32032,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 8, # * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=32000,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        moe_dtype="bfloat16",
        moe_scaling=1.0,
        num_experts=16,
        topk=4,
        output_router_logits=False,
        adapter_dim=512,
        adapter_dropout=0.0,
        router_aux_loss_coef=0.01,
        max_thoughts=16,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.moe_dtype = moe_dtype
        self.moe_scaling = moe_scaling
        self.num_experts = num_experts
        self.topk = topk
        self.output_router_logits = output_router_logits

        self.adapter_dim = adapter_dim
        self.adapter_dropout = adapter_dropout
        self.router_aux_loss_coef = router_aux_loss_coef

        self.max_thoughts = max_thoughts
        self.merged_talk_heads = merged_talk_heads
        self.merged_lm_and_talk_heads = merged_lm_and_talk_heads
        self.merged_lm_and_think_heads = merged_lm_and_think_heads
        self.use_concat_talk_head = use_concat_talk_head
        self.use_shallow_think = use_shallow_think
        self.use_shallow_talk = use_shallow_talk
        self.use_complex_think_head = use_complex_think_head
        self.use_complex_talk_head = use_complex_talk_head
        self.use_weighted_talk_head = use_weighted_talk_head

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
