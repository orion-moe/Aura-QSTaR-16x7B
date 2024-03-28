""" AuraModel Trainer """
import os
import logging
import time
import random

import torch
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model

import pandas as pd

from model.modeling_aura import AuraForCausalLM
from utils.eval_helpers import (
    #preprocess_eval_function_csqa,
    preprocess_eval_function_gsm,
    preprocess_function,
    compute_metrics
)
from utils.utils import (
    print_linear_trainable_parameters
)

from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True

AHEAD_TALK_GLOBAL = 16
PASSES_GLOBAL = 8
AHEAD_GLOBAL = 32
EXAMPLES = 1_000
FULL_BATCH_SIZE = 64
BATCH_SIZE = FULL_BATCH_SIZE // PASSES_GLOBAL
GLOBAL_GRADIENT_ACCUMULATION_STEPS = FULL_BATCH_SIZE // BATCH_SIZE
ROOT_PREFIX = "/ors/"
TMP_PATH = ROOT_PREFIX + "tmp/"

os.environ["WANDB_PROJECT"] = "Aura-QSTaR-16x7B"
os.environ["WANDB_CACHE_DIR"] = "/ors/tmp"

# Configurações de logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def extract_model_info(model):
    """ Exibe os detalhes da estrutura do modelo """

    model_info = {
        "Nome": model.__class__.__name__,
        "Total de parâmetros": sum(p.numel() for p in model.parameters()),
        "Parâmetros treináveis": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "Detalhes das camadas": [],
    }

    for name, module in model.named_modules():
        layer_info = {
            "Nome da camada": name,
            "Tipo": module.__class__.__name__,
            "Parâmetros": sum(p.numel() for p in module.parameters(recurse=False)),
            "Treináveis": any(p.requires_grad for p in module.parameters(recurse=False)),
        }
        model_info["Detalhes das camadas"].append(layer_info)

    layers_df = pd.DataFrame(model_info["Detalhes das camadas"])
    layers_df_string = layers_df.to_string()

    print(f"Nome do Modelo: {model_info['Nome']}")
    print(f"Total de Parâmetros: {model_info['Total de parâmetros']}")
    print(f"Parâmetros Treináveis: {model_info['Parâmetros treináveis']}")
    print("\nDetalhes das Camadas:")
    print(layers_df_string)

def model_init(params):
    """ Incializa o modelo para treinamento """
    original = False

    if params is None:
        params = {}
    else:
        params = params.params

    # Salva os parâmetros
    n_ahead = params.get("n_ahead", AHEAD_GLOBAL if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", AHEAD_TALK_GLOBAL if not original else 1)
    n_passes = params.get("n_passes", PASSES_GLOBAL if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", GLOBAL_GRADIENT_ACCUMULATION_STEPS)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = "/ors/models/Aura-16x7B-QuietSTaR"

    print("Carregando modelo...")
    model = AuraForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        cache_dir=TMP_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        moe_dtype = "bfloat16",
        adapter_dim=512,
        topk=4,
        moe_scaling=1,
        num_experts=16
    )
    #extract_model_info(model)
    print("O modelo foi carregado com sucesso!")

    print("Configurando adaptadores LoRA...")
    lora_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]

    model.adapter_dim = 512
    model.topk = 4
    model.moe_scaling = 1
    model.num_experts = 16

    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=lora_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Adaptadores LoRA configurados!")
    print("Inicializando o Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Adicionando tokens especiais para chat: <|im_start|>, <|im_end|>...")
    special_tokens_to_add = ["<|im_start|>", "<|im_end|>"]

    if model.use_start_thought_token:
        print("Adicionando token especial do Quiet-STaR: <|th_start|>...")
        special_tokens_to_add.append("<|th_start|>")
    if model.use_end_thought_token:
        print("Adicionando token especial do Quiet-STaR: <|th_end|>...")
        special_tokens_to_add.append("<|th_end|>")

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    print("Tokens especiais adicionados ao Tokenizer com sucesso. Redimensionando...")
    model.resize_token_embeddings(len(tokenizer))
    print("Tokenizer redimensionado com sucesso! Tamanho atual do Tokenizer:", len(tokenizer))
    print("Os tokens especiais que foram adicionados foram:")
    for token in special_tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"Token: {token}, ID: {token_id}")

    print("Criando o modelo PEFT...")
    model.tokenizer = tokenizer
    model = get_peft_model(model, config)
    #extract_model_info(model)
    print("Modelo PEFT criado com sucesso!")

    print("Concluíndo configuração do modelo...")
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.moe_dtype = "bfloat16"
    model.adapter_dim = 512
    model.topk = 4
    model.moe_scaling = 1
    model.num_experts = 16

    model.wandb_enabled = True
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    #extract_model_info(model)
    print("Modelo configurado e pronto para treinamento! Iniciando treino...")

    print_linear_trainable_parameters(model)

    model.train()

    return model

def has_checkpoints(_dir):
    dir_list = os.listdir(_dir)
    for item in dir_list:
        full_path = os.path.join(_dir, item)
        if os.path.isdir(full_path) and item.startswith('checkpoint-'):
            return True
    return False

if __name__ == "__main__":
    MODEL_NAME = "Aura-QSTaR"
    OUTPUT_DIR = ROOT_PREFIX + "workdir/" + MODEL_NAME
    os.environ["WANDB_PROJECT"] = MODEL_NAME
    os.environ["WANDB_CACHE_DIR"] = TMP_PATH + "/wandb_cache"

    TRAIN_DATASET = 'teknium/OpenHermes-2.5'

    print("Preparando dataset de treinamento...")
    dataset = load_dataset(
        TRAIN_DATASET,
        "default",
        #split=f"train[:{EXAMPLES}]",
        split="train",
        verification_mode="no_checks",
        num_proc=16,
        cache_dir=TMP_PATH + "datasets/",
    )

    print("Preparando datasets de avaliação...")
    eval_dataset_gsm = load_dataset(
        "gsm8k",
        "main",
        split="test",
        verification_mode="no_checks"
    ).map(
        preprocess_eval_function_gsm,
        batched=True,
        writer_batch_size=512
    )
    # eval_dataset_csqa = load_dataset(
    #     "tau/commonsense_qa",
    #     "default",
    #     split="validation",
    #     verification_mode="no_checks"
    # ).map(
    #     preprocess_eval_function_csqa,
    #     batched=True,
    #     writer_batch_size=200
    # )

    RANDON_SEED = 42
    random.seed(RANDON_SEED)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(RANDON_SEED)

    print("Embaralhando dataset de treinamento...")
    eval_dataset_gsm = eval_dataset_gsm.select(range(min(500, len(eval_dataset_gsm))))

    train_dataset = dataset.shuffle(seed=RANDON_SEED).map(
        preprocess_function,
        batched=True,
        writer_batch_size=512
    )
    eval_datasets = {
        "gsm8k": eval_dataset_gsm,
        # "csqa": eval_dataset_csqa,
    }

    print("Todos os datasets foram preparados com sucesso!")
    
    print("Configurando inicialização do treinamento...")
    trainer = Trainer(
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=2e-04,
            optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_8bit",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GLOBAL_GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=2e-04,
            warmup_steps=25,
            auto_find_batch_size=True,
            weight_decay=0.001,
            label_names=["labels"],
            lr_scheduler_type="cosine",
            include_inputs_for_metrics=True,
            logging_steps=1,
            eval_steps=100_000,
            evaluation_strategy="steps",
            save_steps=100
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=compute_metrics,
        model_init=model_init
    )

    print("Inicialização configurada!")
    print("Iniciando o treinamento...")

    trainer.train(resume_from_checkpoint=has_checkpoints(OUTPUT_DIR))
