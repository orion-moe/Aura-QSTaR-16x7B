# pylint: disable=broad-exception-caught,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,unused-argument,line-too-long,logging-fstring-interpolation,no-member,protected-access
""" Utils """
import os
from os.path import join
import io
import json
import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import bitsandbytes as bnb

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="UTF-8")
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="UTF-8")
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """ Faz dump em um arquivo no formato json. """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def _make_r_io_base(f, mode):
    """Abre o arquivo se f for um caminho de arquivo."""
    if isinstance(f, str):
        return open(f, mode, encoding="UTF-8")
    return f  # Assume que f já é um objeto de arquivo

def jloadl(f, mode="r"):
    """Carrega um arquivo .jsonl em uma lista de dicionários."""
    f = _make_r_io_base(f, mode)
    jlist = [json.loads(line) for line in f]
    f.close()
    return jlist

def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenizar uma lista de strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Pré-processe os dados por tokenização."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Conjunto de dados para ajuste fino supervisionado."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.info("Carregando dataset: %s", data_path)
        data_list = jloadl(data_path)

        logging.info("Pre-processando o dataset")
        self.tokenizer = tokenizer
        self.sources = []
        self.targets = []

        for data in data_list:  # Usando diretamente os itens da lista
            corpus = data["corpus"] if "corpus" in data else ""
            if corpus != "":
                source = f"{tokenizer.bos_token}"
                self.sources.append(source)

                target = f"{corpus}{tokenizer.eos_token}"
                self.targets.append(target)
            else:
                instruction = "Você é uma assistente de inteligência artificial chamada Aura, desenvolvida e treinada pela Orion Research em 2024. Você está em uma conversa com um humano, deve respondê-lo da melhor forma possível, sendo concisa e clara em suas respostas sem informações repetitivas. Nunca abra mão dos princípios éticos e legais e seja simpática."
                conversation = data["conversations"]
                source = f"{tokenizer.bos_token}<|im_start|>system\n{instruction}<|im_end|>"

                for conv in conversation[:-1]:
                    if len(conversation) > 2:
                        if conv['from'] == "human":
                            source += f"<|im_start|>user\n{conv['value']}<|im_end|>"
                        elif conv['from'] == "gpt":
                            source += f"<|im_start|>assistant\n{conv['value']}<|im_end|>"
                    else:
                        if conv['from'] == "human":
                            source += f"<|im_start|>user\n{conv['value']}<|im_end|>"

                self.sources.append(source)
                target = f"<|im_start|>assistant\n{conversation[-1]['value']}{tokenizer.eos_token}"
                self.targets.append(target)

        logging.info("Há um total de %s exemplos no dataset", len(self.sources))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        source = [self.sources[i]]
        target = [self.targets[i]]
        data_dict = preprocess(source, target, self.tokenizer)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset():
    """Agrupa exemplos para finetune supervisionado."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def register_gradient_hooks(model, writer, step):
    """ Registra um hook para cada parâmetro """
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.register_hook(lambda grad, name=name: writer.add_histogram(f"gradients/{name}", grad, step))

class SavePeftModelCallback(TrainerCallback):
    """ Salva o modelo por passos """
    def save_model(self, args, state, kwargs):
        """ Salva o modelolva """
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        moe_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                moe_state.update({param_tensor: model.state_dict()[param_tensor]})
        moe_model_path = os.path.join(checkpoint_folder, "moe_model.bin")

        torch.save(moe_state, moe_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a", encoding="utf-8"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)

class MonitoringCallback(TrainerCallback):
    """Callback para monitorar detalhes durante o treinamento."""

    def on_step_end(self, args, state, control, **kwargs):
        """Executado ao final de cada step de treinamento."""
        super().on_step_end(args, state, control, **kwargs)
        logs = kwargs.get("logs")  # Tentativa de obter logs que podem conter a perda
        if logs and "loss" in logs:
            loss = logs["loss"]
        else:
            loss = "N/A"  # Caso a perda não esteja disponível
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        grad_norm = self.calculate_gradient_norm(model)
        initial_lr = optimizer.param_groups[0]['initial_lr']
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Loss: {loss} || Step: {state.global_step} || Epoch: {state.epoch} || Learning-Rate Inicial: {initial_lr} || Learning-Rate Atual: {current_lr} || Gradient Norm: {grad_norm}")

    def calculate_gradient_norm(self, model):
        """Calcula a norma média dos gradientes."""
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None]
        for param in parameters:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        mean_norm = total_norm / len(parameters) if parameters else 0
        return mean_norm

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """ Cria conjunto de dados e agrupamento para fine tune supervisionado. """
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def find_all_linear_names(model, bits=4):
    """ Procura todos os nomes de módulos lineares para usar no treinamento LoRA """
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # necessário para 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def print_trainable_parameters(model):
    """ Exibe o número de parâmetros treináveis do modelo """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logging.info("Parâmetros treináveis: %s || Total: %s || Treinável: %s", trainable_params, all_param, (100 * trainable_params / all_param))

def print_linear_trainable_parameters(model):
    """ Exibe o número de parâmetros lineares treináveis do modelo """
    linear_trainable_params = 0
    for name, param in model.named_parameters():
        if 'linear' in name and param.requires_grad:
            linear_trainable_params += param.numel()

    print(f"Parâmetros lineares treináveis: {linear_trainable_params}")

def find_latest_checkpoint(checkpoint_dir):
    """Encontra o checkpoint mais recente no diretório especificado."""
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint")]
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)  # Retorna o checkpoint mais recente
    return None

def load_checkpoint(model, optimizer, checkpoint_path):
    """Carrega o modelo e o otimizador do checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
