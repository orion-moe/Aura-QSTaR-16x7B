# pylint: disable=broad-exception-caught,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,unused-argument,line-too-long,logging-fstring-interpolation,no-member,protected-access
""" Funções auxilares para o Transformers """
import os
import collections
import re
import shutil
import tempfile
import gc
from copy import deepcopy

import torch

from transformers.utils import logging
from transformers.utils.bitsandbytes import set_module_quantized_tensor_to_device
from transformers.pytorch_utils import id_tensor_storage
from transformers.modeling_utils import (
    set_initialized_submodules,
    expand_device_map,
    _load_state_dict_into_model,
    get_disk_only_shard_files,
    load_state_dict,
    _load_state_dict_into_meta_model
)
from accelerate.utils import (
    find_tied_parameters,
    set_module_tensor_to_device,
    load_offloaded_weights,
    save_offload_index
)

logger = logging.get_logger(__name__)

def clone_layers(model, clone_range, insert_range):
    """ Clone Layers """
    print(f"Clonando layers {clone_range} e inserindo-os na posição {insert_range}.")
    try:
        start, end = clone_range
        end += 1
        cloned_layers = [deepcopy(model.model.layers[i]) for i in range(start, end)]

        for i, cloned_layer in enumerate(cloned_layers):
            model.model.layers.insert(insert_range + i, cloned_layer)

        return model
    except Exception as e:
        print(f"Ocorreu um erro ao clonar o layer: {e}")
        raise e

def remove_layers(model, remove_range):
    """ Remove Layers """
    print(f"Removendo layers no intervalo {remove_range}.")
    try:
        start, end = remove_range
        end += 1
        diff = end - start
        for _ in range(diff):
            del model.model.layers[start]
        return model
    except Exception as e:
        print(f"Ocorreu um erro ao remover layer: {e}")
        raise e

## Não converter o adaptador MoE da Aura em Low Bit Linear
def get_keys_to_not_convert(model):
    """
        Uma função utilitária para obter a chave do módulo para manter a precisão total se houver
        Por exemplo para módulos CausalLM podemos querer manter lm_head com total precisão por
        razões de estabilidade numérica. Para outras arquiteturas, queremos para manter os pesos
        amarrados do modelo. A função retornará uma lista das chaves dos módulos para não
        converter em int8.

        Parâmetros:
        model (`torch.nn.Module`):
            O modelo para o qual queremos obter as chaves dos módulos para não converter.
    """
    # Crie uma cópia do modelo e amarre os pesos, depois
    # verifica se contém pesos amarrados

    # isso tem custo 0, pois é feito dentro do gerenciador de contexto `init_empty_weights`
    tied_model = deepcopy(model)
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)

    # Para a compatibilidade do Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # Verifica se é um modelo base
    is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignora se for um modelo base
    if (not has_tied_params) and is_base_model:
        return []

    adapter_module = []
    for n, _ in model.named_parameters():
        if 'adapter' in n:
            adapter_module.append(n)

    # caso contrário, eles têm uma cabeça anexada
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # adiciona o último módulo junto com os pesos amarrados
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection) + adapter_module

    # remove ".weight" das chaves
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    # print(filtered_module_names)
    return filtered_module_names

@classmethod
def _load_pretrained_model(
    cls,
    model,
    state_dict,
    loaded_keys,
    resolved_archive_file,
    pretrained_model_name_or_path,
    ignore_mismatched_sizes=False,
    sharded_metadata=None,
    _fast_init=True,
    low_cpu_mem_usage=False,
    device_map=None,
    offload_folder=None,
    offload_state_dict=None,
    dtype=None,
    is_quantized=False,
    keep_in_fp32_modules=None,
    **kwargs
):
    is_safetensors = False

    if device_map is not None and "disk" in device_map.values():
        archive_file = (
            resolved_archive_file[0] if isinstance(
                resolved_archive_file,
                (list, tuple)) else resolved_archive_file
            )
        is_safetensors = archive_file.endswith(".safetensors")
        if offload_folder is None and not is_safetensors:
            raise ValueError(
                "O `device_map` atual teve pesos descarregados para o disco."
                " Forneça um `offload_folder` para eles. Alternativamente,"
                " certifique-se de ter `safetensors` instalados se o modelo que"
                " você está usando oferece os pesos neste formato."
            )
        if offload_folder is not None:
            os.makedirs(offload_folder, exist_ok=True)
        if offload_state_dict is None:
            offload_state_dict = True

    is_sharded_safetensors = is_safetensors and sharded_metadata is not None

    # Recuperar chaves ausentes e inesperadas
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    prefix = model.base_model_prefix

    def _fix_key(key):
        if "beta" in key:
            return key.replace("beta", "bias")
        if "gamma" in key:
            return key.replace("gamma", "weight")
        return key

    original_loaded_keys = loaded_keys
    loaded_keys = [_fix_key(key) for key in loaded_keys]

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # operações de renomeação de chaves nunca são feitas nas chaves
    # que são carregados, mas sempre nas chaves do modelo recém-inicializado
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    if remove_prefix_from_model:
        _prefix = f"{prefix}."
        expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
        expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
    elif add_prefix_to_model:
        expected_keys = [".".join([prefix, s]) for s in expected_keys]

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = set(loaded_keys) - set(expected_keys)

    # Remove buffers não persistentes de chaves inesperadas: eles não estão no estado dict,
    # mas estarão em model_buffers
    model_buffers = { n for n, _ in model.named_buffers() }
    if remove_prefix_from_model:
        model_buffers = {
            key[len(_prefix) :] if key.startswith(_prefix) else key for key in model_buffers
        }
    elif add_prefix_to_model:
        model_buffers = {".".join([prefix, key]) for key in model_buffers}
    unexpected_keys = list(unexpected_keys - model_buffers)

    model.tie_weights()
    ptrs = collections.defaultdict(list)
    for name, tensor in model.state_dict().items():
        id_tensor = id_tensor_storage(tensor) if tensor.device != torch.device("meta") else id(tensor)
        ptrs[id_tensor].append(name)

    # Esses são todos os ponteiros de tensores compartilhados.
    tied_params = [names for _, names in ptrs.items() if len(names) > 1]

    for group in tied_params:
        if remove_prefix_from_model:
            group = [key[len(_prefix) :] if key.startswith(_prefix) else key for key in group]
        elif add_prefix_to_model:
            group = [".".join([prefix, key]) for key in group]
        missing_in_group = [k for k in missing_keys if k in group]
        if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
            missing_keys = [k for k in missing_keys if k not in missing_in_group]

    # Alguns modelos podem ter chaves que não estão no estado projetado, removendo-as
    # antes de avisar desnecessariamente o usuário.
    if cls._keys_to_ignore_on_load_missing is not None:
        for pat in cls._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if cls._keys_to_ignore_on_load_unexpected is not None:
        for pat in cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    # Recupera pesos no meta-device e os põe de volta na CPU.
    # Isso não é ideal em termos de memória, mas se não fizermos isso, não poderemos
    # inicializá-los na próxima etapa
    if low_cpu_mem_usage:
        for key in missing_keys:
            if key not in list(model_state_dict.keys()) and f"{prefix}.{key}" in list(model_state_dict.keys()):
                key = f"{prefix}.{key}"
            elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict.keys()):
                key = ".".join(key.split(".")[1:])

            param = model_state_dict[key]

            # upcast em fp32 se houver
            target_dtype = dtype
            if (
                keep_in_fp32_modules is not None
                and dtype == torch.float16
                and any(module_to_keep_in_fp32 in key for module_to_keep_in_fp32 in keep_in_fp32_modules)
            ):
                target_dtype = torch.float32

            if param.device == torch.device("meta"):
                if not is_quantized:
                    set_module_tensor_to_device(model, key, "cpu", torch.empty(*param.size(), dtype=target_dtype))
                else:
                    set_module_quantized_tensor_to_device(
                        model, key, "cpu", torch.empty(*param.size(), dtype=target_dtype)
                    )

    # Recupera módulos não inicializados e inicializa antes de talvez substituí-los pelos pesos pré-treinados.
    if _fast_init:
        if remove_prefix_from_model:
            _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
        elif add_prefix_to_model:
            _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
        else:
            _loaded_keys = loaded_keys
        set_initialized_submodules(model, _loaded_keys)
        # Isso inicializará apenas os submódulos que não estão marcados como inicializados pela linha acima.
        model.apply(model._initialize_weights)

    # Define alguns módulos para fp32, se houver
    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                param = param.to(torch.float32)

    # Certifica-se de que somos capazes de carregar base models, bem como modelos derivados (com cabeças)
    start_prefix = ""
    model_to_load = model
    if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
        start_prefix = cls.base_model_prefix + "."
    if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
        model_to_load = getattr(model, cls.base_model_prefix)
        base_model_expected_keys = list(model_to_load.state_dict().keys())
        if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
            raise ValueError(
                "O dicionário de estado do modelo que você está tentando carregar está corrompido. Tem certeza de que"
                "foi salvo corretamente?"
            )
        if device_map is not None:
            device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

    def _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        add_prefix_to_model,
        remove_prefix_from_model,
        ignore_mismatched_sizes,
    ):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                # Se o checkpoint estiver fragmentado, talvez não tenhamos a chave aqui.
                if checkpoint_key not in state_dict:
                    continue
                model_key = checkpoint_key
                if remove_prefix_from_model:
                    # A chave do modelo começa com `prefix` mas `checkpoint_key` não, então nós a adicionamos.
                    model_key = f"{prefix}.{checkpoint_key}"
                elif add_prefix_to_model:
                    # A chave do modelo não começa com `prefix`, mas `checkpoint_key` começa, então a removemos.
                    model_key = ".".join(checkpoint_key.split(".")[1:])

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                    )
                    del state_dict[checkpoint_key]

        return mismatched_keys

    if resolved_archive_file is not None:
        folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
    else:
        folder = None
    if device_map is not None and is_safetensors:
        param_device_map = expand_device_map(device_map, original_loaded_keys, None)

        str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
        if sharded_metadata is None:
            archive_file = (
                resolved_archive_file[0]
                if isinstance(resolved_archive_file, (list, tuple))
                else resolved_archive_file
            )
            weight_map = {p: archive_file for p in original_loaded_keys}
        else:
            weight_map = {p: os.path.join(folder, f) for p, f in sharded_metadata["weight_map"].items()}
        offload_index = {
            p: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
            for p, f in weight_map.items()
            if param_device_map[p] == "disk"
        }

    if state_dict is not None:
        # Checkpoint completo
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )
        error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
        offload_index = None
    else:
        # Ponto de verificação fragmentado ou inteiro, mas low_cpu_mem_usage==True

        # Deve ser sempre uma lista, mas só para ter certeza.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        error_msgs = []
        mismatched_keys = []
        if not is_safetensors:
            offload_index = {} if device_map is not None and "disk" in device_map.values() else None
        if offload_state_dict:
            state_dict_folder = tempfile.mkdtemp()
            state_dict_index = {}
        else:
            state_dict_folder = None
            state_dict_index = None

        if is_sharded_safetensors:
            disk_only_shard_files = get_disk_only_shard_files(
                device_map,
                sharded_metadata=sharded_metadata,
                start_prefix=None
            )
            disk_only_shard_files = [
                os.path.join(folder, f) for f in disk_only_shard_files
            ]
        else:
            disk_only_shard_files = []

        if len(resolved_archive_file) > 1:
            resolved_archive_file = logging.tqdm(
                resolved_archive_file,
                desc="Carregando fragmentos do checkpoint"
            )
        for shard_file in resolved_archive_file:
            # Ignora a carga para fragmentos que contenham apenas pesos descarregados de
            # disco ao usar safetensors para a transferência.
            if shard_file in disk_only_shard_files:
                continue
            state_dict = load_state_dict(shard_file)

            # Chaves incompatíveis contêm tuplas key/shape1/shape2 de pesos no checkpoint que possuem uma forma não
            # que não corresponde aos pesos no modelo.
            mismatched_keys += _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            if low_cpu_mem_usage:
                new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    loaded_keys,
                    start_prefix,
                    expected_keys,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    offload_index=offload_index,
                    state_dict_folder=state_dict_folder,
                    state_dict_index=state_dict_index,
                    dtype=dtype,
                    is_safetensors=is_safetensors,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                )
                error_msgs += new_error_msgs
            else:
                error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)

            # força a liberação da memória
            del state_dict
            gc.collect()

        if offload_index is not None and len(offload_index) > 0:
            if model != model_to_load:
                # Precisamos adicionar o prefixo do base model
                prefix = cls.base_model_prefix
                if not is_safetensors:
                    for weight_name in offload_index:
                        shutil.move(
                            os.path.join(offload_folder, f"{weight_name}.dat"),
                            os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                offload_index = {f"{prefix}.{key}": value for key, value in offload_index.items()}
            if not is_safetensors:
                save_offload_index(offload_index, offload_folder)
                offload_index = None

        if offload_state_dict:
            # Carrega de volta o dict do estado temporariamente descarregado
            load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
            shutil.rmtree(state_dict_folder)

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tVocê pode considerar adicionar `ignore_mismatched_sizes=True` no método `from_pretrained` do modelo."
            )
        raise RuntimeError(f"Erro(s) ao carregar state_dict para {model.__class__.__name__}:\n\t{error_msg}")

    if is_quantized:
        unexpected_keys = [elem for elem in unexpected_keys if "SCB" not in elem]
        missing_keys = [elem for elem in missing_keys if "SCB" not in elem]

    missing_keys = list(filter(lambda x: 'adapter' not in x,  missing_keys))

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Alguns pesos do ponto de verificação do modelo em {pretrained_model_name_or_path} não foram usados quando"
            f" inicializando {model.__class__.__name__}: {unexpected_keys}\n- Isso é esperado se você for"
            f" inicializando {model.__class__.__name__} do ponto de verificação de um modelo treinado em outra tarefa ou"
            " com outra arquitetura (por exemplo, inicializando um modelo BertForSequenceClassification de um"
            " modelo BertForPreTraining).\n - Isso NÃO É esperado se você estiver inicializando"
            f" {model.__class__.__name__} do ponto de verificação de um modelo que você espera ser exatamente idêntico"
            " (inicializando um modelo BertForSequenceClassification a partir de um modelo BertForSequenceClassification)."
        )
    else:
        logger.info(f"Todos os pesos dos pontos de verificação do modelo foram usados na inicialização {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Alguns pesos de {model.__class__.__name__} não foram inicializados no ponto de verificação do modelo em"
            f" {pretrained_model_name_or_path} e foram inicializados recentemente: {missing_keys}\nVocê provavelmente deveria"
            " TREINE este modelo em uma tarefa posterior para poder usá-lo para previsões e inferências."
        )
    elif len(mismatched_keys) == 0:
        logger.info(
            f"Todos os pesos de {model.__class__.__name__} foram inicializados a partir do ponto de verificação do modelo em"
            f" {pretrained_model_name_or_path}.\nSe sua tarefa for semelhante à tarefa, o modelo do ponto de verificação"
            f" foi treinado, você já pode usar {model.__class__.__name__} para previsões sem mais"
            " treinamento."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: encontrou a forma {shape1} no checkpoint e {shape2} no modelo instanciado"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Alguns pesos de {model.__class__.__name__} não foram inicializados no ponto de verificação do modelo em"
            f" {pretrained_model_name_or_path} e foram inicializados recentemente porque as formas não"
            f" match:\n{mismatched_warning}\nVocê provavelmente deveria TREINAR este modelo em uma tarefa downstream para poder"
            " para usá-lo para previsões e inferências."
        )

    return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs
