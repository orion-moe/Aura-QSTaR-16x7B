import os
import time
import torch
from transformers import AutoTokenizer

import model.modeling_aura as modeling_aura
from model.modeling_aura import AuraForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True

model_path = "/ors/models/Aura-16x7B-QuietSTaR"

n_ahead = 8
n_ahead_talk = 1
merged_talk_heads = True

model = AuraForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map='auto',
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
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.use_end_thought_token = True
model.tokenizer = tokenizer
model.use_start_thought_token = True
model.wandb_enabled = False
model.n_ahead = n_ahead
model.n_passes = 1
model.eval_mode = True
model.first_run = False
model.rm_initialized = True
model.original_mode = True

input = """
<|im_start|>system
Você se chama Aura, foi desenvolvida pela Orion Research e responde apenas em Português do Brasil.<|im_end|>
<|im_start|>user
Resolva a equação passo a passo: 2x + 3x² = 5.<|im_end|>
<|im_start|>assistant
"""
input_ids = tokenizer.encode(input, return_tensors="pt").to(model.device)
firsts_tokens = len(input_ids[0])

def generate(input_ids, attention_mask, model, temp=0.1, max_length=20):
    with torch.no_grad():
        finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        generated_tokens = []  # Lista para acumular tokens gerados
        for cur_token_idx in range(max_length):
            new_ids = model(
                input_ids[~finished_generating],
                attention_mask=attention_mask[~finished_generating]
            )['logits']
            new_ids[:, :, model.tokenizer.vocab_size:] = -float("inf")
            for list_idx, answer_idx in enumerate((~finished_generating).nonzero(as_tuple=True)[0]):
                base_answer_ids = input_ids[answer_idx]
                new_answer_ids = new_ids[list_idx]
                last_token_idx = (base_answer_ids != model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0].max()

                new_ids_sampled = torch.multinomial(
                        torch.nn.functional.softmax(new_answer_ids[last_token_idx] / temp, dim=-1), 1)
                
                if last_token_idx + 1 >= len(base_answer_ids):
                    new_padding = torch.full((len(input_ids), 1), model.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
                    input_ids = torch.cat([input_ids, new_padding], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.zeros_like(new_padding)], dim=-1)
                
                attention_mask[answer_idx, last_token_idx + 1] = 1
                input_ids[answer_idx, last_token_idx + 1] = new_ids_sampled
                generated_tokens.append(new_ids_sampled.item())  # Adiciona o token gerado à lista
                
                # Imprime os tokens gerados até agora, decodificando-os para formar o texto
                os.system('clear')
                print(tokenizer.decode(generated_tokens, skip_special_tokens=True), end='\r')
                
                if new_ids_sampled in [model.tokenizer.eos_token_id, model.tokenizer.bos_token_id, model.tokenizer.pad_token_id]:
                    finished_generating[answer_idx] = 1

            if finished_generating.all():
                break
    return input_ids, attention_mask

start = time.time()
out = generate(input_ids, torch.ones_like(input_ids), model, max_length=500)
end = time.time()

print(tokenizer.decode(out[0][0], skip_special_tokens=False))
print("---------------------------------------------------------------------")
print(f"Total {modeling_aura.num_token_gen} de {len(out[0][0]) - firsts_tokens} tokens gerados.")
print(f"Tempo total de inferência: {end - start} para {len(out[0][0]) - firsts_tokens} tokens.")

