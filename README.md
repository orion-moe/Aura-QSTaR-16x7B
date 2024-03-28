---
datasets:
- kaykyramos/aura-ft
- teknium/OpenHermes-2.5
language:
- en
- pt
---

<div align="center">
<img src="./assets/AuraGrayMatter.png" width="300"/><br />
Aura-QuietSTaR v0.1 7B
</div>

Developed by: Orion Research

- Model: Aura Archteture
- Model Size: 7.9 billions parameters
- Experts: 16
- Talk Heads: 12
- Talk Ahead: 4
- Context length: 4,096

The Aura-QuietSTaR model is the first AuraModel to use the Aura architecture, with 16 experts and implementing the Quiet-STaR technique (https://arxiv.org/abs/2403.09629). Through meticulous optimization, this model was trained using a computational setup that includes a 24GB RTX 4090 GPU and a 24GB RTX 3090 GPU.

Aura-QuietSTaR is part of the AuraModels series trained to compose our final Aura model. Aura-QuietSTaR was intentionally designed to be powerful like MoE models like Mixtral and lightweight like Mistral 7B, efficient, lightweight VRAM, and cost-effective training.

Aura-QuietSTaR can learn more deeply and quickly thanks to its mixed layout of experts, sparse training, and of course, Quiet-STaR, which allows Aura to have a "conscience" so it can think about what to respond to each token.

---

O modelo Aura-QuietSTaR é o primeiro AuraModel a utilizar a arquitetura Aura, contando com 16 especialistas e implementando a técnica Quiet-STaR (https://arxiv.org/abs/2403.09629). Por meio de otimização meticulosa, deste modelo foi treinado utilizando uma configuração computacional que inclui uma GPU RTX 4090 de 24 GB e uma GPU RTX 3090 de 24 GB.

A Aura-QuietSTaR faz parte da série AuraModels treinada para compor nosso modelo final de Aura. Aura-QuietSTaR foi intencionalmente projetada para ser poderosa como modelos MoE como Mixtral e leve como Mistral 7B, eficiente, VRAM leve e treinamento econômico.

Aura-QuietSTaR pode aprender mais profunda e rapidamente graças ao seu layout misto de especialistas, treinamento esparso e claro, Quiet-STaR, que permite a Aura ter uma "consciência" para que possa pensar no que responder a cada token.

# Training
```sh
python3 train.py
```

# Merging
```sh
python3 -m orion.cli.merge_lora /ors/configs/default.yml --lora_model_dir="/ors/workdir/Aura-QuietSTaR/checkpoint-STEP"
```