### [[dl_moc]]
# **Learning Methods & Theories**

**Tags**:: #dl_theo #learning
**Links**:: [[math_moc]] · [[sequence_llms_moc]] · [[vision_ssl_moc]] · [[generative_moc]] · [[rl_embodied_moc]]

The theory and practice of training neural networks: how models learn, how representations generalize, and the architectural primitives that enable stable optimization at scale.

## Training Paradigms
---
- **[[Pre-training]]** — The two-phase strategy: learn general representations on large unlabeled corpora, then fine-tune on task-specific data; covers BERT/GPT-style pre-training, supervised vs. self-supervised, and the knowledge/behavior split in LLMs
- **[[Few-Shot Learning]]** — Learning from very few labeled examples; covers meta-learning, prototypical networks, and in-context learning as an emergent form of few-shot adaptation

## Optimization
---
- **[[Optimization Algorithms]]** — SGD/momentum, Adam, AdamW, Lion; learning rate schedules (warmup + cosine decay); gradient clipping; batch size tradeoffs and the critical batch size
- **[[Regularization]]** — L1/L2 weight decay, inverted dropout, label smoothing, stochastic depth, data augmentation; entropy regularization for RL; summary table of when to use each technique

## Network Primitives
---
- **[[Activation Functions]]** — ReLU, GELU, SiLU, SwiGLU; dead neuron problem; why smooth activations matter for gradient flow and modern transformer training
- **[[Neural Network Initialization]]** — Xavier/Glorot, Kaiming/He, transformer-specific init; symmetry breaking; why initialization matters for gradient flow at depth
- **[[Normalization]]** — BatchNorm, LayerNorm, RMSNorm, GroupNorm; Pre-LN vs. Post-LN; training stability in deep networks
- **[[Layer Normalization]]** — LayerNorm mechanics and derivation; why it replaced BatchNorm in Transformers; Pre-LN for stable training
- **[[RMSNorm]]** — Root Mean Square normalization: simplified LayerNorm without mean centering; used in LLaMA and modern LLMs for efficiency
- **[[Internal Covariate Shift]]** — The original motivation for BatchNorm; distributional shift of layer inputs during training and how normalization addresses it
- **[[Basics of Neural Network]]** — Index note: forward/backward propagation, the chain rule, GPU parallelism; links to all sub-topic notes

## Generalization Theory
---
- **[[Double Descent]]** — The bias-variance tradeoff revisited: test loss can *decrease again* after the interpolation threshold in overparameterized models; implications for modern deep learning
- **[[Scaling Laws]]** — Chinchilla compute-optimal scaling: loss follows power laws in model size $N$ and data $D$; optimal token-to-parameter ratio; emergent abilities at scale
- **[[FLOPs]]** — Floating-point operations as the fundamental compute currency; how FLOPs scale with model size, sequence length, and batch size

## Evaluation
---
- **[[Benchmark Datasets]]** — GSM-8K, HellaSwag, MMLU, MBPP, MT-Bench; BLEU score derivation and limitations; modern alternatives (BERTScore, ROUGE)

## Dataview
---
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "dl_theo") OR contains(tags, "learning")
SORT file.name ASC
```
