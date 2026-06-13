# Deep Learning Map of Contents

A living index of notes across mathematics, architecture, learning theory, and frontier research. Each section shows key notes directly and a Dataview query for tag-filtered discovery.

---

### Mathematics & Foundations

Core mathematical machinery: linear algebra, probability, information theory, and matrix decompositions.

Key notes: [[Linear Algebra]] · [[SVD]] · [[Spectral Analysis]] · [[Entropy]] · [[KL Divergence]] · [[ELBO]] · [[MCMC]] · [[Manifold Hypothesis]] · [[Hamming Loss]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "math")
SORT file.name ASC
```


### Architecture

Building blocks of modern neural networks: transformers, normalization, activation functions, positional encodings, and efficiency variants.

Key notes: [[Transformer]] · [[Attention]] · [[MoE]] · [[RoPE]] · [[RMSNorm]] · [[Layer Normalization]] · [[Normalization]] · [[Activation Functions]] · [[Neural Network Initialization]] · [[U-Net]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "architecture")
SORT file.name ASC
```


### Large Language Models

Modern LLM internals: training dynamics, tokenization, inference tricks, and model families.

Key notes: [[LLMs]] · [[Attention]] · [[GPTs]] · [[LLaMA]] · [[BPE]] · [[RoPE]] · [[KV Cache]] · [[MoE]] · [[PPL]] · [[Cluster Training Glossary]] · [[CoT Prompting]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "llm")
SORT file.name ASC
```


### Vision & Self-Supervised Learning

Visual representation learning: transformers for vision, self-supervised objectives, multimodal models.

Key notes: [[ViT]] · [[DINO]] · [[CLIP]] · [[VLM]] · [[Contrastive Learning]] · [[U-Net]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "vision")
SORT file.name ASC
```


### Reinforcement Learning & Robotics

Policy optimization, imitation learning, and model-based RL — foundations for embodied AI.

Key notes: [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] · [[Policy Gradient Methods]] · [[Behavior Cloning]] · [[RLHF]] · [[World Models]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "rl")
SORT file.name ASC
```


### Generative Models

Score-based methods, diffusion models, VAEs, and latent diffusion.

Key notes: [[Score-Based Generative Modeling Methods]] · [[DDPM]] · [[LDM]] · [[VAE]] · [[ELBO]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "generative")
SORT file.name ASC
```


### Learning Methods

Supervised, self-supervised, contrastive, and few-shot learning paradigms.

Key notes: [[Contrastive Learning]] · [[DINO]] · [[CLIP]] · [[FSL]] · [[t-SNE]] · [[UMAP]] · [[Benchmark Datasets]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "learning")
SORT file.name ASC
```


### Training Dynamics & Theory

Theoretical underpinnings: generalization, optimization landscapes, normalization effects, initialization.

Key notes: [[Double Descent]] · [[Manifold Hypothesis]] · [[Neural Network Initialization]] · [[Internal Covariate Shift]] · [[Normalization]] · [[Optimization Algorithms]] · [[Scaling Laws]] · [[FLOPs]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "dl_theo")
SORT file.name ASC
```
