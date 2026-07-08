# Deep Learning Map of Contents

A living index organized by research direction. 

### Mathematics & Statistics

Pure mathematical machinery — information theory, linear algebra, probability, and dimensionality reduction — that underpins all of deep learning.

Key notes: [[Linear Algebra]] · [[SVD]] · [[Spectral Analysis]] · [[Entropy]] · [[KL Divergence]] · [[ELBO]] · [[MCMC]] · [[Manifold Hypothesis]] · [[t-SNE]] · [[UMAP]] · [[Hamming Loss]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "math")
SORT file.name ASC
limit 10
```


### Learning Methods & Theories

How neural networks are built and trained: optimization, generalization, normalization, regularization, initialization, and learning paradigms.

Key notes: [[Basics of Neural Network]] · [[Activation Functions]] · [[Neural Network Initialization]] · [[Normalization]] · [[Layer Normalization]] · [[RMSNorm]] · [[Internal Covariate Shift]] · [[Optimization Algorithms]] · [[AdamW]] · [[Shampoo]] · [[SOAP]] · [[Muon]] · [[Regularization]] · [[Double Descent]] · [[Scaling Laws]] · [[Pre-training]] · [[FLOPs]] · [[FSL]] · [[Word Embeddings]] · [[Benchmark Datasets]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "dl_theo") OR contains(tags, "learning")
SORT file.name ASC
limit 10
```


### Sequence Modeling & LLMs

The evolution from RNNs → Transformers → SSMs, plus the full LLM stack: tokenization, positional encoding, efficient scaling, retrieval, and reasoning.

Key notes: [[Transformer]] · [[Attention]] · [[Linear Attention]] · [[RNN]] · [[Mamba]] · [[LLMs]] · [[LLaMA]] · [[GPTs]] · [[BPE]] · [[RoPE]] · [[MoE]] · [[RAG]] · [[CoT Prompting]] · [[PPL]] · [[Cluster Training Glossary]] · [[VLM]] · [[LLaVA]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "llm")
SORT file.name ASC
limit 10
```


### Vision & Self-Supervised Learning

Visual architectures, self-supervised objectives (contrastive, distillation, masked modeling), and vision-language alignment.

Key notes: [[ViT]] · [[DINO]] · [[MAE]] · [[JEPA]] · [[CLIP]] · [[LLaVA]] · [[CLS Token]] · [[CNNs]] · [[Contrastive Learning]] · [[ImageNet]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "vision")
SORT file.name ASC
limit 10
```


### Generative Modeling

Learning to synthesize: score-based methods, denoising diffusion, latent diffusion, and variational autoencoders.

Key notes: [[Autoencoder]] · [[VAE]] · [[VQ-VAE]] · [[RAE]] · [[Score-Based Generative Modeling Methods]] · [[DDPM]] · [[LDM]] · [[U-Net]] · [[FID]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "generative")
SORT file.name ASC
limit 10
```


### RL & Embodied AI

Policy optimization, imitation learning, model-based RL, and the path toward autonomous embodied agents.

Key notes: [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] · [[Policy Gradient Methods]] · [[Behavior Cloning]] · [[RLHF]] · [[World Models]]

```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "rl")
SORT file.name ASC
limit 10
```
