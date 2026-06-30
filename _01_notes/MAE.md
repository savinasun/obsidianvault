### [[dl_moc]]  
# **Masked Autoencoder (MAE)**

**Tags**:: #vision #learning
**Links**:: [[Autoencoder]] · [[ViT]] · [[DINO]] · [[JEPA]] · [[Contrastive Learning]] · [[Transformer]] · [[Normalization]] · [[ImageNet]]


**MAE** ([He et al., 2021](https://arxiv.org/abs/2111.06377)) is a self-supervised pre-training method for vision that learns representations by masking a large fraction of image patches and reconstructing the missing pixels. It is the **vision analogue of BERT's masked language modeling**, adapted to the continuous signal of images.

## Core Idea
---
**Hypothesis**: images have heavy spatial redundancy — neighboring patches are highly correlated, so predicting masked patches from context is easy unless the masking ratio is very high. BERT masks ~15% of tokens; MAE masks **75%** of patches to make the task non-trivial.

The model learns to:
1. **Encode** only the visible 25% of patches into rich representations
2. **Decode** from those representations back to the masked pixel values

The pre-training signal is purely self-supervised — no labels needed.

## Architecture
---
MAE uses an **asymmetric encoder-decoder** design that is key to its efficiency.

### Encoder
A standard [[ViT]] that processes only the **unmasked** patches. Since 75% of patches are removed before encoding:
- Encoder processes $\sim\!25\%$ of the full sequence
- Positional embeddings are applied to visible patches so the encoder knows their spatial location
- Full [[Transformer]] self-attention over the visible subset

### Decoder
A lightweight [[Transformer]] that takes:
- Encoded representations of visible patches (from the encoder)
- Learnable **mask tokens** (a shared learned vector) for the masked positions — these are placed at their original spatial positions with positional embeddings

The decoder reconstructs the **pixel values** of the masked patches. Because the decoder is small and only used during pre-training, the asymmetry makes the design computationally efficient — the expensive encoder only sees 25% of the image.

### Pre-training Target
The loss is the **mean squared error (MSE)** between the predicted and actual pixel values, computed **only over the masked patches**:
$$
\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| p_i - x_i \right\|^2
$$
where $\mathcal{M}$ is the set of masked patch indices, $p_i$ is the predicted patch, and $x_i$ is the normalized target (optionally using per-patch normalization for stability).

### Random Masking
Patches are sampled **uniformly at random** (no structured masking strategy like block masking). This prevents the encoder from inferring masked content from local context, forcing the model to learn global semantic representations.

## Why Asymmetry Matters
---
The key insight is that the **encoder should not see mask tokens**. In contrast, BERT-style models (e.g., BEiT) give the encoder the full sequence including masked positions, wasting compute on placeholder tokens.

| Design choice | MAE | BEiT |
|---|---|---|
| Encoder input | Visible patches only | Full sequence (mask tokens included) |
| Pre-training target | Raw pixel values | Discrete visual tokens (dVAE) |
| Masking ratio | 75% | 40% |
| Decoder | Lightweight Transformer | None (encoder predicts directly) |

By excluding mask tokens from the encoder, MAE processes $4\times$ fewer tokens and trains $3\times$–$4\times$ faster than full-sequence alternatives.

## Training Details
---
- **Backbone**: ViT-L/16 or ViT-H/14
- **Pre-training**: 1600 epochs on ImageNet-1K (no labels)
- **Masking ratio**: 75% (ablated — higher ratios improve representation quality)
- **Decoder**: 8 Transformer blocks, smaller width than encoder
- **Normalization**: per-patch pixel normalization as target improves results
- **No data augmentation**: only random cropping + horizontal flip (the masking is the primary augmentation)

## Fine-tuning Performance
---
After pre-training, the encoder is used as a backbone for downstream tasks via fine-tuning:

| Task | Architecture | Top-1 acc / AP |
|------|-------------|----------------|
| ImageNet classification | ViT-H (MAE pre-train) | 87.8% |
| COCO detection (ViTDet) | ViT-L (MAE pre-train) | 60.4 AP$^\text{box}$ |
| ADE20K segmentation | ViT-L (MAE pre-train) | 53.6 mIoU |

MAE outperforms supervised pre-training baselines and competes with [[DINO]]/DINOv2 on dense prediction tasks when fine-tuned.

## Comparison to Contrastive Learning
---
[[Contrastive Learning]] (SimCLR, MoCo, [[DINO]]) and MAE represent two paradigms of self-supervised visual learning:

| | Contrastive (DINO, SimCLR) | MAE |
|--|---------------------------|-----|
| **Objective** | Invariance to augmentations | Reconstruction |
| **Augmentation sensitivity** | Critical (defines invariances) | Minimal (masking is the signal) |
| **Pre-training cost** | High (large batch / momentum encoder) | Moderate (asymmetric encoder) |
| **Linear probe** | Excellent (features linearly separable) | Good (slightly weaker linear probe) |
| **Fine-tuning** | Good | Excellent (often better than contrastive) |
| **Dense tasks** | Moderate | Strong |

MAE learns representations optimized for **fine-tuning** rather than linear probing — the features are less linearly structured but contain richer detail useful when the full network is updated.

## Connection to NLP Masked Modeling
---
MAE is the direct vision analogue of **BERT** (masked language modeling), but with two key differences:
1. **Continuous targets**: pixels are real-valued, not discrete tokens → use MSE loss instead of cross-entropy
2. **High redundancy**: natural images have much higher spatial redundancy than text, requiring the 75% masking ratio to force meaningful learning

Later work (**BEiT**, **data2vec**) explored predicting discrete visual tokens or latent BERT-like targets rather than raw pixels, closing the gap between NLP and vision masked modeling.

## Comparison to JEPA
---
[[JEPA]] (I-JEPA, Assran et al. 2023) addresses the core limitation of pixel-space reconstruction: the model is forced to predict texture, lighting, and other irrelevant details. JEPA predicts representations of masked regions in the **embedding space** of an EMA target encoder rather than in pixel space.

| | MAE | I-JEPA |
|--|-----|--------|
| Prediction target | Raw pixels (MSE) | EMA encoder representation |
| Masking | Random scattered (75%) | Multi-block, semantically large |
| Augmentations | Minimal | None |
| Reconstructs irrelevant details | Yes | No — target encoder abstracts them away |
| Linear probe | Good | Stronger |

The key tradeoff: MAE's pixel targets preserve every detail (good for fine-tuning on dense tasks), while JEPA's repr targets discard irrelevant variation (better linear probe, no augmentation needed).

## References
---
- [He et al., 2022 — Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [Bao et al., 2022 — BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)
- [Dosovitskiy et al., 2021 — ViT](https://arxiv.org/abs/2010.11929)
- [Assran et al., 2023 — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
