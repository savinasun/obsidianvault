### [[dl_moc]]  
# **Contrastive Language-Image Pre-Training (CLIP)**

**Tags**:: #vision #learning
**Links**:: [[Contrastive Learning]] · [[VLM]] · [[DINO]] · [[ViT]] · [[ImageNet]]


**CLIP** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) learns visual representations by aligning images and text using a contrastive objective over massive internet-scale image–caption pairs. It enables **zero-shot transfer** to downstream vision tasks without any task-specific training.

## Architecture: Dual Encoder
---
CLIP consists of two separate encoders trained jointly:

- **Image encoder** $f_I$: a Vision Transformer ([[ViT]]) or ResNet that maps image $x_I \mapsto \mathbf{v} \in \mathbb{R}^d$
- **Text encoder** $f_T$: a Transformer that maps text $x_T \mapsto \mathbf{u} \in \mathbb{R}^d$

Both encoders project into a **shared embedding space** of dimension $d$. The cosine similarity between an image embedding and a text embedding measures their semantic alignment:
$$
\text{sim}(x_I, x_T) = \frac{f_I(x_I)^\top f_T(x_T)}{\|f_I(x_I)\|\,\|f_T(x_T)\|}
$$

## Pre-Training: InfoNCE Loss
---
Given a batch of $N$ image–text pairs $\{(x_I^{(i)}, x_T^{(i)})\}_{i=1}^N$, CLIP builds an $N \times N$ similarity matrix $S$ where $S_{ij} = \text{sim}(x_I^{(i)}, x_T^{(j)})$.

The diagonal entries are **positive pairs** (matched image–text); all off-diagonal entries are **negatives**.

CLIP applies cross-entropy loss symmetrically across both rows (image-to-text) and columns (text-to-image):
$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^N \left[\log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} + \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}\right]
$$
where $\tau$ is a learned temperature parameter. This is the **InfoNCE** (Noise-Contrastive Estimation) loss.

**Why it works at scale**: with large $N$, the negatives form a rich and diverse set that forces the encoder to learn semantically meaningful representations — not just superficial correlations.

## Zero-Shot Classification
---
CLIP can classify images into *any* category without any training examples of those categories:

1. For each class $c$, form a text prompt: `"a photo of a {c}"`
2. Encode all class prompts: $\mathbf{u}_c = f_T(\text{``a photo of a \{c\}''})$
3. Encode the query image: $\mathbf{v} = f_I(x_I)$
4. Classify by highest cosine similarity:
$$
\hat{c} = \arg\max_c \, \text{sim}(\mathbf{v}, \mathbf{u}_c)
$$

**Prompt engineering** matters significantly: `"a photo of a {c}"` outperforms just `"{c}"` because it better matches the training distribution of image captions.

## What CLIP Learns
---
- **Semantic alignment**: embeddings of `"a dog running"` and an image of a running dog are close in the shared space
- **Compositional understanding**: distinguishes `"dog chasing cat"` from `"cat chasing dog"` (to some degree)
- **Robustness**: since CLIP's training data spans the diversity of the internet, it transfers well to unusual domains

**Limitations**:
- Struggles with **fine-grained** distinctions (e.g., counting objects, exact spatial relationships)
- Biased by training data demographics and language distribution
- Text encoder can be fooled by **typographic attacks** (writing "iPod" on an apple image makes it classify as iPod)

## Relationship to Other Methods
---
| Method | Supervisory signal | Modality |
|--------|-------------------|---------|
| [[DINO]] | Self-distillation (no labels) | Vision only |
| **CLIP** | Contrastive, natural language | Vision + Language |
| [[Contrastive Learning]] (SimCLR) | Contrastive, augmentation views | Vision only |
| [[VLM]] | Various (CLIP + LLM) | Vision + Language + Generation |

CLIP embeddings are widely used as the **visual backbone** in [[VLM]] systems: the image encoder is frozen or fine-tuned, and a language model decoder is added on top.

## Scaling
---
CLIP was trained on 400M image–text pairs (WIT dataset). Scale is critical: smaller CLIP models show poor zero-shot performance; the largest ViT-L/14 variant achieves 76.2% zero-shot on ImageNet, competitive with supervised ResNet-50 at 76.1%.

**SigLIP** (Google, 2023): replaces the softmax normalization in InfoNCE with a sigmoid-based pairwise loss, removing the need for a large batch to form negatives — more training efficient.
