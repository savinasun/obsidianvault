### [[dl_moc]]  
# **Vector Quantized VAE (VQ-VAE)**

**Tags**:: #generative #dl_theo
**Links**:: [[Autoencoder]] · [[VAE]] · [[RAE]] · [[LDM]] · [[Transformer]] · [[GPTs]] · [[ELBO]] · [[KL Divergence]] · [[FID]]


**VQ-VAE** ([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937)) replaces the [[VAE]]'s continuous Gaussian latent space with a **discrete codebook**. Each image is encoded as a sequence of integer indices into a learned vocabulary, enabling clean autoregressive modeling over latents and eliminating the posterior collapse and blurriness problems of the [[VAE]].

## Motivation: Why Discrete Latents?
---
The [[VAE]] assumes a continuous, isotropic Gaussian latent space — a strong prior that doesn't match the structure of natural images. Two practical problems follow:

1. **Posterior collapse**: the KL term can drive the encoder to the prior, ignoring the input entirely
2. **Blurry samples**: the stochastic encoder and MSE reconstruction loss cause the decoder to average over modes
3. **Poor fit for autoregressive priors**: training a prior over a continuous $\mathbb{R}^d$ space is hard; training one over discrete tokens (like language modeling over a vocabulary) is well-understood

Discrete codes address all three: no KL term, deterministic quantization, and a vocabulary that enables a [[GPTs|Transformer]] or PixelCNN prior.

## Vector Quantization
---
Let $\mathcal{E} = \{e_k\}_{k=1}^K \subset \mathbb{R}^d$ be a learned **codebook** of $K$ embedding vectors (the "vocabulary"), each $d$-dimensional.

**Forward pass**:
1. **Encode**: $z_e = f_\phi(x) \in \mathbb{R}^{H \times W \times d}$ (a spatial grid of continuous vectors)
2. **Quantize**: each spatial position is replaced by the nearest codebook entry:
$$
z_q^{(i,j)} = e_{k^*}, \quad k^* = \arg\min_k \|z_e^{(i,j)} - e_k\|_2
$$
3. **Decode**: $\hat{x} = g_\theta(z_q)$

The result is a spatial grid of discrete indices $\{k^*_{i,j}\} \in \{1,\ldots,K\}^{H \times W}$ — analogous to a sequence of tokens. For a $256 \times 256$ image with $8\times$ spatial compression, this is a $32 \times 32 = 1024$-token sequence over a vocabulary of size $K$ (typically $K = 512$ or $8192$).

## Training Objective
---
The quantization step $\arg\min$ is non-differentiable — no gradient flows from the decoder back through the codebook lookup. VQ-VAE uses the **straight-through estimator**: during backpropagation, treat $z_q$ as if it equals $z_e$ (copy gradients through the quantization):

$$
\frac{\partial \hat{x}}{\partial z_e} \approx \frac{\partial \hat{x}}{\partial z_q} \quad \text{(straight-through)}
$$

The full loss has three terms:

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[z_e] - z_q\|^2}_{\text{codebook loss}} + \underbrace{\beta\,\|z_e - \text{sg}[z_q]\|^2}_{\text{commitment loss}}
$$

where $\text{sg}[\cdot]$ is the stop-gradient operator:

| Term | What trains | What it does |
|------|------------|--------------|
| Reconstruction | Encoder + decoder | Minimize pixel error |
| Codebook loss | Codebook $\mathcal{E}$ only | Move codebook entries toward encoder outputs (EMA update is an alternative) |
| Commitment loss | Encoder only | Keep encoder outputs close to the codebook; prevents encoder from fluctuating |

Notably, **there is no KL divergence term** — the regularization comes entirely from the quantization bottleneck itself, not from matching a prior distribution.

## Two-Stage Generation
---
Training VQ-VAE gives a good encoder-decoder pair but not a generative model — there is no distribution over the discrete codes yet.

**Stage 1 — Train VQ-VAE**: learn the encoder, codebook, and decoder end-to-end via the loss above.

**Stage 2 — Train a prior**: collect the discrete code sequences $\{k^*_{i,j}\}$ for all training images. Train an **autoregressive model** (PixelCNN, [[GPTs|GPT-style Transformer]]) to model the distribution over these sequences:

$$
p(k_1, k_2, \ldots, k_N) = \prod_{n=1}^N p(k_n \mid k_1, \ldots, k_{n-1})
$$

**Generation**: sample a code sequence from the prior, look up embeddings in the codebook, decode with $g_\theta$.

This two-stage approach cleanly separates perceptual compression (VQ-VAE) from sequence modeling (prior).

## VQ-VAE-2 ([Razavi et al., 2019](https://arxiv.org/abs/1906.00446))
---
[VQ-VAE-2](https://arxiv.org/abs/1906.00446) scales up with a **hierarchical two-level codebook**:

- **Top level** (coarse, $32\times32$): encodes global structure — layout, object shape, semantic content
- **Bottom level** (fine, $64\times64$): encodes local detail — texture, edges, fine color

The top code is sampled first, then the bottom code is sampled conditioned on the top. Each level has its own PixelSnail (a [[Transformer]]-based autoregressive model) prior:

$$
p(z_\text{top}) \cdot p(z_\text{bottom} \mid z_\text{top})
$$

VQ-VAE-2 produced samples that matched GAN quality on ImageNet ($256\times256$) — the first likelihood-based model to do so — by combining hierarchical structure with a powerful prior.

## VQGAN ([Esser et al., 2020](https://arxiv.org/abs/2012.09841) — "Taming Transformers")
---
[VQGAN](https://arxiv.org/abs/2012.09841) improves VQ-VAE's reconstruction quality by replacing the pixel-level MSE loss with:

1. **Perceptual loss**: MSE in VGG feature space (compares images as the network "sees" them, not pixel-by-pixel)
2. **Patch GAN discriminator**: adversarial loss that penalizes local blurriness

$$
\mathcal{L}_\text{VQGAN} = \mathcal{L}_\text{VQ} + \lambda_\text{perceptual} \mathcal{L}_\text{perc} + \lambda_\text{adv} \mathcal{L}_\text{adv}
$$

With a [[Transformer]] prior trained on the discrete codes, VQGAN enables high-fidelity conditional image synthesis (class-conditional, text-to-image, segmentation-to-image).

VQGAN is also the **perceptual compression stage** used in [[LDM|Latent Diffusion Models]] — the LDM encodes images with a VQGAN-style encoder and runs diffusion in that discrete-friendly latent space.

## Applications and Influence
---

| System | How VQ-VAE is used |
|--------|-------------------|
| **DALL-E 1** (OpenAI, 2021) | DVAE tokenizes images into $32\times32$ discrete codes; GPT-3-scale Transformer models text+image token sequences jointly |
| **Stable Diffusion / LDM** | VQGAN encoder provides the compressed latent space for diffusion |
| **AudioVQ / EnCodec** | VQ applied to audio waveforms for neural audio compression and generation |
| **MAGVIT** (Google, 2023) | VQ-VAE for video tokenization + masked autoregressive video generation |

## Comparison: VQ-VAE vs. VAE vs. RAE
---

| | [[VAE]] | [[RAE]] | VQ-VAE |
|---|---|---|---|
| **Latent type** | Continuous Gaussian | Continuous deterministic | Discrete integer indices |
| **Regularization** | KL divergence | L2 / spectral norm | Quantization bottleneck |
| **Posterior collapse** | Common | Impossible | Impossible |
| **Prior at generation** | $\mathcal{N}(0,I)$ (assumed) | Ex-post density estimator | Learned autoregressive model |
| **Reconstruction** | Blurry (MSE) | Less blurry | Sharp (esp. with VQGAN) |
| **Generation model fit** | Continuous → hard to model | Continuous → hard to model | Discrete → natural token modeling |
| **Use in LDMs** | Compression stage | Drop-in alternative | VQGAN compression stage |

## Relationship to Other Notes
---
- [[Autoencoder]] — VQ-VAE is an [[Autoencoder]] variant; the quantization replaces the bottleneck dimension as the compression mechanism
- [[VAE]] — VQ-VAE replaces the probabilistic encoder and [[KL Divergence]] term with discrete quantization; eliminates posterior collapse
- [[RAE]] — both are deterministic alternatives to VAE; RAE uses continuous L2-regularized codes, VQ-VAE uses discrete codebook entries
- [[ELBO]] — VQ-VAE deliberately avoids the ELBO; straight-through estimation replaces variational inference
- [[KL Divergence]] — eliminated entirely; no prior matching in training
- [[Transformer]] / [[GPTs]] — used as the autoregressive prior over discrete code sequences in Stage 2
- [[LDM]] — uses VQGAN (the adversarially-trained VQ-VAE) as its perceptual compression stage
- [[FID]] — standard metric for evaluating VQ-VAE generation quality
