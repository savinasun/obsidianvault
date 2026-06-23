### [[dl_moc]]  
# **Representation Autoencoder (RAE)**

**Tags**:: #generative #dl_theo #diffusion
**Links**:: [[Autoencoder]] · [[VAE]] · [[DiT]] · [[DINO]] · [[LDM]] · [[FID]] · [[ViT]] · [[GAN]]


**RAE** ([Zheng et al., 2025](https://arxiv.org/abs/2510.11690)) asks: instead of training a VAE from scratch to compress images for diffusion, why not reuse a frozen pretrained representation encoder and just learn a decoder? The answer is that this works far better — frozen encoders like DINOv2 already capture rich semantic structure; a lightweight decoder trained on top recovers reconstruction quality that exceeds SD-VAE, with 47× faster DiT convergence.

## Motivation: VAE Bottlenecks in Latent Diffusion
---
[[LDM]]-style diffusion (including [[DiT]]) relies on a VAE encoder to compress images into a compact latent space, then runs diffusion there. This introduces three compounding problems:

1. **Outdated backbones**: the SD-VAE encoder was trained purely for reconstruction, yielding weak semantic representations (8% linear probe accuracy on ImageNet).
2. **Aggressive compression**: spatial downsampling to 4-channel latents destroys high-frequency detail that diffusion must then hallucinate back.
3. **Reconstruction ceiling**: rFID of 0.62 limits generation quality before diffusion even runs.

RAE decouples the representation from the reconstruction objective entirely.

## RAE Architecture
---
An RAE is the combination of a **frozen pretrained encoder** and a **trained lightweight decoder**:

$$
z = f_\text{enc}(x), \qquad \hat{x} = g_\theta(z)
$$

where $f_\text{enc}$ is frozen (no gradient, no finetuning) and $g_\theta$ is a ViT-XL trained end-to-end.

**Encoder options evaluated** (all at ViT-B scale, 768-dim tokens):

| Encoder | Type | Linear Probe Acc. | rFID |
|---------|------|:-----------------:|:----:|
| DINOv2-B | Self-supervised | 84.5% | 0.49 |
| SigLIP2-B | Language-supervised | — | — |
| MAE-B | Masked AE | — | **0.16** |

All three beat SD-VAE's rFID of 0.62. The encoder produces $N = HW/p_e^2$ tokens at the encoder patch size — **no spatial compression**. For 256×256 with $p_e = 14$: 256 tokens of dim 768.

## Decoder Training
---
The decoder is trained with a combined reconstruction loss:

$$
\mathcal{L}_\text{rec}(x) = \omega_L \cdot \text{LPIPS}(\hat{x}, x) + \|{\hat{x} - x}\|_1 + \omega_G \cdot \lambda \cdot \mathcal{L}_\text{GAN}(\hat{x}, x)
$$

The GAN term is critical for perceptual sharpness. The decoder is ~14× more compute-efficient than SD-VAE's encoder/decoder and outperforms it on rFID.

### Noise-Augmented Decoder Training
A subtle problem: the encoder sees clean real images $x \sim p_\text{data}$, so $z$ lives on a low-dimensional manifold. But at generation time, the diffusion model outputs a continuous distribution over $z$-space — off-manifold samples reach the decoder that it has never seen during training, causing artifacts.

Fix: augment decoder training inputs with additive Gaussian noise:

$$
p_n(z) = \int p(z - \epsilon) \cdot \mathcal{N}(0, \sigma^2 I) \, d\epsilon, \quad \sigma \sim |\mathcal{N}(0, \tau^2)|
$$

This smooths the training distribution over $z$-space at no cost to encoder semantics. There's a small rFID tradeoff (0.49 → 0.57) but generation FID improves (4.81 → 4.28).

## Running Diffusion in RAE Latent Space
---
RAE's 768-dim tokens are far higher-dimensional than VAE's 4-channel latents. Naively dropping a standard DiT into this space fails. Three adaptations are required:

### 1. Width Scaling ($d \geq n$)
The DiT hidden dimension $d$ must satisfy $d \geq n$ (latent token dimension). Theoretical reason: diffusion injects Gaussian noise, expanding the data support to full-rank in $\mathbb{R}^n$. A model with $d < n$ cannot represent full-rank distributions.

Verified empirically: training loss fails to converge whenever $d < 768$ regardless of encoder or model size.

### 2. Dimension-Dependent Noise Schedule Shift
Standard diffusion noise schedules are calibrated for pixel or VAE latents. They ignore the effect of latent dimensionality on the SNR at each noise level. RAE latents have effective dimension $m = \text{tokens} \times \text{dim} = 256 \times 768$, versus $n_0 = 4096$ for VAE baselines.

The schedule is shifted by:

$$
\alpha_\text{shift} = \sqrt{m / n_0}
$$

Without this: FID 23.08. With it: FID 4.81 — a >4× improvement from a single hyperparameter fix.

### 3. DiT$^\text{DH}$: Shallow-but-Wide Head
Width scaling alone causes quadratic FLOP growth in attention. To satisfy $d \geq 768$ efficiently, the paper introduces a **dual-head DiT** ($\text{DiT}^\text{DH}$):

$$
z_t = M(x_t \mid t, y) \quad \text{[base DiT, standard width]}
$$
$$
v_t = H(x_t \mid z_t, t, y) \quad \text{[head: 2 layers, } d=2048\text{]}
$$

The head is shallow (2 transformer blocks) but wide (2048-dim), satisfying $d \geq n$ with ~40% of the FLOPs of a monolithic DiT-XL. The head architecture only helps when latents are high-dimensional — applying $\text{DiT}^\text{DH}$ to VAE latents hurts performance.

## Results
---

**ImageNet 256×256**:

| Model | FID (no guidance) | FID (with guidance) |
|-------|:-----------------:|:-------------------:|
| DiT-XL (VAE) | 2.27 | — |
| REPA | 1.83 | — |
| **RAE-DiT (ours)** | **1.51** | **1.13** |

**ImageNet 512×512**: FID **1.13** with guidance, surpassing EDM2 (1.25).

**Convergence speed**: 47× faster than standard VAE-DiT; 16× faster than REPA-aligned methods.

## Why This Works: Representation vs. Reconstruction
---
The core insight is that representation and reconstruction are separable objectives that have historically been conflated in the VAE design. A VAE encoder is optimized to compress for reconstruction, so it discards semantic signal. A pretrained representation encoder (DINO, SigLIP) optimizes for semantics and discards nothing useful for generation — a lightweight decoder can recover pixel details from semantically rich tokens more easily than from compressed reconstruction tokens.

The [[Manifold Hypothesis]] perspective: pretrained encoders already find a structured, low-dimensional manifold in token space. The decoder just needs to learn the inverse map from that manifold to pixels.

## Comparison: RAE vs. SD-VAE
---

| | SD-VAE | RAE (DINOv2-B) |
|---|---|---|
| **Encoder training** | From scratch, reconstruction | Pretrained, frozen |
| **Latent tokens** | 256 × 4 channels | 256 × 768 channels |
| **rFID** | 0.62 | 0.49 |
| **Linear probe acc.** | 8.0% | 84.5% |
| **DiT convergence** | Baseline | 47× faster |
| **Width requirement** | $d \geq 4$ | $d \geq 768$ |
| **Schedule shift needed** | No | Yes |

## Relationship to Other Notes
---
- [[VAE]] — RAE replaces the VAE encoder entirely; no ELBO, no KL, no reparameterization
- [[DiT]] — RAE is the tokenizer for DiT; requires three adaptations (width, schedule, noise augmentation)
- [[DINO]] — DINOv2-B is the best-performing frozen encoder in RAE; its strong semantic representations drive fast convergence
- [[LDM]] — LDMs use SD-VAE as the compression stage; RAE is a direct, superior drop-in replacement
- [[GAN]] — decoder training uses an adversarial loss for perceptual sharpness
- [[FID]] — primary metric; RAE achieves 1.51 (no guidance) and 1.13 (with guidance) on ImageNet 256×256
- [[ViT]] — both encoder and decoder are ViT architectures; patch size determines token count
