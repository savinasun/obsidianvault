### [[dl_moc]]  
# **Fréchet Inception Distance (FID)**

**Tags**:: #generative #dl_theo
**Links**:: [[LDM]] · [[DDPM]] · [[VAE]] · [[Score-Based Generative Modeling Methods]] · [[CLIP]] · [[KL Divergence]]


**FID** ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)) is the standard metric for evaluating image generation quality. It measures how similar a set of generated images is to a set of real images in a learned feature space — jointly capturing **fidelity** (are images realistic?) and **diversity** (does the model cover the data distribution?).

## Core Idea
---
Rather than comparing images pixel-by-pixel, FID embeds both real and generated images into a perceptual feature space and compares their **distribution statistics**.

**Pipeline:**
1. Pass both real images $\{x_i\}$ and generated images $\{\hat{x}_j\}$ through a pre-trained **Inception v3** network; extract the 2048-dim `pool3` activations
2. Fit a **multivariate Gaussian** to each set of features:
   - Real: $(\mu_r, \Sigma_r)$
   - Generated: $(\mu_g, \Sigma_g)$
3. Compute the **Fréchet distance** (Wasserstein-2 distance under Gaussians) between the two distributions:

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

The matrix square root $(\Sigma_r \Sigma_g)^{1/2}$ is computed via eigendecomposition. Lower FID = better.

## What FID Captures
---
The two terms serve distinct roles:

| Term | What it measures |
|------|-----------------|
| $\|\mu_r - \mu_g\|^2$ | **Fidelity** — are the mean features of generated images close to real ones? |
| $\text{Tr}(\cdots)$ | **Diversity** — do generated images span the same variation as real ones? |

A model that generates one perfect image repeatedly (**mode collapse**) will have a low fidelity term but a large covariance mismatch → high FID. This makes FID more informative than per-image quality scores.

## Interpretation
---
FID = 0 means the generated distribution is statistically identical to the real distribution (in Inception feature space). Reference values on standard benchmarks:

| Setting | Typical FID |
|---------|-------------|
| State-of-the-art (ImageNet 256×256) | ~2–5 |
| Competitive unconditional generation | 5–15 |
| Mediocre / mode-collapsed models | 50–200+ |

Lower is always better; comparisons are only valid when using the **same reference set and preprocessing**.

## Practical Requirements
---
- **Sample count**: FID estimates are noisy with small samples. The standard is **50k generated images** vs. the full training/validation set. Using fewer images biases FID upward — a model can look worse just from smaller N.
- **Preprocessing matters**: resizing method (bicubic vs. bilinear), crop strategy, and normalization all shift the absolute FID value. Results from different papers are often not directly comparable.
- **Reference set**: FID is computed against a fixed reference set (e.g., ImageNet validation). Changing the reference changes the number.

## Relationship to Other Metrics
---

| Metric | What it measures | Limitation |
|--------|-----------------|------------|
| **FID** | Distribution similarity (fidelity + diversity) | Inception-biased; no text-alignment |
| **Inception Score (IS)** | Sharpness + diversity via $p(y\|x)$ vs $p(y)$ | Doesn't compare to real images at all |
| **CLIP Score** | Text-image alignment via [[CLIP]] embeddings | Doesn't measure realism or diversity |
| **Precision / Recall** | Fidelity and diversity separately | More diagnostic, less standard |

In practice, text-to-image models report both FID (realism + diversity) and CLIP Score (text alignment) together — neither alone is sufficient.

## Limitations
---
- **Inception bias**: Inception v3 was trained on ImageNet. FID reflects "ImageNet-likeness" — a photorealistic painting may score poorly even if high quality
- **Gaussian assumption**: the actual feature distributions of real images are far from Gaussian; the metric still works empirically but isn't theoretically tight
- **No spatial understanding**: FID is computed over global pooled features — two structurally different images with similar Inception activations look the same to FID
- **Preprocessing sensitivity**: absolute FID numbers are not comparable across codebases without identical image pipelines

## Relationship to Other Notes
---
- [[LDM]] · [[DDPM]] · [[Score-Based Generative Modeling Methods]] — generative models that FID is used to evaluate
- [[VAE]] — the encoder/decoder stage of LDMs; early VAE-only models also benchmarked with FID
- [[CLIP]] — CLIP Score is the standard complement to FID for text-image alignment
- [[KL Divergence]] — both FID and KL divergence measure distribution discrepancy; FID uses Wasserstein-2 geometry instead
