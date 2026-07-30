### [[dl_moc]]  
# **Diffusion Transformer (DiT)**

**Tags**:: #generative #dl_arch #diffusion
**Links**:: [[DDPM]] · [[LDM]] · [[ViT]] · [[VAE]] · [[RAE]] · [[FID]] · [[Attention]] · [[Layer Normalization]] · [[FiLM]] · [[ImageNet]]


**DiT** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) replaces the U-Net backbone in latent diffusion models with a Vision Transformer, demonstrating that Gflops — not parameter count alone — determine generation quality, and that transformers scale more predictably than U-Nets.

## Background: Why Replace U-Net?
---
[[LDM]]-style diffusion runs a denoising network $\epsilon_\theta$ in the latent space of a [[VAE]]. The standard choice is a U-Net with skip connections and convolutional blocks. U-Nets encode useful inductive biases for images (spatial locality, hierarchical features), but they lack the scaling properties and attention-dominated compute profile of transformers. DiT asks: can a plain ViT replace the U-Net without architectural tricks?

## Architecture
---
DiT follows the standard [[LDM]] pipeline but swaps the U-Net for a transformer:

$$
x \xrightarrow{\text{VAE encoder}} z \xrightarrow{\text{patchify}} \text{tokens} \xrightarrow{\text{DiT blocks}} \xrightarrow{\text{unpatchify}} \hat\epsilon, \hat\Sigma
$$

### Patchify
The latent $z \in \mathbb{R}^{I \times I \times C}$ is split into non-overlapping patches of size $p \times p$, then linearly projected to dimension $d$, producing $T = (I/p)^2$ tokens. Standard ViT positional embeddings (sinusoidal) are added.

Patch size $p$ is the dominant compute knob:
- Halving $p$ quadruples $T$ and roughly quadruples Gflops
- Parameters are nearly unchanged (the transformer width $d$ stays fixed)
- Smaller patches → more tokens → better FID (at higher cost)

### Conditioning: adaLN-Zero
DiT conditions on the denoising timestep $t$ and class label $y$ (for class-conditional generation) — an instance of [[FiLM]]-style feature-wise modulation. Four strategies were evaluated; **adaLN-Zero** won:

$$
h = \text{LayerNorm}(x), \quad h' = (1 + \gamma) \odot h + \beta
$$
$$
\text{out} = x + \alpha \odot \text{Attn}(h')  \quad \text{(and similarly for FFN)}
$$

where $(\gamma, \beta, \alpha)$ are predicted by a small MLP from $\text{emb}(t) + \text{emb}(y)$.

The **Zero** suffix means $\alpha$ is initialized to zero — each DiT block starts as an identity function. This stabilizes training at initialization, analogous to zero-init in ResNets.

The other three strategies tested (for reference):

| Strategy | How conditioning enters | Notes |
|----------|------------------------|-------|
| **In-context** | Append $t, y$ as extra tokens | Simplest; dilutes attention |
| **Cross-attention** | Attend over $[t, y]$ sequence | Works but adds params |
| **adaLN** | Scale/shift after LayerNorm (no gate $\alpha$) | Good but worse than adaLN-Zero |
| **adaLN-Zero** ✓ | Scale/shift/gate, zero-init $\alpha$ | Best FID across all scales |

### Unembedding
The final token sequence is projected back to patch predictions: $\hat\epsilon \in \mathbb{R}^{I \times I \times C}$ and $\hat\Sigma \in \mathbb{R}^{I \times I \times C}$ (the learned noise covariance diagonal, following iDDPM).

## Model Variants
---
Four sizes × three patch sizes = 12 configurations:

| Model | Depth | Width $d$ | Heads | Params | Gflops (/2) |
|-------|-------|-----------|-------|--------|------------|
| DiT-S | 12 | 384 | 6 | 33M | 6 |
| DiT-B | 12 | 768 | 12 | 130M | 23 |
| DiT-L | 24 | 1024 | 16 | 458M | 80 |
| DiT-XL | 28 | 1152 | 16 | 675M | 119 |

Patch sizes: `/2`, `/4`, `/8` — DiT-XL/2 is the largest and best configuration.

## Scaling Laws
---
The central finding: **FID decreases monotonically with Gflops**, regardless of whether the compute comes from depth, width, or token count. This holds across all model sizes and patch sizes tested.

Implications:
- Increasing tokens (smaller patch) is more compute-efficient than increasing width for a fixed parameter budget
- Larger models reach target FID values with less training compute — scale is more efficient than training longer
- This scaling behavior is more predictable than U-Nets, which lack a clean compute-quality tradeoff

## Results
---

| Model | Resolution | FID (cfg=1) | FID (guidance) |
|-------|-----------|:-----------:|:--------------:|
| ADM-U | 256×256 | 7.49 | 3.94 (ADM-G,U) |
| LDM-4 | 256×256 | 10.56 | 3.60 (LDM-4-G) |
| **DiT-XL/2** | 256×256 | 9.62 | **2.27** |
| **DiT-XL/2** | 512×512 | — | **3.04** |

DiT-XL/2 at 256×256 uses 119 Gflops vs. 742 Gflops for ADM-U (a CNN) — 6× more efficient at a better FID.

## Classifier-Free Guidance
---
DiT uses classifier-free guidance at inference: jointly train the model with class conditioning (passing $y$) and unconditionally (passing a null label $\emptyset$). At sampling:

$$
\hat\epsilon = \epsilon_\theta(z_t, \emptyset) + s \cdot (\epsilon_\theta(z_t, y) - \epsilon_\theta(z_t, \emptyset))
$$

Higher guidance scale $s$ improves FID at the cost of sample diversity (FID/IS tradeoff).

## DiT in the Ecosystem
---
DiT became the standard backbone for large-scale image generation after 2023:
- **Stable Diffusion 3, FLUX.1**: scaled DiT architectures (MM-DiT) with text conditioning via joint attention over concatenated text–image tokens
- **[[RAE]]**: replaces the VAE tokenizer with a frozen representation encoder; uses DiT as the diffusion backbone with three adaptations (width scaling, schedule shift, noise-augmented decoder)

## Relationship to Other Notes
---
- [[DDPM]] — the denoising objective DiT trains on; DiT is the $\epsilon_\theta$ network
- [[LDM]] — DiT is a drop-in U-Net replacement; uses the same VAE latent space pipeline
- [[ViT]] — DiT IS a ViT with adaLN-Zero conditioning; patchify/unpatchify are identical
- [[VAE]] — provides the latent space DiT operates in; [[RAE]] is a superior alternative tokenizer
- [[RAE]] — uses DiT with three adaptations for high-dimensional representation latents; achieves 1.51 FID vs. DiT's 2.27
- [[FID]] — primary evaluation metric; DiT-XL/2 sets state-of-the-art at 2.27 on ImageNet 256×256
- [[Layer Normalization]] — adaLN-Zero modulates LayerNorm outputs with predicted scale/shift/gate
