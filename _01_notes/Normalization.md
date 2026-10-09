### [[dl_moc]]  
# **Normalization**

**Tags**:: #dl_theo #architecture
**Links**:: [[Layer Normalization]] · [[RMSNorm]] · [[FiLM]] · [[DiT]] · [[Neural Network Initialization]] · [[Internal Covariate Shift]] · [[LLMs]] · [[Activation Functions]]


Normalization techniques stabilize and accelerate deep network training by controlling the distribution of activations. Different methods differ in *what* they normalize over (batch, layer, group, or instance).

## Why Normalize?
---
Without normalization, activations in deep networks tend to shift and grow as they propagate through layers — a phenomenon called [[Internal Covariate Shift]]. This causes:
- Saturated activations → vanishing gradients
- Exploding gradients in deeper layers
- Need for careful learning rate tuning and warmup

Normalization counteracts this by standardizing activations, giving:
- Faster convergence (allows higher learning rates)
- Slight regularization effect (noise from batch statistics)
- Smoother optimization landscape

## Normalization at Initialization
---
Before any network normalization, inputs are often normalized:
- **Zero-mean, unit-variance**: $(x - \mu) / \sigma$ per feature across the dataset
- **Image normalization**: subtract per-channel dataset mean, divide by std
  - Reason: equalizes the loss gradient magnitudes across input channels — see [stackexchange](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current)
- **Min-max scaling**: $x \mapsto (x - x_\min)/(x_\max - x_\min)$
#### Effect on Initialization

Normalization changes initialization requirements:
- With BN/LN, the effective weight scale matters less — normalization rescales activations regardless
- However, the gain $\gamma$ and bias $\beta$ still need proper initialization (typically $\gamma = 1$, $\beta = 0$)
- Pre-LN transformers (see [[LLMs]]) are significantly more stable at initialization than Post-LN, allowing larger learning rates without warmup
- The opposite move is also useful: zero-initializing the *last* affine of a residual branch ($\gamma = 0$ in the final BN of a ResNet block, [Goyal et al., 2017](https://arxiv.org/abs/1706.02677); the gate $\alpha = 0$ in [[#Adaptive Layer Norm (adaLN) and adaLN-Zero|adaLN-Zero]]) makes every block start as the identity, so a deep network initially behaves like a shallow one


## Batch Normalization (BN)
---
Normalizes across the **batch dimension** for each feature/channel ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)).

Given activations $x \in \mathbb{R}^{B \times D}$ (batch size $B$, feature dim $D$):
$$
\hat x_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \quad y_i = \gamma \hat x_i + \beta,
$$
where $\mu_\mathcal{B} = \frac{1}{B}\sum_b x_{b,i}$ and $\sigma^2_\mathcal{B} = \frac{1}{B}\sum_b (x_{b,i} - \mu_\mathcal{B})^2$ are computed **per feature** over the batch. $\gamma, \beta$ are learned scale and shift parameters that restore representational capacity.

**At inference**: uses running statistics (exponential moving averages of $\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$ accumulated during training) rather than batch statistics.

**Strengths**:
- Highly effective for CNNs and feedforward networks
- Acts as a <span style="color:rgb(255, 255, 0)">regularizer</span> (batch statistics add noise → less dropout needed)
- Enables high learning rates

**Weaknesses**:
- Depends on batch size: small batches → noisy statistics → instability
- <span style="color:rgb(255, 255, 0)">Doesn't work with variable-length sequences</span> (batch statistics change meaning)
- Inference behavior differs from training → subtle bugs

## Layer Normalization (LN)
---
[Ba et al., 2016](https://arxiv.org/abs/1607.06450). Normalizes across the **feature dimension** for each sample independently — see [[Layer Normalization]]:
$$
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat x_i + \beta
$$
where $\mu = \frac{1}{D}\sum_i x_i$ and $\sigma^2 = \frac{1}{D}\sum_i(x_i - \mu)^2$ are computed **per sample** over features.

**No batch dependency**: statistics are computed independently per sample — works the same at training and inference, for any batch size.

**Standard in [[Transformer|Transformers]] and [[LLMs]]**: the sequence-based nature of transformers makes BN impractical (variable lengths, different statistics per position). LN is the default. See [[LLMs]] for Pre-LN vs. Post-LN discussion.

## RMSNorm
---
[Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467). Simplification of Layer Normalization — see [[RMSNorm]]:
$$
\hat x_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{D}\sum_i x_i^2}
$$

Drops the mean subtraction and the bias term $\beta$. Reduces parameters and improves speed with negligible quality loss. Used in [[LLaMA]], PaLM, and most modern LLMs.

## Group Normalization (GN)
---
[Wu & He, 2018](https://arxiv.org/abs/1803.08494). Divides features into $G$ groups; normalizes within each group independently:
$$
\hat x_{g,i} = \frac{x_{g,i} - \mu_g}{\sqrt{\sigma^2_g + \epsilon}} \cdot \gamma + \beta
$$
where $g$ indexes the group.

- At $G = 1$: equivalent to Layer Normalization
- At $G = D$: equivalent to Instance Normalization (one group per feature)

**Use case**: computer vision tasks where batch size is small (e.g., object detection, segmentation). GN is batch-size independent while preserving spatial structure better than LN.

## Instance Normalization (IN)
---
[Ulyanov et al., 2016](https://arxiv.org/abs/1607.08022). Normalizes over spatial dimensions for each sample and channel independently. Used in **style transfer**: normalizes the style (statistics) of each image independently, enabling the model to transfer style without batch-level interference.

## Adaptive Layer Norm (adaLN) and adaLN-Zero
---
The normalizers above all end with a learned, input-independent affine map $y = \gamma \hat x + \beta$. **Conditional normalization** makes that affine map a *function of a conditioning signal* $c$ instead — the normalizer standardizes the features, and a small network decides, per condition, how to re-scale and re-shift them. This is [[FiLM]] applied on top of a normalizer; the family members differ only in which statistics they normalize over (conditional BN, AdaIN, AdaGN, adaLN — see the [[FiLM#Relation to Conditional Normalization|table in FiLM]]).

<span style="color:rgb(255, 255, 0)"><b>adaLN.</b></span> Layer Normalization whose scale and shift are <span style="color:rgb(255, 255, 0)">regressed from the condition</span> rather than learned as free parameters:
$$
\text{adaLN}(x \mid c) = \gamma(c) \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta(c), \qquad [\gamma(c), \beta(c)] = W\,\text{SiLU}(c) + b,
$$
with $\mu, \sigma^2$ the usual per-sample statistics over the feature dimension and the LayerNorm's own affine disabled (`elementwise_affine=False`). In [[DiT]] ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) the condition is the *sum* of the timestep embedding and the class embedding, $c = \text{emb}(t) + \text{emb}(y)$ — a 256-dim sinusoidal frequency embedding of $t$ passed through a two-layer SiLU MLP to width $d$, plus a learned label table (with an extra null class for classifier-free guidance). Because $\gamma(c), \beta(c)$ are vectors in $\mathbb{R}^d$, adaLN applies the *same* function to every token — the only one of DiT's conditioning mechanisms with that property — and adds essentially no FLOPs to the block. The direct ancestor is **AdaGN** in ADM ([Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233)), the same construction on GroupNorm inside a U-Net residual block.

<span style="color:rgb(255, 255, 0)"><b>adaLN-Zero.</b></span> adaLN plus a third regressed vector, a **gate** $\alpha(c)$ applied to each residual branch immediately before it is added back to the stream. A pre-LN Transformer block has two sub-layers, so one linear layer $d \to 6d$ produces $(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2)$:
$$
\begin{aligned}
x &\leftarrow x + \alpha_1(c) \odot \text{Attn}\big((1 + \gamma_1(c)) \odot \text{LN}(x) + \beta_1(c)\big) \\
x &\leftarrow x + \alpha_2(c) \odot \text{FFN}\big((1 + \gamma_2(c)) \odot \text{LN}(x) + \beta_2(c)\big)
\end{aligned}
$$
The scale is parameterized as $1 + \gamma$ so that a zero output means "leave the normalized features alone". The **Zero** is the initialization: the paper states that the MLP is initialized to output $\alpha = 0$, so every residual branch is switched off and **each DiT block starts as the identity function**; the official implementation zeroes the weight *and* bias of the whole $d \to 6d$ linear layer, which also gives $\gamma = \beta = 0$ at init, i.e. the modulation itself starts as the identity. The same zero-init is applied to the final layer (LN → adaLN shift/scale → zero-initialized linear decoder to $p \times p \times 2C$). Peebles & Xie credit the trick to [Goyal et al., 2017](https://arxiv.org/abs/1706.02677), who zero-initialize the last BatchNorm $\gamma$ of every ResNet block, and to diffusion U-Nets, which zero-initialize the last convolution of each residual block — see [[Neural Network Initialization#Effect of Normalization on Initialization]].

```python
class DiTBlock(nn.Module):
    def __init__(self, d, heads):
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn, self.mlp = Attention(d, heads), Mlp(d, 4 * d)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)  # "-Zero"

    def forward(self, x, c):                       # x: (B, T, d), c: (B, d)
        b1, g1, a1, b2, g2, a2 = self.ada(c)[:, None].chunk(6, dim=-1)
        x = x + a1 * self.attn(self.norm1(x) * (1 + g1) + b1)
        x = x + a2 * self.mlp(self.norm2(x) * (1 + g2) + b2)
        return x
```

**Why it won.** DiT compared four ways to inject $(t, y)$ into a ViT — extra in-context tokens, a cross-attention layer over $[t, y]$, adaLN, and adaLN-Zero — at equal backbone size (DiT-XL/2):

| Block design | Gflops | FID trend (400K iters) |
|---|---|---|
| In-context tokens | 119.4 | worst |
| Cross-attention | 137.6 (~15 % overhead) | middle |
| adaLN | 118.6 | better than both |
| **adaLN-Zero** | 118.6 | **best; roughly half the FID of in-context** |

Two lessons: for a *global, low-dimensional* condition (timestep, class) feature-wise modulation beats attention-based routing at lower cost, and identity initialization of the residual branches matters even in a pre-LN Transformer (adaLN-Zero beats plain adaLN with identical FLOPs). Ablations elsewhere in [[FiLM]] point the same way: the multiplicative pathway carries most of the benefit.

**Cost, and the adaLN-single fix.** The modulation linear layer costs $6d^2 + 6d$ parameters per block. For DiT-XL/2 ($d = 1152$, 28 blocks) that is ≈223M of the model's 675M parameters, even though it adds almost no FLOPs. PixArt-α ([Chen et al., 2023](https://arxiv.org/abs/2310.00426)) reports that these projections account for 27 % of its DiT-based backbone and replaces them with **adaLN-single**: one global set of modulation vectors $\bar S = f(t)$ is computed from the timestep *only*, once, and block $i$ uses $S^{(i)} = \bar S + E^{(i)}$ with a per-block trainable embedding $E^{(i)}$; the text condition, being a token sequence, enters through cross-attention instead. This cut PixArt-α from 833M to 611M parameters. Multimodal DiTs (SD3, FLUX.1) keep the DiT-style per-block modulation but drive it from timestep + pooled text embedding, with separate modulation parameters for the image and text streams.

## Comparison
---

| Method | Norm over | Batch-size dep. | Training ≠ inference | Use case |
|--------|-----------|-----------------|---------------------|----------|
| **BatchNorm** | Batch | Yes | Yes | CNNs, feedforward |
| **LayerNorm** | Feature dim | No | No | Transformers, LLMs |
| **RMSNorm** | Feature dim (no mean) | No | No | Modern LLMs |
| **GroupNorm** | Feature groups | No | No | Small-batch vision |
| **InstanceNorm** | Spatial (H×W) | No | No | Style transfer |
| **adaLN / adaLN-Zero** | Feature dim; affine predicted from condition $c$ | No | No | Conditional generation ([[DiT]]) |

