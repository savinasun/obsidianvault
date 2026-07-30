### [[dl_moc]]  
# **Normalization**

**Tags**:: #dl_theo #architecture
**Links**:: [[Layer Normalization]] · [[RMSNorm]] · [[FiLM]] · [[Internal Covariate Shift]] · [[LLMs]] · [[Activation Functions]]


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

## Data Normalization
---
Before any network normalization, inputs are often normalized:
- **Zero-mean, unit-variance**: $(x - \mu) / \sigma$ per feature across the dataset
- **Image normalization**: subtract per-channel dataset mean, divide by std
  - Reason: equalizes the loss gradient magnitudes across input channels — see [stackexchange](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current)
- **Min-max scaling**: $x \mapsto (x - x_\min)/(x_\max - x_\min)$

## Batch Normalization (BN)
---
[Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167). Normalizes across the **batch dimension** for each feature/channel:

Given activations $x \in \mathbb{R}^{B \times D}$ (batch size $B$, feature dim $D$):
$$
\hat x_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \quad y_i = \gamma \hat x_i + \beta
$$
where $\mu_\mathcal{B} = \frac{1}{B}\sum_b x_{b,i}$ and $\sigma^2_\mathcal{B} = \frac{1}{B}\sum_b (x_{b,i} - \mu_\mathcal{B})^2$ are computed **per feature** over the batch.

$\gamma, \beta$ are learned scale and shift parameters that restore representational capacity.

![[normalization_layer_mapping.png|400]]

**At inference**: uses running statistics (exponential moving averages of $\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$ accumulated during training) rather than batch statistics.

**Strengths**:
- Highly effective for CNNs and feedforward networks
- Acts as a regularizer (batch statistics add noise → less dropout needed)
- Enables high learning rates

**Weaknesses**:
- Depends on batch size: small batches → noisy statistics → instability
- Doesn't work with variable-length sequences (batch statistics change meaning)
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

## Comparison
---

| Method | Norm over | Batch-size dep. | Training ≠ inference | Use case |
|--------|-----------|-----------------|---------------------|----------|
| **BatchNorm** | Batch | Yes | Yes | CNNs, feedforward |
| **LayerNorm** | Feature dim | No | No | Transformers, LLMs |
| **RMSNorm** | Feature dim (no mean) | No | No | Modern LLMs |
| **GroupNorm** | Feature groups | No | No | Small-batch vision |
| **InstanceNorm** | Spatial (H×W) | No | No | Style transfer |

## Effect on Initialization
---
Normalization changes initialization requirements:
- With BN/LN, the effective weight scale matters less — normalization rescales activations regardless
- However, the gain $\gamma$ and bias $\beta$ still need proper initialization (typically $\gamma = 1$, $\beta = 0$)
- Pre-LN transformers (see [[LLMs]]) are significantly more stable at initialization than Post-LN, allowing larger learning rates without warmup
