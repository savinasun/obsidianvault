### [[dl_moc]]  
# **Neural Network Initialization**

**Tags**:: #dl_theo
**Links**:: [[Activation Functions]] · [[Normalization]] · [[LLMs]] · [[Double Descent]]


Proper weight initialization prevents **vanishing and exploding gradients** at the start of training, enabling stable and fast convergence from the very first step. The goal is to preserve signal variance through the forward pass and gradient variance through the backward pass.

## The Problem: Vanishing and Exploding Gradients
---
In a deep network with $L$ layers, the gradient at layer 1 involves products of $L$ weight matrices and activation derivatives:
$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=2}^{L} \frac{\partial h_l}{\partial h_{l-1}}
$$

- If each factor has norm $< 1$: gradient vanishes exponentially → early layers learn nothing
- If each factor has norm $> 1$: gradient explodes exponentially → training diverges

![[Pasted image 20250513105524.png]]

## Zero Initialization: Why It Fails
---
Setting all weights $W = 0$ causes **symmetry**: every neuron in a layer receives the same gradient and evolves identically — the network behaves as if it had width 1 regardless of architecture. This is the **symmetry breaking** problem: neurons must start differently to learn different features.

[See also: Symmetry Breaking vs Zero Initialization](https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)

## Random Initialization
---
Setting $W \sim \mathcal{N}(0, \sigma^2)$ or $W \sim U(-a, a)$ breaks symmetry. The key question is: **what $\sigma$ to use?**

**Naive $\sigma = 1$**: for a layer with fan-in $n$, the pre-activation variance is $\text{Var}(z) = n \cdot \sigma^2 = n$, which grows linearly with width — activations explode in wide networks.

**Rule**: set $\sigma$ so that $\text{Var}(z) = 1$ across layers, preserving signal scale.

## Xavier / Glorot Initialization
---
Glorot & Bengio (2010). Designed for **tanh** and **sigmoid** activations (symmetric, near-linear at origin).

Goal: preserve variance through both forward and backward passes. With fan-in $n_\text{in}$ and fan-out $n_\text{out}$:

$$
W \sim U\!\left[-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}},\, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right]
\quad \text{or} \quad
W \sim \mathcal{N}\!\left(0,\, \frac{2}{n_\text{in} + n_\text{out}}\right)
$$

![[Pasted image 20250513105559.png]]
![[initialization_02.png]]

**PyTorch**: `torch.nn.init.xavier_uniform_` / `xavier_normal_`

**Intuition**: the harmonic mean of fan-in and fan-out balances both forward signal propagation and backward gradient propagation.

## Kaiming / He Initialization
---
He et al. (2015). Extends Xavier for **ReLU** and variants, which are asymmetric and kill half the neurons.

For ReLU: $\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \text{Var}(z)$ (half the variance is killed). So we need $2\times$ the variance to compensate:
$$
W \sim \mathcal{N}\!\left(0,\, \frac{2}{n_\text{in}}\right)
$$

![[Pasted image 20250513105622.png]]

**PyTorch**: `torch.nn.init.kaiming_uniform_` / `kaiming_normal_`

**Mode**: use `fan_in` (default) to preserve variance in the forward pass; use `fan_out` to preserve backward gradients.

**Variants**:
- **Leaky ReLU**: $\sigma^2 = \frac{2}{(1+a^2) n_\text{in}}$ where $a$ is the negative slope
- **GELU/SiLU**: Kaiming with `fan_in` works well in practice (GELU behaves similarly to ReLU near zero)

## Effect of Normalization on Initialization
---
[[Normalization|Batch Normalization and Layer Normalization]] reduce sensitivity to initialization because they rescale activations at every layer:
- The effective activation scale is reset to $\sim 1$ after each norm layer regardless of weight initialization
- This is why deep transformers with Pre-LN ([[LLMs]]) are relatively robust to initialization scale

However, initialization still matters for:
- **Convergence speed** in early training (before the optimizer has adapted)
- **Attention logits**: large initial attention weights → uniform or near-zero softmax → poor gradient signal
- **Residual scale**: with many residual connections, the effective depth at initialization affects the initial variance

## Modern Practices for Transformers
---
- **Embedding layers**: $\sim \mathcal{N}(0, d^{-0.5})$ — scale down by model dimension to keep embedding norms reasonable
- **Attention projections**: Xavier or $\mathcal{N}(0, (2 \cdot d)^{-0.5})$
- **Output projection** ($W^O$, FFN second layer): scaled by $1/\sqrt{2L}$ where $L$ is the number of layers — prevents residual stream variance from growing as $O(L)$. Used in GPT-2 ("scaled initialization")
- **Bias**: typically zero-initialized; dropped entirely in modern LLMs (see [[LLMs#Dropping Bias Terms]])

## Orthogonal Initialization
---
Initialize weight matrices as random orthogonal matrices (QR decomposition of random Gaussian). Preserves singular values exactly at 1, making activations and gradients isometric at initialization. Particularly useful for RNNs (where gradients must flow through many timesteps).

**PyTorch**: `torch.nn.init.orthogonal_`
