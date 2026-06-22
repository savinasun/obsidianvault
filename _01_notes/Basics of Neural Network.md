### [[dl_moc]]  
# **Neural Network Fundamentals**

**Tags**:: #architecture #dl_theo
**Links**:: [[Activation Functions]] · [[Neural Network Initialization]] · [[Normalization]] · [[Regularization]] · [[Optimization Algorithms]] · [[CNNs]] · [[RNN]] · [[Transformer]]


This note is an index. The core topics have been split into focused notes.

## Building Blocks
---
- **[[Activation Functions]]** — ReLU, GELU, SiLU, SwiGLU; why non-linearities matter
- **[[Neural Network Initialization]]** — Xavier, Kaiming, transformer-specific init; symmetry breaking
- **[[Normalization]]** — BatchNorm, LayerNorm, RMSNorm, GroupNorm; training stability
- **[[Regularization]]** — L1/L2 weight decay, dropout, label smoothing, stochastic depth, data augmentation

## Optimization
---
- **[[Optimization Algorithms]]** — SGD/momentum, Adam, AdamW, Lion; learning rate schedules; gradient clipping; batch size tradeoffs

## Architectures
---
- **[[CNNs]]** — Convolutional networks; inductive biases; LeNet → ResNet; object detection; ViT transition
- **[[RNN]]** — Recurrent networks; LSTM/GRU; vanishing gradients; why Transformers replaced RNNs
- **[[Transformer]]** — Self-attention; encoder-decoder; Pre-LN; complexity; modern variants
- **[[Mamba]]** — State space models; selective SSMs; Transformer vs. SSM tradeoffs

## Theory
---
- **[[Double Descent]]** — Bias-variance tradeoff; interpolation threshold; overparameterized regime
- **[[Internal Covariate Shift]]** — Motivation for normalization layers

## Forward / Backward Propagation
---
A feedforward network with $L$ layers computes:
$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})
$$

**Backpropagation** applies the chain rule to compute $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$ for each layer:
$$
\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} = (W^{(l+1)})^\top \delta^{(l+1)} \odot \sigma'(z^{(l)})
$$
$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
$$

![[nn_forward_prop_formulas.png|700]]

This is equivalent to the **Optimal Control / Joint State Method** (Pontryagin, 1960s), independently rediscovered for neural networks as backpropagation. GPUs excel at these operations via **SIMD** (single instruction, multiple data) parallelism.

## References
---
- [Dive Into Deep Learning](https://d2l.ai/)
- [NYU Deep Learning (LeCun)](https://atcold.github.io/NYU-DLSP20/)
- [AI Summer — Fundamentals](https://theaisummer.com/learn-ai/fundamentals/)
