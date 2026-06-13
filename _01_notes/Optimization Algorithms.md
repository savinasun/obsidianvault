### [[dl_moc]]  
# **Optimization Algorithms**

**Tags**:: #dl_theo
**Links**:: [[LLMs]] · [[Neural Network Initialization]] · [[Double Descent]] · [[Scaling Laws]]


Optimization algorithms update model parameters $\theta$ to minimize a loss $\mathcal{L}(\theta)$. The choice of optimizer, learning rate schedule, and regularization critically affects both training stability and final performance.

## Gradient Descent Variants
---

### Stochastic Gradient Descent (SGD)
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)
$$
where $\mathcal{B}_t$ is a mini-batch. Pure SGD is rarely used for deep learning — momentum is almost always added.

### SGD with Momentum
$$
\begin{aligned}
m_{t+1} &= \mu\, m_t + \nabla_\theta \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta\, m_{t+1}
\end{aligned}
$$
Momentum $\mu \in [0.9, 0.99]$ accumulates gradient history, dampening oscillations and accelerating convergence in consistent directions.

**Nesterov momentum** computes the gradient at the "looked-ahead" position: $\nabla_\theta \mathcal{L}(\theta_t - \mu\, m_t)$, giving a slight improvement.

## Adam (Adaptive Moment Estimation)
---
Adam (Kingma & Ba, 2014) maintains per-parameter **running estimates** of the first and second moments of gradients:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \quad &\text{(first moment)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \quad &\text{(second moment)} \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t} \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} \quad &\text{(update)}
\end{aligned}
$$

**Defaults**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3 \times 10^{-4}$

**Intuition**: $\hat v_t$ normalizes the update by the magnitude of recent gradients — large gradients get smaller updates. This makes Adam largely insensitive to the scale of the loss and robust across different layers.

## AdamW (Adam + Weight Decay)
---
**Weight decay** (L2 regularization) in standard Adam is coupled into the gradient, distorting the adaptive scaling. **AdamW** (Loshchilov & Hutter, 2017) decouples weight decay from the gradient update:

$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_t\right)
$$

The $\lambda\, \theta_t$ term directly decays weights, independent of the gradient magnitude. This is the correct way to do L2 regularization with adaptive optimizers.

**AdamW is the default optimizer for most modern LLMs** (GPT, LLaMA, etc.) — it gives better generalization than Adam alone.

| Optimizer | Key difference | When to use |
|-----------|---------------|-------------|
| SGD+momentum | No adaptivity | CNNs, fine-tuned ResNets |
| Adam | Adaptive, but L2 is coupled | Quick experiments |
| **AdamW** | Adaptive + decoupled L2 | LLM training (default) |
| Lion | Sign-based update | Large-scale, memory-efficient |

## Lion (EvoLved Sign Momentum)
---
Lion (Chen et al., 2023 — Google Brain) uses only the **sign** of the gradient:
$$
\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
\theta_{t+1} &= \theta_t - \eta(\text{sign}(c_t) + \lambda\, \theta_t) \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2)\, g_t
\end{aligned}
$$

**Benefits**: uses only first-moment state ($m_t$) — half the memory of Adam. All parameter updates have the same magnitude $\eta$, giving more uniform learning across layers. Works well at large scale (image and language models).

**Tradeoff**: requires a smaller learning rate than Adam (typically $3\text{–}10\times$ smaller) because updates are less adaptive.

## Learning Rate Schedules
---
The learning rate $\eta$ is typically varied during training:

### Linear Warmup
Ramp $\eta$ from $0$ to $\eta_\text{max}$ over the first $T_\text{warmup}$ steps. Prevents instability early in training when parameter gradients are noisy and the model is far from any solution.
$$
\eta_t = \eta_\text{max} \cdot \frac{t}{T_\text{warmup}}, \quad t < T_\text{warmup}
$$

### Cosine Decay
After warmup, decay $\eta$ following a cosine curve:
$$
\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})\left(1 + \cos\!\left(\frac{t - T_\text{warmup}}{T - T_\text{warmup}}\pi\right)\right)
$$
Cosine decay is smooth and reaches near-zero at the end of training, allowing the model to fine-tune into a sharp minimum.

### Warmup + Cosine (Standard LLM Recipe)
The de facto schedule for LLMs: linear warmup for $\sim 1\text{–}2\%$ of total steps, then cosine decay to $\eta_\text{min} \approx 0.1 \times \eta_\text{max}$.

## Gradient Clipping
---
To prevent **exploding gradients** (especially in RNNs and early transformer training), clip the gradient norm:
$$
g \leftarrow g \cdot \min\!\left(1,\, \frac{c}{\|g\|}\right)
$$
where $c$ is the clip threshold (typically 1.0). This prevents any single update from being catastrophically large while preserving direction.

**When it fires**: gradient norms spiking during training is a sign of instability — check learning rate, batch size, or model architecture. In healthy training, clipping rarely activates.

## Connection to Training Stability in LLMs
---
Large LLMs are particularly sensitive to optimizer choices because:
- **Depth**: gradients must flow through many layers; improper normalization or LR causes vanishing/exploding gradients
- **Scale**: billions of parameters with heterogeneous learning dynamics (embedding vs. attention vs. FFN)
- **Mixed precision**: float16 training amplifies numerical instability

Best practices from [[LLMs]]: use AdamW + weight decay $0.1$ + gradient clipping $1.0$ + linear warmup + cosine decay + [[RMSNorm]] (pre-LN) to stabilize training.

## Optimizer Memory Cost
---

| Optimizer | States stored | Memory (per param, fp32) |
|-----------|--------------|--------------------------|
| SGD | 1 (momentum) | 4 bytes |
| Adam/AdamW | 2 (m, v) | 8 bytes |
| Lion | 1 (m) | 4 bytes |
| 8-bit Adam | 2 (quantized) | ~1 byte |

For a 70B parameter model, Adam optimizer states alone cost $70 \times 10^9 \times 8 \approx 560$ GB — larger than the model weights themselves. This drives use of Lion, 8-bit Adam, or CPU offloading in large-scale training.
