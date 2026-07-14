### [[dl_moc]]  
# **Optimization Algorithms**

**Tags**:: #dl_theo
**Links**:: [[LLMs]] · [[Neural Network Initialization]] · [[Double Descent]] · [[Scaling Laws]] · [[AdamW]] · [[Shampoo]] · [[SOAP]] · [[Muon]]


Optimization algorithms update model parameters $\theta$ to minimize a loss $\mathcal{L}(\theta)$. The choice of optimizer, learning rate schedule, and regularization critically affects both training stability and final performance.

## Batch Size and Gradient Descent Variants
---

**Full-batch gradient descent**: compute gradient over the entire dataset per step. Exact gradient direction, but impractical for large datasets.

**Mini-batch SGD**: compute gradient over a random subset (mini-batch) of size $B$. Most common in practice.

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t), \quad |\mathcal{B}_t| = B
$$

**Stochastic SGD** ($B = 1$): high variance per step, very noisy updates — loses all benefit of vectorization.

**Batch size tradeoffs**:
- Larger $B$ → smoother gradients, better GPU utilization, but requires scaling learning rate ($\eta \propto B$) and generalizes slightly worse (sharp minima)
- Smaller $B$ → noisier gradients act as **implicit regularization**, favoring flat minima
- Typical values: 256–4096 for image models; 1M–4M tokens per step for LLMs

**Takeaways**:
+ Shuffling and Partitioning are the two steps required to build mini-batches
+ Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

The **critical batch size** $B_\text{crit}$ is where noise and curvature balance — training beyond it gives diminishing returns per step.

### Stochastic Gradient Descent (SGD)
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)
$$
where $\mathcal{B}_t$ is a mini-batch. Pure SGD is rarely used for deep learning — momentum is almost always added.

![[optimization_sgd_vs_gd.png|500]]

### SGD with Momentum

Pure SGD makes progress only from the current gradient. On a loss surface with **different curvatures along different directions** (a "ravine"), pure SGD oscillates across the steep direction while crawling along the flat one.

Consider a 2D quadratic $\mathcal{L}(\theta_1, \theta_2) = \frac{1}{2}(\lambda_1 \theta_1^2 + \lambda_2 \theta_2^2)$ with $\lambda_1 \gg \lambda_2$. The optimal step size is bounded by the steepest direction: $\eta < 2/\lambda_1$. This makes the step excessively small for the flat direction $\theta_2$, so convergence along $\theta_2$ stalls. Momentum fixes this by **accumulating gradient history** — consistent gradient directions amplify; alternating oscillations cancel.

#### Classical Momentum (Heavy Ball, Polyak 1964)

$$
\begin{aligned}
m_{t+1} &= \mu\, m_t + g_t, \qquad g_t = \nabla_\theta \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta\, m_{t+1}
\end{aligned}
$$

Unrolling the recursion, the momentum buffer is a **geometrically-weighted sum** of all past gradients:

$$
m_{t+1} = \sum_{k=0}^{t} \mu^k\, g_{t-k}
$$

For $\mu = 0.9$, the effective gradient memory is $1/(1-\mu) = 10$ steps. Gradients decay by $\mu^k$ with age $k$.

**Quadratic convergence analysis**: on $\mathcal{L}(\theta) = \frac{1}{2}\theta^\top A\theta$ with eigenvalues $\lambda_1 \leq \cdots \leq \lambda_d$, let $\kappa = \lambda_d/\lambda_1$ be the condition number. The update $\theta_{t+1} = (I - \eta A)\theta_t$ (no momentum) converges at rate per step:

$$
\rho_\text{GD} = \max_i |1 - \eta \lambda_i|
$$

With optimal step $\eta^* = 2/(\lambda_1 + \lambda_d)$ this gives $\rho_\text{GD} = (\kappa-1)/(\kappa+1)$. For ill-conditioned problems ($\kappa \gg 1$), this is close to 1 — very slow.

With heavy-ball momentum, the update per eigenspace $i$ becomes a 2nd-order linear recurrence:

$$
\begin{bmatrix} \theta_i^{(t+1)} \\ m_i^{(t+1)} \end{bmatrix} = \underbrace{\begin{bmatrix} 1 - \eta\lambda_i & -\eta\mu \\ \lambda_i & \mu \end{bmatrix}}_{M_i} \begin{bmatrix} \theta_i^{(t)} \\ m_i^{(t)} \end{bmatrix}
$$

The convergence rate per eigenspace is the spectral radius $\rho(M_i)$. With optimal $\mu = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$ and $\eta = \frac{4}{(\sqrt{\lambda_1}+\sqrt{\lambda_d})^2}$, the convergence rate is:

$$
\rho_\text{HB} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}
$$

This is a dramatic improvement: GD requires $O(\kappa)$ steps to reach $\epsilon$ accuracy; heavy ball requires only $O(\sqrt{\kappa})$ steps — a quadratic speedup in the condition number.

**The oscillation problem**: despite the theoretical speedup, heavy ball momentum can **overshoot and oscillate** in practice, especially on non-quadratic losses. The gradient is evaluated at $\theta_t$, but the step carries the parameter far past $\theta_t$ via the accumulated momentum $m_t$. The gradient is "stale" relative to where the step lands.

---

### Nesterov Accelerated Gradient (NAG)

Nesterov's key fix ([Nesterov, 1983](https://link.springer.com/article/10.1007/BF01588080)): evaluate the gradient **where momentum will carry you**, not where you currently are.

#### Derivation

Define the **lookahead position** — the point you'll reach from momentum alone:

$$
\tilde\theta_t = \theta_t - \mu\, \eta\, m_t
$$

Compute the gradient at the lookahead, then form the new momentum and update:

$$
\boxed{
\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}(\tilde\theta_t) = \nabla_\theta \mathcal{L}(\theta_t - \mu\,\eta\, m_t) \\
m_{t+1} &= \mu\, m_t + g_t \\
\theta_{t+1} &= \theta_t - \eta\, m_{t+1}
\end{aligned}
}
$$

**Geometric interpretation**:

```
Classical momentum            Nesterov momentum

θ_t ──(momentum)──► [stale]   θ_t ──(momentum)──► θ̃_t
                       │                              │
                    grad(θ_t)                     grad(θ̃_t)
                       │                              │
                       ▼                              ▼
                    θ_{t+1}                        θ_{t+1}
```

Classical applies momentum from $\theta_t$, then corrects with the gradient at $\theta_t$ — the gradient is computed before the momentum step. Nesterov **takes the momentum step first**, then corrects using the gradient at the new position. The correction is better informed about the actual descent direction after momentum.

#### Why the lookahead gradient is better

Using a first-order Taylor expansion around $\theta_t$:

$$
\nabla\mathcal{L}(\tilde\theta_t) = \nabla\mathcal{L}(\theta_t) - \mu\,\eta\,H_t\, m_t + O(\eta^2)
$$

where $H_t = \nabla^2\mathcal{L}(\theta_t)$ is the Hessian. The Nesterov gradient implicitly includes a **Hessian-gradient product term** $-\mu\eta H_t m_t$ — a second-order correction to the gradient, computed cheaply through the gradient difference. This is why NAG achieves faster convergence without explicitly computing the Hessian.

#### Convergence guarantee

For convex $\mathcal{L}$ with $L$-Lipschitz gradient:

| Method | Rate (general convex) | Quadratic ($\kappa$ condition) |
|--------|----------------------|-------------------------------|
| Gradient descent | $O(1/t)$ | $O(\kappa \log 1/\epsilon)$ |
| Heavy ball | $O(1/t)^*$ | $O(\sqrt{\kappa} \log 1/\epsilon)$ |
| **Nesterov** | $\mathbf{O(1/t^2)}$ | $O(\sqrt{\kappa} \log 1/\epsilon)$ |
| Lower bound (any 1st-order) | $\Omega(1/t^2)$ | $\Omega(\sqrt{\kappa} \log 1/\epsilon)$ |

\* Heavy ball achieves $O(1/t^2)$ only on quadratics; for general convex functions it reverts to $O(1/t)$.

**Nesterov is optimal**: the $O(1/t^2)$ rate matches the information-theoretic lower bound for first-order methods — no algorithm that only uses gradients can do better on general convex functions.

**Proof sketch for $O(1/t^2)$**: NAG maintains an estimate sequence $\{\phi_t\}$ that lower-bounds $\mathcal{L}$ and tightens over time. Specifically, NAG implicitly maintains:

$$
\phi_t(\theta) = \phi_0(\theta) + \frac{\eta}{2}\sum_{\tau=0}^{t-1} \lambda_\tau \left( \mathcal{L}(\theta_\tau) + \langle g_\tau, \theta - \theta_\tau\rangle \right)
$$

with weights $\lambda_\tau \propto \tau$. The linear growth of weights — unlike heavy ball's constant weights — is what gives the $O(1/t^2)$ rate rather than $O(1/t)$.

#### Practical equivalent form (Sutskever et al., 2013)

The lookahead form requires computing $\tilde\theta_t$ before each gradient evaluation. An algebraically equivalent single-sequence form is:

$$
\begin{aligned}
v_{t+1} &= \mu\, v_t - \eta\,\nabla\mathcal{L}(\theta_t + \mu\, v_t) \\
\theta_{t+1} &= \theta_t + v_{t+1}
\end{aligned}
$$

where $v_t = -\eta m_t$ is the "velocity." The gradient is still evaluated at $\theta_t + \mu v_t = \theta_t - \mu\eta m_t = \tilde\theta_t$ — identical to the original formulation, just restructured for implementation.

A further reparameterization in terms of the lookahead variable $\phi_t = \theta_t + \mu v_t$:

$$
\begin{aligned}
v_{t+1} &= \mu\, v_t - \eta\, g(\phi_t) \\
\phi_{t+1} &= \phi_t + (1+\mu) v_{t+1} - \mu\, v_t = \phi_t + v_{t+1} + \mu(v_{t+1} - v_t)
\end{aligned}
$$

This shows NAG as a **momentum correction**: the standard step $v_{t+1}$ plus a correction $\mu(v_{t+1} - v_t)$ proportional to the velocity change — a form of acceleration.

#### Connection to Adam and Muon

**NAdam** (Dozat, 2016) incorporates Nesterov momentum into Adam by replacing $\hat m_t$ with the one-step-ahead estimate:

$$
\hat m_t^{\text{NAG}} = \frac{\beta_1 m_t}{1 - \beta_1^{t+1}} + \frac{(1-\beta_1) g_t}{1 - \beta_1^t}
$$

**[[Muon]]** uses Nesterov momentum (step 1 of its algorithm) before orthogonalizing the update — the orthogonalization operates on the lookahead-corrected gradient, which empirically improves direction quality compared to orthogonalizing the raw gradient.

## Adam (Adaptive Moment Estimation)
---
Adam ([Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)) maintains per-parameter **running estimates** of the first and second moments of gradients:

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
**Weight decay** (L2 regularization) in standard Adam is coupled into the gradient, distorting the adaptive scaling. **AdamW** ([Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101)) decouples weight decay from the gradient update:

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
Lion ([Chen et al., 2023](https://arxiv.org/abs/2302.06675) — Google Brain) uses only the **sign** of the gradient:
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
| 8-bit Adam | 2 (quantized) | ~2 bytes |

For a 70B parameter model, Adam optimizer states alone cost $70 \times 10^9 \times 8 \approx 560$ GB — larger than the model weights themselves. This drives use of Lion, 8-bit Adam, or CPU offloading in large-scale training.
