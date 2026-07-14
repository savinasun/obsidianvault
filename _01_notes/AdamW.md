### [[dl_moc]]  
# **AdamW**

**Tags**:: #dl_theo #learning
**Links**:: [[Optimization Algorithms]] · [[LLMs]] · [[Normalization]] · [[Muon]] · [[SOAP]]

---

**AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) is Adam with **decoupled weight decay** — a small but critical correction that makes L2 regularization behave correctly under adaptive gradient scaling. It is the default optimizer for virtually all modern LLM and ViT training.

---

## 1. The Problem with Adam + L2 Regularization

In vanilla SGD, L2 regularization and weight decay are equivalent. Adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss produces the gradient:

$$
g_t^{\ell_2} = \nabla \mathcal{L}(\theta_t) + \lambda \theta_t
$$

and the SGD update $\theta_{t+1} = \theta_t - \eta g_t^{\ell_2} = (1 - \eta\lambda)\theta_t - \eta \nabla\mathcal{L}$ is identical to weight decay. **This equivalence breaks under adaptive optimizers.**

In Adam, the combined gradient $g_t^{\ell_2}$ is fed into the adaptive scaling:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $\hat{m}_t$ and $\hat{v}_t$ include the $\lambda\theta_t$ term. The weight decay contribution is **divided by the per-parameter adaptive scale** $\sqrt{\hat{v}_t}$. Parameters with large gradient magnitudes get less weight decay than parameters with small gradient magnitudes — the exact opposite of what L2 regularization should do.

Concretely: a parameter in a frequently-updated embedding row has large $\hat{v}_t$, so the effective weight decay applied to it shrinks. L2 regularization's intended effect (penalizing large weights uniformly) is corrupted.

---

## 2. The AdamW Fix: Decoupled Weight Decay

AdamW separates the weight decay into a direct multiplicative shrinkage, applied **after** the adaptive gradient step:

$$
\boxed{
\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\,\theta_t\right)
}
$$

The $\lambda\,\theta_t$ term is **not** passed through the moment estimates — it acts on the raw weight at a fixed rate $\eta\lambda$, independent of gradient statistics. This recovers the intended regularization behavior: a constant fractional shrinkage per step, regardless of the magnitude of the parameter's gradients.

### Bias Correction

At step $t$, the raw moments are initialized at 0 and need correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

This matters most in early training (small $t$) — without it, the optimizer takes excessively large steps initially because $v_t$ underestimates the true gradient variance.

### Full Algorithm

```
Initialize: m_0 = 0, v_0 = 0, t = 0
For each step:
  t ← t + 1
  g_t ← ∇L(θ_{t-1})               # gradient (no L2 term)
  m_t ← β₁ m_{t-1} + (1-β₁) g_t   # 1st moment
  v_t ← β₂ v_{t-1} + (1-β₂) g_t²  # 2nd moment
  m̂_t ← m_t / (1 - β₁ᵗ)           # bias-corrected 1st
  v̂_t ← v_t / (1 - β₂ᵗ)           # bias-corrected 2nd
  θ_t ← θ_{t-1} - η (m̂_t / (√v̂_t + ε) + λ θ_{t-1})
```

---

## 3. Hyperparameters

| Parameter | Typical value | Role |
|-----------|--------------|------|
| $\eta$ (lr) | $1\text{–}3 \times 10^{-4}$ | Step size; most important hyperparameter |
| $\beta_1$ | $0.9$ | Momentum decay; controls smoothing of gradient direction |
| $\beta_2$ | $0.95\text{–}0.999$ | Variance decay; $0.95$ for short runs, $0.999$ for long stable runs |
| $\epsilon$ | $10^{-8}$ to $10^{-6}$ | Numerical stability; larger $\epsilon$ → less adaptivity (more like SGD) |
| $\lambda$ | $0.1$ | Weight decay; surprisingly large values work well for LLMs |

**$\beta_2$ sensitivity**: using $\beta_2 = 0.999$ means $\hat{v}_t$ tracks long-horizon gradient variance — good for long training runs. Using $\beta_2 = 0.95$ is more reactive, suitable when the gradient distribution shifts rapidly (fine-tuning, RL).

**$\epsilon$ sensitivity**: increasing $\epsilon$ toward $10^{-6}$ softens the inverse-variance scaling, making updates more uniform across parameters. This can help with numerical stability in bf16/fp16 mixed-precision training where small $\hat{v}_t$ values underflow.

---

## 4. Why Weight Decay = 0.1 Works for LLMs

The optimal weight decay for LLMs is surprisingly large ($\lambda = 0.1$, vs. $\lambda = 10^{-4}$ typical for CNNs). At each step the multiplicative shrinkage is $1 - \eta\lambda$. For $\eta = 3 \times 10^{-4}$ and $\lambda = 0.1$:

$$
1 - \eta\lambda = 1 - 3 \times 10^{-5} \approx 0.99997
$$

Per step this is negligible — over millions of steps it accumulates to meaningful regularization. The reason large $\lambda$ is tolerable: Adam's adaptive scaling already keeps effective step sizes small for high-variance parameters, so the direct weight decay term doesn't dominate.

---

## 5. Parameters to Exclude from Weight Decay

Weight decay should **not** be applied to:
- **Biases** — too few parameters, acts as nuisance regularization
- **LayerNorm / RMSNorm scales and shifts** — these are affine rescalings; decaying them interferes with normalization behavior
- **Embedding tables** — often large; weight decay interacts poorly with sparse updates

Standard practice: split parameters into two groups — decay group (weight matrices) and no-decay group (everything else).

---

## 6. Memory Cost and Low-Memory Variants

AdamW requires storing $m_t$ and $v_t$ — two fp32 states per parameter. For a 7B model:
$$7 \times 10^9 \times 2 \times 4 \text{ bytes} = 56 \text{ GB}$$

**8-bit Adam** ([Dettmers et al., 2022](https://arxiv.org/abs/2110.02861)): quantizes optimizer states to int8 with block-wise dynamic exponents, reducing to ~$14 \text{ GB}$ with minimal accuracy loss.

**Adafactor** ([Shazeer & Stern, 2018](https://arxiv.org/abs/1804.04235)): factorizes $v_t$ as an outer product of row/column statistics, reducing the second moment from $O(mn)$ to $O(m+n)$ per weight matrix.

---

## 7. Relationship to Other Optimizers

| Optimizer | Relation to AdamW |
|-----------|------------------|
| [[SOAP]] | Runs Adam (like AdamW) in the Shampoo eigenbasis; recovers AdamW as a special case when eigenbasis = identity |
| [[Muon]] | Replaces the adaptive scaling with orthogonalization; used for hidden-layer matrices; AdamW still used for embeddings and head |
| [[Shampoo]] | Second-order generalization; AdaGrad preconditioned with Kronecker factors instead of diagonal $\hat{v}_t$ |
| Lion | Replaces adaptive scaling with sign(gradient); cheaper memory, similar performance |

AdamW remains the dominant baseline — [[Muon]] and [[SOAP]] are increasingly competitive for training from scratch, but AdamW's simplicity and robustness make it the safe default.
