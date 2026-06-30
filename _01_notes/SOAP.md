### [[dl_moc]]  
# **SOAP**

**Tags**:: #dl_theo #learning
**Links**:: [[Optimization Algorithms]] · [[Shampoo]] · [[AdamW]] · [[Muon]]

---

**SOAP** (*Shampoo as Adam in the eigenbasis of Shampoo's Preconditioner*) ([Vyas et al., 2024](https://arxiv.org/abs/2409.11321)) unifies [[Shampoo]]'s second-order preconditioning with Adam's momentum, showing that running Adam in the eigenbasis of Shampoo's Kronecker factors is **exactly equivalent** to Shampoo with Adam momentum — and that this framing makes the algorithm simpler and faster.

---

## 1. The Key Insight

[[Shampoo]] precomputes two Kronecker factors $L_t \in \mathbb{R}^{m \times m}$ and $R_t \in \mathbb{R}^{n \times n}$ and applies the update:

$$
\Delta W_t = L_t^{-1/4}\, G_t\, R_t^{-1/4}
$$

Adding Adam-style momentum to Shampoo naively is awkward because the preconditioner changes at every update-of-roots step, making the momentum accumulation inconsistent.

**SOAP's observation**: if you decompose $L_t = Q_L \Lambda_L Q_L^\top$ and $R_t = Q_R \Lambda_R Q_R^\top$, then the Shampoo update in the eigenbasis of $L_t \otimes R_t$ is:

$$
\tilde{G}_t = Q_L^\top\, G_t\, Q_R \in \mathbb{R}^{m \times n}
$$

In this rotated coordinate system, the preconditioner is **diagonal** (it's $\Lambda_L^{-1/4} \otimes \Lambda_R^{-1/4}$), and diagonal preconditioning with accumulated squared gradients is precisely **Adam**. Therefore:

> **Running Adam on $\tilde{G}_t = Q_L^\top G_t Q_R$ and projecting the update back is equivalent to Shampoo with Adam momentum.**

---

## 2. Algorithm

```
Initialize: L₀ = ε I_m, R₀ = ε I_n, m̃₀ = 0, ṽ₀ = 0, t = 0

For each step t:
  t ← t + 1
  G_t ← ∇L(W_{t-1})                       # gradient

  # Update Kronecker accumulators
  L_t ← L_{t-1} + G_t G_t^T
  R_t ← R_{t-1} + G_t^T G_t

  # Update eigenbases (every T_eig steps)
  if t mod T_eig == 0:
    Q_L, Λ_L ← eigh(L_t)                  # eigendecomposition: O(m³)
    Q_R, Λ_R ← eigh(R_t)                  # O(n³)

  # Project gradient into eigenbasis
  G̃_t ← Q_L^T @ G_t @ Q_R                # O(mn(m+n))

  # Run Adam on the projected gradient
  m̃_t ← β₁ m̃_{t-1} + (1-β₁) G̃_t        # projected 1st moment
  ṽ_t ← β₂ ṽ_{t-1} + (1-β₂) G̃_t²       # projected 2nd moment (elementwise)
  m̂_t ← m̃_t / (1-β₁ᵗ)
  v̂_t ← ṽ_t / (1-β₂ᵗ)
  Δ̃_t ← m̂_t / (√v̂_t + ε)               # Adam update in eigenbasis

  # Project back to original space
  ΔW_t ← Q_L @ Δ̃_t @ Q_R^T              # O(mn(m+n))

  # Weight update with decoupled weight decay
  W_t ← W_{t-1} - η (ΔW_t + λ W_{t-1})
```

### Why eigendecomposition instead of matrix roots?

[[Shampoo]] requires $L_t^{-1/4}$, computed via eigendecomposition anyway: $L = Q\Lambda Q^\top \Rightarrow L^{-1/4} = Q\Lambda^{-1/4}Q^\top$. SOAP recognizes that you can keep $Q_L, Q_R$ directly and apply them as rotation matrices — avoiding the need to materialize $L^{-1/4}$ explicitly. The projected Adam's $\hat{v}_t$ plays the role of $\Lambda^{-1/2}$, so SOAP absorbs the diagonal scaling into the Adam moment estimates.

---

## 3. Advantages Over Shampoo

**Momentum is natural**: since Adam accumulates $\tilde{m}_t$ and $\tilde{v}_t$ in the eigenbasis, momentum is correctly maintained even when the eigenbasis is updated. Shampoo + momentum has an inconsistency: updating $P_L$ mid-training means the old momentum vectors are no longer in the right coordinate system. SOAP avoids this entirely.

**Simpler implementation**: no matrix root computation. Eigendecomposition is cheaper and more numerically stable than computing $A^{-1/4}$ directly (especially for near-singular or ill-conditioned accumulators).

**Familiar hyperparameters**: SOAP uses Adam's $\beta_1, \beta_2, \epsilon$ directly. Shampoo requires additional tuning of the preconditioner update frequency and initialization.

**Amortization**: the eigendecomposition update frequency $T_{\text{eig}}$ (typically every 10–100 steps) gives flexibility. Between updates, the eigenbasis is stale but the Adam moments adapt — a graceful degradation rather than complete loss of second-order information.

---

## 4. Correctness: Equivalence to Shampoo + Adam

**Claim**: SOAP is exactly Shampoo + Adam momentum in the limit where $T_{\text{eig}} = 1$ (eigenbasis updated every step).

**Proof sketch**: at step $t$, the Shampoo update with Adam-accumulated preconditioned moments is:

$$
\Delta W_t = L_t^{-1/4}\, \frac{\tilde{m}_t^{\text{precond}}}{\sqrt{\tilde{v}_t^{\text{precond}}} + \epsilon}\, R_t^{-1/4}
$$

where $\tilde{m}_t^{\text{precond}}$ and $\tilde{v}_t^{\text{precond}}$ are the Adam moments accumulated in the Shampoo-preconditioned space. SOAP achieves the same by working in the eigenbasis: $L_t^{-1/4} = Q_L \Lambda_L^{-1/4} Q_L^\top$, so the left-right application decomposes as a rotation + diagonal scaling. The diagonal scaling is absorbed into Adam's second moment, leaving only the rotation matrices $Q_L, Q_R$ as explicit operations.

When $T_{\text{eig}} > 1$, the eigenbasis is frozen between updates — SOAP becomes an approximation, but the Adam moments continue to adapt to gradient statistics within the (slightly stale) rotated frame.

---

## 5. Computational Cost

Per step (assuming $W \in \mathbb{R}^{m \times n}$, $m \geq n$):

| Operation | Cost |
|-----------|------|
| Accumulate $L_t, R_t$ | $O(m^2 + n^2)$ per step |
| Eigendecomposition (every $T_{\text{eig}}$ steps) | $O(m^3 + n^3) / T_{\text{eig}}$ amortized |
| Project gradient: $Q_L^\top G_t Q_R$ | $O(mn \cdot m + mn \cdot n) = O(mn(m+n))$ |
| Adam on $\tilde{G}$ | $O(mn)$ |
| Project back: $Q_L \tilde{\Delta} Q_R^\top$ | $O(mn(m+n))$ |

The projection operations are the bottleneck — two matrix multiplications per step. For a 4096×4096 weight matrix with $T_{\text{eig}} = 100$:
- Eigendecomposition amortized: $\sim 4096^3 / 100 \approx 687$M FLOPs/step
- Projection: $\sim 2 \times 4096^2 \times 4096 \approx 137$B FLOPs/step — this dominates

In practice SOAP is applied to weight matrices only (not embeddings or biases), and the memory overhead for $Q_L, Q_R, L_t, R_t$ is $O(m^2 + n^2)$.

---

## 6. Empirical Results

([Vyas et al., 2024](https://arxiv.org/abs/2409.11321)) benchmarks SOAP against AdamW and Shampoo on language model pre-training:

- **vs. AdamW**: SOAP reaches the same validation loss in significantly fewer steps (~20–40% improvement on GPT-2 scale)
- **vs. Shampoo**: SOAP is more stable due to coherent momentum; comparable convergence
- **Hyperparameter sensitivity**: SOAP is more robust than Shampoo — using Adam defaults ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$) works well without tuning

The step efficiency gain comes from the second-order preconditioning: by accounting for gradient correlations along weight matrix rows and columns, SOAP takes more curvature-aligned steps than the diagonal AdamW.

---

## 7. Relationship to Other Optimizers

| Optimizer | Preconditioner | Momentum | Basis |
|-----------|---------------|---------|-------|
| [[AdamW]] | Diagonal ($\hat{v}_t^{-1/2}$) | Adam | Standard |
| [[Shampoo]] | Kronecker ($L^{-1/4} \otimes R^{-1/4}$) | None (original) | Standard |
| **SOAP** | Kronecker (via eigenbasis diagonal) | Adam | Shampoo eigenbasis |
| [[Muon]] | Spectral norm (orthogonalization) | Nesterov | N/A |

SOAP sits between AdamW and full Shampoo: it captures the same off-diagonal curvature as Shampoo, adds Adam's robust momentum, and avoids matrix root computation — at the cost of two projection matrix multiplications per step.
