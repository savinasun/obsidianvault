### [[dl_moc]]  
# **Muon**

**Tags**:: #dl_theo #learning
**Links**:: [[Optimization Algorithms]] · [[AdamW]] · [[SOAP]] · [[Shampoo]]

---

**Muon** (*Momentum + Orthogonalization*) is a gradient optimizer for weight matrices that computes Nesterov momentum and then **orthogonalizes the update** via Newton-Schulz iterations ([Jordan et al., 2024](https://github.com/KellerJordan/modded-nanogpt)). It is grounded in the theory of steepest descent under the spectral norm, and has shown competitive or superior results to [[AdamW]] on language model pre-training with significantly fewer hyperparameters.

---

## 1. Motivation: Steepest Descent Under Non-Euclidean Norms

Standard gradient descent minimizes the loss subject to a Euclidean (Frobenius) constraint on the update:

$$
\Delta W^* = \arg\min_{\Delta W} \mathcal{L}(W + \Delta W) \quad \text{s.t.} \quad \|\Delta W\|_F \leq \delta
$$

The solution is $\Delta W^* \propto -G$ (the negative gradient), which treats all parameter directions as equally costly.

But for a weight matrix $W \in \mathbb{R}^{m \times n}$, the spectral norm $\|\cdot\|_2$ (largest singular value) is a natural measure of the update's "strength" — it bounds how much the linear map $W$ can change the magnitude of any input direction. The steepest descent under a **spectral norm** constraint:

$$
\Delta W^* = \arg\min_{\langle G, \Delta W \rangle} \quad \text{s.t.} \quad \|\Delta W\|_2 \leq \delta
$$

is solved by the **polar factor** of $G$ — the orthogonal factor in the polar decomposition $G = UP$ where $U$ has orthonormal columns and $P$ is positive semidefinite.

$$
\Delta W^* = U V^\top \quad \text{where} \quad G = U \Sigma V^\top \text{ (SVD)}
$$

This means the optimal update under spectral norm is the matrix whose singular values are all equal to 1 — i.e., an **isometry** (orthogonal matrix if square, partial isometry if rectangular). Muon approximates this without computing the full SVD.

### Connection to Modular Duality

Bernstein & Newhouse ([2024](https://arxiv.org/abs/2408.00065)) formalize this via *modular duality*: for each layer, the natural update norm is the operator norm (spectral norm), and the steepest descent direction is the dual norm gradient — the polar factor. Muon is the practical implementation of this principle.

---

## 2. Newton-Schulz Orthogonalization

Computing the full SVD costs $O(\min(m,n) \cdot mn)$ — expensive at every step. Muon instead uses a **Newton-Schulz iteration** to approximate the polar factor:

$$
X_0 = G / \|G\|_F, \qquad X_{k+1} = \frac{3}{2} X_k - \frac{1}{2} X_k X_k^\top X_k
$$

This iteration converges quadratically to the orthogonal polar factor of $G$. Starting from $\|X_0\|_2 \leq 1$ (ensured by Frobenius normalization), **5 iterations** are sufficient in practice to orthogonalize a gradient matrix to machine precision.

Each Newton-Schulz step costs two matrix multiplications — $O(m^2 n + mn^2)$ total, comparable to one transformer forward pass for a weight matrix. This is much cheaper than eigendecomposition or SVD.

**Quintic variant** (used in practice for faster convergence):

$$
X_{k+1} = X_k \left(\frac{15}{8} - \frac{5}{4}X_k^\top X_k + \frac{3}{8}(X_k^\top X_k)^2\right)
$$

This converges to the polar factor in fewer iterations (typically 5 steps suffice).

---

## 3. Algorithm

```
Initialize: M₀ = 0 (momentum buffer), t = 0

For each step t (applied to hidden-layer weight matrices only):
  t ← t + 1
  G_t ← ∇L(W_{t-1})                      # gradient
  
  # Nesterov momentum
  M_t ← μ M_{t-1} + G_t                  # update momentum buffer
  G̃_t ← μ M_t + G_t                      # Nesterov lookahead gradient

  # Orthogonalize via Newton-Schulz (5 iterations)
  X ← G̃_t / ‖G̃_t‖_F                     # normalize to unit Frobenius norm
  for k in range(5):
    X ← (3/2) X - (1/2) X X^T X          # Newton-Schulz step
  O_t ← X                                # approximate polar factor

  # Update
  W_t ← W_{t-1} - η O_t                  # no adaptive scaling, no weight decay on W
```

**Key differences from AdamW**:
- No per-parameter adaptive scaling ($\hat{v}_t$ term) — the update norm is controlled globally via the learning rate
- No second moment state — Muon stores only the momentum buffer $M_t$ (half the memory of AdamW)
- Updates are orthogonal matrices (all singular values = 1) — the update magnitude is uniform in all singular directions

---

## 4. Practical Usage: Hybrid with AdamW

Muon is **not** applied to all parameters. The standard recipe:

| Parameter type | Optimizer |
|---------------|-----------|
| Hidden-layer weight matrices (Q, K, V, O, FFN) | **Muon** |
| Input embedding table | [[AdamW]] |
| Output (LM head) weight | [[AdamW]] |
| Biases, LayerNorm params | [[AdamW]] |

**Why not embeddings?** Embeddings are sparse — most rows receive zero gradient per step. Orthogonalization is not meaningful for sparse updates and would distort the embedding space. AdamW's per-parameter adaptivity handles this correctly.

**Why not the LM head?** The output projection couples to the vocabulary distribution; orthogonal updates can destabilize the logit scale. AdamW's scaling adapts to this naturally.

This hybrid approach is used in the [NanoGPT speed-run](https://github.com/KellerJordan/modded-nanogpt) and subsequent LLM training experiments.

---

## 5. Why Orthogonalization Works: Intuition

### Gradient correlation in weight space

A weight matrix $W$ applies a linear map. The singular value decomposition $W = U \Sigma V^\top$ reveals that $U$ and $V$ are the "output" and "input" basis vectors, while $\Sigma$ controls scaling. During training, gradients tend to concentrate in the top singular directions of $W$ — the most "active" input-output pairs. An Adam-style update would amplify these already-large gradient components (since $\hat{v}_t$ tracks gradient variance per coordinate, not per singular direction), potentially creating imbalanced singular value dynamics.

Orthogonalizing the gradient redistributes update energy **equally across all singular directions**. This acts like a form of spectral regularization on the update — preventing any single direction from dominating.

### Connection to Riemannian optimization

Muon's update lives on the Stiefel manifold (the set of matrices with orthonormal columns). Riemannian gradient descent on this manifold naturally prevents weight matrices from collapsing (all singular values going to zero) or exploding in specific directions. This may contribute to improved training stability relative to AdamW.

---

## 6. Empirical Results

From the NanoGPT speed-run benchmark (training GPT-2 124M on FineWeb):

- **Muon reaches the AdamW baseline loss in ~40% fewer steps** (or equivalently, lower loss at the same step budget)
- **Memory**: Muon uses 1 optimizer state per parameter (vs. 2 for AdamW) — ~50% reduction in optimizer memory for the weight matrices
- **Hyperparameter robustness**: orthogonalization removes the need to tune $\beta_2$ and $\epsilon$; only $\mu$ (momentum) and $\eta$ (learning rate) matter

Results at larger scale (GPT-2 1.5B, Llama-style models) show consistent improvements, though the relative gain tends to narrow with scale — second-order methods benefit proportionally more from the curvature correction at smaller model sizes.

---

## 7. Theoretical Properties

**Scale invariance**: if the gradient $G$ is scaled by a constant $c$, the polar factor is unchanged: $\text{polar}(cG) = \text{polar}(G)$. This makes Muon robust to learning rate sensitivity in a way that Adam is not — the direction of the update doesn't depend on gradient magnitude, only the step size $\eta$.

**No second moment = implicit regularization**: Adam's $\hat{v}_t$ adapts step sizes to gradient variance. In practice, directions with high gradient variance get smaller updates. Muon ignores variance — all directions get equal-magnitude updates. This is a stronger regularization signal but can be harmful when gradient variance is genuinely informative (e.g., sparse gradients in embeddings — hence the exclusion).

**Steepest descent optimality**: under the spectral norm constraint, the orthogonalized gradient is the exact steepest descent direction. In contrast, the Frobenius-normalized gradient (standard GD) and Adam's scaled gradient both correspond to different implicit norm constraints, neither of which is the spectral norm.

---

## 8. Open Questions

- **Optimal orthogonalization frequency**: current implementations orthogonalize every step. Could amortized orthogonalization (every $k$ steps) trade accuracy for speed?
- **Interaction with learning rate schedules**: Muon's scale invariance means cosine decay interacts differently — the effective step size is purely $\eta$, not modulated by gradient variance. Standard LR schedules may need recalibration.
- **Beyond square/near-square matrices**: Newton-Schulz convergence is fastest when $\sigma_{\max}/\sigma_{\min}$ is close to 1 (well-conditioned gradient). For very rectangular matrices with ill-conditioned gradients, convergence may require more iterations.
- **Combination with SOAP**: can Muon's orthogonalization be combined with SOAP's eigenbasis rotation for even better preconditioning? Initial experiments (early 2025) suggest possible synergy.

---

## 9. Summary vs. Related Optimizers

| Property | [[AdamW]] | [[Shampoo]] | [[SOAP]] | **Muon** |
|----------|-----------|-------------|----------|---------|
| Update direction | Scaled gradient | Kronecker-precond. gradient | Eigenbasis Adam | Polar factor of momentum |
| Curvature information | Diagonal (per-param variance) | Row/column correlations | Row/column correlations | Spectral (all singular directions equal) |
| Momentum | Adam (1st/2nd moment) | None (original) | Adam in eigenbasis | Nesterov |
| Memory / param | 2 states | $O(m^2+n^2)$ per matrix | $O(m^2+n^2)$ per matrix | 1 state |
| Scale invariant | No | No | No | **Yes** |
| Applies to embeddings | Yes | Yes | Yes | **No** (use AdamW) |
| Compute overhead | Low | High (matrix roots) | Medium (projections) | Low (5 NS iterations) |
