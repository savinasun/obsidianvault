### [[dl_moc]]  
# **Shampoo**

**Tags**:: #dl_theo #learning
**Links**:: [[Optimization Algorithms]] · [[AdamW]] · [[SOAP]] · [[Muon]]

---

**Shampoo** ([Gupta, Koren & Singer, 2018](https://arxiv.org/abs/1802.09568)) is a **second-order optimizer** that preconditions stochastic gradients with a Kronecker-factored approximation to the full matrix AdaGrad preconditioner. It is the theoretical foundation for [[SOAP]] and related structured preconditioning methods.

---

## 1. Motivation: Preconditioning and Curvature

Gradient descent ignores the curvature of the loss surface:

$$
\theta_{t+1} = \theta_t - \eta\, g_t
$$

Newton's method uses the full Hessian to account for curvature:

$$
\theta_{t+1} = \theta_t - H_t^{-1} g_t
$$

but $H_t \in \mathbb{R}^{p \times p}$ (for $p$ parameters) is intractable: storing it costs $O(p^2)$, inverting it costs $O(p^3)$.

**AdaGrad** approximates $H$ by its diagonal:

$$
G_t = \sum_{\tau=1}^t g_\tau g_\tau^\top, \qquad \theta_{t+1} = \theta_t - \eta\, G_t^{-1/2} g_t
$$

The diagonal approximation $\text{diag}(G_t)^{-1/2}$ is exactly Adam's $1/\sqrt{\hat{v}_t}$ (in the full-gradient limit). It ignores all off-diagonal curvature — correlations between different parameters.

**Shampoo** exploits the matrix structure of weight tensors to capture off-diagonal curvature without paying $O(p^2)$.

---

## 2. Kronecker-Factored Preconditioning

For a weight matrix $W \in \mathbb{R}^{m \times n}$ with gradient $G_t \in \mathbb{R}^{m \times n}$, the full AdaGrad accumulator is:

$$
\mathcal{G}_t = \sum_{\tau=1}^t \text{vec}(G_\tau)\,\text{vec}(G_\tau)^\top \in \mathbb{R}^{mn \times mn}
$$

The optimal preconditioned update is $\mathcal{G}_t^{-1/2}\,\text{vec}(g_t)$ — storing $\mathcal{G}_t$ costs $O(m^2 n^2)$ which is completely infeasible.

Shampoo approximates $\mathcal{G}_t \approx (L_t \otimes R_t)^{1/2}$ (Kronecker product), where:

$$
L_t = \sum_{\tau=1}^t G_\tau G_\tau^\top \in \mathbb{R}^{m \times m}, \qquad R_t = \sum_{\tau=1}^t G_\tau^\top G_\tau \in \mathbb{R}^{n \times n}
$$

By the identity $(A \otimes B)^{-1/4} = A^{-1/4} \otimes B^{-1/4}$ for Kronecker products, the preconditioned update becomes:

$$
\boxed{
\Delta W_t = L_t^{-1/4}\, G_t\, R_t^{-1/4}
}
$$

This requires storing only $m \times m$ and $n \times n$ matrices — $O(m^2 + n^2)$ instead of $O(m^2 n^2)$.

### Why $-1/4$, not $-1/2$?

Because $L_t$ and $R_t$ each accumulate the full gradient history, $L_t \otimes R_t$ double-counts it: the accumulator satisfies $\mathcal{G}_t \approx (L_t \otimes R_t)^{1/2}$ (exact for a single rank-1 gradient; a bound in general — Gupta et al., Lemma 8). The full-matrix AdaGrad target is therefore:

$$
\mathcal{G}_t^{-1/2} \approx (L_t \otimes R_t)^{-1/4} = L_t^{-1/4} \otimes R_t^{-1/4}
$$

using the exact identity $(A \otimes B)^s = A^s \otimes B^s$ — each Kronecker factor inherits half of the $-1/2$ exponent.

In matrix form, applying $L^{-1/4}$ on the left and $R^{-1/4}$ on the right is equivalent to applying $L^{-1/4} \otimes R^{-1/4}$ to the vectorized gradient.

---

## 3. Full Algorithm

```
Initialize: L₀ = ε I_m, R₀ = ε I_n (small diagonal to ensure invertibility)

For each step t:
  g_t ← stochastic gradient for weight W
  L_t ← L_{t-1} + g_t g_t^T          # left accumulator update
  R_t ← R_{t-1} + g_t^T g_t          # right accumulator update

  if t mod T_inv == 0:               # compute matrix roots every T_inv steps
    P_L ← L_t^{-1/4}                 # expensive: O(m³)
    P_R ← R_t^{-1/4}                 # expensive: O(n³)
  
  ΔW_t ← P_L @ g_t @ P_R            # preconditioned update: O(m·n·(m+n))
  W_t ← W_{t-1} - η ΔW_t
```

**Computing matrix roots**: the dominant cost is $L^{-1/4}$ and $R^{-1/4}$. These are computed via eigendecomposition $L = Q\Lambda Q^\top$, then $L^{-1/4} = Q\Lambda^{-1/4}Q^\top$. Cost: $O(m^3)$ per update of $P_L$, amortized by computing every $T_{\text{inv}}$ steps (typically 100–1000).

---

## 4. Extension to Higher-Order Tensors

For a 4D convolution weight $W \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times k \times k}$, Shampoo maintains one preconditioner per dimension:

$$
L^{(1)} \in \mathbb{R}^{C_\text{out} \times C_\text{out}}, \quad L^{(2)} \in \mathbb{R}^{C_\text{in} \times C_\text{in}}, \quad L^{(3)}, L^{(4)} \in \mathbb{R}^{k \times k}
$$

and applies $L^{(1),-1/8} \times_1 L^{(2),-1/8} \times_2 L^{(3),-1/8} \times_3 L^{(4),-1/8} \times_4 G_t$ where $\times_i$ is the mode-$i$ tensor product. The exponent is now $-1/8$ (= $-1/2 \div 4$ dimensions).

---

## 5. Distributed Shampoo

**Scalable Second-Order Optimization** ([Anil et al., 2020](https://arxiv.org/abs/2002.09018)) makes Shampoo practical at the scale of modern LLM training:

- **Blocking**: for very large matrices, split into smaller blocks and apply Shampoo block-wise
- **Distributed preconditioning**: compute matrix roots in parallel across workers; each worker owns a subset of preconditioner matrices
- **Coupled Newton steps**: use iterative matrix root approximation (Coupled Newton iterations) instead of eigendecomposition — converges in a fixed number of steps, cheaper and more numerically stable for large matrices

Distributed Shampoo was used at Google to train production-scale language models, demonstrating ~1.35× wallclock speedup over Adam for the same loss.

---

## 6. Convergence and Theory

For convex problems, Shampoo provably converges faster than AdaGrad when the gradient correlation structure matches the Kronecker factorization. In the deep learning setting, the convergence guarantees are less tight, but the empirical picture is consistent:

- Shampoo converges in **fewer steps** (better sample efficiency) than Adam
- At small model sizes, the per-step overhead dominates and Shampoo is slower in wallclock
- At large model sizes (GPT-scale), the compute budget shifts in Shampoo's favor

**Key intuition**: Adam's diagonal preconditioner misses off-diagonal gradient correlations. A weight matrix row that has correlated gradient components will be over-stepped in some directions and under-stepped in others. Shampoo's Kronecker preconditioner corrects for this, enabling more aggressive (and directionally correct) updates.

---

## 7. Limitations

- **Memory**: $O(m^2 + n^2)$ per weight matrix — non-trivial for large matrices. For a 4096×4096 attention weight, $L_t$ alone is ~16.8M floats ≈ 67 MB.
- **Compute for matrix roots**: despite amortization, eigendecompositions for large blocks are a significant overhead
- **No momentum**: original Shampoo lacks Adam-style momentum; this matters for training stability
- **Hyperparameter complexity**: the preconditioning frequency $T_{\text{inv}}$, blocking strategy, and epsilon initialization all require tuning

[[SOAP]] addresses the momentum issue and simplifies the algorithm by reframing Shampoo preconditioning as Adam in a rotated basis.

---

## 8. Comparison

| Property | AdaGrad / Adam | Shampoo | [[SOAP]] |
|----------|---------------|---------|---------|
| Preconditioner | Diagonal | Kronecker ($L \otimes R$) | Kronecker eigenbasis |
| Gradient correlations captured | None | Row-column | Row-column |
| Momentum | Adam: yes | No (original) | Yes (Adam) |
| Memory per matrix $m\times n$ | $O(mn)$ | $O(m^2+n^2)$ | $O(m^2+n^2)$ |
| Step convergence | Slower | Faster | Faster |
| Wallclock at scale | Fast | Competitive | Competitive |
