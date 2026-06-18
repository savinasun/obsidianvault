### [[dl_moc]]  
# **Linear Attention**

**Tags**:: #architecture #llm
**Links**:: [[Attention]] · [[Transformer]] · [[Mamba]] · [[RNN]] · [[LLMs]]


Standard [[Attention|self-attention]] runs in $O(n^2 d)$ time and $O(n^2)$ memory due to the full $n \times n$ attention matrix. **Linear attention** replaces the softmax with a kernel that lets you reorder the matrix multiplications, dropping the quadratic dependence on sequence length to $O(nd^2)$.

## The Kernel Trick
---
Scaled dot-product attention computes:
$$
\text{Attn}(Q,K,V)_i = \frac{\sum_j \exp(q_i^\top k_j / \sqrt{d}) \, v_j}{\sum_j \exp(q_i^\top k_j / \sqrt{d})}
$$

The softmax couples all $j$ positions through the row normalization, forcing the $n \times n$ matrix to be materialized. **Linear attention** replaces $\exp(q^\top k / \sqrt{d})$ with a factored kernel $\phi(q)^\top \phi(k) \ge 0$:

$$
\text{LinearAttn}(Q,K,V)_i = \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j) \, v_j^\top\right)}{\phi(q_i)^\top \left(\sum_j \phi(k_j)\right)}
$$

By **associativity** of matrix multiplication, the sums $\sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{d \times d}$ and $\sum_j \phi(k_j) \in \mathbb{R}^d$ are **independent of the query** — compute them once and reuse across all $i$. Total cost: $O(nd^2)$ instead of $O(n^2 d)$.

### Common Feature Maps $\phi$

| Feature map | Definition | Notes |
|---|---|---|
| ELU + 1 | $\phi(x) = \text{ELU}(x) + 1$ | Non-negative; Katharopoulos et al. 2020 |
| Squared ReLU | $\phi(x) = \text{ReLU}(x)^2$ | Simple; used in sub-LN variants |
| RBF (FAVOR+) | Random Fourier Features | Unbiased softmax approximation; Performer |
| Identity | $\phi(x) = x$ | Simplest but allows negative similarities |

The choice of $\phi$ trades off between approximation quality (how closely it mimics softmax) and numerical stability.

## Causal (Autoregressive) Form
---
For causal (left-to-right) modeling, the sum runs only over $j \le i$:
$$
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}, \quad \text{where } S_i = \sum_{j \le i} \phi(k_j) v_j^\top, \quad z_i = \sum_{j \le i} \phi(k_j)
$$

This admits a simple **recurrence** that mirrors an [[RNN]]:
$$
S_i = S_{i-1} + \phi(k_i) \, v_i^\top \in \mathbb{R}^{d \times d}
$$
$$
z_i = z_{i-1} + \phi(k_i) \in \mathbb{R}^{d}
$$
$$
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
$$

**Training**: use the parallel form (left side, $O(nd^2)$).  
**Inference**: use the recurrent form (right side, $O(d^2)$ fixed state, $O(1)$ per step).

## Decay and Forgetting
---
Pure linear attention **accumulates** all past context into $S$ without forgetting. In practice most variants add a **decay factor** $\lambda \in (0,1)$ per step or per channel:
$$
S_i = \lambda \, S_{i-1} + \phi(k_i) v_i^\top
$$

This creates an **exponential moving average** over past $(k, v)$ pairs — recent tokens contribute more than distant ones. Learnable per-channel decay allows different "memory timescales" per feature dimension.

Without decay, the state is dominated by early tokens on very long sequences. With too-small decay, the model forgets useful long-range context. Most modern variants make decay **input-dependent** (see [[#Gated Linear Attention]]).

## Chunk-Parallel Training
---
The recurrence is fast at inference but sequential to train. **Chunkwise parallel** computation resolves this:

1. Split the sequence into chunks of size $C$.
2. Within each chunk: use the **parallel (attention-matrix) form** — parallelizable over the chunk.
3. Across chunks: carry the recurrent state $S$ forward — sequential but only $n/C$ steps.

Total cost: $O(nCd + nd^2/C)$, minimized at $C = d$ → $O(nd^{3/2})$. In practice $C$ is tuned for hardware (e.g., $C = 64$–$256$).

## Notable Models
---

### Performer (2020)
[Choromanski et al.](https://arxiv.org/abs/2009.14794) uses **random Fourier features** (FAVOR+) to construct an *unbiased* approximation of the softmax kernel:
$$
\phi(x) = \frac{1}{\sqrt{m}} \left[\exp(\omega_r^\top x - \|\omega_r\|^2/2)\right]_{r=1}^m, \quad \omega_r \sim \mathcal{N}(0, I)
$$

Unbiased but high-variance; memory cost shifts from $O(n^2)$ to $O(nm)$ where $m \ll n$.

### RWKV (2023)
[Peng et al.](https://arxiv.org/abs/2305.13048) replaces attention with a **WKV** (weighted key-value) operator using time-decay:
$$
\text{wkv}_t = \frac{\sum_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i<t} e^{-(t-1-i)w + k_i} + e^{u+k_t}}
$$
where $w$ is a learnable per-channel decay and $u$ is a bonus for the current token. Fully recurrent at inference; trained in parallel via prefix-sum.

### RetNet (2023)
[Sun et al.](https://arxiv.org/abs/2307.08621) introduces a **retention** mechanism with three equivalent representations:
- **Parallel**: like attention with a decay mask — trainable on GPU
- **Recurrent**: $O(1)$ inference
- **Chunkwise**: balanced efficiency

Retention uses a fixed geometric decay per head: $\gamma_h = 1 - 2^{-5 - h}$, and complex-valued rotations for positional encoding (analogous to [[RoPE]]).

### Gated Linear Attention (2023)
[Yang et al.](https://arxiv.org/abs/2312.06635) extends linear attention with an **input-dependent gate**:
$$
S_i = G_i \odot S_{i-1} + \phi(k_i) v_i^\top, \quad G_i = \sigma(W_g x_i)
$$

The gate $G_i \in [0,1]^{d \times d}$ (or structured as $\alpha_i \beta_i^\top$) allows the model to selectively forget parts of the state. Enables a hardware-efficient chunkwise CUDA kernel. Input-dependent gating is the key differentiator from models with fixed decay.

## Connection to SSMs and Mamba
---
[[Mamba]] (SSMs) and linear attention are two perspectives on the same structure. The **SSD (State Space Duality)** framework in Mamba-2 makes this explicit:

An SSM recurrence:
$$
x_i = A_i x_{i-1} + B_i u_i, \quad y_i = C_i^\top x_i
$$

is equivalent to linear attention with $\phi(k) = B$, $\phi(q) = C$, and a structured decay/transition matrix playing the role of $\lambda$. Specifically:
- **Linear attention with scalar decay** ↔ SSM with $A_i = \lambda I$
- **GLA with input-dependent gate** ↔ Selective SSM (Mamba-style)
- **S4/S6** ↔ linear attention with specific structured kernels

The SSD framing enables tensor-core-friendly implementations and unifies the two research lines.

## Complexity Comparison
---

| Model | Training | Inference / step | Memory (inference) | Precise retrieval |
|---|---|---|---|---|
| Full [[Attention]] | $O(n^2 d)$ | $O(n \cdot d)$ KV cache | $O(n)$ grows | ✓ Exact |
| Linear Attention | $O(n d^2)$ | $O(d^2)$ fixed state | $O(d^2)$ fixed | ✗ Approximate |
| [[Mamba]] / SSM | $O(nNd)$ scan | $O(Nd)$ fixed state | $O(N)$ fixed | ✗ Approximate |
| [[RNN]] / LSTM | $O(nd^2)$ sequential | $O(d^2)$ fixed state | $O(d^2)$ fixed | ✗ Approximate |

$N$ = SSM state size (typically $N \ll d$).

## The State Compression Tradeoff
---
Linear attention's fundamental limitation: all past context must be **lossily compressed** into the fixed $d \times d$ state. Full attention's KV cache grows with context and allows **exact lookup** of any past token.

Tasks that require retrieving a specific detail from far back in context (e.g., "find the phone number mentioned in paragraph 1") are hard for linear attention. Tasks that benefit from long-horizon compression or streaming (long audio, genomics, embodied sensorimotor sequences) are where linear attention excels.

**Hybrid architectures** (e.g., Griffin, Jamba, Zamba) address this by mixing a few full-attention layers with many linear-attention / SSM layers: the sparse attention layers handle precise retrieval; the linear layers handle long-range compression cheaply.
