### [[dl_moc]]  
# **Mamba / State Space Models**

**Tags**:: #architecture #llm
**Links**:: [[Transformer]] · [[Attention]] · [[Linear Attention]] · [[LLMs]] · [[RoPE]]


**State Space Models (SSMs)** are a family of sequence models that offer an alternative to the [[Transformer]] architecture. While Transformers scale as $O(n^2)$ in sequence length due to full attention, SSMs achieve **linear-time inference and training** through a recurrent formulation.

## State Space Models (SSMs)
---
An SSM maps an **input sequence $u(t)$** to an **output sequence $y(t)$** through a **latent state $x(t)$**:

$$
\begin{aligned}
x'(t) &= Ax(t) + Bu(t) \\
y(t) &= Cx(t) + Du(t)
\end{aligned}
$$

- $A \in \mathbb{R}^{N \times N}$: state transition matrix (how the latent state evolves)
- $B \in \mathbb{R}^{N \times 1}$: input projection
- $C \in \mathbb{R}^{1 \times N}$: output projection
- $D$: skip connection (often $D = 0$)

This continuous-time system is discretized for sequence modeling:
$$
\begin{aligned}
x_k &= \bar{A} x_{k-1} + \bar{B} u_k \\
y_k &= C x_k
\end{aligned}
$$
where $\bar{A}, \bar{B}$ are the discretized matrices via zero-order hold (ZOH).

## S4: Structured State Space Sequences
---
S4 ([Gu et al., 2021](https://arxiv.org/abs/2111.00396)) made SSMs practical for deep learning by:

1. **HiPPO initialization**: initialize $A$ using the HiPPO (High-order Polynomial Projection Operators) framework — this compresses the history of the input signal into the state optimally
2. **Diagonal plus low-rank (DPLR) structure**: represent $A$ in a structured form that enables $O(N \log N)$ computation via the convolution theorem

**Key insight**: an SSM can be computed either as a **recurrence** (efficient for inference, $O(1)$ per step) or as a **convolution** (efficient for training, parallelizable over the sequence):
$$
y_k = \sum_{j=0}^{k} \bar{C}\bar{A}^{k-j}\bar{B}\, u_j = (K * u)_k, \quad K_j = \bar{C}\bar{A}^j\bar{B}
$$

==**Training uses the convolutional mode (fast on GPUs); inference uses the recurrent mode (constant memory per step).**==

## Mamba: Selective SSMs
---
[Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) identified a fundamental limitation of prior SSMs: **$A$, $B$, $C$ are fixed** for all inputs — the model cannot selectively remember or forget based on content. This makes them poor at **content-based reasoning** (e.g., answering questions about specific tokens in a long context).

**Mamba's solution**: make $B$, $C$, and $\Delta$ (the discretization step) **functions of the input**:
$$
B_k = s_B(u_k), \quad C_k = s_C(u_k), \quad \Delta_k = \tau_\Delta(\text{Linear}(u_k))
$$

This **selective mechanism** allows the model to dynamically decide:
- What information to integrate from new input (via $B_k$)
- What part of the state to expose as output (via $C_k$)
- How much to discretize the continuous-time dynamics (via $\Delta_k$ — larger $\Delta$ → more forgetting)

**Hardware-aware implementation**: naive selective SSMs can't use the convolution trick (since $K$ now depends on input). Mamba uses a **parallel scan** (prefix sum) algorithm in CUDA with kernel fusion to run efficiently despite the input-dependent parameters.

## Mamba Architecture
---
A Mamba block:

```
Input x (B, L, D)
  ↓  Linear projection (expand to 2D)
  ├→ SSM branch: selective_scan(u, Δ, A, B, C)
  └→ Gate branch: SiLU(z)
  ↓  Element-wise multiply (gated output)
  ↓  Linear projection (down to D)
  ↓  Residual add
```

No attention mechanism — the model is **purely recurrent**.

## Mamba-2
---
[Gu & Dao, 2024](https://arxiv.org/abs/2405.21060) reformulated Mamba's selective scan as a **semi-separable matrix multiplication**, connecting SSMs to a theoretical framework that also unifies with linear attention. This enables:
- Even faster hardware implementation via tensor-core-friendly operations
- Cleaner theoretical analysis
- SSD (State Space Duality) — a family that includes both SSMs and [[Linear Attention]] as special cases

## Comparison: Mamba vs. Transformer
---

| | Transformer | Mamba |
|---|---|---|
| **Training complexity** | $O(L^2 D)$ attention | $O(LDN)$ scan |
| **Inference per step** | $O(L \cdot D)$ (KV cache) | $O(N \cdot D)$ (fixed state) |
| **Memory (inference)** | $O(L)$ KV cache grows with context | $O(N)$ fixed state |
| **Content-based lookup** | Full attention → exact | Selective state → approximate |
| **Long-range dependencies** | Exact (up to context length) | Compressed into $N$-dim state |
| **Parallelism (training)** | High | High (via parallel scan) |

**Where Mamba excels**: very long sequences (>100K tokens), streaming inference (constant memory), audio and genomics sequences.

**Where Transformers excel**: tasks requiring precise retrieval from long context ("find the phone number mentioned earlier"), in-context learning, and tasks where exact attention patterns matter.

## Hybrid Architectures
---
In practice, **mixing** SSM layers and attention layers often outperforms either alone:

- **Jamba** (AI21 Labs): alternating Mamba and Transformer layers
- **Zamba** (Zyphra): a few attention layers mixed with many Mamba layers
- **Griffin** (DeepMind): gated linear recurrent layers + local attention

The intuition: SSM layers handle long-range context compression efficiently; attention layers provide high-precision retrieval for the most recent context.
