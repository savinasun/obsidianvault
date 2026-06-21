### [[dl_moc]]
# **Attention**

**Tags**:: #architecture #llm
**Links**:: [[Transformer]] · [[Linear Attention]] · [[LLMs]] · [[RoPE]] · [[LLaMA]] · [[MoE]]


Large Language Models ([[LLMs]]) rely heavily on the **attention mechanism**, a core component that enables the model to selectively focus on parts of the input. It’s foundational to the [[Transformer|transformer]] architecture and all its variants (e.g., [[GPTs]], BERT, T5).

## Self-Attention
---
**Self-Attention** (a.k.a. **scaled dot-product attention**) allows each token in a sequence to attend to all other tokens (including itself) to compute contextualized representations.

Given an input matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$ of $n$ tokens (each with $d_{\text{model}}$ dimensions), self-attention is computed as:
$$
\textrm{Attention}(Q,K,V) = \textrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\cdot V
$$
- $n$: sequence length
- $d_{\text{model}}$: hidden dimensionality

We project $X$ into **Query**, **Key**, and **Value** matrices:

- $Q = XW^Q \in \mathbb{R}^{n \times d_k}$
- $K = XW^K \in \mathbb{R}^{n \times d_k}$
- $V = XW^V \in \mathbb{R}^{n \times d_v}$

where
- $d_k$: key/query dimension
- $d_v$: value dimension
- $W^Q, W^K, W^V$ are learned weights of size $\mathbb{R}^{d_{\text{model}} \times d_k}$ or $\mathbb{R}^{d_{\text{model}} \times d_v}$
- $QK^\top \in \mathbb{R}^{n \times n}$: similarity between all pairs of tokens
- $\sqrt{d_k}$: scaling factor to prevent large dot products (see below)
- Output: $\in \mathbb{R}^{n \times d_v}$

**Why Scale by $\sqrt{d_k}$?**
- Dot products $Q \cdot K$ can grow large in magnitude as $d_k$ increases.
- Large values push softmax into extreme regions → vanishing gradients.
- Scaling by $\sqrt{d_k}$ stabilizes the gradient and learning dynamics.


## Masked Self-Attention
---
Used in **causal (autoregressive)** models like GPT to ensure each token can only attend to **previous or current** tokens (not future ones).

A **causal mask** $M \in \mathbb{R}^{n \times n}$ is applied:
$$
M_{ij} = 
\begin{cases}
0 & \text{if } j \le i \\
-\infty & \text{if } j > i
\end{cases}
$$
Then:
$$
\text{MaskedAttention}(Q, K, V) = \text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} + M \right) V
$$
This ensures that token $i$ only uses information from tokens $\le i$.

Q could only see K before it, K could be seen by all Qs after it.

## Cross-Attention
---
Used in **encoder-decoder** architectures (e.g., T5, BART).
- Queries $Q \in \mathbb{R}^{n_t \times d_k}$: from decoder tokens
- Keys $K \in \mathbb{R}^{n_s \times d_k}$: from encoder tokens
- Values $V \in \mathbb{R}^{n_s \times d_v}$: from encoder tokens
$$
\text{CrossAttention}(Q, K, V) = \text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$
This allows decoder tokens to condition on the entire encoded input.

**Multi-Modal Models**: encoder trained on images, speech → context aware embeddings → text based decoder to generate captions or audio-prompt responses



## Multi-Head Attention (MHA)
---
Multi-Head Attention extends self-attention by learning **multiple attention heads** in parallel.

For $h$ heads:
+ $n$ is the number of tokens in the input sequence
- Project inputs to $h$ sets of $Q_i, K_i, V_i \in \mathbb{R}^{n \times d_k}$
- Projection matrices $W_i^Q, W_i^K, W_i^V$ are learned weights of size $\mathbb{R}^{d_{\text{model}} \times d_k}$
- Each head computes its own attention
- Concatenate the heads and project:
$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \textrm{Attention}\left(XW_i^Q, XW_i^K, XW_i^V\right),
$$
$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \in \mathbb{R}^{n\times d_{model}},
$$
where $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ is the output projection.

$$
\begin{aligned}
h_i &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\cdot V \\
&= \text{softmax}\left(\frac{XW_i^Q {W_i^K}^T X^T}{\sqrt{d_k}}\right)\cdot XW_i^V \\
\end{aligned}
$$
Note that in [[RMSNorm]] layer, 
$$
X = \frac{X}{\sqrt{\|X\|_2 + \epsilon}}\cdot \Gamma = \tilde{X}\cdot\Gamma$$
where $\Gamma \in\mathbb{R}^{d_{\text{model}}}$.

$$
\begin{aligned}
h_i &= \text{softmax}\left(\frac{\tilde{X}\Gamma W_i^Q {W_i^K}^T \Gamma {\tilde{X}}^T}{\sqrt{d_k}}\right)\cdot\tilde{X}\Gamma W_i^V \\
\end{aligned}
$$

Denote the attention weights matrix of head $i$ as $A_i$, based on the tricks in [[Linear Algebra#Four Types of Matrix Multiplication|matrix multiplication]], we may rewrite the outputs of attention block as
$$
X = X+ \Sigma_i A_i \cdot \tilde{X}\cdot\Gamma \cdot (W_i^V) \cdot ({W_i^O}^T)^T
$$
where $A_i\in\mathbb{R}^{n\times n}$ is the $i$-th head of attention matrix.

**Benefits**
- **Allows the model to jointly attend to information from different subspaces.**
- **Provides richer representations than a single attention head.**


## Grouped-Query Attention (GQA)
---
![[attention_03.png]]

**Grouped-Query Attention** is a memory-efficient variant used in [[LLaMA]], PaLM, etc.

- Use **many query heads**, but **fewer key/value heads**:
- $h_q$ query heads, $h_{kv}$ key/value heads, where $h_q > h_{kv}$
- A group of query heads shares the same $K, V$

This reduces the memory/computation cost for storing and computing large key/value matrices.

- $Q \in \mathbb{R}^{n \times h_q \times d_k}$
- $K, V \in \mathbb{R}^{n \times h_{kv} \times d_k}$
- Each group of $G = h_q / h_{kv}$ query heads uses the same $K, V$

### Benefits

- Significantly reduces attention memory bottleneck
- Enables scaling up query resolution while keeping KV costs manageable


## Attention Heads are Independent and Additive
----
https://transformer-circuits.pub/2021/framework/index.html

+ Dong (2021) showed that without skip connections or multi-layer perceptrons (MLPs), the output converges doubly exponentially to a rank-1 matrix. On the other hand, skip connections and MLPs stop the output from degeneration.
	+ The intuition is that attention matrices ‘**mix**’ information, and **repeated mixing converges to a space that is of low-rank**.
+ The same phenomenon in Graph-Neural-Networks is called **over-smoothing**.



## Paired-Head Attention
---
In normal attention, each head independently computes `q·k` within itself. **Paired heads make adjacent heads cross-attend — head A's queries attend to head B's keys, and vice versa.**

Normal MHA has **independent** heads — they can't share information with each other within a single layer. Paired heads allow **cross-head communication**: head A can leverage patterns that head B detects. This increases expressivity without adding parameters.

Paired heads make **adjacent heads' queries attend to each other's keys**:

```
Normal:    Q1→K1,  Q2→K2  (independent)
Paired:    Q1→K2,  Q2→K1  (cross-attend)
```

This is achieved by interleaving two heads' tokens into a single doubled-length sequence, then running standard causal attention — the causal mask naturally allows cross-head visibility.

- Every head has its **own** Q, K, and V (no sharing)
- Purpose: **cross-head communication** within a single layer
- Tradeoff: the effective window size is halved (since `T` doubles but the sliding window `bm_size` stays the same), so it's only used on selected layers.
## Computational Complexity
---
From [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) Table 1 — comparing per-layer complexity, sequential operations, and maximum path length across layer types ($n$ = sequence length, $d$ = representation dimension, $k$ = kernel size):

![[attention_complexity_table.png|700]]

Self-attention's $O(n^2 \cdot d)$ complexity is the bottleneck for long sequences — all $n^2$ token pairs must be compared. This motivates [[Linear Attention]] and other efficient attention variants that reduce this to $O(n \cdot d^2)$.
