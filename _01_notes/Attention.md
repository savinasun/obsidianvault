### [[dl_moc]]  
# **Attention**

**Tags**::  
**Links**::


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

### Why Scale by $\sqrt{d_k}$?

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

Denote the attention weights matrix of head $i$ as $A_i$, based on the tricks in [[Linear Algebra#Four Types of Matrix Multiplication|matrix multiplication]], we may rewrite the outputs of attention block as
$$
X = X+ \Sigma_i A_i \cdot X \cdot (W_i^V) \cdot ({W_i^O}^T)^T
$$
where $A_i$ is the $i$-th head of attention matrix $A\in\mathbb{R}^{n\times n}$.

### Benefits

- **Allows the model to jointly attend to information from different subspaces.**
- **Provides richer representations than a single attention head.**


## Grouped-Query Attention
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
