### [[dl_moc]]  
# **Attention**

**Tags**::  
**Links**::

## Self-Attention
---
$$
\textrm{Attention}(Q,K,V) = \textrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\cdot V
$$
where $d_k$ represents the dimension of matrix $K$, or the dimension of head matrix.
Usually we will use $d_K = d_Q = d_V$.
#### Interpretations of Scaled Dot-Product Attention Function
calculate the **scaled dot-product similarities** among each input tokens
→ convert these similarities into percentages by softmax function
→ calculate the self-attention score by using these percentages to weight the values embedding

## Masked Self-Attention
---


## Cross-Attention (Encoder-Decoder Attention)
---
+ K,V from encoder
+ Q from decoder    
→ calculate encoder-decoder attention scores to generate text (these are from the Seq2Seq models)

**Multi-Modal Models**: encoder trained on images, speech → context aware embeddings → text based decoder to generate captions or audio-prompt responses



## Multi-Head Attention
---
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. While with a single attention head, averaging inhibits this. Before jumping to the formula of multi-head attention, we first specify the notations of architecture parameters (here we suppose batch size is 1, and we will omit the dimension for batch size in this section):

- $D$: dimension of the hidden states (or embedding vector), or $d_{\mathrm{model}}$
- $d$: dimension of each head, where $d_q=d_k=d_v=d$
- $H= D / d$: number of heads
- $q$: sequence length

Multi-head attention function is the concatenation of attention scores from all the heads and projected back to the higher dimension:
$$
\textrm{MultiHead}(Q,K,V) = \textrm{Concat}(h_1, \dots, h_H)W^O
$$
where
$$
h_i = \textrm{Attention}\left(XW_i^Q, XW_i^K, XW_i^V\right),
$$
+ $X\in\mathbb{R}^{q\times D}$: hidden states (or outputs from previous module)
+ $W_i^Q\in\mathbb{R}^{D\times d_K}$: projection matrix for query states
+ $W_i^K\in\mathbb{R}^{D\times d_K}$: projection matrix for key states
+ $W_i^V\in\mathbb{R}^{D\times d_V}$: projection matrix for value states
+ $W^O\in\mathbb{R}^{{Hd_V}\times D}$:  projection matrix for outputs
Then
$$
Q = XW_i^Q,\ K=XW_i^K\in\mathbb{R}^{q\times d_K},\ V = XW_i^V\in\mathbb{R}^{q\times d_V}, \quad i=1,\dots, H,
$$
and
$$
\textrm{Concat}(h_1, \dots, h_H)\in\mathbb{R}^{q\times Hd_V}.
$$
Denote the attention weights matrix of head $i$ as $A_i$, we may write outputs of attention block as
$$
X = X+ \Sigma_i A_j \cdot X \cdot (W_V)_j \cdot (W_O^T)_j^T
$$
where $A_j$ is the $j$-th column of attention matrix $A\in\mathbb{R}^{q\times q}$.




## Grouped-Query Attention
---
![[attention_03.png]]

