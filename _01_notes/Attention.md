### [[dl_moc]]  
# **Attention**

**Tags**::  
**Links**::

## Self-Attention
---
$$
\textrm{Attention}(Q,K,V) = \textrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\cdot V
$$
where $d_k$ represents the dimension of matrix $K$.
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
Before giving the formula of multi-head attention, we specify the following notations (suppose batch size is 1, and we omit the dimension for batch size here):

- $D$: dimension of the hidden states (or embedding vector), or $d_{\mathrm{model}}$
- $d$: dimension of each head
- $H= D / d$: number of heads
- $q$: sequence length
$$
\textrm{MultiHead}(Q,K,V) = \textrm{Concat}(h_1, \dots, h_H)W^O
$$
where
$$
h_i = \textrm{Attention}\left(XW_i^Q, XW_i^K, XW_i^V\right),
$$
+ $X\in\mathbb{R}^{q\times D}$: hidden states (or outputs from previous module)
+ 
![[attention_03.png]]

$$
X = X+ \Sigma_i A_j \cdot X \cdot (W_V)_j \cdot (W_O^T)_j^T
$$
+ $X \in \mathbb{R}^{q\times D}$: hidden states
+ $A_j$: $j$-th column of attention matrix $A\in\mathbb{R}^{q\times q}$

## Grouped-Query Attention
---