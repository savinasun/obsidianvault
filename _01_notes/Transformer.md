### [[dl_moc]]  
# **Transformer Network Architecture**

**Tags**:: #architecture #llm
**Links**:: [[Attention]] · [[LLMs]] · [[RoPE]] · [[RMSNorm]] · [[Layer Normalization]] · [[Activation Functions]]


The Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) introduced a purely attention-based architecture for sequence-to-sequence tasks, replacing recurrence and convolution entirely. It is now the foundation of virtually all modern deep learning: [[LLMs]], [[ViT|vision models]], and [[VLM|multimodal models]].

## Encoder-Decoder Architecture
---
The original Transformer is an **encoder-decoder** model for tasks like machine translation where input and output distributions differ.

![[transformer_encoder_decoder.png|300]]

### Encoder
Maps an input sequence $x_{1:n}$ to **context-aware embeddings** $z_{1:n}$:
- Two sublayers per block: **Multi-Head Self-Attention** + **Feed-Forward Network (FFN)**
- Residual connection around each sublayer: $x \leftarrow x + \text{sublayer}(x)$
- Layer normalization after each sublayer (Post-LN original; Pre-LN modern — see [[LLMs]])

### Decoder
Maps context embeddings $z$ + target prefix $y_{<t}$ to next token logits:
- Three sublayers: **Masked Multi-Head Self-Attention** (causal) + **Cross-Attention** (to encoder) + **FFN**
- The masked attention prevents attending to future tokens (causal mask)
- Cross-attention: queries from decoder, keys/values from encoder — see [[Attention#Cross-Attention|Cross-Attention]]

**Training**: parallel (teacher forcing — feed ground-truth tokens as decoder input)
**Inference**: sequential (autoregressive — feed own predictions step by step)

## Attention Block
---
Each attention block computes scaled dot-product attention — see [[Attention]] for full derivation:
$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Multi-Head Attention (MHA)**: $h$ attention heads in parallel, each with its own $W^Q_i, W^K_i, W^V_i$, concatenated and projected:
$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

Modern variants ([[LLaMA]], [[LLMs#MHA vs. MQA vs. GQA vs. MLA|GQA/MQA]]): reduce the number of KV heads to save memory.

## Feed-Forward Network (FFN)
---
Applied independently to each token's representation:

**Original (ReLU)**:
$$
\text{FFN}(x) = \max(0,\, xW_1 + b_1)W_2 + b_2, \quad d_{ff} = 4 d_\text{model}
$$

**Modern (SwiGLU)** — used in [[LLaMA]], PaLM:
$$
\text{FFN}_\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_3)\, W_2, \quad d_{ff} \approx \tfrac{8}{3} d_\text{model}
$$

See [[Activation Functions]] for SwiGLU, GELU, and other activation choices.

## Positional Encoding
---
Transformers are permutation-invariant; positional information must be injected explicitly.

**Sinusoidal** (original): fixed encodings based on $\sin/\cos$ at different frequencies — not learned, generalizes to longer sequences in theory.

**Learnable absolute** (GPT, ViT): learned embedding per position — simple but doesn't generalize beyond training length.

**[[RoPE]] (Rotary Position Embedding)**: rotate $Q$ and $K$ by an angle proportional to their positions before the dot product. The inner product then depends only on the *relative* position $m - n$. Used in [[LLaMA]], GPT-NeoX, Qwen, and most modern LLMs. See [[RoPE]] for full derivation.

## Normalization Placement
---
See [[LLMs#Pre-Layer Normalization vs. Post-Layer Normalization]] for the full analysis:

**Post-LN** (original Transformer):
$$
x_{l+1} = \text{LN}(x_l + f_l(x_l))
$$
Gradient norms grow with layer index → unstable with large LR; requires careful warmup.

**Pre-LN** (modern default):
$$
x_{l+1} = x_l + f_l(\text{LN}(x_l))
$$
Residual path is clean → valid gradient flow → no warmup required, allows higher LR.

Modern LLMs (LLaMA, Qwen, DeepSeek) use **Pre-LN with [[RMSNorm]]** instead of LayerNorm.

## Complexity Analysis
---
For sequence length $n$ and model dimension $d$:

| | Self-attention | FFN |
|---|---|---|
| **Compute** | $O(n^2 d)$ | $O(n d^2)$ |
| **Memory** | $O(n^2)$ (attention matrix) | $O(nd)$ |

Self-attention dominates when $n \gg d$; FFN dominates when $d \gg n$.

![[transformer_attention_complexity.png]]

**Efficient attention variants** address the $O(n^2)$ bottleneck: sparse attention, local/sliding-window attention (Longformer), [[Linear Attention]], FlashAttention (IO-aware exact computation).

## Architecture Variants
---

| Architecture | Type | Key example |
|-------------|------|-------------|
| Encoder-only | Bidirectional attention | BERT, RoBERTa |
| Decoder-only | Causal (masked) attention | [[GPTs\|GPT]], [[LLaMA]] |
| Encoder-decoder | Full attention | T5, BART |

**Decoder-only** (causal) is now dominant for language models: simpler, scales better, and in-context learning emerges naturally from the autoregressive objective.

## Notable Model Sizes
---
- GPT-3: 175B parameters, 96 attention heads, $d_\text{model} = 12288$
- GPT-4: ~1.8T parameters (MoE, unconfirmed)
- LLaMA-3 70B: 80 layers, 64 heads, $d_\text{model} = 8192$, GQA

See [[LLaMA]] for full LLaMA architecture details and [[MoE]] for Mixture-of-Experts variants.

## Interpretability
---
- Different attention heads specialize in different linguistic patterns (syntax, coreference, positional)
- [[Attention#Attention Heads are Independent and Additive|Attention heads are additive]] — each head contributes independently to the residual stream
- The residual stream is the carrier of information between layers; attention and FFN "write" to it additively
