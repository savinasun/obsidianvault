### [[dl_moc]]
# **Sequence Modeling & LLMs**

**Tags**:: #llm #architecture
**Links**:: [[math_moc]] · [[learning_methods_moc]] · [[vision_ssl_moc]] · [[generative_moc]] · [[rl_embodied_moc]]

The evolution of sequence modeling — from RNNs to Transformers to state space models — and the full stack of modern large language models: architecture, tokenization, training, inference, and alignment.

## Sequence Architectures
---
- **[[RNN]]** — Recurrent Neural Networks: vanilla RNN, LSTM (4-gate cell state), GRU; vanishing gradient problem; why Transformers replaced RNNs; beam search decoding
- **[[Transformer]]** — The dominant architecture: multi-head self-attention, positional encodings, Pre-LN, encoder-decoder vs. decoder-only; $O(L^2)$ complexity
- **[[Attention]]** — Scaled dot-product attention $\text{softmax}(QK^\top/\sqrt{d_k})V$; multi-head attention; cross-attention; causal masking; Flash Attention
- **[[Mamba]]** — Selective State Space Models: input-dependent state transitions, $O(L)$ parallel scan training, $O(1)$ inference memory; the viable alternative to Transformers for long sequences

## LLM Architecture Components
---
- **[[RoPE]]** — Rotary Position Embeddings: encode position via rotation in complex space; length generalization; used in LLaMA, GPT-NeoX, Mistral
- **[[MoE]]** — Mixture of Experts: sparse activation of expert FFN layers; top-$k$ routing; expert load balancing; used in Mixtral, GPT-4 (rumored), Switch Transformer
- **[[Word Embeddings]]** — Word2Vec, GloVe, FastText; vector arithmetic analogies; debiasing; superseded by contextual embeddings in modern LLMs

## Tokenization
---
- **[[BPE]]** — Byte Pair Encoding: iterative merging of frequent byte pairs; the tokenization algorithm underlying GPT-2/3/4, LLaMA, and most modern LLMs
- **[[PPL]]** — Perplexity: $\exp(-\frac{1}{N}\sum_i \log p(w_i|w_{<i}))$; the standard intrinsic evaluation metric for language models

## Model Families
---
- **[[LLMs]]** — Architecture internals of modern large language models: Pre-LN Transformer stack, AdamW + cosine LR, mixed precision, gradient checkpointing, KV cache inference
- **[[LLaMA]]** — Meta's open-weight LLM family: RoPE, SwiGLU, RMSNorm, GQA; training recipe and token budget; influence on open-source LLM ecosystem
- **[[GPTs]]** — OpenAI's Generative Pre-trained Transformers: GPT-1 through GPT-4 lineage; scaling hypothesis; in-context learning emergence; RLHF alignment

## Applications & Capabilities
---
- **[[CoT Prompting]]** — Chain-of-Thought: standard CoT, zero-shot CoT ("think step by step"), self-consistency via majority vote; Tree-of-Thought; why scratchpad reasoning improves accuracy
- **[[RAG]]** — Retrieval-Augmented Generation: dense retrieval (DPR), chunking strategies, embedding lookup, fusion-in-decoder; why RAG beats fine-tuning for factual grounding
- **[[Cluster Training Glossary]]** — Distributed training vocabulary: data parallelism, tensor parallelism, pipeline parallelism, ZeRO optimizer stages, FSDP

## Multimodal
---
- **[[VLM]]** — Vision Language Models: ViT image encoder + LLM decoder; cross-attention or prefix fusion; BLIP-2, LLaVA, GPT-4V; visual instruction tuning

## Dataview
---
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "llm")
SORT file.name ASC
```
