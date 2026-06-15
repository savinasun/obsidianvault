### [[dl_moc]] · [[sequence_llms_moc]]  
# **Retrieval-Augmented Generation (RAG)**

**Tags**:: #llm
**Links**:: [[LLMs]] · [[CLIP]] · [[Attention]] · [[BPE]]


**RAG** ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)) augments a language model's generation with **retrieved documents** from an external knowledge store. Instead of relying solely on parametric memory (weights), the model can look up relevant information at inference time — enabling factual, up-to-date, and verifiable responses.

## Motivation
---
LLMs suffer from:
- **Hallucination**: generating plausible but false facts confidently
- **Knowledge staleness**: training cutoff means the model doesn't know recent events
- **Attribution**: unable to cite sources
- **Long-tail knowledge**: rare facts are underrepresented in training

RAG addresses all four by grounding generation in retrieved documents. It also allows updating the knowledge base without retraining (just update the index).

**Why RAG beats naive fine-tuning for factual tasks**: fine-tuning stores facts in weights — brittle, hard to update, and prone to forgetting. RAG stores facts in a retrievable index — exact, updatable, and traceable.

## Architecture
---

### Stage 1: Offline Indexing
1. **Chunk** the document corpus into passages of $\sim 100$–$500$ tokens
2. **Embed** each chunk using a dense encoder (e.g., DPR, sentence-transformers, OpenAI embeddings): $e_i = \text{Enc}(d_i) \in \mathbb{R}^d$
3. **Index** all embeddings in a vector store (FAISS, Pinecone, Weaviate) for fast approximate nearest-neighbor (ANN) search

### Stage 2: Online Retrieval
Given a user query $q$:
1. Embed the query: $e_q = \text{Enc}(q)$
2. Retrieve top-$k$ chunks by cosine similarity: $\{d_1, \ldots, d_k\} = \text{TopK}_{i}\,\cos(e_q, e_i)$
3. Rank or rerank the retrieved chunks (optional: cross-encoder reranker)

### Stage 3: Augmented Generation
Concatenate the retrieved documents with the query as context for the LLM:
```
Context:
[Document 1]: {retrieved chunk 1}
[Document 2]: {retrieved chunk 2}
...
Question: {user query}
Answer:
```
The LLM generates its response conditioned on both the query and the retrieved evidence.

## Dense Passage Retrieval (DPR)
---
DPR ([Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)) trains a **bi-encoder** for retrieval using contrastive learning:

- **Query encoder** $\text{Enc}_Q$: encodes the query into a dense vector
- **Passage encoder** $\text{Enc}_P$: encodes each passage into a dense vector

Training: for each (question, positive passage, negative passages) triple, maximize similarity to the positive and minimize to negatives:
$$
\mathcal{L} = -\log \frac{e^{\text{sim}(q, p^+)}}{\sum_j e^{\text{sim}(q, p_j^-)}}
$$

This is essentially the same InfoNCE loss as [[CLIP]] but for text-text retrieval.

## Chunking Strategies
---
Chunking affects retrieval quality more than most practitioners expect:

| Strategy | Description | Best for |
|----------|-------------|----------|
| **Fixed-size** | Split every $n$ tokens with $k$ token overlap | Simple, fast baseline |
| **Sentence-based** | Split at sentence boundaries | Preserves semantic units |
| **Paragraph-based** | Split at paragraph breaks | Documents with clear structure |
| **Recursive** | Try larger splits, fall back to smaller | Hierarchical documents |
| **Semantic chunking** | Split where embedding similarity drops | Best quality, expensive |

**Overlap**: always use $10$–$20\%$ overlap between chunks to avoid splitting a concept across chunk boundaries.

## Reranking
---
ANN retrieval (bi-encoder) is fast but imprecise. A **cross-encoder reranker** takes the (query, passage) pair jointly and produces a single relevance score:

$$
s(q, d) = \text{CrossEncoder}([q;\, d])
$$

Cross-encoders use full attention between query and document (unlike bi-encoders, which embed them separately), giving more precise relevance judgments. Typical pipeline: retrieve top-50 with bi-encoder, rerank with cross-encoder, pass top-5 to the LLM.

## Fusion-in-Decoder (FiD)
---
Izacard & Grave (2021): rather than concatenating all retrieved passages in one context, **encode each passage independently** and then **fuse** in the decoder via cross-attention:

1. For each retrieved passage $d_i$, prepend the question: `Q: {q} Context: {d_i}`
2. Encode each independently with the encoder: $h_i = \text{Enc}(q, d_i)$
3. Concatenate all encoder outputs: $H = [h_1; h_2; \ldots; h_k]$
4. Decoder cross-attends to all of $H$

**Benefits**: scales to many more retrieved passages (encoder is independent per passage; only decoder sees all); outperforms concatenation for knowledge-intensive tasks.

## Advanced RAG Patterns
---

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer using the LLM, embed it, and use it as the retrieval query. Bridges the semantic gap between short questions and long document passages.

### Self-RAG
Train the model to decide *when* to retrieve (not every query needs retrieval), generate retrieval tokens, and critique its own outputs.

### GraphRAG
Build a knowledge graph from the corpus; traverse the graph for multi-hop reasoning instead of flat chunk retrieval.

## Limitations
---
- **Retrieval errors**: if top-$k$ chunks don't contain the answer, the LLM will either hallucinate or say it doesn't know — accuracy bounded by recall
- **Context length**: retrieved chunks consume context window; LLMs often struggle with relevant information in the middle of long contexts ("lost in the middle" problem)
- **Latency**: retrieval adds roundtrip time; not suitable for latency-critical applications without caching
- **Chunking artifacts**: poorly chunked documents split crucial context across boundaries
