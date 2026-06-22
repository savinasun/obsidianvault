### [[dl_moc]]  
# **Word Embeddings**

**Tags**:: #learning
**Links**:: [[BPE]] · [[LLMs]] · [[Transformer]] · [[Contrastive Learning]] · [[RNN]]


**Word embeddings** map discrete tokens to dense real-valued vectors in a continuous space where semantic relationships are captured by geometric proximity. They are the foundation of modern NLP — replacing one-hot encodings with representations that encode meaning.

## Motivation
---
One-hot encodings treat words as independent symbols: $\text{cat} = [0,0,1,0,\ldots]$. No similarity is encoded — cosine similarity between any two one-hot vectors is zero. This fails to generalize: a model that learns "cat" must relearn everything for "kitten."

Distributed representations solve this: $\text{cat}$ and $\text{kitten}$ have nearby vectors, so knowledge transfers across similar words. The **distributional hypothesis** (Firth, 1957) underpins this: *"a word is characterized by the company it keeps."*

## Word2Vec
---
[Mikolov et al., 2013](https://arxiv.org/abs/1301.3781) introduced two architectures to learn embeddings from co-occurrence statistics:

### Continuous Bag of Words (CBOW)
Predict the center word from its surrounding context:
$$
\hat{y} = \text{softmax}(W' \cdot \bar{v}_\text{context}), \quad \bar{v}_\text{context} = \frac{1}{2k}\sum_{-k \le j \le k, j\ne 0} v_{w_{t+j}}
$$
Context window of size $k$; average the context embeddings, then predict the center.

### Skip-Gram
Predict surrounding context words from the center word — the inverse of CBOW:
$$
\mathcal{L} = \sum_{t} \sum_{-k \le j \le k, j\ne 0} \log p(w_{t+j} \mid w_t)
$$

Skip-gram works better on rare words; CBOW is faster to train.

### Negative Sampling
Softmax over a large vocabulary is expensive. **Negative sampling** reformulates the objective as binary classification: distinguish true context pairs $(w, c)$ from randomly sampled "negative" pairs:
$$
\mathcal{L}_\text{NS} = \log\sigma(v_c^\top v_w) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n}\left[\log\sigma(-v_{w_i}^\top v_w)\right]
$$
where $P_n(w) \propto f(w)^{3/4}$ (unigram distribution raised to $3/4$ to upsample rare words).

### Analogies via Vector Arithmetic
Word2Vec vectors exhibit **linear algebraic structure** capturing semantic and syntactic relationships:
$$
v_\text{king} - v_\text{man} + v_\text{woman} \approx v_\text{queen}
$$
$$
v_\text{Paris} - v_\text{France} + v_\text{Italy} \approx v_\text{Rome}
$$
This emergent structure was surprising — it means syntax and semantics are linearly encoded in the embedding space.

## GloVe (Global Vectors)
---
[Pennington et al., 2014](https://aclanthology.org/D14-1162.pdf). Word2Vec trains on local context windows; **GloVe** explicitly models global co-occurrence statistics.

Build a co-occurrence matrix $X$ where $X_{ij}$ = number of times word $j$ appears in the context of word $i$. GloVe minimizes:
$$
\mathcal{L} = \sum_{i,j} f(X_{ij})\left(v_i^\top \tilde{v}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
$$

where $f(X_{ij}) = \min\!\left(1, \left(\frac{X_{ij}}{X_\text{max}}\right)^{3/4}\right)$ is a weighting function that downweights very frequent co-occurrences.

**Intuition**: the dot product of two word vectors should equal the log probability of their co-occurrence. GloVe achieves similar analogy performance to Word2Vec but with a cleaner probabilistic interpretation.

## FastText
---
[Bojanowski et al., 2017](https://arxiv.org/abs/1607.04606). Extends Word2Vec by representing each word as a bag of **character n-grams**:
$$
v_\text{where} = v_{\texttt{<wh}} + v_{\texttt{whe}} + v_{\texttt{her}} + v_{\texttt{ere}} + v_{\texttt{re>}}
$$

**Benefits**:
- Handles **OOV words** — any unseen word can be approximated from its n-gram components
- Learns morphologically aware representations (e.g., "running" and "runs" share n-grams)
- Better for languages with rich morphology (German, Turkish)

## Debiasing Word Embeddings
---
Word embeddings trained on internet text inherit **social biases** from their training corpora:
$$
v_\text{man} - v_\text{woman} \approx v_\text{programmer} - v_\text{homemaker}
$$

[Bolukbasi et al., 2016](https://arxiv.org/abs/1607.06520) proposed debiasing by:
1. Identify the **bias direction** (e.g., the gender subspace): $g = v_\text{he} - v_\text{she}$
2. **Neutralize** non-gender words by projecting out the bias direction: $v_w \leftarrow v_w - (v_w \cdot \hat{g})\hat{g}$
3. **Equalize** gendered word pairs (e.g., "grandmother"/"grandfather") to be equidistant from the neutral embedding

This highlights that embeddings are not just mathematical objects — they encode cultural biases from data, raising fairness concerns in production systems.

## Contextualized Embeddings
---
Word2Vec/GloVe produce **static** embeddings — the word "bank" has the same vector whether referring to a financial institution or a river bank.

**Contextualized embeddings** produce different vectors for the same word depending on context:
- **ELMo** (2018) — two-layer LSTM trained with LM objective; context-dependent representations
- **BERT** (2019) — Transformer encoder with masked LM; each token vector attends to the full sequence
- **GPT** (2018) — Transformer decoder; left-to-right contextualization

Modern [[LLMs]] use **learned token embeddings** inside the model rather than pre-computed static embeddings, so the embedding itself is trained end-to-end with the full model. The embedding layer is effectively a lookup table mapping token IDs to $d_\text{model}$-dimensional vectors.

## Comparison
---
| Method | Type | OOV | Analogy | Context-aware |
|--------|------|-----|---------|---------------|
| Word2Vec (Skip-gram) | Static, local | No | Good | No |
| GloVe | Static, global | No | Good | No |
| FastText | Static, character | Yes | Good | No |
| ELMo | Contextual, RNN | Via chars | N/A | Yes |
| BERT/GPT | Contextual, Transformer | Via BPE | N/A | Yes |

Modern models use [[BPE]] (Byte Pair Encoding) tokenization to handle OOV by construction, making static word embeddings largely superseded for production NLP.
