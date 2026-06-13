### [[dl_moc]]  
# Perplexity Score

**Tags**:: #llm
**Links**::

## Definition and Formula
---
**Perplexity** is a standard metric used to evaluate the performance of probabilistic models, especially in **language modeling**. It measures how well a model predicts a sample and can be interpreted as the average branching factor of the language model — i.e., how "confused" the model is when predicting the next token.

For a language model assigning probabilities $P(w_1, w_2, \dots, w_N)$ to a sequence of $N$ words:
$$
\text{Perplexity} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_1, \dots, w_{i-1}) \right)
$$
It is the **exponentiated average log-likelihood** of the predicted words in a sequence.


## Relationship with [[Entropy#Cross Entropy|Cross Entropy]]
---
- **Cross-entropy** measures the expected number of bits/nats needed to encode a token using the predicted probability distribution.
- ~={red}**Perplexity is simply the exponential of cross-entropy**=~, translating the log-scale loss into a more intuitive scale.
	If the average **cross-entropy loss** $H(P)$ is known (usually in nats or bits), then:

$$
\text{Perplexity} = \exp(H(P)) \quad \text{(when using natural log)}
$$
$$
\text{Perplexity} = 2^{H(P)} \quad \text{(when using log base 2)}
$$

- Minimizing cross-entropy is equal with minimizing perplexity.


## Interpretation
---
- Lower perplexity = better performance (model assigns higher probability to true sequences).
- A perplexity of $k$ can be interpreted as: on average, the model is as uncertain as if it were choosing uniformly among $k$ equally likely options for each word.
- For a perfect model: perplexity = 1 (certainty).
- For a uniform random model over $V$ vocabulary tokens: perplexity = $V$.


## Usage
---
- Widely used to evaluate **language models** (e.g., GPT, BERT pretraining, etc.)
- Can be computed on a **validation or test set** to measure generalization.
- Often reported on standard corpora like **Penn Treebank**, **WikiText**, or **OpenWebText**
