### [[dl_moc]] · [[sequence_llms_moc]]  
# **Byte Pair Encoding**

**Tags**:: #llm
**Links**:: [[LLMs]] · [[GPTs]]

## Overview
---
**Byte Pair Encoding (BPE)** is a subword tokenization algorithm that balances the trade-off between word-level and character-level tokenization. Originally introduced for data compression by Philip Gage in 1994, BPE was adapted to NLP for neural machine translation by [Sennrich et al., 2016](https://aclanthology.org/P16-1162/), BPE was then used by [[GPTs#GPT-2|GPT-2]] [@radford].


## Motivation
---
- **Word-level tokenization** suffers from large vocabulary size and out-of-vocabulary (OOV) issues (the \<UNK>).
- **Character-level tokenization** avoids OOV but results in longer sequences and weaker semantics.
- **BPE** provides a middle ground by learning a fixed vocabulary of subword units from data, allowing efficient representation of both frequent and rare words.


## Algorithm
---
The basic BPE algorithm is a **greedy frequency-based merge** procedure. It start with each byte as a token, and successively merge the most common pair of adjacent tokens. It operates as follows:

1. **Initialize** the vocabulary with all individual characters from the training corpus.
2. **Count** frequency of symbol pairs (adjacent characters or subwords).
3. **Merge** the most frequent pair into a new symbol.
4. **Repeat** steps 2–3 for a fixed number of iterations or until the desired vocabulary size is reached.

**Example**
Input: `"low low lower"`
Initial symbols: `l o w`, `l o w`, `l o w e r`
Steps:
1. Count pairs: (`l`, `o`), (`o`, `w`), (`w`, `e`), (`e`, `r`)
2. Most frequent: (`l`, `o`) → merge to `lo`
3. Update sequence: `lo w lo w lo w e r`
4. Repeat...

## Implementation Details
---
- Corpus is usually pre-tokenized by whitespace before applying BPE.
- Merges are applied deterministically according to a learned list of merge rules.
- Vocabulary can include:
  - Regular subwords (e.g., `low`, `##er`)
  - Special tokens (e.g., `<pad>`, `<unk>`, `<s>`, `</s>`)


## Applications
---
- Widely used in NLP models like **GPT**, **BERT**, **RoBERTa**, and **MarianMT**.
- Helps with handling **morphologically rich languages** and rare words.
- Used in frameworks like:
  - [SentencePiece](https://github.com/google/sentencepiece) (Google)
  - [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)


## Pros and Cons
---
### Pros
- Compact vocabulary size.
- Handles unseen words gracefully.
- Works well across different languages and domains.
### Cons
- Greedy merges are not optimal.
- Not reversible without extra markers.
- Merged subwords may not align well with linguistic units.


## Related Methods
---
- **Unigram Language Model**: Probabilistic alternative used in SentencePiece.
- **WordPiece**: Used in BERT, similar to BPE but scores merges differently.
- **Morfessor**: Unsupervised morphological segmentation model.


## References
---
- Sennrich, R., Haddow, B., & Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/). *ACL 2016*.
- Gage, P. (1994). A New Algorithm for Data Compression. *C Users Journal*.

