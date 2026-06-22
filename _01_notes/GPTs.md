### [[dl_moc]]  

# **Generative Pre-Training Transformer Models**

**Tags**:: #llm
**Links**:: [[Transformer]] · [[BPE]] · [[Scaling Laws]] · [[LLMs]] · [[RLHF]]


## GPT Families
---
### GPT-1
+ [Improving Language Understanding by Generative Pre-Training | GPT](https://paperswithcode.com/paper/improving-language-understanding-by)
- a multi-layer [[Transformer|transformer decoder]]
	
- pre-train on large unlabeled data (SSL) + supervised fine-tune on task-specific data
	
- self-supervised learning + next-token prediction (uses [[BPE]] tokenization)
	
	![[gpt_architecture.png]]
	
- ~110M parameters

### BERT (Bidirectional Encoder Representations from Transformers)
	
- [Devlin et al., 2018](https://arxiv.org/abs/1810.04805), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional)
- bidirectional encoder → an encoder-only transformer model
- pre-train on large unlabeled data + supervised fine-tune on task-specific data
	- randomly masks some of the tokens from the input;
	- pre-train a deep bidirectional transformer
	- next-sentence prediction task
- BERT-Base: ~110M parameters
- BERT-Large: ~340M parameters


### GPT-2
- [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask)
- ~1.5 billion parameters
- a **zero-shot** learner


### GPT-3
- [Brown et al., 2020](https://arxiv.org/abs/2005.14165), [Language Models are Few-Shot Learners](https://paperswithcode.com/paper/language-models-are-few-shot-learners)
- ~175 billion parameters
- no fine-tuning + no gradient update in few-shot learning
- as you increase the layers, the width per layer should also be increased
- for smaller models, batch size should not be too big to prevent overfitting


