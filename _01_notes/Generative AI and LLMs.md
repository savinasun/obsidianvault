### **Generative Pre-Training Transformer Models**
	
- GPT
	
	- [[Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [Improving Language Understanding by Generative Pre-Training](https://paperswithcode.com/paper/improving-language-understanding-by)]
	
	- a multi-layer transformer decoder
		
	- pre-train on large unlabeled data (SSL) + supervised fine-tune on task-specific data
		
	- self-supervised learning + next-token prediction
		
		![[Pasted image 20250513120752.png]]
		
	- ~110M parameters
		
- **BERT (Bidirectional Encoder Representations from Transformers)**
	
	- [[Devlin et al., 2018](https://arxiv.org/abs/1810.04805), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional)]
	- bidirectional encoder → an encoder-only transformer model
	- pre-train on large unlabeled data + supervised fine-tune on task-specific data
		- randomly masks some of the tokens from the input;
		- pre-train a deep bidirectional transformer
		- next-sentence prediction task
	- BERT-Base: ~110M parameters
	- BERT-Large: ~340M parameters
	
- **GPT-2:**
	
	- [[Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask)]
	- ~1.5 billion parameters
	- a **zero-shot** learner
	
- **GPT-3:**
	
	- [[Brown et al., 2020](https://arxiv.org/abs/2005.14165v4), [Language Models are Few-Shot Learners](https://paperswithcode.com/paper/language-models-are-few-shot-learners)]
	- ~175 billion parameters
	- no fine-tuning + no gradient update in few-shot learning
	- as you increase the layers, the width per layer should also be increased
	- for smaller models, batch size should not be too big to prevent overfitting
	
- **InstructGPT**
	
- **GPT-3.5 (`text-davinci-003`)**
	
- **ChatGPT**
	
	- a GPT-3.5 fine-tuned on human instructions by RLHF → a decoder-only transformer model

### In-Context Learning (ICL) - Zero shot inference

- Zero-Shot Learning V.S. One-Shot Learning V.S. Few-Shot Learning

- One-shot learning example: face-recognition task

[https://www.geeksforgeeks.org/zero-shot-vs-one-shot-vs-few-shot-learning/](https://www.geeksforgeeks.org/zero-shot-vs-one-shot-vs-few-shot-learning/)



+ LLaMA Architecture 
	![[llama_encoder_architecture.png | 500]]
	+ RMS Norm & Pre-Norm
	+ Rotary Positional Encodings
	+ Self-Attention with KV Cache
	+ Grouped Multi-Query Attention
	+ SwissGLU in FFC
+ LLaMA 1 & 2 Params
	![[llama12_params.png]]