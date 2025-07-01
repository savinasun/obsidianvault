### [[dl_moc]]  
## **LLaMa**

**Tags**::  
**Links**::

---

### Llama Architecture
LLaMA Architecture 
	![[llama_encoder_architecture.png | 500]]
	+ [[RMSNorm]] & Pre-Norm
	+ [[RoPE|Rotary Positional Encodings]]
	+ Self-Attention with KV Cache
	+ Grouped Multi-Query Attention
	+ SwissGLU in FFC


 Llama 1 & 2 Params
	![[llama12_params.png]]


The MLP layer in LLaMa2 uses [[Activation Functions#SwiGLU (Swish Gated Linear Unit)|SwiGLU]] activation functions
$$
FFN_{SwiGLU}(x, W_1, W_2, W_3) = (Swish_1(xW_1)\otimes xW_3)W_2
$$