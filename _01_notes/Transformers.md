### [[dl_moc]]  
### Transformer Network Architecture
**Tags**::  #transformer 
**Links**::

[[Transformers]]
[[!https://arxiv.org/abs/1706.03762]]
[[Vaswani, et al., 2017.](https://arxiv.org/abs/1706.03762) Attention is all you need.]

- **Encoder-Decoder Architecture**
    
    weights are not necessarily shared across the encoder and decoder, when the input distribution are different from the output distribution, we should use an encoder-decoder architecture
	    ![[image-3-x173-y371.png|300]]
    - encoder: train simple word embeddings to **context-aware embeddings**
        - two sublayers: a multi-head attention layer + a dense layer
        - residual connection for each sublayer + post-LN → evolve to pre-LN nowadays
    - decoder:
        - three sublayers: masked multi-head attention layer (solve the problem of data-leakage) + a multi-head cross-attention layer + a dense layer
    - parallel training + sequential inferencing (auto-regressive)
    - pre-train phase + fine-tune phase
- **Self-Attention (Scaled Dot-Product Attention)**
    
    ![[image-4-x206-y296.png|400]]
    
    - QKV attention function
        
        → calculate the scaled dot-product similarities among each input tokens
        
        → convert these similarities into percentages by softmax function
        
        → calculate the self-attention score by using these percentages to weight the values embedding
        
    - scale factor: d_k represents the dimension of matrix K, **note that for logits to be input into softmax function, its variance should be approx to 1**
        
    - scaled dot-product attention V.S. additive attention
        
- **Masked Self-Attention**
	
- **Cross-Attention (Encoder-Decoder Attention):**
    
    K,V from encoder
    
    Q from decoder
    
    → calculate encoder-decoder attention scores to generate text (these are from the Seq2Seq models)
    
    - **Multi-Modal Models**: encoder trained on images, speech → context aware embeddings → text based decoder to generate captions or audio-prompt responses
    
- **Multi-Head Attention**
    - d_model: dimension of the embedding vector
    - n_head * d_k = d_model
    
	![[image-5-x104-y560.png]]

in post-LN transformer, the expected gradients of the parameters near the output layer are large at initialization → unstable training using large LR

- _If_ X ∈ R^d _is a Gaussian vector,_ X ∼ N(0, σ^2 I^d)_, then_ E(||_ReLU_(X)||^2) = 1/2 σ^2 d*.*
    
- **Positional Encoding**
    
- Computation complexity + Long Range Dependencies
	![[image-6-x104-y597.png]]
    - n << d: self-attention outperforms
- Interpretability:
    
    - different heads perform different tasks
- Examples:
    
    - GPT-3: 175 billion parameters, 350 GB, 96 heads
    - GPT-4: 1.8 trillion parameters, 3.6 TB
    - LLaMA
        - 7B to 65B parameters
        - main differences from original transformer architecture
            - RMSNorm normalizing function is used to improve the training stability, by normalizing the input of each transformer sub-layer, instead of normalizing the output.
            - The ReLU non-linearity is replaced by the SwiGLU activation function to improve performance.
            - Absolute positional embeddings are removed and instead rotary positional embeddings (RoPE) are added at each layer of the network.
- Local/Restricted Attention???
    
- Making inference less sequential


