### [[dl_moc]]  
# **Attention Sink**

**Tags**::  
**Links**::

+ From Attention Sink of Massive Activation: why attention sink? for sink head, key of sink token has larger inner product with query of all tokens
	+ observation 1: examples, two senarios: llama and qwen, cite Gu
	+ observation 2: almost all Qs are at the same space, Q/K space dimensional collapse
	+ It's easy to orthogonal to the space, for sink token, <- contributed by massive activations, + RMS  one-hot vector
+ From Massive Activation to Attention Sink
	+ observation 1:	first token almost definitely lead to massive activation, only rare tokens does not have massive activation, how many?? and it does not interplay with attention sink since it is a 1\times 1 matrix
		 explain why mlp layer -> massive activation
		 explain why the second token does not have massive activation
	+ observation 2: why 2 token no massive activation/attention sink?
			Hello: cache QKV
			Hello You: cache QKV
			analyze why the internal representation changed
			mlp cannot differentiate position
			the output of attention is always superposition of v
			token 0: w1=1, w2=0
			token 1: w1=0.99, w2=0.01
			if all heads in the layer are sink heads, then the output of token0 and token1 suppose to be close
			what about the norms of value states of the two tokens
			why early layers does not have sink??? -> if so, it might loss a lot of representation power
+ How Massive Activations Associated with Attention Sink
	+ Norm suppresion of Q/K/V at sink tokens
	+ Representation collapse of query space, the relationship between query space and key space
+ Wq \cdot Wk \cdot one-hot: what happened to the norm
+ 
+ 

## Related Work
---
The internal mechanisms governing the behavior of large Transformer-based models, including Large Language Models (LLMs) and Vision Language Models (VLMs), remain an active area of investigation. Recent research has highlighted several intriguing and seemingly paradoxical phenomena, notably **attention sinks** and **massive activations (MAs)**, which occur within the hidden states and attention layers of these models. Understanding the origins and functions of these phenomena is crucial for improving model efficiency, stability, interpretability, and performance, particularly in long-context scenarios relevant to both language and vision tasks.

### Attention Sinks

Consistent phenomenon of "attention sink", where attention heads allocate significant portion of weights to specific tokens despite limited semantic information, has been observed across various Transformer-architectured models, both large and small models (Xiao). It is also observed that attention sink develops during pre-training, and instruction-tuning does not change its prevalence (Gu, Guo).

Several hypotheses have been proposed to explain the emergence and utility of attention sinks. 
Gu found that cosine similarity instead of norm product contributes to attention sinks in Llama series models. 

Several hypotheses have been proposed to explain the emergence and utility of attention sinks. One line of reasoning suggests they act as a mechanism to **prevent over-mixing** of information in deep networks, effectively slowing down information propagation and countering representational or rank collapse, phenomena related to over-smoothing in graph networks666666666666666666. Barbero et al. 777connect sinks to theoretical concepts like over-squashing and representational collapse, arguing their necessity increases with model depth and context length 888888888, a finding empirically supported by training ablations999999999. Another perspective links sinks to **positional biases** learned during pre-training101010101010101010. Gu et al. further suggest that sinks may arise partly due to the **softmax normalization** constraint, forcing attention scores to sum to one, thus requiring a default location to "dump" excess attention111111111111111111. Pre-training LMs with alternative attention mechanisms lacking this normalization, such as sigmoid attention, has been shown to prevent sink emergence121212121212121212. Sinks have also been connected to spectral properties of the network 13131313and the formation of approximate "no-op" mechanisms within specific heads14141414.

### Massive Activations (MAs)

Concurrent research identified **"massive activations" (MAs)**—scalar values within transformer hidden states exhibiting magnitudes orders of magnitude larger (e.g., ![](data:,) times the median) than typical activations1515151515151515151515151515. Sun et al. 16161616provided the first systematic characterization, demonstrating that MAs are widespread across LLM families 17171717, appear in fixed, input-agnostic feature dimensions 18181818, and often co-locate with attention sink tokens (e.g., initial tokens, delimiters)19191919191919191919191919191919191919191919191919. Crucially, MAs appear to function as **indispensable, implicit bias terms**; their values remain relatively constant across inputs 20202020, and removing them causes catastrophic model failure, whereas setting them to their mean values preserves performance21212121212121212121212121212121. Sun et al. also established a direct link between MAs and attention concentration, suggesting MAs _cause_ attention to focus on their corresponding tokens 222222222222222222, potentially by creating distinct representations that dominate attention computations23232323. They proposed that explicit attention biases (learnable key/value pairs) added during training can eliminate the need for models to learn MAs242424242424242424. Gallego-Feliciano et al. 25252525further investigated the **temporal dynamics**, showing that MA emergence during training follows predictable, mathematically modelable patterns governed by architectural parameters262626262626262626, allowing for potential prediction and control. This work also relaxed the strict definition of MAs to better capture the phenomenon across model scales27. MAs are distinct from the previously studied "outlier features" 28282828, which describe entire feature dimensions with large average magnitudes across _many_ tokens29292929.

### Interplay and Unifying Mechanisms

Recently, researchers have started exploring the deep **interplay between attention sinks, MAs, and representation geometry**. Cancedda 30303030 linked sinks to specific spectral subspaces and large hidden state norms, aligning with the MA findings. Gu et al. 31313131 also noted the large norms of hidden states at sink positions, reinforcing the connection. Queipo-de-Llano et al. 32323232explicitly proposed a **unified mechanism**, arguing that MAs in the residual stream are the common cause for both attention sinks _and_ the **"compression valley"** phenomenon 33333333—where intermediate layer representations exhibit unexpectedly low entropy3434343434. They provided theoretical proof that large activation norms necessarily induce spectral dominance and compression 353535353535353535, validating this through experiments and ablations36363636363636363636363636363636. Building on this, they proposed the **Mix-Compress-Refine theory** of information flow, suggesting transformers organize computation depth-wise through phases controlled by MAs37373737373737373737373737373737. Other works have also explored the training dynamics linking these phenomena38.

### Vision Transformers (ViTs) and VLMs

Similar phenomena have been observed in vision models. Sun et al. 39393939reported the existence of MAs in several ViT families (CLIP, DINOv2), noting they also act as biases but appear later in the network40404040. Darcet et al. 41414141identified attention artifacts and high-norm activations in ViTs, proposing **"register tokens"**—additional learnable tokens—to absorb these activations and stabilize attention maps42424242. Sun et al. and Gallego-Feliciano et al. subsequently argued that these register tokens effectively function as learned biases, similar to MAs in LLMs434343434343434343. Gan et al. 44 explored modulating MAs in diffusion transformers for visual correspondence. While less explored, the implications for VLMs, which combine language and vision processing often using Transformer backbones, are significant, suggesting these mechanisms may operate across modalities.

### Gap and Our Contribution

Previous work has independently characterized attention sinks 454545454545454545and massive activations 46464646, proposed various functional roles (over-mixing prevention 47, implicit biases 48, stream anchoring 49), linked them empirically 505050505050505050, and explored their training dynamics51515151. Some works have begun unifying sinks, MAs, and representational properties like compression52525252. However, a definitive, mechanistic explanation connecting the _fundamental cause_ of MAs to their _manifestations_ as attention sinks and their role in information processing across both LLMs and VLMs remains elusive. This paper aims to fill this gap by proposing and validating an ultimate underlying mechanism that comprehensively explains the origin, function, and interplay of these critical phenomena in Transformer-based architectures across modalities.


