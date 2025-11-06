 format: subtitle + explanation + findings
ConvNext
MetaMorph

+ **From Attention Sink to Massive Activation:** 
	+ explains attention sink from pure mathematical perspective, attention sink means there exist a sink token whose key has large inner product with query states of all the tokens, review the softmax attention function here, it indicates two scenarios or both:
		+ cosine similarities between the key states of sink token and query states of other tokens are large 
		+ the norm of sink token key states is very large among all tokens
	    we now define two types of attention sink based on the above mathematical analysis
	+ from experimental perspective show that attention sinks could only possibly to be cosine similarities induced, based on observations on llama and qwen models, we found two types of patterns:
		+ pattern 1 (almost all llama sink heads and part of qwen sink heads have show this pattern): the key states of sink tokens always has positive cosine similarities with all query states whereas the key states of non-sink tokens always has negative cosine similarities with all query states.
		+ pattern 2 (part of qwen sink heads have show this pattern): all key states and query states have positive cosine similarities but for sink tokens the cosine values are always larger than the non-sink tokens
		we may conclude that the main cause of attention sinks is from the q/k angles based on experimental results
	+ through further analysis, we may derive the conclusion that the reason behind the experimental behavior is that the key states of sink tokens live in a different space with other tokens
	+ we develop a hypothesis that there is representational collapse for key space and query space, and sink token lives differently with other tokens, so it is easily for its key/query states to be orthogonal with other tokens' K/Q



- Why no matter what token being input to the first position, it all show massive activations after MLP layer?
- Why the same token being input to a different location, the massive activation disappeared.
- first token -> linear transformation -> a special vector -> same vector before MLP -> massive activation