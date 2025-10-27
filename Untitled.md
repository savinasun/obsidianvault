 format: subtitle + explanation + findings
ConvNext
MetaMorph

+ **From Attention Sink to Massive Activation:** 
	+ explains attention sink from pure mathematical perspective, attention sink means there exist a sink token whose key has large inner product with query states of all the tokens, review the softmax attention function here, it indicates two scenarios or both:
		+ cosine similarities between the key states of sink token and query states of other tokens are large 
		+ the norm of sink token key states is very large among all tokens
	    we now define two types of attention sink based on the above mathematical analysis
	+ from experimental perspective show that attention sinks are only possibly to be cosine similarities induced
	+ We found most attention sinks are cosine induced, experiments show that llama attention sinks are introduced and norm alleviate the effect, qwen has two types of sinks, one is the same as llama, the other one is 
		+ the main cause of attention sink is from the angle
	+ the reason behind it is the sink token key states live in a different space than the query space of all tokens
	+ representational collapse: is the reason behind the key space and query space are very different