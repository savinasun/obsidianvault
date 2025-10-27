 format: subtitle + explanation + findings
ConvNext
MetaMorph

+ **From Attention Sink to Massive Activation:** 
	+ attention sink means the sink token key to other query states has large values after softmax stage from the formula of softmax attention function
		+ there are two possibilities: cosine large / norm large or mixed cases
		+ introduce two types of attention sink
		Findings: from mathematical perspective, there are two cases ....
	+ We found most attention sinks are cosine induced, experiments show that llama attention sinks are introduced and norm alleviate the effect, qwen has two types of sinks, one is the same as llama, the other one is 
		+ the main cause of attention sink is from the angle
	+ the reason behind it is the sink token key states live in a different space than the query space of all tokens
	+ representational collapse: is the reason behind the key space and query space are very different0008i8i