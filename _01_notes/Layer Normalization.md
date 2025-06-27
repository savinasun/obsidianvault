### [[dl_moc]]  
## **Layer Normalization**
**Tags**:: #architecture/transformer 
**Links**:: [[ba2016]]

[[Ba et al., 2016](http://arxiv.org/pdf/1607.06450v1), Layer Normalization]
+ Notice that for sequence models, it is more common to use LN, since there could be a varying sequence size
- modern transformer architecture → pre-layer normalization
	
	Post-LN requires a **LR warm-up stage** for the optimizer to perform well
	
	Pre-LN: requires a final layer normalization right before prediction
	
	![[Pasted image 20250513120914.png]]

- according to [[Xiong et al., 2020](https://arxiv.org/pdf/1908.03265), On Layer Normalization in the Transformer Architecture], the scale of the activations in the post-LN transformer keeps to be the same in activations

+ ![[Screenshot 2025-05-22 at 22.20.15.png]]