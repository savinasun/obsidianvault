### [[dl_moc]]  
# **Large Language Models**

**Tags**::  
**Links**::

## Architecture Variants
---
### Pre-Layer Normalization V.S. Post-Layer Normalization
![[llms_01.png]]
- In the **original-designed Post-LN** [[Transformer|transformer]], [[Layer Normalization|layer normalization]] is placed after the residual blocks, [[BERT]] also uses **Post-LN**. 
+ The function of post-LN could be expressed as$$
	x_{l+1} = \textrm{LN}(x_{l}+f(x_l)),
	$$here $f(\cdot)$ represents FFN or attention function.
- In Post-LN transformers, **the expected gradients** of the parameters near the output layer are **large** in the forward pass[@xiong2020]; and the expected gradient-norms decay exponentially in the backward pass, which leads to gradient vanishing in deep network trainings.
	Suppose 

$$
x_{l+1} = 
$$
- Experiments are conducted to show that this learning rate warm-up stage is essential for training Post-LN transformer models.
	![[llms_02.png]]
- Warm-up stage significantly slows down the optimization process and requires hyper-parameter tuning which is computationally expensive.
- According to [@xiong2020], the scale of the activations in the pre-LN transformer keeps to be the same in activations and therefore warm-up stage could be skipped.
- The architecture of Post-LN versus Pre-LN:

+ Post-LN requires a **LR warm-up stage** for the optimizer to perform well
+ Pre-LN: requires a final layer normalization right before prediction


- 