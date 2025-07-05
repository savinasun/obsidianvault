### [[dl_moc]]  
# **Large Language Models**

**Tags**::  
**Links**::

## Architecture Variants
---
### Pre-Layer Normalization V.S. Post-Layer Normalization
![[llms_01.png|600]]
- In the **original-designed Post-LN** [[Transformer|transformer]], [[Layer Normalization|layer normalization]] is placed between the residual blocks, this architecture has achieved state-of-the-art performance in many tasks. The very early pre-trained large language models like [[GPT]] and [[BERT]] also uses Post-LN. 
 + The function of post-LN could be expressed as$$
	x_{l+1} = \textrm{LN}(x_{l}+f_l(x_l)),
	$$where $f_l(\cdot)$ represents FFN or attention function in the $l$-th layer.
- It was noticed that in Post-LN transformers, the **gradient-norm decay exponentially** in backward propagation, which shadows the original intention that applys residual connections here and leading to **gradient vanishing** when training deep network. It therefore requires a subtly tuned **LR warm-up stage** for the optimizer to perform well.$$
	x_{l+1} = \textrm{LN}(x_{l}+f_l(x_l)) = \frac{x_l + f_l(x_l)}{\sqrt{2}} = \frac{x_{l-1}}{2} + \frac{f_{l-1}(x_{l-1})}{2} + \frac{f_l(x_l)}{\sqrt{2}} = \cdots = \frac{x_0}{2^{(l+1)/2}} + \frac{f_0(x_0)}{2^{(l+1)/2}} + \cdots + \frac{f_l(x_l)}{\sqrt{2}}
	$$Experiments are conducted to show that this learning rate warm-up stage is essential for training Post-LN transformer models, ![[llms_02.png]]
- In [@xiong2020], the author proposes the Pre-LN Transformer which puts **layer normalization inside the residual connection** and places it before all other non-linear transformations. Additionally, the Pre-LN Transformer uses a final layer normalization right before the prediction.
- The function of pre-LN could be expressed as$$
x_{l+1} = x_l + f_l (\textrm{LN}(x_l))
	$$where $f_l(\cdot)$ represents FFN or attention function in the $l$-th layer. $$
	x_{l+1} = x_{l}+f_l(\textrm{LN}(x_l)) = \cdots = x_0 + f_0(\textrm{LN}(x_0)) + \dots+ f_l(\textrm{LN}(x_l))
	$$Now due to the valid residual connection here, there is no more gradient vanishing during back propagation and it no longer requires the warm-up stage for the optimizer to perform well. ![[llms_03.png]]
+ N stability and larger LRs for large networks
+ There is other explanation for why pre-LN is better than post-LN as well, 