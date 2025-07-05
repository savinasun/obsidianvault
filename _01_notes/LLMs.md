### [[dl_moc]]  
# **Large Language Models**

**Tags**::  
**Links**::

## Architecture Variations
---
### Pre-Layer Normalization V.S. Post-Layer Normalization
![[llms_01.png|550]]
- In the **original-designed Post-LN** [[Transformer|transformer]], [[Layer Normalization|layer normalization]] is placed between the residual blocks, this architecture has achieved state-of-the-art performance in many tasks. The very early pre-trained large language models like [[GPT]] and [[BERT]] also uses Post-LN. 
 + The function of post-LN could be expressed as$$
	x_{l+1} = \textrm{LN}(x_{l}+f_l(x_l)),
	$$where $f_l(\cdot)$ represents FFN or attention function in the $l$-th layer.
- It was noticed that in Post-LN transformers, the scale of the expected gradients grows along with the layer index for the Post-LN Transformer. On the contrary, the scale almost keeps the same for different layers in the Pre-LN Transformer.![[llms_03.png]]
- It causes the **gradient-norm decay exponentially** in backward propagation, which shadows the original intention of introducing residual connections here and leading to **gradient vanishing** when training deep network. $$
	x_{l+1} = \textrm{LN}(x_{l}+f_l(x_l)) = \frac{x_l + f_l(x_l)}{\sqrt{2}} = \frac{x_{l-1}}{2} + \frac{f_{l-1}(x_{l-1})}{2} + \frac{f_l(x_l)}{\sqrt{2}} = \cdots = \frac{x_0}{2^{(l+1)/2}} + \frac{f_0(x_0)}{2^{(l+1)/2}} + \cdots + \frac{f_l(x_l)}{\sqrt{2}}
	$$- It therefore requires a subtly tuned **LR warm-up stage** for the optimizer to perform well. Experiments are conducted to show that this learning rate warm-up stage is essential for training Post-LN transformer models, ![[llms_02.png]]
- In [@xiong2020], the author proposes the **Pre-LN Transformer** which puts **layer normalization inside the residual connection** and places it before all other non-linear transformations. Additionally, the Pre-LN Transformer uses a final layer normalization right before the prediction.
- The function of pre-LN could be expressed as$$
x_{l+1} = x_l + f_l (\textrm{LN}(x_l))
	$$where $f_l(\cdot)$ represents FFN or attention function in the $l$-th layer. $$
	x_{l+1} = x_{l}+f_l(\textrm{LN}(x_l)) = \cdots = x_0 + f_0(\textrm{LN}(x_0)) + \dots+ f_l(\textrm{LN}(x_l))
	$$Now due to the valid residual connection here, there is no more gradient vanishing during back propagation and it no longer requires the warm-up stage for the optimizer to perform well. 
+ N stability and larger LRs for large networks
+ Nowadays, there are other explanations in terms of why pre-LN being better than post-LN as well. In [@nguyen2019], the experiments on recording the global norm of gradients are conducted for both pre-LN and post-LN. We could see that the global norms of gradients in post-LN are also more spiky compared to pre-LN.
	![[llms_04.png|400]]
+ So pre-LN method is also viewed as the stability-inducing aids for training large neural networks.
+ Nowadays there is another variant called "double" norm being used in Grok and Gemma2 which places layer normalization both before and after the FFN or attention functions within the residual block.
	![[llms_05.png|200]]


### LayerNorm V.S. RMSNorm

+ The original transformer uses LayerNorm proposed in [@ba2016],
$$
y = \frac{x-\mathbb{E}[x]}{\sqrt{Var[x]+\epsilon}}\times\gamma + \beta
$$
+ Notable 
+ However, in many modern LMs: [[RMSNorm]] 