### [[dl_moc]]  
# **Large Language Models**

**Tags**:: #llm #architecture
**Links**:: [[Transformer]] · [[Attention]] · [[RoPE]] · [[RMSNorm]] · [[LLaMA]] · [[MoE]] · [[Scaling Laws]] · [[Optimization Algorithms]]


**Architecture Variations**
+ Pre-LN vs. Post-LN
+ LayerNorm vs. RMSNorm
+ [[RoPE|Rotary Positional Encodings]]
+ No Bias Terms
+ Parallel Layers vs. Serial Layers
+ KV Cache
+ 


## Pre-Layer Normalization vs. Post-Layer Normalization
---
![[llms_01.png|550]]
- In the **original-designed Post-LN** [[Transformer|transformer]], [[Layer Normalization|layer normalization]] is placed between the residual blocks, this architecture has achieved state-of-the-art performance in many tasks. The very early pre-trained large language models like [[GPTs]] and [[BERT]] also uses Post-LN. 
 + The function of post-LN could be expressed as$$
	x_{l+1} = \textrm{LN}(x_{l}+f_l(x_l)),
	$$where $f_l(\cdot)$ represents FFN or attention function in the $l$-th layer.
- It was noticed that in Post-LN transformers, the scale of the expected gradients grows along with the layer index for the Post-LN Transformer. On the contrary, the scale almost keeps the same for different layers in the Pre-LN Transformer.![[llms_03.png]]
- It causes the **gradient-norm decay exponentially** in backward propagation, which shadows the original intention of introducing residual connections here and leading to **gradient vanishing** when training deep network. $$
	x_{l+1} = \textrm{LN}(x_{l}+f_l(x_l)) = \frac{x_l + f_l(x_l)}{\sqrt{2}} = \frac{x_{l-1}}{2} + \frac{f_{l-1}(x_{l-1})}{2} + \frac{f_l(x_l)}{\sqrt{2}} = \cdots = \frac{x_0}{2^{(l+1)/2}} + \frac{f_0(x_0)}{2^{(l+1)/2}} + \cdots + \frac{f_l(x_l)}{\sqrt{2}}
	$$- It therefore requires a subtly tuned **LR warm-up stage** for the optimizer to perform well. Experiments are conducted to show that this learning rate warm-up stage is essential for training Post-LN transformer models, ![[llms_02.png]]
- In [Xiong et al., 2020](https://arxiv.org/abs/2002.04745), the author proposes the **Pre-LN Transformer** which puts **layer normalization inside the residual connection** and places it before all other non-linear transformations. Additionally, the Pre-LN Transformer uses a final layer normalization right before the prediction.
- The function of pre-LN could be expressed as$$
x_{l+1} = x_l + f_l (\textrm{LN}(x_l))
	$$where $f_l(\cdot)$ represents FFN or attention function in the $l$-th layer. $$
	x_{l+1} = x_{l}+f_l(\textrm{LN}(x_l)) = \cdots = x_0 + f_0(\textrm{LN}(x_0)) + \dots+ f_l(\textrm{LN}(x_l))
	$$Now due to the valid residual connection here, there is no more gradient vanishing during back propagation and it no longer requires the warm-up stage for the optimizer to perform well. 
+ N stability and larger LRs for large networks
+ Nowadays, there are other explanations in terms of why pre-LN being better than post-LN as well. In [Nguyen & Salazar, 2019](https://arxiv.org/abs/1910.05895), the experiments on recording the global norm of gradients are conducted for both pre-LN and post-LN. We could see that the global norms of gradients in post-LN are also more spiky compared to pre-LN.
	![[llms_04.png|400]]
+ So pre-LN method is also viewed as the stability-inducing aids for training large neural networks.
+ Nowadays there is another variant called "double" norm being used in Grok and Gemma2 which places layer normalization both before and after the FFN or attention functions within the residual block.
	![[llms_05.png|200]]


## LayerNorm vs. RMSNorm
---
+ The original transformer uses **LayerNorm** proposed in [Ba et al., 2016](https://arxiv.org/abs/1607.06450),
$$
y = \frac{x-\mathbb{E}[x]}{\sqrt{Var[x]+\epsilon}}\times\gamma + \beta
$$
where $\gamma$ and $\beta$ are learnable parameters.
+ Notable models uses LayerNorm: GPT-1, GPT-2, GPT-3, GPT-J, BLOOM
+ However, in most modern LMs, [[RMSNorm]] is more often used due to its fewer parameters to move around and being memory efficient. It also works as well as LayerNorm. The runtime (and surprisingly, perf) gains have been seen in papers (it shows the results of ablation studies)
![[llms_06.png]]
+ Notable models uses RMSNorm: LLaMA, PaLM, Chinchilla


## Dropping Bias Terms
---
+ **Most modern transformers don’t have bias terms.**
+ Original Transformer:$$
	\textrm{FFN}(x) = max(0, xW_1+b_1)W_2+b2
	$$
+ Modern Transformer (if not gated):$$
	\textrm{FFN}(x) = \sigma(xW_1)W_2
	$$
+ The reason for the change is also due to memory efficiency and training stability.


## Parallel Layers vs. Serial Layers
---
+ Normal transformer blocks are serial – they compute attention, then the MLP
+ A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. Originally in GPT-J!
+ ![[llms_07.png]]
+ If implemented correctly, LayerNorm can be shared, and matrix multiplies can be fused
+ Recent Models: Cohere Command A, Falcon 2 11B, Command R+


## KV Cache
---
![[llama_03.png|500]]
+ Number of entries in caching: 
$$
2\cdot n\cdot d_h \cdot n_h \cdot \ell,
$$
+ $n$ is the number of tokens in the input sequence
+ $d_h$ is dimension of key/value matrices
+ $n_h$ is number of attention heads per layer
+ $\ell$ is number of layers

For DeepSeek R1-V3 architecture, 
$$d_h=128, n_h=128, \ell=61, n=10000$$
the caching size is around $4MB$ per token




## MHA vs. MQA vs. GQA vs. MLA
---

![[llms_08.png|center|500]]







## Hyperparamters
---
### Feed-Forward Ratio
The feed-forward ratio is referring to the ratio between the feed-forward dimension $d_{ff}$ and the model dimension $d_{model}$. It controls the up projection dimension in MLP layer
There are two types of default feed-forward ratios that are commonly used. For those ReLU architecture-typed models, it almost always be true that
$$
d_{ff} = 4 d_{model}.
$$
For those GLU variants-typed models, since GLU variants scale down by 2/3rd, it is by default that people would use
$$𝑑_{𝑓𝑓} = \frac 8 3 d_{model}.$$This is mostly what happens. Some notable such example models are roughly in this range, though PaLM, LLaMA2 and Mistral are slightly larger.

| Model        | $d_{ff}/d_{model}$ |
| ------------ | ------------------ |
| PaLM         | 4                  |
| Mistral 7B   | 3.5                |
| LLaMA-2 70B  | 3.5                |
| LLaMA 70B    | 2.68               |
| Qwen 14B     | 2.67               |
| DeepSeek 67B | 2.68               |
| Yi 34B       | 2.85               |
| T5 v1.1      | 2.5                |


### $d_{head}\times n_{heads} / d_{model}$


### Aspect Ratio



## Weight Decay
---

