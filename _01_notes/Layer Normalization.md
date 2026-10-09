### [[dl_moc]]  
# **Layer Normalization**

**Tags**:: #dl_theo
**Links**:: [[Normalization]] · [[RMSNorm]] · [[LLMs]] · [[Transformer]] · [[Internal Covariate Shift]] · [[FiLM]] · [[DiT]]

## Overview
---
For [[Transformer|Transformers]] and other NLP models, layer normalization ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) yields significantly better performance than [[Normalization|batch normalization]] ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)), in part because **NLP models tend to exhibit greater variance in batch statistics during training**, for example compared to computer vision ([Shen et al., 2020](https://arxiv.org/abs/2003.07845)). Layer normalization boosts performance in deeper networks chiefly by controlling their gradients ([Xu et al., 2019](https://arxiv.org/abs/1911.07013)). It re-scales and re-centers activation distributions

## Conditional Variant: adaLN / adaLN-Zero
---
The learned affine $\gamma \hat x + \beta$ can be replaced by a scale and shift *predicted from a conditioning vector* (LN's own affine disabled): this is **adaLN**, a [[FiLM]] layer on top of LayerNorm. Adding a zero-initialized gate on each residual branch gives **adaLN-Zero**, the conditioning mechanism of [[DiT]], which makes every Transformer block start as the identity. Equations, initialization details and the DiT ablation live in [[Normalization#Adaptive Layer Norm (adaLN) and adaLN-Zero]].
