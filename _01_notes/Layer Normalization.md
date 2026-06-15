### [[dl_moc]] · [[learning_methods_moc]]  
# **Layer Normalization**

**Tags**:: #dl_theo
**Links**:: [[Normalization]] · [[RMSNorm]] · [[LLMs]] · [[Transformer]] · [[Internal Covariate Shift]]

## Overview
---
For [[Transformer|Transformers]] and other NLP models, layer normalization ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) yields significantly better performance than [[Normalization|batch normalization]] ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)), in part because **NLP models tend to exhibit greater variance in batch statistics during training**, for example compared to computer vision (Shen et al., 2020). Layer normalization boosts performance in deeper networks chiefly by controlling their gradients (Xu et al., 2019). It re-scales and re-centers activation distributions