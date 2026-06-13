### [[dl_moc]]  
# **Layer Normalization**

**Tags**:: #dl_theo
**Links**::

## Overview
---
For Transformers and other NLP models, layer normalization (Ba et al., 2016) yields significantly better performance than batch normalization (Ioffe and Szegedy, 2015), in part because **NLP models tend to exhibit greater variance in batch statistics during training**, for example compared to computer vision (Shen et al., 2020). Layer normalization boosts performance in deeper networks chiefly by controlling their gradients (Xu et al., 2019). It re-scales and re-centers activation distributions