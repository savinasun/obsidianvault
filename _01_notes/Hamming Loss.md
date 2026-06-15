### [[dl_moc]] · [[math_moc]]  
# **Hamming Loss**

**Tags**::  #math
**Links**:: [[Benchmark Datasets]]


The Hamming Loss (HL) is the fraction of the wrong labels to the total number of labels.

## Hamming Distance
---
The Hamming loss is the average of the **hamming distances** between the predicted label $\hat y$ and the true label $y$ over $N$ samples and $L$ classes. 

$$
\frac1 {N\cdot L} \sum\limits_{i=1}^N\sum\limits_{\ell=1}^L \mathbb{1}\{y_{i\ell}\neq \hat{y}_{i\ell}\}
$$
For binary classification, HL = 1 - Accuracy;