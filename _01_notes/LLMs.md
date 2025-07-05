### [[dl_moc]]  
# **Large Language Models**

**Tags**::  
**Links**::

## Architecture Variants
---
### Pre-Layer Normalization V.S. Post-Layer Normalization

- In the **original-designed Post-LN** [[Transformer|transformer]](https://sh-tsang.medium.com/review-attention-is-all-you-need-transformer-96c787ecdec1), which places the [[Layer Normalization|layer normalization]](https://sh-tsang.medium.com/overview-my-reviewed-paper-lists-tutorials-946ce59fbf9e#:~:text=%5D%20%5BGELU%5D%20%5B-,Layer%20Norm,-%2C%20LN%5D%0A2017) between the residual blocks, **the expected gradients** of the parameters near the output layer are **large**. 
- Experiments are conducted to show that this learning rate warm-up stage is essential for training Post-LN transformer models.
	![[llms_02.png]]
- Warm-up stage significantly slows down the optimization process and requires hyper-parameter tuning which is computationally expensive.
- According to [@xiong2020], the scale of the activations in the pre-LN transformer keeps to be the same in activations and therefore warm-up stage could be skipped.
- The architecture of Post-LN versus Pre-LN:
![[llms_01.png]]

+ Post-LN requires a **LR warm-up stage** for the optimizer to perform well
+ Pre-LN: requires a final layer normalization right before prediction


- 