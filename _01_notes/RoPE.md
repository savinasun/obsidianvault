### [[dl_moc]]  
# **Rotary Position Embeddings**

**Tags**::  
**Links**::

## Implementation
---
+ **High level intuition**: A relative position embedding should be some $f(x,i)$ which defines the inner product s.t.
$$
<f(x,i), f(x,j)> = g(x, y, i-j)
$$ which means the attention function should only depend on relative position $(i-j)$.
+ Existing variations in position embeddings, like Sine embeddings in the original transformer and Absolute embeddings all fail to be 'relative'. And Relative embeddings is not an inner product.
+ RoPE was proposed in [@su2023] to solve the problem.
+ We want our embeddings to be invariant to absolute position and we know that inner products are invariant to arbitrary rotation. So the math here is to pair up the coordinates into $d/2$ subspaces and rotate them in 2d
+ ![[rope_01.png]]

