### [[dl_moc]]  
# **Rotary Position Embeddings**

**Tags**::  
**Links**::

## Implementation
---
**High level intuition**: A relative position embedding should be some $f(x,i)$ which defines the inner product s.t.
$$
<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m-n)
$$which means the attention function should **only depend on relative position $(m-n)$**. And there is a solution s.t.
$$
\begin{aligned}
& f_q(x_m, m) = (W_q x_m)e^{im\theta},\\\\
& f_k(x_n, n) = (W_k x_n)e^{in\theta},\\\\
& g(x_m, x_n, m-n) = \text{Re}\left[(W_q x_m) (W_k x_n)^* e^{i(m-n)\theta}\right].
\end{aligned}
$$

+ Existing variations in position embeddings, like Sine embeddings in the original transformer and Absolute embeddings all fail to be 'relative'. And Relative embeddings is not an inner product.
+ RoPE was proposed in [@su2023] to solve the problem.
+ The main idea is that **the q and k vectors incorporate absolute position information**, and then **the updated q and k vectors will bring relative position information by doing inner product during attention computation**.

+ We want our embeddings to be invariant to absolute position and we know that inner products are invariant to arbitrary rotation. So the math here is to pair up the coordinates into $d/2$ subspaces and rotate them in $2d$ spaces


## Proof
---
Let's start with a 2D case (i.e. $d=2$).

$$
f_q(x_m, m) = 
\begin{pmatrix}
  \cos{m\theta} & -\sin{m\theta} \\
\sin{m\theta} & \cos{m\theta}
\end{pmatrix}
\begin{pmatrix}
q_m^1 \\
q_m^2
\end{pmatrix}
$$
$$
f_k(x_n, n) = 
\begin{pmatrix}
  \cos{n\theta} & -\sin{n\theta} \\
\sin{n\theta} & \cos{n\theta}
\end{pmatrix}
\begin{pmatrix}
k_n^1 \\
k_n^2
\end{pmatrix}
$$
Then
$$
\begin{aligned}
<f_q(x_m, m), f_q(x_n, n)> &= f_q(x_m, m)^T f_k(x_n, n)\\
&= (q_m^1 k_n^1 + q_m^2 k_n^2)\cos{(m-n)\theta} - (q_m^2 k_n^1 - q_m^1 k_n^2)\sin{(m-n)\theta}
\end{aligned}
$$
$$
\begin{aligned}
g(x_m, x_n, m-n) &= \text{Re}\left[(W_qx_m)(W_kx_n)^* \cdot e^{i(m-n)\theta} \right]\\
&= \text{Re} \left[q_m k_n^* e^{i(m-n)\theta}\right]\\
&= \text{Re} \left[(q_m^1+iq_m^2)(k_n^1-ik_n^2)\left[cos(m-n)\theta+i\cdot \sin{(m-n)\theta} \right]\right]\\
&= (q_m^1 k_n^1 + q_m^2 k_n^2)\cos{(m-n)\theta} - (q_m^2 k_n^1 - q_m^1 k_n^2)\sin{(m-n)\theta}\\
&= <f_q(x_m, m), f_k(x_n, n)>
\end{aligned}
$$

Generalize to $d>2$ cases


The authors try to formulate the relative relationship between tokens like below, where **the dot product between tokens injected with absolute position information inherently contains relative position information**.
$$
<f_q(x_m, m), f_q(x_n, n)> = g(x_m, x_n, m-n)
$$

+ ![[rope_01.png|782]]

