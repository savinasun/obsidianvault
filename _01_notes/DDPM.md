### [[dl_moc]]  
## **Denoising Diffusion Probabilistic Model**
**Tags**::  
**Links**:: [Diffusion Model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

DDPM; [Ho et al. 2020](https://arxiv.org/abs/2006.11239) defines a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise.
![[Pasted image 20260601154614.png]]

## Forward Diffusion Process
---
Given a data point sampled from a real data distribution $x_0 \sim q(x)$, we define a **forward diffusion process** $\{x_t\}$ by adding small amount of Gaussian noise to the sample in $T$ steps. The step sizes are controlled by a **variance schedule** $\{\beta_t \in (0,1)\}_{t=1}^T$.
$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$
and $q(x_{1:T}|x_0)=\prod_{t=1}^T q(x_t|x_{t-1})$.

The closed form for $x_t$ could be derived as
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
$$
using re-parameterization trick, where $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.
Therefore
$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
$$

## Reverse Diffusion Process
---
To recreate the true sample $x_0$ from a Gaussian noise input $x_T \sim \mathcal{N}(0, I)$, we need to estimate and sample from $q(x_{t-1}|x_t)$, and it is called the **reverse diffusion process**.

We may train a neural network $p_{\theta}$ to approximate the conditioned probability distributions in the reverse diffusion process $q(x_{t-1}|x_t)$ in order to run the reverse diffusion process.
Note that
$$
p_{\theta}(x_{0:T}) = p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)
$$
 and
 $$
p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma(x_t, t))
$$
 
 ![[Pasted image 20260601151304.png]]