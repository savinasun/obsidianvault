### [[dl_moc]]  
## **Denoising Diffusion Probabilistic Model**
**Tags**:: #generative
**Links**:: [[DDIM]] · [[Score-Based Generative Modeling Methods]] · [[Score Matching]] · [[Flow Matching]] · [[MCMC]] · [[LDM]] · [[VAE]] · [[ELBO]] · [[FID]]

DDPM; [Ho et al. 2020](https://arxiv.org/abs/2006.11239) defines a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise.

**Diffusion Process (Forward Diffusion) $q_{\phi}$**
![[ddpm_forward_diffusion.png|500]]

**Denoise Process (Backward Diffusion) $p_{\theta}$**
![[ddpm_backward_diffusion.png|500]]

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
 
 ![[ddpm_reverse_process.png]]

## Training Objective
---
We cannot directly minimize the KL divergence between $q(x_{t-1}\mid x_t)$ and $p_\theta(x_{t-1}\mid x_t)$ because $q(x_{t-1}\mid x_t)$ is intractable. However, the **posterior conditioned on $x_0$** is tractable:
$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\, \tilde\mu_t(x_t, x_0),\, \tilde\beta_t I\right)
$$
$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t, \quad \tilde\beta_t = \frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}
$$

[Ho et al., 2020](https://arxiv.org/abs/2006.11239) showed that training $p_\theta$ to predict the **added noise** $\epsilon$ (rather than $\mu_\theta$ directly) gives the best sample quality. Using the reparameterization $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$:

$$
\boxed{\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\,\epsilon \sim \mathcal{N}(0,I),\,t}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}
$$

This is a simple MSE loss: the U-Net $\epsilon_\theta(x_t, t)$ predicts the noise $\epsilon$ that was added to $x_0$ to get $x_t$. At inference, the predicted noise is used to compute $\mu_\theta(x_t, t)$ for the reverse step.

**Connection to score matching**: $\epsilon_\theta(x_t, t) = -\sqrt{1-\bar\alpha_t}\, s_\theta(x_t, t)$, where $s_\theta \approx \nabla_{x_t} \log q(x_t)$. See [[Score-Based Generative Modeling Methods]] for the full score matching framework, and [[Score Matching]] for the denoising-score-matching derivation of $\mathcal{L}_\text{simple}$ (this same relation in flow-matching time reads $\epsilon_t^\theta = -\beta_t s_t^\theta$).

## DDIM: Deterministic Sampling
---
Standard DDPM sampling is stochastic (requires $T = 1000$ steps). **[[DDIM]]** ([Song et al., 2020](https://arxiv.org/abs/2010.02502)) reuses the same trained network with a non-Markovian forward process sharing the marginals $q(x_t \mid x_0)$, giving a **deterministic** noise-to-image mapping that enables few-step sampling (e.g. 50 instead of 1000), latent interpolation via inversion, and consistent outputs — see [[DDIM]] for the sampling rule and the probability-flow-ODE view.

## Variance Schedule
---
The noise schedule $\{\beta_t\}$ controls how quickly the data is destroyed:

- **Linear** (original DDPM): $\beta_t$ linearly from $10^{-4}$ to $0.02$; too much noise at small $t$ for low-resolution images
- **Cosine** ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)): $\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1+s}\frac{\pi}{2}\right)$; preserves more structure at small $t$, improving sample quality
- **EDM schedule** ([Karras et al., 2022](https://arxiv.org/abs/2206.00364)): continuous-time, treats noise level $\sigma$ as a continuous variable; enables second-order ODE solvers