### [[dl_moc]]  
# **Score-Based Generative Modeling Methods**

**Tags**:: #generative #dl_theo
**Links**:: [[DDPM]] · [[MCMC]] · [[Manifold Hypothesis]] · [[ELBO]] · [Score-Based Generative Modeling](https://yang-song.net/blog/2021/score/)

## Score Matching
---
Let  $f_{\theta}(x)\in\mathbb{R}$ be a real-valued function parameterized by a learnable parameter $\theta$, we may define a p.d.f. by
$$
p_{\theta}(x) = \frac{e^{-f_{\theta}(x)}}{z_{\theta}},
$$
where $z_{\theta}>0$ s.t. $\int p_{\theta}(x) dx = 1$.  

$f_{\theta}(x)$ is often called an **unnormalized probabilistic model**, or **energy-based model**.

We can train $p_{\theta}(x)$ by maximizing the log-likelihood of the data
$$
\max_{\theta} \sum\limits_{i=1}^N \log p_{\theta}(x_i).
$$
#### Stein Score Function
The **score function** of a distribution $p(x)$ is defined as
$$
\nabla_x \log p(x)
$$
and a model for the score function is called a **score-based model**, which we denote as $s_{\theta}(x)$. Since
$$
s_{\theta}(x)=\nabla_x \log p_{\theta}(x) = -\nabla_x f_{\theta}(x) - \nabla_x \log z_{\theta} = -\nabla_x f_{\theta}(x).
$$
The **score-based model** is learned such that $s_{\theta}(x) \approx \nabla_x \log p_\theta(x)$, and can be parameterized without worrying about the normalizing constant.

#### Fisher Divergence
Similar to likelihood-based models, we can train score-based models by minimizing the **Fisher divergence** between the model and the data distributions, defined as
$$
\mathbb{E}_{p(x)} \left[\|\nabla_x \log p(x) - s_{\theta}(x)\|_2^2\right] = \int p(x) \|\nabla_x \log p(x) - s_{\theta}(x)\|_2^2 dx.
$$

#### Score Matching Method
Score Matching is a family of methods which minimizes the Fisher divergence without knowledge of the ground-truth data score $\nabla_x \log p(x)$. 

**Score matching objectives** can directly be estimated on a dataset and optimized with SGD, analogous to the log-likelihood objective for training likelihood-based models **(with known normalizing constants)**. 

In summary, we can represent a distribution by modeling its score function $s_{\theta}(x)$, which can be estimated by **training a score-based model of free-form architectures** with score matching. A score network $s_{\theta}: \mathbb{R}^D \to \mathbb{R}^D$ is often trained to estimate the score function.

## Langevin Dynamics
---
Once we have trained a score-based model , we can use an iterative procedure called **Langevin dynamics**(a special [[MCMC]] method) to draw samples from it.

**Initialization:** an arbitrary prior distribution $x_0\sim\pi(x)$
**Iteration:** 
$$
x_{i+1} \leftarrow x_i+ \epsilon \nabla_x \log p(x) + \sqrt{2\epsilon} z_i, \quad i=0, 1,\dots, K
$$
where $z_i \sim \mathcal{N}(0, I)$.

When $\epsilon\to 0$ and $K\to \infty$, $X_K$ converges to a sample drawn from distribution $p(x)$ under some regularity conditions.

#### Caveat
According to [[Manifold Hypothesis]], most of the data lives in a low-dimensional manifold in high-dimensional space. It brings a negative effect on score estimation since in low density regions, where few data points are available for computing the score matching objective. Therefore, score matching has had limited success in practice, and a noise-perturbed score-based method come up to address this problem.
#### Annealed Langevin Dynamics
---
The manifold problem means that score estimation is unreliable in low-density regions (few data points → noisy score estimates). **Annealed Langevin Dynamics** (Song & Ermon, 2019) addresses this by perturbing the data with noise at multiple scales:

1. Define a noise schedule $\sigma_1 > \sigma_2 > \cdots > \sigma_L$ (e.g., geometric sequence)
2. For each noise level $\sigma_i$, define a perturbed data distribution:
$$
q_{\sigma_i}(x) = \int p_\text{data}(x') \mathcal{N}(x; x', \sigma_i^2 I)\, dx'
$$
3. Train a **single noise-conditioned score network** $s_\theta(x, \sigma)$ that estimates the score at all noise levels:
$$
s_\theta(x, \sigma) \approx \nabla_x \log q_\sigma(x)
$$
   Training objective (denoising score matching at each noise level):
$$
\mathcal{L} = \sum_i \lambda(\sigma_i)\, \mathbb{E}_{x \sim p_\text{data},\, \tilde x \sim \mathcal{N}(x, \sigma_i^2 I)}\left[\left\|s_\theta(\tilde x, \sigma_i) + \frac{\tilde x - x}{\sigma_i^2}\right\|^2\right]
$$
4. **Sampling** via annealed Langevin: run Langevin dynamics starting from large $\sigma_1$ (data is spread across all of $\mathbb{R}^D$, score is well-defined everywhere), then gradually anneal to small $\sigma_L$ (score approaches $\nabla_x \log p_\text{data}$)

**Why it works**: at large $\sigma$, the noisy distribution fills the entire space — there are no low-density voids — so score estimation is reliable. As $\sigma$ decreases, the samples are guided toward the data manifold.

## Connection to DDPM
---
The connection between score-based models and [[DDPM]] (Ho et al., 2020) was unified by Song et al. (2020) under a **stochastic differential equation (SDE)** framework:

- The **forward diffusion process** in DDPM is a discrete approximation of adding Gaussian noise (a variance-exploding or variance-preserving SDE)
- The **reverse denoising process** corresponds to reversing this SDE, which requires the score function $\nabla_x \log p_t(x)$ at each time $t$
- The DDPM noise prediction network $\epsilon_\theta(x_t, t)$ is a **reparameterized score network**:
$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sigma_t}
$$

Both DDPM and score-based models are instances of the same diffusion-score framework. The SDE formulation also enables:
- **Probability flow ODE**: deterministic sampling (no stochastic noise), enabling exact likelihood computation
- **DDIM**: faster sampling by taking larger steps along the probability flow ODE (see [[DDPM]])

## Denoising Score Matching
---
Training the score network with the exact score $\nabla_x \log p(x)$ is intractable. **Denoising score matching** (Vincent, 2011) provides a tractable alternative:

Perturb data $x$ with noise $\tilde x = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$. Then:
$$
\nabla_{\tilde x} \log p_\sigma(\tilde x | x) = -\frac{\tilde x - x}{\sigma^2}
$$

A score network trained to predict this *conditional* score implicitly learns the *marginal* score $\nabla_x \log p_\sigma(x)$. The training objective becomes:
$$
\mathcal{L}_\text{DSM} = \mathbb{E}_{x \sim p_\text{data},\, \epsilon \sim \mathcal{N}(0, I)}\left[\left\|s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\|^2\right]
$$
This is the same as the DDPM noise prediction loss up to reparameterization — the two frameworks are formally equivalent.
