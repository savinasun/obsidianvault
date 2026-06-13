### [[dl_moc]]  
# **Score-Based Generative Modeling Methods**

**Tags**:: #generative #dl_theo
**Links**:: [Score-Based Generative Modeling](https://yang-song.net/blog/2021/score/#the-score-function-score-based-models-and-score-matching), 

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

