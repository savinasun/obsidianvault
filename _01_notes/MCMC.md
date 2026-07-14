### [[dl_moc]]  
# **Markov Chain Monte Carlo**

**Tags**:: #math #dl_theo
**Links**:: [[Score-Based Generative Modeling Methods]] · [[ELBO]] · [[Entropy]]


**MCMC** methods sample from probability distributions that are known up to a normalizing constant — exactly the case in Bayesian inference and generative modeling where $p(x) \propto \tilde{p}(x)$ but $\int \tilde{p}(x)\,dx$ is intractable.

## Rejection Sampling
---
**Rejection sampling** generates samples from a target $f(x)$ using a simpler proposal $g(x)$:

#### Algorithm
1. Find constant $M$ such that $f(x) \le M\cdot g(x)$ for all $x$
2. Sample candidate $x^* \sim g$
3. Sample $u \sim \text{Uniform}(0, M\cdot g(x^*))$
4. **Accept** $x^*$ if $u \le f(x^*)$; **reject** otherwise
5. Accepted samples are distributed as $f$

![[mcmc_sampling.png]]

**Limitation**: acceptance rate $= 1/M$ decreases exponentially in high dimensions — unusable for deep learning applications.

## Importance Sampling
---
Rather than generating samples from $f$, compute expectations under $f$ using samples from $g$:
$$
\mathbb{E}_f[h(x)] = \int h(x) f(x)\,dx = \int h(x) \frac{f(x)}{g(x)} g(x)\,dx \approx \frac{1}{N}\sum_{i=1}^N h(x^{(i)}) w^{(i)}
$$
where $x^{(i)} \sim g$ and **importance weights** $w^{(i)} = f(x^{(i)}) / g(x^{(i)})$.

Importance sampling is unbiased but has **high variance** when $g$ poorly covers the tails of $f$.

## Markov Chains
---
A **Markov chain** is a sequence of random variables $x_0, x_1, x_2, \ldots$ where the distribution of $x_{t+1}$ depends only on $x_t$:
$$
P(x_{t+1} \mid x_t, x_{t-1}, \ldots, x_0) = T(x_{t+1} \mid x_t)
$$
where $T$ is the **transition kernel**.

A chain has a **stationary distribution** $\pi(x)$ if $\pi(x') = \int T(x' \mid x)\pi(x)\,dx$. Under mild conditions (irreducibility + aperiodicity), the chain converges to $\pi$ from any starting point.

**Detailed balance** (sufficient condition for $\pi$ to be stationary):
$$
\pi(x)\, T(x' \mid x) = \pi(x')\, T(x \mid x') \quad \forall x, x'
$$

## Metropolis-Hastings Algorithm
---
MH constructs a chain with target stationary distribution $\pi(x) \propto \tilde\pi(x)$:

1. Start at $x_0$
2. At step $t$, propose $x^* \sim q(\cdot \mid x_t)$ (any proposal distribution)
3. Compute acceptance ratio:
$$
\alpha = \min\left(1,\, \frac{\tilde\pi(x^*)\, q(x_t \mid x^*)}{\tilde\pi(x_t)\, q(x^* \mid x_t)}\right)
$$
4. Accept: $x_{t+1} = x^*$ with probability $\alpha$; otherwise $x_{t+1} = x_t$

**Key insight**: we only need $\tilde\pi$ (unnormalized), since the normalizing constant cancels in $\alpha$.

**Random Walk MH**: proposal $q(x^* \mid x_t) = \mathcal{N}(x_t, \sigma^2 I)$ — symmetric, so $q$ terms cancel.

## Gibbs Sampling
---
For multivariate distributions $\pi(x_1, \ldots, x_d)$, Gibbs sampling updates one variable at a time by sampling from the **conditional**:

1. Initialize $(x_1^{(0)}, \ldots, x_d^{(0)})$
2. For each iteration:
   - Sample $x_1^{(t+1)} \sim \pi(x_1 \mid x_2^{(t)}, \ldots, x_d^{(t)})$
   - Sample $x_2^{(t+1)} \sim \pi(x_2 \mid x_1^{(t+1)}, x_3^{(t)}, \ldots, x_d^{(t)})$
   - ... continue for all dimensions

Gibbs is a special case of MH with acceptance rate $\alpha = 1$. It works when conditionals are tractable (e.g., conjugate priors in Bayesian models).

## Langevin Dynamics
---
**Langevin dynamics** is an MCMC method that uses the **score function** $\nabla_x \log \pi(x)$ (gradient of log-density) to guide sampling:
$$
x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log \pi(x_t) + \sqrt{\eta}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

- The gradient term **drifts** the chain toward high-probability regions
- The noise term ensures **exploration** and prevents collapse

**Stochastic Gradient Langevin Dynamics (SGLD)**: uses mini-batch gradient estimates, making Langevin dynamics scalable to large datasets.

### Connection to Score-Based Generative Models
This is the direct link to [[Score-Based Generative Modeling Methods]]: if we train a neural network $s_\theta(x) \approx \nabla_x \log p_\text{data}(x)$ (the **score function**), we can run Langevin dynamics using $s_\theta$ instead of the true score — effectively sampling from $p_\text{data}$ without ever knowing the normalizing constant.

$$
x_{t+1} = x_t + \frac{\eta}{2} s_\theta(x_t) + \sqrt{\eta}\, \epsilon_t
$$

This is the core idea behind denoising score matching and, ultimately, [[DDPM|diffusion models]].

## Burn-In and Mixing
---
- **Burn-in**: the first $B$ samples are discarded, since the chain hasn't yet reached stationarity
- **Mixing time**: how many steps until the chain distribution is close to $\pi$; poor mixing → highly correlated samples
- **Thinning**: keep every $k$-th sample to reduce autocorrelation

In practice for deep learning, these issues motivate using diffusion models (which sample via a fixed, finite sequence of learned denoising steps) over raw MCMC.
