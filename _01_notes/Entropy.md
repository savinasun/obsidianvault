### [[dl_moc]] · [[math_moc]]  
# **Entropy**

**Tags**:: #math
**Links**:: [[KL Divergence]] · [[ELBO]] · [[PPL]] · [[MCMC]]


**Information entropy** (Shannon entropy) quantifies the average uncertainty in a random variable. High entropy = more uncertainty = more information needed to describe an outcome.

## Shannon Entropy
---
Introduced by Claude Shannon (1948). For a discrete random variable $X$ with distribution $p: \mathcal{X} \to [0,1]$:
$$
H(X) = \mathbb{E}_p[-\log p(X)] = -\sum_{x \in \mathcal{X}} p(x) \log p(x)
$$

The unit depends on the log base: **bits** (base 2) or **nats** (base $e$). Deep learning almost always uses nats.

**Intuition**: $-\log p(x)$ is the "surprise" or information content of outcome $x$. Rare events are more surprising. Entropy is the expected surprise.

### Properties
1. **Non-negativity**: $H(X) \ge 0$, with equality iff $X$ is deterministic ($p(x^*) = 1$ for some $x^*$)
2. **Maximum entropy**: $H(X) \le \log |\mathcal{X}|$, with equality when $X$ is uniform over $\mathcal{X}$
3. **Concavity**: $H(\lambda p + (1-\lambda)q) \ge \lambda H(p) + (1-\lambda)H(q)$
4. **Continuity**: $H$ is continuous in the probabilities

## Joint and Conditional Entropy
---
**Joint entropy** $H(X, Y)$: uncertainty in the pair $(X, Y)$
$$
H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)
$$

**Conditional entropy** $H(X \mid Y)$: remaining uncertainty in $X$ after observing $Y$
$$
H(X \mid Y) = -\sum_{x,y} p(x,y) \log p(x \mid y) = H(X, Y) - H(Y)
$$

**Chain rule**: $H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$

**Conditioning reduces entropy**: $H(X \mid Y) \le H(X)$, with equality when $X$ and $Y$ are independent.

## Mutual Information
---
**Mutual information** $I(X; Y)$ measures how much knowing $Y$ reduces uncertainty about $X$:
$$
I(X; Y) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X, Y)
$$

In terms of [[KL Divergence]]:
$$
I(X; Y) = D_\text{KL}(p(X,Y) \| p(X)p(Y))
$$

Mutual information is **symmetric**: $I(X;Y) = I(Y;X)$, and **non-negative**: $I(X;Y) \ge 0$.

**Applications in deep learning**:
- Information bottleneck principle: maximize $I(Z; Y)$ (task-relevant info) while minimizing $I(Z; X)$ (redundancy)
- Contrastive learning objectives can be interpreted as maximizing mutual information between views

## Differential Entropy (Continuous Variables)
---
For a continuous random variable $X$ with density $f(x)$, **differential entropy** is:
$$
h(X) = -\int f(x) \log f(x)\, dx
$$

Unlike discrete entropy, differential entropy can be **negative** and is not invariant to reparameterization.

**Gaussian has maximum entropy for fixed variance**: among all distributions with variance $\sigma^2$, the Gaussian $\mathcal{N}(0, \sigma^2)$ maximizes differential entropy:
$$
h(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)
$$

This property motivates using Gaussian priors in [[VAE|VAEs]] and [[ELBO|variational inference]].

## Cross Entropy
---
[[Cross entropy](https://en.wikipedia.org/wiki/Cross-entropy)] between distributions $p$ (true) and $q$ (model) measures the average number of bits needed to encode data from $p$ using a code optimized for $q$:
$$
H(p, q) = -\mathbb{E}_p[\log q] = -\sum_x p(x) \log q(x)
$$

Relationship to entropy and [[KL Divergence]]:
$$
H(p, q) = H(p) + D_\text{KL}(p \| q)
$$

When $p = q$: $H(p, p) = H(p)$ — the cross entropy equals the true entropy.

**Cross entropy as a training loss**: in classification, the target distribution $p$ is a one-hot vector over $C$ classes. The cross entropy loss is:
$$
\mathcal{L}_\text{CE} = -\sum_{c=1}^C \mathbf{1}[y = c] \log \hat p(c) = -\log \hat p(y)
$$

Minimizing this is equivalent to minimizing $D_\text{KL}(p_\text{data} \| p_\theta)$ when $H(p_\text{data})$ is fixed.

## Entropy in Language Models
---
The entropy of a language model $p_\theta$ over a test corpus measures how uncertain the model is per token:
$$
H(p_\theta) \approx -\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})
$$

**Perplexity** ([[PPL]]) is simply the exponentiated entropy: $\text{PPL} = e^{H(p_\theta)}$. A perplexity of $k$ means the model is "as uncertain as if choosing uniformly among $k$ options" at each step.

**Maximum entropy principle**: when all we know about a distribution is some constraints (e.g., its mean), the distribution that maximizes entropy subject to those constraints is the most "honest" — it encodes only what we know and nothing more. This principle underlies exponential family distributions.

## Entropy in Reinforcement Learning
---
Maximum entropy RL (used in SAC, see [[Deep Reinforcement Learning]]) adds an entropy bonus to the reward:
$$
J(\pi) = \sum_t \mathbb{E}\left[r(s_t, a_t) + \alpha H(\pi(\cdot \mid s_t))\right]
$$

The entropy term discourages premature commitment to a single action — the policy is encouraged to explore by maintaining high action diversity. This improves robustness and sample efficiency.
