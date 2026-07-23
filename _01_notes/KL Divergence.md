### [[dl_moc]]  
# **Kullback-Leibler Divergence**

**Tags**:: #math #dl_theo
**Links**:: [[Entropy]] · [[ELBO]] · [[VAE]] · [[Score-Based Generative Modeling Methods]] · [[Policy Gradient Methods]]


**KL divergence** (**relative entropy**) measures how much one probability distribution $q$ differs from a reference distribution $p$. It is fundamental to variational inference, generative models, and policy optimization.

## Definition
---
$$
D_\text{KL}(q \| p) = \mathbb{E}_{X\sim q}\!\left[\log \frac{q(X)}{p(X)}\right] = \int q(x) \log \frac{q(x)}{p(x)}\, dx \ge 0
$$

**Always non-negative** (**Gibbs' inequality**): $D_\text{KL}(q \| p) = 0$ if and only iff $q = p$ almost everywhere.

**Proof sketch**: by Jensen's inequality applied to the convex function $-\log$:
$$
-D_\text{KL}(q\|p) = \mathbb{E}_q\!\left[\log\frac{p}{q}\right] \le \log \mathbb{E}_q\!\left[\frac{p}{q}\right] = \log \int p\, dx = 0
$$

## Properties
---
1. **Asymmetry**: $D_\text{KL}(q \| p) \neq D_\text{KL}(p \| q)$ in general — KL divergence is **not a metric**
2. **Relationship to entropy**: $D_\text{KL}(p \| q) = H(p, q) - H(p)$ where $H(p,q)$ is cross entropy (see [[Entropy]])
3. **Information gain**: $D_\text{KL}(p \| q)$ measures the information gained when updating from **prior** $q$ to **posterior** $p$
4. **Undefined when $p(x) > 0$ but $q(x) = 0$**: the KL term diverges to $+\infty$; $q$ must cover the support of $p$


## KL-Divergence of Two Normal Distributions
---
$$q(x)\sim\mathcal{N}(x;\mu_q,\sigma_q^2 I^d), \quad p(x)\sim\mathcal{N}(x;\mu_p,\sigma_p^2 I^d)$$
$$
D_{KL}(q\|p)=\frac 1 2 \left[\kappa \bigg(\frac{\sigma_q^2}{\sigma_p^2}\bigg) + \frac{\|\mu_q - \mu_p\|^2}{\sigma_p^2} \right],
$$
where $\kappa(\alpha) = \sum_i \alpha_i - \log\alpha_i -1$.


## Forward KL vs. Backward KL
---
![[entropy_01.png]]

The direction of KL divergence determines the mode behavior of the approximating distribution $q$:

### Forward KL: $D_\text{KL}(p \| q)$
$$
D_\text{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)}\, dx
$$
- **Zero-avoiding** (mass-covering): when $p(x) > 0$, the term $p(x)\log(p(x)/q(x))$ blows up if $q(x) \to 0$. So $q$ is forced to place probability mass everywhere $p$ is non-zero.
- **Result**: $q$ tends to be **overdispersed** — it covers all modes of $p$, potentially spreading mass in low-probability regions between modes.
- **Used in**: maximum likelihood estimation (minimizing $D_\text{KL}(p_\text{data} \| p_\theta)$), expectation propagation.

### Backward KL: $D_\text{KL}(q \| p)$
$$
D_\text{KL}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)}\, dx
$$
- **Zero-forcing** (mode-seeking): when $q(x) > 0$, the term blows up if $p(x) \to 0$. So $q$ avoids regions where $p$ is near zero.
- **Result**: $q$ tends to be **underdispersed** — it latches onto one mode of $p$ and ignores others.
- **Used in**: variational inference (ELBO maximization — see [[ELBO]]), VAE training.

**Mode-seeking vs. mode-covering**: in a multimodal $p$, forward KL produces a broad $q$ that averages over modes; backward KL produces a sharp $q$ that picks one mode. Neither is universally better — depends on the application.

## Relationships to Entropy and Cross-Entropy
---
The triangular relationship:
$$
D_\text{KL}(p \| q) = H(p, q) - H(p)
$$

where $H(p,q) = -\mathbb{E}_p[\log q]$ is cross entropy and $H(p) = -\mathbb{E}_p[\log p]$ is entropy.

**Consequence**: minimizing cross entropy $H(p, q)$ with respect to $q$ is equivalent to minimizing $D_\text{KL}(p \| q)$, since $H(p)$ is constant w.r.t. $q$. This is why cross-entropy loss in classification is equivalent to KL minimization.

## Jensen-Shannon Divergence
---
The **Jensen-Shannon divergence** (JSD) is a symmetrized, bounded version of KL:
$$
D_\text{JS}(p \| q) = \frac{1}{2} D_\text{KL}\!\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2} D_\text{KL}\!\left(q \,\Big\|\, \frac{p+q}{2}\right)
$$

- **Symmetric**: $D_\text{JS}(p\|q) = D_\text{JS}(q\|p)$
- **Bounded**: $0 \le D_\text{JS}(p\|q) \le \log 2$ (for base-2 log)
- $\sqrt{D_\text{JS}}$ is a proper metric (satisfies the triangle inequality)
- **Used in GANs**: the original GAN objective minimizes JSD between the real and generated distributions.

## Applications in Deep Learning
---
### Variational Inference / VAE
The [[ELBO]] objective contains a backward KL term that regularizes the encoder:
$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_\text{KL}(q_\phi(z|x) \| p(z))
$$
The KL term keeps the variational posterior close to the prior $\mathcal{N}(0,I)$.

### Policy Optimization (PPO/TRPO)
TRPO explicitly constrains the KL divergence between old and new policies:
$$
\mathbb{E}_s[D_\text{KL}(\pi_\text{old}(\cdot|s) \| \pi_\theta(\cdot|s))] \le \delta
$$
PPO approximates this constraint with a clipped objective. See [[Policy Gradient Methods]].

### Knowledge Distillation
The student model is trained to minimize $D_\text{KL}(p_\text{teacher} \| p_\text{student})$ — forward KL, so the student covers all modes of the teacher's soft predictions.

### Score-Based Models
The score function $\nabla_x \log p(x)$ is the gradient of the log-density, which appears in the [[ELBO]] derivation and in [[MCMC|Langevin dynamics]]. Fisher divergence (used in [[Score-Based Generative Modeling Methods]]) can be interpreted as a KL in function space.

## Numerical Stability
---
In practice, compute KL in log-space to avoid underflow:
```python
# PyTorch: F.kl_div expects log-probabilities for input
kl = F.kl_div(q.log(), p, reduction='batchmean')

# Closed-form KL for Gaussians (used in VAE):
kl = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(-1)
```

The closed-form KL between two Gaussians:
$$
D_\text{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$
