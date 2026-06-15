### [[dl_moc]]  
# **Variational Autoencoder (VAE)**

**Tags**:: #generative #dl_theo
**Links**:: [[ELBO]] · [[KL Divergence]] · [[LDM]] · [[Score-Based Generative Modeling Methods]] · [[Manifold Hypothesis]]


A **VAE** ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)) is a latent variable generative model that learns to encode data into a structured latent space and decode back. Unlike standard autoencoders, the latent space is **probabilistic** and **regularized**, making it suitable for generation.

## Generative Model
---
The VAE models data $x$ as generated from a latent variable $z$:

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz
$$

- **Prior**: $p(z) = \mathcal{N}(0, I)$ — standard Gaussian
- **Decoder** (generative model): $p_\theta(x \mid z)$ — neural network parameterized by $\theta$
- **True posterior**: $p_\theta(z \mid x) = p_\theta(x\mid z)\,p(z) / p_\theta(x)$ — **intractable** (requires computing $p_\theta(x)$)

The marginal $p_\theta(x)$ involves an integral over all latent codes — intractable for continuous $z$ in high dimensions. We can't directly maximize $\log p_\theta(x)$.

## Variational Inference: ELBO
---
Introduce a variational posterior $q_\phi(z \mid x)$ (the **encoder**) to approximate the true posterior. By the [[ELBO]] derivation:
$$
\log p_\theta(x) \ge \mathcal{L}(\theta, \phi;\, x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_\text{KL}(q_\phi(z\mid x) \| p(z))}_{\text{regularization}}
$$

Maximizing the ELBO:
1. Makes the decoder reconstruct $x$ well from samples of $q_\phi$
2. Keeps the encoder posterior close to the prior $\mathcal{N}(0, I)$

## Architecture
---
**Encoder** $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$:
- Neural network that maps input $x$ → mean $\mu \in \mathbb{R}^d$ and log-variance $\log\sigma^2 \in \mathbb{R}^d$
- Output is a diagonal Gaussian (assumes independent latent dimensions)

**Decoder** $p_\theta(x \mid z)$:
- For continuous data: $\mathcal{N}(f_\theta(z), \sigma^2 I)$ — reconstruction loss is MSE
- For binary data: $\text{Bernoulli}(f_\theta(z))$ — reconstruction loss is binary cross-entropy

Both encoder and decoder are jointly trained end-to-end.

## Reparameterization Trick
---
We need to backpropagate through the sampling step $z \sim q_\phi(z \mid x)$. Sampling is non-differentiable, so we can't compute $\partial z / \partial \phi$ directly.

**Solution**: reparameterize the sampling as:
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Now $z$ is a deterministic function of $\phi$ (given $\epsilon$), and gradients flow through $\mu_\phi$ and $\sigma_\phi$ normally. $\epsilon$ is sampled from a fixed distribution with no parameters.

This trick is fundamental — it extends to other distributions (Gumbel-softmax for discrete, etc.).

## KL Term: Closed Form
---
For Gaussian encoder and Gaussian prior, the KL has an analytic form:
$$
D_\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)) = \frac{1}{2} \sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
$$

This doesn't require Monte Carlo estimation — the full ELBO gradient can be computed exactly (reconstruction term via reparameterization, KL term analytically).

## Training Objective
---
The final per-sample training loss (to minimize):
$$
\mathcal{L} = -\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\log p_\theta(x \mid \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)\right] + \frac{1}{2}\sum_j \left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)
$$

In practice, the expectation is estimated with a **single Monte Carlo sample** ($L=1$) — this is sufficient because each training step processes a different mini-batch.

## Latent Space Properties
---
The KL regularization forces the posterior to be close to $\mathcal{N}(0, I)$, which means:
- Latent codes for different inputs are distributed around the origin with unit variance
- The space is **smooth**: nearby points in latent space decode to similar outputs
- The space is **complete**: random samples from $p(z) = \mathcal{N}(0, I)$ decode to valid-looking outputs

This makes VAE latent spaces suitable for:
- **Interpolation**: $z = \lambda z_1 + (1-\lambda) z_2$ produces plausible intermediate images
- **Sampling**: draw $z \sim \mathcal{N}(0, I)$, decode to get a new sample
- **Compression**: use $\mu_\phi(x)$ as a compact representation (e.g., LDM's latent space)

## Posterior Collapse
---
A failure mode where the KL term drives $q_\phi(z \mid x) \to p(z)$. The encoder ignores the input; the decoder models $p(x)$ marginally. The model generates but doesn't encode — it's a pure generative model, not an autoencoder.

**Causes**: powerful decoders can model $p_\theta(x)$ without $z$; in that case, the KL penalty's most efficient solution is to set the posterior to the prior.

**Mitigations**:
- **KL annealing**: start with KL weight $w = 0$ (pure reconstruction), anneal to $w = 1$ over training
- **Free bits**: allow each latent dimension at least $\lambda$ nats of KL; only penalize dimensions that exceed $\lambda$
- **Weaker decoder**: reduce decoder capacity so it *must* use $z$

## $\beta$-VAE
---
$\beta$-VAE ([Higgins et al., 2017](https://openreview.net/forum?id=Sy2fchgwl)) upweights the KL term by $\beta > 1$:
$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[\log p_\theta(x\mid z)] - \beta\, D_\text{KL}(q_\phi(z\mid x) \| p(z))
$$

Higher $\beta$ encourages more **disentangled** representations — each latent dimension controls one independent factor of variation (e.g., rotation, scale, lighting). At the cost of reduced reconstruction quality.

## VAE vs. GAN vs. Diffusion
---

| | VAE | GAN | Diffusion |
|---|---|---|---|
| **Training** | Stable (ELBO) | Unstable (adversarial) | Stable (denoising) |
| **Sample quality** | Blurry | Sharp | Sharp |
| **Latent space** | Structured, smooth | Unstructured | Via inversion only |
| **Likelihood** | Tractable (via ELBO) | Intractable | Tractable |
| **Use in LDMs** | **Encoder/decoder** | Alternative encoder | Diffusion core |

VAEs produce blurrier samples than GANs or diffusion models because the MSE reconstruction loss averages over possible reconstructions. However, their structured latent spaces make them ideal as the **compression stage** in [[LDM|Latent Diffusion Models]].
