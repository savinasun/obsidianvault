### [[dl_moc]]  
# **Evidence Lower Bound (ELBO)**

**Tags**:: #dl_theo #math
**Links**:: [[KL Divergence]] · [[VAE]] · [[Score-Based Generative Modeling Methods]] · [[MCMC]]


The **ELBO** **(Evidence Lower Bound)** is the **==fundamental objective for training latent variable models==** when the **true posterior** $p_\theta(z \mid x)$ is **intractable**. It provides a **tractable lower bound** on the log-likelihood $\log p_\theta(x)$.

## Derivation
---
Let $X$ and $Z$ be random variables jointly distributed with $p_\theta(X, Z)$. For a sample $x$ drawn from the marginal distribution $p_{\theta}(X)$, we call $\log p_{\theta}(x)$ the **evidence** of $x$.
$$
\begin{aligned}
\log p_\theta(x) = \log \int p_\theta(x, z)\, dz
\end{aligned}
$$
For any distribution $q_\phi(z \mid x)$ (the **variational posterior** or **encoder**), multiply the above equation and divide by $q_\phi(z \mid x)$:
$$
\begin{aligned}
&= \log \int \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \cdot q_\phi(z \mid x) dz \\
&= \log \mathbb{E}_{z \sim q_\phi(\cdot \mid x)}\left[\frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]
\end{aligned}
$$

Apply **Jensen's inequality** (log is concave, so $\log\mathbb{E}[\cdot] \ge \mathbb{E}[\log\cdot]$):
$$
\ge \mathbb{E}_{z \sim q_\phi(\cdot \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \eqqcolon \mathcal{L}(\phi, \theta;\, x)
$$
$\mathcal{L}(\phi, \theta;\, x)$ is called the **ELBO**, i.e.
$$
\log p_{\theta}(x) \geq \mathcal{L}(\phi, \theta; x).
$$
The gap between $\log p_\theta(x)$ and $\mathcal{L}$ is exactly the [[KL Divergence|KL divergence]] of the variational posterior from the true posterior:
$$
\log p_\theta(x) = \mathcal{L}(\phi, \theta;\, x) + D_\text{KL}(q_\phi(z\mid x) \| p_\theta(z\mid x))
$$

Since $D_\text{KL} \ge 0$, we have $\mathcal{L} \le \log p_\theta(x)$ — hence "lower bound."

## Decomposed Form
---
The ELBO decomposes into a **reconstruction term** and a **regularization term**:
$$
\mathcal{L}(\phi, \theta;\, x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_\text{KL}(q_\phi(z \mid x) \| p(z))}_{\text{regularization}}
$$

| Term | Interpretation |
|------|----------------|
| $\mathbb{E}[\log p_\theta(x \mid z)]$ | How well the decoder reconstructs $x$ from latent $z$ |
| $D_\text{KL}(q_\phi \| p)$ | How far the variational posterior deviates from the prior $p(z) = \mathcal{N}(0, I)$ |

Maximizing the ELBO simultaneously:
1. **Improves reconstruction**: the decoder $p_\theta(x \mid z)$ must explain the data well
2. **Regularizes the latent space**: the encoder $q_\phi(z \mid x)$ must stay close to the prior

## Connection to VAE
---
In a [[VAE]]:
- **Encoder**: $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$
- **Prior**: $p(z) = \mathcal{N}(0, I)$
- **Decoder**: $p_\theta(x \mid z)$ — Gaussian or Bernoulli depending on data type

The KL term has a closed form for Gaussian posteriors:
$$
D_\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\sum_j \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
$$

The VAE is trained end-to-end by maximizing the ELBO using the **reparameterization trick** to backpropagate through the sampling step: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.

## Posterior Collapse
---
A failure mode in VAE training: the KL term dominates, pushing $q_\phi(z \mid x) \to p(z)$. The encoder learns to ignore the input, and the decoder learns to model $p(x)$ marginally (without using $z$ at all). The latent space becomes uninformative.

**Mitigations**: KL annealing (gradually increase KL weight during training), free bits (allow a minimum KL per dimension), or using stronger decoders.

## ELBO in Diffusion Models
---
DDPM training can also be derived as an ELBO on the data log-likelihood. The variational posterior $q(x_{1:T} \mid x_0)$ is fixed (the forward noising process), and the learned reverse $p_\theta(x_{0:T})$ plays the role of the generative model. The simplified MSE loss in [[DDPM]] corresponds to one term of this ELBO.

## Why ELBO Matters
---
The ELBO is the core of **variational inference** — a general technique for approximating intractable posteriors that underlies VAEs, VQ-VAE, LDMs, and many probabilistic models. Whenever exact inference is intractable (which is most of the time), we optimize the ELBO instead.
