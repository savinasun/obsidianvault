### [[dl_moc]]  
# **Regularized Autoencoder (RAE)**

**Tags**:: #generative #dl_theo
**Links**:: [[Autoencoder]] · [[VAE]] · [[VQ-VAE]] · [[ELBO]] · [[KL Divergence]] · [[LDM]] · [[FID]] · [[Manifold Hypothesis]]


**RAE** ([Ghosh et al., 2019](https://arxiv.org/abs/1903.12436)) asks: do we actually need a stochastic encoder to get a well-structured latent space? The answer is no. RAE replaces the [[VAE]]'s variational encoder and KL term with a **deterministic encoder** plus explicit **regularization on the latent codes**, recovering comparably structured latent spaces without the pathologies of the ELBO objective.

## Motivation: Problems with the VAE
---
The [[VAE]] optimizes the [[ELBO]]:
$$
\mathcal{L}_\text{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_\text{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}}
$$

This objective introduces two well-known failure modes:

1. **Posterior collapse**: when the decoder is powerful, the KL term drives $q_\phi(z|x) \to p(z)$, causing the encoder to be ignored entirely. The model generates but doesn't encode.
2. **Blurry samples**: the stochastic encoder averages over multiple plausible reconstructions — the decoder must explain away injected noise, which pushes it toward blurry mean predictions.

Both stem from tying the *regularization mechanism* (KL) to the *probabilistic encoder*. RAE decouples them.

## RAE Objective
---
RAE uses a **deterministic encoder** $z = f_\phi(x)$ (no sampling, no KL) and adds an explicit penalty directly on the latent codes:

$$
\mathcal{L}_\text{RAE} = \underbrace{\|x - g_\theta(f_\phi(x))\|^2}_{\text{reconstruction}} + \underbrace{\lambda \, \|f_\phi(x)\|^2}_{\text{L2 latent regularization}}
$$

where $g_\theta$ is the decoder. The L2 penalty prevents the encoder from collapsing to a degenerate (e.g., unbounded) latent space, replacing the role of the KL term without the stochasticity.

Two regularization variants are proposed:

| Variant | Mechanism | Effect |
|---------|-----------|--------|
| **RAE-L2** | $\lambda\|z\|^2$ on latent codes | Keeps codes bounded near origin |
| **RAE-SN** | Spectral normalization on encoder weights | Bounds the Lipschitz constant of $f_\phi$; more implicit regularization |

Both variants avoid posterior collapse by design — there is no KL term to drive $q_\phi \to p$.

## Ex-Post Density Estimation
---
A standard autoencoder doesn't define a generative model. To sample, RAE uses **ex-post density estimation**:

1. After training, compute latent codes for all training data: $\{z_i = f_\phi(x_i)\}$
2. Fit a simple density model $p_\lambda(z)$ to these codes (Gaussian, GMM, or a small normalizing flow)
3. At generation: sample $z \sim p_\lambda(z)$, then decode $\hat x = g_\theta(z)$

This separates learning the **latent geometry** (encoder/decoder training) from learning the **prior over that geometry** (density estimation). In contrast, the VAE tries to do both simultaneously via the KL term, which creates the tension that causes posterior collapse.

## Why RAE Latent Spaces Are Still Structured
---
At first glance, removing the KL term should break the smooth, continuous latent space that makes VAEs useful. In practice, the L2 penalty + reconstruction loss jointly enforce structure:

- **Bounded**: L2 prevents codes from drifting to infinity
- **Dense**: reconstruction pressure keeps codes informative — the decoder must use $z$ to reconstruct $x$, so nearby $x$'s map to nearby $z$'s
- **Smooth**: unlike a standard AE with no regularization, the penalty prevents the encoder from exploiting high-frequency directions in the latent space

The [[Manifold Hypothesis]] perspective: the data manifold is still recovered; RAE just regularizes the coordinate system (latent codes) rather than regularizing the posterior family.

## Comparison: RAE vs. VAE
---

| | VAE | RAE |
|---|---|---|
| **Encoder** | Stochastic $q_\phi(z\|x) = \mathcal{N}(\mu, \sigma^2)$ | Deterministic $z = f_\phi(x)$ |
| **Regularization** | KL divergence (implicit, via ELBO) | Explicit L2 / spectral norm |
| **Reparameterization** | Required | Not needed |
| **Posterior collapse** | Common failure mode | Impossible by construction |
| **Sample blurriness** | High (stochastic encoder averages modes) | Lower (deterministic encoder) |
| **Prior at generation** | $p(z) = \mathcal{N}(0,I)$ (assumed) | Ex-post density model (learned) |
| **ELBO** | Optimized directly | Not used |
| **Sample quality (FID)** | Worse than diffusion | Competitive with VAE; better sharpness |

## Relationship to Other Notes
---
- [[VAE]] — RAE is a direct drop-in alternative; addresses posterior collapse and blurriness while preserving a structured latent space
- [[ELBO]] — RAE deliberately avoids the ELBO; the KL–reconstruction tradeoff is the root cause of VAE failures
- [[KL Divergence]] — replaced by explicit L2 regularization; the two serve the same structural role but RAE's version can't collapse
- [[LDM]] — LDMs use a VAE encoder; RAE could substitute as the compression stage with potentially sharper reconstructions
- [[FID]] — the standard metric used to compare RAE vs. VAE generation quality
- [[Manifold Hypothesis]] — motivates why deterministic encoders with mild regularization still recover the data manifold
