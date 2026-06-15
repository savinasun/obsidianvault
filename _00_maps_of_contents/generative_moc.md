### [[dl_moc]]
# **Generative Modeling**

**Tags**:: #generative
**Links**:: [[math_moc]] · [[learning_methods_moc]] · [[sequence_llms_moc]] · [[vision_ssl_moc]] · [[rl_embodied_moc]]

Learning to synthesize data: variational autoencoders, score-based models, and diffusion processes — the research direction that produced Stable Diffusion, DALL-E, and modern image/video generation.

## Variational Methods
---
- **[[VAE]]** — Variational Autoencoder: encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$, reparameterization trick $z = \mu + \sigma\epsilon$; ELBO objective (reconstruction + KL penalty); posterior collapse; foundation for latent diffusion
- **[[ELBO]]** — Evidence Lower Bound derivation: $\log p(x) \ge \mathbb{E}_q[\log p(x|z)] - D_\text{KL}(q\|p)$; the variational inference objective; see also [[math_moc]] for the mathematical object

## Score-Based & Diffusion Models
---
- **[[Score-Based Generative Modeling Methods]]** — Score functions $\nabla_x \log p(x)$; Stein score; Langevin MCMC sampling; the theoretical unification of diffusion and score matching
- **[[DDPM]]** — Denoising Diffusion Probabilistic Models: forward noising process $q(x_t|x_{t-1})$, learned reverse denoising $p_\theta(x_{t-1}|x_t)$; DDIM deterministic sampling; cosine noise schedule
- **[[LDM]]** — Latent Diffusion Models: run diffusion in a VAE latent space instead of pixel space; text conditioning via cross-attention; the architecture behind Stable Diffusion

## Architectures for Generation
---
- **[[U-Net]]** — Encoder-decoder with skip connections; the standard denoiser backbone for DDPM and LDM; time-step conditioning via AdaGN; see also [[vision_ssl_moc]] for the segmentation origin

## Mathematical Prerequisites
---
The generative modeling stack draws heavily from [[math_moc]]:
- [[KL Divergence]] — the regularization term in ELBO and VAE training
- [[ELBO]] — the variational lower bound that makes VAE training tractable
- [[MCMC]] — Langevin dynamics connects score functions to sampling; Metropolis-Hastings as a baseline sampler
- [[Entropy]] — maximum entropy perspective on diffusion

## Dataview
---
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "generative")
SORT file.name ASC
```
