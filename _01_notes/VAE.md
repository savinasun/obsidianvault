### [[dl_moc]]  
# **Variational Autoencoder (VAE)**

**Tags**:: #generative #dl_theo
**Links**:: [[Autoencoder]] · [[VQ-VAE]] · [[ELBO]] · [[KL Divergence]] · [[LDM]] · [[RAE]] · [[Flow Matching]] · [[Score-Based Generative Modeling Methods]] · [[Manifold Hypothesis]]


A **VAE** ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)) is a latent-variable generative model that encodes data into a structured latent space and decodes back. Unlike standard [[Autoencoder|autoencoders]], the latent space is **probabilistic** and **regularized** toward a prior — which makes it both a generative model in its own right and the standard compression stage for latent generative modeling ([[LDM]], SD3, Movie Gen). This note incorporates the treatment in §6.2.2 and Appendix D of the MIT 6.S184 lecture notes ([Holderrieth & Erives](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)).

**Notation** (6.S184): data $x \in \mathbb{R}^d$, latent $z \in \mathbb{R}^k$ ($k \ll d$); encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$; prior $p_\text{prior} = \mathcal{N}(0, I_k)$; KL weight $\beta \ge 0$. ⚠️ *Symbol clash*: here $z$ is the **latent code** — in [[Flow Matching]]/[[Score Matching]] notes $z$ is the data point.

## Generative Model
---
The VAE models data $x$ as generated from a latent variable $z$:
$$
p_\theta(x) = \int p_\theta(x \mid z)\, p_\text{prior}(z)\, dz
$$
- **Prior**: $z \sim \mathcal{N}(0, I_k)$
- **Decoder** (generative model): $p_\theta(x \mid z)$, a neural network
- **True posterior**: $p_\theta(z \mid x) = p_\theta(x\mid z)\,p_\text{prior}(z) / p_\theta(x)$ — **intractable** (requires $p_\theta(x)$)

The marginal $p_\theta(x)$ integrates over all latent codes — intractable in high dimension, so $\log p_\theta(x)$ cannot be maximized directly.

## Variational Inference: the ELBO
---
Introduce a **variational posterior** $q_\phi(z \mid x)$ (the **encoder**) to approximate the true posterior. By the [[ELBO]] derivation:
$$
\log p_\theta(x) \ge \text{ELBO}(x; \phi, \theta) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_\text{KL}(q_\phi(z\mid x) \,\|\, p_\text{prior})}_{\text{regularization}} .
$$
Maximizing the ELBO (i) makes the decoder reconstruct $x$ well from encoder samples and (ii) keeps the posterior close to the prior. The "variational" in VAE (6.S184 Rem. 43): $q_\phi(\cdot|x)$ is a tractable **variational approximation** of the intractable posterior $p_\theta(z|x)$ — we cannot just set them equal because $p_\theta(z|x)$ requires the unknown $p_\theta(x)$.

## The β-VAE Loss (6.S184 formulation)
---
**Probabilistic encoder/decoder.** The standard Gaussian choice — with $\sigma_\phi = \sigma_\theta = 0$ recovering the deterministic [[Autoencoder]]:

$$
q_\phi(z|x) = \mathcal{N}\big(z;\, \mu_\phi(x),\, \mathrm{diag}(\sigma_\phi^2(x))\big), \qquad
p_\theta(x|z) = \mathcal{N}\big(x;\, \mu_\theta(z),\, \sigma_\theta^2(z) I_d\big).
$$

For binary data the decoder is Bernoulli and reconstruction becomes cross-entropy.

**Two losses.** Reconstruction = negative log-likelihood of $x$ under encode-then-decode; prior enforcement = [[KL Divergence]] of the encoder to the prior:

$$
\mathcal{L}_\text{VAE-Recon} = -\mathbb{E}_{x \sim p_\text{data},\, z \sim q_\phi(\cdot|x)}\big[\log p_\theta(x|z)\big],
\qquad
\mathcal{L}_\text{VAE-Prior} = \mathbb{E}_{x \sim p_\text{data}}\big[D_\text{KL}\big(q_\phi(\cdot|x)\,\|\, p_\text{prior}\big)\big].
$$

**Closed-form KL.** For diagonal Gaussians (6.S184 Ex. 31), $D_\text{KL}(q\|p) = \tfrac{1}{2}\big(K(\sigma_q^2/\sigma_p^2) + \|\mu_q - \mu_p\|^2/\sigma_p^2\big)$ with $K(\alpha) = \sum_i (\alpha_i - \log\alpha_i - 1)$, minimized at $\alpha = 1$; against $\mathcal{N}(0, I_k)$ this is $\tfrac{1}{2}\sum_j (\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$ — no Monte Carlo needed for the KL term.

**Four-term loss.** Combining with weight $\beta$, $\mathcal{L}_\text{VAE} = \mathcal{L}_\text{VAE-Recon} + \beta\,\mathcal{L}_\text{VAE-Prior}$ reads (6.S184 Eq. 83):

$$
\mathcal{L}_\text{VAE} = \mathbb{E}_{x,\, z \sim q_\phi}\Big[
\underbrace{\tfrac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2}_{\text{recon. error}}
+ \underbrace{\tfrac{d}{2}\log \sigma_\theta^2(z)}_{\text{decoder confidence}}
+ \underbrace{\tfrac{\beta}{2}K(\sigma_\phi^2(x))}_{\text{latent variance} \to 1}
+ \underbrace{\tfrac{\beta}{2}\|\mu_\phi(x)\|^2}_{\text{latent mean} \to 0}
\Big].
$$

## Training: the Reparameterization Trick
---
The expectation is over $q_\phi$, which depends on $\phi$ — and sampling is non-differentiable. **Reparameterize**:
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_k),
$$
so all randomness comes from a parameter-free $\epsilon$ and gradients flow through $\mu_\phi, \sigma_\phi$; a **single Monte Carlo sample** per step suffices. (The trick generalizes — e.g. Gumbel-softmax for discrete latents.) Training loop (6.S184 Alg. 6, fixed decoder variance):

```python
# beta-VAE training step
x = sample_data()
mu, log_var = encoder(x)                                   # q_phi(z|x)
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)   # reparameterization
recon = ((x - decoder(z))**2).sum(-1) / (2 * sigma2)
kl    = 0.5 * (mu**2 + log_var.exp() - log_var - 1).sum(-1)
loss  = (recon + beta * kl).mean()
```

## Theory: Joint KL, ELBO, and the Amortization Gap
---
**One-line identity** (6.S184 App. D, $\beta = 1$): with $q_\phi(x,z) = p_\text{data}(x)\,q_\phi(z|x)$ and $p_\theta(x,z) = p_\text{prior}(z)\,p_\theta(x|z)$,

$$
\mathcal{L}_\text{VAE}(\phi, \theta) = D_\text{KL}\big(q_\phi(x, z)\,\|\,p_\theta(x, z)\big) + \text{const}
$$

— the VAE minimizes a KL **in the joint data–latent space**; equivalently $\mathcal{L}_\text{VAE} = -\mathbb{E}_x[\text{ELBO}(x)] + \text{const}$. The KL chain rule $D_\text{KL}(q(z,x)\|p(z,x)) = D_\text{KL}(q(x)\|p(x)) + \mathbb{E}[D_\text{KL}(q(z|x)\|p(z|x))]$ (data-processing inequality) then gives two upper bounds:

$$
\mathcal{L}_\text{VAE} \ge D_\text{KL}\big(p_\text{data}\,\|\,p_\theta(x)\big) + \text{const},
\qquad
\mathcal{L}_\text{VAE} \ge D_\text{KL}\big(q_\phi(z)\,\|\,p_\text{prior}\big) + \text{const},
$$

so training simultaneously (i) makes the VAE a **generative model in its own right** ($z \sim p_\text{prior}$, $x \sim p_\theta(\cdot|z)$) and (ii) pushes the **aggregate latent** $q_\phi(z)$ toward the prior.

**Amortization gap — why not stop at VAEs?** The bounds are tight iff $q_\phi(z|x) = p_\theta(z|x)$; in practice a gap remains, so $q_\phi(z) \ne p_\text{prior}$ — and the decoder was trained to reconstruct from $q_\phi(z)$, so sampling from $p_\text{prior}$ at inference is out-of-distribution. The fix *is* latent generative modeling: train a [[Flow Matching]]/diffusion model to sample $q_\phi(z)$ exactly, and let the (weaker, convolutional) decoder do less generative work — see [[Autoencoder]] (Latent Generative Modeling) and [[LDM]].

## Latent Space Properties
---
The KL regularization keeps posteriors near $\mathcal{N}(0, I_k)$, so the latent space is:
- **Smooth**: nearby latents decode to similar outputs → **interpolation** $z = \lambda z_1 + (1-\lambda) z_2$ yields plausible intermediates
- **Complete**: random $z \sim p_\text{prior}$ decodes to valid-looking outputs → **sampling**
- **Compact**: $\mu_\phi(x)$ is a usable representation → **compression** (LDM's latent space)

## Posterior Collapse and Practical Remarks
---
**Posterior collapse**: the KL term drives $q_\phi(z|x) \to p_\text{prior}$, the encoder ignores $x$, and the model generates without encoding. Cause: a decoder powerful enough to model $p_\theta(x)$ without $z$ makes collapsing the posterior the cheapest way to pay the KL penalty.

Mitigations and practice (6.S184 + classics):
- **KL warm-up / annealing**: start at $\beta = 0$, ramp up; modern compression autoencoders keep $\beta \ll 1$ throughout.
- **Free bits**: only penalize per-dimension KL above a floor $\lambda$.
- **Weaker decoder**: force the model to use $z$.
- **Fixed decoder variance**: learning $\sigma_\theta^2$ is numerically delicate; fixing it makes reconstruction ∝ MSE.
- **Perceptual losses**: pixel MSE yields blurry outputs; feature-space losses restore sharpness.
- **Adversarial hybrid** (VAE-GAN / VQGAN): a discriminator on decodings sharpens further, at the cost of stability.

## $\beta$-VAE: Disentanglement vs. Compression
---
The same knob $\beta$ serves opposite goals. $\beta$-VAE ([Higgins et al., 2017](https://openreview.net/forum?id=Sy2fchgwl)) uses $\beta > 1$ to encourage **disentangled** latents (each dimension one factor of variation) at the cost of reconstruction; compression autoencoders for [[LDM]]-style pipelines use $\beta \ll 1$ to prioritize reconstruction, leaving distribution-matching to the latent generative model.

## VAE vs. GAN vs. Diffusion
---

|                        | VAE                  | RAE                 | GAN                    | Diffusion          |
| ---------------------- | -------------------- | ------------------- | ---------------------- | ------------------ |
| **Training**           | Stable (ELBO)        | Stable (L2 reg)     | Unstable (adversarial) | Stable (denoising) |
| **Sample quality**     | Blurry               | Less blurry         | Sharp                  | Sharp              |
| **Latent space**       | Structured, smooth   | Structured, smooth  | Unstructured           | Via inversion only |
| **Posterior collapse** | Common               | Impossible          | N/A                    | N/A                |
| **Likelihood**         | Tractable (via ELBO) | Ex-post density     | Intractable            | Tractable          |
| **Use in LDMs**        | **Encoder/decoder**  | Drop-in alternative | Alternative encoder    | Diffusion core     |

VAEs produce blurrier samples than GANs or diffusion models because MSE reconstruction averages over possible reconstructions. [[RAE]] replaces the stochastic encoder and KL with a deterministic encoder and explicit regularization, avoiding blurriness and collapse. VAE latent spaces nonetheless remain the standard compression stage in [[LDM|Latent Diffusion Models]] and flow-matching generators (SD3, Movie Gen — see [[Autoencoder]]).
