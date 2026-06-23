### [[dl_moc]]  
# **Generative Adversarial Network (GAN)**

**Tags**:: #generative #dl_theo
**Links**:: [[VAE]] · [[DDPM]] · [[FID]] · [[KL Divergence]] · [[Autoencoder]] · [[RAE]]


**GANs** ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)) frame generative modeling as a two-player zero-sum game: a **generator** $G$ tries to produce realistic samples, and a **discriminator** $D$ tries to tell real from fake. At equilibrium, $G$ perfectly recovers the data distribution and $D$ cannot distinguish them.

## The Minimax Game
---
Let $p_\text{data}$ be the real data distribution and $p_z$ be a simple noise prior (e.g., $\mathcal{N}(0, I)$). Define:
- **Generator**: $G_\theta: z \mapsto \hat{x}$, mapping noise to fake samples
- **Discriminator**: $D_\phi: x \mapsto [0, 1]$, estimating $P(\text{real} \mid x)$

The objective is:

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

$D$ maximizes this (getting better at distinguishing). $G$ minimizes it (getting better at fooling $D$).

In practice, minimizing $\log(1 - D(G(z)))$ early in training gives vanishing gradients when $D$ is confident. The **non-saturating loss** for $G$ is used instead:

$$
\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

Same Nash equilibrium, better gradient signal throughout training.

## Optimal Discriminator and Global Optimum
---
For a fixed $G$, the optimal discriminator is:

$$
D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)}
$$

Substituting $D^*$ into $V$, the generator's objective becomes minimizing the **Jensen-Shannon Divergence** between $p_\text{data}$ and $p_G$:

$$
V(D^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_\text{data} \| p_G)
$$

The global minimum is $\text{JSD} = 0 \Leftrightarrow p_G = p_\text{data}$, where $D^*(x) = \frac{1}{2}$ everywhere — the discriminator is at chance.

## Training Procedure
---
GANs are trained by alternating gradient updates. Each iteration:

1. Sample minibatch of real data $\{x^{(i)}\} \sim p_\text{data}$
2. Sample minibatch of noise $\{z^{(i)}\} \sim p_z$
3. **Update $D$**: ascend $\nabla_\phi \frac{1}{m}\sum_i [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$
4. **Update $G$**: descend $\nabla_\theta \frac{1}{m}\sum_i \log(1 - D(G(z^{(i)})))$ (or ascend the non-saturating form)

The discriminator is often updated $k > 1$ times per generator step to keep it informative.

## Training Instabilities
---
GAN training is notoriously unstable. The core issues:

### Mode Collapse
$G$ learns to produce a small subset of the data distribution — modes that reliably fool $D$. $D$ adapts, but $G$ then shifts to other modes rather than broadening coverage. The generator cycles through modes rather than covering all of $p_\text{data}$.

**Cause**: the minimax objective gives $G$ no incentive to cover all modes, only to fool $D$ on whatever it outputs.

### Vanishing Gradients
When $D$ becomes too confident (near-zero loss), $\nabla_\theta \mathcal{L}_G \approx 0$ — $G$ gets no training signal. This happens when $p_\text{data}$ and $p_G$ have disjoint support (common in high dimensions), which makes $\text{JSD} = \log 2$ everywhere with gradient 0.

### Non-Convergence
The alternating gradient updates don't minimize a single loss — they seek a Nash equilibrium, which gradient descent isn't guaranteed to find. Oscillations and divergence are common.

## Key Variants
---

| Variant | Key change | Problem addressed |
|---------|-----------|-------------------|
| **DCGAN** (Radford et al., 2015) | Convolutional G/D, batch norm, no pooling | Training stability for images |
| **WGAN** (Arjovsky et al., 2017) | Wasserstein distance, weight clipping | Vanishing gradients, mode collapse |
| **WGAN-GP** (Gulrajani et al., 2017) | Gradient penalty instead of clipping | More stable Lipschitz constraint |
| **StyleGAN** (Karras et al., 2019) | Mapping network, AdaIN style injection, progressive growing | High-resolution, disentangled control |
| **StyleGAN2** (Karras et al., 2020) | Removes progressive growing, redesigns normalization | Eliminates artifacts, better quality |

### WGAN: Wasserstein Distance
Replace the discriminator with a **critic** $f_w$ (unbounded output) and train it to estimate the Earth Mover's distance:

$$
W(p_\text{data}, p_G) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_\text{data}}[f(x)] - \mathbb{E}_{\hat{x} \sim p_G}[f(\hat{x})]
$$

The Lipschitz constraint ($\|f\|_L \leq 1$) is enforced via gradient penalty (WGAN-GP):

$$
\mathcal{L}_\text{GP} = \lambda \, \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} f(\hat{x})\|_2 - 1)^2]
$$

where $\hat{x}$ is interpolated between real and fake. This gives smooth, informative gradients even when $p_\text{data}$ and $p_G$ are far apart.

## GAN as a Perceptual Loss
---
Even after diffusion models surpassed GANs for generation quality, the **discriminator loss** remains widely used as a perceptual training signal for image reconstruction tasks. A discriminator trained to tell real images from reconstructions provides a patch-level realism signal that L1/L2 losses cannot:

$$
\mathcal{L}_\text{GAN}(\hat{x}, x) = \log D(x) + \log(1 - D(\hat{x}))
$$

This is used in:
- **[[RAE]] decoder training**: combined with LPIPS and L1 for perceptually sharp reconstructions
- **VQ-VAE-2 / VQGAN**: adversarial loss on the decoder produces sharper tokens for autoregressive models
- **Pix2Pix, image inpainting**: where reconstruction quality matters more than diversity

The discriminator here is a "perceptual critic" — it's not the generator being trained, it's providing loss signal to train an encoder/decoder.

## GAN vs. Other Generative Models
---

| | GAN | VAE | Diffusion |
|---|---|---|---|
| **Training** | Unstable, adversarial | Stable, ELBO | Stable, denoising |
| **Sample quality** | Sharp (historically best) | Blurry | Sharp (state-of-the-art) |
| **Mode coverage** | Poor (mode collapse) | Good | Good |
| **Likelihood** | Intractable | Tractable (ELBO lower bound) | Tractable |
| **Sampling speed** | Fast (single forward pass) | Fast | Slow (many denoising steps) |
| **Latent space** | Unstructured | Smooth, regularized | Via inversion only |
| **Current use** | Perceptual loss component | Compression (LDM encoder) | Primary generation method |

## Relationship to Other Notes
---
- [[VAE]] — both are generative models; GANs produce sharper samples but lack a structured latent space and tractable likelihood; VAE comparison in [[VAE#VAE vs. GAN vs. Diffusion]]
- [[DDPM]] — diffusion models supplanted GANs as the dominant generative paradigm; GANs are now used mainly as perceptual losses within diffusion pipelines
- [[RAE]] — RAE's decoder training uses a GAN loss for perceptual sharpness; the GAN here is a perceptual critic, not a standalone generative model
- [[KL Divergence]] — the original GAN objective minimizes JSD, not KL; WGAN replaces this with Wasserstein distance
- [[FID]] — the standard metric for comparing generative models, introduced partly to better capture the failure modes of GANs (mode collapse doesn't show up in per-sample likelihood)
