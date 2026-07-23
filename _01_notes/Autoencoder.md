### [[dl_moc]]  
# **Autoencoder**

**Tags**:: #generative #learning
**Links**:: [[VAE]] · [[VQ-VAE]] · [[RAE]] · [[MAE]] · [[LDM]] · [[Flow Matching]] · [[Score Matching]] · [[DiT]] · [[ELBO]] · [[KL Divergence]] · [[Manifold Hypothesis]] · [[Contrastive Learning]] · [[t-SNE]] · [[UMAP]] · [[FID]]

![[autoencoder.png|500]]

An **autoencoder** compresses data into a low-dimensional **latent code** and reconstructs it back — learning a compact, lossy representation **without supervision**. Over four decades it evolved from a dimensionality-reduction tool into the **compression backbone of modern generative models**: nearly all state-of-the-art image/video generators train a flow or diffusion model *inside* an autoencoder's latent space.

This note follows §6.2 and Appendix D of the MIT 6.S184 lecture notes ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf); [course site](https://diffusion.csail.mit.edu/)), which motivate autoencoders as the answer to: *how do we model high-dimensional images within a reasonable compute budget?* (A $1024^2$ RGB image has $d \approx 3\times 10^6$, and a velocity field's output must be just as large as its input.) The variant history and taxonomy go beyond the course.

**Notation** (6.S184 §6.2): data $x \in \mathbb{R}^d$, latent $z \in \mathbb{R}^k$ with $k \ll d$; encoder $\mu_\phi$ (or stochastic $q_\phi(z|x)$), decoder $\mu_\theta$ (or $p_\theta(x|z)$); latent prior $p_\text{prior} = \mathcal{N}(0, I_k)$; weight $\beta \ge 0$. ⚠️ *Symbol clash*: here $z$ is the **latent code** — in the [[Flow Matching]]/[[Score Matching]] notes $z$ is the **data point**.

## Standard Autoencoder
---
**Input data**: $x \in \mathbb{R}^d$, 
**Latent variable**: $z \in \mathbb{R}^k$ with $k \ll d$; 
**Encoder**: $\mu_\phi : \mathbb{R}^d \to \mathbb{R}^k$ or stochastic $q_\phi(z|x)$
**Decoder**: $\mu_\theta : \mathbb{R}^k \to \mathbb{R}^d$ or $p_\theta(x|z)$ 
**Latent prior**: $p_\text{prior} = \mathcal{N}(0, I_k)$, weight $\beta \ge 0$
**Autoencoder.** An encoder–decoder pair $(\mu_{\phi}, \mu_{\theta})$ with $k \ll d$, trained with the **reconstruction loss**
$$
\mathcal{L}_\text{Recon}(\phi, \theta) = \mathbb{E}_{x \sim p_\text{data}}\big\|\mu_\theta(\mu_\phi(x)) - x\big\|^2 .
$$

The bottleneck forbids the trivial identity map — the model must keep only the most informative structure. By the [[Manifold Hypothesis]], real data lies near a low-dimensional manifold; the autoencoder learns to parameterize it.

**Why reconstruction alone is not enough**. The eventual goal is to train a generative model on the **latent distribution** $p_\text{latent}(z)$, $z = \mu_\phi(x)$, $x \sim p_\text{data}$. Pure reconstruction gives *no control over $p_\text{latent}$* — compression may have warped the data into an arbitrarily hard-to-learn latent distribution. We need to **regularize the latent distribution toward something "nice"** — which leads to the VAE.

## Variational Autoencoder
---
**Probabilistic encoder/decoder.** Relax determinism ([[VAE]]; Kingma & Welling, 2013):

$$
q_\phi(z|x) = \mathcal{N}\big(z;\, \mu_\phi(x),\, \mathrm{diag}(\sigma_\phi^2(x))\big), \qquad
p_\theta(x|z) = \mathcal{N}\big(x;\, \mu_\theta(z),\, \sigma_\theta^2(z) I_d\big),
$$

with $\sigma_\phi = \sigma_\theta = 0$ recovering the standard autoencoder. Encoding/decoding become sampling.

**Two losses.** Reconstruction = negative log-likelihood of $x$ under encode-then-decode; prior enforcement = [[KL Divergence]] of the encoder to $p_\text{prior} = \mathcal{N}(0, I_k)$:

$$
\mathcal{L}_\text{VAE-Recon} = -\mathbb{E}_{x \sim p_\text{data},\, z \sim q_\phi(\cdot|x)}\big[\log p_\theta(x|z)\big],
\qquad
\mathcal{L}_\text{VAE-Prior} = \mathbb{E}_{x \sim p_\text{data}}\big[D_\text{KL}\big(q_\phi(\cdot|x)\,\|\, p_\text{prior}\big)\big].
$$

With the closed-form Gaussian KL, $D_\text{KL}(q\|p) = \tfrac{1}{2}\big(K(\sigma_q^2/\sigma_p^2) + \|\mu_q - \mu_p\|^2/\sigma_p^2\big)$ where $K(\alpha) = \sum_i (\alpha_i - \log\alpha_i - 1)$ (minimized at $\alpha = 1$), the **β-VAE loss** $\mathcal{L}_\text{VAE} = \mathcal{L}_\text{VAE-Recon} + \beta\, \mathcal{L}_\text{VAE-Prior}$ has four intuitive terms (6.S184 Eq. 83):

$$
\mathcal{L}_\text{VAE} = \mathbb{E}_{x,\, z \sim q_\phi}\Big[
\underbrace{\tfrac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2}_{\text{recon. error}}
+ \underbrace{\tfrac{d}{2}\log \sigma_\theta^2(z)}_{\text{decoder confidence}}
+ \underbrace{\tfrac{\beta}{2}K(\sigma_\phi^2(x))}_{\text{latent variance} \to 1}
+ \underbrace{\tfrac{\beta}{2}\|\mu_\phi(x)\|^2}_{\text{latent mean} \to 0}
\Big].
$$

**Reparameterization trick.** The expectation is over $q_\phi$, which depends on $\phi$; rewrite $z = \mu_\phi(x) + \sigma_\phi(x)\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I_k)$ so all randomness is $\phi$-independent and SGD applies. Training loop (6.S184 Alg. 6, fixed decoder variance):

```python
# beta-VAE training step
x = sample_data()
mu, log_var = encoder(x)                                   # q_phi(z|x)
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)   # reparameterization
recon = ((x - decoder(z))**2).sum(-1) / (2 * sigma2)
kl    = 0.5 * (mu**2 + log_var.exp() - log_var - 1).sum(-1)
loss  = (recon + beta * kl).mean()
```

**Practical remarks** (6.S184): (1) large $\beta$ hurts reconstructions and can cause **posterior collapse** ($q_\phi(z|x) \approx \mathcal{N}(0,I_k)$ ignoring $x$) — use KL warm-up, and modern autoencoders use $\beta \ll 1$; (2) learned decoder variance is numerically delicate — most implementations fix $\sigma_\theta^2$ constant, making reconstruction ∝ MSE; (3) pixel MSE gives blurry outputs — add **perceptual losses**; (4) optionally add an **adversarial loss** (VAE-GAN / VQGAN style) for sharpness at the cost of stability.

## Theory: Joint KL, ELBO, and the Amortization Gap
---
**One-line identity** (6.S184 App. D, $\beta = 1$): with $q_\phi(x,z) = p_\text{data}(x)\,q_\phi(z|x)$ and $p_\theta(x,z) = p_\text{prior}(z)\,p_\theta(x|z)$,

$$
\mathcal{L}_\text{VAE}(\phi, \theta) = D_\text{KL}\big(q_\phi(x, z)\,\|\,p_\theta(x, z)\big) + \text{const}
$$

— the VAE minimizes a KL **in the joint data–latent space**. The KL chain rule $D_\text{KL}(q(z,x)\|p(z,x)) = D_\text{KL}(q(x)\|p(x)) + \mathbb{E}[D_\text{KL}(q(z|x)\|p(z|x))]$ (data-processing inequality) then gives two upper bounds:

$$
\mathcal{L}_\text{VAE} \ge D_\text{KL}\big(p_\text{data}\,\|\,p_\theta(x)\big) + \text{const},
\qquad
\mathcal{L}_\text{VAE} \ge D_\text{KL}\big(q_\phi(z)\,\|\,p_\text{prior}\big) + \text{const},
$$

so training simultaneously (i) makes the VAE a **generative model in its own right** ($z \sim p_\text{prior}$, $x \sim p_\theta(\cdot|z)$) and (ii) pushes the **aggregate latent** $q_\phi(z)$ toward the prior.

**ELBO view.** Rearranging via Bayes gives $\text{ELBO}(x) = \mathbb{E}_{z\sim q_\phi(\cdot|x)}\big[\log\tfrac{p_\theta(x|z)\,p_\text{prior}(z)}{q_\phi(z|x)}\big] \le \log p_\theta(x)$, and $\mathcal{L}_\text{VAE} = -\mathbb{E}_{x}[\text{ELBO}(x)] + \text{const}$ — the VAE maximizes the expected [[ELBO]]. The "variational" in VAE: $q_\phi(\cdot|x)$ is a tractable **variational approximation** of the intractable posterior $p_\theta(z|x)$ (6.S184 Rem. 43).

**Amortization gap — why not stop at VAEs?** The bounds are tight iff $q_\phi(z|x) = p_\theta(z|x)$; in practice a gap remains, so $q_\phi(z) \ne p_\text{prior}$, and the decoder was trained to reconstruct from $q_\phi(z)$ — sampling from $p_\text{prior}$ at inference is out-of-distribution. The fix *is* latent generative modeling: train a flow/diffusion model to sample $q_\phi(z)$ exactly, and let the (weaker, convolutional) decoder do less generative work.

## Latent Generative Modeling
---
**The recipe** (6.S184 Rem. 32; = [[LDM]]): train the autoencoder first and freeze it; train a [[Flow Matching]]/diffusion model on encoded latents $z \sim q_\phi(\cdot|x)$; at inference, sample the latent model and decode with the **mean** $\mu_\theta(z)$ (avoids noise artifacts). The autoencoder filters out perceptually meaningless high-frequency detail so the generative model can focus on semantics — but total quality now also depends on the autoencoder.

**Reconstruction vs. generation trade-off** (App. D). Compare the *reconstruction sampler* (encode data, decode) and *generative sampler* (sample latent model, decode) by their [[FID]]s: **rFID** measures information loss, **gFID** end-to-end generation. They are in tension: lower rFID (less compression) ⟹ harder latent distribution ⟹ higher gFID, and vice versa. The optimal **division of labor** sits at the knee of the rate–distortion Pareto frontier: compress as much as possible *without* meaningful distortion (Intuition 44).

**Case studies** (6.S184 §6.3): **Stable Diffusion 3** — CFM objective in a pretrained autoencoder's latent space, MM-DiT backbone ([[DiT]] with joint text–image attention over CLIP + T5-XXL embeddings), CFG scale 2–5, 50 Euler steps, 8B params. **Movie Gen Video** — a **temporal autoencoder (TAE)** compressing $8\times$ along time, height, and width (temporal tiling for long videos), cond-OT flow matching, DiT backbone with UL2/ByT5/MetaCLIP text embeddings.

## History and Variants
---
- **1986 — Original autoencoder** (Rumelhart et al.): unsupervised hidden representations through a narrow bottleneck — nonlinear PCA; limited to shallow nets by vanishing gradients.
- **2006 — Deep autoencoders** ([Hinton & Salakhutdinov](https://doi.org/10.1126/science.1127647)): deep AEs beat PCA, but only with greedy layer-wise RBM pre-training — a landmark of the first deep-learning wave and the era's unsupervised pre-training recipe.
- **2008–10 — Denoising autoencoder** ([Vincent et al., 2010](https://www.jmlr.org/papers/v11/vincent10a.html)): reconstruct clean $x$ from corrupted $\tilde x$ (noise, masking) — forces robust features. With Gaussian corruption the DAE implicitly learns the **score of the noise-smoothed data distribution** — the precursor of [[Score Matching]] and [[DDPM]].
- **2011 — Sparse autoencoder** (Ng): sparsity penalty on activations $\lambda\sum_j |\hat\rho_j - \rho|^2$ — few units fire per input; recently revived for **mechanistic interpretability** of LLM activations.
- **2011 — Contractive autoencoder** (Rifai et al.): penalize the encoder Jacobian $\lambda\|J_{f_\phi}(x)\|_F^2$ — locally invariant representations, smooth along the data manifold.
- **2013 — [[VAE]]** ([Kingma & Welling](https://arxiv.org/abs/1312.6114)): the probabilistic version above — continuous, regularized latent space; the standard compression stage of [[LDM]].
- **2017 — [[VQ-VAE]]** ([van den Oord et al.](https://arxiv.org/abs/1711.00937)): **discrete codebook** latents via nearest-neighbor lookup + straight-through gradients — enables autoregressive priors over latent tokens; VQGAN (+ discriminator, perceptual loss) became LDM's tokenizer.
- **2019 — [[RAE]]** ([Ghosh et al.](https://arxiv.org/abs/1903.12436)): deterministic encoder + explicit L2 latent regularization — no posterior collapse by design; prior fit ex-post.
- **2021 — [[MAE]]** ([He et al.](https://arxiv.org/abs/2111.06377)): masking as the bottleneck — encode 25% of patches, reconstruct the rest; BERT-style vision pre-training that matches/exceeds supervised backbones.

| Variant | Regularization | Latent space | Primary use |
|---------|----------------|--------------|-------------|
| **Vanilla AE** | bottleneck dimension | deterministic | dimensionality reduction |
| **Denoising AE** | corrupted input | deterministic | robust features, denoising |
| **Sparse AE** | activation sparsity | deterministic | interpretable features |
| **Contractive AE** | Jacobian norm | deterministic | smooth manifold encoding |
| **[[VAE]]** | KL to prior ([[ELBO]]) | probabilistic | generation, compression |
| **[[VQ-VAE]]** | discrete codebook | discrete | autoregressive latents |
| **[[RAE]]** | L2 / spectral norm | deterministic | generation, no collapse |
| **[[MAE]]** | masking (75%) | high-dim patch | vision pre-training |

## Applications and Evaluation
---
**Applications**: dimensionality reduction (vs. [[t-SNE]], [[UMAP]]); the compression stage of latent generative models ([[LDM]], SD3, Movie Gen); self-supervised pre-training ([[MAE]]); anomaly detection (high reconstruction error = anomaly); denoising/inpainting; mechanistic interpretability (sparse AEs on LLM activations).

**Evaluation**: reconstruction — PSNR/SSIM and **rFID**; generation — **gFID** ([[FID]] of the full latent-model + decoder pipeline); representations — linear probe accuracy.

## Relationship to Contrastive Learning
---
The two dominant self-supervised paradigms:

| | Autoencoders (generative SSL) | [[Contrastive Learning]] |
|---|---|---|
| **Objective** | reconstruct input | align views, repel negatives |
| **Signal** | reconstruction error | embedding similarity |
| **Augmentation role** | corruption (DAE, MAE) | defines invariances (SimCLR, DINO) |
| **Latent structure** | explicit bottleneck | implicit (loss geometry) |
| **Fine-tuning / linear probe** | excellent / good (MAE) | good / excellent |

## Relationship to Other Notes
---
- [[VAE]] — the probabilistic autoencoder above: stochastic encoder + KL regularization; trained by maximizing the [[ELBO]]
- [[VQ-VAE]] / [[RAE]] — discrete-codebook and deterministic-regularized alternatives to the VAE bottleneck
- [[MAE]] — masking as the bottleneck; the reconstruction principle applied to vision pre-training at scale
- [[LDM]] / [[Flow Matching]] / [[DiT]] — latent generative modeling: FM/diffusion trained inside the (frozen) autoencoder latent space (SD3, Movie Gen)
- [[Score Matching]] — the denoising autoencoder implicitly learns the smoothed-data score; DSM is its rigorous descendant
- [[KL Divergence]] — the VAE's prior-enforcement term; the whole objective is one joint-space KL
- [[Manifold Hypothesis]] — why a bottleneck works: data lives near a low-dimensional manifold
- [[Contrastive Learning]] — the discriminative alternative to generative self-supervision
- [[t-SNE]] · [[UMAP]] — alternative nonlinear dimensionality reduction
- [[FID]] — rFID/gFID quantify the autoencoder's reconstruction–generation trade-off
