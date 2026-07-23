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
The [[VAE]] (Kingma & Welling, 2013) fixes this by making encoder and decoder **probabilistic** — $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x)))$, $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \sigma_\theta^2(z) I_d)$, with $\sigma_\phi = \sigma_\theta = 0$ recovering the standard autoencoder — and adding a [[KL Divergence]] penalty pushing each encoding toward the prior $p_\text{prior} = \mathcal{N}(0, I_k)$:

$$
\mathcal{L}_\text{VAE} = \underbrace{-\,\mathbb{E}_{x,\, z \sim q_\phi(\cdot|x)}\big[\log p_\theta(x|z)\big]}_{\text{reconstruction}} \;+\; \beta\, \underbrace{\mathbb{E}_{x}\big[D_\text{KL}\big(q_\phi(\cdot|x)\,\|\, p_\text{prior}\big)\big]}_{\text{prior enforcement}} .
$$

The latent distribution becomes Gaussian-like *by construction* — exactly the "nice" latent a downstream generative model needs. Full treatment in [[VAE]]: the four-term β-VAE loss, reparameterization trick and training code, posterior collapse and practical remarks, and the joint-KL / [[ELBO]] / amortization-gap theory (6.S184 §6.2.2 + App. D).

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
