### [[dl_moc]]  
# **Autoencoder**

**Tags**:: #generative #learning
**Links**:: [[VAE]] · [[VQ-VAE]] · [[RAE]] · [[MAE]] · [[LDM]] · [[ELBO]] · [[KL Divergence]] · [[Manifold Hypothesis]] · [[Contrastive Learning]] · [[t-SNE]] · [[UMAP]] · [[FID]]

Autoencoder = Encoder + Decoder


An **autoencoder** is a neural network trained to compress input data into a low-dimensional **latent code** and **reconstruct it back** — learning a compact, lossy representation **without supervision**. The reconstruction objective forces the model to discover the most informative structure in the data. Over four decades, autoencoders evolved from dimensionality reduction tools into the **compression backbone of modern generative models**.

## Latent Space
---



## Core Architecture
---
An autoencoder has two components:
$$
z = f_\phi(x) \qquad \hat{x} = g_\theta(z)
$$
- **Encoder** $f_\phi$: maps input $x \in \mathbb{R}^D$ → latent code $z \in \mathbb{R}^d$, with $d \ll D$ (the **bottleneck**)
- **Decoder** $g_\theta$: maps $z$ back to reconstruction $\hat{x} \in \mathbb{R}^D$
- **Loss**: reconstruction error, typically MSE for continuous data or cross-entropy for binary:
$$
\mathcal{L} = \|x - g_\theta(f_\phi(x))\|^2
$$
The bottleneck prevents the trivial identity solution — the model must compress, forcing it to retain only the most informative features. By the [[Manifold Hypothesis]], real data lies on a low-dimensional manifold; the autoencoder learns to parameterize that manifold.

## History and Advancements
---
### 1986 — Original Autoencoder (Rumelhart et al., 1986)
Introduced as part of the backpropagation paper as a demonstration that hidden representations could be learned unsupervised. A shallow network was trained to reconstruct its input through a narrow bottleneck — conceptually equivalent to **nonlinear PCA**. At the time, limited to shallow architectures due to vanishing gradients.

### 2006 — Deep Autoencoders and Layer-Wise Pre-Training ([Hinton & Salakhutdinov, 2006](https://doi.org/10.1126/science.1127647))
Hinton & Salakhutdinov showed that deep autoencoders could dramatically outperform PCA for dimensionality reduction — but only with careful initialization via **greedy layer-wise pre-training** (stacking Restricted Boltzmann Machines). This was a landmark result in the first wave of deep learning, demonstrating that depth was worth the training difficulty. Also used as an **unsupervised pre-training** technique before fine-tuning on labeled data, before dropout and better optimizers made random initialization viable.

### 2008–2010 — Denoising Autoencoder ([Vincent et al., 2010](https://www.jmlr.org/papers/v11/vincent10a.html))
**Key insight**: rather than reconstructing clean input from clean input, train to reconstruct clean $x$ from *corrupted* $\tilde{x}$:

$$
\mathcal{L}_\text{DAE} = \|x - g_\theta(f_\phi(\tilde{x}))\|^2, \quad \tilde{x} = \text{corrupt}(x)
$$

Corruption can be Gaussian noise addition, masking (zeroing out features), or salt-and-pepper noise. This forces the encoder to learn **robust, generalizing features** rather than memorizing the identity. Critically, the Denoising Autoencoder objective is theoretically equivalent to **score matching** — implicitly learning $\nabla_x \log p(x)$, making it a precursor to [[Score-Based Generative Modeling Methods|score-based generative models]] and [[DDPM]].

### 2011 — Sparse Autoencoder (Ng)
Adds a **sparsity penalty** on hidden activations:

$$
\mathcal{L}_\text{SAE} = \|x - \hat{x}\|^2 + \lambda \sum_j |\hat\rho_j - \rho|^2
$$

where $\hat\rho_j$ is the average activation of hidden unit $j$ and $\rho$ is the target sparsity (e.g., 0.05). Only a few neurons fire for any given input, producing interpretable, disentangled features. Sparse autoencoders have recently found renewed interest as a tool for **mechanistic interpretability** of large language models — decomposing activations into sparse, human-interpretable features.

### 2011 — Contractive Autoencoder (Rifai et al., 2011)
Penalizes the **Frobenius norm of the encoder's Jacobian**:

$$
\mathcal{L}_\text{CAE} = \|x - \hat{x}\|^2 + \lambda \|J_{f_\phi}(x)\|_F^2
$$

This forces the representation to be **locally invariant to small input perturbations** — the encoder's output should not change much if the input changes slightly. Contractive AEs learn representations that are smooth along the data manifold and sensitive only to directions of true variation, related to the geometry s tudied in [[Manifold Hypothesis]].

### 2013 — Variational Autoencoder ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114))
The [[VAE]] transforms the autoencoder into a **probabilistic generative model** by replacing the deterministic encoder with a distribution $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ and adding a [[KL Divergence]] penalty to the [[ELBO]] objective. The latent space becomes continuous and regularized — any point $z \sim \mathcal{N}(0,I)$ decodes to a plausible image. VAEs enabled controllable generation, interpolation, and became the standard compression stage in [[LDM|Latent Diffusion Models]].

### 2017 — VQ-VAE ([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937))
[[VQ-VAE|Vector Quantized VAE]] replaces the continuous Gaussian latent with a **discrete codebook** $\{e_k\}_{k=1}^K$. The encoder maps $x$ to a continuous vector, snapped to the nearest codebook entry via $\arg\min$. A **straight-through estimator** passes gradients through the non-differentiable lookup. The discrete bottleneck enables autoregressive modeling over latent tokens (PixelCNN or [[Transformer]] prior). VQ-VAE-2 extended this hierarchically (coarse + fine codes); VQGAN added a patch discriminator and perceptual loss for sharper reconstructions and became the compression stage in [[LDM]].

### 2019 — Regularized Autoencoder ([Ghosh et al., 2019](https://arxiv.org/abs/1903.12436))
[[RAE]] showed that neither stochastic encoding nor KL divergence is necessary to get well-structured latent spaces. A **deterministic encoder** with explicit L2 regularization on latent codes avoids posterior collapse by design and produces less blurry samples. Separates the latent geometry (encoder training) from the prior (ex-post density estimation), a cleaner design than the VAE's intertwined ELBO.

### 2021 — Masked Autoencoder ([He et al., 2021](https://arxiv.org/abs/2111.06377))
[[MAE]] applied the autoencoder principle to self-supervised **vision pre-training**. The encoder processes only 25% of visible image patches (a [[ViT]]); the lightweight decoder reconstructs the masked 75%. The reconstruction target is raw pixel values. MAE is the vision analogue of BERT's masked language modeling — the bottleneck is created by masking rather than a narrow layer, forcing the encoder to learn globally coherent representations. MAE pre-trained backbones match or exceed supervised pre-training on detection and segmentation.

## Taxonomy of Autoencoder Variants
---

| Variant | Regularization mechanism | Latent space | Primary use |
|---------|-------------------------|--------------|-------------|
| **Vanilla AE** | Bottleneck dimension | Deterministic | Dimensionality reduction |
| **Denoising AE** | Corrupted input | Deterministic | Robust features, denoising |
| **Sparse AE** | Activation sparsity | Deterministic | Interpretable features |
| **Contractive AE** | Jacobian norm | Deterministic | Smooth manifold encoding |
| **[[VAE]]** | KL divergence (ELBO) | Probabilistic | Generation, compression |
| **[[VQ-VAE]]** | Discrete codebook | Discrete | Autoregressive generation |
| **[[RAE]]** | L2 / spectral norm | Deterministic | Generation (no collapse) |
| **[[MAE]]** | Masking (75% patches) | High-dim patch | Vision pre-training |

## Applications
---
- **Dimensionality reduction**: nonlinear alternative to PCA; compare with [[t-SNE]] and [[UMAP]] for visualization
- **Generative modeling**: [[VAE]], [[VQ-VAE]], [[RAE]]; the encoder/decoder is also the compression stage of [[LDM]]
- **Self-supervised pre-training**: [[MAE]] for vision; denoising objective connects to [[Score-Based Generative Modeling Methods]]
- **Anomaly detection**: train on normal data; high reconstruction error at test time indicates anomalies
- **Denoising / inpainting**: denoising AE directly; masked AE for inpainting
- **Mechanistic interpretability**: sparse AEs applied to LLM residual stream activations to extract interpretable features

## Relationship to Contrastive Learning
---
Autoencoders and [[Contrastive Learning]] are the two dominant self-supervised paradigms:

| | Autoencoders (generative SSL) | Contrastive Learning |
|---|---|---|
| **Objective** | Reconstruct input | Align views, push apart negatives |
| **Signal** | Reconstruction error | Similarity in embedding space |
| **Augmentation role** | Corruption (DAE, MAE) | Defines invariances (SimCLR, DINO) |
| **Latent structure** | Explicit bottleneck | Implicit (via loss geometry) |
| **Fine-tuning** | Excellent (MAE) | Good |
| **Linear probe** | Good | Excellent |

## Evaluation
---
- **Reconstruction quality**: PSNR, SSIM for image autoencoders
- **Generation quality**: [[FID]] for VAE/RAE-based generative models
- **Representation quality**: linear probe accuracy on downstream classification tasks (for MAE, Contrastive AEs)

## Relationship to Other Notes
---
- [[VAE]] — probabilistic autoencoder; adds stochastic encoder and KL regularization for structured generation
- [[RAE]] — deterministic alternative to VAE; avoids posterior collapse via explicit L2 regularization
- [[MAE]] — masked autoencoder; applies the reconstruction principle to vision pre-training at scale
- [[LDM]] — uses a VQ-VAE/VAE encoder–decoder as its perceptual compression stage
- [[ELBO]] — the training objective for VAEs, derived by lower-bounding $\log p(x)$
- [[KL Divergence]] — the regularization term in the VAE ELBO
- [[Manifold Hypothesis]] — theoretical motivation: bottleneck forces learning of the low-dim data manifold
- [[Contrastive Learning]] — the alternative self-supervised paradigm; generative (AE) vs. discriminative (contrastive)
- [[t-SNE]] · [[UMAP]] — alternative nonlinear dimensionality reduction methods
- [[FID]] — standard metric for evaluating autoencoder-based generative model quality
