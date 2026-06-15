### [[dl_moc]]
# **Mathematics**

**Tags**:: #math
**Links**:: [[learning_methods_moc]] · [[sequence_llms_moc]] · [[vision_ssl_moc]] · [[generative_moc]] · [[rl_embodied_moc]]

The mathematical foundations underlying deep learning: linear algebra, probability theory, information theory, and the geometric tools used for dimensionality reduction and representation analysis.

## Linear Algebra & Matrix Decompositions
---
- **[[Linear Algebra]]** — Vector spaces, matrix operations, eigendecomposition, and the core mechanics used throughout deep learning (weight matrices, projections, attention)
- **[[SVD]]** — Singular Value Decomposition: factorizes any matrix $M = U\Sigma V^\top$; used for low-rank approximation, PCA, and understanding weight matrix geometry
- **[[Spectral Analysis]]** — Eigenvalues of the Hessian and weight matrices; connections to optimization landscape, sharpness, and generalization

## Information Theory
---
- **[[Entropy]]** — Shannon entropy $H(X) = -\sum_x p(x)\log p(x)$; measures uncertainty; foundation for cross-entropy loss and perplexity
- **[[KL Divergence]]** — Kullback-Leibler divergence $D_\text{KL}(P\|Q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$; measures distribution mismatch; used in VAEs, RL policy optimization, and distillation
- **[[ELBO]]** — Evidence Lower Bound: $\log p(x) \ge \mathbb{E}_q[\log p(x|z)] - D_\text{KL}(q(z|x)\|p(z))$; the variational inference objective underpinning VAEs and latent variable models

## Probabilistic Methods
---
- **[[MCMC]]** — Markov Chain Monte Carlo: sampling from intractable distributions via Metropolis-Hastings and Gibbs sampling; connects to Langevin dynamics and score-based generative models
- **[[Manifold Hypothesis]]** — Real-world high-dimensional data lies near low-dimensional manifolds; theoretical basis for why dimensionality reduction and representation learning work

## Dimensionality Reduction
---
- **[[t-SNE]]** — t-Distributed Stochastic Neighbor Embedding: non-linear 2D/3D visualization that preserves local neighborhood structure; uses [[KL Divergence]] between pairwise similarity distributions
- **[[UMAP]]** — Uniform Manifold Approximation and Projection: faster alternative to t-SNE with better global structure preservation; grounded in Riemannian geometry and fuzzy topology

## Metrics & Loss Functions
---
- **[[Hamming Loss]]** — Fraction of incorrectly predicted labels in multi-label classification; used as an evaluation metric in structured prediction tasks

## Dataview
---
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "math")
SORT file.name ASC
```
