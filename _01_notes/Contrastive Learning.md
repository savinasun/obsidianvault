### [[dl_moc]]  
# **Contrastive Learning**

**Tags**:: #learning
**Links**:: [[CLIP]] · [[DINO]] · [[Manifold Hypothesis]] · [[Entropy]] · [[FSL]]


**Contrastive learning** is a self-supervised learning paradigm that learns representations by pulling together embeddings of semantically similar (positive) pairs and pushing apart dissimilar (negative) pairs — without requiring human labels.

## Core Idea
---
Given a sample $x$, create two augmented **views** $\tilde x_1, \tilde x_2$ via data augmentation (crop, color jitter, blur, etc.). Train an encoder $f_\theta$ such that:
- $f(\tilde x_1)$ and $f(\tilde x_2)$ are close in embedding space (**positive pair**)
- $f(\tilde x_1)$ and $f(\tilde x_j)$ from different images are far (**negative pairs**)

The representation is forced to capture *invariant* features (shared across views) and discard *variant* features (augmentation-specific).

## SimCLR (Chen et al., 2020)
---
**Simple Framework for Contrastive Learning of Visual Representations** — the canonical contrastive learning method.

**Components**:
- **Data augmentation**: random crop + color distortion + Gaussian blur → two views $(x_i, x_j)$
- **Base encoder** $f$: ResNet; produces feature vector $h = f(x)$
- **Projection head** $g$: 2-layer MLP; maps $h \to z$ where the loss is computed (discarded after training)

**NT-Xent loss** (Normalized Temperature-scaled Cross Entropy):
$$
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}
$$
where $\text{sim}(u,v) = u^\top v / (\|u\|\|v\|)$ is cosine similarity and $\tau$ is a temperature hyperparameter.

**Training**: sample a batch of $N$ images; each gives 2 views → $2N$ total embeddings; for each of the $2N$ views, compute loss against its positive and $2N-2$ negatives. Total loss is the mean over all $2N$ views.

**After training**: discard the projection head; use the encoder $f$ for downstream tasks via linear probe or fine-tuning.

**Key findings**:
- Composition of augmentations matters more than individual augmentations
- Larger batch sizes improve performance (more negatives per step)
- The projection head is critical — without it, representations are worse

## MoCo (Momentum Contrast, He et al., 2019)
---
SimCLR requires large batches for good negatives. **MoCo** decouples the negative pool from the batch size using a **memory queue** and a **momentum encoder**:

- **Query encoder** $f_\theta$: updated by gradient
- **Key encoder** $f_{\xi}$: updated by **exponential moving average (EMA)**: $\xi \leftarrow m\xi + (1-m)\theta$
- **Queue**: stores the last $K$ key embeddings (e.g., $K = 65536$)

The query's positive is the corresponding key from the same image; negatives are all entries in the queue. The momentum update ensures the key encoder changes slowly, giving consistent representations in the queue.

**Advantage**: enables large effective batch sizes ($K \gg$ batch size) at low memory cost.

## BYOL (Bootstrap Your Own Latent, Grill et al., 2020)
---
BYOL achieves strong representations **without negative pairs** — avoiding mode collapse through asymmetry:

- **Online network** ($\theta$): encoder + projector + predictor $q_\theta$
- **Target network** ($\xi$): encoder + projector only, updated by EMA of $\theta$

Loss:
$$
\mathcal{L} = 2 - 2 \cdot \frac{\langle q_\theta(z), \bar z' \rangle}{\|q_\theta(z)\| \cdot \|\bar z'\|}
$$
where $z = \text{proj}(f_\theta(\tilde x_1))$, $\bar z' = \text{proj}(f_\xi(\tilde x_2))$ (target, stop-gradient).

**Why doesn't it collapse?** The predictor $q_\theta$ creates an asymmetry: the online network must *predict* the target network's output. The EMA target creates a slowly-moving target that prevents trivial solutions. The combination prevents the collapse that a pure symmetric negative-free approach would suffer.

## Barlow Twins (Zbontar et al., 2021)
---
Reduces redundancy between features by minimizing the **cross-correlation matrix** between the two views' embeddings away from identity:
$$
\mathcal{L} = \underbrace{\sum_i (1 - \mathcal{C}_{ii})^2}_{\text{invariance}} + \lambda \underbrace{\sum_i\sum_{j\neq i} \mathcal{C}_{ij}^2}_{\text{redundancy reduction}}
$$
where $\mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,j}^B)^2}}$.

No negatives needed. Encourages each feature dimension to encode a distinct factor of variation.

## Comparison
---

| Method | Negatives | Momentum encoder | Batch size sensitivity |
|--------|-----------|-----------------|----------------------|
| **SimCLR** | In-batch | No | High (needs large batches) |
| **MoCo** | Queue | Yes | Low |
| **BYOL** | None | Yes | Low |
| **Barlow Twins** | None | No | Medium |
| **[[DINO]]** | None (distillation) | Yes | Low |
| **[[CLIP]]** | In-batch (cross-modal) | No | High |

## Why Contrastive Learning Works: Theory
---
Contrastive objectives lower-bound the **mutual information** between the two views $I(Z_1; Z_2)$:
$$
\mathcal{L}_\text{InfoNCE} \le -I(Z_1; Z_2) + \log N
$$
where $N$ is the number of negatives. Maximizing the objective (minimizing the loss) maximizes the mutual information between the two views' representations.

By the **data processing inequality**, $I(Z_1; Z_2) \le I(X_1; X_2)$ — so the representation captures the mutual information that exists between different views of the same image, which corresponds to the **invariant content** (semantics) rather than view-specific details (augmentation artifacts).

## Extensions
---
- **SimCLR v2**: semi-supervised learning with larger models + knowledge distillation from fine-tuned to linear models
- **[[CLIP]]**: contrastive learning between image–text pairs, enabling zero-shot transfer
- **[[DINO]]**: self-distillation with no labels; connects to contrastive learning via the centering mechanism
- **Masked Autoencoders (MAE)**: generative alternative — predict masked patches instead of contrasting views
