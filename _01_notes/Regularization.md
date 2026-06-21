### [[dl_moc]]  
# **Regularization**

**Tags**:: #dl_theo #learning
**Links**:: [[Neural Network Initialization]] · [[Optimization Algorithms]] · [[Double Descent]] · [[Normalization]] · [[Activation Functions]]


**Regularization** techniques reduce overfitting by constraining model capacity, injecting noise, or modifying the training data distribution — improving generalization beyond the training set.

## Why Regularize?
---
Without regularization, high-capacity networks can memorize training data exactly (interpolating all training examples). The key insight from [[Double Descent]]: overparameterized models can still generalize if implicit or explicit regularization is applied. Classical regularization prevents memorization in the underparameterized regime.

## Weight Penalties (L1 / L2)
---
**L2 regularization (weight decay)**: add a penalty proportional to the squared norm of weights to the loss:
$$
\mathcal{L}_\text{reg} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2
$$
The gradient update becomes $\theta \leftarrow \theta(1 - \eta\lambda) - \eta \nabla_\theta \mathcal{L}$, which **shrinks weights toward zero each step** ("weight decay"). Strongly penalizes large weights; solutions prefer many small weights over a few large ones.

**L1 regularization**: penalty proportional to the $\ell_1$ norm $\lambda \|\theta\|_1$. Produces **sparse** solutions (many weights exactly zero) because the $\ell_1$ ball has corners at coordinate axes. Useful for feature selection.

**Elastic Net**: weighted combination of L1 and L2 — inherits sparsity from L1 and stability from L2:
$$
\mathcal{L}_\text{elastic} = \mathcal{L} + \lambda_1\|\theta\|_1 + \frac{\lambda_2}{2}\|\theta\|^2
$$

**In modern deep learning**: L2 weight decay is standard, but implemented as **AdamW** (decoupled weight decay) rather than in the loss, because Adam's adaptive scaling corrupts the L2 penalty when applied in the loss.

## Dropout
---
[Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html). During training, randomly **zero out** each neuron with probability $p$ (the *drop probability*), then scale surviving neurons by $\frac{1}{1-p}$ (**inverted dropout**):

$$
\tilde{h} = h \odot \text{Bernoulli}(1 - p) \cdot \frac{1}{1-p}
$$

At inference, the full network is used without scaling (the scaling at training time already corrects expectations).

**Why it works**:
- Forces **redundant representations**: no single neuron can be relied on, so the network learns overlapping features
- Approximate **ensemble effect**: each forward pass uses a different subnetwork; inference averages over an exponential number of subnetworks

**Key rules**:
- Apply dropout **only during training** — never at inference
- Typical $p = 0.5$ for hidden layers; $p = 0.1$–$0.2$ for input layers
- Works less well with batch normalization (they conflict — both add noise, but BN normalizes it away)
- Less used in modern Transformers; replaced by other regularization

## Label Smoothing
---
Instead of hard one-hot targets $y \in \{0, 1\}$, use soft labels:
$$
y_\text{smooth} = (1 - \epsilon)\, y_\text{hard} + \frac{\epsilon}{K}
$$
where $K$ is the number of classes and $\epsilon$ is typically 0.1.

This prevents the model from becoming **overconfident** (logits growing unboundedly toward the correct class), which improves calibration and generalization. Widely used in image classification (e.g., ViT training) and language models (pre-training loss).

## Stochastic Depth (DropPath)
---
Randomly **drop entire residual blocks** during training with probability $p$, scaling the surviving blocks:
$$
\text{output} = x + \text{Bernoulli}(1-p) \cdot F(x)
$$

At inference, all blocks are used with their outputs scaled by $(1-p)$. Effectively trains an ensemble of shallower networks. Used in vision Transformers ([[ViT]], [[DINO]]) and modern CNN training.

## Data Augmentation
---
Artificially expand the training set by applying label-preserving transformations. Effective regularization because the model must learn features invariant to the applied transforms.

**Image augmentations**:
| Transform | Description |
|-----------|-------------|
| Random crop | Crop to random location + resize |
| Horizontal flip | Mirror left-right (not vertical for natural images) |
| Color jitter | Brightness, contrast, saturation, hue perturbation |
| Gaussian blur | Low-frequency smoothing |
| RandAugment | Randomly select from a large set of transforms |
| Mixup | Linear interpolation between two training examples: $\tilde x = \lambda x_i + (1-\lambda)x_j$ |
| CutMix | Replace a rectangular patch with a patch from another image |

**Why strong augmentation matters**: both [[Contrastive Learning]] (SimCLR, MoCo, BYOL) and supervised training ([[ViT]], [[DINO]]) rely heavily on augmentation to define what features are invariant. The augmentation set implicitly encodes the inductive biases you want the model to learn.

## Early Stopping
---
Monitor the **validation loss** during training; stop when it stops decreasing (or starts increasing). Prevents the model from memorizing training noise in the later training phases.

![[regularization_overfitting_curve.png|600]]

**Practical note**: in overparameterized models ([[Double Descent]]), early stopping can avoid the interpolation threshold peak. However, training *past* this threshold often recovers generalization. Modern practice (large transformers) is to train to completion with a proper learning rate schedule rather than using early stopping.

## Entropy Regularization
---
Used in **reinforcement learning** to prevent the policy from collapsing to a single action. An entropy bonus $\mathcal{H}[\pi]$ is added to the reward:
$$
\mathcal{L}_\text{RL} = \mathbb{E}[R] + \alpha \mathcal{H}[\pi(\cdot \mid s)]
$$
This encourages exploration by keeping the policy distribution spread out. Used in SAC (Soft Actor-Critic) and maximum entropy RL — see [[Deep Reinforcement Learning]].

## Summary
---

| Technique | Mechanism | When to use |
|-----------|-----------|-------------|
| **L2 / AdamW** | Penalize large weights | Almost always |
| **Dropout** | Random neuron zeroing | MLPs, some transformers |
| **Label smoothing** | Soft targets | Classification, LM pre-training |
| **Stochastic depth** | Drop residual blocks | Deep ResNets, ViTs |
| **Data augmentation** | Transform training data | Vision, speech, text |
| **Early stopping** | Stop at validation minimum | Small/medium models |
| **Entropy bonus** | Spread policy distribution | Reinforcement learning |
