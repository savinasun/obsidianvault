### [[dl_moc]]  
# **Flow Matching**

**Tags**:: #generative #dl_theo #diffusion
**Links**:: [[DDPM]] · [[LDM]] · [[DiT]] · [[Score-Based Generative Modeling Methods]] · [[VAE]] · [[FID]] · [[RAE]]


**Flow Matching** ([Lipman et al., 2022](https://arxiv.org/abs/2210.02747)) is a simulation-free method for training Continuous Normalizing Flows (CNFs) by directly regressing a learned velocity field onto a target. It avoids the expensive ODE simulation required by prior CNF training, recovers diffusion as a special case, and supports optimal-transport paths that produce straighter trajectories — enabling high-quality generation with far fewer sampling steps.

## Background: Continuous Normalizing Flows
---
A CNF defines a time-dependent vector field $v_\theta(x, t)$ that generates a probability path $p_t$ from a simple prior $p_0 = \mathcal{N}(0, I)$ to the data distribution $p_1 = p_\text{data}$ via the ODE:

$$
\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]
$$

To sample: integrate from $x_0 \sim p_0$ forward to $t = 1$. To train classically: maximize log-likelihood using the instantaneous change-of-variables formula — which requires solving the ODE during training. This is prohibitively expensive at scale.

## Flow Matching Objective
---
Flow Matching sidesteps likelihood by directly regressing $v_\theta$ onto a target field $u_t$ known to generate $p_t$:

$$
\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]
$$

**Problem**: computing $u_t(x)$ and sampling from the marginal $p_t$ requires knowledge of the full coupling between $p_0$ and $p_1$ — which is exactly what we're trying to learn.

## Conditional Flow Matching (CFM): The Key Insight
---
Instead of working with intractable marginal quantities, condition on individual data points $x_1 \sim p_\text{data}$ and define:
- A **conditional probability path** $p_t(x \mid x_1)$ — a simple path per data point
- A **conditional velocity field** $u_t(x \mid x_1)$ — analytically known

The **CFM loss** is:

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\, x_1 \sim p_\text{data},\, x_0 \sim \mathcal{N}(0,I)}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]
$$

where $x_t = (1 - t)x_0 + tx_1$ is the linear interpolant at time $t$.

**Why this is valid**: the marginal velocity $u_t(x)$ is a posterior-weighted mixture of conditional velocities:

$$
u_t(x) = \mathbb{E}[u_t(x \mid x_1) \mid x_t = x]
$$

It can be shown that $\mathcal{L}_\text{CFM}$ and $\mathcal{L}_\text{FM}$ have identical gradients w.r.t. $\theta$ — they differ only by a constant that doesn't depend on $\theta$.

## Gaussian Conditional Paths
---
The standard choice for $p_t(x \mid x_1)$ is a Gaussian that interpolates from $\mathcal{N}(0, I)$ to a delta at $x_1$:

$$
p_t(x \mid x_1) = \mathcal{N}(x;\; t \cdot x_1,\; (1-t)^2 I)
$$

Sampling a point on this path: $x_t = (1-t)x_0 + tx_1$, with conditional velocity $u_t(x_t \mid x_1) = x_1 - x_0$ — a **constant** displacement vector, independent of $t$.

This makes the training target trivially computable: the model just needs to predict the direction from noise to data, at any time $t$.

## Training and Sampling
---
**Training** (one step):
1. Sample $x_1 \sim p_\text{data}$, $x_0 \sim \mathcal{N}(0, I)$, $t \sim \mathcal{U}[0, 1]$
2. Interpolate: $x_t = (1-t)x_0 + tx_1$
3. Predict: $v_\text{pred} = v_\theta(x_t, t)$
4. Loss: $\|v_\text{pred} - (x_1 - x_0)\|^2$

No noise schedule, no SNR weighting, no variance terms. Simpler than diffusion training.

**Sampling** (Euler integration):
$$
x_{t + \Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t
$$

Integrate from $t = 0 \to 1$ using any ODE solver. Fewer steps are needed when trajectories are straight.

## Optimal Transport Paths
---
The Gaussian path pairs $x_0$ with $x_1$ **randomly** — two noise samples near $x_1$ take different paths that may cross, requiring the model to resolve intersections. Crossing paths must curve, forcing more integration steps.

**OT-CFM** uses mini-batch optimal transport to find pairings $\pi$ that minimize total squared displacement:

$$
\pi^* = \arg\min_\pi \sum_i \|x_{0,i} - x_{1,\pi(i)}\|^2
$$

solved via the Sinkhorn or Hungarian algorithm on each mini-batch. The result: **non-crossing, globally straighter trajectories** — the model learns a near-linear flow. This is the version that outperforms diffusion paths on ImageNet.

## Rectified Flow
---
**Rectified Flow** ([Liu et al., 2022](https://arxiv.org/abs/2209.03003)) is a closely related method that arrives at the same straight-line interpolation from a different motivation. The key addition is **reflow**: after training, generate synthetic $(x_0, x_1)$ pairs by running the learned ODE, then retrain on those pairs. Reflow provably straightens trajectories across iterations.

Straight trajectories have a critical property: if $v_\theta$ is constant along the path, a **single Euler step** recovers the sample exactly. Reflow enables distillation into 1–2 step generators.

Rectified Flow underlies **Stable Diffusion 3** and **FLUX.1**.

## Relationship to Diffusion Models
---

| | Diffusion (DDPM) | Flow Matching (CFM) |
|---|---|---|
| **Network predicts** | Noise $\epsilon_\theta(x_t, t)$ | Velocity $v_\theta(x_t, t)$ |
| **Interpolant** | $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ | $x_t = (1-t)x_0 + tx_1$ |
| **Training target** | Noise $\epsilon$ | Displacement $x_1 - x_0$ |
| **Path geometry** | Curved (marginal Gaussian) | Straight (linear) |
| **Training loss** | $\|\epsilon_\theta - \epsilon\|^2$ | $\|v_\theta - (x_1 - x_0)\|^2$ |
| **Sampling** | Reverse SDE / DDIM | Forward ODE integration |
| **Noise schedule** | Required ($\bar\alpha_t$ schedule) | Not needed |
| **Steps (typical)** | 20–1000 | 5–50; 1–2 after reflow |
| **Simulation-free training** | Yes | Yes |

Diffusion models are a special case of flow matching with a specific curved path choice. Flow matching with OT paths finds shorter, less-curved routes between $p_0$ and $p_1$, which requires fewer NFE (number of function evaluations) at inference.

## Why Straight Paths Matter
---
Euler integration error scales with path curvature: $\text{error} \propto \Delta t^2 \cdot \kappa$, where $\kappa$ is curvature. Perfectly straight paths ($\kappa = 0$) have zero integration error at any step size — a single Euler step suffices. Diffusion paths are curved by construction (the noise schedule introduces a non-linear trajectory in probability space), requiring many steps for accurate integration.

This explains the $10\text{–}100\times$ step-count advantage of flow matching over DDPM-style samplers.

## Applications
---
- **Stable Diffusion 3 / FLUX.1**: use rectified flow (straight-line FM) with a [[DiT]] backbone for text-to-image generation
- **Image generation**: FM with OT paths achieves state-of-the-art likelihood and FID on ImageNet, outperforming diffusion-based CNFs
- **Protein structure**: AlphaFold 3 uses a flow matching diffusion head for atomic coordinate generation
- **Audio**: F5-TTS, E2-TTS use flow matching for fast text-to-speech synthesis

## Relationship to Other Notes
---
- [[DDPM]] — diffusion is a special case of flow matching with a curved (cosine/linear schedule) path; FM generalizes and improves upon it
- [[Score-Based Generative Modeling Methods]] — score matching learns $\nabla_x \log p_t(x)$; flow matching learns $v_t(x)$ directly; both train CNFs/SDEs without simulation
- [[LDM]] — LDMs run diffusion in latent space; SD3/FLUX replace the diffusion process with flow matching in the same VAE latent pipeline
- [[DiT]] — FLUX and SD3 use DiT architectures with flow matching objectives; the backbone is unchanged, only the training target shifts
- [[RAE]] — RAE + DiT uses a diffusion objective in the RAE latent space; could also be trained with FM for fewer-step sampling
- [[FID]] — FM with OT paths achieves better FID than diffusion-based CNFs on ImageNet at comparable or lower compute
