### [[dl_moc]]
# **DDIM: Denoising Diffusion Implicit Models**

**Tags**:: #generative
**Links**:: [[DDPM]] · [[Score-Based Generative Modeling Methods]] · [[Score Matching]] · [[Flow Matching]] · [[LDM]]

Standard [[DDPM]] sampling is stochastic and requires all $T = 1000$ steps. **DDIM** ([Song et al., 2020](https://arxiv.org/abs/2010.02502)) gives a **deterministic** mapping from noise to image using a non-Markovian forward process that shares the same marginals $q(x_t \mid x_0)$ — so the *same trained DDPM network* can be reused, only the sampler changes. Notation follows the DDPM note ($\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{i\le t}\alpha_i$, noise predictor $\epsilon_\theta$).

### Sampling Rule
---
$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted }x_0} + \sqrt{1-\bar\alpha_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t\epsilon_t
$$

Each step: predict $x_0$ from the current noise estimate, then re-noise it to the *previous* noise level, with a free parameter $\sigma_t$ controlling how much fresh noise $\epsilon_t \sim \mathcal N(0, I)$ is injected. The paper parameterizes the family by $\eta$:
$$
\sigma_t(\eta) = \eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}},
$$
where $\eta = 1$ recovers the stochastic DDPM ancestral sampler and $\eta = 0$ (i.e. $\sigma_t = 0$) makes the process **deterministic** — no random $\epsilon_t$ at all.

### What Determinism Buys
---
- **Fewer steps**: skip to every $S$-th timestep, e.g., 50 steps instead of 1000 (same network, same weights)
- **Latent interpolation**: since the mapping is deterministic and invertible, you can encode real images to their latent $x_T$ (**DDIM inversion**) and interpolate between them
- **Consistency**: same starting $x_T$ always gives the same image

### Continuous-Time View
---
DDIM's deterministic sampler is equivalent to integrating the **probability flow ODE** of the corresponding SDE — see [[Score-Based Generative Modeling Methods]] for the SDE unification. In the [[Flow Matching]] view, this ODE is exactly the FM marginal velocity for the corresponding Gaussian probability path — DDPM training is FM with noise-prediction parameterization and inverted time.
