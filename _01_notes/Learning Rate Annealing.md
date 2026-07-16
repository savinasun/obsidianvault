### [[dl_moc]]  
# **Learning Rate Annealing**

**Tags**:: #dl_theo #learning
**Links**:: [[Optimization Algorithms]] · [[AdamW]] · [[LLMs]] · [[Scaling Laws]] · [[Pre-training]] · [[Neural Network Initialization]]


**Learning rate annealing** is the practice of varying the step size $\eta_t$ over training — typically a **warmup** ramp, a long high-LR phase, and a **decay** to (near) zero. The LR is the single most consequential optimizer hyperparameter, and its *schedule* often matters as much as its peak value. This note is the deep-dive companion to the schedules overview in [[Optimization Algorithms#Learning Rate Schedules|Optimization Algorithms]].

## Why Anneal
---
**Classical view (stochastic approximation).** Constant-step-size SGD does not converge — it reaches a **noise ball** around the optimum whose radius grows with $\eta$ (fluctuation variance $\propto \eta$). Robbins & Monro (1951) give the classical conditions for convergence of a decaying schedule:
$$
\sum_{t} \eta_t = \infty \quad (\text{reach anywhere}), \qquad \sum_{t} \eta_t^2 < \infty \quad (\text{noise dies out})
$$
e.g. $\eta_t \propto 1/t$. Modern schedules violate the second condition on purpose — they keep $\eta$ high for most of training and only kill the noise at the end.

**Modern picture.** The high-LR phase makes fast progress and acts as an implicit regularizer (larger noise → flatter basins); the decay phase descends into the basin found, harvesting the loss reduction that the gradient noise was hiding. Empirically, **a large fraction of the final loss improvement is realized during the decay phase** — which is why comparing runs mid-training at unequal LRs is misleading.

## Warmup
---
Linear ramp $\eta_t = \eta_\text{max}\, t / T_\text{warmup}$ for the first ~0.1–2% of steps. Why it's needed:

- **Adaptive-optimizer variance**: Adam's second-moment estimate $v_t$ is unreliable in the first steps, producing pathologically large adaptive steps. RAdam ([Liu et al., 2019](https://arxiv.org/abs/1908.03265)) derives a variance-rectification term and shows warmup acts as its heuristic equivalent.
- **Large batches**: the linear LR-scaling rule ($\eta \propto$ batch size) only works with gradual warmup ([Goyal et al., 2017](https://arxiv.org/abs/1706.02677) — ImageNet in 1 hour, batch 8192).
- **Early sharpness**: at initialization the loss landscape changes rapidly (progressive sharpening); a large LR immediately after init causes loss spikes or divergence. See [[Neural Network Initialization]].

Too-short warmup at scale shows up as early loss spikes; GPT-3 warmed up over the first 375M tokens, Llama 3 405B over 8,000 steps.

## The Schedule Zoo
---
| Schedule | Form (after warmup) | Notes |
|---|---|---|
| **Step decay** | $\eta_0\, \gamma^{\lfloor t/s \rfloor}$ | classic CNN recipe (ResNet: ÷10 at epochs 30/60/90) |
| **Exponential** | $\eta_0\, e^{-t/\tau}$ | smooth step-decay analogue |
| **Linear** | $\eta_\text{max}(1 - t/T)$ | strong baseline; provably near-optimal rates for last-iterate SGD ([Defazio et al., 2023](https://arxiv.org/abs/2310.07831)) |
| **Cosine** | $\eta_\text{min} + \tfrac12(\eta_\text{max}-\eta_\text{min})(1 + \cos \pi t/T)$ | the LLM default 2020–2024; slow early decay, fast middle, gentle landing |
| **Cosine + warm restarts** | cosine cycles of growing length | SGDR ([Loshchilov & Hutter, 2016](https://arxiv.org/abs/1608.03983)); restarts explore multiple basins; snapshot ensembles |
| **Inverse-sqrt (Noam)** | $d_\text{model}^{-1/2} \min(t^{-1/2},\, t\, T_w^{-3/2})$ | original [[Transformer]] schedule; budget-agnostic but decays slowly |
| **Cyclical (CLR)** | triangular oscillation between bounds | [Smith, 2015](https://arxiv.org/abs/1506.01186); LR range test for picking $\eta_\text{max}$ |
| **One-cycle** | one up-down triangle + short annihilation | "super-convergence" ([Smith & Topin, 2017](https://arxiv.org/abs/1708.07120)); strong for fast supervised training |
| **WSD / trapezoidal** | warmup → **constant** → short decay (last ~10–20%) | MiniCPM ([Hu et al., 2024](https://arxiv.org/abs/2404.06395)); matches cosine at equal compute ([Hägele et al., 2024](https://arxiv.org/abs/2405.18392)) |

## Cosine in LLM Practice
---
- **GPT-3**: cosine decay to 10% of peak over 260B tokens ([Brown et al., 2020](https://arxiv.org/abs/2005.14165))
- **Chinchilla's schedule lesson** ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)): the cosine cycle length must **match the intended training duration** — stretching the cycle >25% beyond the actual stopping point measurably degrades loss. Consequence: a compute-optimal cosine run must fix its token budget *in advance*, and every budget in a [[Scaling Laws]] sweep needs its own full run.
- **Llama 3 405B** ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)): peak $8\times10^{-5}$, linear warmup 8,000 steps, cosine to $8\times10^{-7}$ (1% of peak) over 1.2M steps — alongside a batch-size ramp $4\text{M} \to 8\text{M} \to 16\text{M}$ tokens. Decaying to ~1% rather than the older 10% convention reflects evidence that a low final LR helps at long token horizons.

## WSD: Warmup–Stable–Decay
---
$$
\eta_t = \begin{cases} \eta_\text{max}\, t/T_w & t < T_w \quad \text{(warmup)}\\ \eta_\text{max} & T_w \le t < T_s \quad \text{(stable)}\\ \eta_\text{max} \cdot f\!\left(\tfrac{t - T_s}{T - T_s}\right) & t \ge T_s \quad \text{(decay, last 10–20\%)} \end{cases}
$$

with $f$ linear or the slightly better $(1-\sqrt{x})$ shape ([Hägele et al., 2024](https://arxiv.org/abs/2405.18392)). During the stable phase the loss plateaus above a cosine run; the decay phase then drops it sharply to match — the gains are "realized" in the cooldown.

Why WSD displaced cosine for frontier pre-training:
- **Budget-free**: no cycle length to pre-commit; training can be extended indefinitely and cooled down whenever desired
- **Branchable**: any stable-phase checkpoint can spawn a decayed model — one long run yields every point of a [[Scaling Laws]] data-budget sweep for the cost of the cooldowns (Hägele et al.)
- **Data annealing**: the cooldown is a natural place to upweight high-quality data (MiniCPM mixes in SFT-grade data during decay; Llama 3 similarly "anneals" on curated data at the end)
- **Continual pre-training**: resume from the stable checkpoint, not the decayed one

## Annealing ≈ Weight Averaging
---
Averaging iterates along a high-LR trajectory approximates the effect of having decayed the LR:
- **SWA** ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)): average checkpoints under a high/cyclical LR → flatter minimum, better generalization
- **EMA** of weights: standard in diffusion-model training; a continuous version of the same idea
- **Schedule-Free** ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)): replaces the schedule with an online interpolation of iterate averaging — no decay phase, no horizon commitment, competitive with tuned cosine
- Hägele et al. find SWA over a constant-LR run recovers much (not all) of the cooldown's gain — a cheap mid-training estimate of "what would this run give if annealed now?"

## Interactions & Pitfalls
---
- **Schedule–horizon mismatch** is the cosine failure mode (Chinchilla's >25% rule); WSD and inverse-sqrt sidestep it
- **Mid-training comparisons lie**: a constant-LR (stable-phase) run looks worse than a cosine run at the same step until it is cooled down
- **LR ↔ batch size**: linear scaling holds for SGD up to a critical batch size (Goyal et al.); for Adam-family optimizers the empirical rule is closer to square-root scaling
- **LR ↔ weight decay**: in [[AdamW]] what matters is the product $\eta\lambda$ (the weight-decay timescale $\sim 1/(\eta\lambda)$ steps) — changing the schedule silently changes the effective regularization
- **Warmup after resume**: restarting from a checkpoint with a fresh optimizer state needs a short re-warmup, or the stale-curvature problem reappears
- **Fine-tuning**: use a schedule an order of magnitude shorter and lower — typically constant or cosine from $\sim\!10^{-5}$ with brief warmup; the pre-training schedule intuitions do not transfer directly

## References
---
- [Robbins & Monro, 1951 — A Stochastic Approximation Method (Ann. Math. Stat. 22)](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full)
- [Loshchilov & Hutter, 2016 — SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- [Smith, 2015 — Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) · [Smith & Topin, 2017 — Super-Convergence](https://arxiv.org/abs/1708.07120)
- [Goyal et al., 2017 — Accurate, Large Minibatch SGD (linear scaling + warmup)](https://arxiv.org/abs/1706.02677)
- [Liu et al., 2019 — RAdam: On the Variance of the Adaptive Learning Rate](https://arxiv.org/abs/1908.03265)
- [Hoffmann et al., 2022 — Chinchilla (cosine cycle-length finding)](https://arxiv.org/abs/2203.15556)
- [Hu et al., 2024 — MiniCPM (WSD schedule)](https://arxiv.org/abs/2404.06395)
- [Hägele et al., 2024 — Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392)
- [Izmailov et al., 2018 — Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
- [Defazio et al., 2023 — Optimal Linear Decay Learning Rate Schedules](https://arxiv.org/abs/2310.07831) · [Defazio et al., 2024 — The Road Less Scheduled (Schedule-Free)](https://arxiv.org/abs/2405.15682)
