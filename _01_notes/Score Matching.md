### [[dl_moc]]  
# **Score Matching**

**Tags**:: #generative #dl_theo #diffusion
**Links**:: [[Generative Modeling]] · [[Flow Matching]] · [[ODEs, SDEs, and Probability Paths]] · [[Score-Based Generative Modeling Methods]] · [[DDPM]] · [[Divergence Theorem]] · [[MCMC]]


**Score Matching** is the training algorithm of diffusion models: regress a neural network $s_t^\theta$ onto the **score function** $\nabla \log p_t$ of a probability path. This note follows §4 of the MIT 6.S184 lecture notes ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf); [course site](https://diffusion.csail.mit.edu/)), which presents score matching as the exact stochastic sibling of [[Flow Matching]]: the same conditional-path machinery, the same "conditional loss has the same gradients" theorem — plus one extra reward, since a learned score unlocks **SDE sampling** and **guidance**.

**Notation** (matches the [[Flow Matching]] note): source $p_\text{init} = \mathcal{N}(0, I_d)$, data $p_\text{data}$; data point $z \sim p_\text{data}$, noise $\epsilon \sim \mathcal{N}(0, I_d)$; noise schedulers $(\alpha_t, \beta_t)$ with $\alpha_0 = \beta_1 = 0$, $\alpha_1 = \beta_0 = 1$; FM time ($t=0$ noise → $t=1$ data, the reverse of [[DDPM]] time).

## The Score Function
---
For any distribution $q$ with density, the **score** function is

$$
\nabla_x \log q(x)
$$

a vector field pointing in the **direction of steepest ascent of log-likelihood**, i.e. "toward the data," with magnitude growing in low-density regions. 

Two properties make it the preferred handle on a distribution:
- **Normalization-free**: $\nabla \log \frac{e^{-f_\theta(x)}}{Z_\theta} = -\nabla f_\theta(x)$ — the intractable constant $Z_\theta$ drops out, the original motivation for score matching in energy-based models (Hyvärinen; see [[Score-Based Generative Modeling Methods]]).
- **Closed-form along Gaussian paths** (below), which is what diffusion training exploits.

**Conditional and marginal scores.** For a conditional path $p_t(x|z)$ and its marginal $p_t(x) = \int p_t(x|z)\,p_\text{data}(z)\,dz$, define the **conditional score** $\nabla \log p_t(x|z)$ and **marginal score** $\nabla \log p_t(x)$. The marginal score is a posterior-weighted mixture of conditional scores (6.S184 Eq. 38):

$$
\nabla \log p_t(x) = \int \nabla \log p_t(x|z)\, \frac{p_t(x|z)\, p_\text{data}(z)}{p_t(x)}\, dz
$$

— *the same structure as the marginal velocity field* $u_t^\text{target}$ in [[Flow Matching]]. One-line proof: $\nabla \log p_t = \frac{\nabla p_t}{p_t}$, push $\nabla$ inside the mixture integral, and use $\nabla p_t(x|z) = p_t(x|z)\nabla\log p_t(x|z)$.

**Gaussian paths (6.S184 Ex. 15).** For $p_t(x|z) = \mathcal{N}(x;\, \alpha_t z,\, \beta_t^2 I_d)$:

$$
\nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}
\;\;\overset{x = \alpha_t z + \beta_t\epsilon}{=}\;\;
-\frac{\epsilon}{\beta_t},
\qquad
\nabla \log p_t(x) = -\frac{\hat\epsilon_t(x)}{\beta_t},
$$

where $\hat\epsilon_t(x) = \mathbb{E}[\epsilon\,|\,x_t = x]$: the conditional score is the (negative, rescaled) **noise that corrupted $z$**, and the marginal score is the rescaled posterior-mean noise.

## Score Matching Losses
---
Train a **score network** $s^\theta : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ to approximate $\nabla\log p_t$:

$$
\mathcal{L}_\text{SM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_\text{data},\, x \sim p_t(\cdot|z)} \big\|s_t^\theta(x) - \nabla \log p_t(x)\big\|^2
\qquad \text{(score matching loss — intractable)}
$$

$$
\mathcal{L}_\text{CSM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_\text{data},\, x \sim p_t(\cdot|z)} \big\|s_t^\theta(x) - \nabla \log p_t(x|z)\big\|^2
\qquad \text{(conditional / denoising score matching)}
$$

**Theorem (6.S184 Thm 22).** $\mathcal{L}_\text{SM}(\theta) = \mathcal{L}_\text{CSM}(\theta) + C$ with $C$ independent of $\theta$, so their gradients coincide, and the minimizer is $s_t^{\theta^*} = \nabla \log p_t$.

The proof is *identical* to the CFM theorem in [[Flow Matching]] (their Thm 12): because the marginal score has the same posterior-mixture form as the marginal velocity, expanding the square and swapping integrals works verbatim. Both are instances of one lemma — **regressing on tractable single-sample targets learns their conditional expectation** (the Bregman-divergence principle in the FM note).

## Denoising Score Matching = the DDPM Loss
---
Instantiating $\mathcal{L}_\text{CSM}$ on the Gaussian path (6.S184 Ex. 23):

$$
\mathcal{L}_\text{CSM}(\theta)
= \mathbb{E}_{t,\, z,\, \epsilon}\Big\|s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t}\Big\|^2
= \mathbb{E}_{t,\, z,\, \epsilon}\,\frac{1}{\beta_t^2}\big\|\beta_t\, s_t^\theta(\alpha_t z + \beta_t \epsilon) + \epsilon\big\|^2 ,
$$

so the network literally learns to predict the corrupting noise — hence "**denoising** score matching" (Vincent, 2011). The $\tfrac{1}{\beta_t^2}$ weight **blows up as $\beta_t \to 0$** (near the data end $t \to 1$; equivalently, DSM only works with enough noise). The [[DDPM]] fix (Ho et al., 2020): drop the weight and reparameterize into a **noise-prediction network** $\epsilon_t^\theta(x) = -\beta_t\, s_t^\theta(x)$:

$$
\mathcal{L}_\text{DDPM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_\text{data},\, \epsilon \sim \mathcal{N}(0, I_d)}\big\|\epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon\big\|^2
$$

— exactly $\mathcal{L}_\text{simple}$ of [[DDPM]], written in FM time. The training loop (6.S184 Alg. 4) mirrors the FM one, with a different target:

```python
# score matching training step (Gaussian path)
z = sample_data(); eps = torch.randn_like(z); t = torch.rand(b, 1)
x_t  = alpha(t) * z + beta(t) * eps
loss = ((eps_model(x_t, t) - eps)**2).mean()          # noise-prediction form
# or: ((s_model(x_t, t) + eps / beta(t))**2).mean()   # score form (unstable near t=1)
```

## Score ↔ Velocity: Two Sides of One Model
---
For Gaussian paths, both the conditional velocity and the conditional score are **linear in $(x, z)$**, so they interconvert (6.S184 Prop. 1):

$$
u_t^\text{target}(x) = a_t\, \nabla\log p_t(x) + b_t\, x,
\qquad
a_t = \beta_t^2 \frac{\dot\alpha_t}{\alpha_t} - \dot\beta_t \beta_t, \quad b_t = \frac{\dot\alpha_t}{\alpha_t},
$$

(and identically for the conditional pair). Consequences:
- **Train either one.** A velocity model contains the score for free and vice versa — SD3 trains velocity, DDPMs train $\epsilon$/score; same information, different parameterization.
- **Any posterior-mean surrogate works** (6.S184 Rem. 16): denoiser $D_t(x) = \mathbb{E}[z|x_t = x]$, noise-prediction $\hat\epsilon_t$, score, velocity — full conversion table in [[Flow Matching]].
- **Singularities differ**: the score $-\hat\epsilon_t/\beta_t$ and the DSM weight blow up at $\beta_t \to 0$ (data end, $t \to 1$), while the score→velocity coefficients $(a_t, b_t)$ blow up at $\alpha_t \to 0$ (noise end, $t \to 0$). Direct velocity prediction avoids both — one practical argument for the FM parameterization.

## Why Learn the Score? Stochastic Sampling and Guidance
---
**The SDE sampling family (6.S184 Thm 17 / Summary 24).** For *every* diffusion coefficient $\sigma_t \ge 0$,

$$
X_0 \sim p_\text{init}, \qquad
dX_t = \Big[u_t^\text{target}(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\Big]dt + \sigma_t\, dW_t
\;\;\Rightarrow\;\; X_t \sim p_t .
$$

The score term is exactly **Langevin dynamics along the path** — divergence-free w.r.t. $p_t$ ([[Divergence Theorem]]), so it injects noise without changing any marginal. $\sigma_t = 0$ recovers the deterministic flow (probability-flow ODE); a *static* path recovers pure Langevin [[MCMC]] (the sampler of [[Score-Based Generative Modeling Methods]]). For Gaussian paths the drift can be written purely in the score (6.S184 Eq. 56): $dX_t = [(a_t + \tfrac{\sigma_t^2}{2})\, s_t^\theta(X_t) + b_t X_t]\,dt + \sigma_t dW_t$, simulated with Euler–Maruyama. In theory every $\sigma_t$ samples $p_\text{data}$; in practice an **empirically optimal $\sigma_t$** exists only because of training and discretization error.

**Guidance needs scores.** Classifier guidance adds $w\, a_t \nabla \log p_t(y|x)$ to the velocity, and classifier-free guidance rests on the Bayes decomposition $\nabla\log p_t(x|y) = \nabla\log p_t(x) + \nabla\log p_t(y|x)$ — see the guidance section of [[Flow Matching]].

## The Score Matching Family (beyond 6.S184)
---
| Variant | Objective | Trick | Covered in |
|---|---|---|---|
| **Implicit SM** ([Hyvärinen, 2005](https://jmlr.org/papers/v6/hyvarinen05a.html)) | $\mathbb{E}_p[\tfrac{1}{2}\|s_\theta\|^2 + \mathrm{div}(s_\theta)]$ | integration by parts / Stein's identity | [[Divergence Theorem]] |
| **Sliced SM** ([Song et al., 2019](https://arxiv.org/abs/1905.07088)) | random projections of ISM | Hutchinson-style $v^\top \partial_x s_\theta v$ | — |
| **Denoising SM** (Vincent, 2011) | $\mathcal{L}_\text{CSM}$ above | regress on the *conditional* score | this note |
| **Annealed / multi-scale** (NCSN, [Song & Ermon, 2019](https://arxiv.org/abs/1907.05600)) | DSM over noise levels + annealed Langevin | large noise fixes low-density score estimates | [[Score-Based Generative Modeling Methods]] |

DSM won at scale because it needs **no divergence term** — ISM's $\mathrm{div}(s_\theta)$ is a Hessian trace of $\log p_\theta$, expensive in high dimension ([[Divergence Theorem]]) — and because it plugs directly into the probability-path/SDE view that unifies it with [[Flow Matching]].

## Relationship to Other Notes
---
- [[Flow Matching]] — the same recipe with velocities instead of scores; 6.S184 Prop. 1 converts between them; the SM/CSM and FM/CFM theorems are the same lemma
- [[Score-Based Generative Modeling Methods]] — the broader score-based picture: energy-based models, Langevin and annealed Langevin sampling, the SDE unification; this note is the training-algorithm core in FM time
- [[DDPM]] — $\mathcal{L}_\text{DDPM}$ above **is** $\mathcal{L}_\text{simple}$; the diffusion-time relation $\epsilon_\theta = -\sqrt{1-\bar\alpha_t}\,s_\theta$ is the FM-time $\epsilon_t^\theta = -\beta_t s_t^\theta$
- [[Divergence Theorem]] — Hyvärinen's implicit SM comes from integration by parts; the Langevin term is divergence-free w.r.t. $p_t$, which is why SDE sampling preserves the path
- [[MCMC]] — Langevin dynamics as the stochastic component of diffusion sampling and as a standalone sampler given a learned score
