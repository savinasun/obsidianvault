### [[dl_moc]]  
# **Flow Matching**

**Tags**:: #generative #dl_theo #diffusion
**Links**:: [[Generative Modeling]] · [[DDPM]] · [[Score Matching]] · [[Score-Based Generative Modeling Methods]] · [[Divergence Theorem]] · [[LDM]] · [[DiT]] · [[VAE]] · [[ELBO]] · [[MCMC]] · [[FID]] · [[RAE]] · [[VLA]]


**Flow Matching (FM)** is a *simulation-free* framework for training generative models: pick a probability path $p_t$ interpolating a source distribution (noise) $p_\text{init}$ into the data distribution $p_\text{data}$, train a neural velocity field to generate that path by simple regression, and sample by ODE integration. Introduced concurrently by [Lipman et al., 2022](https://arxiv.org/abs/2210.02747) (Flow Matching), [Liu et al., 2022](https://arxiv.org/abs/2209.03003) (Rectified Flow), and [Albergo & Vanden-Eijnden, 2022](https://arxiv.org/abs/2209.15571) (stochastic interpolants), it now powers Stable Diffusion 3 ([Esser et al., 2024](https://arxiv.org/abs/2403.03206)), Movie Gen (video), Voicebox/Audiobox (speech/audio), SE(3) FM (protein backbones), and π0 (robot policies — [[VLA]]).

This note follows **"Flow Matching Guide and Code"** ([Lipman et al., 2024, arXiv 2412.06264](https://arxiv.org/abs/2412.06264), with the [`flow_matching` library](https://github.com/facebookresearch/flow_matching)) and, for the background, the **[MIT 6.S184](https://diffusion.csail.mit.edu/2026/index.html) lecture notes** ([Holderrieth & Erives](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)). 

## Background
---
Source $X_0 \sim p_\text{init} = \mathcal{N}(0, I_d)$, data point $z\sim p_\text{data}$; noise $\epsilon \sim \mathcal{N}(0, I_d)$; schedulers $(\alpha_t, \beta_t)$; target fields $u_t^\text{target}(x)$, $u_t^\text{target}(x|z)$; network $u_t^\theta$.  **FM time runs noise → data** ($t=0$ noise, $t=1$ data) — the *reverse* of [[DDPM]] time, related by a decreasing reparameterization $r = k(t)$. The FM guide writes $p, q$; $x_1$; $x_0$; $(\alpha_t, \sigma_t)$; $u_t(x|x_1)$ — translate when reading it.

### Probability Paths
A **probability path** $(p_t)_{0\le t\le 1}$ is a time-indexed family of distributions — a trajectory in distribution space, prescribing what law $X_t$ should have at each time.

**<span style="color:rgb(255, 255, 0)"><b>Conditional probability path.</b></span>** Interpolates noise into a *single data point* $z$, ending at the Dirac $\delta_z$:
$$
p_0(\cdot|z) = p_\text{init}, \qquad p_1(\cdot|z) = \delta_z \qquad \forall z .
$$

**<span style="color:rgb(255, 255, 0)"><b>Marginal probability path.</b></span>** Mix over the data: $p_t(x) = \int p_t(x|z)\, p_\text{data}(z)\, dz$, interpolating $p_\text{init} \to p_\text{data}$. Sampling from $p_t$ is easy ($z \sim p_\text{data}$, then $x \sim p_t(\cdot|z)$) even though its density is intractable. E.g. the **Gaussian path** with schedulers $(\alpha_t, \beta_t)$, $\alpha_0 = \beta_1 = 0$, $\alpha_1 = \beta_0 = 1$:
$$
p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\, \beta_t^2 I_d), \qquad x_t = \alpha_t z + \beta_t \epsilon .
$$

### Flow Models
**<span style="color:rgb(255, 255, 0)"><b>Vector field.</b></span>** A time-dependent **vector field** $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ defines an **ODE**; its solution from a start point is a <span style="color:rgb(255, 255, 0)"><b>trajectory</b></span> $t \mapsto X_t$:
$$
\frac{d}{dt}X_t = u_t(X_t), \qquad X_0 = x_0 .
$$

<span style="color:rgb(255, 255, 0)"><b><span style="color:rgb(255, 255, 0)">Flow.</b></span></span> Collecting solutions from *every* start point gives the **flow** $\psi_t$, with $X_t = \psi_t(X_0)$:
$$
\frac{d}{dt}\psi_t(x_0) = u_t(\psi_t(x_0)), \qquad \psi_0(x_0) = x_0 .
$$
Conversely $u_t(x) = \dot\psi_t(\psi_t^{-1}(x))$. Vector fields define ODEs whose solutions are flows.

<span style="color:rgb(255, 255, 0)"><b>Flow model.</b></span> Make *only the initial condition* random: $$X_0 \sim p_\text{init},\quad \frac{d}{dt}X_t = u_t^\theta(X_t)$$goal $X_1 = \psi_1^\theta(X_0) \sim p_\text{data}$. The network parameterizes the **vector field, never the flow** — evaluating $\psi_t^\theta$ always means simulating the ODE.

### Lipschitz Condition
<span style="color:rgb(255, 255, 0)"><b>Lipschitz condition.</b></span> $u_t$ is globally **Lipschitz** if $\|u_t(x) - u_t(y)\| \le L\,\|x - y\|$ for all $x, y$.

<span style="color:rgb(255, 255, 0)"><b>Picard–Lindelöf theorem.</b></span> If $u_t(x)$ is continuous in $t$ and Lipschitz in $x$ (constant $L$ uniform in $t$), the ODE has exactly one solution through every initial condition — on all of $[0,1]$ in the globally Lipschitz case; only on a maximal interval (possible finite-time blow-up) if merely locally Lipschitz.
	Both hypotheses earn their keep: mere continuity keeps existence (Peano) but **uniqueness fails** ($\dot x = \sqrt{|x|},\ x(0)=0$ is solved by $x \equiv 0$ *and* $x = t^2/4$); local-only Lipschitzness allows **blow-up** ($\dot x = x^2,\ x(0)=1$ gives $x = 1/(1-t)$, gone at $t=1$).

<span style="color:rgb(255, 255, 0)"><b>Diffeomorphism</b></span> (微分同胚): if $u_t$ is $C^1$ with bounded derivative (⟹ Lipschitz, $L = \sup\|\partial_x u\|$), the flow exists, is unique, and $\psi_t$ is a **diffeomorphism**. 
	<span style="color:rgb(0, 176, 240)">Uniqueness is what makes flows well-behaved: two trajectories can never cross or merge at the same time</span> — they would be two solutions of one initial-value problem — <span style="color:rgb(0, 176, 240)">which is why</span> $\psi_t$ <span style="color:rgb(0, 176, 240)">is a bijection</span>, and why the *marginal* FM velocity must average conditional velocities wherever conditional paths cross. Neural networks with Lipschitz activations satisfy this automatically. 
	The FM guide's weaker hypothesis — *local* Lipschitzness + integrability $\int_0^1 \mathbb{E}\|u_t(X_t)\|\,dt < \infty$ — rules out blow-up for almost every trajectory instead, which is what's needed near $t=1$ where conditional velocities like $(z-x)/(1-t)$ blow up (hence "generates" on $[0,1)$).

### Numerical Simulation
<span style="color:rgb(255, 255, 0)"><b>Euler method.</b></span> $X_{t+h} = X_t + h\,u_t(X_t)$, $h = 1/n$ ($o(h)$ local error). 
<span style="color:rgb(255, 255, 0)"><b>Heun method.</b></span> <span style="color:rgb(255, 255, 0)">(predictor–corrector)</span> predict $X'_{t+h} = X_t + h\,u_t(X_t)$, then average velocities, $X_{t+h} = X_t + \frac{h}{2}\big(u_t(X_t) + u_{t+h}(X'_{t+h})\big)$ — second-order, like the midpoint method used in the guide's library.

### From ODEs to SDEs
**Stochastic process, diffusion coefficient, SDE.** A **stochastic process** $(X_t)$ is a random trajectory. An **SDE** adds Brownian noise, scaled by a **diffusion coefficient** $\sigma_t \ge 0$:
$$
dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t,
$$
shorthand for $X_{t+h} = X_t + h\,u_t(X_t) + \sigma_t(W_{t+h} - W_t) + h\,R_t(h)$. Simulate with **Euler–Maruyama**: $X_{t+h} = X_t + h\,u_t(X_t) + \sqrt{h}\,\sigma_t\,\epsilon_t$. Randomness enters the dynamics, so there is no flow map. **Every ODE is an SDE with $\sigma_t = 0$** — a "diffusion model" is literally a flow model plus a diffusion coefficient.

### Vector Fields
<span style="color:rgb(255, 255, 0)"><b>Conditional vector field.</b></span> Any field whose ODE transports $p_\text{init}$ along the conditional path: $X_0 \sim p_\text{init},\ \frac{d}{dt}X_t = u_t^\text{target}(X_t|z) \Rightarrow X_t \sim p_t(\cdot|z)$. Typically derivable *by hand*: the Gaussian conditional flow $\psi_t(x|z) = \alpha_t z + \beta_t x$ gives (6.S184 Ex. 10)
$$
u_t^\text{target}(x|z) = \Big(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\Big) z + \frac{\dot\beta_t}{\beta_t}\, x .
$$
Alone it just collapses everything onto $z$ — but it is the building block of what we actually train.

<span style="color:rgb(255, 255, 0)"><b>Marginal vector field.</b></span> The posterior-weighted average of conditional fields — for each $z$, the direction toward $z$, weighted by how much we believe $x$ came from $z$:
$$
u_t^\text{target}(x) = \int u_t^\text{target}(x|z)\, \frac{p_t(x|z)\,p_\text{data}(z)}{p_t(x)}\, dz = \mathbb{E}\big[u_t^\text{target}(x_t|z)\,\big|\,x_t = x\big].
$$
This is the training target: its flow turns noise into data.

<span style="color:rgb(255, 255, 0)"><b>Marginalization trick.</b></span> If $u_t^\text{target}(\cdot|z)$ generates $p_t(\cdot|z)$, then the marginal $u_t^\text{target}$ generates the marginal $p_t$ (under mild regularity: C¹ paths/velocities, bounded data support, $p_t > 0$). Proof: plug the mixture into the Continuity Equation and swap integral/divergence. The guide generalizes the conditioning from $z$ to any $Z$ — source samples or two-sided pairs $(x_0, z)$ — all essentially equivalent.

### The Continuity Equation and Fokker–Planck Equation
<span style="color:rgb(255, 255, 0)"><b>Continuity Equation (Mass Conservation).</b></span> Say $u_t$ **generates** $p_t$ if $X_t \sim p_t$ for all $t$ whenever $X_0 \sim p_0$. This holds **iff**
$$
\partial_t\, p_t(x) = -\,\mathrm{div}\big(p_t u_t\big)(x) \qquad \forall x, t .
$$
**Intuition**: $\partial_t p_t$ is the local change of probability; $-\mathrm{div}(p_t u_t)$ is the **net inflow** of probability mass carried by the flux $j_t = p_t u_t$; conservation of total mass forces them equal — the [[Divergence Theorem]] in differential form. This single PDE licenses everything above.

<span style="color:rgb(255, 255, 0)"><b>Fokker–Planck (SDE generalization).</b></span> For $dX_t = u_t\,dt + \sigma_t\,dW_t$,
$$
X_t\sim p_t\quad \mathrm{iff.}\quad \partial_t\, p_t = -\,\mathrm{div}(p_t u_t) + \frac{\sigma_t^2}{2}\Delta p_t ,
$$
with the Laplacian $\Delta = \nabla\cdot \nabla = \mathrm{div}\,\nabla$ as heat-equation term; $\sigma_t = 0$ recovers the Continuity Equation. Since $\frac{\sigma_t^2}{2}\Delta p_t = \mathrm{div}(p_t \frac{\sigma_t^2}{2}\nabla\log p_t)$, the SDE
$$
dX_t = \Big[u_t(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\Big]dt + \sigma_t\,dW_t
$$
has the **same marginals for every $\sigma_t \ge 0$** — Langevin dynamics ([[MCMC]]) is the static-path special case. This is where diffusion's stochastic samplers come from; the "optimal" $\sigma_t$ exists only because of training and simulation error. See [[Score Matching]].

### Exact Likelihoods
The Continuity Equation implies the *instantaneous change of variables* ([Chen et al., 2018](https://arxiv.org/abs/1806.07366); guide §3.6):
$$
\frac{d}{dt}\log p_t(\psi_t(x)) = -\,\mathrm{div}(u_t)(\psi_t(x)),
$$
so $\log p_1$ is an integral of the divergence along the trajectory; Hutchinson's estimator $\mathrm{div}(u_t)(x) = \mathbb{E}_Z\,\mathrm{tr}[Z^\top \partial_x u_t(x) Z]$ makes it an **unbiased** estimate at one vector–Jacobian product per step — an advantage over stochastic processes, which only admit lower bounds ([[ELBO]]). Classical CNFs trained by maximum likelihood needed this ODE *inside* the training loop — exactly what FM eliminates.

## FM and Conditional FM Losses
---
Regress the network onto the *intractable* marginal field, then swap in the tractable conditional target:
$$
\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, x \sim p_t} \big\|u_t^\theta(x) - u_t^\text{target}(x)\big\|^2
\qquad \text{(flow matching loss — intractable)}
$$
$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_\text{data},\, x \sim p_t(\cdot|z)} \big\|u_t^\theta(x) - u_t^\text{target}(x|z)\big\|^2
\qquad \text{(conditional flow matching)}
$$$$\mathcal{L}_\text{FM}(\theta) = \mathcal{L}_\text{CFM}(\theta) + \text{const} \qquad \text{(relationship)}$$so gradients coincide, and the minimizer is $u_t^{\theta^*} = u_t^\text{target}$.

<span style="color:rgb(255, 255, 0)"><b>Bregman divergences.</b></span> The gradient equality holds for any **Bregman divergence** $$D(u,v) = \Phi(u) - \Phi(v) - \langle u - v, \nabla\Phi(v)\rangle$$ ($\Phi$ strictly convex; MSE is $\Phi = \|\cdot\|^2$): their gradients are affine-invariant in the second argument, so $\nabla_v D(\mathbb{E}[Y], v) = \mathbb{E}[\nabla_v D(Y, v)]$. General principle: for any $f_t(x_0, z)$, minimizing $\mathbb{E}\, D(f_t(x_0, z), g_t^\theta(x_t))$ yields $g_t^\theta(x) = \mathbb{E}[f_t(x_0, z) \mid x_t = x]$ — **regression on single-sample targets learns their conditional expectation**. Bregman-ness is *necessary* for this (Holderrieth et al., 2024).
 
**Time weighting.** Training with $t \sim \omega(t)$ (equivalently, reweighting) changes nothing mathematically, but non-uniform sampling (e.g. SD3's logit-normal) works better at scale.

## Designing Paths via Conditional Flows
---
Rather than hand-designing $p_t(\cdot|z)$, pick a **conditional flow** $\psi_t(x|z)$ with $\psi_0(x|z) = x$, $\psi_1(x|z) = z$ (smooth, diffeomorphic in $x$). Then $X_t = \psi_t(X_0|z)$ defines both path and velocity, and the CFM loss needs no densities — the target is just the **time-derivative of the interpolant**:

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\,(x_0,\,z) \sim \pi}\, \big\|u_t^\theta(x_t) - \dot\psi_t(x_0|z)\big\|^2 ,
$$

with $\pi$ the source–data coupling (independent by default — see *Data Couplings*).

**The conditional-OT (linear) path.** Minimizing a Jensen upper bound on the kinetic energy $\int_0^1 \mathbb{E}\|u_t^\text{target}(X_t)\|^2 dt$ over conditional flows gives straight lines (Euler–Lagrange: $\ddot\gamma_t = 0$):

$$
\psi_t(x|z) = t\,z + (1-t)\,x
\;\;\Rightarrow\;\;
x_t = t z + (1-t)\epsilon, \quad
u_t^\text{target}(x|z) = \frac{z - x}{1-t}, \quad
\dot\psi_t(\epsilon|z) = z - \epsilon,
$$

i.e. the Gaussian path with $\alpha_t = t$, $\beta_t = 1-t$, and the simplest FM loss $\mathbb{E}\|u_t^\theta(x_t) - (z - \epsilon)\|^2$. Training loop (6.S184 Alg. 3):

```python
# flow matching training step (cond-OT path)
z = sample_data(); eps = torch.randn_like(z); t = torch.rand(b, 1)
x_t  = t * z + (1 - t) * eps
loss = ((model(x_t, t) - (z - eps))**2).mean()
```

Facts worth remembering:
- If $p_\text{data} = \delta_z$, the linear conditional flow **is** the exact dynamic optimal transport.
- In general the *marginal* flow is **not** OT — the linear choice only minimizes a bound; marginal kinetic energy ≤ coupling cost $\mathbb{E}\|z - x_0\|^2$.
- **Conditional** straight lines ≠ **marginal** straight trajectories: with independent couplings the conditional lines cross, so marginal paths curve — what multisample couplings and reflow address (below).

## Affine Paths, Schedulers, and Parameterizations
---
Generalize to **affine conditional flows** $\psi_t(x|z) = \alpha_t z + \beta_t x$ with scheduler $(\alpha_t, \beta_t)$: $\alpha_0 = 0,\ \alpha_1 = 1,\ \beta_0 = 1,\ \beta_1 = 0$, $\dot\alpha_t > 0 > \dot\beta_t$; CFM target $\dot\alpha_t z + \dot\beta_t \epsilon$. Examples: cond-OT ($t,\ 1-t$), polynomial ($t^n$), LinearVP ($t,\ \sqrt{1-t^2}$), cosine ($\sin\tfrac{\pi t}{2},\ \cos\tfrac{\pi t}{2}$). Diffusion's VP ($\alpha_t^2 + \beta_t^2 = 1$, exponential form) and VE ($\alpha_t \equiv 1$, $\beta_0 \gg 1$) are approximate members — neither hits an exact Gaussian at $t=0$.

**Equivalent parameterizations.** From $u_t^\text{target}(x) = \dot\alpha_t\,\mathbb{E}[z|x_t{=}x] + \dot\beta_t\,\mathbb{E}[\epsilon|x_t{=}x]$:
- **denoiser / $z$-prediction** ($D_t(x) := \mathbb{E}[z|x_t = x]$, 6.S184 Rem. 16): $u_t^\text{target}(x) = \frac{\dot\beta_t}{\beta_t}x + \big(\dot\alpha_t - \alpha_t\frac{\dot\beta_t}{\beta_t}\big)D_t(x)$
- **noise / $\epsilon$-prediction** ($\hat\epsilon_t(x) := \mathbb{E}[\epsilon|x_t = x]$): $u_t^\text{target}(x) = \frac{\dot\alpha_t}{\alpha_t}x + \big(\dot\beta_t - \beta_t\frac{\dot\alpha_t}{\alpha_t}\big)\hat\epsilon_t(x)$
- **score** (Gaussian paths, below).

All interconvert in closed form (guide Table 1; any posterior-mean surrogate works), and each is learnable by conditional matching ($f_t(\epsilon, z) = z$, $\epsilon$, …). Caveat — **endpoint singularities**: denoiser blows up as $t \to 1$, ε-pred/score as $t \to 0$; direct velocity prediction avoids both.

**Gaussian paths and the score.** $\nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}$, $\nabla \log p_t(x) = -\hat\epsilon_t(x)/\beta_t$ — the score is rescaled noise-prediction — and velocity ↔ score convert (6.S184 Prop. 1):

$$
u_t^\text{target}(x) = a_t\,\nabla \log p_t(x) + b_t\, x, \qquad
a_t = \beta_t^2\frac{\dot\alpha_t}{\alpha_t} - \dot\beta_t\beta_t, \quad
b_t = \frac{\dot\alpha_t}{\alpha_t}.
$$

A trained velocity model contains the score for free and vice versa ([[Score Matching]]); this form is also kinetically optimal for the fixed Gaussian path (guide §4.8.3).

**Post-training scheduler change (guide).** Affine paths with the same coupling are related by a **scale-time transformation** $\bar\psi_r = s_r\,\psi_{t_r}$, matched via the SNR $\rho_t = \alpha_t/\beta_t$ ($t_r = \rho^{-1}(\bar\rho(r))$), giving $\bar u_r(x) = \frac{\dot s_r}{s_r}x + s_r \dot t_r\, u_{t_r}(x/s_r)$. So a scheduler can be **swapped at inference** (e.g. VP → cond-OT), and *all schedulers yield the same final map* $\psi_1$ in continuous time — the choice matters only through discretization error and learning dynamics.

## Data Couplings
---
FM trains on pairs $(x_0, z) \sim \pi$; the default is the **independent coupling** $\pi = p_\text{init} \otimes p_\text{data}$, but this is a design axis:

- **Paired data** ([I²SB, Liu et al., 2023](https://arxiv.org/abs/2302.05872); Albergo et al., 2024): for inpainting/super-resolution/deblurring, set $\pi(x_0, z) = \pi(x_0|z)\,p_\text{data}(z)$ by corrupting data samples (+ noise so the source has a density) — FM learns the restoration map directly, no guidance needed.
- **Multisample / minibatch-OT** ([Pooladian et al., 2023](https://arxiv.org/abs/2304.14772); [Tong et al., 2023](https://arxiv.org/abs/2302.00482)): solve the OT assignment within each batch and train on matched pairs — provably lowers transport cost and **straightens marginal trajectories** (fewer ODE steps); approaches true OT as batch size $k \to \infty$.
- **Reflow** (Rectified Flow, [Liu et al., 2022](https://arxiv.org/abs/2209.03003); beyond the guide): re-couple with model-generated pairs $(x_0, \hat z)$, iteratively straightening toward 1–2-step generation — behind SD3's and FLUX's fast variants.

## Guidance
---
Train a **guided velocity** $u_t^\theta(x|y)$ with the same CFM loss ("vanilla guidance", 6.S184 §5). To *reinforce* the prompt, use the score conversion and Bayes' rule $\nabla\log p_t(x|y) = \nabla\log p_t(x) + \nabla\log p_t(y|x)$:

- <span style="color:rgb(255, 255, 0)"><b>Classifier guidance</b></span>: $\tilde u_t(x|y) = u_t^\text{target}(x) + w\,a_t \nabla \log p_t(y|x)$ — scale up a noised-data classifier's gradient.
- <span style="color:rgb(255, 255, 0)"><b>Classifier-free guidance</b></span> ([Ho & Salimans, 2021](https://arxiv.org/abs/2207.12598); for flows [Zheng et al., 2023](https://arxiv.org/abs/2311.13443)) eliminates the classifier:

$$
\tilde u_t^\theta(x|y) = (1 - w)\, u_t^\theta(x|\varnothing) + w\, u_t^\theta(x|y),
$$

with guidance scale $w > 1$ and condition dropout $\xi \sim \text{Bernoulli}(p_\text{uncond})$ during training. For $w \ne 1$ this is a heuristic — not the true guided field, and what distribution it samples is theoretically murky — yet it is the standard for large-scale text-to-image FM (SD3).

## Diffusion Models as a Special Case
---
The guide's unifying claim (§10): **a diffusion model = an FM model with a specific Gaussian path + score parameterization + inverted time.**

- The **forward noising SDE** $dX_r = a_r X_r\,dr + g_r\,dW_r$ (affine drift → Gaussian marginals) merely *constructs* the path $p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, with $\alpha, \beta$ determined by $a, g$.
- **Denoising score matching** ([[Score Matching]]; the DDPM loss) = conditional matching for ε-prediction with $s_t^\theta = -\hat\epsilon_t^\theta / \beta_t$; minimizer $\nabla \log p_t$.
- The **probability-flow ODE** ([Song et al., 2021](https://arxiv.org/abs/2011.13456)) = the FM marginal velocity written via the score (Prop. 1); deterministic diffusion sampling = FM sampling.
- **Stochastic (SDE) sampling** = adding Langevin dynamics $\tfrac{\sigma_t^2}{2} \nabla \log p_t\,dt + \sigma_t dW_t$ — divergence-free w.r.t. $p_t$, so any $\sigma_t \ge 0$ preserves marginals (Thm 17). Anderson time-reversal is one particular $\sigma_t$; with model/discretization error an empirically tuned $\sigma_t$ can help (Karras et al., 2022).
- **Time-reversal is unnecessary**: "reverse the forward marginals" ≡ "find a velocity generating $p_t$" ≡ "solve the Continuity/Fokker–Planck equation" — FM solves the third directly.

| | Diffusion ([[DDPM]]) | Flow Matching (cond-OT) |
|---|---|---|
| **Path construction** | forward noising SDE, closed-form marginals | direct choice of conditional flow |
| **Network target** | noise $\epsilon$ / score | velocity $z - \epsilon$ (or any parameterization) |
| **Boundary at noise end** | Gaussian only as $r \to \infty$ (approx.) | exactly $p_\text{init}$ at $t = 0$ |
| **Schedule** | required, baked into training | swappable post-training |
| **Sampling** | reverse SDE or prob-flow ODE | ODE (SDE optional via Langevin) |
| **Likelihood** | ELBO (SDE) / exact (ODE) | exact, unbiased estimator |
| **Endpoint singularities** | score/ε-pred singular near noise | velocity prediction non-singular |

Practical upshot: straighter conditional paths → fewer function evaluations (Euler error scales with trajectory curvature); perfectly straight marginal flows would sample in one step — the motivation for OT couplings and reflow.

## Extension: Riemannian Flow Matching
---
FM transfers to Riemannian manifolds $\mathcal{M}$ (sphere, tori, SE(3) — proteins, molecules, geoscience) ([Chen & Lipman, 2024](https://arxiv.org/abs/2302.03660)): velocities live in tangent spaces, the continuity equation uses the Riemannian divergence, and the Marginalization Trick + CFM equivalence hold verbatim with tangent-space Bregman divergences. Straight lines become **geodesic interpolation**:

$$
\psi_t(x|z) = \exp_{x}\!\big(\kappa(t)\, \log_{x}(z)\big), \qquad \kappa(0)=0,\ \kappa(1)=1,
$$

**simulation-free** whenever exp/log maps are closed-form — a real advantage over Riemannian diffusion (in-training simulation, approximate scores). For hard geometries, replace geodesics with any smooth **premetric** $d(x,z)$ (e.g. spectral distances), with minimal-norm velocity $u_t^\text{target}(x|z) = \frac{d \log \bar\kappa(t)}{dt}\, d(x, z) \frac{\nabla d(x, z)}{\|\nabla d(x, z)\|_g^2}$.

## Extension: Discrete Flow Matching (CTMC)
---
For discrete data $x = (x^1, \dots, x^d) \in \mathcal{V}^d$ (tokens, vocabulary size $V$; 6.S184 §7), the flow becomes a **Continuous-Time Markov Chain** ([[CTMC]]; [Campbell et al., 2024](https://arxiv.org/abs/2402.04997); [Gat et al., 2024](https://arxiv.org/abs/2407.15595)): transitions $p_{t+h|t}(y|x) = \delta_x(y) + h\,Q_t(y|x) + o(h)$ with a **rate matrix** $Q_t$ ($Q_t(y|x) \ge 0$ for $y \ne x$, $\sum_y Q_t(y|x) = 0$). The Continuity Equation becomes the **Kolmogorov equation** $\frac{d}{dt}p_t(y) = \sum_x Q_t(y|x)\,p_t(x)$, and the whole FM recipe transfers.

- **Factorized rates**: all-$V^d$-states rates are infeasible; allow only single-token transitions, parameterized per position as $Q_t(v, i \mid x)$ — output dimension $d \cdot V$; per-token parallel Euler sampling is valid to $o(h)$ (6.S184 Alg. 7).
- **Mixture paths**: per token, $X_t^i = z^i$ w.p. $\kappa_t$, else $x_0^i$; conditional rates $Q_t(v, i \mid x; x_0, z) = \frac{\dot\kappa_t}{1-\kappa_t}\big[\delta_{z^i}(v) - \delta_{x^i}(v)\big]$. Source: uniform noise or a **mask token** (mask-DFM = discrete noise→data).
- **Posterior parameterization**: learn a per-token denoiser $p_{1|t}^{\theta,i}(z^i | x)$ (discrete analog of $D_t$); train with cross-entropy or **generalized KL**, which doubles as an [[ELBO]] for evaluation.
- **Sampling**: per token, draw $z^i \sim p^{\theta,i}_{1|t}(\cdot|x)$, then a CTMC Euler step (jump-to-target vs. stay); divergence-free rate components add corrector-style noise (un-/re-masking trade-offs).

This is the FM view of "discrete diffusion" language models and multimodal protein co-design — and, over tokenized action chunks, of robot policies. Full treatment: [[Discrete Flow Matching]].

## Extension: Generator Matching — One Framework for Everything
---
**Generator Matching (GM)** ([Holderrieth et al., 2024](https://arxiv.org/abs/2410.20587)) abstracts FM to *any* Continuous-Time Markov Process on *any* state space via the **generator** $\mathcal{L}_t$, the first-order derivative of the transition kernel on test functions: $\mathbb{E}[f(X_{t+h})|X_t = x] = f(x) + h\,[\mathcal{L}_t f](x) + o(h)$.

| Process | Generator $\mathcal{L}_t f$ | Parameterized by | KFE ≙ |
|---|---|---|---|
| Flow (ODE) | $\nabla f^\top u_t$ | velocity $u_t$ | Continuity Eq. |
| Diffusion (SDE) | $\frac{1}{2}\sigma_t^2 \cdot \nabla^2 f$ | diffusion coeff $\sigma_t^2$ | Fokker–Planck Eq. |
| Jump process | $\int (f(y) - f(x))\,Q_t(dy, x)$ | jump kernel $Q_t$ | Jump continuity Eq. |
| CTMC (discrete) | $f^\top Q_t$ | rate matrix $Q_t$ | Kolmogorov Eq. |

All marginals obey the **Kolmogorov Forward Equation** $\frac{d}{dt}\langle p_t, f\rangle = \langle p_t, \mathcal{L}_t f \rangle$, unifying every "mass conservation" PDE above. The FM recipe holds in full generality: conditional generators marginalize under the posterior, and a conditional Bregman loss on the generator's linear parameterization has unbiased gradients — FM, diffusion, and discrete FM are all instances.

Generators are **linear** and the KFE is a linear equation, so trained models *combine*:
1. **Markov superposition**: $\lambda_t \mathcal{L}_t + (1-\lambda_t)\mathcal{L}_t'$ still generates the shared path (jump + flow hybrids).
2. **Divergence-free components**: add any $\mathcal{L}^\text{div}_t$ with $\langle p_t, \mathcal{L}^\text{div}_t f\rangle = 0$ — Langevin and Metropolis–Hastings kernels ([[MCMC]]); the origin of stochastic samplers and predictor–corrector schemes.
3. **Multimodal models**: factorized paths on product spaces let continuous FM (images) and discrete FM (text) train jointly.

## Practical Summary
---
**Training** (cond-OT default): sample $z \sim p_\text{data}$, $\epsilon \sim \mathcal{N}(0,I_d)$, $t$ (uniform or logit-normal); regress $u_t^\theta(x_t)$ onto $z - \epsilon$ at $x_t = t z + (1-t)\epsilon$. No noise schedule, no SNR weighting; any backbone ([[U-Net]], [[DiT]]) — only the target changes.

**Sampling**: any ODE solver; midpoint/Heun beats Euler at equal NFE; typical budgets 5–50 steps, 1–4 after distillation/reflow. Optional Langevin stochasticity; optional post-hoc scheduler swap.

```python
# sampling: integrate dx/dt = model(x, t) from t=0 to 1
x = torch.randn(n, d)
for t in torch.linspace(0, 1, steps + 1)[:-1]:
    x = x + (1 / steps) * model(x, t)   # Euler; midpoint/Heun work better
```

**Where it's used**: SD3 / FLUX (rectified-flow [[DiT]] in [[LDM]]-style latent space), Movie Gen (video), Voicebox / Audiobox / F5-TTS (speech & audio), SE(3) FM for protein backbones (FoldFlow, FrameFlow, Multiflow), π0's flow-matching action expert ([[VLA]]), discrete FM for text.

## Relationship to Other Notes
---
- [[DDPM]] — diffusion = FM with a Gaussian path from a forward SDE + score parameterization + inverted time; DDIM/prob-flow sampling = FM's ODE sampling
- [[Score Matching]] — the score-side training algorithm (6.S184 §4): $\mathcal{L}_\text{SM}/\mathcal{L}_\text{CSM}$ mirror $\mathcal{L}_\text{FM}/\mathcal{L}_\text{CFM}$ theorem-for-theorem; Prop. 1 converts the trained models into each other
- [[Score-Based Generative Modeling Methods]] — the broader score-based picture: EBMs, Langevin/annealed Langevin, the SDE unification
- [[Divergence Theorem]] — the continuity equation is its differential form; instantaneous change of variables and Hutchinson's estimator run on divergences
- [[LDM]] / [[DiT]] — SD3/FLUX pair FM objectives with DiT backbones in VAE latent space; backbone ⊥ objective
- [[VLA]] — π0's flow-matching action expert: the same CFM math with action chunks as the data point $z$
- [[ELBO]] — flows give exact unbiased likelihoods; discrete FM's generalized-KL loss *is* an ELBO
- [[MCMC]] — Langevin and Metropolis–Hastings are divergence-free generator components (predictor–corrector sampling)
- [[VAE]] / [[RAE]] / [[FID]] — latent-space FM and generative-quality evaluation
