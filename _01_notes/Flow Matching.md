### [[dl_moc]]  
# **Flow Matching**

**Tags**:: #generative #dl_theo #diffusion
**Links**:: [[DDPM]] · [[Score-Based Generative Modeling Methods]] · [[Divergence Theorem]] · [[LDM]] · [[DiT]] · [[VAE]] · [[ELBO]] · [[MCMC]] · [[FID]] · [[RAE]] · [[VLA]]


**Flow Matching (FM)** is a simulation-free framework for training generative models: pick a probability path $p_t$ that interpolates a source distribution $p_\text{init}$ (noise) into the data distribution $p_\text{data}$, then train a neural velocity field to generate that path by simple regression. Sampling is ODE integration. It was introduced concurrently by [Lipman et al., 2022](https://arxiv.org/abs/2210.02747) (Flow Matching), [Liu et al., 2022](https://arxiv.org/abs/2209.03003) (Rectified Flow), and [Albergo & Vanden-Eijnden, 2022](https://arxiv.org/abs/2209.15571) (stochastic interpolants), and now powers state-of-the-art systems for images (SD3, [Esser et al., 2024](https://arxiv.org/abs/2403.03206)), video (Movie Gen), speech/audio (Voicebox, Audiobox), protein backbones (SE(3) FM), and robot policies (π0 — see [[VLA]]).

This note follows the comprehensive treatment in **"Flow Matching Guide and Code"** ([Lipman et al., 2024, arXiv 2412.06264](https://arxiv.org/abs/2412.06264), FAIR/MIT/Weizmann), which comes with the [`flow_matching` PyTorch library](https://github.com/facebookresearch/flow_matching). The background section follows the more introductory **MIT 6.S184 lecture notes** ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)).

## The FM Blueprint
---
1. **Choose** a probability path $(p_t)_{0\le t\le 1}$ with boundary conditions $p_0 = p_\text{init}$ (known source, e.g. $\mathcal{N}(0,I_d)$) and $p_1 = p_\text{data}$.
2. **Train** a velocity field $u_t^\theta$ (neural network) to match the velocity $u_t^\text{target}$ known to generate $p_t$, via regression.
3. **Sample** by drawing $X_0 \sim p_\text{init}$ and solving the ODE $\frac{d}{dt}X_t = u_t^\theta(X_t)$ from $t=0$ to $1$.

The simplest instantiation fits in ~10 lines (6.S184 Alg. 3):

```python
# training step (linear / cond-OT path)
z = sample_data(); eps = torch.randn_like(z); t = torch.rand(b, 1)
x_t  = t * z + (1 - t) * eps
loss = ((model(x_t, t) - (z - eps))**2).mean()

# sampling: integrate dx/dt = model(x, t) from t=0 to 1 (Euler / midpoint)
x = torch.randn(n, d)
for t in torch.linspace(0, 1, steps + 1)[:-1]:
    x = x + (1 / steps) * model(x, t)   # Euler; midpoint works better
```

*This background follows the MIT 6.S184 lecture notes ([Holderrieth & Erives, "An Introduction to Flow Matching and Diffusion Models"](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf), §2–4; [course site](https://diffusion.csail.mit.edu/) with recordings and labs) — a gentler build-up of the same machinery as Ch. 3 of the FM guide.*

## Background
---
**Time convention (important!)**: FM uses $t=0$ for **noise/source** and $t=1$ for **data/target** — the *opposite* of the diffusion convention in [[DDPM]] where $x_0$ is data. Diffusion time $r$ relates to FM time by a decreasing reparameterization $r = k(t)$, with data at $r=0$ and noise at $r \to \infty$.

**Notation**: source $p_\text{init}$, data $p_\text{data}$; data point $z \sim p_\text{data}$; Gaussian noise $\epsilon \sim p_\text{init} = \mathcal{N}(0, I_d)$; noise schedulers $(\alpha_t, \beta_t)$; ground-truth marginal velocity $u_t^\text{target}(x)$, conditional velocity $u_t^\text{target}(x|z)$, neural network $u_t^\theta$. The FM guide instead writes $p, q$; $x_1$; $x_0$; $(\alpha_t, \sigma_t)$; $u_t(x)$, $u_t(x|x_1)$ — translate accordingly when reading it.

+ **Trajectory**. A time-dependent **vector field** $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$, $(x,t) \mapsto u_t(x)$, assigns a velocity to every point and time. It defines an **ODE**, whose solution from a start point $x_0$ is a **trajectory** $t \mapsto X_t$:
$$
\frac{d}{dt}X_t = u_t(X_t), \qquad X_0 = x_0 .
$$

+ **Flow**. Collecting the solutions from *every* start point into a single map gives the **flow** $\psi : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$:
$$
\frac{d}{dt}\psi_t(x_0) = u_t(\psi_t(x_0)), \qquad \psi_0(x_0) = x_0, \qquad X_t = \psi_t(X_0).
$$
	Conversely a flow determines its field via $u_t(x) = \dot\psi_t(\psi_t^{-1}(x))$. In the 6.S184 slogan: ==*vector fields define ODEs whose solutions are flows*.== 

+ **Flow model**. A **flow model** turns this into a generative model by making *only the initial condition* random: $X_0 \sim p_\text{init}$, $\frac{d}{dt}X_t = u_t^\theta(X_t)$, with the goal $X_1 = \psi_1^\theta(X_0) \sim p_\text{data}$. The network parameterizes the **vector field, never the flow itself** — evaluating $\psi_t^\theta$ always means simulating the ODE.

+ **Lipschitz condition.** A vector field is global **Lipschitz** if $$\|u_t(x) - u_t(y)\| \le L\,\|x - y\|, \forall x, y.$$
	**Picard–Lindelöf theorem** (Cauchy–Lipschitz; existence & uniqueness of the solution). *Let $u_t(x)$ be continuous in $t$ and Lipschitz in $x$ with constant $L$ uniform in $t$. Then for every initial condition $X_{t_0} = x_0$, the ODE $\frac{d}{dt}X_t = u_t(X_t)$ has **exactly one** solution, defined on all of $[0,1]$ in the globally Lipschitz case.
	
	If $u$ is only **locally** Lipschitz, the unique solution exists on a maximal time interval around $t_0$, which may end in finite-time blow-up. Both hypotheses earn their keep. Weaken Lipschitz to mere continuity and existence survives (Peano's theorem) but **uniqueness fails**: $\dot x = \sqrt{|x|}$, $x(0) = 0$ is solved by both $x \equiv 0$ and $x(t) = t^2/4$ — the field has infinite slope at the origin. Keep only *local* Lipschitzness and solutions can **blow up** before $t=1$: $\dot x = x^2$, $x(0) = 1$ gives $x(t) = 1/(1-t)$, which escapes to infinity at $t = 1$.
	
	Specially, if $u_t(x)$ is continuously differentiable with bounded derivative (which implies Lipschitz with $L = \sup\|\partial_x u\|$), the ODE has a unique solution and $\psi_t$ is a **diffeomorphism**. ==Uniqueness is what makes flows well-behaved: two trajectories can never cross or merge at the same time== — they would be two solutions of one initial-value problem — ==which is why $\psi_t$ is a bijection==, and why the *marginal* FM velocity must average conditional velocities wherever conditional paths cross. Neural networks with Lipschitz activations satisfy the condition automatically, so the theorem is "good news rather than a concern" (6.S184). The FM guide's weaker hypothesis — *local* Lipschitzness plus integrability $\int_0^1 \mathbb{E}\|u_t(X_t)\|\,dt < \infty$ — rules out blow-up for almost every trajectory instead, which is what's needed near $t = 1$, where conditional velocities like $(z - x)/(1-t)$ blow up (hence "generates" on $[0,1)$). Worked example (6.S184 Ex. 4): $u_t(x) = -\theta x$ has the explicit flow $\psi_t(x_0) = e^{-\theta t}x_0$.

+ **Simulation.** The **Euler method** updates $X_{t+h} = X_t + h\,u_t(X_t)$ with step size $h = 1/n$ ($o(h)$ local error). 
	**Heun's method**: a predictor-corrector scheme, first predicts $X'_{t+h} = X_t + h\,u_t(X_t)$, then averages velocities at the current and guessed states $X_{t+h} = X_t + \frac{h}{2}\big(u_t(X_t) + u_{t+h}(X'_{t+h})\big)$ (mid-point method).  

+ **Stochastic process, diffusion coefficient, SDE.** A **stochastic process** $(X_t)_{0\le t\le 1}$ is a random trajectory — each simulation gives a different path. An **SDE** adds Brownian noise, scaled by a **diffusion coefficient** $\sigma_t \ge 0$, to the ODE's infinitesimal update
$$
dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t
$$
	is shorthand for $X_{t+h} = X_t + h\,u_t(X_t) + \sigma_t(W_{t+h} - W_t) + h\,R_t(h)$ with a vanishing error term. 
	**Euler–Maruyama Simulation**: $X_{t+h} = X_t + h\,u_t(X_t) + \sqrt{h}\,\sigma_t\,\epsilon_t$, $\epsilon_t \sim \mathcal{N}(0, I_d)$. Randomness now enters the dynamics themselves, so there is no flow map. **Every ODE is an SDE with $\sigma_t = 0$** — in 6.S184's framing, a "diffusion model" is literally a flow model plus a diffusion coefficient. 

+ **Probability path.** A **probability path** $(p_t)_{0 \le t \le 1}$ is a time-indexed family of distributions — a trajectory in the space of distributions, prescribing *what distribution $X_t$ should have at each time*. 

+ **Conditional probability path**. A conditional probability path interpolates noise into a *single data point* $z$ with Dirac distribution
$$
p_0(\cdot|z) = p_\text{init}, \qquad p_1(\cdot|z) = \delta_z \qquad \forall z,
$$
+ **Marginal probability path.** A marginal probability path is by mixing over the data: $p_t(x) = \int p_t(x|z)\, p_\text{data}(z)\, dz$, which interpolates $p_0 = p_\text{init}$ to $p_1 = p_\text{data}$. Sampling from $p_t$ is easy (draw $z \sim p_\text{data}$, then $x \sim p_t(\cdot|z)$) even though its density is intractable. 
	Eg. **Gaussian path** with noise schedulers $(\alpha_t, \beta_t)$, $\alpha_0 = \beta_1 = 0$, $\alpha_1 = \beta_0 = 1$:$$
p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\, \beta_t^2 I_d), \qquad x_t = \alpha_t z + \beta_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_d).
$$
+ **Conditional vector field.** A conditional vector field $u_t^\text{target}(x|z)$ is any field whose ODE transports $p_\text{init}$ along the conditional path $$X_0 \sim p_\text{init},\ \frac{d}{dt}X_t = u_t^\text{target}(X_t|z) \Rightarrow X_t \sim p_t(\cdot|z).$$
	It can typically be derived *by hand*: for the Gaussian path, the conditional flow $\psi_t(x|z) = \alpha_t z + \beta_t x$ gives$$
u_t^\text{target}(x|z) = \Big(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\Big) z + \frac{\dot\beta_t}{\beta_t}\, x .
$$

+ Marginal vector field. 

	The flow corresponds to the marginal vector field, and it is what we plan to train for.

+ Marginalization Trick. 



+ **Continuity Equation (Mass Conservation).** Say $u_t$ **generates** $p_t$ if $X_t \sim p_t$ for all $t$ whenever $X_0 \sim p_0$. The central characterization (6.S184 Thm 11; = the guide's Mass Conservation theorem) is: this holds **if and only if**

$$
\partial_t\, p_t(x) = -\,\mathrm{div}\big(p_t u_t\big)(x) \qquad \text{for all } x,\ t.
$$

Intuition (6.S184): $\partial_t p_t(x)$ is the local change of probability at $x$; divergence measures net *outflow* of a field, so $-\mathrm{div}(p_t u_t)$ is the **net inflow of probability mass** carried by the probability flux $j_t = p_t u_t$; since total mass is conserved (always integrates to 1), the two sides must agree — the [[Divergence Theorem]] in differential form. This single PDE licenses everything that follows: the Marginalization Trick is proven by checking that the marginal pair $(p_t, u_t^\text{target})$ satisfies it.

**Fokker–Planck: the SDE generalization.** For $dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t$: $X_t \sim p_t$ for all $t$ **iff**

$$
\partial_t\, p_t = -\,\mathrm{div}(p_t u_t) + \frac{\sigma_t^2}{2}\Delta p_t ,
$$

where the Laplacian $\Delta = \mathrm{div}\,\nabla$ contributes a heat-equation term for the added noise; $\sigma_t = 0$ recovers the Continuity Equation (6.S184 Thm 19). Immediate payoff (their Thm 17): rewriting $\frac{\sigma_t^2}{2}\Delta p_t = \mathrm{div}\big(p_t \frac{\sigma_t^2}{2}\nabla\log p_t\big)$ shows that

$$
dX_t = \Big[u_t(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\Big]dt + \sigma_t\,dW_t
$$

has the **same marginals $p_t$ for every noise level $\sigma_t \ge 0$** — Langevin dynamics ([[MCMC]]) is the special case of a static path. This is where diffusion's stochastic samplers come from (see *Diffusion Models as a Special Case*); in theory $\sigma_t$ is a free knob, and a practically "optimal" $\sigma_t$ exists only because of training and simulation error.

**Exact likelihoods.** The Continuity Equation implies the *instantaneous change of variables* ([Chen et al., 2018](https://arxiv.org/abs/1806.07366), Neural ODEs; guide §3.6 — not covered in 6.S184):

$$
\frac{d}{dt}\log p_t(\psi_t(x)) = -\,\mathrm{div}(u_t)(\psi_t(x)),
$$

so $\log p_1$ can be computed by integrating the divergence along the trajectory — with Hutchinson's trace estimator $\mathrm{div}(u_t)(x) = \mathbb{E}_Z\,\mathrm{tr}[Z^\top \partial_x u_t(x) Z]$ giving an **unbiased** log-likelihood estimate via one vector–Jacobian product per step. This is a genuine advantage of flows over stochastic processes, which only admit lower bounds ([[ELBO]]).

Classical CNFs trained by maximum likelihood need full ODE simulation (and its differentiation) *inside* the training loop — this is what FM eliminates.

## FM and Conditional FM Losses
---
The FM loss regresses onto the target velocity along the path (6.S184 Eq. 24):

$$
\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, x \sim p_t}\, \big\|u_t^\theta(x) - u_t^\text{target}(x)\big\|^2 .
$$

This is intractable — $u_t^\text{target}$ marginalizes over the whole dataset. The fix is **conditioning**.

**Marginal field from conditional fields.** The marginal path and its generating field are posterior mixtures over the data point $z$:

$$
p_t(x) = \int p_t(x|z)\, p_\text{data}(z)\, dz, \qquad
u_t^\text{target}(x) = \int u_t^\text{target}(x|z)\, \frac{p_t(x|z)\,p_\text{data}(z)}{p_t(x)}\, dz = \mathbb{E}\big[u_t^\text{target}(x_t|z)\,\big|\,x_t = x\big],
$$

i.e. the marginal velocity is the **posterior-weighted average** of conditional velocities $u_t^\text{target}(x|z)$ — for each $z$, the direction that would carry $x$ toward $z$, weighted by "how much we believe $x$ came from $z$" (6.S184).

**Marginalization Trick (6.S184 Thm 9; guide Thm 3)**: under mild regularity (C¹ conditional paths/velocities, bounded data support, $p_t > 0$), if $u_t^\text{target}(\cdot|z)$ generates $p_t(\cdot|z)$, then the marginal $u_t^\text{target}$ generates the marginal $p_t$. Proof: plug into the Continuity Equation and swap integral/divergence. The guide generalizes the conditioning from $z$ to any variable $Z$ — source samples, or two-sided pairs $(x_0, z)$ — all essentially equivalent.

**Conditional FM loss** — the tractable objective (6.S184 Eq. 26):

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_\text{data},\, x \sim p_t(\cdot|z)}\, \big\|u_t^\theta(x) - u_t^\text{target}(x|z)\big\|^2 ,
\qquad \mathcal{L}_\text{FM}(\theta) = \mathcal{L}_\text{CFM}(\theta) + C,
$$

equal up to a $\theta$-independent constant (6.S184 Thm 12, by expanding the square), so gradients coincide: $\nabla_\theta \mathcal{L}_\text{FM} = \nabla_\theta \mathcal{L}_\text{CFM}$, and the minimizer is $u_t^{\theta^*} = u_t^\text{target}$.

**Why this works in general — Bregman divergences (guide).** The gradient equality holds not just for MSE but for any **Bregman divergence** $D(u,v) = \Phi(u) - \Phi(v) - \langle u - v, \nabla\Phi(v)\rangle$ ($\Phi$ strictly convex; squared Euclidean is $\Phi = \|\cdot\|^2$). Bregman divergences have affine-invariant gradients in the second argument, so $\nabla_v D(\mathbb{E}[Y], v) = \mathbb{E}[\nabla_v D(Y, v)]$ — expectation and gradient can be swapped. General principle: for any $f_t(x_0, z)$, the conditional matching loss $\mathbb{E}\, D(f_t(x_0, z), g_t^\theta(x_t))$ has minimizer $g_t^\theta(x) = \mathbb{E}[f_t(x_0, z) \mid x_t = x]$ — **regression targets can be single samples; the network learns their conditional expectation**. Notably, Bregman-ness is *necessary* for this property (Holderrieth et al., 2024).

**Time weighting**: $t$ need not be uniform — training with $t \sim \omega(t)$ (equivalently reweighting the loss) changes nothing mathematically, but sampling $t$ non-uniformly (e.g. SD3's logit-normal) works better in practice at scale.

## Designing Paths via Conditional Flows
---
Rather than hand-designing $p_t(\cdot|z)$, define a **conditional flow** $\psi_t(x|z)$ with boundary conditions $\psi_0(x|z) = x$, $\psi_1(x|z) = z$ (smooth, diffeomorphic in $x$). Then $X_t = \psi_t(X_0|z)$ defines both the conditional path and its velocity, and the CFM loss simplifies to

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\,(x_0,\,z) \sim \pi}\, \big\|u_t^\theta(x_t) - \dot\psi_t(x_0|z)\big\|^2 ,
$$

— the regression target is just the **time-derivative of the interpolant**, no densities needed. ($\pi$ is the coupling of source and data samples; independent by default — see *Data Couplings*.)

### The conditional-OT (linear) path
Minimizing a Jensen upper bound on the **kinetic energy** $\int_0^1 \mathbb{E}\|u_t^\text{target}(X_t)\|^2 dt$ over conditional flows gives (Euler–Lagrange: $\ddot\gamma_t = 0$) straight lines:

$$
\psi_t(x|z) = t\,z + (1-t)\,x
\;\;\Rightarrow\;\;
x_t = t z + (1-t)\epsilon, \quad
u_t^\text{target}(x|z) = \frac{z - x}{1-t}, \quad
\dot\psi_t(\epsilon|z) = z - \epsilon.
$$

With Gaussian source this makes $p_t(x|z) = \mathcal{N}(x \mid t z, (1-t)^2 I_d)$ (schedulers $\alpha_t = t$, $\beta_t = 1-t$) and yields the simplest FM loss $\mathbb{E}\|u_t^\theta(x_t) - (z - \epsilon)\|^2$.

Facts worth remembering:
- If $p_\text{data} = \delta_z$ (single point), the linear conditional flow **is** the exact dynamic optimal transport solution.
- In general the *marginal* flow is **not** OT — the linear choice only minimizes a bound; the marginal kinetic energy is bounded by the coupling cost $\mathbb{E}\|z - x_0\|^2$.
- **Conditional** straight lines ≠ **marginal** straight trajectories: marginal paths of independent couplings still curve (conditional lines cross), which is what multisample couplings and reflow address (below).

## Affine Paths, Schedulers, and Parameterizations
---
The linear path generalizes to **affine conditional flows** $\psi_t(x|z) = \alpha_t z + \beta_t x$ with a **scheduler** $(\alpha_t, \beta_t)$ satisfying $\alpha_0 = 0,\ \alpha_1 = 1,\ \beta_0 = 1,\ \beta_1 = 0$, $\dot\alpha_t > 0$, $\dot\beta_t < 0$. The CFM target becomes $\dot\alpha_t z + \dot\beta_t \epsilon$. Examples: cond-OT ($\alpha_t = t, \beta_t = 1-t$), polynomial ($t^n$), LinearVP ($\alpha_t = t, \beta_t = \sqrt{1-t^2}$), cosine ($\alpha_t = \sin\tfrac{\pi t}{2}, \beta_t = \cos\tfrac{\pi t}{2}$). Diffusion's VP/VE schedules are (approximate) members: VP has $\alpha_t^2 + \beta_t^2 = 1$ with exponential form; VE has $\alpha_t \equiv 1$, $\beta_0 \gg 1$ — both only *approximately* hit a Gaussian at $t = 0$.

**Equivalent parameterizations.** Using $x_t = \alpha_t z + \beta_t \epsilon$, the marginal velocity $u_t^\text{target}(x) = \dot\alpha_t\,\mathbb{E}[z|x_t{=}x] + \dot\beta_t\,\mathbb{E}[\epsilon|x_t{=}x]$ can be rewritten via:
- **denoiser / $z$-prediction** (6.S184's $D_t(x) := \mathbb{E}[z|x_t = x]$, the posterior mean): $u_t^\text{target}(x) = \frac{\dot\beta_t}{\beta_t}x + \big(\dot\alpha_t - \alpha_t\frac{\dot\beta_t}{\beta_t}\big)\,D_t(x)$
- **noise / $\epsilon$-prediction** ($\hat\epsilon_t(x) := \mathbb{E}[\epsilon|x_t = x]$; diffusion's ε-prediction): $u_t^\text{target}(x) = \frac{\dot\alpha_t}{\alpha_t}x + \big(\dot\beta_t - \beta_t\frac{\dot\alpha_t}{\alpha_t}\big)\,\hat\epsilon_t(x)$
- **score** (Gaussian paths): see below.

All are related by closed-form conversions (Table 1 of the guide; 6.S184 Remark 16 — any quantity recovering the posterior mean $\mathbb{E}[z|x]$ works) and each is learnable with a conditional matching loss ($f_t(\epsilon, z) = z$, $\epsilon$, etc.). Caveat: the conversions have **singularities** at the endpoints (denoiser blows up as $t \to 1$, ε-pred/score as $t \to 0$); direct velocity prediction avoids them.

**Gaussian paths and the score.** For $p_t(x|z) = \mathcal{N}(x|\alpha_t z, \beta_t^2 I_d)$:

$$
\nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}, \qquad
\nabla \log p_t(x) = -\frac{\hat\epsilon_t(x)}{\beta_t},
$$

so the score is just rescaled noise-prediction, and velocity and score convert into each other (6.S184 Prop. 1):

$$
u_t^\text{target}(x) = a_t\,\nabla \log p_t(x) + b_t\, x, \qquad
a_t = \beta_t^2\frac{\dot\alpha_t}{\alpha_t} - \dot\beta_t\beta_t, \quad
b_t = \frac{\dot\alpha_t}{\alpha_t}.
$$

This expression is a **gradient field** plus a radial term, hence kinetically optimal *for that fixed Gaussian path* (it's the minimum-energy velocity generating $p_t$ — guide §4.8.3); and it means a trained velocity model contains the score for free, and vice versa.

**Post-training scheduler change (guide).** Any two affine paths with the same coupling are related by a **scale-time transformation** $\bar\psi_r = s_r\,\psi_{t_r}$ with $t_r = \rho^{-1}(\bar\rho(r))$ matched via the signal-to-noise ratio $\rho_t = \alpha_t/\beta_t$, giving $\bar u_r(x) = \frac{\dot s_r}{s_r}x + s_r \dot t_r\, u_{t_r}(x / s_r)$. Consequences: a trained model's scheduler can be **swapped at inference** (e.g. VP → cond-OT), and *all schedulers produce the same final map* $\psi_1$ in continuous time — scheduler choice matters only through discretization error and learning dynamics.

## Data Couplings
---
FM trains on pairs $(x_0, z) \sim \pi$; the default is the **independent coupling** $\pi = p_\text{init} \otimes p_\text{data}$ (draw $\epsilon$ and $z$ separately), but this is a design axis:

- **Paired data** ([Liu et al., 2023](https://arxiv.org/abs/2302.05872) I²SB; Albergo et al., 2024): for inpainting/super-resolution/deblurring, build $\pi(x_0, z) = \pi(x_0|z)\,p_\text{data}(z)$ by corrupting data samples (+ noise so the source has a density). FM then learns the restoration map directly — no guidance needed.
- **Multisample / minibatch-OT couplings** ([Pooladian et al., 2023](https://arxiv.org/abs/2304.14772); [Tong et al., 2023](https://arxiv.org/abs/2302.00482)): draw $k$ source and $k$ target samples, solve the discrete OT assignment within the batch, train on matched pairs. This provably lowers transport cost vs. independent coupling and **straightens marginal trajectories** (→ fewer ODE steps); as $k \to \infty$ it approaches true OT.

Related (beyond the guide): **Rectified Flow's reflow** ([Liu et al., 2022](https://arxiv.org/abs/2209.03003)) replaces the coupling with $(x_0, \hat z)$ pairs generated by simulating the trained model, iteratively straightening paths toward 1–2-step generation; this underlies SD3's and FLUX's fast variants.

## Guidance
---
With labels/conditioning $y$, train a **guided velocity** $u_t^\theta(x|y)$ with the same CFM loss (the theory carries through per-$y$; "vanilla guidance" in 6.S184 §5). To *reinforce* the prompt, use the velocity ↔ score conversion (Prop. 1): Bayes' rule gives $\nabla\log p_t(x|y) = \nabla\log p_t(x) + \nabla\log p_t(y|x)$, so **classifier guidance** scales up the classifier term, $\tilde u_t(x|y) = u_t^\text{target}(x) + w\,a_t \nabla \log p_t(y|x)$, and **classifier-free guidance** eliminates the classifier ([Ho & Salimans, 2021](https://arxiv.org/abs/2207.12598); for flows [Zheng et al., 2023](https://arxiv.org/abs/2311.13443)):

$$
\tilde u_t^\theta(x|y) = (1 - w)\, u_t^\theta(x|\varnothing) + w\, u_t^\theta(x|y),
$$

with guidance scale $w > 1$, training with condition dropout $\xi \sim \text{Bernoulli}(p_\text{uncond})$ — exactly CFG, applied to velocities. (For $w \ne 1$ this is a heuristic — it is *not* the true guided field, and what distribution it samples remains theoretically murky; it is nonetheless the standard for large-scale text-to-image FM like SD3.)

## Diffusion Models as a Special Case
---
The guide's unifying claim (§10): **a diffusion model is an FM model with a specific Gaussian path and score parameterization** (plus inverted time).

- The **forward noising SDE** $dX_r = a_r X_r\,dr + g_r\,dW_r$ (affine drift, so marginals are Gaussian in closed form) is just a way of *constructing* the probability path: $p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ with $\alpha, \beta$ determined by $a, g$.
- **Denoising score matching** (the DDPM loss; 6.S184's $\mathcal{L}_\text{CSM}$) is precisely the conditional matching loss for ε-prediction, reparameterized by $s_t^\theta = -\hat\epsilon_t^\theta / \beta_t$; its minimizer is the marginal score $\nabla \log p_t$.
- The **probability-flow ODE** of [Song et al., 2021](https://arxiv.org/abs/2011.13456) is exactly the FM marginal velocity written via the score (Prop. 1); deterministic diffusion sampling = FM sampling.
- **Stochastic (SDE) sampling** = adding **Langevin dynamics** $\tfrac{\sigma_t^2}{2} \nabla \log p_t\,dt + \sigma_t dW_t$, which is *divergence-free* w.r.t. $p_t$: any $\sigma_t \ge 0$ preserves the marginals (6.S184 Thm 17). The classical Anderson time-reversal is one particular $\sigma_t$ — with a perfect model all choices are equivalent; with errors, an empirically tuned noise level can help (Karras et al., 2022).
- **Time-reversal is not needed**: "find a process with the forward marginals reversed" ≡ "find a velocity generating $p_t$" ≡ "solve the Continuity/Fokker–Planck equation" — FM solves the third problem directly.

| | Diffusion ([[DDPM]]) | Flow Matching (cond-OT) |
|---|---|---|
| **Path construction** | forward noising SDE, closed-form marginals | direct choice of conditional flow |
| **Network target** | noise $\epsilon$ / score | velocity $z - \epsilon$ (or any parameterization) |
| **Boundary at noise end** | Gaussian only as $r \to \infty$ (approx.) | exactly $p_\text{init}$ at $t = 0$ |
| **Schedule** | required, baked into training | scheduler swappable post-training |
| **Sampling** | reverse SDE or prob-flow ODE | ODE (SDE optional via Langevin) |
| **Likelihood** | ELBO (SDE) / exact (ODE) | exact, unbiased estimator |
| **Endpoint singularities** | score/ε-pred singular near noise | velocity prediction non-singular |

Practical upshot: with straight(er) conditional paths, Euler/midpoint integration needs far fewer function evaluations (local truncation error scales with trajectory curvature); perfectly straight marginal flows would sample in one step — the motivation for OT couplings and reflow.

## Extension: Riemannian Flow Matching
---
FM transfers to Riemannian manifolds $\mathcal{M}$ (sphere, tori, SE(3) — proteins, molecules, geoscience) ([Chen & Lipman, 2024](https://arxiv.org/abs/2302.03660)). Velocities live in tangent spaces, the continuity equation uses the Riemannian divergence, and the Marginalization Trick + CFM gradient equivalence hold verbatim with tangent-space Bregman divergences. The conditional flow generalizes straight lines to **geodesic interpolation**:

$$
\psi_t(x|z) = \exp_{x}\!\big(\kappa(t)\, \log_{x}(z)\big), \qquad \kappa(0)=0,\ \kappa(1)=1,
$$

which is **simulation-free** whenever exp/log maps are closed-form (simple manifolds) — a real advantage over Riemannian diffusion, which needs in-training simulation and score approximations. For hard geometries, geodesics can be replaced by any smooth **premetric** $d(x, z)$ (e.g. spectral distances) with the minimal-norm velocity $u_t^\text{target}(x|z) = \frac{d \log \bar\kappa(t)}{dt}\, d(x, z) \frac{\nabla d(x, z)}{\|\nabla d(x, z)\|_g^2}$.

## Extension: Discrete Flow Matching (CTMC)
---
For discrete data $x = (x^1, \dots, x^d) \in \mathcal{V}^d$ (tokens, vocabulary size $V$; 6.S184 §7), the flow is replaced by a **Continuous-Time Markov Chain** ([Campbell et al., 2024](https://arxiv.org/abs/2402.04997); [Gat et al., 2024](https://arxiv.org/abs/2407.15595)): transitions $p_{t+h|t}(y|x) = \delta_x(y) + h\,Q_t(y|x) + o(h)$, where the velocity is now a **rate matrix** $Q_t$ satisfying $Q_t(y|x) \ge 0$ for $y \ne x$ and $\sum_y Q_t(y|x) = 0$. The Continuity Equation becomes the **Kolmogorov equation** $\frac{d}{dt}p_t(y) = \sum_x Q_t(y|x)\,p_t(x)$, and the whole FM recipe transfers: marginalization trick, conditional rates, Bregman CFM loss.

Key design points:
- **Factorized rates**: modeling rates over all $V^d$ states is infeasible; restrict to transitions changing **at most one token** (neighbors $N(x)$), parameterized per position as $Q_t(v, i \mid x)$ — output dimension $d \cdot V$ (a transformer with a $V$-way head), and per-token parallel Euler sampling is valid to $o(h)$ (6.S184 Alg. 7).
- **Mixture paths**: per token, $X_t^i = z^i$ with probability $\kappa_t$, else $x_0^i$ ($\kappa_0 = 0, \kappa_1 = 1$), with closed-form conditional rates $Q_t(v, i \mid x; x_0, z) = \frac{\dot\kappa_t}{1-\kappa_t}\big[\delta_{z^i}(v) - \delta_{x^i}(v)\big]$. Source: uniform noise or a **mask token** $m$ (mask-DFM is the discrete analog of noise→data).
- **Posterior parameterization**: learn $p_{1|t}^{\theta,i}(z^i | x)$ — a per-token denoiser predicting clean tokens given the current sequence (the discrete analog of $D_t$). Trained with cross-entropy (KL Bregman) or **generalized KL**, which doubles as an [[ELBO]] on target likelihood for evaluation.
- **Sampling**: at each step, per token, sample $z^i \sim p^{\theta,i}_{1|t}(\cdot|x)$, then take the CTMC Euler step (decide jump-to-target vs. stay). Divergence-free rate components can be added freely (corrector-style noise ↔ un-masking/re-masking trade-offs).

This is the FM view of "discrete diffusion" language models and multimodal protein co-design.

## Extension: Generator Matching — One Framework for Everything
---
**Generator Matching (GM)** ([Holderrieth et al., 2024](https://arxiv.org/abs/2410.20587)) abstracts FM to *any* Continuous-Time Markov Process on *any* state space. Every well-behaved (Feller) Markov process is characterized by its **generator** $\mathcal{L}_t$, the first-order derivative of its transition kernel acting on test functions: $\mathbb{E}[f(X_{t+h})|X_t = x] = f(x) + h\,[\mathcal{L}_t f](x) + o(h)$.

| Process | Generator $\mathcal{L}_t f$ | Parameterized by | KFE ≙ |
|---|---|---|---|
| Flow (ODE) | $\nabla f^\top u_t$ | velocity $u_t$ | Continuity Eq. |
| Diffusion (SDE) | $\frac{1}{2}\sigma_t^2 \cdot \nabla^2 f$ | diffusion coeff $\sigma_t^2$ | Fokker–Planck Eq. |
| Jump process | $\int (f(y) - f(x))\,Q_t(dy, x)$ | jump kernel $Q_t$ | Jump continuity Eq. |
| CTMC (discrete) | $f^\top Q_t$ | rate matrix $Q_t$ | Kolmogorov Eq. |

The marginals of any such process obey the **Kolmogorov Forward Equation** $\frac{d}{dt}\langle p_t, f\rangle = \langle p_t, \mathcal{L}_t f \rangle$, which unifies all the "mass conservation" PDEs above. The FM recipe then holds in full generality: conditional generators (found by solving the KFE analytically for simple conditional paths) marginalize under the posterior $\mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)}$, and a conditional Bregman loss on the generator's linear parameterization has unbiased gradients — FM, diffusion, and discrete FM are all instances.

Because generators are **linear** and the KFE is a linear equation, trained models can be *combined*:
1. **Markov superposition**: $\lambda_t \mathcal{L}_t + (1-\lambda_t)\mathcal{L}_t'$ of two generators for the same path still generates it (e.g. jump + flow hybrids).
2. **Divergence-free components**: add any $\mathcal{L}^\text{div}_t$ with $\langle p_t, \mathcal{L}^\text{div}_t f\rangle = 0$ — Langevin dynamics and Metropolis–Hastings kernels ([[MCMC]]) are exactly such components, which is where diffusion's stochastic samplers and predictor–corrector schemes come from.
3. **Multimodal models**: factorized paths over product state spaces let continuous FM (images) and discrete FM (text) train jointly in one model.

## Practical Summary
---
**Training** (cond-OT, the default): sample $z \sim p_\text{data}$, $\epsilon \sim \mathcal{N}(0,I_d)$, $t$ (uniform or logit-normal); regress $u_t^\theta(x_t)$ onto $z - \epsilon$ at $x_t = t z + (1-t) \epsilon$. No noise schedule, no SNR weighting; any architecture ([[U-Net]], [[DiT]]) works — only the target changes.

**Sampling**: any ODE solver; midpoint/Heun typically beats Euler at equal NFE; typical budgets 5–50 steps, 1–4 after distillation/reflow. Optional stochasticity via Langevin addition; optional post-hoc scheduler swap.

**Where it's used**: SD3 / FLUX (rectified-flow [[DiT]] in [[LDM]]-style latent space), Movie Gen (video), Voicebox / Audiobox / F5-TTS (speech & audio), SE(3) FM for protein backbones (FoldFlow, FrameFlow, Multiflow), π0's flow-matching action expert for robot control ([[VLA]]), discrete FM for text.

## Relationship to Other Notes
---
- [[DDPM]] — diffusion = FM with a Gaussian path built from a forward SDE + score parameterization + inverted time; DDIM/probability-flow sampling = FM's ODE sampling
- [[Score-Based Generative Modeling Methods]] — the score is one of several equivalent parameterizations of the FM velocity for Gaussian paths (6.S184 Prop. 1); denoising score matching = conditional matching with ε-prediction; Fokker–Planck = the diffusion KFE
- [[LDM]] / [[DiT]] — SD3/FLUX pair FM objectives with DiT backbones in VAE latent space; the backbone is orthogonal to the objective
- [[VLA]] — π0 uses a flow-matching action expert to model continuous action chunks; the same CFM math with actions as the data point $z$
- [[ELBO]] — flows give exact unbiased likelihoods (vs. ELBOs); discrete FM's generalized-KL loss *is* an ELBO
- [[MCMC]] — Langevin and Metropolis–Hastings are divergence-free generator components; adding them to any FM/GM model preserves marginals (predictor–corrector sampling)
- [[VAE]] / [[RAE]] / [[FID]] — latent-space FM and evaluation of generative quality
