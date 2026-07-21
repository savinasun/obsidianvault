### [[dl_moc]]  
# **Flow Matching**

**Tags**:: #generative #dl_theo #diffusion
**Links**:: [[DDPM]] · [[Score-Based Generative Modeling Methods]] · [[Divergence Theorem]] · [[LDM]] · [[DiT]] · [[VAE]] · [[ELBO]] · [[MCMC]] · [[FID]] · [[RAE]] · [[VLA]]


**Flow Matching (FM)** is a simulation-free framework for training generative models: pick a probability path $p_t$ that interpolates a source distribution $p$ (noise) into the data distribution $q$, then train a neural velocity field to generate that path by simple regression. Sampling is ODE integration. It was introduced concurrently by [Lipman et al., 2022](https://arxiv.org/abs/2210.02747) (Flow Matching), [Liu et al., 2022](https://arxiv.org/abs/2209.03003) (Rectified Flow), and [Albergo & Vanden-Eijnden, 2022](https://arxiv.org/abs/2209.15571) (stochastic interpolants), and now powers state-of-the-art systems for images (SD3, [Esser et al., 2024](https://arxiv.org/abs/2403.03206)), video (Movie Gen), speech/audio (Voicebox, Audiobox), protein backbones (SE(3) FM), and robot policies (π0 — see [[VLA]]).

This note follows the comprehensive treatment in **"Flow Matching Guide and Code"** ([Lipman et al., 2024, arXiv 2412.06264](https://arxiv.org/abs/2412.06264), FAIR/MIT/Weizmann), which comes with the [`flow_matching` PyTorch library](https://github.com/facebookresearch/flow_matching). The background section follows the more introductory **MIT 6.S184 lecture notes** ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)).

**Time convention (important!)**: FM uses $t=0$ for **noise/source** and $t=1$ for **data/target** — the *opposite* of the diffusion convention in [[DDPM]] where $x_0$ is data. Diffusion time $r$ relates to FM time by a decreasing reparameterization $r = k(t)$, with data at $r=0$ and noise at $r \to \infty$.

## The FM Blueprint
---
1. **Choose** a probability path $(p_t)_{0\le t\le 1}$ with boundary conditions $p_0 = p$ (known source, e.g. $\mathcal{N}(0,I)$) and $p_1 = q$ (data).
2. **Train** a velocity field $u_t^\theta$ (neural network) to match the velocity $u_t$ known to generate $p_t$, via regression.
3. **Sample** by drawing $X_0 \sim p$ and solving the ODE $\frac{d}{dt}X_t = u_t^\theta(X_t)$ from $t=0$ to $1$.

The simplest instantiation fits in ~10 lines:

```python
# training step (linear / cond-OT path)
x_1 = sample_data(); x_0 = torch.randn_like(x_1); t = torch.rand(b, 1)
x_t  = (1 - t) * x_0 + t * x_1
loss = ((model(x_t, t) - (x_1 - x_0))**2).mean()

# sampling: integrate dx/dt = model(x, t) from t=0 to 1 (Euler / midpoint)
x = torch.randn(n, d)
for t in torch.linspace(0, 1, steps + 1)[:-1]:
    x = x + (1 / steps) * model(x, t)   # Euler; midpoint works better
```

*This background follows the MIT 6.S184 lecture notes ([Holderrieth & Erives, "An Introduction to Flow Matching and Diffusion Models"](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf), §2–4; [course site](https://diffusion.csail.mit.edu/) with recordings and labs) — a gentler build-up of the same machinery as Ch. 3 of the FM guide. Dictionary: their $p_\text{init}, p_\text{data}$ are this note's $p, q$; their data point $z$ is our $x_1$; their noise schedulers $(\alpha_t, \beta_t)$ are the guide's $(\alpha_t, \sigma_t)$.*

## Background: Flows, Velocity Fields, and the Continuity Equation
---
**Trajectory**. A time-dependent **vector field** $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$, $(x,t) \mapsto u_t(x)$, assigns a velocity to every point and time. It defines an **ODE**, whose solution from a start point $x_0$ is a **trajectory** $t \mapsto X_t$:
$$
\frac{d}{dt}X_t = u_t(X_t), \qquad X_0 = x_0 .
$$

**Flow**. Collecting the solutions from *every* start point into a single map gives the **flow** $\psi : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$:
$$
\frac{d}{dt}\psi_t(x_0) = u_t(\psi_t(x_0)), \qquad \psi_0(x_0) = x_0, \qquad X_t = \psi_t(X_0).
$$
Conversely a flow determines its field via $u_t(x) = \dot\psi_t(\psi_t^{-1}(x))$. In the 6.S184 slogan: ==*vector fields define ODEs whose solutions are flows*.== 

**Flow model**. A **flow model** turns this into a generative model by making *only the initial condition* random: $X_0 \sim p$, $\frac{d}{dt}X_t = u_t^\theta(X_t)$, with the goal $X_1 = \psi_1^\theta(X_0) \sim q$. The network parameterizes the **vector field, never the flow itself** — evaluating $\psi_t^\theta$ always means simulating the ODE.

**Lipschitz condition, existence, uniqueness.** A vector field is **Lipschitz** (globally) if $\|u_t(x) - u_t(y)\| \le L\,\|x - y\|$ for all $x, y$ — it changes at most linearly in space. This is precisely the hypothesis of the classical well-posedness theorem for ODEs:

**Picard–Lindelöf theorem** (Cauchy–Lipschitz; existence & uniqueness): *Let $u_t(x)$ be continuous in $t$ and Lipschitz in $x$ with constant $L$ uniform in $t$. Then for every initial condition $X_{t_0} = x_0$, the ODE $\frac{d}{dt}X_t = u_t(X_t)$ has **exactly one** solution — defined on all of $[0,1]$ in the globally Lipschitz case; if $u$ is only **locally** Lipschitz, the unique solution exists on a maximal time interval around $t_0$, which may end in finite-time blow-up.*

*Proof idea*: rewrite the initial-value problem as a fixed-point equation $X = \Phi X$ for the integral operator

$$
(\Phi X)_t = x_0 + \int_{t_0}^{t} u_s(X_s)\, ds .
$$

On a time window of length $h < 1/L$, $\Phi$ is a contraction on the space of continuous trajectories ($\|\Phi X - \Phi Y\|_\infty \le Lh\,\|X - Y\|_\infty$), so the Banach fixed-point theorem gives a unique fixed point, and the **Picard iteration** $X^{(k+1)} = \Phi X^{(k)}$ converges to it from any initial guess; chaining windows covers $[0,1]$. (Grönwall's inequality gives the same uniqueness plus the stability bound $\|\psi_t(x_0) - \psi_t(y_0)\| \le e^{Lt}\|x_0 - y_0\|$.)

Both hypotheses earn their keep. Weaken Lipschitz to mere continuity and existence survives (Peano's theorem) but **uniqueness fails**: $\dot x = \sqrt{|x|}$, $x(0) = 0$ is solved by both $x \equiv 0$ and $x(t) = t^2/4$ — the field has infinite slope at the origin. Keep only *local* Lipschitzness and solutions can **blow up** before $t=1$: $\dot x = x^2$, $x(0) = 1$ gives $x(t) = 1/(1-t)$, which escapes to infinity at $t = 1$.

6.S184's Thm 3 is the specialization used in the course: if $u_t(x)$ is continuously differentiable with bounded derivative (which implies Lipschitz with $L = \sup\|\partial_x u\|$), the ODE has a unique solution and $\psi_t$ is a **diffeomorphism**. Uniqueness is what makes flows well-behaved: two trajectories can never cross or merge at the same time — they would be two solutions of one initial-value problem — which is why $\psi_t$ is a bijection, and why the *marginal* FM velocity must average conditional velocities wherever conditional paths cross. Neural networks with Lipschitz activations satisfy the condition automatically, so the theorem is "good news rather than a concern" (6.S184). The FM guide's weaker hypothesis — *local* Lipschitzness plus integrability $\int_0^1 \mathbb{E}\|u_t(X_t)\|\,dt < \infty$ — rules out blow-up for almost every trajectory instead, which is what's needed near $t = 1$, where conditional velocities like $(x_1 - x)/(1-t)$ blow up (hence "generates" on $[0,1)$). Worked example (6.S184 Ex. 4): $u_t(x) = -\theta x$ has the explicit flow $\psi_t(x_0) = e^{-\theta t}x_0$.

**Simulation.** The **Euler method** updates $X_{t+h} = X_t + h\,u_t(X_t)$ with step size $h = 1/n$ ($o(h)$ local error). One rung up, **Heun's method** takes an Euler "guess" $X'_{t+h} = X_t + h\,u_t(X_t)$, then averages velocities at the current and guessed states: $X_{t+h} = X_t + \frac{h}{2}\big(u_t(X_t) + u_{t+h}(X'_{t+h})\big)$ — a predictor–corrector scheme; the FM guide's library defaults to the similar midpoint method. Higher-order solvers pay off exactly when trajectories curve.

**Stochastic process, diffusion coefficient, SDE.** A **stochastic process** $(X_t)_{0\le t\le 1}$ is a random trajectory — each simulation gives a different path. The building block is **Brownian motion** $W_t$ (a continuous random walk): $W_0 = 0$, Gaussian increments $W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$, independent across disjoint intervals. An **SDE** adds Brownian noise, scaled by a **diffusion coefficient** $\sigma_t \ge 0$, to the ODE's infinitesimal update — the symbolic notation

$$
dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t
$$

is shorthand for $X_{t+h} = X_t + h\,u_t(X_t) + \sigma_t(W_{t+h} - W_t) + h\,R_t(h)$ with a vanishing error term. Simulation by **Euler–Maruyama**: $X_{t+h} = X_t + h\,u_t(X_t) + \sqrt{h}\,\sigma_t\,\epsilon_t$, $\epsilon_t \sim \mathcal{N}(0, I_d)$. Randomness now enters the dynamics themselves, so there is no flow map. **Every ODE is an SDE with $\sigma_t = 0$** — in 6.S184's framing, a "diffusion model" is literally a flow model plus a diffusion coefficient. (Canonical example: the Ornstein–Uhlenbeck process $dX_t = -\theta X_t\,dt + \sigma\,dW_t$, which relaxes toward $\mathcal{N}(0, \tfrac{\sigma^2}{2\theta})$.)

**Probability paths: conditional and marginal.** A **probability path** $(p_t)_{0 \le t \le 1}$ is a time-indexed family of distributions — a trajectory in the space of distributions, prescribing *what distribution $X_t$ should have at each time*. A **conditional probability path** interpolates noise into a *single data point*:

$$
p_0(\cdot|z) = p, \qquad p_1(\cdot|z) = \delta_z \qquad \text{for all } z,
$$

and induces the **marginal probability path** by mixing over the data: $p_t(x) = \int p_t(x|z)\, q(z)\, dz$, which interpolates $p_0 = p$ to $p_1 = q$. Sampling from $p_t$ is easy (draw $z \sim q$, then $x \sim p_t(\cdot|z)$) even though its density is intractable. The dominant example is the **Gaussian path** with noise schedulers $(\alpha_t, \beta_t)$, $\alpha_0 = \beta_1 = 0$, $\alpha_1 = \beta_0 = 1$:

$$
p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\, \beta_t^2 I_d), \qquad x_t = \alpha_t z + \beta_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_d).
$$

**Conditional vector field.** A conditional vector field $u_t^\text{target}(x|z)$ is any field whose ODE transports $p$ along the conditional path: $X_0 \sim p,\ \frac{d}{dt}X_t = u_t^\text{target}(X_t|z) \Rightarrow X_t \sim p_t(\cdot|z)$. It can typically be derived *by hand*: for the Gaussian path, the conditional flow $\psi_t(x|z) = \alpha_t z + \beta_t x$ gives (6.S184 Ex. 10)

$$
u_t^\text{target}(x|z) = \Big(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\Big) z + \frac{\dot\beta_t}{\beta_t}\, x .
$$

Alone it is useless — its ODE just collapses everything onto $z$ — but it is the *building block* of the marginal field: the next section shows the marginal velocity is the posterior-weighted average of conditional ones (the Marginalization Trick), and that is what FM trains.

**The Continuity Equation (Mass Conservation).** Say $u_t$ **generates** $p_t$ if $X_t \sim p_t$ for all $t$ whenever $X_0 \sim p_0$. The central characterization (6.S184 Thm 11; = the guide's Mass Conservation theorem) is: this holds **if and only if**

$$
\partial_t\, p_t(x) = -\,\mathrm{div}\big(p_t u_t\big)(x) \qquad \text{for all } x,\ t.
$$

Intuition (6.S184): $\partial_t p_t(x)$ is the local change of probability at $x$; divergence measures net *outflow* of a field, so $-\mathrm{div}(p_t u_t)$ is the **net inflow of probability mass** carried by the probability flux $j_t = p_t u_t$; since total mass is conserved (always integrates to 1), the two sides must agree — the [[Divergence Theorem]] in differential form. This single PDE licenses everything that follows: the Marginalization Trick is proven by checking that the marginal pair $(p_t, u_t)$ satisfies it.

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
The FM loss regresses onto the target velocity along the path:

$$
\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, X_t \sim p_t}\, D\big(u_t(X_t),\, u_t^\theta(X_t)\big),
$$

with $D$ a dissimilarity like $\|u - v\|^2$. This is intractable — $u_t$ marginalizes over the whole dataset. The fix is **conditioning**.

**Marginal path from conditional paths.** Condition on any random variable $Z$ (typically $Z = X_1$, but also $Z = X_0$ or $Z = (X_0, X_1)$ — all essentially equivalent):

$$
p_t(x) = \int p_{t|Z}(x|z)\, p_Z(z)\, dz, \qquad
u_t(x) = \mathbb{E}\big[u_t(X_t|Z)\,\big|\,X_t = x\big],
$$

i.e. the marginal velocity is the **posterior-weighted average** of conditional velocities $u_t(x|z)$, each of which generates its simple conditional path $p_{t|Z}(\cdot|z)$.

**Marginalization Trick (Theorem 3 of the guide)**: under mild regularity (C¹ conditional paths/velocities, bounded support of $p_Z$, $p_t > 0$), if $u_t(\cdot|z)$ generates $p_{t|Z}(\cdot|z)$, then the marginal $u_t$ generates the marginal $p_t$. Proof: plug into the Continuity Equation and swap integral/divergence.

**Conditional FM loss** — the tractable objective:

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\,Z,\,X_t \sim p_{t|Z}}\, D\big(u_t(X_t|Z),\, u_t^\theta(X_t)\big),
\qquad \nabla_\theta \mathcal{L}_\text{FM} = \nabla_\theta \mathcal{L}_\text{CFM}.
$$

**Why gradients coincide — Bregman divergences.** The equality holds for any **Bregman divergence** $D(u,v) = \Phi(u) - \Phi(v) - \langle u - v, \nabla\Phi(v)\rangle$ ($\Phi$ strictly convex; squared Euclidean is $\Phi = \|\cdot\|^2$). Bregman divergences have affine-invariant gradients in the second argument, so $\nabla_v D(\mathbb{E}[Y], v) = \mathbb{E}[\nabla_v D(Y, v)]$ — expectation and gradient can be swapped. General principle (Proposition 1): for any $f_t(X_0, X_1)$, the conditional matching loss $\mathbb{E}\, D(f_t(X_0,X_1), g_t^\theta(X_t))$ has minimizer $g_t^\theta(x) = \mathbb{E}[f_t(X_0,X_1) \mid X_t = x]$ — **regression targets can be single samples; the network learns their conditional expectation**. Notably, Bregman-ness is *necessary* for this property (Holderrieth et al., 2024).

**Time weighting**: $t$ need not be uniform — training with $t \sim \omega(t)$ (equivalently reweighting the loss) changes nothing mathematically, but sampling $t$ non-uniformly (e.g. SD3's logit-normal) works better in practice at scale.

## Designing Paths via Conditional Flows
---
Rather than hand-designing $p_{t|1}$, define a **conditional flow** $\psi_t(x|x_1)$ with boundary conditions $\psi_0(x|x_1) = x$, $\psi_1(x|x_1) = x_1$ (smooth, diffeomorphic in $x$). Then $X_{t|1} = \psi_t(X_0|x_1)$ defines both the conditional path and its velocity, and the CFM loss simplifies to

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\,(X_0,X_1) \sim \pi_{0,1}}\, D\big(\dot\psi_t(X_0|X_1),\, u_t^\theta(X_t)\big),
$$

— the regression target is just the **time-derivative of the interpolant**, no densities needed.

### The conditional-OT (linear) path
Minimizing a Jensen upper bound on the **kinetic energy** $\int_0^1 \mathbb{E}\|u_t(X_t)\|^2 dt$ over conditional flows gives (Euler–Lagrange: $\ddot\gamma_t = 0$) straight lines:

$$
\psi_t(x|x_1) = t\,x_1 + (1-t)\,x
\;\;\Rightarrow\;\;
X_t = t X_1 + (1-t) X_0, \quad
u_t(x|x_1) = \frac{x_1 - x}{1-t}, \quad
\dot\psi_t(X_0|X_1) = X_1 - X_0.
$$

With Gaussian source this makes $p_{t|1}(x|x_1) = \mathcal{N}(x \mid t x_1, (1-t)^2 I)$ and yields the simplest FM loss $\mathbb{E}\|u_t^\theta(X_t) - (X_1 - X_0)\|^2$.

Facts worth remembering:
- If $q = \delta_{x_1}$ (single point), the linear conditional flow **is** the exact dynamic optimal transport solution.
- In general the *marginal* flow is **not** OT — the linear choice only minimizes a bound; the marginal kinetic energy is bounded by the coupling cost $\mathbb{E}\|X_1 - X_0\|^2$.
- **Conditional** straight lines ≠ **marginal** straight trajectories: marginal paths of independent couplings still curve (conditional lines cross), which is what multisample couplings and reflow address (below).

## Affine Paths, Schedulers, and Parameterizations
---
The linear path generalizes to **affine conditional flows** $\psi_t(x|x_1) = \alpha_t x_1 + \sigma_t x$ with a **scheduler** $(\alpha_t, \sigma_t)$ satisfying $\alpha_0 = 0,\ \alpha_1 = 1,\ \sigma_0 = 1,\ \sigma_1 = 0$, $\dot\alpha_t > 0$, $\dot\sigma_t < 0$. The CFM target becomes $\dot\alpha_t X_1 + \dot\sigma_t X_0$. Examples: cond-OT ($\alpha_t = t, \sigma_t = 1-t$), polynomial ($t^n$), LinearVP ($\alpha_t = t, \sigma_t = \sqrt{1-t^2}$), cosine ($\alpha_t = \sin\tfrac{\pi t}{2}, \sigma_t = \cos\tfrac{\pi t}{2}$). Diffusion's VP/VE schedules are (approximate) members: VP has $\alpha_t^2 + \sigma_t^2 = 1$ with exponential form; VE has $\alpha_t \equiv 1$, $\sigma_0 \gg 1$ — both only *approximately* hit a Gaussian at $t = 0$.

**Equivalent parameterizations.** Using $X_t = \alpha_t X_1 + \sigma_t X_0$, the marginal velocity $u_t(x) = \dot\alpha_t\,\mathbb{E}[X_1|X_t{=}x] + \dot\sigma_t\,\mathbb{E}[X_0|X_t{=}x]$ can be rewritten via:
- **$x_1$-prediction** (denoiser / target prediction): $u_t(x) = \frac{\dot\sigma_t}{\sigma_t}x + \big(\dot\alpha_t - \alpha_t\frac{\dot\sigma_t}{\sigma_t}\big)\,x_{1|t}(x)$
- **$x_0$-prediction** (= diffusion's noise/ε-prediction): $u_t(x) = \frac{\dot\alpha_t}{\alpha_t}x + \big(\dot\sigma_t - \sigma_t\frac{\dot\alpha_t}{\alpha_t}\big)\,x_{0|t}(x)$
- **score** (Gaussian paths): see below.

All are related by closed-form conversions (Table 1 of the guide) and each is learnable with a conditional matching loss ($f_t(X_0,X_1) = X_1$, $X_0$, etc.). Caveat: the conversions have **singularities** at the endpoints ($x_1$-pred blows up as $t \to 1$, $x_0$-pred/score as $t \to 0$); direct velocity prediction avoids them.

**Gaussian paths and the score.** For $p_{t|1}(x|x_1) = \mathcal{N}(x|\alpha_t x_1, \sigma_t^2 I)$:

$$
\nabla \log p_{t|1}(x|x_1) = -\frac{x - \alpha_t x_1}{\sigma_t^2}, \qquad
\nabla \log p_t(x) = -\frac{x_{0|t}(x)}{\sigma_t},
$$

so the score is just rescaled noise-prediction, and the marginal velocity is

$$
u_t(x) = \frac{\dot\alpha_t}{\alpha_t}\,x + \Big(\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2 - \dot\sigma_t \sigma_t\Big) \nabla \log p_t(x).
$$

This expression is a **gradient field**, hence kinetically optimal *for that fixed Gaussian path* (it's the minimum-energy velocity generating $p_t$).

**Post-training scheduler change.** Any two affine paths with the same coupling are related by a **scale-time transformation** $\bar\psi_r = s_r\,\psi_{t_r}$ with $t_r = \rho^{-1}(\bar\rho(r))$ matched via the signal-to-noise ratio $\rho_t = \alpha_t/\sigma_t$, giving $\bar u_r(x) = \frac{\dot s_r}{s_r}x + s_r \dot t_r\, u_{t_r}(x / s_r)$. Consequences: a trained model's scheduler can be **swapped at inference** (e.g. VP → cond-OT), and *all schedulers produce the same final map* $\psi_1$ in continuous time — scheduler choice matters only through discretization error and learning dynamics.

## Data Couplings
---
FM trains on pairs $(X_0, X_1) \sim \pi_{0,1}$; the default is the **independent coupling** $\pi_{0,1} = p \otimes q$, but this is a design axis:

- **Paired data** ([Liu et al., 2023](https://arxiv.org/abs/2302.05872) I²SB; Albergo et al., 2024): for inpainting/super-resolution/deblurring, build $\pi_{0,1}(x_0, x_1) = \pi_{0|1}(x_0|x_1)q(x_1)$ by corrupting data samples (+ noise so the source has a density). FM then learns the restoration map directly — no guidance needed.
- **Multisample / minibatch-OT couplings** ([Pooladian et al., 2023](https://arxiv.org/abs/2304.14772); [Tong et al., 2023](https://arxiv.org/abs/2302.00482)): draw $k$ source and $k$ target samples, solve the discrete OT assignment within the batch, train on matched pairs. This provably lowers transport cost vs. independent coupling and **straightens marginal trajectories** (→ fewer ODE steps); as $k \to \infty$ it approaches true OT.

Related (beyond the guide): **Rectified Flow's reflow** ([Liu et al., 2022](https://arxiv.org/abs/2209.03003)) replaces the coupling with $(X_0, \hat X_1)$ pairs generated by simulating the trained model, iteratively straightening paths toward 1–2-step generation; this underlies SD3's and FLUX's fast variants.

## Guidance
---
With labels/conditioning $y$, train a **guided velocity** $u_t^\theta(x|y)$ with the same CFM loss (the theory carries through per-$y$). For Gaussian paths, velocity ↔ score conversions transfer classifier and classifier-free guidance from diffusion ([Zheng et al., 2023](https://arxiv.org/abs/2311.13443)):

$$
\tilde u_t^\theta(x|y) = (1 - w)\, u_t^\theta(x|\varnothing) + w\, u_t^\theta(x|y),
$$

with guidance scale $w > 1$, training with condition dropout $\xi \sim \text{Bernoulli}(p_\text{uncond})$ — exactly CFG, applied to velocities. (What distribution CFG actually samples remains theoretically murky; it is nonetheless the standard for large-scale text-to-image FM like SD3.)

## Diffusion Models as a Special Case
---
The guide's unifying claim (§10): **a diffusion model is an FM model with a specific Gaussian path and score parameterization** (plus inverted time).

- The **forward noising SDE** $dX_r = a_r X_r\,dr + g_r\,dW_r$ (affine drift, so marginals are Gaussian in closed form) is just a way of *constructing* the probability path: $p_{t|1}(x|x_1) = \mathcal{N}(\alpha_t x_1, \sigma_t^2 I)$ with $\alpha, \sigma$ determined by $a, g$.
- **Denoising score matching** (the DDPM loss) is precisely the conditional matching loss for $x_0$-prediction, reparameterized by $s_t^\theta = -x_{0|t}^\theta / \sigma_t$; its minimizer is the marginal score $\nabla \log p_t$.
- The **probability-flow ODE** of [Song et al., 2021](https://arxiv.org/abs/2011.13456) is exactly the FM marginal velocity written via the score; deterministic diffusion sampling = FM sampling.
- **Stochastic (SDE) sampling** = adding **Langevin dynamics** $\tfrac{1}{2}\beta_t^2 \nabla \log p_t\,dt + \beta_t dW_t$, which is *divergence-free* w.r.t. $p_t$: any $\beta_t \ge 0$ preserves the marginals. The classical Anderson time-reversal is one particular $\beta_t$ — with a perfect model all choices are equivalent; with errors, an empirically tuned noise level can help (Karras et al., 2022).
- **Time-reversal is not needed**: "find a process with the forward marginals reversed" ≡ "find a velocity generating $p_t$" ≡ "solve the Continuity/Fokker–Planck equation" — FM solves the third problem directly.

| | Diffusion ([[DDPM]]) | Flow Matching (cond-OT) |
|---|---|---|
| **Path construction** | forward noising SDE, closed-form marginals | direct choice of conditional flow |
| **Network target** | noise $\epsilon$ / score | velocity $X_1 - X_0$ (or any parameterization) |
| **Boundary at noise end** | Gaussian only as $r \to \infty$ (approx.) | exactly $p$ at $t = 0$ |
| **Schedule** | required, baked into training | scheduler swappable post-training |
| **Sampling** | reverse SDE or prob-flow ODE | ODE (SDE optional via Langevin) |
| **Likelihood** | ELBO (SDE) / exact (ODE) | exact, unbiased estimator |
| **Endpoint singularities** | score/ε-pred singular near noise | velocity prediction non-singular |

Practical upshot: with straight(er) conditional paths, Euler/midpoint integration needs far fewer function evaluations (local truncation error scales with trajectory curvature); perfectly straight marginal flows would sample in one step — the motivation for OT couplings and reflow.

## Extension: Riemannian Flow Matching
---
FM transfers to Riemannian manifolds $\mathcal{M}$ (sphere, tori, SE(3) — proteins, molecules, geoscience) ([Chen & Lipman, 2024](https://arxiv.org/abs/2302.03660)). Velocities live in tangent spaces, the continuity equation uses the Riemannian divergence, and the Marginalization Trick + CFM gradient equivalence hold verbatim with tangent-space Bregman divergences. The conditional flow generalizes straight lines to **geodesic interpolation**:

$$
\psi_t(x_0|x_1) = \exp_{x_0}\!\big(\kappa(t)\, \log_{x_0}(x_1)\big), \qquad \kappa(0)=0,\ \kappa(1)=1,
$$

which is **simulation-free** whenever exp/log maps are closed-form (simple manifolds) — a real advantage over Riemannian diffusion, which needs in-training simulation and score approximations. For hard geometries, geodesics can be replaced by any smooth **premetric** $d(x, y)$ (e.g. spectral distances) with the minimal-norm velocity $u_t(x|x_1) = \frac{d \log \bar\kappa(t)}{dt}\, d(x, x_1) \frac{\nabla d(x, x_1)}{\|\nabla d(x, x_1)\|_g^2}$.

## Extension: Discrete Flow Matching (CTMC)
---
For discrete data $x = (x^1, \dots, x^d) \in \mathcal{T}^d$ (tokens, vocabulary size $K$), the flow is replaced by a **Continuous-Time Markov Chain** ([Campbell et al., 2024](https://arxiv.org/abs/2402.04997); [Gat et al., 2024](https://arxiv.org/abs/2407.15595)): transitions $p_{t+h|t}(y|x) = \delta(y,x) + h\,u_t(y,x) + o(h)$, where the velocities are now **rates** satisfying $u_t(y,x) \ge 0$ for $y \ne x$ and $\sum_y u_t(y,x) = 0$. The Continuity Equation becomes the **Kolmogorov equation** $\frac{d}{dt}p_t(y) = \sum_x u_t(y,x)\,p_t(x)$, and the whole FM recipe transfers: marginalization trick, conditional velocities, Bregman CFM loss.

Key design points:
- **Factorized velocities**: modeling rates over all $K^d$ states is infeasible; restrict to transitions changing **at most one token**, $u_t(y,x) = \sum_i \delta(y^{\bar i}, x^{\bar i})\,u_t^i(y^i, x)$ — output dimension $d \cdot K$, and coordinate-wise Euler sampling is valid to $o(h)$.
- **Mixture paths**: per token, $X_t^i = x_1^i$ with probability $\kappa_t$, else $x_0^i$ ($\kappa_0 = 0, \kappa_1 = 1$), with closed-form conditional rates $u_t^i(y^i, x^i|x_0, x_1) = \frac{\dot\kappa_t}{1-\kappa_t}\big[\delta(y^i, x_1^i) - \delta(y^i, x^i)\big]$. Source: uniform noise or a **mask token** $m$ (mask-DFM is the discrete analog of noise→data).
- **Posterior parameterization**: learn $p_{1|t}^{\theta,i}(x_1^i | x)$ — a per-token denoiser predicting clean tokens given the current sequence, i.e. discrete $x_1$-prediction. Trained with cross-entropy (KL Bregman) or **generalized KL**, which doubles as an [[ELBO]] on target likelihood for evaluation.
- **Sampling**: at each step, per token, sample $X_1^i \sim p^{\theta,i}_{1|t}(\cdot|x)$, then take the CTMC Euler step (decide jump-to-target vs. stay). Divergence-free rate components can be added freely (corrector-style noise ↔ un-masking/re-masking trade-offs).

This is the FM view of "discrete diffusion" language models and multimodal protein co-design.

## Extension: Generator Matching — One Framework for Everything
---
**Generator Matching (GM)** ([Holderrieth et al., 2024](https://arxiv.org/abs/2410.20587)) abstracts FM to *any* Continuous-Time Markov Process on *any* state space. Every well-behaved (Feller) Markov process is characterized by its **generator** $\mathcal{L}_t$, the first-order derivative of its transition kernel acting on test functions: $\mathbb{E}[f(X_{t+h})|X_t = x] = f(x) + h\,[\mathcal{L}_t f](x) + o(h)$.

| Process | Generator $\mathcal{L}_t f$ | Parameterized by | KFE ≙ |
|---|---|---|---|
| Flow (ODE) | $\nabla f^\top u_t$ | velocity $u_t$ | Continuity Eq. |
| Diffusion (SDE) | $\frac{1}{2}\sigma_t^2 \cdot \nabla^2 f$ | diffusion coeff $\sigma_t^2$ | Fokker–Planck Eq. |
| Jump process | $\int (f(y) - f(x))\,Q_t(dy, x)$ | jump kernel $Q_t$ | Jump continuity Eq. |
| CTMC (discrete) | $f^\top u_t$ | rate matrix $u_t$ | Kolmogorov Eq. |

The marginals of any such process obey the **Kolmogorov Forward Equation** $\frac{d}{dt}\langle p_t, f\rangle = \langle p_t, \mathcal{L}_t f \rangle$, which unifies all the "mass conservation" PDEs above. The FM recipe then holds in full generality: conditional generators (found by solving the KFE analytically for simple conditional paths) marginalize under the posterior $\mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)}$, and a conditional Bregman loss on the generator's linear parameterization has unbiased gradients — FM, diffusion, and discrete FM are all instances.

Because generators are **linear** and the KFE is a linear equation, trained models can be *combined*:
1. **Markov superposition**: $\alpha_t \mathcal{L}_t + (1-\alpha_t)\mathcal{L}_t'$ of two generators for the same path still generates it (e.g. jump + flow hybrids).
2. **Divergence-free components**: add any $\mathcal{L}^\text{div}_t$ with $\langle p_t, \mathcal{L}^\text{div}_t f\rangle = 0$ — Langevin dynamics and Metropolis–Hastings kernels ([[MCMC]]) are exactly such components, which is where diffusion's stochastic samplers and predictor–corrector schemes come from.
3. **Multimodal models**: factorized paths over product state spaces let continuous FM (images) and discrete FM (text) train jointly in one model.

## Practical Summary
---
**Training** (cond-OT, the default): sample $x_1 \sim q$, $x_0 \sim \mathcal{N}(0,I)$, $t$ (uniform or logit-normal); regress $u^\theta(x_t, t)$ onto $x_1 - x_0$ at $x_t = t x_1 + (1-t) x_0$. No noise schedule, no SNR weighting; any architecture ([[U-Net]], [[DiT]]) works — only the target changes.

**Sampling**: any ODE solver; midpoint typically beats Euler at equal NFE; typical budgets 5–50 steps, 1–4 after distillation/reflow. Optional stochasticity via Langevin addition; optional post-hoc scheduler swap.

**Where it's used**: SD3 / FLUX (rectified-flow [[DiT]] in [[LDM]]-style latent space), Movie Gen (video), Voicebox / Audiobox / F5-TTS (speech & audio), SE(3) FM for protein backbones (FoldFlow, FrameFlow, Multiflow), π0's flow-matching action expert for robot control ([[VLA]]), discrete FM for text.

## Relationship to Other Notes
---
- [[DDPM]] — diffusion = FM with a Gaussian path built from a forward SDE + score parameterization + inverted time; DDIM/probability-flow sampling = FM's ODE sampling
- [[Score-Based Generative Modeling Methods]] — the score is one of several equivalent parameterizations of the FM velocity for Gaussian paths; denoising score matching = conditional matching with $x_0$-prediction; Fokker–Planck = the diffusion KFE
- [[LDM]] / [[DiT]] — SD3/FLUX pair FM objectives with DiT backbones in VAE latent space; the backbone is orthogonal to the objective
- [[VLA]] — π0 uses a flow-matching action expert to model continuous action chunks; the same CFM math with actions as $x_1$
- [[ELBO]] — flows give exact unbiased likelihoods (vs. ELBOs); discrete FM's generalized-KL loss *is* an ELBO
- [[MCMC]] — Langevin and Metropolis–Hastings are divergence-free generator components; adding them to any FM/GM model preserves marginals (predictor–corrector sampling)
- [[VAE]] / [[RAE]] / [[FID]] — latent-space FM and evaluation of generative quality
