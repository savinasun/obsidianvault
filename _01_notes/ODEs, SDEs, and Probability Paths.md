### [[dl_moc]]  
# **ODEs, SDEs, and Probability Paths**

**Tags**:: #math #dl_theo #generative
**Links**:: [[Flow Matching]] · [[Score Matching]] · [[DDPM]] · [[Generative Modeling]] · [[Divergence Theorem]] · [[CTMC]] · [[Discrete Flow Matching]] · [[MCMC]] · [[ELBO]] · [[Robot Kinematics]]


The mathematical machinery behind flow- and diffusion-based generative models: vector fields and the ODEs/flows they generate, well-posedness (Picard–Lindelöf), numerical solvers, SDEs, probability paths, conditional and marginal vector fields, and the continuity / Fokker–Planck equations that tie dynamics to distributions. Follows the **[MIT 6.S184](https://diffusion.csail.mit.edu/2026/index.html) lecture notes** ([Holderrieth & Erives](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)) and the **"Flow Matching Guide and Code"** ([Lipman et al., 2024, arXiv 2412.06264](https://arxiv.org/abs/2412.06264)); this is the background section of [[Flow Matching]], split out as a standalone reference.

**Notation** (6.S184 convention, used throughout the FM/SM notes): source $X_0 \sim p_\text{init} = \mathcal{N}(0, I_d)$, data point $z\sim p_\text{data}$; noise $\epsilon \sim \mathcal{N}(0, I_d)$; schedulers $(\alpha_t, \beta_t)$; target fields $u_t^\text{target}(x)$, $u_t^\text{target}(x|z)$; network $u_t^\theta$.  **FM time runs noise → data** ($t=0$ noise, $t=1$ data) — the *reverse* of [[DDPM]] time, related by a decreasing reparameterization $r = k(t)$. The FM guide writes $p, q$; $x_1$; $x_0$; $(\alpha_t, \sigma_t)$; $u_t(x|x_1)$ — translate when reading it.

## Probability Paths
---
A **probability path** $(p_t)_{0\le t\le 1}$ is a time-indexed family of distributions — a trajectory in distribution space, prescribing what law $X_t$ should have at each time.

**<span style="color:rgb(255, 255, 0)"><b>Conditional probability path.</b></span>** Interpolates noise into a *single data point* $z$, ending at the Dirac $\delta_z$:
$$
p_0(\cdot|z) = p_\text{init}, \qquad p_1(\cdot|z) = \delta_z \qquad \forall z .
$$

**<span style="color:rgb(255, 255, 0)"><b>Marginal probability path.</b></span>** Mix over the data: $p_t(x) = \int p_t(x|z)\, p_\text{data}(z)\, dz$, interpolating $p_\text{init} \to p_\text{data}$. Sampling from $p_t$ is easy ($z \sim p_\text{data}$, then $x \sim p_t(\cdot|z)$) even though its density is intractable. E.g. the **Gaussian path** with schedulers $(\alpha_t, \beta_t)$, $\alpha_0 = \beta_1 = 0$, $\alpha_1 = \beta_0 = 1$:
$$
p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\, \beta_t^2 I_d), \qquad x_t = \alpha_t z + \beta_t \epsilon .
$$

## Flow Models
---
**<span style="color:rgb(255, 255, 0)"><b>Vector field.</b></span>** A time-dependent **vector field** $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ defines an <b><span style="color:rgb(255, 255, 0)">ODE</span></b> (瞬时位移=瞬时速度$\times$时间); its solution from a start point is a <span style="color:rgb(255, 255, 0)"><b>trajectory</b></span> $t \mapsto X_t$:
$$
\frac{d}{dt}X_t = u_t(X_t), \qquad X_0 = x_0 .
$$

<span style="color:rgb(255, 255, 0)"><b><span style="color:rgb(255, 255, 0)">Flow.</b></span></span> Collecting solutions from *every* start point gives the **flow** $\psi_t$, with $X_t = \psi_t(X_0)$:
$$
\frac{d}{dt}\psi_t(x_0) = u_t(\psi_t(x_0)), \qquad \psi_0(x_0) = x_0 .
$$
Conversely $u_t(x) = \dot\psi_t(\psi_t^{-1}(x))$. Vector fields define ODEs whose solutions are flows.

<span style="color:rgb(255, 255, 0)"><b>Flow model.</b></span> Make *only the initial condition* random: $$X_0 \sim p_\text{init},\quad \frac{d}{dt}X_t = u_t^\theta(X_t)$$goal $X_1 = \psi_1^\theta(X_0) \sim p_\text{data}$. The network parameterizes the **vector field, never the flow** — evaluating $\psi_t^\theta$ always means simulating the ODE.

<span style="color:rgb(255, 255, 0)"><b>Continuous Normalizing Flow (CNF).</b></span> The flow model seen as a **normalizing flow** — a model that pushes $p_\text{init}$ through an invertible map $f^\theta$ and evaluates densities by change of variables,
$$
\log p_1(x) = \log p_\text{init}\big((f^\theta)^{-1}(x)\big) + \log \big|\det \partial_x (f^\theta)^{-1}(x)\big| ,
$$
which in discrete normalizing flows ([Rezende & Mohamed, 2015](https://arxiv.org/abs/1505.05770)) forces architectures with cheap Jacobian determinants (coupling layers; $O(d^3)$ in general). A **CNF** ([Chen et al., 2018](https://arxiv.org/abs/1806.07366), Neural ODEs) instead takes $f^\theta = \psi_1^\theta$, the flow of a learned ODE: invertibility comes for free (Picard–Lindelöf — invert by running the ODE backwards in time), $u_t^\theta$ is architecturally unconstrained, and the Jacobian log-determinant collapses into a divergence integral along the trajectory — the *instantaneous change of variables* of [[#Exact Likelihoods]]. Classical CNFs (FFJORD, [Grathwohl et al., 2018](https://arxiv.org/abs/1810.01367)) are trained by maximizing this exact likelihood, which puts ODE simulation + divergence estimation *inside* the training loop; [[Flow Matching]] trains the very same model class simulation-free — an FM model **is** a CNF, only the training signal differs.

## Lipschitz Condition
---
<span style="color:rgb(255, 255, 0)"><b>Lipschitz condition.</b></span> $u_t$ is globally **Lipschitz** if $\|u_t(x) - u_t(y)\| \le L\,\|x - y\|$ for all $x, y$.

<span style="color:rgb(255, 255, 0)"><b>Picard–Lindelöf theorem.</b></span> If $u_t(x)$ is continuous in $t$ and Lipschitz in $x$ (constant $L$ uniform in $t$), the ODE has exactly one solution through every initial condition — on all of $[0,1]$ in the globally Lipschitz case; only on a maximal interval (possible finite-time blow-up) if merely locally Lipschitz.
	Both hypotheses earn their keep: mere continuity keeps existence (Peano) but **uniqueness fails** ($\dot x = \sqrt{|x|},\ x(0)=0$ is solved by $x \equiv 0$ *and* $x = t^2/4$); local-only Lipschitzness allows **blow-up** ($\dot x = x^2,\ x(0)=1$ gives $x = 1/(1-t)$, gone at $t=1$).

<span style="color:rgb(255, 255, 0)"><b>Diffeomorphism</b></span> (微分同胚): if $u_t$ is $C^1$ with bounded derivative (⟹ Lipschitz, $L = \sup\|\partial_x u\|$), the flow exists, is unique, and $\psi_t$ is a **diffeomorphism**. 
	<span style="color:rgb(0, 176, 240)">Uniqueness is what makes flows well-behaved: two trajectories can never cross or merge at the same time</span> — they would be two solutions of one initial-value problem — <span style="color:rgb(0, 176, 240)">which is why</span> $\psi_t$ <span style="color:rgb(0, 176, 240)">is a bijection</span>, and why the *marginal* FM velocity must average conditional velocities wherever conditional paths cross. Neural networks with Lipschitz activations satisfy this automatically. 
	The FM guide's weaker hypothesis — *local* Lipschitzness + integrability $\int_0^1 \mathbb{E}\|u_t(X_t)\|\,dt < \infty$ — rules out blow-up for almost every trajectory instead, which is what's needed near $t=1$ where conditional velocities like $(z-x)/(1-t)$ blow up (hence "generates" on $[0,1)$).

## Numerical Simulation
---
<span style="color:rgb(255, 255, 0)"><b>Euler method.</b></span> $X_{t+h} = X_t + h\,u_t(X_t)$, $h = 1/n$ ($o(h)$ local error). 
<span style="color:rgb(255, 255, 0)"><b>Heun method.</b></span> <span style="color:rgb(255, 255, 0)">(predictor–corrector)</span> predict $X'_{t+h} = X_t + h\,u_t(X_t)$, then average velocities, $X_{t+h} = X_t + \frac{h}{2}\big(u_t(X_t) + u_{t+h}(X'_{t+h})\big)$ — second-order, like the midpoint method used in the guide's library.

## From ODEs to SDEs
---
**Stochastic process, diffusion coefficient, SDE.** A **stochastic process** $(X_t)$ is a random trajectory. An **SDE** adds Brownian noise, scaled by a **diffusion coefficient** $\sigma_t \ge 0$:
$$
dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t,
$$
shorthand for $X_{t+h} = X_t + h\,u_t(X_t) + \sigma_t(W_{t+h} - W_t) + h\,R_t(h)$. Simulate with **Euler–Maruyama**: $X_{t+h} = X_t + h\,u_t(X_t) + \sqrt{h}\,\sigma_t\,\epsilon_t$. Randomness enters the dynamics, so there is no flow map. **Every ODE is an SDE with $\sigma_t = 0$** — a "diffusion model" is literally a flow model plus a diffusion coefficient.

## Vector Fields for Probability Paths
---
<span style="color:rgb(255, 255, 0)"><b>Conditional vector field.</b></span> Any field whose ODE transports $p_\text{init}$ along the conditional path: $X_0 \sim p_\text{init},\ \frac{d}{dt}X_t = u_t^\text{target}(X_t|z) \Rightarrow X_t \sim p_t(\cdot|z)$. Typically derivable *by hand*: the Gaussian conditional flow $\psi_t(x|z) = \alpha_t z + \beta_t x$ gives (6.S184 Ex. 10)
$$
u_t^\text{target}(x|z) = \Big(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\Big) z + \frac{\dot\beta_t}{\beta_t}\, x .
$$
Alone it just collapses everything onto $z$ — but it is the building block of what [[Flow Matching]] actually trains.

<span style="color:rgb(255, 255, 0)"><b>Marginal vector field.</b></span> The posterior-weighted average of conditional fields — for each $z$, the direction toward $z$, weighted by how much we believe $x$ came from $z$:
$$
u_t^\text{target}(x) = \int u_t^\text{target}(x|z)\, \frac{p_t(x|z)\,p_\text{data}(z)}{p_t(x)}\, dz = \mathbb{E}\big[u_t^\text{target}(x_t|z)\,\big|\,x_t = x\big].
$$
This is the FM training target: its flow turns noise into data.

<span style="color:rgb(255, 255, 0)"><b>Marginalization trick.</b></span> If $u_t^\text{target}(\cdot|z)$ generates $p_t(\cdot|z)$, then the marginal $u_t^\text{target}$ generates the marginal $p_t$ (under mild regularity: C¹ paths/velocities, bounded data support, $p_t > 0$). Proof: plug the mixture into the Continuity Equation and swap integral/divergence. The guide generalizes the conditioning from $z$ to any $Z$ — source samples or two-sided pairs $(x_0, z)$ — all essentially equivalent.

## The Continuity Equation and Fokker–Planck Equation
---
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

## Discrete Probability Paths (CTMC)
---
On a **discrete state space** $S = \mathcal{V}^d$ (sequences of $d$ tokens from a vocabulary $\mathcal{V}$ of size $V$), the dynamics object is a **rate matrix** $Q_t(y|x)$ and the process a [[CTMC]]; the generative construction mirrors the continuous case, with PMFs replacing densities.

**Conditional probability path.** A family of PMFs $p_t(\cdot|z)$ on $S$, indexed by a data point $z \in S$ and $t \in [0,1]$, interpolating noise into that single data point:

$$
p_0(\cdot|z) = p_\text{init}, \qquad p_1(\cdot|z) = \delta_z \qquad \forall z \in S .
$$

The standard choice is the <span style="color:rgb(255, 255, 0)"><b>factorized mixture path</b></span> — each token independently revealed with probability $\kappa_t$ (a monotone schedule with $\kappa_0 = 0$, $\kappa_1 = 1$):

$$
p_t(x \mid z) = \prod_{j=1}^{d} \Big[ \kappa_t\, \delta_{z^j}(x^j) + (1-\kappa_t)\, p_\text{init}(x^j) \Big],
$$

with the **masked** special case $p_\text{init} = \delta_{[\text{mask}]}$ (masked diffusion: tokens are either revealed or `[mask]`).

**Marginal probability path.** Mix the conditional paths over the data:

$$
p_t(x) = \sum_{z \in S} p_t(x \mid z)\, p_\text{data}(z),
$$

so that $p_0 = p_\text{init}$ and $p_1 = p_\text{data}$. Sampling from $p_t$ is easy ($z \sim p_\text{data}$, then $x \sim p_t(\cdot|z)$) even though the sum itself is intractable.

**Conditional rate matrix.** Any rate matrix $Q_t^z$ whose CTMC follows the conditional path: $X_0 \sim p_\text{init}$, $X_t$ CTMC of $Q_t^z$ $\Rightarrow$ $X_t \sim p_t(\cdot|z)$. For the mixture path it exists in closed form and only permits jumps straight to the data token:

$$
\begin{aligned}
Q_t^z (y|x) &= (Q_t^z (v_i, j|x_j))_{v_i,j}, \\
Q^z_t(v, j \mid x) &= \frac{\dot\kappa_t}{1-\kappa_t}\,\big(\delta_{z^j}(v) - \delta_{x^j}(v)\big).
\end{aligned}
$$

## Kolmogorov Forward Equation
---
A CTMC with rate matrix $Q_t$ follows the probability path $X_t\sim p_t$, $t\in [0,1]$ if and only if the KFE holds,

$$
\frac{d}{dt} p_t(y) = \sum_{x \in S} Q_t(y|x)\, p_t(x)
\;=\; \underbrace{\sum_{x \ne y} Q_t(y|x)\,p_t(x)}_{\text{incoming flux}} \;-\; \underbrace{\sum_{x \ne y} Q_t(x|y)\,p_t(y)}_{\text{outgoing flux}},
$$

i.e., the change of probability is equal to the probability net inflow. 

The Kolmogorov forward equation is the [[#The Continuity Equation and Fokker–Planck Equation|Continuity Equation]] under discrete state space. A standard consequence:
- **Probability-preserving rates**: any $R_t$ with $\sum_x R_t(y|x)p_t(x) = 0$ (and valid rates) can be added without changing the marginals — the discrete divergence-free freedom behind re-masking correctors and Metropolis–Hastings-style components.

## Exact Likelihoods
---
The Continuity Equation implies the *instantaneous change of variables* ([Chen et al., 2018](https://arxiv.org/abs/1806.07366); guide §3.6):
$$
\frac{d}{dt}\log p_t(\psi_t(x)) = -\,\mathrm{div}(u_t)(\psi_t(x)),
$$
so $\log p_1$ is an integral of the divergence along the trajectory; Hutchinson's estimator $\mathrm{div}(u_t)(x) = \mathbb{E}_Z\,\mathrm{tr}[Z^\top \partial_x u_t(x) Z]$ makes it an **unbiased** estimate at one vector–Jacobian product per step — an advantage over stochastic processes, which only admit lower bounds ([[ELBO]]). Classical CNFs ([[#Flow Models|defined above]]) trained by maximum likelihood needed this ODE *inside* the training loop — exactly what FM eliminates.

## Relationship to Other Notes
---
- [[Flow Matching]] — uses this machinery end-to-end: regresses $u_t^\theta$ onto the marginal vector field via the conditional (CFM) loss; this note was split out of its Background section
- [[Score Matching]] — the score-side view; the same-marginals SDE family (Fokker–Planck + Langevin) is where its stochastic samplers live
- [[DDPM]] — diffusion time runs data → noise, the reverse of the convention here; its forward SDE constructs a Gaussian probability path
- [[Divergence Theorem]] — the continuity equation is its differential form; instantaneous change of variables and Hutchinson's estimator run on divergences
- [[CTMC]] — the discrete-state analog: rate matrices replace vector fields, the Kolmogorov equation replaces the continuity equation; the Discrete Probability Paths and KFE sections here were split out of that note
- [[Discrete Flow Matching]] — trains a CTMC model along the factorized mixture path via the marginal rate matrix (posterior-average of conditional rates)
- [[MCMC]] — Langevin dynamics as the divergence-free / static-path special case of the same-marginals SDE family
- [[ELBO]] — SDEs only admit likelihood lower bounds; ODE flows give exact likelihoods
