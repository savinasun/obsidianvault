### [[dl_moc]]  
# **CTMC (Continuous-Time Markov Chain)**

**Tags**:: #math #generative #dl_theo
**Links**:: [[Flow Matching]] · [[Discrete Flow Matching Policy]] · [[MCMC]] · [[Generative Modeling]] · [[LLMs]]


A **Continuous-Time Markov Chain** is a memoryless stochastic process on a **discrete state space** — the discrete-state analog of the ODEs and SDEs behind [[Flow Matching]], and the process class powering discrete diffusion / discrete FM (language models, protein co-design, token-action robot policies — [[Discrete Flow Matching Policy]]). This note follows §7.1 and Appendix C of the MIT 6.S184 lecture notes ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf); [course site](https://diffusion.csail.mit.edu/)), with the FM-guide view (jump processes, generators) noted at the end.

## Notations
---
+ **State space**: $S = \mathcal{V}^d$, sequences of $d$ tokens from a vocabulary $\mathcal{V}$ of size $V$, e.g. DNA bases, text tokens, action bins
+ **Transition probabilities**: $p_{t+h|t}(y|x) := p(X_{t+h} = y \mid X_t = x)$
+ **Transition-rate matrix**: a bounded function $$
\begin{aligned}
Q : S \times S \times [0,1] &\to \mathbb{R} \\
(x, y, t) &\mapsto Q_t(y|x)
\end{aligned}
$$continuous in $t$, defined entry-wise as the derivative of the transition probabilities at zero time-lag, $$
Q_t(y|x) := \frac{d}{dh}\Big|_{h=0}\, p_{t+h|t}(y \mid x),
$$and required to satisfy the two <span style="color:rgb(255, 255, 0)"><b>rate conditions</b></span>
$$
\text{(1)}\;\; Q_t(y|x) \ge 0 \quad (y \ne x), \qquad
	\text{(2)}\;\; Q_t(x|x) = -\sum_{y \ne x} Q_t(y|x) \le 0,$$so $Q_t(y|x)$ is the instantaneous rate of jumping $x \to y$, and each column sums to zero, $\sum_{y} Q_t(y|x) = 0$ (probability conservation).
+ **Dirac-delta function**: $\delta_z(x) = 1$ iff $x = z$.

## Markov Chains in Continuous Time
---
A stochastic process $(X_t)_{0 \le t \le 1}$ on $S$ is **Markov** if the future depends on the past only through the present:

$$
p(X_{t+h} \mid X_t, X_{t_1}, \dots, X_{t_k}) = p(X_{t+h} \mid X_t) \qquad (t_1 < \dots < t_k < t,\ h > 0).
$$

ODEs and SDEs are Markov too; on a *discrete* space such a process is a **Markov chain**, and in continuous time a **CTMC**. It is fully determined by its transition probabilities $p_{t+h|t}(y|x)$ plus the initial distribution $X_0 \sim p_\text{init}$.

#### Rate Matrices — the Discrete Velocity Field
On a discrete space there are no directions, only **jumps**, so *the analog of a velocity field is a **rate matrix*** $Q_t(y|x)$ — the instantaneous rate of switching $x \to y$. Derive from Taylor expansion,
$$
\frac{d}{dh}\Big|_{h=0}\, p_{t+h|t}(y \mid x) = Q_t(y|x)
\qquad\Longleftrightarrow\qquad
p_{t+h|t}(y|x) = \delta_x(y) + h\,Q_t(y|x) + o(h).
$$
**The dictionary between the continuous and discrete worlds:**

|                             | Continuous state ($\mathbb{R}^d$)                               | Discrete state ($S = \mathcal{V}^d$)       |
| --------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| **Dynamics object**         | velocity field $u_t(x)$ (+ diffusion $\sigma_t$)                | rate matrix $Q_t(y\|x)$                    |
| **Defining equation**       | ODE / SDE                                                       | transition-rate condition                  |
| **Solution**                | flow $\psi_t$ / process                                         | transition kernel $p_{t+h\|t}$             |
| **Marginal law evolves by** | Continuity / Fokker–Planck ([[Divergence Theorem]])             | Kolmogorov Forward Equation                |
| **Simulation**              | Euler / Euler–Maruyama                                          | categorical Euler step                     |
| **Well-posedness**          | Picard–Lindelöf (nonlinear; local)                              | linear ODE (global, no extra conditions)   |
| **Generative training**     | [[Flow Matching]] (regression) / [[Score Matching]] (denoising) | discrete FM (cross-entropy classification) |
#### Existence and Uniqueness
**Theorem (existence & uniqueness).** *For any rate matrix $Q_t$ (bounded, continuous in $t$), there exists a **unique** CTMC — a unique transition kernel — whose transition probabilities change at the rates prescribed by $Q_t$.*

The proof is pure **linear ODE theory**, and simpler than the flow case: for fixed $(x, t)$, the kernel $t' \mapsto p_{t'|t}(\cdot|x)$ satisfies the linear ODE $\frac{d}{dt'} p_{t'|t}(y|x) = \sum_z Q_{t'}(y|z)\, p_{t'|t}(z|x)$ with initial condition $\delta_x$ — uniqueness follows as in the Picard–Lindelöf theorem ([[Flow Matching]] Background), and *because the ODE is linear, solutions exist globally with no Lipschitz caveats or blow-up*. One then checks the solution is a valid kernel: rows stay normalized (columns of $Q_t$ sum to zero), probabilities stay non-negative (at $p = 0$ the derivative is $\ge 0$, so zero is a barrier), and Chapman–Kolmogorov holds (both sides solve the same linear ODE). For ML purposes: **any neural-network rate matrix defines a unique, well-behaved CTMC.**

**Worked example** — two states $S = \{a, b\}$, constant flip rate $\lambda > 0$: $Q = \begin{pmatrix} -\lambda & \lambda \\ \lambda & -\lambda \end{pmatrix}$ gives the closed-form kernel
$$
p_{t+h|t}(\text{stay}) = \tfrac{1}{2}\big(1 + e^{-2\lambda h}\big), \qquad
p_{t+h|t}(\text{switch}) = \tfrac{1}{2}\big(1 - e^{-2\lambda h}\big),
$$
which decays to the uniform distribution as $h \to \infty$ — the chain *forgets its start*, faster for larger $\lambda$ (the discrete analog of mixing in [[MCMC]] and of the OU process relaxing to its stationary Gaussian).

## Sampling Algorithm
---
Closed-form kernels are rare, but the first-order expansion of the kernel gives the **categorical Euler method**: for small step $h$,

$$
X_{t+h} \sim \tilde p_{t+h|t}(\cdot \mid X_t), \qquad \tilde p_{t+h|t}(y|x) = \delta_x(y) + h\,Q_t(y|x),
$$

a valid categorical distribution for small $h$ by the rate conditions (the FM guide's variant with $\exp[h\,Q_t(x|x)]$ staying-probability is valid for *every* $h$). Each step is just sampling from a discrete distribution.

![[CTMC_algorithm.png|500]]
 
## The Kolmogorov Forward Equation
---
The marginals $p_t$ of a CTMC evolve by the **KFE** — the discrete continuity equation:

$$
\frac{d}{dt} p_t(y) = \sum_{x \in S} Q_t(y|x)\, p_t(x)
\;=\; \underbrace{\sum_{x \ne y} Q_t(y|x)\,p_t(x)}_{\text{incoming flux}} \;-\; \underbrace{\sum_{x \ne y} Q_t(x|y)\,p_t(y)}_{\text{outgoing flux}},
$$

and conversely (as with Mass Conservation / Fokker–Planck in [[Flow Matching]]): $X_t \sim p_t$ for all $t$ **iff** the KFE holds — necessity by differentiating, sufficiency by linear-ODE uniqueness. A standard consequence:
- **Probability-preserving rates**: any $R_t$ with $\sum_x R_t(y|x)p_t(x) = 0$ (and valid rates) can be added without changing the marginals — the discrete divergence-free freedom behind re-masking correctors and Metropolis–Hastings-style components.

## Probability Paths and the Marginalization Trick
---
The generative construction mirrors continuous [[Flow Matching]], with PMFs replacing densities.

**Conditional probability path.** A family of PMFs $p_t(\cdot|z)$ on $S$, indexed by a data point $z \in S$ and $t \in [0,1]$, interpolating noise into that single data point:

$$
p_0(\cdot|z) = p_\text{init}, \qquad p_1(\cdot|z) = \delta_z \qquad \forall z \in S .
$$

The standard choice is the **factorized mixture path** — each token independently revealed with probability $\kappa_t$ (a monotone schedule with $\kappa_0 = 0$, $\kappa_1 = 1$):

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
Q^z_t(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\,\big(\delta_{z^j}(v) - \delta_{x^j}(v)\big).
$$

**Theorem (discrete marginalization trick).** *Let $Q_t^z$ generate the conditional path $p_t(\cdot|z)$ for every $z \in S$. Then the **marginal rate matrix***

$$
Q_t(y|x) := \sum_{z \in S} Q_t^z(y|x)\; p_{1|t}(z|x),
\qquad
p_{1|t}(z|x) := \frac{p_t(x|z)\, p_\text{data}(z)}{p_t(x)},
$$

*is a valid rate matrix, and the CTMC it defines follows the marginal path: $X_0 \sim p_\text{init} \Rightarrow X_t \sim p_t$ for all $t \in [0,1]$ — in particular $X_1 \sim p_\text{data}$.*

*Proof sketch.* The rate conditions survive the convex combination (posterior weights are non-negative and sum to one over $z$). For the path property, differentiate $p_t(x) = \sum_z p_t(x|z)\,p_\text{data}(z)$, apply the conditional KFE to each term, multiply-and-divide by $p_t(x)$, and recognize the marginal KFE for $Q_t$; sufficiency then follows from linear-ODE uniqueness. $\square$

This theorem is the training target of discrete FM: the posterior average $\sum_z (\cdot)\, p_{1|t}(z|x)$ is a conditional expectation, so it can be learned from single data samples — and for the mixture path it collapses to per-position denoising posteriors $p_{1|t}(z^j = v \mid x)$, i.e. a classifier per token.

## CTMC as a Generative Model
---
A **CTMC model** (also called "discrete diffusion model" sometimes) is $p_\text{init}=\text{Uniform}(S)$ plus a network $Q_t^\theta$ that outputs, for input $x$, one column $\{Q_t^\theta(y|x)\}_y$ of the rate matrix. The catch: $|S| = V^d$, so a full column can't even be stored. The fix is **factorization**, a sparsity constraint.

<span style="color:rgb(255, 255, 0)"><b>Factorization condition.</b></span> A CTMC model is **factorized** if, for all $x = (x^1, \dots, x^d),\ y = (y^1, \dots, y^d) \in S$,

$$
Q_t^\theta(y \mid x) = 0 \qquad \text{whenever } y^i \ne x^i \text{ for more than one position } i,
$$

i.e. rates are supported on the <span style="color:rgb(255, 255, 0)"><b>neighbors</b></span> $N(x) := \{\,y \in S : y^i \ne x^i \text{ for at most one } i\,\}$ — the chain changes at most one token at a time. The surviving rates are parameterized per position,

$$
Q_t^\theta(v, i \mid x) := Q_t^\theta\big((x^1, \dots, x^{i-1}, v, x^{i+1}, \dots, x^d) \,\big|\, x\big), \qquad v \in \mathcal{V},\ i \in \{1, \dots, d\},
$$

each obeying the rate conditions per position: $Q_t^\theta(v, i \mid x) \ge 0$ for $v \ne x^i$ and $Q_t^\theta(x^i, i \mid x) = -\sum_{v \ne x^i} Q_t^\theta(v, i \mid x)$. The network output shrinks from $V^d$ to a tractable $d \times V$ matrix (a [[Transformer]] with a $V$-way head), and per-token parallel Euler steps agree with the full CTMC step to first order in $h$.

Training uses the discrete FM recipe (see [[Flow Matching]], Discrete FM section): choose a mixture/masked path, derive the conditional rate matrix in closed form, and learn the per-position denoising posteriors $p_{1|t}^{\theta}(z^j|x)$ with plain **cross-entropy** — generative modeling reduced to per-token classification. Applications: masked diffusion language models ([[LLMs]] beyond autoregression), multimodal protein co-design, and token-action robot policies ([[Discrete Flow Matching Policy]]).

**FM-guide view**: a CTMC is a **jump process** on a discrete space; its generator is $\mathcal{L}_t f(x) = \sum_y f(y) Q_t(y|x)$, sitting in the same Generator Matching table as flows ($\nabla f^\top u_t$) and diffusions ($\tfrac{1}{2}\sigma_t^2 \cdot \nabla^2 f$) — one framework, four process classes.

## CTMC vs. MCMC
---
Both are Markov chains used for sampling, with opposite designs: [[MCMC]] runs a **time-homogeneous, discrete-time** chain whose *stationary* distribution is the target (detailed balance; run until mixed, convergence only asymptotic); a CTMC generative model is **time-inhomogeneous by construction** and follows a *prescribed* probability path $p_t$ exactly over $[0,1]$ — no waiting for convergence, the endpoint *is* the target. Metropolis–Hastings kernels reappear inside CTMC models as probability-preserving (divergence-free) rate components.

## Relationship to Other Notes
---
- [[Flow Matching]] — the continuous-state parent framework; its Discrete FM section is the training recipe for CTMC models, and its Background (Picard–Lindelöf, continuity equation) is the continuous mirror of this note
- [[Discrete Flow Matching Policy]] — CTMC action decoding for robots; masked mixture paths and re-masking correctors in action
- [[MCMC]] — the stationary-sampler cousin; mixing intuition, detailed balance, Metropolis–Hastings as divergence-free component
- [[Divergence Theorem]] — the KFE is the discrete continuity equation; flux in/out reading
- [[Generative Modeling]] — where CTMC-based discrete diffusion sits in the family taxonomy
- [[LLMs]] / [[Transformer]] — masked diffusion language models are CTMC models; the factorized rate network is a standard transformer
