### [[dl_moc]]  
# **CTMC (Continuous-Time Markov Chain)**

**Tags**:: #math #generative #dl_theo
**Links**:: [[Flow Matching]] · [[Discrete Flow Matching]] · [[ODEs, SDEs, and Probability Paths]] · [[MCMC]] · [[Generative Modeling]] · [[LLMs]]


A **Continuous-Time Markov Chain** is a memoryless stochastic process on a **discrete state space** — the discrete-state analog of the ODEs and SDEs behind [[Flow Matching]], and the process class powering discrete diffusion / [[Discrete Flow Matching]] (language models, protein co-design, token-action robot policies). This note follows §7.1 and Appendix C of the MIT 6.S184 lecture notes ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf); [course site](https://diffusion.csail.mit.edu/)), with the FM-guide view (jump processes, generators) noted at the end.

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

The proof is pure **linear ODE theory**, and simpler than the flow case: for fixed $(x, t)$, the kernel $t' \mapsto p_{t'|t}(\cdot|x)$ satisfies the linear ODE $\frac{d}{dt'} p_{t'|t}(y|x) = \sum_z Q_{t'}(y|z)\, p_{t'|t}(z|x)$ with initial condition $\delta_x$ — uniqueness follows as in the Picard–Lindelöf theorem ([[ODEs, SDEs, and Probability Paths]]), and *because the ODE is linear, solutions exist globally with no Lipschitz caveats or blow-up*. One then checks the solution is a valid kernel: rows stay normalized (columns of $Q_t$ sum to zero), probabilities stay non-negative (at $p = 0$ the derivative is $\ge 0$, so zero is a barrier), and Chapman–Kolmogorov holds (both sides solve the same linear ODE). For ML purposes: **any neural-network rate matrix defines a unique, well-behaved CTMC.**

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

![[CTMC_sampling.png|500]]

## Probability Paths and the Kolmogorov Forward Equation
---
The generative construction — conditional/marginal probability paths over $S$, the factorized mixture path, conditional rate matrices, and the **Kolmogorov Forward Equation** (the discrete continuity equation) — lives in [[ODEs, SDEs, and Probability Paths]], alongside its continuous mirror. What the rest of this note relies on:

- **Factorized mixture path** (conditional path $p_\text{init} \to \delta_z$, schedule $\kappa_0 = 0$, $\kappa_1 = 1$): $p_t(x|z) = \prod_j \big[\kappa_t\, \delta_{z^j}(x^j) + (1-\kappa_t)\, p_\text{init}(x^j)\big]$, with the **masked** special case $p_\text{init} = \delta_{[\text{mask}]}$; marginal path $p_t(x) = \sum_z p_t(x|z)\, p_\text{data}(z)$, interpolating $p_\text{init} \to p_\text{data}$.
- **Conditional rate matrix** generating the mixture path (only jumps straight to the data token): $Q^z_t(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\,\big(\delta_{z^j}(v) - \delta_{x^j}(v)\big)$.
- **KFE**: a CTMC with rates $Q_t$ follows $X_t \sim p_t$ **iff** $\frac{d}{dt} p_t(y) = \sum_x Q_t(y|x)\, p_t(x)$ — change of probability = net inflow; **probability-preserving rates** $R_t$ with $\sum_x R_t(y|x)\,p_t(x) = 0$ can be added freely (re-masking correctors, Metropolis–Hastings-style components).

## Marginal Rate Matrix
---
<span style="color:rgb(255, 255, 0)"><b>Theorem (discrete marginalization trick).</b></span> *Let $Q_t^z$ generate the conditional path $p_t(\cdot|z)$ for every $z \in S$. Then the <span style="color:rgb(255, 255, 0)"><b>marginal rate matrix</b></span>*

$$
Q_t(y|x) := \sum_{z \in S} Q_t^z(y|x)\; p_{1|t}(z|x),
\qquad
p_{1|t}(z|x) := \frac{p_t(x|z)\, p_\text{data}(z)}{p_t(x)},
$$

*is a valid rate matrix, and the CTMC it defines follows the marginal path: $X_0 \sim p_\text{init} \Rightarrow X_t \sim p_t$ for all $t \in [0,1]$ — in particular $X_1 \sim p_\text{data}$.*

Proof. 
$$
\begin{aligned}
\frac{d}{dt}p_t(x) &= \frac{d}{dt} \sum\limits_{z\in S} p_t(x|z) p_{\text{data}}(z) \\
&= \sum_{z\in S} \frac{d}{dt} p_t(x|z) p_{\text{data}}(z) \quad \textit{(apply convexity)}\\
&= \sum_{z\in S} \left[\sum\limits_{y\in S} Q^z_t (x|y) p_t(y) \right] p_{\text{data}}(z)   \quad (\textit{apply Kolmogorov forward equation})\\
&= \sum\limits_{y\in S} \left[\sum\limits_{z\in S} Q^z_t(x|y) \cdot \frac{p_t(y)p_{\text{data}}(z)}{p_t(y)}\right] \cdot p_t(y) \\
&= \sum\limits_{y\in S} Q_t(x|y) \cdot p_t(y)\quad \textit{(apply marginal prob. path definition)}
\end{aligned}
$$
therefore $X_t \sim p_t$.$\square$

This theorem is the training target of discrete FM: the posterior average $\sum_z (\cdot)\, p_{1|t}(z|x)$ is a conditional expectation, so it can be learned from single data samples. For the factorized mixture path it collapses into closed form:

<span style="color:rgb(255, 255, 0)"><b>Theorem (marginal rate matrix of the factorized mixture path).</b></span> *For the factorized mixture path with conditional rates $Q^z_t(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\big(\delta_{z^j}(v) - \delta_{x^j}(v)\big)$, the marginal rate matrix is itself factorized, with per-position entries*

$$
Q_t(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\,\Big( p_{1|t}(z^j = v \mid x) \;-\; \delta_{x^j}(v) \Big),
$$

*where $p_{1|t}(z^j = v \mid x) := \sum_{z \,:\, z^j = v} p_{1|t}(z|x)$ is the posterior probability that the clean token at position $j$ equals $v$, given the noisy sequence $x$.*

Proof. (i) *Factorization*: whenever $y$ and $x$ differ in more than one position, $Q_t^z(y|x) = 0$ for every $z$, so the posterior average vanishes there too — the marginal rate matrix is supported on neighbors. (ii) *Per position*:

$$
\begin{aligned}
Q_t(v, j \mid x)
&= \sum_{z \in S} Q_t^z(v, j \mid x)\; p_{1|t}(z|x) \\
&= \frac{\dot\kappa_t}{1-\kappa_t} \Bigg( \sum_{z \in S} \delta_{z^j}(v)\, p_{1|t}(z|x) \;-\; \delta_{x^j}(v)\, \underbrace{\sum_{z \in S} p_{1|t}(z|x)}_{=\,1} \Bigg) \quad \textit{(plug in conditional rates)}\\
&= \frac{\dot\kappa_t}{1-\kappa_t}\,\Big( p_{1|t}(z^j = v \mid x) - \delta_{x^j}(v) \Big). \quad \textit{(marginalize the posterior over positions $\ne j$)} \qquad \square
\end{aligned}
$$

**Why it matters.** The marginal rate matrix is a pure **reparameterization of per-position denoising posteriors** — learning the generative model reduces to training a classifier per token position, a network $x \mapsto \big(p^\theta_{1|t}(z^j = v \mid x)\big)_{j \le d,\, v \in \mathcal{V}}$ with output shape $d \times V$ (softmax per position), trained by cross-entropy $\mathbb{E}\big[-\sum_j \log p^\theta_{1|t}(z^j \mid x)\big]$. As continuous FM reduces to regression and score matching to denoising, **discrete FM reduces to classification**.

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

Training uses the discrete FM recipe (see [[Discrete Flow Matching]]): choose a mixture/masked path, derive the conditional rate matrix in closed form, and learn the per-position denoising posteriors $p_{1|t}^{\theta}(z^j|x)$ with plain **cross-entropy** — generative modeling reduced to per-token classification. Applications: masked diffusion language models ([[LLMs]] beyond autoregression), multimodal protein co-design, and token-action robot policies.

![[CTMC_algo.png|500]]

**FM-guide view**: a CTMC is a **jump process** on a discrete space; its generator is $\mathcal{L}_t f(x) = \sum_y f(y) Q_t(y|x)$, sitting in the same Generator Matching table as flows ($\nabla f^\top u_t$) and diffusions ($\tfrac{1}{2}\sigma_t^2 \cdot \nabla^2 f$) — one framework, four process classes.

## CTMC vs. MCMC
---
Both are Markov chains used for sampling, with opposite designs: [[MCMC]] runs a **time-homogeneous, discrete-time** chain whose *stationary* distribution is the target (detailed balance; run until mixed, convergence only asymptotic); a CTMC generative model is **time-inhomogeneous by construction** and follows a *prescribed* probability path $p_t$ exactly over $[0,1]$ — no waiting for convergence, the endpoint *is* the target. Metropolis–Hastings kernels reappear inside CTMC models as probability-preserving (divergence-free) rate components.

## Relationship to Other Notes
---
- [[Flow Matching]] — the continuous-state parent framework; its Discrete FM section is the training recipe for CTMC models
- [[ODEs, SDEs, and Probability Paths]] — the continuous mirror of this note (Picard–Lindelöf, continuity equation); now also holds this note's former Probability Paths and Kolmogorov Forward Equation sections
- [[Discrete Flow Matching]] — the generative training framework built on CTMCs: masked mixture paths, cross-entropy loss, re-masking correctors, and applications (LLMs, robot policies)
- [[MCMC]] — the stationary-sampler cousin; mixing intuition, detailed balance, Metropolis–Hastings as divergence-free component
- [[Divergence Theorem]] — the KFE is the discrete continuity equation; flux in/out reading
- [[Generative Modeling]] — where CTMC-based discrete diffusion sits in the family taxonomy
- [[LLMs]] / [[Transformer]] — masked diffusion language models are CTMC models; the factorized rate network is a standard transformer
