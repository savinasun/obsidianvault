### [[dl_moc]]  
# **CTMC (Continuous-Time Markov Chain)**

**Tags**:: #math #generative #dl_theo
**Links**:: [[Flow Matching]] · [[Discrete Flow Matching Policy]] · [[MCMC]] · [[Generative Modeling]] · [[LLMs]]


A **Continuous-Time Markov Chain** is a memoryless stochastic process on a **discrete state space** — the discrete-state analog of the ODEs and SDEs behind [[Flow Matching]], and the process class powering discrete diffusion / discrete FM (language models, protein co-design, token-action robot policies — [[Discrete Flow Matching Policy]]). This note follows §7.1 and Appendix C of the MIT 6.S184 lecture notes ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf); [course site](https://diffusion.csail.mit.edu/)), with the FM-guide view (jump processes, generators) noted at the end.

## Notations
---
**State space**: $S = \mathcal{V}^d$, sequences of $d$ tokens from a vocabulary $\mathcal{V}$ of size $V$, e.g. DNA bases, text tokens, action bins
**Transition probabilities**: $p_{t+h|t}(y|x) := p(X_{t+h} = y \mid X_t = x)$
**Transition-rate matrix**: a bounded function 
$$
\begin{aligned}
Q : S \times S \times [0,1] &\to \mathbb{R} \\
(x, y, t) &\mapsto Q_t(y|x)
\end{aligned}
$$
continuous in $t$, defined entry-wise as the derivative of the transition probabilities at zero time-lag,
$$
Q_t(y|x) := \frac{d}{dh}\Big|_{h=0}\, p_{t+h|t}(y \mid x),
$$
and required to satisfy the two **rate conditions**
$$
\text{(1)}\;\; Q_t(y|x) \ge 0 \quad (y \ne x), \qquad
\text{(2)}\;\; Q_t(x|x) = -\sum_{y \ne x} Q_t(y|x) \le 0,
$$
so $Q_t(y|x)$ is the instantaneous rate of jumping $x \to y$, and each column sums to zero, $\sum_{y} Q_t(y|x) = 0$ (probability conservation).
**Dirac-delta function**: $\delta_z(x) = 1$ iff $x = z$.

## Markov Chains in Continuous Time
---
A stochastic process $(X_t)_{0 \le t \le 1}$ on $S$ is **Markov** if the future depends on the past only through the present:

$$
p(X_{t+h} \mid X_t, X_{t_1}, \dots, X_{t_k}) = p(X_{t+h} \mid X_t) \qquad (t_1 < \dots < t_k < t,\ h > 0).
$$

ODEs and SDEs are Markov too; on a *discrete* space such a process is a **Markov chain**, and in continuous time a **CTMC**. It is fully determined by its transition probabilities $p_{t+h|t}(y|x)$ plus the initial distribution $X_0 \sim p_\text{init}$.

## Rate Matrices — the Discrete Velocity Field
---
On a discrete space there are no directions, only **jumps**, so the analog of a velocity field is a **rate matrix** $Q_t(y|x)$ — the instantaneous rate of switching $x \to y$. Derive from Taylor expansion,
$$
\frac{d}{dh}\Big|_{h=0}\, p_{t+h|t}(y \mid x) = Q_t(y|x)
\qquad\Longleftrightarrow\qquad
p_{t+h|t}(y|x) = \delta_x(y) + h\,Q_t(y|x) + o(h).
$$

The dictionary between the continuous and discrete worlds:

|                             | Continuous state ($\mathbb{R}^d$)                               | Discrete state ($S = \mathcal{V}^d$)       |
| --------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| **Dynamics object**         | velocity field $u_t(x)$ (+ diffusion $\sigma_t$)                | rate matrix $Q_t(y\|x)$                    |
| **Defining equation**       | ODE / SDE                                                       | transition-rate condition                  |
| **Solution**                | flow $\psi_t$ / process                                         | transition kernel $p_{t+h\|t}$             |
| **Marginal law evolves by** | Continuity / Fokker–Planck ([[Divergence Theorem]])             | Kolmogorov Forward Equation                |
| **Simulation**              | Euler / Euler–Maruyama                                          | categorical Euler step                     |
| **Well-posedness**          | Picard–Lindelöf (nonlinear; local)                              | linear ODE (global, no extra conditions)   |
| **Generative training**     | [[Flow Matching]] (regression) / [[Score Matching]] (denoising) | discrete FM (cross-entropy classification) |
![[CTMC_algorithm.png|500]]

## Existence and Uniqueness
---
**Theorem (6.S184 Thm 33).** *For any rate matrix $Q_t$ (bounded, continuous in $t$), there exists a **unique** CTMC — a unique transition kernel — satisfying Eq. 87.*

The proof (App. C) is pure **linear ODE theory**, and simpler than the flow case: for fixed $(x, t)$, the kernel $t' \mapsto p_{t'|t}(\cdot|x)$ satisfies the linear ODE $\frac{d}{dt'} p_{t'|t}(y|x) = \sum_z Q_{t'}(y|z)\, p_{t'|t}(z|x)$ with initial condition $\delta_x$ — uniqueness follows as in the Picard–Lindelöf theorem ([[Flow Matching]] Background), and *because the ODE is linear, solutions exist globally with no Lipschitz caveats or blow-up*. One then checks the solution is a valid kernel: rows stay normalized (columns of $Q_t$ sum to zero), probabilities stay non-negative (at $p = 0$ the derivative is $\ge 0$, so zero is a barrier), and Chapman–Kolmogorov holds (both sides solve the same linear ODE). For ML purposes: **any neural-network rate matrix defines a unique, well-behaved CTMC.**

**Worked example (6.S184 Ex. 34)** — two states $S = \{a, b\}$, constant flip rate $\lambda > 0$: $Q = \begin{pmatrix} -\lambda & \lambda \\ \lambda & -\lambda \end{pmatrix}$ gives the closed-form kernel

$$
p_{t+h|t}(\text{stay}) = \tfrac{1}{2}\big(1 + e^{-2\lambda h}\big), \qquad
p_{t+h|t}(\text{switch}) = \tfrac{1}{2}\big(1 - e^{-2\lambda h}\big),
$$

which decays to the uniform distribution as $h \to \infty$ — the chain *forgets its start*, faster for larger $\lambda$ (the discrete analog of mixing in [[MCMC]] and of the OU process relaxing to its stationary Gaussian).

## Simulation
---
Closed-form kernels are rare, but Eq. 87 gives the **categorical Euler method**: for small step $h$,

$$
X_{t+h} \sim \tilde p_{t+h|t}(\cdot \mid X_t), \qquad \tilde p_{t+h|t}(y|x) = \delta_x(y) + h\,Q_t(y|x),
$$

a valid categorical distribution for small $h$ by the rate conditions (the FM guide's variant with $\exp[h\,Q_t(x|x)]$ staying-probability is valid for *every* $h$). Each step is just sampling from a discrete distribution.

## The Kolmogorov Forward Equation
---
The marginals $p_t$ of a CTMC evolve by the **KFE** (6.S184 Prop. 2) — the discrete continuity equation:

$$
\frac{d}{dt} p_t(y) = \sum_{x \in S} Q_t(y|x)\, p_t(x)
\;=\; \underbrace{\sum_{x \ne y} Q_t(y|x)\,p_t(x)}_{\text{incoming flux}} \;-\; \underbrace{\sum_{x \ne y} Q_t(x|y)\,p_t(y)}_{\text{outgoing flux}},
$$

and conversely (as with Mass Conservation / Fokker–Planck in [[Flow Matching]]): $X_t \sim p_t$ for all $t$ **iff** the KFE holds — necessity by differentiating, sufficiency by linear-ODE uniqueness. Two standard consequences:
- **Marginalization trick** (6.S184 Thm 36): if conditional rate matrices $Q^z_t$ generate conditional paths $p_t(\cdot|z)$, their posterior average $Q_t(y|x) = \sum_z Q_t^z(y|x)\,p_{1|t}(z|x)$ generates the marginal path — verified by plugging into the KFE, exactly as in continuous FM.
- **Probability-preserving rates**: any $R_t$ with $\sum_x R_t(y|x)p_t(x) = 0$ (and valid rates) can be added without changing the marginals — the discrete divergence-free freedom behind re-masking correctors and Metropolis–Hastings-style components.

## CTMC as a Generative Model
---
A **CTMC model** (= "discrete diffusion model") is $p_\text{init}$ plus a network $Q_t^\theta$ that outputs, for input $x$, one column $\{Q_t^\theta(y|x)\}_y$ of the rate matrix. The catch: $|S| = V^d$, so a full column can't even be stored. The fix is **factorization** (a sparsity constraint): allow non-zero rates only between **neighbors** — states differing in at most one token — parameterized per position as $Q_t^\theta(v, j \mid x)$ with output shape $d \times V$ (a [[Transformer]] with a $V$-way head); per-token parallel Euler steps agree with the full CTMC step to first order in $h$ (6.S184 Alg. 7).

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
