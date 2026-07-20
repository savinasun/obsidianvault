### [[dl_moc]]  
# **Divergence Theorem**

**Tags**:: #math #dl_theo
**Links**:: [[Flow Matching]] · [[Score-Based Generative Modeling Methods]] · [[DDPM]] · [[MCMC]] · [[Linear Algebra]] · [[KL Divergence]]


The **Divergence Theorem** (also **Gauss's theorem** or the **Gauss–Ostrogradsky theorem**) is the fundamental bridge between the *local* behavior of a vector field (its divergence at each point) and its *global* behavior (net flux through a closed surface):

$$
\int_V \mathrm{div}(F)(x)\, dV \;=\; \oint_{\partial V} \langle F(y), n(y)\rangle\, dS_y ,
$$

for a compact region $V \subset \mathbb{R}^d$ with piecewise-smooth boundary $\partial V$, outward unit normal $n$, and a $C^1$ vector field $F$. In words: **accumulating all the infinitesimal sources and sinks inside a region equals the net outflow through its boundary**.

In deep learning this theorem is the engine behind the continuity equation of [[Flow Matching]], exact likelihoods of continuous normalizing flows, Hyvärinen's tractable score matching objective, Stein's identity, and the Fokker–Planck equation — anywhere a **velocity field** transports probability mass.

> **Naming caution**: the divergence of a *vector field* ($\mathrm{div}\,F$, this note) and a *statistical divergence* between distributions ([[KL Divergence]], Bregman, Fisher) are unrelated concepts that share a word.

## Divergence of a Vector Field
---
For $F : \mathbb{R}^d \to \mathbb{R}^d$,

$$
\mathrm{div}(F)(x) \;=\; \sum_{i=1}^d \frac{\partial F_i}{\partial x_i}(x) \;=\; \mathrm{tr}\big[\partial_x F(x)\big],
$$

the **trace of the Jacobian** — only the $d$ diagonal entries matter, not the full $d \times d$ matrix. Equivalent coordinate-free definition as *local flux density*:

$$
\mathrm{div}(F)(x_0) = \lim_{r \to 0} \frac{1}{|B_r(x_0)|} \oint_{\partial B_r(x_0)} \langle F, n \rangle\, dS ,
$$

the net outflow per unit volume from an infinitesimal ball. Interpretation of sign, thinking of $F$ as the velocity of a fluid/gas:
- $\mathrm{div}(F)(x) > 0$: $x$ is a **source** — flow expands there
- $\mathrm{div}(F)(x) < 0$: $x$ is a **sink** — flow compresses there
- $\mathrm{div}(F) \equiv 0$: **divergence-free** (solenoidal, incompressible) — flow rearranges mass without changing local volume

Examples: the radial field $F(x) = x$ has $\mathrm{div} = d$ (uniform expansion); a rigid rotation in 2D, $F(x,y) = (-y, x)$, has $\mathrm{div} = 0$; the composition $\mathrm{div}(\nabla \phi) = \Delta \phi$ defines the **Laplacian**.

## The Theorem: Intuition and the Vector-Calculus Family
---
**Why it's true (telescoping picture)**: tile $V$ into tiny cells. The flux out of one cell through a shared internal face is exactly the flux *into* its neighbor — all interior contributions cancel pairwise, leaving only the outer boundary. Each cell's net outflow is $\mathrm{div}(F)\,dV$ by the limit definition; summing gives the theorem. In $d = 1$ it *is* the fundamental theorem of calculus: $\int_a^b f'(x)\,dx = f(b) - f(a)$, with "boundary flux" $f(b)\cdot(+1) + f(a)\cdot(-1)$ at the two endpoint "normals."

The theorem sits in a hierarchy, all special cases of the **generalized Stokes theorem** $\int_\Omega d\omega = \int_{\partial \Omega} \omega$ on manifolds ("the integral of a derivative over a region = the integral of the object over the boundary"):

| Theorem | Dimension | Statement |
|---|---|---|
| Fundamental theorem of calculus | 1D | $\int_a^b f'\,dx = f(b) - f(a)$ |
| Gradient theorem | curve in $\mathbb{R}^d$ | $\int_\gamma \langle \nabla\phi, d\ell\rangle = \phi(\gamma_1) - \phi(\gamma_0)$ |
| Green's theorem | 2D region | circulation form of Stokes in the plane |
| Kelvin–Stokes theorem | surface in $\mathbb{R}^3$ | $\int_S \langle \nabla \times F, n\rangle\, dS = \oint_{\partial S} \langle F, d\ell \rangle$ |
| **Divergence theorem** | $d$-dim volume | $\int_V \mathrm{div} F\, dV = \oint_{\partial V} \langle F, n \rangle dS$ |

**Physics origins**: Gauss's law of electrostatics ($\oint \langle E, n\rangle dS = Q_\text{enc}/\varepsilon_0 \Leftrightarrow \mathrm{div}\,E = \rho/\varepsilon_0$), conservation of mass in fluids, and the derivation of the heat equation are all this theorem.

## Key Corollaries: Integration by Parts and Stein's Identity
---
Applying the theorem to the product $\psi F$ (scalar $\psi$, vector field $F$) and using $\mathrm{div}(\psi F) = \psi\, \mathrm{div}F + \langle \nabla \psi, F\rangle$ gives **multivariate integration by parts**:

$$
\int_V \psi\, \mathrm{div}(F)\, dV = \oint_{\partial V} \psi \langle F, n\rangle\, dS - \int_V \langle \nabla\psi, F\rangle\, dV .
$$

Taking $V = \mathbb{R}^d$, $\psi = p$ a density, and assuming $p F \to 0$ at infinity kills the boundary term and yields **Stein's identity**:

$$
\mathbb{E}_{x \sim p}\big[\langle \nabla \log p(x), F(x)\rangle + \mathrm{div}(F)(x)\big] = 0 ,
$$

(using $\nabla p = p\,\nabla\log p$). This one line lets you **trade an unknown data score $\nabla \log p$ for a computable divergence of your model** — the foundation of score matching (below), Stein discrepancies, and control variates.

## Conservation Laws: the Continuity Equation
---
The divergence theorem converts "mass is conserved" into a PDE. Let $p_t$ be a density transported by a velocity field $u_t$, and define the **probability flux** $j_t = p_t u_t$ (mass crossing a unit area per unit time). Conservation says: the mass inside any region $D$ changes only by flux through its boundary,

$$
\frac{d}{dt}\int_D p_t(x)\,dx \;=\; -\oint_{\partial D} \langle p_t u_t, n\rangle\, dS \;\overset{\text{Div.\ Thm.}}{=}\; -\int_D \mathrm{div}(p_t u_t)(x)\, dx .
$$

Since $D$ is arbitrary, the integrands match pointwise — the **Continuity Equation**:

$$
\frac{\partial}{\partial t} p_t(x) + \mathrm{div}\big(p_t u_t\big)(x) = 0 .
$$

This is exactly the equation characterizing "$u_t$ generates $p_t$" in [[Flow Matching]] (Mass Conservation theorem), and the FM guide's own interpretation of it is the integral form above: *local change in probability = minus net outgoing probability flux*. The Marginalization Trick — the theorem licensing conditional-flow training — is proven by verifying the continuity equation for the marginal pair $(p_t, u_t)$.

Special cases:
- $\mathrm{div}(u_t) = 0$ (incompressible): the flow preserves volume; density is constant along trajectories.
- Adding any field $v_t$ with $\mathrm{div}(p_t v_t) = 0$ leaves the path $p_t$ unchanged — the **divergence-free design freedom** used for stochastic/corrector sampling in FM and Generator Matching.

## Velocity Fields and Likelihoods: Instantaneous Change of Variables
---
Along a flow trajectory $\psi_t(x)$ driven by $u_t$, the continuity equation implies

$$
\frac{d}{dt} \log p_t(\psi_t(x)) = -\,\mathrm{div}(u_t)(\psi_t(x)) ,
$$

the **instantaneous change of variables** of Neural ODEs: log-density decreases exactly at the rate the flow expands volume. (Equivalently, Liouville's formula: $\frac{d}{dt}\log \det \partial_x\psi_t(x) = \mathrm{div}(u_t)(\psi_t(x))$ — divergence is the instantaneous log-volume distortion, the continuous analog of the $\log|\det J|$ term in discrete normalizing flows.) Integrating along the trajectory gives **exact flow likelihoods**:

$$
\log p_1(\psi_1(x)) = \log p_0(x) - \int_0^1 \mathrm{div}(u_t)(\psi_t(x))\, dt .
$$

**Hutchinson's trace estimator** makes the divergence affordable in high dimension: since $\mathrm{div}(u) = \mathrm{tr}[\partial_x u]$ and $\mathrm{tr}(A) = \mathbb{E}_Z[Z^\top A Z]$ for any $Z$ with zero mean and identity covariance,

$$
\mathrm{div}(u_t)(x) = \mathbb{E}_Z\, \mathrm{tr}\big[Z^\top \partial_x u_t(x)\, Z\big],
$$

computable with a single vector–Jacobian product per sample instead of $d$ backward passes — this is how FFJORD and [[Flow Matching]] models report unbiased log-likelihoods.

## Score Matching via Integration by Parts
---
The Fisher divergence objective of [[Score-Based Generative Modeling Methods]] contains the unknown data score:

$$
\tfrac{1}{2}\,\mathbb{E}_{p}\big\|s_\theta(x) - \nabla_x \log p(x)\big\|^2 .
$$

Expanding the square, the cross term is handled by Stein's identity / integration by parts (Hyvärinen, 2005): $-\mathbb{E}_p\langle s_\theta, \nabla \log p\rangle = -\int \langle s_\theta, \nabla p\rangle\,dx = \mathbb{E}_p\big[\mathrm{div}(s_\theta)\big]$, giving the tractable **implicit score matching** objective

$$
\mathcal{J}(\theta) = \mathbb{E}_{p}\Big[\tfrac{1}{2}\|s_\theta(x)\|^2 + \mathrm{div}\big(s_\theta\big)(x)\Big] + \text{const},
$$

which needs **no ground-truth score** — only your model and data samples. The $\mathrm{div}(s_\theta) = \mathrm{tr}[\partial_x s_\theta]$ term is the expensive part (Hessian trace of $\log p_\theta$), motivating sliced score matching (Hutchinson-style projections) and denoising score matching (which avoids the divergence entirely and became the diffusion-model loss — [[DDPM]]).

## Fokker–Planck and Divergence-Free Langevin
---
For an SDE $dX_t = a_t(X_t)\,dt + g_t\,dW_t$, the density evolves by the **Fokker–Planck equation** — a continuity equation with a diffusion term (both terms are divergences):

$$
\frac{\partial}{\partial t} p_t = -\,\mathrm{div}(p_t\, a_t) + \frac{g_t^2}{2} \Delta p_t .
$$

Two staple computations, both one-line applications of $\mathrm{div}$ algebra:
- **Probability flow ODE**: rewriting $\frac{g_t^2}{2}\Delta p_t = \mathrm{div}\big(p_t \cdot \frac{g_t^2}{2}\nabla \log p_t\big)$ folds diffusion into the drift, turning Fokker–Planck into a pure continuity equation with velocity $a_t - \frac{g_t^2}{2}\nabla\log p_t$ — the deterministic sampler shared by diffusion and FM.
- **Langevin is divergence-free**: the Langevin generator's action on $p_t$ is $-\mathrm{div}\big(p_t\,\tfrac{\beta_t^2}{2}\nabla\log p_t\big) + \tfrac{\beta_t^2}{2}\Delta p_t = -\tfrac{\beta_t^2}{2}\Delta p_t + \tfrac{\beta_t^2}{2}\Delta p_t = 0$, so Langevin dynamics ([[MCMC]]) can be added to *any* generative process at *any* strength without changing its marginals — the origin of stochastic sampling and predictor–corrector schemes in [[Flow Matching]]/diffusion.

## On Manifolds
---
Everything generalizes to a Riemannian manifold $(\mathcal{M}, g)$: in local coordinates $\mathrm{div}_g(u) = \frac{1}{\sqrt{|g|}}\partial_i\big(\sqrt{|g|}\, u^i\big)$, the divergence theorem holds with the Riemannian volume/area elements, and the Riemannian continuity equation $\partial_t p_t + \mathrm{div}_g(p_t u_t) = 0$ underpins Riemannian Flow Matching (velocities in tangent spaces, geodesic conditional flows — see [[Flow Matching]]).

## Relationship to Other Notes
---
- [[Flow Matching]] — the continuity equation (= divergence theorem applied to probability flux $p_t u_t$) defines what it means for a velocity field to generate a probability path; instantaneous change of variables gives its exact likelihoods; divergence-free components give sampling freedom
- [[Score-Based Generative Modeling Methods]] — Hyvärinen's tractable score matching objective is integration by parts (Stein's identity); the Fokker–Planck equation governs the SDE view
- [[DDPM]] — denoising score matching sidesteps the divergence term; the probability-flow ODE is a divergence rewrite of Fokker–Planck
- [[MCMC]] — Langevin dynamics preserve $p_t$ because their generator is divergence-free w.r.t. $p_t$; Metropolis–Hastings' detailed balance is the discrete analog
- [[Linear Algebra]] — divergence = trace of the Jacobian; Hutchinson's estimator = randomized trace estimation
- [[KL Divergence]] — unrelated despite the name: statistical divergences compare distributions; vector-field divergence measures local flux
