### [[dl_moc]]  
# **Discrete Flow Matching**

**Tags**:: #generative #dl_theo
**Links**:: [[Flow Matching]] · [[CTMC]] · [[Generative Modeling]] · [[LLMs]] · [[Transformer]] · [[Score Matching]] · [[VLA]]


**Discrete Flow Matching (DFM)** transfers the [[Flow Matching]] recipe to **discrete data** — token sequences instead of vectors: pick a probability path interpolating noise (typically all-`[mask]`) into the data distribution, derive its conditional rate matrix in closed form, learn the marginal dynamics by posterior averaging, and sample by simulating a [[CTMC]]. The punchline mirrors the continuous story: as continuous FM reduces to *regression* and [[Score Matching]] to *denoising*, **discrete FM reduces to classification** — the loss is the **same cross-entropy as language modeling**. This note follows **§7 of the MIT 6.S184 lecture notes** ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf); [course site](https://diffusion.csail.mit.edu/)); original papers: [Gat et al., 2024](https://arxiv.org/abs/2407.15595) (Discrete Flow Matching) and [Campbell et al., 2024](https://arxiv.org/abs/2402.04997) (generative flows on discrete spaces).

**Notation** (6.S184 §7; matches the [[Flow Matching]]/[[Score Matching]] notes' $z$-for-data convention): vocabulary $\mathcal{V}$ (size $V$), state space $S = \mathcal{V}^d$, data point $z = (z^1, \dots, z^d) \in S$; CTMC $(X_t)$ with rate matrix $Q_t(y|x)$; schedule $\kappa_t$ monotone with $\kappa_0 = 0, \kappa_1 = 1$; posterior $p_{1|t}(z|x)$. The process machinery — rate conditions, Kolmogorov Forward Equation, existence/uniqueness, categorical Euler sampling — lives in [[CTMC]]; this note is the *generative training framework* built on top of it.

## The DFM Recipe
---
The construction mirrors continuous [[Flow Matching]] step for step, with PMFs replacing densities and rate matrices replacing velocity fields (see the dictionary table in [[CTMC]]).

**1. Probability path — masked mixture path.** Extend the vocabulary with a `[mask]` token and set $p_\text{init} = \delta_{[\text{mask}]^d}$ (the all-masked sequence). The conditional path reveals each token independently with probability $\kappa_t$:

$$
p_t(x \mid z) = \prod_{j=1}^{d} \Big[ \kappa_t\, \delta_{z^j}(x^j) + (1-\kappa_t)\, \delta_{[\text{mask}]}(x^j) \Big],
$$

i.e. $X_t^j = z^j$ w.p. $\kappa_t$, else `[mask]` — a "teleporting" interpolation from mask to data (the MDLM construction, 6.S184 Ex. 39; uniform noise $p_\text{init} = \text{Unif}(\mathcal{V})^d$ is the other standard choice). The marginal path $p_t(x) = \sum_z p_t(x|z)\, p_\text{data}(z)$ interpolates $p_\text{init} \to p_\text{data}$.

**2. Conditional rate matrix.** For the mixture path it exists in closed form and only permits jumps straight to the data token (per position $j$, target value $v$):

$$
Q_t^z(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\,\big(\delta_{z^j}(v) - \delta_{x^j}(v)\big).
$$

**3. Conditional → marginal** (6.S184 Thm 36, the **discrete Marginalization Trick** — statement and proof in [[CTMC]]): the marginal rate matrix is the posterior average $Q_t(y|x) = \sum_z Q_t^z(y|x)\, p_{1|t}(z|x)$, and for the factorized mixture path it collapses to a **per-position classifier** (Thm 38):

$$
Q_t(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\Big( p_{1|t}(z^j = v \mid x) - \delta_{x^j}(v) \Big),
$$

so learning the generative model *is* learning the denoising posteriors $p^\theta_{1|t}(z^j = v \mid x)$ — a $d \times V$ softmax head on a standard [[Transformer]] (the factorization condition in [[CTMC]] is what makes the output tractable).

## Loss = Cross-Entropy
---
Since a posterior is a conditional expectation, it is learnable from single data samples by the same Bregman-divergence principle as $\mathcal{L}_\text{CFM}$ ([[Flow Matching]], FM and CFM Losses) — here with the log-loss as the Bregman divergence:

$$
\mathcal{L}_\text{DFM}(\theta) = \mathbb{E}_{z \sim p_\text{data},\, t \sim \text{Unif},\, x \sim p_t(\cdot|z)}\Big[ -\sum_{j=1}^{d} \log p^\theta_{1|t}(z^j \mid x) \Big],
$$

whose minimizer is the true posterior. A **generalized-KL** variant of the loss doubles as an [[ELBO]] for likelihood evaluation. No rate matrices appear in training at all — they are reconstructed from the classifier at sampling time.

## Training and Sampling Algorithms
---
**Training** (6.S184 Alg. 8): sample $z \sim p_\text{data}$, $t \sim \text{Unif}[0,1]$; mask each token independently w.p. $1 - \kappa_t$; predict the original tokens at the masked positions with cross-entropy. This is exactly masked-language-model training with a random masking rate.

**Sampling** (6.S184 Alg. 7): start from $X_0 = [\text{mask}]^d$; take parallel per-token categorical Euler steps of the CTMC with the reconstructed marginal rates — each round, sample $\hat z^j \sim p^\theta_{1|t}(\cdot|x)$ per position and jump-or-stay. Per-token parallel steps agree with the full CTMC step to $o(h)$ ([[CTMC]], factorization).

**Corrector steps / re-masking.** Any probability-preserving (divergence-free) rate component can be added without changing the marginals ([[CTMC]], KFE consequence) — for masked paths this licenses **re-masking already-decided tokens**, giving iterative refinement an error-correction mechanism that autoregressive decoding lacks. This is the discrete analog of Langevin corrector steps in continuous FM/diffusion.

## Where DFM Sits
---
| | Continuous FM | Score Matching | **Discrete FM** |
|---|---|---|---|
| Data | $\mathbb{R}^d$ | $\mathbb{R}^d$ | $\mathcal{V}^d$ (tokens) |
| Process | ODE flow | SDE | [[CTMC]] |
| Learned object | velocity $u_t^\theta$ | score $\nabla \log p_t$ | posteriors $p^\theta_{1|t}(z^j\|x)$ |
| Loss reduces to | regression | denoising | **classification (CE)** |
| Mass conservation | Continuity Eq. | Fokker–Planck | Kolmogorov Forward Eq. |

Generator Matching ([[Flow Matching]], last section) unifies all three: a CTMC is a jump process whose generator $f^\top Q_t$ sits beside $\nabla f^\top u_t$ (flow) and $\tfrac{1}{2}\sigma_t^2 \nabla^2 f$ (diffusion), and factorized product paths let continuous and discrete FM train jointly in one multimodal model.

**Applications**: masked diffusion language models ([[LLMs]] beyond autoregression — "discrete diffusion" LMs are exactly this construction), multimodal protein co-design (Multiflow), and token-action robot policies (below).

## Application: Discrete FM Policies ([[VLA]])
---
A [[VLA]]/[[Behavior Cloning]] policy is a conditional generative model $\pi(\text{action chunk} \mid \text{observation}, \text{language})$ — demonstrations are multimodal, so the *decoder family* matters. The design axis (see [[VLA]]): **autoregressive tokens** (RT-2, OpenVLA, π0-FAST — left-to-right, slow), **continuous regression** (OpenVLA-OFT — fast but unimodal), **continuous diffusion/flow heads** (Diffusion Policy, π0 — multimodal but a separate expert module). Discrete FM takes the fourth cell: **token actions, generated in parallel by iterative refinement** — multimodal *and* backbone-native. Conditioning enters every network input, exactly as guidance $y$ does in FM. The flagship instantiation is **Discrete Diffusion VLA** ([Liang et al., 2025, arXiv 2508.20072](https://arxiv.org/abs/2508.20072)).

**The recipe, instantiated:**
1. **Tokenize actions**: discretize each control dimension (position, orientation, gripper) into bins — quantile-based binning in Discrete Diffusion VLA — or compress the chunk first (FAST's DCT+BPE); group into a fixed-length chunk of $d$ tokens. This chunk is the data point $z$.
2. **One backbone, one loss**: a single-transformer VLM consumes (images, instruction, masked action tokens) and outputs per-position logits — trained with the **same cross-entropy objective as its text pretraining**. No separate action expert, no new loss type.
3. **Decode by iterative unmasking**: start from $[\text{mask}]^d$; over a *fixed, small* number of refinement rounds, commit the highest-confidence tokens first (**adaptive easy-first ordering**) and **re-mask** low-confidence ones for revision (the corrector freedom above).

**Discrete Diffusion VLA results** ([2508.20072](https://arxiv.org/abs/2508.20072)): 96.3% average success on LIBERO, 71.2% visual matching on SimplerEnv-Fractal, 54.2% overall on SimplerEnv-Bridge — beating both autoregressive and continuous-diffusion VLA baselines with the same backbone class.

**Why discrete FM for action decoding:**
- **vs. autoregressive tokens**: parallel any-order refinement replaces the left-to-right bottleneck — a few rounds instead of $d$ sequential steps; ordering adapts to difficulty (easy action dims commit first); re-masking gives **error correction** AR decoding cannot do.
- **vs. continuous flow/diffusion heads**: no freshly-initialized expert module whose gradients degrade the pretrained VLM — the *knowledge insulation* problem of π0-style experts ([[VLA]]) disappears because action decoding **is** the VLM's native objective; scaling recipes and infrastructure transfer unchanged.
- **Costs**: quantization error from binning (mitigated by finer bins/FAST compression); still multi-round inference (though rounds ≪ $d$); per-token independence within a round can hurt chunk consistency (re-masking helps). OpenVLA-OFT's 76.5% → 97.1% LIBERO jump from *changing only the decoding scheme* is the standing evidence that this axis, not the backbone, is the bottleneck.

| Action decoder | Objective | Inference | Backbone integration | Weakness | Examples |
|---|---|---|---|---|---|
| AR tokens | next-token CE | $d$ sequential steps | native | slow, rigid order | RT-2, OpenVLA, π0-FAST |
| Continuous regression | L1/MSE | 1 pass | head swap | unimodal | OpenVLA-OFT |
| Continuous FM/diffusion head | CFM / DSM regression | ~10 ODE steps | separate expert (+insulation issues) | extra module | π0, GR00T N1, Diffusion Policy |
| **Discrete FM / diffusion** | masked-token CE | few parallel rounds | native (same CE loss) | quantization | Discrete Diffusion VLA |

## Relationship to Other Notes
---
- [[Flow Matching]] — the continuous-state parent framework; its Discrete FM (CTMC) section is the summary this note expands, and the Bregman-divergence principle behind $\mathcal{L}_\text{CFM}$ is what makes cross-entropy a valid loss here
- [[CTMC]] — the underlying process: rate matrices, KFE, marginalization-trick proofs, factorization condition, categorical Euler sampling
- [[Score Matching]] — the same conditional-expectation lemma, in its classification instance here
- [[Generative Modeling]] — where discrete FM sits in the family taxonomy
- [[LLMs]] / [[Transformer]] — masked-diffusion language modeling is the same construction on text; DFM training *is* MLM with a random masking rate
- [[VLA]] — the action-representation design axis the policy application extends; knowledge insulation, FAST tokens, OpenVLA-OFT evidence
- [[Behavior Cloning]] — the underlying imitation objective for the policy application; chunking and compounding errors
