### [[dl_moc]]  
# **Discrete Flow Matching Policy**

**Tags**:: #generative #rl
**Links**:: [[VLA]] · [[Flow Matching]] · [[Behavior Cloning]] · [[Generative Modeling]] · [[Transformer]] · [[LLMs]]


A **discrete flow matching policy** generates robot actions as **discrete token sequences refined by a CTMC** (discrete flow matching / discrete diffusion), instead of decoding them autoregressively or regressing them with a continuous flow head: tokenize the action chunk, mask it, and train the VLM backbone to *un-mask* it — with the **same cross-entropy loss as language modeling**. The machinery is 6.S184 §7 ([Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)) / Discrete Flow Matching ([Gat et al., 2024](https://arxiv.org/abs/2407.15595); [Campbell et al., 2024](https://arxiv.org/abs/2402.04997)); the flagship policy instantiation is **Discrete Diffusion VLA** ([Liang et al., 2025, arXiv 2508.20072](https://arxiv.org/abs/2508.20072)). (This is a concept note — no single paper carries this exact title.)

**Notation** (6.S184 §7; matches the [[Flow Matching]]/[[Score Matching]] notes' $z$-for-data convention): vocabulary $\mathcal{V}$ (size $V$), data point $z = (z^1, \dots, z^d) \in \mathcal{V}^d$ — here a **tokenized action chunk**; CTMC $(X_t)$, rate matrix $Q_t(y|x)$, schedule $\kappa_t$ with $\kappa_0 = 0, \kappa_1 = 1$; conditioning (observation, language) enters every network input, exactly as guidance $y$ does in FM.

## Background: Policies as Conditional Generative Models
---
A [[VLA]]/[[Behavior Cloning]] policy is a conditional generative model $\pi(\text{action chunk} \mid \text{observation}, \text{language})$ — demonstrations are multimodal, so the *decoder family* matters. The design axis (see [[VLA]]): **autoregressive tokens** (RT-2, OpenVLA, π0-FAST — left-to-right, slow), **continuous regression** (OpenVLA-OFT — fast but unimodal), **continuous diffusion/flow heads** (Diffusion Policy, π0 — multimodal but a separate expert module). Discrete FM takes the fourth cell: **token actions, generated in parallel by iterative refinement** — multimodal *and* backbone-native.

## The Machinery: Discrete FM in Brief (6.S184 §7)
---
**CTMC + masked mixture path.** Extend the vocabulary with a `[mask]` token and set $p_\text{init} = \delta_{[\text{mask}]^d}$ (the all-masked sequence). The conditional path reveals each token independently with probability $\kappa_t$: $X_t^j = z^j$ w.p. $\kappa_t$, else `[mask]` — a "teleporting" interpolation from mask to data (MDLM construction, 6.S184 Ex. 39).

**Conditional → marginal rate matrix** (6.S184 Thm 36, the discrete Marginalization Trick): the conditional rate matrix only ever jumps a token straight to its data value, $Q_t^z(v, j|x) = \frac{\dot\kappa_t}{1-\kappa_t}(\delta_{z^j}(v) - \delta_{x^j}(v))$, and the marginal rate is its posterior average — which collapses to a **per-position classifier** (Thm 38):

$$
Q_t(v, j \mid x) = \frac{\dot\kappa_t}{1-\kappa_t}\Big( p_{1|t}(z^j = v \mid x) - \delta_{x^j}(v) \Big),
$$

so learning the generative model *is* learning the denoising posteriors $p^\theta_{1|t}(z^j = v|x)$ — a $d \times V$ softmax head on a standard [[Transformer]].

**Loss = cross-entropy** (6.S184's $\mathcal{L}_\text{DFM}$): as continuous FM reduces to regression and [[Score Matching]] to denoising, discrete FM reduces to **classification**:

$$
\mathcal{L}_\text{DFM}(\theta) = \mathbb{E}_{z \sim p_\text{data},\, t \sim \text{Unif},\, x \sim p_t(\cdot|z)}\Big[ -\sum_{j=1}^{d} \log p^\theta_{1|t}(z^j \mid x) \Big].
$$

Training (Alg. 8): sample $z$, mask each token w.p. $1 - \kappa_t$, predict the original tokens. Sampling (Alg. 7): start all-masked, take parallel per-token CTMC Euler steps — each round, unmask some positions by sampling from $p^\theta_{1|t}$; divergence-free rate components allow **re-masking** already-decided tokens (corrector steps).

## The Policy Instantiation
---
1. **Tokenize actions**: discretize each control dimension (position, orientation, gripper) into bins — quantile-based binning in Discrete Diffusion VLA — or compress the chunk first (FAST's DCT+BPE); group into a fixed-length chunk of $d$ tokens.
2. **One backbone, one loss**: a single-transformer VLM consumes (images, instruction, masked action tokens) and outputs per-position logits — trained with the **same cross-entropy objective as its text pretraining**. No separate action expert, no new loss type.
3. **Decode by iterative unmasking**: start from $[\text{mask}]^d$; over a *fixed, small* number of refinement rounds, commit the highest-confidence tokens first (**adaptive easy-first ordering**) and **re-mask** low-confidence ones for revision (the corrector freedom above).

**Discrete Diffusion VLA results** ([2508.20072](https://arxiv.org/abs/2508.20072)): 96.3% average success on LIBERO, 71.2% visual matching on SimplerEnv-Fractal, 54.2% overall on SimplerEnv-Bridge — beating both autoregressive and continuous-diffusion VLA baselines with the same backbone class.

## Why Discrete FM for Action Decoding
---
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
- [[VLA]] — the action-representation design axis this note extends; knowledge insulation, FAST tokens, OpenVLA-OFT evidence
- [[Flow Matching]] — the parent framework; its Discrete FM (CTMC) section is the theory this note applies, and π0's continuous action expert is the alternative
- [[Behavior Cloning]] — the underlying imitation objective; chunking and compounding errors
- [[Score Matching]] — the same conditional-expectation lemma, in its classification instance here
- [[Generative Modeling]] — where discrete FM sits in the family taxonomy
- [[LLMs]] / [[Transformer]] — masked-diffusion language modeling is the same construction on text; the policy inherits the VLM's training recipe
