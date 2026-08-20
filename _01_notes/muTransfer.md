### [[dl_moc]]
# **Maximal Update Parametrization (μP)**

**Tags**:: #dl_theo #learning #llm
**Links**:: [[Scaling Laws]] · [[Optimization Algorithms]] · [[AdamW]] · [[Muon]] · [[Learning Rate Annealing]] · [[Neural Network Initialization]] · [[Transformer]] · [[Attention]] · [[LLMs]]

**μTransfer** ([Yang, Hu et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", NeurIPS 2021](https://arxiv.org/abs/2203.03466)) solves the problem that frontier-scale models can only be trained once: hyperparameters cannot be tuned directly at the target scale. The method: parametrize the model in the **Maximal Update Parametrization (μP)**, under which many optimal HPs become *stable across model width*; tune a small proxy model cheaply; copy the HPs to the full-size model **zero-shot** — no tuning of the large model at all. Headline results: BERT-large beaten by transferring pretraining HPs from a **13M**-parameter proxy, and published GPT-3 6.7B beaten by transferring from a **40M** proxy (168× smaller) at a tuning cost of only **7% of pretraining compute**. Notation: width $n$ (fan_in of a hidden layer), width multiplier $m_d = n / n_{\text{base}}$, learning rate $\eta$.

### The Problem: HPs Don't Transfer in Standard Parametrization
---
In **standard parametrization (SP)** — PyTorch defaults, $1/\text{fan\_in}$ init variance, one global LR — the optimal learning rate **shifts by roughly an order of magnitude** as width grows (paper Fig. 1), so small-model tuning is misleading. Worse, in SP **wider is not always better** at a fixed HP setting: some layers update too fast and others too slow as width grows (logits and attention logits blow up while embeddings barely move), forcing a global LR reduction that starves the slow layers. In μP, at fixed HPs, the optimal LR is stable and **wider is monotonically better** — the sanity check the paper suggests for any scaling setup.

*Why SP blows up:* after one SGD step on a linear net, the correlation between the updated weights and the forward activations makes the output scale like $\Theta(n)$ by the Law of Large Numbers rather than the $\Theta(\sqrt n)$ that independence-based init analysis assumes. Init variance is chosen for the first forward pass; it says nothing about *training* dynamics.

### Theory: abc-Parametrizations and Maximal Feature Learning
---
An **abc-parametrization** writes each weight as $W = n^{-a} w$, with $w \sim \mathcal N(0, n^{-2b})$ and per-layer LR $\eta\, n^{-c}$. (For SGD there is a symmetry $(a,b,c) \to (a+\theta,\, b-\theta,\, c-2\theta)$ leaving dynamics unchanged, which is why the same μP can be written with scalings moved between multiplier, init, and LR — the paper's Table 3 vs Table 8 forms.)

The **dynamical dichotomy** of Tensor Programs IV ([Yang & Hu, "Feature Learning in Infinite-Width Neural Networks"](https://arxiv.org/abs/2011.14522)): every non-trivial stable abc-parametrization has an infinite-width limit that is *either* a **kernel regime** (features frozen, e.g. the NTK parametrization) *or* a **feature-learning regime**. **μP is the unique parametrization (up to the symmetry) in which every layer learns features maximally** — each layer's activations change by $\Theta(1)$ per step, independent of width, with nothing blowing up and nothing freezing. SP is not stable in the wide limit at fixed LR; NTK is stable but learns no features. Because the training dynamics converge to a width-independent limit, the optimal HPs converge too — *that* is why they transfer across width.

### The μP Recipe (Table 3)
---
Classify weights by whether their dimensions grow with width: **input weights** (fan_in fixed, fan_out $= n$), **hidden weights** ($n \times n$), **output weights** (fan_in $= n$, fan_out fixed); biases and LayerNorm gains count as input-like (kept $\Theta(1)$, constant Adam LR).

| | Input weights & biases | Hidden weights | Output weights |
|---|---|---|---|
| Init variance | $1/\text{fan\_in}$ (= $\Theta(1)$) | $1/\text{fan\_in}$ | $1/\text{fan\_in}^2$ |
| SGD LR | $\eta \cdot \text{fan\_out}$ | $\eta$ | $\eta / \text{fan\_in}$ |
| **Adam LR** | $\eta$ | $\eta/\text{fan\_in}$ | $\eta / \text{fan\_in}$ |

**Why Adam's hidden LR shrinks as $1/n$:** Adam's update entries are $\Theta(\eta)$ regardless of gradient scale (sign-like normalization). The update is *correlated* with the incoming activations, so the change in each output coordinate is a sum of $n$ correlated $\Theta(\eta)$ terms $\approx n\eta$ by LLN; keeping it $\Theta(1)$ forces $\eta \propto 1/n$. The same LLN argument gives the two other famous μP rules:

- **$1/d$ attention instead of $1/\sqrt d$**: attention logits are computed as $q^\top k / d_{\text{head}}$, because during training $q$ and $k$ become correlated, so $q^\top k$ scales like $d$, not $\sqrt d$.
- **Output logits scaled down**: via $1/\text{fan\_in}^2$ init variance plus $\eta/\text{fan\_in}$ LR (equivalently in the $m_d$ form: keep $1/\text{fan\_in}$ init but multiply the logits by $1/m_d$).

**Practical transformer form** (paper Table 8, width multiplier $m_d$, Adam): embedding and all vector-like params keep LR $\eta$ and $\Theta(1)$ init; every matrix (attention/MLP/output) uses LR $\eta / m_d$; output logits get a $1/m_d$ multiplier; attention uses $1/d_{\text{head}}$. At $m_d = 1$ (base width) μP coincides with SP — μP only changes how things *scale*.

**Implementation**: the official [`mup` package](https://github.com/microsoft/mup) — wrap the model, call `set_base_shapes(model, base_model)` *before* init and optimizer creation, use `MuAdam`/`MuSGD` and `MuReadout` for the output layer. Correctness is verified with a **coord check**: plot per-layer activation coordinate sizes across widths during the first training steps — under correct μP the curves are horizontal (width-independent); under SP they fan out.

### What Transfers (Table 1)
---
| | Examples | Status |
|---|---|---|
| **μTransferable** | learning rate, LR schedule, momentum / Adam $\beta$s, per-layer init scales, parameter multipliers (output/embedding, attention temperature) | transfer across width — theoretically justified |
| **Transferred across (empirical)** | depth, batch size, sequence length, training steps | works in practice with caveats, no width-style theory |
| **Not μTransferable** | regularization HPs: dropout, weight decay | depend on both model *and data* size, not just width |

Minimums for reliable transfer (paper's rule of thumb): proxy width ≥ 256, depth ≥ 4, batch ≥ 32, sequence length ≥ 128, ≥ 5000 training steps.

### Results
---
- **BERT-large (350M)**: pretraining HPs transferred from a 13M proxy outperform the published BERT-large numbers; total tuning cost ≈ training one BERT-large once.
- **GPT-3 6.7B**: random-search HP tuning (LR, multipliers, init scale, schedule) entirely on a 40M proxy, then one zero-shot 6.7B run: "The resulting µTransfer model outperforms the 6.7B [of Brown et al.], and is in fact comparable to the twice-as-large 13B model across our evaluation suite." Tuning cost: 7% of pretraining compute.

### Impact and Follow-Ups
---
- **Adoption**: [Cerebras-GPT](https://arxiv.org/abs/2304.03208) (111M–13B family trained compute-optimally with μP), [MiniCPM](https://arxiv.org/abs/2404.06395) (μP-style tensor programs scaling + WSD schedule — see [[Learning Rate Annealing]]). μP also makes [[Scaling Laws]] sweeps cleaner: each model size runs near *its* optimal HPs, removing a major confounder in fitted laws.
- **Depth-μP**: width transfer is the theoretically grounded axis; depth transfer was empirical (and weakest for post-LN). [Tensor Programs VI](https://arxiv.org/abs/2310.02244) extends feature-learning limits to infinite depth for residual networks.
- **Spectral view → [[Muon]]**: [Yang, Simon & Bernstein 2023](https://arxiv.org/abs/2310.17813) re-derive μP as a **spectral condition**: feature learning at every width follows if each layer's weights and updates satisfy $\|W\|_2 \asymp \|\Delta W\|_2 \asymp \sqrt{\text{fan\_out}/\text{fan\_in}}$. This norm-based reading of μP is the conceptual bridge to spectral-norm steepest descent and hence [[Muon]] and modular duality — μP by parametrization vs μP by optimizer design.
- **u-μP** ([Blake et al. 2024](https://arxiv.org/abs/2407.17465)): combines μP with unit scaling for low-precision-friendly training and simpler default multipliers.

### Caveats
---
- μP fixes *width* scaling only; transferring across data size / training duration is a separate problem (regularization HPs explicitly do not transfer).
- The proxy must not be too small (minimums above), and anything that changes with scale outside the parametrization — data mix, tokenizer, architecture tricks sensitive to init — can still break transfer.
- HP stability claims are about the *optimum location*, not the loss value: the transferred large model still needs its one training run; μTransfer removes the sweep, not the run.
