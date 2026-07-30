### [[dl_moc]]
# **FiLM: Feature-wise Linear Modulation**

**Tags**:: #architecture #dl_theo
**Links**:: [[Normalization]] · [[Layer Normalization]] · [[DiT]] · [[DDPM]] · [[GAN]] · [[VLA]] · [[CNNs]] · [[VLM]] · [[Attention]]

**FiLM** ([Perez et al., AAAI 2018](https://arxiv.org/abs/1709.07871)) is a general-purpose conditioning mechanism: a network's intermediate features are modulated by a **feature-wise affine transformation** whose scale and shift are *predicted from the conditioning input*. Introduced for visual question answering, it has since become the default way to inject side information — a language instruction, a class label, a diffusion timestep, a style code — into a backbone, appearing everywhere from BigGAN and [[DiT]]'s adaLN-Zero to RT-1's language-conditioned image encoder. The unifying survey is Dumoulin et al.'s Distill article ["Feature-wise transformations"](https://distill.pub/2018/feature-wise-transformations/) (2018).

### Definition
---
Given features $x$ and conditioning input $z$, a **FiLM generator** (any small network) predicts per-feature coefficients, and the **FiLM layer** applies them:

$$
\text{FiLM}(x \mid z) = \gamma(z) \odot x + \beta(z).
$$

For a CNN feature map $F \in \mathbb{R}^{H \times W \times C}$, $\gamma, \beta \in \mathbb{R}^{C}$ act **per channel**, broadcast over all spatial locations — the same modulation everywhere in space. For a Transformer, they act per hidden dimension. The transformation is linear in $x$ but the *interaction* between $z$ and $x$ is **multiplicative** (via $\gamma$), which is what concatenation-based conditioning lacks:

**Concatenation is shift-only FiLM.** Concatenating $z$ to the input of a linear layer, $W[x; z] = W_x x + W_z z$, contributes only an input-dependent *bias* — a $\beta(z)$ with no $\gamma$. FiLM upgrades this to a scaling that can amplify, suppress ($\gamma \approx 0$), or sign-flip each feature, and can be inserted at every layer rather than only the input.

**FiLM is a constrained hypernetwork.** A full [hypernetwork](https://arxiv.org/abs/1609.09106) predicts an entire weight matrix from $z$; the FiLM generator predicts only $2C$ numbers per layer (a diagonal transformation). This is the "happy compromise between effectiveness and efficiency" (Distill): parameters scale linearly in the number of features, yet the modulation touches every layer of the computation.

### Original Architecture: Visual Reasoning on CLEVR
---
The AAAI paper's pipeline: a GRU encodes the question into $z$; a linear FiLM generator maps $z$ to $(\gamma_{n,c}, \beta_{n,c})$ for every channel $c$ of every residual block $n$; the image passes through a CNN whose 4 residual blocks each apply (conv → ReLU → conv → **BN without learned affine** → **FiLM** → ReLU) with a skip connection. The question thus reprograms the visual pipeline at every depth.

Results: **97.7% overall accuracy on CLEVR**, halving the previous state-of-the-art error — without program supervision, explicit reasoning modules, or data augmentation that competing neuro-symbolic approaches required.

**What γ and β learn.** The paper's analysis shows $\gamma$ values cluster around zero and take both signs: FiLM acts largely as a learned, condition-dependent **gating** over channels — selectively silencing or sign-flipping feature detectors. Ablations found scaling alone ($\gamma$-only) hurts far less than shifting alone ($\beta$-only), i.e. the multiplicative pathway carries most of the conditioning signal. FiLM parameters cluster by question type, and the model generalizes compositionally (CLEVR-CoGenT) better than baselines when fine-tuned.

### Relation to Conditional Normalization
---
FiLM is the normalization-free generalization of a family of "conditional normalization" methods — each is (normalize, then FiLM), differing in what they normalize and where $\gamma, \beta$ come from:

| Method | Normalizer | $\gamma, \beta$ predicted from | Origin |
|---|---|---|---|
| BN/LN learned affine | batch / layer stats | — (free parameters, no $z$) | standard |
| Conditional Instance Norm | instance stats | style *index* (embedding lookup) | style transfer ([Dumoulin et al. 2017](https://arxiv.org/abs/1610.07629)) |
| Conditional BatchNorm | batch stats | language embedding (MLP) | VQA ([de Vries et al. 2017](https://arxiv.org/abs/1707.00683)) |
| AdaIN | instance stats | *statistics* of a style image (no learned generator) | style transfer ([Huang & Belongie 2017](https://arxiv.org/abs/1703.06868)) |
| AdaGN | group stats | timestep + class embedding | diffusion ([ADM, Dhariwal & Nichol 2021](https://arxiv.org/abs/2105.05233)) |
| adaLN / adaLN-Zero | layer stats | timestep + class embedding | [[DiT]] ([Peebles & Xie 2023](https://arxiv.org/abs/2212.09748)) |
| **FiLM** | **none required** | any conditioning network | [Perez et al. 2018](https://arxiv.org/abs/1709.07871) |

The FiLM paper's key observation: the benefit does **not** depend on normalization — modulation works applied directly to activations. Pairing it with a normalizer whose own affine is disabled (as in the CLEVR model, ADM, and DiT) is convenient because normalization hands FiLM a standardized input, making $\gamma, \beta$'s job scale-free.

### FiLM in the Generative Stack
---
**Class-conditional GANs.** BigGAN ([Brock et al. 2019](https://arxiv.org/abs/1809.11096)) conditions its generator through class-conditional BatchNorm at every resolution. StyleGAN ([Karras et al. 2019](https://arxiv.org/abs/1812.04948)) is FiLM-as-architecture: the mapped latent $w$ drives AdaIN at every layer, so *all* generator conditioning flows through feature-wise modulation.

**Diffusion models.** The original [[DDPM]] U-Net injects the timestep embedding *additively* into each residual block — shift-only FiLM. ADM upgraded this to **AdaGN**, $\text{AdaGN}(h, z) = \gamma(z) \odot \text{GroupNorm}(h) + \beta(z)$, and found it clearly better. [[DiT]] carried the same idea to Transformers as **adaLN-Zero**, adding a third predicted coefficient — a residual-branch gate $\alpha$ initialized to zero so every block starts as the identity — which beat both cross-attention and in-context conditioning on FID at lower FLOPs (see the [[DiT]] note for the four-way comparison).

**Identity initialization** is the recurring design rule: parameterize the scale as $1 + \Delta\gamma$ with $\Delta\gamma$ (and $\beta$, and any gate) initialized to zero, so the conditioned network begins as the unconditioned one. RT-1 and DiT both rely on it — essential when bolting FiLM onto a *pretrained* backbone without wrecking its features.

### FiLM in Robotics
---
Feature-wise conditioning is the workhorse for language- and goal-conditioned policies:

- **RT-1** ([Brohan et al. 2022](https://arxiv.org/abs/2212.06817)): the instruction's Universal Sentence Encoder embedding conditions a pretrained EfficientNet-B3 through identity-initialized FiLM layers, so task-relevant visual features are selected *early* in the encoder rather than fused late; the resulting 81 vision-language tokens feed TokenLearner → decoder-only Transformer (see [[VLA]]).
- **Diffusion Policy** ([Chi et al. 2023](https://arxiv.org/abs/2303.04137)): the CNN (1D temporal U-Net) variant conditions every conv block on the observation embedding via FiLM — the same AdaGN pattern as image diffusion, applied to action denoising.

The appeal in robotics is the same as elsewhere: the conditioning signal (instruction, goal image, timestep) has a different modality and dimensionality than the features it must steer, and FiLM couples them with negligible parameter cost and no architectural surgery.

### FiLM vs Attention
---
FiLM and [[Attention]] are complementary conditioning mechanisms, not competitors:

- **FiLM is spatially/temporally uniform** — one $(\gamma, \beta)$ per channel, identical at every position. It reweights *what* (which features matter given $z$) but cannot select *where*. Cross-attention does position-dependent retrieval from the conditioning sequence, at quadratic cost and with per-layer KV projections.
- **Rule of thumb**: a global, low-dimensional condition (class, timestep, style, a single instruction embedding) → FiLM is cheaper and often better (DiT's ablation); a structured, many-token condition (a full text sequence for T2I, retrieved documents) → cross-attention, or both (e.g. text-to-image U-Nets use AdaGN for the timestep *and* cross-attention for the prompt).

### Summary
---
FiLM distills conditioning to its minimal effective form: *predict a per-feature scale and shift from the thing you want to condition on, and apply it everywhere*. Its special cases (conditional BN/IN, AdaIN, AdaGN, adaLN) differ only in bookkeeping; its descendants power class-conditional GANs, diffusion backbones, and language-conditioned robot policies. When a design question is "how do I inject $z$ into this network?", FiLM — with identity init — is the strong default before reaching for cross-attention.
