### [[dl_moc]]  
# **Cambrian: A Vision-Centric Exploration of Multimodal LLMs**

**Tags**:: #vision #llm #learning
**Links**:: [[VLM]] · [[LLaVA]] · [[CLIP]] · [[DINO]] · [[ViT]] · [[MAE]] · [[JEPA]] · [[Attention]] · [[World Models]] · [[CLS Token]]


**Cambrian-1** ([Tong et al., 2024](https://arxiv.org/abs/2406.16860), *A Fully Open, Vision-Centric Exploration of Multimodal LLMs*, NeurIPS 2024 **Oral**) is a large, systematic study of the **vision side** of multimodal LLMs, from Saining Xie's group at NYU (with Rob Fergus and Yann LeCun). Its thesis is a corrective one: while the field has poured effort into the language backbone, *"the design choices for vision components are often insufficiently explored and disconnected from visual representation learning research"* — and this gap *"hinders accurate sensory grounding in real-world scenarios."* Cambrian therefore flips the usual framing. Instead of treating the vision encoder as a solved, frozen component behind an ever-stronger LLM, it repurposes an instruction-tuned [[VLM]] as an **evaluation harness for visual representations** — a way to ask *which* visual features actually help a language model see.

The paper is deliberately a **"fully open cookbook"**: model weights, code, tooling, the instruction-tuning datasets, and detailed training *and* evaluation recipes are all released. It contributes (1) a benchmark of 20+ vision encoders, (2) the **Spatial Vision Aggregator (SVA)** connector, (3) the **CV-Bench** vision-centric benchmark plus a critique of existing MLLM evaluation, and (4) the **Cambrian-10M / Cambrian-7M** instruction-tuning data with curation findings. It ships three models — **Cambrian-1-8B** ([[LLaMA]]-3-8B-Instruct), **-13B** (Vicuna-1.5-13B), and **-34B** (Nous-Hermes-2-Yi-34B) — all of which fuse **four** vision encoders into just **576 visual tokens**.

## The Vision-Centric Thesis
---
Cambrian organizes its exploration of the MLLM design space around **five pillars**, each studied as a controlled ablation with the rest of the pipeline held fixed:

1. **Visual Representations** — which vision encoder(s)? (language-supervised vs self-supervised, resolution, ConvNet vs [[ViT]]).
2. **Connector Design** — how to feed visual features into the LLM (→ SVA).
3. **Instruction-Tuning Data** — what data mix produces a capable *and* conversational assistant (→ Cambrian-7M).
4. **Instruction-Tuning Recipes** — freezing/unfreezing, staging, learning rates.
5. **Benchmarking** — measuring genuine visual grounding rather than language priors (→ CV-Bench).

A key methodological move: because the MLLM itself is the measuring instrument, Cambrian argues that **ImageNet linear-probe accuracy is a poor proxy** for an encoder's usefulness inside an MLLM. The ranking of encoders under the MLLM harness differs markedly from their linear-probe ranking — e.g., [[MAE]] probes near the bottom yet is not the worst inside an MLLM, and SigLIP tops the MLLM ranking. Real-world sensory grounding, not ImageNet top-1, is the target.

## Visual Representations: Which Encoder Can an LLM See Through?
---
Cambrian benchmarks **23 backbone configurations** (13 model families) — each paired with Vicuna-1.5-7B under matched two-stage training — grouped into four supervision types:

- **Language-supervised**: [[CLIP|OpenAI CLIP]], SigLIP, DFN-CLIP, EVA-CLIP, and **OpenCLIP ConvNeXt** (a CLIP-trained ConvNet).
- **Self-supervised**: [[DINO|DINOv2]], [[MAE]], MoCo v3, [[JEPA|I-JEPA]].
- **Class-supervised**: ImageNet-22K supervised ViT.
- **Other**: SAM (segmentation), MiDaS (depth), Stable Diffusion VAE (generative).

**Findings:**

- **Language supervision wins across the board, most decisively on OCR & Chart.** The hypothesized cause is data: CLIP-style web captions (LAION) are text- and OCR-rich, whereas self-supervised encoders train on natural images with little embedded text. A footnote insists CLIP be read as *strongly supervised* — language is a dense supervisory signal.
- **The gap narrows on vision-centric tasks**, where a high-resolution **DINOv2 (ViT-L/14@518)** even beats several CLIP variants — it ranks last among language-supervised encoders overall but jumps to the top tier on vision-centric benchmarks.
- **The gap is an artifact of scale, and it is closable.** Language-supervised encoders train on 400M ([[CLIP]]) to 10B samples; DINOv2's ~142M is the largest self-supervised set. Scaling instruction-tuning data (737K → 5M) and **unfreezing the encoder** lets a DINOv2-based MLLM on 5M data surpass a CLIP-based one on 0.7M (*Finding 6*).
- **High-resolution and ConvNet encoders excel at OCR/Chart and vision-centric tasks**: OpenCLIP **ConvNeXt-XXL@1024** ranks ~2nd overall and 1st on OCR & Chart.
- **Combining encoders helps** (*Finding 7*). Concatenating features (at a fixed 576-token budget) from **CLIP + SigLIP + ConvNeXt + DINOv2** beats any single encoder; adding the self-supervised DINOv2 helps most on vision-centric tasks and even lifts OCR. This is the scaled-up descendant of the **Mixture-of-Features** idea from the *Eyes Wide Shut* precursor (see below).

Naïve concatenation has two flaws, though — **interpolation loss** (down-sampling high-res feature maps to a common grid destroys detail) and **equal treatment** of every encoder regardless of its resolution. Fixing these motivates the connector.

## Spatial Vision Aggregator (SVA)
---
SVA is a **spatially-aware, multi-encoder connector**. It fuses $N$ heterogeneous encoders — with different native resolutions and token counts — into a small, fixed set of visual tokens, *without* exploding the token budget and *without* interpolating everything to a common grid. (For context, LLaVA-NeXT needs **2,880** tokens for one 672px image; Cambrian uses **576**.)

The connector maintains a grid of **$L \times L$ learnable latent tokens** $X \in \mathbb{R}^{L^2 \times C}$ as queries; the encoders' feature maps $\{F_k\}_{k=1}^N$ are the keys and values. The defining property is a **spatial inductive bias**: the query at grid position $(i,j)$ attends *only* to the local sub-region of each encoder's feature map that aligns to that cell — not globally. For a per-cell cross-attention with a shared query projection $W_Q$ and per-encoder projections $W_K^k, W_V^k$:

$$
X_{(i,j)} \;\leftarrow\; X_{(i,j)} + \sum_{k=1}^{N} \text{CrossAttn}\!\Big(X_{(i,j)}W_Q,\ \; F_k^{(i,j)}W_K^k,\ \; F_k^{(i,j)}W_V^k\Big)
$$

where $F_k^{(i,j)}$ is the $m_k \times m_k$ block of encoder $k$'s feature map covering cell $(i,j)$ (each encoder's map is sized to a positive-integer multiple $m_k L$ of the query grid). Each of the $L^2$ tokens thus gathers $\sum_k m_k^2$ features while **preserving 2D structure** — the property a global [[Attention|cross-attention]] resampler / Q-Former lacks.

Two capacity knobs control the connector: **$D$**, the number of SVA cross-attention layers, and **$G$**, the number of parallel query groups. Critically, SVA is a **multi-layer** design — aggregation blocks are re-inserted throughout the LLM (standard stride 3), so the model repeatedly re-reads the *uncompressed* visual features at multiple depths rather than seeing them only once at the input. The released models use $L=24$ ($24^2 = 576$ tokens) with $D=3, G=1$.

**SVA vs. alternatives** (Table 4; four category averages General / Knowledge / OCR&Chart / Vision-Centric):

| Connector | General | Knowledge | OCR & Chart | Vision-Centric |
|-----------|:---:|:---:|:---:|:---:|
| **SVA** | **68.5** | **49.7** | **55.5** | **53.2** |
| Concatenation | 67.2 | 48.9 | 50.1 | 52.6 |
| Resampler (Q-Former-style) | 63.1 | 46.5 | **27.1** | 42.6 |

The plain resampler **collapses on OCR & Chart (27.1)**: with only global cross-attention it cannot condense high-resolution, spatially-structured text into a few tokens. SVA's largest gains are exactly there. *Finding 8*: "spatial inductive bias and deep interaction between LLM and vision features help to better aggregate and condense vision features."

## Benchmarking: CV-Bench and the "Who's Answering?" Problem
---
Cambrian argues that many popular MLLM benchmarks **under-measure visual grounding**. By clustering benchmarks and re-running each with the image *removed*, it finds several — ScienceQA-I, MMMU, MathVista, AI2D — that a model answers almost as well **without the image** (a <5% drop). These are dominated by the LLM's language priors: they test knowledge and reasoning, not perception. The paper accordingly works with four benchmark **categories** — General, Knowledge, OCR & Chart, and **Vision-Centric** — and builds a benchmark that genuinely requires sight.

**CV-Bench** repurposes ground-truth vision annotations into natural-language VQA, giving **2,638** manually-inspected questions:

- **2D** (~1,440 questions from ADE20K + COCO): **spatial relationships** and **object counting**.
- **3D** (~1,200 questions from Omni3D): **depth ordering** and **relative distance**.

At ~3.5× the size of RealWorldQA and ~8.8× MMVP, it is the paper's yardstick for vision-centric ability.

**Precursor — *Eyes Wide Shut?*** ([Tong et al., 2024](https://arxiv.org/abs/2401.06209), CVPR 2024). By the same lead/senior authors, this is the conceptual seed of Cambrian. It constructs **CLIP-blind pairs** — images [[CLIP]] encodes as nearly identical despite obvious visual differences — and turns them into the **MMVP** benchmark across nine basic visual patterns, on which even GPT-4V fails and hallucinates. Its remedy, **Mixture-of-Features (MoF)** — blending self-supervised [[DINO|DINOv2]] features into the MLLM — is precisely the multi-encoder intuition Cambrian scales up and formalizes as SVA.

## Instruction-Tuning Data: Cambrian-10M → Cambrian-7M
---
Cambrian assembles **Cambrian-10M** (~9.78M samples) from public sources across seven categories — general VQA, OCR, counting, code, math, science, and a slice of **language-only** data (to counter catastrophic forgetting). A *Targeted Internet Data Engine* synthesizes ~161K science QA pairs (pick a field → GPT-4 picks topics → search Wikipedia → extract image–caption tuples → generate Q&A), roughly +400% over prior science sources.

The central data lesson is **balance beats volume**:

- **Per-source thresholding** caps the long tail of over-represented datasets; the sweet spot is a threshold of ~250K–350K samples per source.
- **Category-ratio balancing** matters most for General / OCR / Language; too much OCR *hurts* general and vision-centric ability.

Applying this curation yields **Cambrian-7M** (~7M), which **outperforms the full 10M despite fewer samples** and lower training cost:

| Data | Avg | General | Knowledge | OCR & Chart | Vision-Centric |
|------|:---:|:---:|:---:|:---:|:---:|
| LLaVA-665K | 40.7 | 64.7 | 45.2 | 20.8 | 32.0 |
| Cambrian-10M | 54.8 | 68.7 | 51.6 | 47.3 | 51.4 |
| **Cambrian-7M** | **55.9** | **69.6** | **52.6** | **47.3** | **54.1** |

**The "answer machine" phenomenon.** A model tuned on too many short-answer VQA samples becomes benchmark-strong but conversationally broken — it *"defaults to outputting short, curt responses"* regardless of the question. The fix is a **system prompt** (e.g., *"Answer the question using a single word or phrase."*) prepended to short-answer training samples: it isolates the terse behavior to when it is asked for, restoring conversational ability with **benchmark scores unchanged**, and even nudges the model toward chain-of-thought ([[CoT Prompting]]) before answering.

**Recipe.** Cambrian keeps [[LLaVA]]'s two-stage structure:

| Stage | Vision encoders | SVA connector | LLM | Data |
|-------|:---:|:---:|:---:|------|
| 1 · Alignment | ❄️ frozen | 🔥 train | ❄️ frozen | ~2.5M adapter captions |
| 2 · Instruction | ❄️ frozen | 🔥 train | 🔥 train | Cambrian-7M |

Connector LR is 1e-4, LLM LR 2e-5. (Ablations show *unfreezing* the vision encoder is broadly beneficial, especially for self-supervised encoders on vision-centric tasks — *Finding 4* — but the released headline models keep encoders frozen.)

**Results.** Cambrian-1-34B is competitive with proprietary GPT-4V while using only 576 tokens. Note the paper reports **per-category averages, not a single overall number** — a frequently-misquoted point:

| Category | Cambrian-1-34B | GPT-4V |
|----------|:---:|:---:|
| General | **76.8** | 63.0 |
| Knowledge | **67.0** | 65.2 |
| OCR & Chart | 71.9 | **77.4** |
| Vision-Centric | **68.5** | 62.4 |

It leads GPT-4V on General, Knowledge, and Vision-Centric (and on MME-P, 1689.3 vs 1409.4, and MMVP, 52.7 vs 50.0) but **GPT-4V still leads on OCR & Chart**. Against open 34B peers **LLaVA-NeXT-34B and Mini-Gemini-HD-34B — which use 2,880 tokens** — Cambrian matches or beats them on OCR & Chart and Vision-Centric with **one-fifth the visual tokens**, the payoff of SVA.

## Follow-ups: Toward Spatial Supersensing
---
The Cambrian line extends its vision-centric agenda from static images into **video and 3D/spatial** understanding.

**Thinking in Space** ([Yang et al., 2024](https://arxiv.org/abs/2412.14171), NYU / Yale / Stanford, with Fei-Fei Li and Saining Xie) introduces **VSI-Bench** (Visual-Spatial Intelligence Benchmark, >5,000 QA pairs from egocentric indoor video). It finds MLLMs are *competitive but distinctly subhuman* at spatial reasoning, that **spatial reasoning is the bottleneck** (not perception or language), and — strikingly — that linguistic reasoning tricks (chain-of-thought, self-consistency, tree-of-thoughts) **fail to help**, whereas prompting the model to build an explicit **cognitive map** does. This is the conceptual bridge to Cambrian-S.

**Cambrian-S** ([Yang, Yang et al., 2025](https://arxiv.org/abs/2511.04670), *Towards Spatial Supersensing in Video*) argues that scaling data and context length alone will not produce true spatial understanding, and proposes a taxonomy of **spatial supersensing** in four stages:

1. **Semantic perception** — naming what is seen.
2. **Streaming event cognition** — memory across continuous experience.
3. **Implicit 3D spatial cognition** — inferring the world behind the pixels.
4. **Predictive world modeling** — an internal model that filters and organizes incoming information (cf. [[World Models]], [[JEPA]]).

To probe stages beyond perception it introduces **VSI-Super** (VSI-Super-**Recall** for long-horizon spatial recall, VSI-Super-**Count** for continual counting) and the **VSI-590K** spatial instruction dataset. Its mechanism, **predictive sensing**, trains a self-supervised **next-latent-frame predictor** whose **prediction error ("surprise")** gates memory management and event segmentation — a small step toward LeCun-style predictive world models. Training on VSI-590K yields **+30% on VSI-Bench** without hurting general ability, yet VSI-Super remains hard — the paper's evidence that *scale alone is insufficient* for spatial supersensing.

## Relationship to Other Methods
---
| System | Vision → LLM bridge | # Visual tokens | Core idea |
|--------|---------------------|:---:|-----------|
| [[CLIP]] | Shared contrastive space | — | Dual-encoder alignment; no generation |
| [[LLaVA]] (1.5) | Linear / MLP projector | ~576 | One frozen CLIP + visual instruction tuning |
| LLaVA-NeXT / Mini-Gemini-HD | Image tiling + MLP | 2,880 | High resolution via *more tokens* |
| [[VLM|Qwen-VL]] | Resampler (global cross-attn) | compressed | Condense features with a learned query set |
| **Cambrian-1** | **SVA** (spatial cross-attn, 4 encoders) | **576** | Multi-encoder fusion + spatial bias + multi-layer aggregation |

Cambrian is best read as the rigorous continuation of [[LLaVA]]'s recipe: it inherits the two-stage visual-instruction-tuning pipeline, uses LLaVA-665K as the baseline it dwarfs, and benchmarks against LLaVA-NeXT — but replaces the single frozen [[CLIP]] encoder with a **studied combination** of encoders ([[CLIP]] + SigLIP + ConvNeXt + [[DINO|DINOv2]]) and a spatially-aware connector. Its lasting contributions are methodological: use the MLLM *itself* as the yardstick for visual representations; take **self-supervised** features ([[DINO]], [[MAE]], [[JEPA]]) seriously as complements to language-supervised ones; measure real perception with CV-Bench; and treat data **curation** as a first-class lever. The follow-up thread ([[World Models]], predictive sensing) pushes the same "vision-centric" philosophy toward embodied, spatial understanding.
