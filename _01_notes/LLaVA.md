### [[dl_moc]]  
# **Large Language and Vision Assistant (LLaVA)**

**Tags**:: #llm #vision #learning
**Links**:: [[VLM]] · [[CLIP]] · [[LLaMA]] · [[LLMs]] · [[ViT]] · [[RLHF]] · [[Pre-training]]


**LLaVA** ([Liu et al., 2023](https://arxiv.org/abs/2304.08485), *Visual Instruction Tuning*, NeurIPS 2023) is the first attempt to extend **instruction tuning** to the multimodal setting. Its central insight is *data-centric*: use a **text-only** GPT-4 to synthesize multimodal instruction-following data, then train a lightweight connector to bridge a frozen [[CLIP]] vision encoder and an open LLM ([[LLaMA|Vicuna]]). The result is a general-purpose visual assistant that follows free-form language instructions grounded in an image — a template that most later open VLMs ([[VLM]]) inherit.

## Architecture
---
LLaVA connects three components; only the middle one is new:

- **Vision encoder** $g(\cdot)$: a frozen [[CLIP]] **ViT-L/14** ([[ViT]]). 
	For an input image $X_v$ --- it produces grid features $Z_v = g(X_v)$ (the patch-token grid, before/after the last transformer layer).
- **Projection** $W$: a trainable matrix (a *linear* layer in LLaVA-1.0) that maps visual features into the language embedding space, yielding **visual tokens** $H_v = W \cdot Z_v$. These have the same dimensionality as the LLM's word embeddings.
- **LLM** $f_\phi$: [[LLaMA|Vicuna]] (7B / 13B), which autoregressively decodes a response conditioned on the interleaved sequence of visual tokens $H_v$ and text instruction tokens $H_{\text{instruct}}$.

The projector is deliberately minimal — the paper's thesis is that the visual features are already rich (CLIP is pre-trained at scale), so a thin adapter plus instruction data suffices.

## Training Objective
---
LLaVA is trained with the standard **autoregressive language-modeling loss**, but the loss is computed **only over the assistant's answer tokens** $X_a$ (the image and human-instruction tokens are context, not targets). For a response of length $L$:
$$
p(X_a \mid X_v, X_{\text{instruct}}) = \prod_{i=1}^{L} p_\theta\big(x_i \mid X_v,\; X_{\text{instruct}, <i},\; X_{a, <i}\big)
$$
where $\theta = \{W, \phi\}$ are the trainable parameters. Multi-turn conversations are formatted into a single sequence, masking everything except the current-turn answer.

## Two-Stage Training
---
**Stage 1 — Feature Alignment Pre-training.**
Only the projection $W$ is trained; both the vision encoder and the LLM are **frozen**. Data is a filtered subset of CC3M (~595K image–text pairs) reformatted into naive instruction pairs ("describe the image" → caption). This aligns visual tokens with the LLM's word-embedding space — effectively training a "visual tokenizer" compatible with the frozen LLM.

**Stage 2 — End-to-End Instruction Fine-tuning.**
The vision encoder stays frozen; **both** the projection $W$ and the LLM weights $\phi$ are updated on the synthesized instruction data. This is where the model learns to *follow instructions* rather than just caption.

| Stage | Vision encoder | Projection $W$ | LLM $\phi$ | Data |
|-------|:---:|:---:|:---:|------|
| 1 · Alignment | ❄️ frozen | 🔥 train | ❄️ frozen | 595K CC3M pairs |
| 2 · Instruction | ❄️ frozen | 🔥 train | 🔥 train | 158K GPT-4 instructions |

## GPT-4-Generated Instruction Data
---
The key enabler: there was no large multimodal instruction dataset, so LLaVA **bootstraps one using a language-only GPT-4**. Since GPT-4 (text) cannot see images, each image is presented via its **symbolic representation** — the ground-truth **captions** plus **bounding boxes** (from COCO) — and GPT-4 is prompted to generate three data types (158K samples total):

1. **Conversation** (58K) — multi-turn Q&A about visible objects, counts, actions, and locations.
2. **Detailed description** (23K) — a rich, comprehensive paragraph describing the scene.
3. **Complex reasoning** (77K) — questions requiring inference beyond what is directly visible (e.g., "why might this be dangerous?").

This *distills* a strong text model's reasoning into a multimodal model without any human-labeled instruction data — a self-instruct pattern applied across modalities.

## LLaVA-1.5
---
[Liu et al., 2023b](https://arxiv.org/abs/2310.03744) shows that a few simple changes make LLaVA SOTA on 11 benchmarks while remaining strikingly **data- and compute-efficient** (trained in ~1 day on 8×A100):

- **MLP projector** — replace the single linear layer with a **2-layer MLP**, improving the vision–language connection.
- **Higher resolution** — CLIP **ViT-L/336px** instead of 224px, giving more visual detail.
- **Academic-task data** — add VQA / OCR / region-level datasets with explicit **response-format prompts** (e.g., "answer with a single word or phrase") to control output length for short-answer benchmarks.
- **Scale** — Vicuna-13B backbone.

These made LLaVA the de-facto baseline recipe: *frozen CLIP + MLP projector + instruction-tuned open LLM*.

## Significance & Limitations
---
- **Template for open VLMs**: the "connector + frozen encoder + LLM, trained on synthetic instruction data" recipe underlies many later systems; contrast with [[CLIP]]'s contrastive dual-encoder (alignment only, no generation) and cross-attention approaches like BLIP-2/Flamingo.
- **Data-centric lesson**: capability came largely from *instruction data quality*, not architectural novelty.
- **Limitations**: inherits CLIP's weaknesses (fine-grained perception, counting, precise spatial relations); prone to **hallucination** (describing objects not present) — later addressed by RLHF-style methods (e.g., LLaVA-RLHF, see [[RLHF]]); low input resolution limits dense OCR / small-text reading (motivating dynamic-resolution designs in [[VLM|Qwen-VL]]).

## Relationship to Other Methods
---
| Model | Vision–language bridge | Generation? | Training signal |
|-------|------------------------|:---:|-----------------|
| [[CLIP]] | Shared contrastive space | No | Contrastive (InfoNCE) |
| BLIP-2 | Q-Former (cross-attn) | Yes | Frozen encoder + LLM |
| **LLaVA** | Linear / MLP projector | Yes | Visual instruction tuning |
| [[VLM|Qwen2.5-VL]] | MLP merger + dynamic res | Yes | Large-scale multimodal |



$$\text{AUC} = \frac{\sum_{i \in \text{TRUE}} \text{rank}_i - \frac{n_T(n_T+1)}{2}}{n_T \cdot n_F}$$