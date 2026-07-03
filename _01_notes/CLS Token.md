### [[dl_moc]]  
# **CLS Token in Vision Models & VLMs**

**Tags**:: #vision #architecture #llm
**Links**:: [[ViT]] · [[DINO]] · [[CLIP]] · [[VLM]] · [[Attention]] · [[Transformer]] · [[MAE]]

---

The **`[CLS]` token** (classification token) is a special learnable embedding prepended to the input sequence of a Transformer. It serves as a global **"summary"** of the entire input — attending to all other tokens over many layers, **==its final hidden state is used as the aggregate representation for downstream tasks.==**

---

## 1. Origins: BERT (NLP)

Introduced in **BERT** ([Devlin et al., 2019](https://arxiv.org/abs/1810.04805)), the `[CLS]` or `[BOS]` token is the first token in every input sequence. After the full Transformer encoder stack, the output at position 0 is taken as the **sequence-level representation** and fed to a linear classification head.

$$
\mathbf{h}_0^{(L)} = \text{BERT}([{\tt CLS}], x_1, \ldots, x_n)_0 \in \mathbb{R}^d
$$

The intuition: because self-attention is global, by the final layer the CLS position has access to information from every other token, making it a natural aggregate. **It is not initialized with any privileged information** — its meaning emerges entirely from training.

---

## 2. CLS Token in ViT

**ViT** ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) directly transfers this idea to vision. An image $x \in \mathbb{R}^{H \times W \times C}$ is split into $N = HW/P^2$ non-overlapping patches. Each patch is linearly projected into a $d$-dimensional embedding, and a learnable CLS token is prepended:

$$
\mathbf{z}_0 = [\mathbf{e}_{\tt cls};\, \mathbf{e}_1 + \mathbf{p}_1;\, \ldots;\, \mathbf{e}_N + \mathbf{p}_N]
$$

where $\mathbf{e}_{\tt cls} \in \mathbb{R}^d$ is the learnable CLS embedding and $\mathbf{p}_i$ are positional embeddings. After $L$ Transformer blocks, **$\mathbf{z}^{(L)}_0$ — ==the CLS output — is the image representation** fed to the classification head.==

### What does the CLS token learn?
- Early layers: attends broadly and somewhat uniformly across patches
- Later layers: develops **task-specific attention** — focusing on discriminative regions (object parts, textures) relevant to classification
- Its attention maps in the final layer often resemble a rough semantic segmentation of the image

**DeiT** ([Touvron et al., 2021](https://arxiv.org/abs/2012.12877)) adds a **distillation token** alongside CLS, supervised by a teacher CNN — demonstrating that additional global tokens can coexist and specialize. DeiT-III later showed that removing CLS and using only mean pooling over patch tokens can match or exceed CLS-based performance at scale.

---

## 3. CLS Token in Self-Supervised Vision

### [[DINO]] ([Caron et al., 2021](https://arxiv.org/abs/2104.14294))

DINO uses a **self-distillation** objective: a student network is trained to match the output of a momentum teacher. Crucially, DINO operates **on the CLS token alone** — both student and teacher compute $\mathbf{z}^{(L)}_0$, and the cross-entropy loss between their softmax-normalized CLS outputs is minimized.

$$
\mathcal{L}_{\tt DINO} = -\sum_x P_t(\mathbf{z}^{(L),t}_0) \log P_s(\mathbf{z}^{(L),s}_0)
$$

A remarkable emergent property: the **CLS attention maps** develop semantically meaningful segmentation without any supervision signal for localization. The CLS token learns to integrate spatial information in a way that reflects object boundaries, which motivated its use as a feature for k-NN classification and dense prediction.

### [[MAE]] ([He et al., 2022](https://arxiv.org/abs/2111.06377))

MAE takes the opposite approach — it **does not use a CLS token during pre-training**. The encoder only sees unmasked patches; the decoder reconstructs masked pixels. The CLS token is added only at **fine-tuning time**. This raises an important question about what the CLS token actually stores versus what can be recovered by pooling patch tokens — and MAE's competitive performance suggests that CLS is not strictly necessary for learning good representations, though it does help during fine-tuning.

### [[DINO#DINO-v2 Scaling Self-Supervision for Foundation Models|DINO-v2]] ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193))

DINOv2 adds **register tokens** (see §5) to address artifacts in intermediate layers, and scales self-supervised ViT training. The CLS token in DINOv2 remains the primary representation for downstream tasks and achieves state-of-the-art performance on linear probing and k-NN benchmarks.

---

## 4. CLS Token in VLMs

VLMs connect a vision encoder (typically ViT) to a language model. How the visual information is extracted and fed to the LM varies significantly:

| Model | Visual token strategy | Role of CLS |
|-------|----------------------|-------------|
| **CLIP** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) | CLS token output after projection | Primary image embedding for contrastive alignment with text |
| **BLIP** ([Li et al., 2022](https://arxiv.org/abs/2201.12086)) | CLS + patch tokens via cross-attention | CLS used for image-text matching head |
| **BLIP-2** ([Li et al., 2023](https://arxiv.org/abs/2301.12597)) | Q-Former queries all patch tokens | CLS is bypassed; Q-Former learns to extract relevant content |
| **LLaVA** ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)) | All patch tokens (no CLS) | CLS discarded; patch grid preserves spatial detail |
| **InstructBLIP** ([Dai et al., 2023](https://arxiv.org/abs/2305.06500)) | Q-Former + instruction conditioning | CLS optional depending on backbone |
| **SigLIP** ([Zhai et al., 2023](https://arxiv.org/abs/2303.15343)) | EOS-position pooling (BERT-style) | Functional equivalent of CLS, but uses EOS |

### The LLaVA Design Choice
LLaVA deliberately **discards the CLS token** and feeds the full $N \times d$ patch embedding grid to the LLM (after a linear projection layer). The rationale: CLS compresses spatial detail into a single vector, losing fine-grained spatial information that the LLM needs for dense visual reasoning. Passing all patch tokens preserves this at the cost of a longer context length.

### The Q-Former Alternative
BLIP-2 replaces CLS-based compression with a learned cross-attention module (Q-Former) that uses $K$ learnable query vectors to extract task-relevant summaries of the patch tokens. This is functionally a generalization of the CLS token: instead of one global summary vector, you get $K$ specialized summary vectors.

---

## 5. Attention Sink Phenomenon

### Discovery in LLMs

**StreamingLLM** ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)) identified the **attention sink**: in autoregressive LLMs, the initial tokens (e.g., `<BOS>`) accumulate **disproportionately large attention scores** across all heads and layers — even when they are semantically irrelevant to the query.

**Why does this happen?** Softmax attention requires scores to sum to 1. When all tokens are semantically irrelevant to a query, the model needs somewhere to "dump" excess probability mass without corrupting other representations. Initial tokens provide a structurally safe position: attending heavily to BOS does not corrupt any semantically meaningful token. Through training, the model learns to use initial tokens as attention **garbage collectors**.

$$
\alpha_{ij} = \text{softmax}\!\left(\frac{\mathbf{q}_i \mathbf{K}^\top}{\sqrt{d_k}}\right)_j, \quad \sum_j \alpha_{ij} = 1
$$

If no token is informative, one must still receive the probability mass — the attention sink is the learned solution.

**Consequence for streaming inference**: if you apply a KV cache window and evict the initial tokens, performance collapses — not because those tokens contain useful content, but because their key-value vectors serve as the "garbage bin" for attention. StreamingLLM fixes this by always retaining a few "sink" tokens in the window.

### Mechanistic View

Subsequent work ([Sun et al., 2024](https://arxiv.org/abs/2402.10685)) showed that sink tokens develop **near-zero value vectors** — so even when attended to heavily, they contribute minimally to the output hidden state. The softmax "wasted" mass goes to these near-zero-value positions, which is functionally equivalent to attention dropout.

---

## 6. CLS Token as Attention Sink in Vision

The attention sink mechanism has a direct analog in ViTs: **the CLS token is a structurally privileged attention sink**.

### Why CLS is the natural sink position

- CLS is at position 0 — structurally analogous to BOS in LLMs
- It is learnable and not tied to any image region, so it can develop arbitrary key/value vectors
- Patch tokens that have no useful content to aggregate can attend heavily to CLS without semantic cost

### Empirical Evidence: Artifact Patches

**"Vision Transformers Need Registers"** ([Darcet et al., 2023](https://arxiv.org/abs/2309.16588)) revealed a related but distinct failure mode: certain **background patch tokens** (typically in informative-region-free areas like uniform sky or solid backgrounds) develop **anomalously high L2 norms** in intermediate layers. These "artifact patches" accumulate global information and attract attention from many other patches — i.e., they become **local attention sinks**.

The mechanism: when background patches have no local content to summarize, the model repurposes them as information-gathering nodes. This corrupts the spatial feature maps because these tokens no longer represent local image content.

```
Standard ViT:   [CLS, patch_1, ..., patch_N]
                        ↑
              artifact patches emerge here
              (high-norm, global-info accumulators)

ViT + Registers: [CLS, reg_1, ..., reg_R, patch_1, ..., patch_N]
                       ↑
              registers explicitly absorb
              the global-info accumulation role
```

### Registers as Explicit Attention Sinks

Register tokens provide designated positions for global-information accumulation, eliminating the incentive for patch tokens to develop artifacts. The model freely uses registers as attention sinks; patch tokens are freed to represent only their local content.

**Key insight**: the CLS token alone is insufficient as a global sink at every layer — it is only present at one position, and its attention sink capacity saturates in deeper models or when processing informationally sparse images. Registers provide additional capacity.

### Comparison: CLS Sink vs. BOS Sink

| Property | CLS in ViT | BOS/initial tokens in LLM |
|----------|-----------|--------------------------|
| Position | 0 (prepended) | 0 (start of sequence) |
| Semantic content | None (learnable) | Minimal (`<s>`, `<bos>`) |
| Sink behavior | Absorbs global info across layers | Absorbs excess attention probability |
| Pathology when saturated | Artifact patches emerge as secondary sinks | KV cache corruption when evicted |
| Fix | Register tokens | StreamingLLM sink retention |
| Value vector | Can carry meaningful aggregate repr. | Near-zero (dampens contribution) |

---

## 7. Evolving Role Through Layers

CLS token behavior is not uniform across depth:

| Layer range | CLS behavior |
|-------------|-------------|
| Early (L=1–3) | Broad, diffuse attention; CLS begins aggregating low-level statistics |
| Middle (L=4–8) | Becomes selective; attends to semantically relevant patches; risk of artifacts in adjacent patch tokens |
| Late (L=9–12) | Highly task-specific; attention maps resemble semantic segmentation; serves as the final summary |

==DINO-style training amplifies the semantic quality of CLS across all layers; supervised training tends to produce more diffuse intermediate CLS representations.==

---

## 8. Literature Summary

| Paper | Year | Contribution |
|-------|------|-------------|
| [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al.) | 2019 | Introduced CLS token for sequence classification |
| [ViT](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al.) | 2021 | Ported CLS to vision; CLS = image representation |
| [DeiT](https://arxiv.org/abs/2012.12877) (Touvron et al.) | 2021 | Added distillation token; CLS + extra global tokens |
| [DINO](https://arxiv.org/abs/2104.14294) (Caron et al.) | 2021 | CLS-only self-supervised loss; emergent segmentation in CLS attn |
| [MAE](https://arxiv.org/abs/2111.06377) (He et al.) | 2022 | Pre-trains without CLS; adds CLS only at fine-tuning |
| [CLIP](https://arxiv.org/abs/2103.00020) (Radford et al.) | 2021 | CLS output as the image embedding for contrastive VL alignment |
| [BLIP-2](https://arxiv.org/abs/2301.12597) (Li et al.) | 2023 | Q-Former replaces single CLS with K learned query vectors |
| [LLaVA](https://arxiv.org/abs/2304.08485) (Liu et al.) | 2023 | Discards CLS; passes all patch tokens to LLM |
| [DINOv2](https://arxiv.org/abs/2304.07193) (Oquab et al.) | 2023 | Scaled DINO with registers; CLS remains primary repr. |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) (Xiao et al.) | 2023 | Identifies attention sink in LLMs; initial tokens as probability dumping ground |
| [Registers](https://arxiv.org/abs/2309.16588) (Darcet et al.) | 2023 | Artifact patches as secondary attention sinks in ViT; registers as fix |
| [Sun et al.](https://arxiv.org/abs/2402.10685) | 2024 | Mechanistic analysis: sink tokens have near-zero value vectors |

---

## 9. Open Questions

- **Is a single CLS token optimal?** MAE and LLaVA suggest patch pooling can match or exceed it. Q-Former generalizes to $K$ summary tokens.
- **Should VLMs retain CLS?** For dense tasks (grounding, segmentation), no — spatial patch features are needed. For global tasks (image classification, retrieval), yes.
- **Can attention sinks be designed away?** Softmax always requires unit-sum attention; some form of sink behavior is inevitable without changing the normalization (e.g., sigmoid attention as in SigLIP or FlashAttention variants with temperature scaling).
- **Do registers fully eliminate the CLS-as-sink problem?** Registers address artifact patches, but CLS may still behave as an attention sink in the first few transformer blocks. Their interaction at scale is not fully characterized.
