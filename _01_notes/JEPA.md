### [[dl_moc]]  
# **JEPA — Joint Embedding Predictive Architecture**

**Tags**:: #vision #learning #architecture
**Links**:: [[ViT]] · [[MAE]] · [[DINO]] · [[Contrastive Learning]] · [[World Models]] · [[VLM]] · [[CLS Token]]

---

**JEPA** (*Joint Embedding Predictive Architecture*) is a self-supervised learning framework in which a model learns by **predicting the representations of masked or future observations in embedding space**, rather than reconstructing raw pixels or matching augmented views. It was proposed by Yann LeCun as part of a broader theoretical program for autonomous AI ([LeCun, 2022](https://openreview.net/forum?id=BZ55e1Xt8zs)) and instantiated concretely in I-JEPA and V-JEPA.

---

## 1. The Core Problem with Prior Paradigms

### Generative models predict too much

**MAE**-style methods predict raw pixel values. Pixels contain enormous redundancy — exact texture, lighting, and color that carry no semantic information. A model that reconstructs pixels must waste capacity representing these irrelevant details. This leads to a mismatch: the pre-training task rewards pixel-level accuracy, but downstream tasks care about semantic content.

### Contrastive methods define invariances via augmentations

**DINO** / SimCLR learn representations that are invariant to hand-crafted augmentations (crops, color jitter, blur). The augmentation pipeline acts as a hand-coded prior about what information to discard. This works well but requires task-specific augmentation design and does not generalize cleanly to modalities beyond images.

### JEPA's alternative

Instead of predicting pixels or matching views, JEPA **predicts the representation of a masked region given the representation of its context** — both computed by encoder networks. The prediction target lives in a learned embedding space, not in the raw observation space.

**Key property**: the encoder can collapse irrelevant variation (textures, exact colors) before the prediction target is defined. The predictor never has to account for these — it only predicts the semantically meaningful part of the representation. This is the fundamental advantage over pixel-space reconstruction.

---

## 2. General JEPA Architecture

The canonical JEPA architecture has three components:

```
Image x
  ├─ Context region x_c ──► Context Encoder (f_θ) ──► h_c  ─┐
  │                                                           ├─► Predictor (g_φ) ──► ŝ_t
  └─ Target region x_t ──► Target Encoder (f_ξ)  ──► s_t  ──┘
```

1. **Context encoder** $f_\theta$: encodes the visible (unmasked) region $x_c$ into a context representation $h_c$
2. **Target encoder** $f_\xi$: encodes the masked target region $x_t$ into a target representation $s_t$; this encoder is an **EMA of the context encoder** (not trained by backprop directly)
3. **Predictor** $g_\phi$: a lightweight network that maps $h_c$ (+ positional information about the target location) to a prediction $\hat{s}_t \approx s_t$

**Loss**: minimize the L2 distance between the predicted and actual target representations:
$$
\mathcal{L}_{\text{JEPA}} = \left\| \hat{s}_t - \text{sg}(s_t) \right\|_2^2
$$
where $\text{sg}(\cdot)$ is stop-gradient — the target is treated as a fixed supervision signal. The target encoder's parameters $\xi$ are updated by EMA: $\xi \leftarrow \tau\xi + (1-\tau)\theta$.

### Why EMA target encoder?

Without the EMA target encoder, the model can satisfy the loss by setting both encoders to constant functions (representation collapse). The EMA encoder provides stable targets that evolve slowly, preventing collapse while not requiring contrastive negatives or explicit collapse prevention (unlike DINO's centering/sharpening).

### Why is the predictor important?

The predictor must be expressive enough to infer masked content from context but not so powerful that it can memorize the target encoder. In practice, a small Transformer predictor (4–8 blocks) works well. Crucially, the predictor receives **positional information** (the spatial location of the target block), so it can reason about *where* the masked region is, not just learn a generic statistic.

---

## 3. I-JEPA: Image JEPA

**I-JEPA** ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)) applies JEPA to still images.

### Multi-block masking strategy

Unlike MAE's random uniform masking, I-JEPA uses a **multi-block masking** strategy specifically designed to force semantic reasoning:

- **Target blocks**: 4 large contiguous blocks (aspect ratio sampled uniformly, scale sampled from $[0.15, 0.2]$ of the image area) — larger, semantically coherent regions
- **Context**: everything outside the target blocks (no random scattered patches)

The rationale: a model predicting large contiguous regions cannot rely on local texture extrapolation from neighboring pixels (as a pixel-prediction model can). It must learn about global object structure.

### Architecture details

- **Context encoder**: ViT-H/14, processes the full image minus the masked target blocks. No mask tokens — only visible patches are encoded (asymmetric like MAE).
- **Predictor**: narrow Transformer (small depth/width), receives encoded context + sinusoidal position embeddings for the target locations.
- **Target encoder**: EMA of context encoder, $\tau = 0.996$; processes all patches including the target regions.

### Results

| Task | Metric | Value |
|------|--------|-------|
| ImageNet linear probe | Top-1 acc | 79.3% (ViT-H/14) |
| ImageNet fine-tuning | Top-1 acc | 87.1% (ViT-H/16@448) |
| Low-shot (1%) | Top-1 acc | 73.3% (ViT-H/14) |

I-JEPA achieves strong linear probe performance **without any hand-crafted data augmentation** during pre-training — a notable contrast to contrastive methods that require heavy augmentation pipelines.

### I-JEPA vs. MAE vs. DINO

| Property | MAE | I-JEPA | DINO |
|----------|-----|--------|------|
| Prediction target | Raw pixels | EMA encoder repr. | — (matching, not prediction) |
| Mask strategy | Random uniform (75%) | Multi-block (large regions) | Multi-crop views |
| Augmentations required | Minimal | **None** | Yes (color jitter, blur, etc.) |
| Pixel-level details learned | Yes | No (only semantic) | No |
| Linear probe quality | Good | **Strong** | Excellent |
| Fine-tuning quality | Excellent | Strong | Good |
| Collapse prevention | N/A (pixel target) | EMA + predictor bottleneck | Centering + sharpening |
| CLS token | Added at fine-tune | Used throughout | Core to loss |

---

## 4. V-JEPA: Video JEPA

**V-JEPA** ([Bardes et al., 2024](https://arxiv.org/abs/2404.08471)) extends the framework to video, predicting masked **spatiotemporal** regions.

### Key changes from I-JEPA

- Input: $T$ frames of video, patchified into a spatiotemporal tube grid
- Masking: large contiguous 3D tubes (spatial blocks × temporal span) are masked
- Predictor receives spatiotemporal positional embeddings

V-JEPA predicts both spatial content *and temporal evolution* of the masked region — forcing the model to understand object motion and dynamics rather than just static appearance.

### Why video JEPA is qualitatively different from image JEPA

In images, predicting a masked region requires understanding spatial structure and object appearance. In video, predicting a masked temporal tube requires understanding **dynamics**: how objects move, deform, and interact over time. This is fundamentally a form of physical world modeling.

V-JEPA achieves strong performance on action recognition and video understanding tasks without labeling, and transfer-learns physical dynamics that image models cannot capture.

---

## 5. JEPA as a World Model

LeCun's 2022 paper ("A Path Towards Autonomous Machine Intelligence") frames JEPA as an **architecture for building world models** — models that predict the future state of the world in representation space, used to plan and reason.

In the JEPA world model view:
- The **context encoder** encodes the current state of the world into a latent representation
- The **predictor** models the transition dynamics: given the current state (and optionally an action), predict the future state representation
- The **target encoder** provides training signal for the future state

This is distinct from generative world models (Dreamer, RSSM) which predict raw observations (pixels). Generative world models must model *everything* visible — textures, lighting, irrelevant background. A JEPA world model predicts only the semantically relevant aspects of the future, ignoring unpredictable variation.

**The fundamental claim**: for autonomous AI, we do not need to predict every pixel of the future — we need to predict which high-level events and states will occur. JEPA's embedding-space prediction is structurally aligned with this goal.

See [[World Models]] for Dreamer/MuZero comparison.

---

## 6. Relationship to data2vec

**data2vec** ([Baevski et al., 2022](https://arxiv.org/abs/2202.03555)) is the closest precursor: it predicts averaged representations of masked tokens computed from a momentum encoder, across audio, vision, and text.

| | data2vec | I-JEPA |
|--|---------|--------|
| Target | Average of top-K layer outputs from EMA encoder | Final-layer output of EMA encoder |
| Masking | Random scattered tokens | Multi-block (semantic) |
| Positional info to predictor | No | Yes (explicit target location) |
| Architecture | Modality-specific | ViT-centric |

I-JEPA improves on data2vec primarily through the multi-block masking strategy and the explicit target location conditioning, which force semantic rather than local reasoning.

---

## 7. JEPA and VLMs

JEPA-trained encoders produce strong visual representations that transfer well to multimodal settings:

- **I-JEPA features** have been used as the visual backbone in VLM pipelines, replacing or augmenting CLIP features
- The absence of augmentation-defined invariances means JEPA features retain more spatial detail than CLIP's contrastively-trained features
- **V-JEPA features** are particularly relevant for video-understanding VLMs (e.g., models that reason about video content with language)

A notable difference from CLIP: CLIP's visual encoder is trained with language supervision, so its features align well with text but may over-emphasize text-describable properties. JEPA features are purely vision-trained — they encode structural and dynamic visual properties that language doesn't easily describe.

---

## 8. Limitations and Open Questions

- **No explicit negative control**: JEPA relies on the EMA/predictor bottleneck to prevent collapse; this can fail if the predictor is too powerful or the EMA decay is too slow
- **Masking design**: multi-block masking is manually designed to encourage semantic reasoning — an automatic curriculum for mask difficulty would be preferable
- **Action conditioning**: current I/V-JEPA models don't condition on actions; extending to action-conditioned JEPA (as in LeCun's full vision) is an open problem
- **Language grounding**: JEPA learns visual representations without language alignment; combining with CLIP-style text supervision could bridge both strengths

---

## 9. Literature Summary

| Paper | Year | Contribution |
|-------|------|-------------|
| [LeCun — Path to AMI](https://openreview.net/forum?id=BZ55e1Xt8zs) | 2022 | Theoretical framework for JEPA-based world models; motivation for repr-space prediction |
| [data2vec](https://arxiv.org/abs/2202.03555) (Baevski et al.) | 2022 | Predicts averaged EMA representations; direct precursor to I-JEPA |
| [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al.) | 2023 | JEPA applied to images; multi-block masking; EMA target encoder |
| [V-JEPA](https://arxiv.org/abs/2404.08471) (Bardes et al.) | 2024 | JEPA applied to video; spatiotemporal tube masking; physical dynamics |
