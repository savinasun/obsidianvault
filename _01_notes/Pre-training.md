### [[dl_moc]]  
# **Pre-training**

**Tags**:: #learning #dl_theo
**Links**:: [[MAE]] · [[DINO]] · [[Contrastive Learning]] · [[CLIP]] · [[LLMs]] · [[RLHF]] · [[Word Embeddings]] · [[Scaling Laws]]


**Pre-training** is the first phase of a two-phase training strategy: train a model on a large, cheap-to-obtain dataset to learn general-purpose representations, then **fine-tune** on a smaller, task-specific dataset. The core motivation is that many tasks share underlying structure — language syntax, visual textures, physical dynamics — that can be learned once and reused across many downstream applications.

## Why Pre-train?
---
Three compounding reasons:

**Data efficiency** — task-specific labeled data is expensive and scarce. Unlabeled data (internet text, images, video) is abundant. Pre-training extracts signal from unlabeled data, so fine-tuning requires far fewer labels.

**Generalization** — pre-trained representations encode broad world knowledge that acts as a regularizer during fine-tuning. Starting from a rich initialization reduces overfitting on small datasets.

**Compute amortization** — one expensive pre-training run produces a model that can be fine-tuned cheaply for many tasks. This is how a single GPT-4 pre-training run becomes usable for coding, reasoning, writing, and tool use.

## The Pre-training Objective
---
When labels are unavailable, a **self-supervised proxy task** is designed so that solving it forces the model to learn useful internal representations. The objective is not the end goal — it is a vehicle for acquiring features.

| Domain | Objective | Forces the model to learn |
|--------|-----------|--------------------------|
| Vision — generative | Reconstruct masked patches ([[MAE]]) | Global image semantics |
| Vision — contrastive | Invariance to augmentations ([[DINO]], SimCLR) | View-invariant features |
| Language — masked | Predict masked tokens (BERT) | Bidirectional context |
| Language — causal | Predict next token (GPT, [[LLMs]]) | World knowledge, language structure |
| Multimodal | Match image–text pairs ([[CLIP]]) | Cross-modal alignment |

The objective is carefully chosen so the **only viable strategy is to build the representations we care about**. MAE cannot reconstruct a masked face without a model of what faces look like. GPT cannot predict the next token in a medical text without encoding medical knowledge.

## Pre-training → Fine-tuning Pipeline
---
```
Large unlabeled corpus
        ↓
  Pre-training (self-supervised proxy task)
        ↓
  Pre-trained model (general-purpose representations)
        ↓  (cheap, repeated for each task)
  Fine-tuning (task-specific labeled data)
        ↓
  Task-specific model
```

**Fine-tuning strategies**:
- **Full fine-tuning**: update all parameters — best performance, most compute
- **Linear probe**: freeze the backbone, train only a linear head — fast, tests representation quality
- **LoRA / adapter fine-tuning**: inject small trainable modules, freeze most weights — parameter-efficient
- **Prompt tuning**: learn soft prompts prepended to frozen model input

## Self-supervised vs. Supervised Pre-training
---
Historically, models pre-trained on ImageNet with supervised labels (class labels for 1K categories) were the standard backbone. Self-supervised pre-training has largely replaced this because:
- It scales to arbitrary unlabeled data (not bottlenecked by labeling)
- It learns richer representations not biased toward the 1K ImageNet categories
- Strong self-supervised methods ([[DINO]], [[MAE]]) now outperform supervised pre-training on dense tasks (detection, segmentation)

## Pre-training Scale and the Bitter Lesson
---
Empirically, pre-training quality scales predictably with compute, data, and model size (see [[Scaling Laws]]). This is the foundation of the **bitter lesson** ([Sutton, 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)): general methods that leverage computation eventually outperform hand-crafted inductive biases. Pre-training on massive corpora is the canonical example — a language model trained on enough text acquires emergent capabilities (reasoning, translation, coding) that were never explicitly trained for.

The [[Scaling Laws]] (Chinchilla) quantify this: loss follows a power law in both model parameters $N$ and training tokens $D$:
$$
\mathcal{L}(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \mathcal{L}_\infty
$$
Optimal compute allocation pre-trains on roughly $20\times$ more tokens than the model has parameters.

## Pre-training in LLMs: Knowledge vs. Behavior
---
In modern [[LLMs]], pre-training and fine-tuning play distinct roles:

- **Pre-training** (next-token prediction on trillions of tokens) — teaches **world knowledge**, language structure, reasoning patterns, factual associations
- **Instruction fine-tuning / [[RLHF]]** — steers **behavior**: following instructions, being helpful, refusing harmful requests, adopting a persona

This separation matters: fine-tuning does not efficiently inject new knowledge (the model will hallucinate if the fact wasn't in pre-training). Knowledge comes from pre-training data; behavior comes from fine-tuning signals. This is why retrieval-augmented generation (RAG) is used to inject facts at inference time rather than continually re-training.

## Continual Pre-training
---
A variant where a pre-trained model is continued on domain-specific unlabeled data before task fine-tuning:

```
General pre-training → Domain pre-training → Task fine-tuning
```

Used in: biomedical NLP (PubMedBERT), code (CodeBERT, StarCoder), robotics (RT-2 on robot trajectories). Lets the model adapt vocabulary and token distributions to a new domain without starting from scratch.

## References
---
- [Devlin et al., 2019 — BERT](https://arxiv.org/abs/1810.04805)
- [He et al., 2022 — MAE](https://arxiv.org/abs/2111.06377)
- [Radford et al., 2021 — CLIP](https://arxiv.org/abs/2103.00020)
- [Sutton, 2019 — The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
