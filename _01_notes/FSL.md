### [[dl_moc]]  
# **Few-Shot Learning**

**Tags**:: #learning
**Links**:: [[Contrastive Learning]] · [[CLIP]] · [[DINO]] · [[Benchmark Datasets]]


**Few-Shot Learning (FSL)** is the problem of learning to generalize to new classes or tasks from very few labeled examples — typically $K$ examples per class (K-shot). It sits at the intersection of transfer learning, meta-learning, and metric learning.

## Definitions
---

| Aspect | Zero-Shot (ZSL) | One-Shot (OSL) | Few-Shot (FSL) |
|--------|----------------|----------------|----------------|
| **Examples per new class** | 0 | 1 | Typically 1–20 |
| **Approach** | Semantic descriptions / attributes | Similarity metrics | Meta-learning |
| **Key challenge** | Attribute quality | Single-example variance | Overfitting |
| **Example methodology** | [[CLIP]] zero-shot | Siamese networks | MAML, Prototypical Nets |

## Episode-Based Training (N-way K-shot)
---
FSL models are trained and evaluated using **episodes** that simulate the few-shot regime:

- **Support set** $S$: $N$ classes × $K$ examples per class ($N$-way $K$-shot)
- **Query set** $Q$: unlabeled examples from the same $N$ classes to classify

The model must use $S$ to classify $Q$ — mimicking test-time deployment. Training uses many sampled episodes so the model learns *how to learn from few examples*, not just how to classify specific classes.

## Metric-Based Methods
---
These methods learn an embedding space where support and query examples from the same class are close.

### Prototypical Networks ([Snell et al., 2017](https://arxiv.org/abs/1703.05175))
Compute a **prototype** for each class by averaging its support embeddings, then classify queries by nearest prototype:
$$
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\theta(x_i)
$$
$$
p(y = k \mid x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))}
$$
where $d$ is Euclidean distance (or cosine). Simple, fast, and surprisingly strong. Prototypical networks implicitly learn a metric space where intra-class variance is small.

### Matching Networks ([Vinyals et al., 2016](https://arxiv.org/abs/1606.04080))
Classify a query using a weighted sum over support examples:
$$
\hat y = \sum_{i=1}^{|S|} a(x, x_i) y_i, \quad a(x, x_i) = \frac{\exp(\text{cos}(f(x), g(x_i)))}{\sum_j \exp(\text{cos}(f(x), g(x_j)))}
$$
The attention-weighted nearest-neighbor is computed end-to-end. Both embedders $f$ and $g$ can be conditioned on the support set via an LSTM (full context embeddings).

## Optimization-Based Methods
---
These methods learn an initialization or update rule that can rapidly adapt to new tasks.

### MAML (Model-Agnostic Meta-Learning, [Finn et al., 2017](https://arxiv.org/abs/1703.03400))
**Goal**: find an initial parameter vector $\theta$ such that a few gradient steps on any new task's loss leads to good performance.

**Algorithm**:
1. Sample a batch of tasks $\{\mathcal{T}_i\}$
2. For each task $\mathcal{T}_i$:
   a. Compute adapted parameters via 1 (or few) gradient step(s) on support set:
      $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^S(\theta)$$
   b. Evaluate adapted parameters on query set: $\mathcal{L}_{\mathcal{T}_i}^Q(\theta_i')$
3. **Meta-update**: update $\theta$ to minimize expected query loss across all tasks:
   $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}^Q(\theta_i')$$

**Key insight**: the meta-gradient $\nabla_\theta \mathcal{L}^Q(\theta')$ involves second-order derivatives (gradient through the inner update). MAML is **model-agnostic** — it works with any architecture trained with gradient descent.

**FOMAML**: first-order approximation (ignore second-order terms) — nearly as good in practice but cheaper.

**Reptile** (OpenAI): simpler meta-algorithm that just moves $\theta$ toward $\theta_i'$ without second-order gradients.

## Large Model Era: In-Context Few-Shot
---
Modern large language/vision models (GPT-3, [[CLIP]], [[VLM|VLMs]]) enable **in-context few-shot learning** without any gradient updates:

- Provide examples directly in the context window: `[example 1, answer 1], [example 2, answer 2], [query]`
- The model predicts the answer from context alone
- [[CLIP]] zero-shot: form text prompts for each class and classify by cosine similarity to image embedding

This emergent capability in large models has largely superseded classical FSL for many tasks, but FSL methods remain important for:
- Low-compute deployment (small models that must adapt quickly)
- Robotics and embodied AI (new objects/tasks with few demonstrations)
- Medical imaging (rare conditions with few labeled cases)

## Benchmark Datasets
---

| Dataset | Domain | Classes | Notes |
|---------|--------|---------|-------|
| Omniglot | Handwritten characters | 1,623 | 20 examples per class |
| miniImageNet | General images | 100 | 600 images per class |
| tieredImageNet | General images (hierarchical) | 608 | ImageNet subset |
| Meta-Dataset | Multi-domain | Variable | Mixed-domain evaluation |
| FC100 | CIFAR-100 subset | 100 | Harder splits |

Evaluation standard: 5-way 1-shot and 5-way 5-shot accuracy, averaged over 600 episodes.
