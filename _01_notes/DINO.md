### [[dl_moc]]  
# **Self-Distillation with No Labels**

**Tags**::  
**Links**::

## Introduction
---
DINO (self-**DI**stillation with **NO** labels) is a groundbreaking self-supervised learning method for computer vision which is prompted in the paper [@caron2021] -- "Emerging Properties in Self-Supervised Vision Transformers" by Caron et al., and touch upon its influential successor, DINO-v2. Note that it is not a contrastive learning method.

**Self-Supervised Learning (SSL)** is a paradigm in machine learning where a model learns representations from unlabeled data. Instead of relying on human-provided labels (like "cat" or "dog"), the model generates its own supervision from the data itself. A common technique is to create a "pretext task," such as predicting a missing part of an image, which forces the model to learn meaningful features about the world.

DINO is a specific SSL method that uses a form of knowledge distillation, but without needing any pre-trained, labeled teacher network.


## Framework
---
The core idea of DINO is to train a **student network** ($g_{\theta_s}$) to match the output of a **teacher network** ($g_{\theta_t}$). The key is that both networks have the same architecture, but the teacher's weights are an exponential moving average (EMA) of the student's weights.
### Student-Teacher Architecture
+ **Teacher Network ($g_{\theta_t}$):** 
	Vit-Large, it is already trained, strong and slow.
	This network is not trained directly. Its weights ($\theta_t$) are updated using an EMA of the student's weights ($\theta_s$). This provides a more stable target for the student to learn from. The update rule is: $\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s$, where $\lambda$ is a momentum parameter that is slowly increased during training.
+ **Student Network ($g_{\theta_s}$):** 
	Vit-Base, it doesn't have as many parameters as the teacher, not as strong but it is faster
	This network is trained with standard back-propagation.

### Multi-Crop Augmentation
+ From a single image, multiple crops (views) are generated. These include two larger "global" crops and several smaller "local" crops.
* All crops are passed through the student network.
* Only the global crops are passed through the teacher network.
* This encourages the model to learn features that are consistent across different scales and views of the same object (**"local-to-global"** correspondence).
* **Loss Function:**
    * The model aims to minimize the cross-entropy loss between the softmax-normalized outputs of the student and teacher for the same image. The teacher's output is sharpened using a temperature parameter to prevent the model from collapsing to a uniform distribution.
    * The probability distribution for a view $x$ is $P_s(x) = \text{softmax}(g_{\theta_s}(x) / \tau_s)$ for the student and $P_t(x) = \text{softmax}(g_{\theta_t}(x) / \tau_t)$ for the teacher.
    * The loss function is:$$\mathcal{L} = - \sum_{x \in \{x_{global}^1, x_{global}^2\}} \sum_{x' \in \mathcal{V}, x' \neq x} P_t(x) \log P_s(x')$$where $\mathcal{V}$ is the set of all crops from the image, $\tau_s$ and $\tau_t$ are the temperature parameters for the student and teacher, respectively.


## Emergent Properties in Vision Transformers
---
The most remarkable finding of the DINO paper was the set of properties that emerged in the [[ViT|Vision Transformer]] (ViT) architecture when trained with this method, without any explicit supervision for these tasks.

* **High-Quality Semantic Segmentation:** The self-attention maps from the final layer of the ViT model naturally segment objects from the background. By visualizing the attention of the `[CLS]` token, one can produce object masks of surprising quality.
* **Feature Re-use:** DINO-trained ViTs learn features that are highly effective for downstream tasks like image classification, even when using a simple k-NN classifier on the frozen features.


## DINO-v2
---
The success of DINO leads to DINO-v2, which scales up the core ideas to create a powerful vision "foundation model."

* **Massive, Curated Dataset:** DINO-v2 was trained on a new, curated dataset of 142 million images, carefully selected from a larger pool to remove low-quality or irrelevant images.
* **Architectural Improvements:** It incorporated techniques like SwiGLU activations and Nested Dropout for better performance and stability during training at scale.
* **State-of-the-Art Performance:** DINO-v2 established itself as a leading general-purpose vision backbone, achieving excellent results on tasks like depth estimation, semantic segmentation, and image classification.