### [[dl_moc]]  
# **Self-Distillation with No Labels**

**Tags**::  #learning
**Links**::

## Introduction
---
DINO (self-**DI**stillation with **NO** labels) is a groundbreaking self-supervised learning method for computer vision which is prompted in the paper [@caron2021] -- "Emerging Properties in Self-Supervised Vision Transformers" by Caron et al., and touch upon its influential successor, DINOv2. Note that it is not a [[Contrastive Learning|Contrastive Learning]] method.

**Self-Supervised Learning (SSL)** is a paradigm in machine learning where a model learns representations from unlabeled data. Instead of relying on human-provided labels (like "cat" or "dog"), the model generates its own supervision from the data itself. A common technique is to create a "pretext task," such as predicting a missing part of an image, which forces the model to learn meaningful features about the world.

DINO is a specific SSL method that uses a form of knowledge distillation, but without needing any pre-trained, labeled teacher network.


## Framework
---
The core idea of DINO is to train a **student network** ($g_{\theta_s}$) to match the output of a **teacher network** ($g_{\theta_t}$). The key is that both networks have the same architecture, but the teacher's weights are an exponential moving average (EMA) of the student's weights.
![[DINO_01.png|350]]
### Student-Teacher Architecture
+ **Teacher Network ($g_{\theta_t}$):** 
	Vit-Large, it is already trained, strong and slow.
	This network is not trained directly. Its weights ($\theta_t$) are updated using an EMA of the student's weights ($\theta_s$). This provides a more stable target for the student to learn from. 
+ **Student Network ($g_{\theta_s}$):** 
	Vit-Base, it doesn't have as many parameters as the teacher, not as strong but it is faster
	This network is trained with standard back-propagation.

### Multi-Crop Training 
From a single image, multiple crops (views) are generated. These include two larger "global" crops and several smaller "local" crops.

**Global Views**: crops that are more than 50% of the image
**Local Views**: crops that are less than 50% of the image
**Local-to-Global Correspondence**：student uses both local and global views and teacher uses only global views, which encourages the model to learn features that are consistent across different scales and views of the same object.

**Loss Function:**
The model aims to minimize the cross-entropy loss between the softmax-normalized outputs of the student and teacher for the same image. To prevent the model from **collapsing** to a uniform distribution, the author sharpens the teacher's output by using a **temperature parameter** and to prevent being dominated by a single dimension by adding a **center operation**.

* The probability distribution for a view $x$ is $P_s(x) = \text{softmax}(g_{\theta_s}(x) / \tau_s)$ for the student and $P_t(x) = \text{softmax}(g_{\theta_t}(x) / \tau_t)$ for the teacher.
* The loss function is:$$\mathcal{L} = - \sum_{x \in \{x_{global}^1, x_{global}^2\}} \sum_{x' \in \mathcal{V}, x' \neq x} P_t(x) \log P_s(x')$$where $\mathcal{V}$ is the set of all crops from the image, $\tau_s$ and $\tau_t$ are the temperature parameters for the student and teacher, respectively.

**Update Rules:**
The update rule for the **teacher** is: $\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s$, where $\lambda$ is a momentum parameter that is slowly increased during training.


## Emergent Properties in Vision Transformers
---
The most remarkable finding of the DINO paper was the set of properties that emerged in the [[ViT|Vision Transformer]] (ViT) architecture when trained with this method, without any explicit supervision for these tasks.

* **High-Quality Semantic Segmentation:** The **self-attention maps from the final layer** of the ViT model naturally segment objects from the background. By visualizing the attention of the `[CLS]` token, one can produce object masks of surprising quality.
![[DINO_02.png|350]]
* **Feature Re-use:** DINO-trained ViTs learn features that are highly effective for downstream tasks like image classification, even when using a simple k-NN classifier on the frozen features.
![[DINO_03.png]]


## DINOv2: Scaling Self-Supervision for Foundation Models
---
The success of DINO leads to DINOv2 [@oquab2024], which scales up the core ideas to create a powerful vision "foundation model", achieving state-of-the-art performance across a wide range of tasks without fine-tuning.

The goal of DINOv2 was to overcome the limitations of existing pre-trained models, which often required fine-tuning on large, labeled datasets to perform well. The hypothesis was that by scaling up the pre-training data and refining the training algorithm, it would be possible to create a single, general-purpose vision backbone that excels at many tasks with only a simple linear head.

* **Massive, Curated Dataset:** DINOv2 was trained on a new, curated dataset of 142 million images, carefully selected from a larger pool to remove low-quality or irrelevant images.
* **Architectural Improvements:** It incorporated techniques like SwiGLU activations and Nested Dropout for better performance and stability during training at scale.
* **State-of-the-Art Performance:** DINOv2 established itself as a leading general-purpose vision backbone, achieving excellent results on tasks like depth estimation, semantic segmentation, and image classification.


### Key Innovations

DINOv2 introduced several crucial improvements over the original DINO.
#### A. Data Curation at Scale (The LVD-142M Dataset)

Instead of using an existing, off-the-shelf dataset, the DINOv2 authors built their own: **LVD-142M**. This involved a massive data-sourcing and curation pipeline:

1.  **Sourcing:** An initial, un-curated dataset of 1.2 billion unique images was sourced from the web.
2.  **Deduplication & Similarity Search:** Self-supervised models (like DINO) were used to compute image embeddings. These embeddings helped identify and remove near-duplicate images.
3.  **Curation:** The remaining images were clustered. Clusters that were too close to ImageNet or other standard benchmarks were removed to ensure the pre-training data was sufficiently different from evaluation datasets. This process resulted in a final, high-quality dataset of **142 million images**.

#### B. Algorithmic and Architectural Improvements

To make training stable and effective at this new scale, several changes were made:

* **Image-Text Foundation Model Filtering:** [[CLIP]] was used to filter the web-scale dataset to ensure a good alignment between images and their alt-text, improving the quality of the image source.
* **Combined Loss Function:** DINOv2 incorporates the **iBOT loss**, which adds a masked image modeling objective. This forces the model to learn not just global features (from the `[CLS]` token) but also dense, patch-level features.
* **KoLeo Regularizer:** A regularization term was added to the loss to encourage batch-wise feature diversity, preventing the model from collapsing and ensuring a broader representation space.
* **Architectural Tweaks:** The ViT architecture was updated with modern components like SwiGLU activations and Nested Dropout for improved performance and stability.

#### C. Efficient Implementation

Training on over 100 million images is computationally expensive. The team made significant engineering efforts to make it feasible:

* **PyTorch 2.0 Compilation:** The model was compiled with `torch.compile` for significant speedups.
* **Fully Sharded Data Parallel (FSDP):** This technique was used to distribute the model and optimizer states across GPUs, enabling training of the largest ViT models.
* **Mixed-Precision Training:** Using `bfloat16` precision reduced memory usage and further accelerated training.




## Remark
---
**Dense Prediction Task**: In computer vision, unlike image classification, tasks such as **semantic segmentation**, **instance segmentation**, etc. are considered as dense prediction task because label for each pixel is predicted.