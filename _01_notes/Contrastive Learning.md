### [[dl_moc]]  
# **Contrastive Learning**

**Tags**:: #learning
**Links**::

## SimCLR
---
**Simple Framework for Contrastive Learning of Visual Representations** [1](https://arxiv.org/abs/2002.05709): ==a SSL method that trains a neural network to recognize visual features by comparing different views of the same image without supervision==.
+ **Data Augmentation:** transforms a single image into two distinct but correlated views (e.g., random crop, color distortion, and Gaussian blur). These two views act as a "positive pair."
+ **Base Encoder f:** a standard neural network, often a ResNet-50. It extracts a feature vector h from both augmented images.
+ **Projection Head g:** a small MLP that maps the feature vectors h into a different space, z = g(h), where the contrastive loss is computed.
+ **Contrastive Loss (NT-Xent):** the normalized temperature-scaled cross-entropy loss. It mathematically calculates the similarity of a positive pair against all other "negative" samples in the training batch.
	The loss function of a positive pair of examples $(i,j)$ is denoted as:
$$
\mathcal{L}_{i,j} = -\log \frac{\exp (sim(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{\{k\neq i\}} \exp (sim(z_i, z_k)/\tau)}
$$
	$sim(u,v)=\frac{u^T v}{\|u\|\|v\|}$ represents the cosine similarity
	$\tau$ is temperature hyper-parameter that controls the hardness of negatives.

#### Training Procedure
1. Sample a batch of unlabeled images.
2. Apply two random augmentations to each image to create positive pairs.
3. Pass both views through the encoder and projection head.
4. Calculate contrastive loss for each positive pair.
5. [Backpropagate](https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/) and update the encoder and projection head weights.
6. After training, discard the projection head and use the encoder for downstream tasks

### Extensions and Variants
- SimCLR v2: Adds larger models, longer training, and knowledge distillation
- CLIP, DINO, and MAE are newer paradigms built on/around SimCLR ideas.

