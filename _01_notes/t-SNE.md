### [[dl_moc]]  
# **t-Distributed Stochastic Neighbor Embedding**

**Tags**:: #math #learning
**Links**:: [[KL Divergence]] · [[UMAP]] · [[Manifold Hypothesis]]

## Introduction
---
t-SNE is a powerful non-linear **dimensionality reduction technique** which is particularly well-suited for embedding high-dimensional data into a low-dimensional space (typically 2 or 3 dimensions) for visualization. 

It starts from Stochastic Neighbor Embedding (SNE), which is first proposed by Geoffrey Hinton and Sam Roweis in [Hinton & Roweis, 2002](https://proceedings.neurips.cc/paper/2002/file/6150ccc6069bea6b5716254057a9e187-Paper.pdf) in 2002. In a follow-up paper [Maaten & Hinton, 2008](https://jmlr.org/papers/v9/vandermaaten08a.html) by Laurens van der Maaten and Geoffrey Hinton, a couple of improvements were proposed that led to t-Distributed Stochastic Neighbor Embedding (t-SNE). Then finally Barnes-Hut-SNE by Laurens van der Maaten provides the toolbox to scale t-SNE to millions of datapoints.

t-SNE is a probabilistic method for non-linear dimensionality reduction. The core idea of t-SNE is to preserve local neighborhood structures from the high-dimensional space in the low-dimensional map by **modeling similarities as joint probabilities and minimizing the [[KL Divergence|KL divergence]] between the high-dimensional Gaussian distributions $P$ and the low-dimensional Students' t-probability distributions $Q$**, it creates intuitive and often insightful visualizations. The use of the Student's t-distribution for the low-dimensional similarities is crucial for its effectiveness, as it helps to separate distinct clusters and reveal the underlying structure of the data.


## Algorithm
---
1. Initialize points randomly with PCA
2. Compute pairwise distances
3. Convert distances to conditional probabilities
4. Perform binary search to find $\sigma$ that matches the desired perplexity
5. Optimize KL divergence for each point via gradient descent

#### From Euclidian Distances to Probabilities

t-SNE models similarity between data points using **conditional probabilities**. For a set of $N$ high-dimensional data points $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$, the similarity of datapoint $\mathbf{x}_j$ to datapoint $\mathbf{x}_i$ is modeled as the conditional probability, $p_{j|i}$, that $\mathbf{x}_i$ would pick $\mathbf{x}_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $\mathbf{x}_i$.

Mathematically, this conditional probability is given by:
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$
where $\sigma_i$ is the variance of the Gaussian centered on datapoint $\mathbf{x}_i$. The value of $\sigma_i$ is determined for each point individually through a binary search process that depends on a user-defined parameter called *perplexity*. 

#### Perplexity

Perplexity can be interpreted as a smooth measure of **the effective number of neighbors for each point**. For a chosen perplexity, $\text{Perp}$, the **binary search** finds the $\sigma_i$ that produces a probability distribution $P_i$ with that perplexity:
$$2^{H(P_i)} = \text{Perp}$$
where $H(P_i)$ is the Shannon entropy of the distribution $P_i$ measured in bits:
$$H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$$
Generally, the entropy increases if the variance of the distribution increases, we may see the perplexity here in essence is the same as the [[PPL|perplextiy score]] we use to measure the prediction power of a model.

#### Symmetrize SNE

The conditional probabilities $p_{j|i}$ are not symmetric ($p_{j|i} \neq p_{i|j}$). To simplify the optimization process and deal with outliers, t-SNE uses joint probabilities, $p_{ij}$, which are symmetrized versions of the conditional probabilities. This is done by averaging the conditional probabilities:
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
This ensures that $\sum_{i,j} p_{ij} = 1$ and $p_{ij} = p_{ji}$. The matrix $P$ with entries $p_{ij}$ represents the pairwise similarities of the high-dimensional data points.

#### The Crowding Problem in SNE

A key innovation of t-SNE over its predecessor, Stochastic Neighbor Embedding (SNE), is the use of a Student's t-distribution to model similarities in the low-dimensional space. This change was introduced to solve a significant issue known as the **"crowding problem"**.

In the original SNE algorithm, the low-dimensional similarities $q_{ij}$ were modeled using a Gaussian distribution, just like the high-dimensional similarities $p_{ij}$:

$$q_{ij}^{\text{(SNE)}} = \frac{\exp(-\|\mathbf{y}_i - \mathbf{y}_j\|^2)}{\sum_{k \neq l} \exp(-\|\mathbf{y}_k - \mathbf{y}_l\|^2)}$$
*(Note: The variance is typically fixed to $1/2$ for simplicity in the low-dimensional space).*

The crowding problem arises from a "mismatch of dimensionalities." In a high-dimensional space, there is a lot of room, and points can have many neighbors at a moderate distance. When we try to project this structure onto a 2D or 3D map, there isn't enough space to accommodate all the neighbors accurately.

Specifically, the volume of a sphere of radius $r$ scales as $r^d$ with the dimension $d$, so there is far more room at moderate distances in high-dimensional space. This means that points that are moderately distant in a high-dimensional space (e.g., 100-D) must be squeezed into a much smaller volume in the low-dimensional map (e.g., 2-D). The Gaussian distribution used in SNE has a "light" tail, meaning the probability density drops off very quickly with distance. This creates significant attractive forces between points that are only moderately close in the high-D space, forcing them to be placed very close together in the low-D map. This results in a visualization where distinct clusters are not well separated and are often clumped together in a single, crowded ball.

#### Student's t-Distribution in Low-Dimensional Space Solving the Crowding Problem

t-SNE addresses this by replacing the Gaussian distribution in the low-dimensional space with a heavy-tailed distribution: the Student's t-distribution with one degree of freedom.

The similarity metric $q_{ij}$ is redefined as:

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

The key term here is $(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$, which is the probability density of a t-distribution with one degree of freedom (also known as a Cauchy distribution).

#### The "Heavy Tails" Advantage

The t-distribution has "heavier tails" than the Gaussian distribution. This means that as the distance $\|\mathbf{y}_i - \mathbf{y}_j\|$ increases, the similarity score $q_{ij}$ decreases much more slowly than it would under a Gaussian.

This property provides two main benefits:

1.  **Alleviates Crowding:** The heavy tails allow dissimilar points to be placed much further apart in the low-dimensional map. For a pair of points to be considered truly dissimilar (i.e., for $q_{ij}$ to be very small), their distance $\|\mathbf{y}_i - \mathbf{y}_j\|$ must be very large. This means that moderately dissimilar points (small but not-quite-zero $p_{ij}$) are pushed far apart by repulsive forces, creating more space and clearer separation between clusters. It effectively translates moderate distances in the high-dimensional space to much larger distances in the low-dimensional map, which solves the crowding problem.

2.  **Improves Cluster Separation:** As a direct consequence, the global structure of the data is better represented. Groups of points that form distinct clusters in the high-dimensional space can be separated by large empty spaces in the t-SNE map, making the visualization far more intuitive and useful.

In summary, the introduction of the Student's t-distribution is not an arbitrary choice. It is a principled solution to the geometric constraints of dimensionality reduction, allowing t-SNE to create clean, well-separated visualizations of high-dimensional data by exaggerating the distance between dissimilar clusters.

#### The Cost Function: [[Entropy#Kullback–Leibler Divergence|Kullback-Leibler Divergence]]

The goal of t-SNE is to find a low-dimensional embedding $\mathcal{Y}$ that minimizes the mismatch between the joint probability distributions $P$ (from the high-dimensional space) and $Q$ (from the low-dimensional space). This mismatch is measured by the Kullback-Leibler (KL) divergence of $Q$ from $P$:
$$C = \text{KL}(P || Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
Minimizing this cost function will arrange the points $\mathbf{y}_i$ in the low-dimensional space to best reflect the similarities $p_{ij}$ of their high-dimensional counterparts.

#### Optimization via Gradient Descent

The KL divergence is minimized using gradient descent. The gradient of the cost function $C$ with respect to the embedding point $\mathbf{y}_i$ is given by:
$$\frac{\partial C}{\partial \mathbf{y}_i} = 4 \sum_{j} (p_{ij} - q_{ij}) q_{ij} Z (\mathbf{y}_i - \mathbf{y}_j)$$
where $Z = \sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}$ is the normalization constant from the denominator of $q_{ij}$.

The term $(p_{ij} - q_{ij})$ can be interpreted as the error between the similarities in the high and low-dimensional spaces. The gradient can be seen as a sum of spring-like forces:
* If $p_{ij}$ is large and $q_{ij}$ is small, there is a strong attractive force between points $\mathbf{y}_i$ and $\mathbf{y}_j$.
* If $p_{ij}$ is small and $q_{ij}$ is large, there is a strong repulsive force pushing $\mathbf{y}_i$ and $\mathbf{y}_j$ apart.

The optimization is typically performed using momentum to speed up convergence and avoid poor local minima.

## Goodness-Of-Fit
---
#### Shepard Diagram

A Shepard diagram compares how far apart your data points are before and after you transform them (ie: goodness-of-fit) as a scatter plot. It could be easily used to evaluate data reduction techniques like _principal components analysis_ (PCA), _multidimensional scaling_ (MDS), or t-SNE.


Note that the distances between clusters of t-SNE points are meaningless, this is because the [[KL Divergence|KL-divergence]] cost function used in t-SNE doesn't care about the global structure.