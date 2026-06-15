### [[dl_moc]]  
# **Uniform Manifold Approximation and Projection **

**Tags**:: #math #learning
**Links**:: [[t-SNE]] · [[Manifold Hypothesis]] · [[Contrastive Learning]]

## Introduction
---
Ambiant Space

Manifold: the shape of the data in the ambiant space

Homology
Simplicial Complex

Filtration

Persistant Homology



Uniform Manifold Approximation and Projection (UMAP) is a modern dimensionality reduction algorithm developed by Leland McInnes, John Healy, and James Melville in [@mcinnes2020] in 2018. Like [[t-SNE]], it is highly effective for visualizing high-dimensional data in low-dimensional spaces. However, UMAP is often significantly faster and is argued to provide a better balance between preserving local and global data structure. Its theoretical underpinnings are based on Riemannian geometry and algebraic topology.

#### Theoretical Foundation: Learning the Manifold

The core assumption of UMAP is that the observed data points $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$ are sampled from a (potentially non-linear) **[[Manifold Hypothesis|manifold]]** that is embedded in the high-dimensional space (**the ambient space**). The algorithm aims to **learn the underlying topology of this manifold** and then find the closest possible low-dimensional representation of that topology.

The key steps are:
1.  Construct a weighted graph representing the topological structure of the data in the high-dimensional space.
2.  Find a low-dimensional embedding of the points that has the most similar topological structure.

#### 1. Constructing the High-Dimensional Fuzzy Topological Representation

First, UMAP constructs a "fuzzy simplicial set," (a **simplex** is a generalization of the notion of a triangle or tetrahedron to arbitrary dimensions; and a **simplicial complex** is a structured set composed of $k$-simplices of any $k$) which can be thought of as a weighted graph where edge weights represent the likelihood that two points are connected on the manifold.

**A. Finding Local Distances and Neighbors**

For each data point $\mathbf{x}_i$, UMAP identifies its $k$ nearest neighbors. It then learns a local, point-specific distance metric. This is achieved by assuming the manifold is *locally uniform*. Under this assumption, for each point $\mathbf{x}_i$, UMAP finds a radius $\sigma_i$ such that the following equation holds:

$$\sum_{j=1}^{k} \exp\left(-\frac{\max(0, d(\mathbf{x}_i, \mathbf{x}_{i_j}) - \rho_i)}{\sigma_i}\right) = \log_2(k)$$

where:
* $\{\mathbf{x}_{i_1}, \dots, \mathbf{x}_{i_k}\}$ are the $k$ nearest neighbors of $\mathbf{x}_i$.
* $d(\mathbf{x}_i, \mathbf{x}_{i_j})$ is the distance between $\mathbf{x}_i$ and its $j$-th neighbor.
* $\rho_i$ is the distance from $\mathbf{x}_i$ to its nearest neighbor. This ensures that the closest point has a high connectivity weight.
* $\sigma_i$ is a scaling factor, found via binary search, which normalizes distances for each point's local neighborhood.

**B. Computing High-Dimensional Similarities**

The directed edge weight from $\mathbf{x}_i$ to $\mathbf{x}_j$ is then given by:

$$w_{j|i} = \exp\left(-\frac{\max(0, d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i)}{\sigma_i}\right)$$

This gives a non-symmetric adjacency matrix. To create an undirected graph that represents the overall topological structure, UMAP symmetrizes these weights. The joint probability (or similarity) $p_{ij}$ between points $i$ and $j$ is calculated using a probabilistic t-conorm (fuzzy set union):

$$p_{ij} = w_{j|i} + w_{i|j} - w_{j|i} \cdot w_{i|j}$$

This formula combines the two directed edge weights into a single undirected weight, representing the strength of connection between the two points on the manifold. The collection of these weights forms the high-dimensional graph.

#### 2. Finding the Low-Dimensional Embedding

The next step is to find a set of low-dimensional points $\mathcal{Y} = \{\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_N\}$ whose own topological structure is as similar as possible to the one defined by the $p_{ij}$ values.

**A. Low-Dimensional Similarities**

The similarity $q_{ij}$ between two points in the low-dimensional space is modeled using a family of curves similar to the Student's t-distribution:

$$q_{ij} = \left(1 + a\|\mathbf{y}_i - \mathbf{y}_j\|_2^{2b}\right)^{-1}$$

The parameters $a$ and $b$ are typically determined by a non-linear least squares fit to the curve $(1+y)^{-1}$, with default values in the UMAP implementation around $a \approx 1.93$ and $b \approx 0.79$. This provides a heavy-tailed distribution, which, like in t-SNE, helps separate clusters and avoid crowding.

#### Cost Function and Optimization

UMAP's goal is to arrange the points $\mathbf{y}_i$ such that the similarities $q_{ij}$ closely match the high-dimensional similarities $p_{ij}$. This is framed as minimizing the **cross-entropy (CE)** between the two distributions:

$$C = \sum_{i \neq j} \left[ p_{ij} \log\left(\frac{p_{ij}}{q_{ij}}\right) + (1 - p_{ij}) \log\left(\frac{1 - p_{ij}}{1 - q_{ij}}\right) \right]$$

This cost function has two distinct parts:
* **Attractive Force:** The first term, $p_{ij} \log\left(\frac{p_{ij}}{q_{ij}}\right)$, acts as an attractive force. It is active when $p_{ij}$ is large (i.e., for connected points in the high-dimensional graph), pulling their low-dimensional counterparts $\mathbf{y}_i$ and $\mathbf{y}_j$ together.
* **Repulsive Force:** The second term, $(1 - p_{ij}) \log\left(\frac{1 - p_{ij}}{1 - q_{ij}}\right)$, acts as a repulsive force. It is active for all pairs of points but is strongest when $p_{ij}$ is small, pushing their low-dimensional representations apart.

**Optimization via Stochastic Gradient Descent**

The cost function is optimized using Stochastic Gradient Descent (SGD). A major performance advantage of UMAP comes from how it handles the repulsive force. Since the high-dimensional graph is very sparse (most $p_{ij}$ are zero or near-zero), computing the repulsive force for all pairs of points would be computationally expensive.

Instead, UMAP uses **[[Contrastive Learning|negative sampling]]**. For each sample (an edge with a non-zero $p_{ij}$), the attractive force is computed. Then, a small number of "negative samples" (pairs of points with $p_{ij}=0$) are randomly chosen, and the repulsive force is computed only for them. This approximation dramatically reduces the computational complexity of each optimization step, making UMAP much faster than t-SNE, which calculates repulsive forces between all pairs.

#### Conclusion

UMAP provides a powerful framework for dimensionality reduction by building on a solid mathematical foundation in topology and manifold learning. It constructs a fuzzy topological representation of high-dimensional data and then optimizes a low-dimensional embedding to match this structure. Its use of a cross-entropy cost function combined with efficient negative sampling during optimization allows it to be both fast and highly effective at preserving both the local and global structure of the original data.