### [[dl_moc]]  
# **Linear Algebra**

**Tags**::  
**Links**:: [Gilbert Strung Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)

![[linear_algebra_01.png]]
The above figure from [@zotero-item-237] captures a central idea of linear algebra.

## Basic Concepts
---
### Linear Independence
Vectors ${v_1, \dots, v_k}$ are independent if $\sum\limits_{i=1}^{k} \alpha_i v_i = 0 \Rightarrow \alpha_i = 0 \; \forall i$
It also implies that no vector lies in the span of the combinations of others.


### Basis Vectors
A **basis** of a vector space $V$ is a set of vectors ${v_1, \dots, v_k}$ that is **linearly independent** and **spans $V$**. 

We have the following could be proved,
- Every vector $v\in V$ has a unique representation as a linear combination of the basis vectors $v = \sum \alpha_i v_i$ .
- The number of basis vectors equals the **dimension** of $V$.

**Orthonormal Basis:** An orthonormal basis is a set of mutually perpendicular (orthogonal) vectors in a vector space, where each vector is also a unit vector.


### Span of Vectors
The **span** of vectors ${v_1, \dots, v_k}$ in $\mathbb{R}^n$ is: 
$$\text{Span}({v_1, \dots, v_k}) = \left\{\sum_{i=1}^{k} \alpha_i v_i \mid \alpha_i \in \mathbb{R}\right\}$$
Span is the smallest subspace containing all linear combinations of the given vectors.


### Four Fundamental Subspaces
**Column Space**: the **image** of the linear map; the span of the columns of $A$, denoted as $$\text{Im}(A) = \mathcal{C}(A) \coloneqq \{ Ax \mid x \in \mathbb{R}^n \}\subseteq\mathbb{R}^m$$**Row Space**: the span of the rows of $A$, , equivalently the column space of $A^\top$, often denoted as$$
\text{Im}(A^\top) =\mathcal{C}(A^\top) \coloneqq \{ A^\top y \mid y \in \mathbb{R}^m \}\subseteq\mathbb{R}^n$$**Null Space (Kernel)**: the **kernel** of the linear map; it is the part of the domain which is mapped to the zero-vector of the co-domain, often denoted as
$$
\text{Ker}(A)= \mathcal{N}(A) \coloneqq \{ x \in \mathbb{R}^n \mid Ax = 0 \}\subseteq\mathbb{R}^n
$$
**Left Null Space**: the set of vectors $y \in \mathbb{R}^m$ such that $A^\top y = 0$, often denoted as
$$
\text{Ker}(A^\top)=\mathcal{N}(A^\top) \coloneqq \{ y \in \mathbb{R}^m \mid A^\top y = 0\} = \{x \in \mathbb{R}^m \mid x A = 0\} \subseteq \mathbb{R}^m
$$

**The four fundamental subspaces form two orthogonal pairs:**
$$
\mathcal{C}(A)^\perp = \mathcal{N}(A^\top)\subseteq\mathbb{R}^m, \quad \mathcal{C}(A^\top)^\perp = \mathcal{N}(A)\subseteq\mathbb{R}^n
$$

### Rank
The **rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the dimension of its column space: $\text{rank}(A) = \dim(\mathcal{C}(A))$, we also know that
+ rank is the maximal number of linearly independent rows or columns
- $\text{rank}(A) = \text{rank}(A^\top)$
- $\text{rank}(A) \le \min(m,n)$


### Matrix Multiplication
There are four types of interpretations for matrix multiplication.
Denote $A\in\mathbb{R}^{m\times n}$, $B\in\mathbb{R}^{n\times r}$ and $C = AB$.
#### 1. Matrix-Matrix Multiplication
Multiplying matrices $A$ and $B$ will produce matrix $C$ of size $m\times r$ with elements,
$$
c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}, \quad i=1,\dots, m, \quad j=1,\dots, r
$$
#### 2. Matrix-Vector Multiplication
The columns in $C$ could be represented as the linear combination of columns in $A$,
$$
C_j = \sum_{k=1}^n A_k b_{kj}, \quad j=1, \dots, r
$$
The following figure from [@zotero-item-192] shows how it works
![[linear_algebra_mat_mul_01.png]]
#### 3. Vector-Matrix Multiplication
The row in matrix $C$ could be represented as the linear combination of rows in $B$,
$$
C_i^T = \sum_{k=1}^n a_{ik} B_k^T, \quad i=1,\dots, m
$$
![[linear_algebra_mat_mul_02.png]]
#### 4. Vector-Vector Multiplication
 The product of $A$ and $B$ could be represented as a sum of (columns of $A$) times (rows of $B$),
 $$
 C = \sum_{k=1}^n A_k B_k^T
$$
![[linear_algebra_mat_mul_03.png]]
#### 5. Block-Block Multiplication
The fifth way is to chop matrices in blocks and multiply blocks by any of the previous methods.
For example, $$
C_1 = A_1 B_1 + A_2 B_3
$$
![[linear_algebra_mat_mul_04.png]]


### Inverse of Matrix
The **inverse** of a square matrix $A$ is a matrix $A^{-1}$ such that $$A A^{-1} = A^{-1} A = I$$Only **invertible** matrices (also called **non-singular**) have an inverse.

- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$ for invertible matrices $A$, $B$
- $(A^\top)^{-1} = (A^{-1})^\top$
- $(cA)^{-1} = \frac{1}{c}A^{-1}$ for $c \ne 0$

The following methods are used to derive the matrix inverses:
- **Gaussian elimination** (row reduction to identity)
- **LU decomposition** (solve $Ax = b$ for each column of $I$)
- **Adjugate formula**: (not practical for large $n$)


### Moore–Penrose Pseudoinverse
- If $A$ is not square or not invertible, its **Moore–Penrose pseudoinverse** $A^+$ generalizes the inverse: from the SVD $A = U\Sigma V^\top$


### Eigenvalues and Eigenvectors
An **eigenvector** $\mathbf{v}\in\mathbb{C}^n$ (or **characteristic vector**) is a non-zero vector that has its direction unchanged (or reversed) by a given linear transformation $A\in \mathbb{R}^{n\times n}$, i.e. $\exists \lambda\in\mathbb{C}$ s.t.
$$ A\mathbf{v} = \lambda \mathbf{v}$$
Here scalar $\lambda$ is the **eigenvalue** of $A$ and could be solved through equation
$$p(\lambda) := \det(A−\lambda I)=0$$
where $p(\lambda)$ is the **characteristic polynomial** of matrix $A$

This equation will have $N_\lambda$ distinct solutions, where $1 \leq N_\lambda \leq n$. The set of solutions, that is, the eigenvalues, is called the **spectrum** of $A$ and $\rho(A) = \max |\lambda_i|$ is called the **spectral radius** of $A$.

**Diagonalization:** If $A$ has $n$ linearly independent eigenvectors, it can be diagonalized: 
$$A = V\Lambda V^{-1}$$


### Determinant
When a matrix applies transformation to a space, how much is the area scaled by this transformation is given by the determinant.

If determinant of matrix is 0, it squishes the space into a smaller dimension than the input dimension.

Negative determinant mean the orientation of the transformed space is inverted.

+ For square matrix $A$: $\det(A) = 0 \iff A \text{ is not invertible}$
- $\det(AB) = \det(A)\det(B)$
- $\det(A^\top) = \det(A)$
- $\det(Q) = \pm 1$ for orthogonal $Q$
#### Geometric interpretation
Volume of the parallelepiped formed by the columns of $A$.




## Special Matrices
---
### Symmetric Matrices
A matrix $A$ is called symmetric iff. $A=A^T$.
**Spectral Theorem**: Real symmetric matrices have real eigenvalues and are diagonalizable and thus having orthogonal eigenvectors.
**Spectral Decomposition**


### Gram Matrix
$G = AA^T$ is called Gram matrix, we have the properties for any $A$,
$$\text{rank}(AA^T) = \text{rank}(A^TA)= \text{rank}(A)= \text{rank}(A^T)$$


### Normal Matrices
A matrix $A$ is **normal** iff. $AA^\ast = A^\ast A$
Normal matrices include symmetric, Hermitian, unitary, and orthogonal matrices.
#### Properties
- Diagonalizable by a unitary (complex) or orthogonal (real) matrix: $A = Q \Lambda Q^\ast$
- Eigenvectors of normal matrices form an orthonormal basis.


### Orthogonal Matrices
A real $n \times n$ matrix $Q$ is called orthogonal if $Q^T Q = I$, i.e. $Q^{-1}=Q^T$.
In geometric interpretation, the orthogonal matrix represents rotations and reflections
#### Properties
- $Q^{-1} = Q^\top$
- Preserves length and angles
- Preserves inner products: $|Q| = \pm1$, $|Qx|=|x|$


### Diagonalizable Matrices
A matrix $A$ is called diagonalizable matrix if there exists a diagonal matrix $D$ and an invertible matrix $P$ where
$$A = PDP^{-1}.$$
A diagonalizable matrix implies a full set of linearly independent eigenvectors.


### Orthogonally Diagonalizable
A matrix $A$ is called diagonalizable matrix if there exists a diagonal matrix $D$ and an orthogonal matrix $Q$ where
$$A = QDQ^{T}.$$
It is only possible if $A$ is symmetric.
An orthogonally diagonalizable matrix implies eigenvectors form an orthonormal basis.


### Positive Definite and Semidefinite
**Positive definite**: $x^\top A x > 0 ; \forall x \ne 0$
**Positive semidefinite**: $x^\top A x \geq 0 ; \forall x$
All eigenvalues are > 0 (PD) or $\geq 0$ (PSD)


## Matrix Factorizations
---
There are six matrix factorizations commonly used in different domains:
### Cholesky Decomposition
For symmetric, positive-definite $A$: $A = LL^\top$, where $L$ is lower triangular

### LU Factorization
$A = LU$, where $L$ is lower-triangular, $U$ is upper-triangular
May require row permutations: $PA = LU$

### QR Factorization
$A = QR$, where $Q$ is orthogonal, $R$ is upper-triangular
Used in least squares problems

### Schur Decomposition
$A = Q T Q^\top$, where $T$ is upper triangular, $Q$ is orthogonal
All square matrices have a Schur decomposition

### [[Eigen Decomposition]]
$A = V \Lambda V^{-1}$ if $A$ has $n$ independent eigenvectors
$\Lambda$: diagonal matrix of eigenvalues

### [[SVD|Singular Value Decomposition (SVD)]]
$A = U\Sigma V^\top$ where:

- $U, V$ are orthogonal
	
- $\Sigma$ is diagonal with non-negative singular values $\sigma_i$
	
- $\sigma_i = \sqrt{\lambda_i}$ of $A^\top A$

Works for any $m \times n$ matrix.


## Theorems
---
### Rank-Nullity Theorem
![[linear_algebra_02.png]]
Let $A: V\to W$, be a linear transformation between two vector spaces where $A$'s domain $V$ is finite dimensional. Then:
$$\text{rank}(A) + \dim(\mathcal{N}(A)) = \dim(V)$$
or
$$
\text{Im}(A)\oplus \text{Ker}(A)\cong V.
$$


### Spectral Theorem
A matrix is called symmetric if $A=A^T$.

- A real symmetric matrix $A$ can be diagonalized by an orthogonal matrix: $$A = Q \Lambda Q^\top$$
- Eigenvalues are real, eigenvectors are orthonormal

### Min-Max Theorem (Courant-Fischer)

- For symmetric $A$ with eigenvalues $\lambda_1 \le \cdots \le \lambda_n$: λk=min⁡dim⁡(V)=kmax⁡x∈V,∥x∥=1x⊤Ax\lambda_k = \min_{\dim(V)=k} \max_{x \in V, \|x\|=1} x^\top A x
    
- Provides variational characterization of eigenvalues


The singular values {$σ_k$} of a square matrix $A$ are the square roots of the eigenvalues of $A^T A$. Denote $\sigma_k$ the $k$-th entry in the decreasing sequence of the singular values, then

$$
\sigma_k = \max \min\limits_{x\in S^k, \|x\|=1} (A^T Ax, x)^{\frac 1 2} = \max \min \|Ax\|.
$$


## Algorithms
---
## Gram-Schidmt






---

References:

- Gilbert Strang, "Introduction to Linear Algebra", 5th Edition: [https://math.mit.edu/~gs/linearalgebra/](https://math.mit.edu/~gs/linearalgebra/)
    
- Horn & Johnson, "Matrix Analysis"
    
- Trefethen & Bau, "Numerical Linear Algebra"
