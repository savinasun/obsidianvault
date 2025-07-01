### [[dl_moc]]  
## **Linear Algebra**

**Tags**::  
**Links**:: [Gilbert Strung Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)

---

## Basic Concepts
---
### Basis Vectors
A **basis** of a vector space $V$ is a set of vectors ${v_1, \dots, v_k}$ that is:
- Linearly independent
- Spans $V$
#### Properties
- Every vector $v\in V$ has a unique representation as a linear combination of the basis vectors $v = \sum \alpha_i v_i$ .
- The number of basis vectors equals the **dimension** of $V$.

**Orthonormal Basis:** An orthonormal basis is a set of mutually perpendicular (orthogonal) vectors in a vector space, where each vector is also a unit vector.

### Four Types of Matrix Multiplication
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
![[Pasted image 20250701113323.png]]
#### 5. Block-Block Multiplication
The fifth way is to chop matrices in blocks and multiply blocks by any of the previous methods.
For example, $$
C_1 = A_1 B_1 + A_2 B_3
$$
![[Pasted image 20250701113427.png]]

### Span of Vectors
The **span** of vectors ${v_1, \dots, v_k}$ is: 
$$\text{Span}({v_1, \dots, v_k}) = \left\{\sum_{i=1}^{k} \alpha_i v_i \mid \alpha_i \in \mathbb{R}\right\}$$
Span is the smallest subspace containing all linear combinations of the given vectors.


### Linear Independence
Vectors ${v_1, \dots, v_k}$ are independent if $\sum\limits_{i=1}^{k} \alpha_i v_i = 0 \Rightarrow \alpha_i = 0 \; \forall i$
It also implies that no vector lies in the span of the combinations of others.


### Null Space
the **kernel** of a [linear map](https://en.wikipedia.org/wiki/Linear_map "Linear map"), also known as the **null space** or **nullspace**, is the part of the [domain](https://en.wikipedia.org/wiki/Domain_of_a_function "Domain of a function") which is mapped to the [zero vector](https://en.wikipedia.org/wiki/Zero_element#Additive_identities "Zero element") of the [co-domain](https://en.wikipedia.org/wiki/Codomain "Codomain"); the kernel is always a [linear subspace](https://en.wikipedia.org/wiki/Linear_subspace "Linear subspace") of the domain.


### Rank
The **rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the dimension of the column space: $\text{rank}(A) = \dim(\text{Col}(A))$
#### Properties
- $\text{rank}(A) = \text{rank}(A^\top)$
- $\text{rank}(A) \le \min(m,n)$
- Rank is the maximal number of linearly independent rows or columns
#### Rank-Nullity Theorem
$$\text{rank}(A) + \dim(\text{Null}(A)) = n$$
![[Pasted image 20250628213712.png|300]]

**Null Space**: $\text{Null}(A) = \{x \mid Ax = 0\}$
**Column Space**: Span of the columns of $A$, usually denoted as $\textrm{Col}(A)$
**Row Space**: Span of the rows of $A$, usually denoted as $\textrm{Col}(A^T)$


### Gram Matrix
$G = AA^T$
$$rank(AA^T) = rank(A^TA)=rank(A)=rank(A^T)$$


### Inverse of Matrix
The **inverse** of a square matrix $A$ is a matrix $A^{-1}$ such that $$A A^{-1} = A^{-1} A = I$$Only **invertible** matrices (also called **non-singular**) have an inverse.
#### Properties
- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$ for invertible matrices $A$, $B$
- $(A^\top)^{-1} = (A^{-1})^\top$
- $(cA)^{-1} = \frac{1}{c}A^{-1}$ for $c \ne 0$
#### Methods to compute
- **Gaussian elimination** (row reduction to identity)
- **LU decomposition** (solve $Ax = b$ for each column of $I$)
- **Adjugate formula**: (not practical for large $n$)
#### Moore–Penrose Pseudoinverse
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
#### Properties
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


### Positive Definite / Semidefinite
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

### [[Singular Value Decomposition]] (SVD)
$A = U\Sigma V^\top$ where:

- $U, V$ are orthogonal
	
- $\Sigma$ is diagonal with non-negative singular values $\sigma_i$
	
- $\sigma_i = \sqrt{\lambda_i}$ of $A^\top A$

Works for any $m \times n$ matrix.

## Theorems
---
### Spectral Theorem
A matrix is called symmetric if $A=A^T$.

- A real symmetric matrix $A$ can be diagonalized by an orthogonal matrix: A=QΛQ⊤A = Q \Lambda Q^\top
    
- Eigenvalues are real, eigenvectors are orthonormal

### Min-Max Theorem (Courant-Fischer)

- For symmetric $A$ with eigenvalues $\lambda_1 \le \cdots \le \lambda_n$: λk=min⁡dim⁡(V)=kmax⁡x∈V,∥x∥=1x⊤Ax\lambda_k = \min_{\dim(V)=k} \max_{x \in V, \|x\|=1} x^\top A x
    
- Provides variational characterization of eigenvalues


The singular values {$σ_k$} of a square matrix $A$ are the square roots of the eigenvalues of $A^T A$. Denote $\sigma_k$ the $k$-th entry in the decreasing sequence of the singular values, then

$$
\sigma_k = \max \min\limits_{x\in S^k, \|x\|=1} (A^T Ax, x)^{\frac 1 2} = \max \min \|Ax\|.
$$
![[Screenshot 2025-06-28 at 23.45.06.png]]


---

References:

- Gilbert Strang, "Introduction to Linear Algebra", 5th Edition: [https://math.mit.edu/~gs/linearalgebra/](https://math.mit.edu/~gs/linearalgebra/)
    
- Horn & Johnson, "Matrix Analysis"
    
- Trefethen & Bau, "Numerical Linear Algebra"
