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

**Properties:**
- Every vector $v\in V$ has a unique representation as a linear combination of the basis vectors $v = \sum \alpha_i v_i$ .
- The number of basis vectors equals the **dimension** of $V$.

**Geometric interpretation:** In $\mathbb{R}^n$, basis vectors define the coordinate axes.

### Span of Vectors
The **span** of vectors ${v_1, \dots, v_k}$ is: $\text{Span}({v_1, \dots, v_k}) = \left\{\sum_{i=1}^{k} \alpha_i v_i \mid \alpha_i \in \mathbb{R}\right\}$

**Geometric interpretation:** Span is the smallest subspace containing all linear combinations of the given vectors.

### Rank
The **rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the dimension of the column space: $\text{rank}(A) = \dim(\text{Col}(A))$

**Properties:**
- $\text{rank}(A) = \text{rank}(A^\top)$
- $\text{rank}(A) \le \min(m,n)$
- Rank is the maximal number of linearly independent rows or columns

**Rank-Nullity Theorem**: $$\text{rank}(A) + \dim(\text{Null}(A)) = n$$

### Eigenvalues and Eigenvectors

An **eigenvector** $\mathbf{v}\in\mathbb{C}^n$ (or **characteristic vector**) is a non-zero vector that has its direction unchanged (or reversed) by a given linear transformation $A\in \mathbb{R}^{n\times n}$, i.e. $\exists \lambda\in\mathbb{C}$ s.t.
$$ A\mathbf{v} = \lambda \mathbf{v}$$
Here scalar $\lambda$ is the **eigenvalue** of $A$ and could be solved through equation
$$p(\lambda) := \det(A−\lambda I)=0$$
where $p(\lambda)$ is the **characteristic polynomial** of matrix $A$

This equation will have $N_\lambda$ distinct solutions, where $1 \leq N_\lambda \leq n$. The set of solutions, that is, the eigenvalues, is called the **spectrum** of $A$ and $\rho(A) = \max |\lambda_i|$ is called the **spectral radius** of $A$.

**Diagonalization:** If $A$ has $n$ linearly independent eigenvectors, it can be diagonalized: $A = V\Lambda V^{-1}$


### Determinant
- For square matrix \$A\$: \(\det(A) = 0 \iff A \text{ is not invertible}\)

- **Properties:**

- \$\det(AB) = \det(A)\det(B)\$

- \$\det(A^\top) = \det(A)\$

- \$\det(Q) = \pm 1\$ for orthogonal \$Q\$

- **Geometric interpretation:** Volume of the parallelepiped formed by the columns of \$A\$.


- A scalar value that encodes volume scaling and invertibility: det⁡(A)=0  ⟺  A is singular (not invertible)\det(A) = 0 \iff A \text{ is singular (not invertible)}
    
- Geometric interpretation: $|\det(A)|$ is the volume of the parallelepiped defined by columns of $A$
    

### Linear Independence

- Vectors ${v_1, \dots, v_k}$ are linearly independent if: ∑i=1kαivi=0⇒αi=0  ∀i\sum_{i=1}^{k} \alpha_i v_i = 0 \Rightarrow \alpha_i = 0 \; \forall i
    

## 2. Special Matrices

### Symmetric Matrices

- $A^\top = A$
    
- Real symmetric matrices have real eigenvalues and are diagonalizable
    

### Orthogonal Matrices

- $Q^\top Q = QQ^\top = I$
    
- Preserves inner products: $|Qx| = |x|$
    

### Diagonalizable Matrices

- $A = PDP^{-1}$ where $D$ is diagonal, $P$ is invertible
    
- Implies a full set of linearly independent eigenvectors
    

### Orthogonally Diagonalizable

- $A = QDQ^\top$ where $Q$ is orthogonal
    
- Only possible if $A$ is symmetric
    

### Positive Definite / Semidefinite

- **Positive definite**: $x^\top A x > 0 ; \forall x \ne 0$
    
- **Positive semidefinite**: $x^\top A x \geq 0 ; \forall x$
    
- All eigenvalues are > 0 (PD) or $\geq 0$ (PSD)
    

## 3. Matrix Factorizations

### Cholesky Decomposition

- For symmetric, positive-definite $A$: $A = LL^\top$, where $L$ is lower triangular
    

### LU Factorization

- $A = LU$, where $L$ is lower-triangular, $U$ is upper-triangular
    
- May require row permutations: $PA = LU$
    

### QR Factorization

- $A = QR$, where $Q$ is orthogonal, $R$ is upper-triangular
    
- Used in least squares problems
    

### Schur Decomposition

- $A = Q T Q^\top$, where $T$ is upper triangular, $Q$ is orthogonal
    
- All square matrices have a Schur decomposition
    

### Eigen Decomposition

- $A = V \Lambda V^{-1}$ if $A$ has $n$ independent eigenvectors
    
- $\Lambda$: diagonal matrix of eigenvalues
    

### Singular Value Decomposition (SVD)

- $A = U\Sigma V^\top$ where:
    
    - $U, V$ are orthogonal
        
    - $\Sigma$ is diagonal with non-negative singular values $\sigma_i$
        
    - $\sigma_i = \sqrt{\lambda_i}$ of $A^\top A$
        
- Works for any $m \times n$ matrix
    

## 4. Theorems

### Spectral Theorem

- A real symmetric matrix $A$ can be diagonalized by an orthogonal matrix: A=QΛQ⊤A = Q \Lambda Q^\top
    
- Eigenvalues are real, eigenvectors are orthonormal
    

### Min-Max Theorem (Courant-Fischer)

- For symmetric $A$ with eigenvalues $\lambda_1 \le \cdots \le \lambda_n$: λk=min⁡dim⁡(V)=kmax⁡x∈V,∥x∥=1x⊤Ax\lambda_k = \min_{\dim(V)=k} \max_{x \in V, \|x\|=1} x^\top A x
    
- Provides variational characterization of eigenvalues
    

---

### Additional Concepts

- **Null Space**: $\text{Null}(A) = {x \mid Ax = 0}$
    
- **Column Space**: Span of the columns of $A$
    
- **Row Space**: Span of the rows of $A$
    
- **Rank-Nullity Theorem**: rank(A)+dim⁡(Null(A))=n\text{rank}(A) + \dim(\text{Null}(A)) = n where $A \in \mathbb{R}^{m \times n}$
    
- **Pseudoinverse**: For $A \in \mathbb{R}^{m\times n}$, $A^+ = V\Sigma^+ U^\top$
    

---

References:

- Gilbert Strang, "Introduction to Linear Algebra", 5th Edition: [https://math.mit.edu/~gs/linearalgebra/](https://math.mit.edu/~gs/linearalgebra/)
    
- Horn & Johnson, "Matrix Analysis"
    
- Trefethen & Bau, "Numerical Linear Algebra"

### Eigenvalues and Eigenvectors 



![[Pasted image 20250628213712.png|600]]



## Special Matrices
---
### Symmetric Matrices 

A matrix is called symmetric if $A=A^T$.

### Normal Matrices


### Orthogonal Matrices

A real $n \times n$ matrix $Q$ is called orthogonal if $Q^T Q = I$, i.e. $Q^{-1}=Q^T$.

### Diagonalizable Matrices 

If your transformation (matrix) has lots of eigenvectors, enough so you can choose a set of vectors that spans a full space, then you could change the basis of your coordinate system to these eigenvectors. These new basis vectors are called "Eigenbasis". This has as a nice consequence of guaranteeing that the new transformation matrix is always a diagonal matrix.

In this new diagonal matrix, all the columns represent eigenvectors with the values in the diagonal as their eigen values.

For example, let's say A is the new change of basis matrix, with eigenvectors as its columns. Then our transformation matrix D can be factorized in the new space as A−1DA, where D is guaranteed to be a diagonal matrix of eigen vectors.

Diagonal matrix are much easier to handle as the matrix multiplication simply boils down to raising the power of basis vectors to N number of times.

### Orthogonally Diagonalizable Matrices 

An n×n matrix A is said to be orthogonally diagonalizable when an orthogonal matrix P can be found such that P−1AP=PTAP is a diagonal matrix.


## Theorems
---
### Spectral Theorem


### Min-Max Theorem

The singular values {$σ_k$} of a square matrix $A$ are the square roots of the eigenvalues of $A^T A$. Denote $\sigma_k$ the $k$-th entry in the decreasing sequence of the singular values, then

$$
\sigma_k = \max \min\limits_{x\in S^k, \|x\|=1} (A^T Ax, x)^{\frac 1 2} = \max \min \|Ax\|.
$$
![[Screenshot 2025-06-28 at 23.45.06.png]]
## Matrix Factorizations 
---
The following are known as the big six matrix factorizations:

1. Cholesky Factorization
2. LU Factorization
3. QR Factorization
4. Schur Decomposition
5. [[Eigen Decomposition]]
6. [[Singular Value Decomposition]]