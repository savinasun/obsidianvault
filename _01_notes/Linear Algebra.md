### [[dl_moc]]  
## **Linear Algebra**

**Tags**::  
**Links**::

## Basic Concepts
---
### Eigenvalues and eigenvectors 

An **eigenvector** $\mathbf{v}$ (or **characteristic vector**) is a vector that has its direction unchanged (or reversed) by a given linear transformation $A$, i.e. $\exists \lambda\in\mathbb{R}$ s.t.
$$ A\mathbf{v} = \lambda \mathbf{v}$$

Here scalar $\lambda$ is the **eigenvalue** of $A$ and could be solved through equation
$$p(\lambda) := \det(A−\lambda I)=0$$
where $p(\lambda)$ is the **characteristic polynomial** of matrix $A$

This equation will have $N_\lambda$ distinct solutions, where $1 \leq N_\lambda \leq N$. The set of solutions, that is, the eigenvalues, is called the **spectrum** of $A$.

## Special Matrices
---
### Symmetric Matrices 

A matrix is called symmetric if $A=A^T$.

### Normal Matrices


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



## Matrix Factorizations 
---
The following are known as the big six matrix factorizations:

1. Cholesky Factorization
2. LU Factorization
3. QR Factorization
4. Schur Decomposition
5. [[Eigenvalue Decomposition]]
6. [[Singular Value Decomposition]]