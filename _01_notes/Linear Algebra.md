### [[dl_moc]]  
## **Linear Algebra**

**Tags**::  
**Links**:: [Gilbert Strung Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)

---

## Basic Concepts
---
### Eigenvalues and Eigenvectors 

An **eigenvector** $\mathbf{v}$ (or **characteristic vector**) is a vector that has its direction unchanged (or reversed) by a given linear transformation $A$, i.e. $\exists \lambda\in\mathbb{R}$ s.t.
$$ A\mathbf{v} = \lambda \mathbf{v}$$

Here scalar $\lambda$ is the **eigenvalue** of $A$ and could be solved through equation
$$p(\lambda) := \det(A−\lambda I)=0$$
where $p(\lambda)$ is the **characteristic polynomial** of matrix $A$

This equation will have $N_\lambda$ distinct solutions, where $1 \leq N_\lambda \leq N$. The set of solutions, that is, the eigenvalues, is called the **spectrum** of $A$.

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

The singular values {$σ_k$} of a square matrix $A$ are the square roots of the eigenvalues of $A^T A$. An immediate consequence of the first equality in the min-max theorem is:

σk↓=maxS:dim⁡(S)=kminx∈S,‖x‖=1(M∗Mx,x)12=maxS:dim⁡(S)=kminx∈S,‖x‖=1‖Mx‖.![{\displaystyle \sigma _{k}^{\downarrow }=\max _{S:\dim(S)=k}\min _{x\in S,\|x\|=1}(M^{*}Mx,x)^{\frac {1}{2}}=\max _{S:\dim(S)=k}\min _{x\in S,\|x\|=1}\|Mx\|.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/1d42895c98a7513fc5091061debfade994e2591d)

Similarly,

σk↓=minS:dim⁡(S)=n−k+1maxx∈S,‖x‖=1‖Mx‖.![{\displaystyle \sigma _{k}^{\downarrow }=\min _{S:\dim(S)=n-k+1}\max _{x\in S,\|x\|=1}\|Mx\|.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/6bef0c507f13beccbf3a0d8dc0f7860a145f66f9)

Here σk↓![{\displaystyle \sigma _{k}^{\downarrow }}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a09c2a2ee7f6ce9e88ff7be991b6dc1f32c725c0) denotes the _k_th entry in the decreasing sequence of the singular values, so that σ1↓≥σ2↓≥⋯![{\displaystyle \sigma _{1}^{\downarrow }\geq \sigma _{2}^{\downarrow }\geq \cdots }](https://wikimedia.org/api/rest_v1/media/math/render/svg/e773352c547e7b24d7d3e0e94650a679abe2afed).


## Matrix Factorizations 
---
The following are known as the big six matrix factorizations:

1. Cholesky Factorization
2. LU Factorization
3. QR Factorization
4. Schur Decomposition
5. [[Eigenvalue Decomposition]]
6. [[Singular Value Decomposition]]