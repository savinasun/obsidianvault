### [[dl_moc]]  
## **Singular Value Decomposition**

**Tags**::  #dl_theo
**Links**::

---
### Singular Value Decomposition
$$
A = U\Sigma V^{\star} \in \mathbb{C}^{m\times n}
$$
+ $U:=(u_1, \dots, u_m) \in\mathbb{C}^{m\times m}$: complex unitary matrix whose columns are left singular vectors 
	+ $u_1, \dots, u_m$ form an orthonormal basis of $\mathbb{C}^m$
+ $V:=(v_1, \dots, v_n)\in\mathbb{C}^{n\times n}$: complex unitary matrix whose columns are right singular vectors
	+ $v_1, \dots, v_m$ form an orthonormal basis of $\mathbb{C}^n$
+ $\Sigma\in\mathbb{R}^{m\times n}$: rectangle diagonal matrix whose diagonal entries are non-negative and are called the **singular values** of $A$
+ $V^{\star}$ is the conjugate transpose of $V$
+ The number of non-negative singular values is equal to the rank of $A$
+ **Existence and Uniqueness**: 
	A SVD **always exists for any real or complex matrix** though it is **not unique**.
	It is always possible to choose the decomposition such that the singular values $\Sigma_{ii}$ are in descending order. In this case, $\Sigma$ is uniquely determined by ⁠$A$.


### SVD of Real Matrices

If $A\in\mathbb{R}^{m\times n}$, then $U$ and $V$ are guaranteed to be real orthogonal matrices and $$A = U\Sigma V^T \in \mathbb{R}^{m\times n}$$![[svd1.png]]
+ $U \in \mathbb{R}^{m\times m}$: orthogonal matrix
+ $V \in \mathbb{R}^{n\times n}$: orthogonal matrix
+ $\Sigma \in \mathbb{R}^{m\times n}$: rectangle diagonal matrix whose diagonal entries are non-negative


### Geometric Interpretation of the largest singular value (C. Jordan)

![[Pasted image 20250628182448.png]]


**Theorem**. Let $A\in\mathbb{R}^{m\times n}$, then the largest singular value of $A$ is given by
$$
\sigma_1 = max{\|Ax\|}
$$

Proof. Based on [[Linear Algebra#Min-Max Theorem|min-max theorem]]
### Applications of SVD

1. Computation of bases of the four fundamental subspaces of A. 
2. Polar decomposition of square matrices 
3. Least squares approximation of vectors and data fitting 
4. Data compression 
5. Approximation of A by matrices of lower rank 
6. Computation of matrix norms