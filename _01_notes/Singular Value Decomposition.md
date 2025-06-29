### [[dl_moc]]  
## **Singular Value Decomposition**

**Tags**::  #dl_theo
**Links**::

---
### Singular Value Decomposition
$$
A = U\Sigma V^{\star} \in \mathbb{C}^{m\times n}
$$
+ $U:=(u_1, \dots, u_m) \in\mathbb{C}^{m\times m}$: complex unitary matrix whose columns made of left singular vectors 
	+ $u_1, \dots, u_m$ form an orthonormal basis of $\mathbb{C}^m$
+ $V:=(v_1, \dots, v_n)\in\mathbb{C}^{n\times n}$: complex unitary matrix whose columns made of right singular vectors
	+ $v_1, \dots, v_m$ form an orthonormal basis of $\mathbb{C}^n$
+ $\Sigma\in\mathbb{R}^{m\times n}$: rectangle diagonal matrix of singular values (non-negative real numbers)
+ $V^{\star}$ is the conjugate transpose of $V$
+ The number of non-negative singular values is equal to the rank of $A$
+ **Existence and Uniqueness**: 
	A SVD **always exists for any real or complex matrix** though it is **not unique**.
	It is always possible to choose the decomposition such that the singular values $\Sigma_{ii}$ are in descending order. In this case, $\Sigma$ is uniquely determined by ⁠$A$.
+ If $A\in\mathbb{R}^{m\times n}$, then $U$ and $V$ are guaranteed to be real orthogonal matrices and $$A = U\Sigma V^T \in \mathbb{R}^{m\times n}$$![[svd1.png]]
+ 
![[Pasted image 20250628182448.png|300]]

### Eigen-Value Decomposition

