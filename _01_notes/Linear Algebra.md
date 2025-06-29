### [[dl_moc]]  
## **Linear Algebra**

**Tags**::  
**Links**::

## Basic Concepts
---
### Eigenvalues and eigenvectors 

	Most vectors get knocked off from their span when undergoing transforms (i.e. when being multiplied with matrices), for example during rotation. But some vectors may remain on their span (none in rotation), and only get scaled. These vectors are called the eigenvectors of the matrix.

The factor with which the eigenvector get squished or stretched is called the eigenvalue of that eigenvector.

Let's say we have a matrix A that represents a linear transformation.  

An **eigenvector** $\mathbf{v}$ (or **characteristic vector**) is a vector that has its direction unchanged (or reversed) by a given linear transformation $A$, i.e. $\exists \lambda\in\mathbb{R}$ s.t.
$$ A\mathbf{v} = \lambda \mathbf{v}$$

Then v→ represents the eigen vector and λ it's eigen value when

Av→=λv→

For eigenvalues, we have

det(A−λI)=0

this expression can be used to solve for possible eigen values of the matrix A.


## Special Matrices
---
### Symmetric matrices 

A matrix is called symmetric if A=AT.

### Diagonalizable matrices 

If your transformation (matrix) has lots of eigenvectors, enough so you can choose a set of vectors that spans a full space, then you could change the basis of your coordinate system to these eigenvectors. These new basis vectors are called "Eigenbasis". This has as a nice consequence of guaranteeing that the new transformation matrix is always a diagonal matrix.

In this new diagonal matrix, all the columns represent eigenvectors with the values in the diagonal as their eigen values.

For example, let's say A is the new change of basis matrix, with eigenvectors as its columns. Then our transformation matrix D can be factorized in the new space as A−1DA, where D is guaranteed to be a diagonal matrix of eigen vectors.

Diagonal matrix are much easier to handle as the matrix multiplication simply boils down to raising the power of basis vectors to N number of times.

### Orthogonally diagonalizable matrices 

An n×n matrix A is said to be orthogonally diagonalizable when an orthogonal matrix P can be found such that P−1AP=PTAP is a diagonal matrix.

## Matrix Factorizations 
---
The following are known as the big six matrix factorizations:

1. Cholesky Factorization
2. LU Factorization
3. QR Factorization
4. Schur Decomposition
5. [[Eigenvalue Decomposition]]
6. [[Singular Value Decomposition]]