### [[dl_moc]]  
# **Spectral Analysis**

**Tags**::  
**Links**::

## Spectrum Norm
---
The [natural norm](https://mathworld.wolfram.com/NaturalNorm.html) induced by the [L2-norm](https://mathworld.wolfram.com/L2-Norm.html). Let $A^H$ be the [conjugate transpose](https://mathworld.wolfram.com/ConjugateTranspose.html) of the [square matrix](https://mathworld.wolfram.com/SquareMatrix.html) ![A], so that ![(a_(ij))^(H)=(a^__(ji))], then the spectral norm is defined as the [square root](https://mathworld.wolfram.com/SquareRoot.html) of the maximum [eigenvalue] of ![A^(H)A], i.e.,

|                                                                                     |      |                                          |     |
| ----------------------------------------------------------------------------------- | ---- | ---------------------------------------- | --- |
| ![\|A\|_2](https://mathworld.wolfram.com/images/equations/SpectralNorm/Inline5.svg) | ![=] | ![(maximum eigenvalue of A^(H)A)^(1/2)]  | (1) |
| ![](https://mathworld.wolfram.com/images/equations/SpectralNorm/Inline8.svg)        | ![=] | ![max_(\|x\|_2!=0)(\|Ax\|_2)/(\|x\|_2),] | (2) |

This [matrix norm](https://mathworld.wolfram.com/MatrixNorm.html) is implemented as [Norm](http://reference.wolfram.com/language/ref/Norm.html)[_m_, 2].


## Spectral Theorem
---
