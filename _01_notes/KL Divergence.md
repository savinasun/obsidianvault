### [[dl_moc]]  
## **Kullback–Leibler Divergence**

**Tags**::  #dl_theo
**Links**::

---
### Definition

KL divergence or relative entropy is a measure of how one probability distribution $q(x)$ is different from a reference probability distribution $p(x)$.

$$D_{KL}(q||p)=-\mathbb{E}_{q}\left[\log\frac{p(X)}{q(X)}\right] = -\int q(x)\left[\log\frac{p(x)}{q(x)}\right]$$

### Forward and Backward KL

![[KL_Divergence_forward_vs_backward.png]]

+ **Forward KL**: $KL(p||q) = \int p\log \frac{p}{q}dz$
	
	+ **Zero Avoiding**: the model will try to avoid placing zero probability mass anywhere where $p>0$ since this would lead to an exploding KL. This means it is safer to place non-zero $q$ anywhere where plausibly $p>0$ which leads to overestimation.
	
+ **Backward KL**: $KL(q||p)=\int q\log\frac{q}{p}dz$
	
	+ **Zero Forcing**: the model will try to avoid situations where $q≈0$ and $p>0$, thus it is safer to choose a mode and underestimate variance, rather than overestimate and risking to overshoot.

### Relationships to Entropy and Cross-Entropy

Minimizing KL divergence between two distributions corresponds to minimizing the [[Cross Entropy]] between the distributions, and can be used equivalently when the entropy one distribution is stationary i.e. training data.

---
## References
