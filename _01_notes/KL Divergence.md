### [[dl_moc]]  
# **Kullback-Leibler Divergence*

**Tags**:: #math #dl_theo
**Links**::

## KL Divergence
---
**KL divergence** (or **relative entropy**) is a measure of how one probability distribution $q(x)$ is different from a reference probability distribution $p(x)$.

$$D_{KL}(q||p)=-\mathbb{E}_{q}\left[\log\frac{p(X)}{q(X)}\right] = -\int q(x)\left[\log\frac{p(x)}{q(x)}\right]\geq 0$$
Therefore KL divergence is **always non-negative**.

There are a few important properties about KL divergence:
1. **Asymmetry**: KL divergence is not a distance metric since it is asymmetric.$$D_{KL}(q||p) \neq D_{KL}(p||q)$$
2. **Relationship with [[Entropy]] / Cross Entropy**: $$H(p,q)=H(p)+D_{KL}(p,q)$$
3. **Relationship with Information Gain**: KL divergence of $p$ relative to $q$, i.e. $D_{KL} (p||q)$, could be deemed as information gain achieved changing from $q$ to $p$.

### Forward KL and Backward KL

![[entropy_01.png]]

1. **Forward KL**: $KL(p||q) = \int p\log \frac{p}{q}dz$
	
	+ **Zero Avoiding**: the model will try to avoid placing zero probability mass anywhere where $p>0$ since this would lead to an exploding KL. This means it is safer to place non-zero $q$ anywhere where plausibly $p>0$ which leads to overestimation.
	
2. **Backward KL**: $KL(q||p)=\int q\log\frac{q}{p}dz$
	
	+ **Zero Forcing**: the model will try to avoid situations where $q≈0$ and $p>0$, thus it is safer to choose a mode and underestimate variance, rather than overestimate and risking to overshoot.

### Relationships to Entropy and Cross-Entropy

Minimizing KL divergence between two distributions corresponds to minimizing the cross entropy between the distributions, and can be used equivalently when the entropy one distribution is stationary i.e. training data.




