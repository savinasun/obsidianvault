### [[dl_moc]]  
# **Entropy**

**Tags**::  
**Links**::

## Definition
---
The concept of **information entropy** (or **Shannon entropy**) was introduced by Claude Shannon in 1948 and has various applications physics, statistics and computer science. 
It quantifies the average level of uncertainty or information associated with a random variable $X$, we may denote its associated distribution as $p: \mathcal{X}\to [0,1]$, the entropy is defined as
$$
H(X) = \mathbb{E}_p \left[-\log p(X)\right],
$$
or we may define it as
$$
H(p) = \mathbb{E}_p\left[-log(p)\right].
$$


## Cross Entropy
---
[Cross entropy](https://en.wikipedia.org/wiki/Cross-entropy) between two probability distribution $p$ and $q$ over same underlying events measures the average number of bits needed to identify an event drawn from the set if a coding scheme optimized for $q$ instead of true distribution $p$ is used.

$$H(p,q)=− \mathbb{E}_p \left[log(q)\right]$$

Under discrete cases, it is
$$
H(p,q)= \Sigma p_i \log(qi)
$$

If the true probability density $p$ is equal to predicted density $q$, then cross entropy becomes entropy.

$$H(p,q)=H(p)+D_{KL}(p,q)$$
The amount of bits that differ cross entropy from entropy is called KL divergence.


## Kullback–Leibler Divergence (Relative Entropy)
---
KL divergence (or relative entropy) is a measure of how one probability distribution $q(x)$ is different from a reference probability distribution $p(x)$.

$$D_{KL}(q||p)=-\mathbb{E}_{q}\left[\log\frac{p(X)}{q(X)}\right] = -\int q(x)\left[\log\frac{p(x)}{q(x)}\right]$$

There are a few important properties about KL divergence:
1. **Asymmetry**: KL divergence is not a distance metric since it is asymmetric.$$D_{KL}(q||p) \neq D_{KL}(p||q)$$
2. **Relationship with Entropy / [[Cross Entropy]]**: $$H(p,q)=H(p)+D_{KL}(p,q)$$
3. **Relationship with Information Gain**: KL divergence of $p$ relative to $q$, i.e. $D_{KL} (p||q)$, could be deemed as information gain achieved changing from $q$ to $p$.

### Forward and Backward KL

![[entropy_01.png]]

1. **Forward KL**: $KL(p||q) = \int p\log \frac{p}{q}dz$
	
	+ **Zero Avoiding**: the model will try to avoid placing zero probability mass anywhere where $p>0$ since this would lead to an exploding KL. This means it is safer to place non-zero $q$ anywhere where plausibly $p>0$ which leads to overestimation.
	
2. **Backward KL**: $KL(q||p)=\int q\log\frac{q}{p}dz$
	
	+ **Zero Forcing**: the model will try to avoid situations where $q≈0$ and $p>0$, thus it is safer to choose a mode and underestimate variance, rather than overestimate and risking to overshoot.

### Relationships to Entropy and Cross-Entropy

Minimizing KL divergence between two distributions corresponds to minimizing the cross entropy between the distributions, and can be used equivalently when the entropy one distribution is stationary i.e. training data.




