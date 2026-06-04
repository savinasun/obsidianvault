### [[dl_moc]]  
# **Entropy**

**Tags**::  
**Links**::

## Definition
---
The concept of **information entropy** (or **Shannon entropy**) was introduced by Claude Shannon in 1948 in his master thesis and has various applications physics, statistics and computer science. 

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
The amount of bits that differ cross entropy from entropy is called [[KL Divergence|KL divergence]].