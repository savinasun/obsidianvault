### [[dl_moc]]  
## **Cross Entropy**

**Tags**::  #dl_theo
**Links**::

---
### Definition

Cross entropy between two probability distribution $p$ and $q$ over same underlying events measures the average number of bits needed to identify an event drawn from the set if a coding scheme optimized for $q$ instead of true distribution $p$ is used.

$$H(p,q)=− \Sigma p_i \log(qi)$$

If the true probability density $p$ is equal to predicted density $q$, then cross entropy becomes entropy.

$$H(p,q)=H(p)+D_{KL}(p,q)$$

The amount of bits that differ cross entropy from entropy is called [[KL Divergence|KL divergence]].

