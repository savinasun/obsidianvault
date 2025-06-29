### [[dl_moc]]  
## **Proximal Policy Optimization**
**Tags**::  #llm/gpt 
**Links**::

**Probability Ratio**: the probability of taking an action in the current policy $\pi_{\theta}$ to that of the old policy $\pi_{\theta_{old}}$
$$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
it measures how much the policy has changed between updates


**Advantage Function**: advantage function $A^{\pi}(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action $a$ in state $s$, over randomly selecting an action according to $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after. Mathematically, the advantage function is defined by
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

**PPO (Clipping)**: PPO uses a clipping function that modifies the objective function to restrict the size of policy updates:
$$L^{CLIP}(θ)=E_t​[min(r_t​(θ)A_t​,clip(r_t​(θ), 1−ϵ, 1+ϵ)A_t​)]$$

![[RLHF.png]]
