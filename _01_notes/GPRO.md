### [[dl_moc]]  
## **Group Relative Policy Optimization**

**Tags**::  
**Links**:: [@deepseek-ai2025][deepseek-ai2025]

---

![[Pasted image 20250623160931.png]]

+ **An on-policy algorithm**: this means the agent learns from its own experiences, updating the policy it uses to act based on the outcomes of those actions.
+ Reward Function: 
+ Agent Goal: to maximize rewards received over time
+ Requires: Diversity in responses and diversity of rewards
+ **Advantage**: Normalized rewards scores
$$A_i = \frac{r_i - mean(r_1, r_2, \dots, r_G)}{std(r_1, r_2, \dots, r_G)}, $$
	where $A_i$ is the advantage of the $i$-th completion and $r_i$ is the reward assigned to $i$-th completion

 

### Loss Function of GRPO Algorithm
![[_02_references/deepseek-ai2025/image-5-x65-y74.png]]

	1. Ratio of Token Probability Distributions of Reference Model (no adapter) and Updated Model (with adapter)
	2. Advantages
	3. Ratio Clipping
	4. KL-Divergence: make sure the new model does not deviate too much from the baseline model during update


![[_04_images/GRPO_kl_divergence.png|500]]
