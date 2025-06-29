### [[dl_moc]]  
## **Group Relative Policy Optimization**

**Tags**:: 
**Links**:: [@deepseek-ai2025][deepseek-ai2025]

---
### Loss Function of GRPO Algorithm

+ **Goal**: to maximize rewards for agents received over time

+ **Requires**: Diversity in responses and diversity of rewards

+ **Algorithm**:
	![[GRPO_algorithm.png]]
	
+ **An on-policy algorithm**: this means the agent learns from its own experiences, updating the policy it uses to act based on the outcomes of those actions.

+ **Loss Function**: 
	![[grpo_loss_function.png]]
	
+ **Advantage**: Normalized rewards scores
$$A_i = \frac{r_i - mean(r_1, r_2, \dots, r_G)}{std(r_1, r_2, \dots, r_G)}, $$
	where $A_i$ is the advantage of the $i$-th completion and $r_i$ is the reward assigned to $i$-th completion
	
+ **Components**:
	1. Ratio of Token Probability Distributions of Reference Model (no adapter) and Updated Model (with adapter)
	2. Advantages
	3. Ratio Clipping
	4. [[KL Divergence]]: make sure the new model does not deviate too much from the baseline model during update
		![[GRPO_kl_divergence.png]]




