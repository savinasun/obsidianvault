### [[dl_moc]]  
# **Policy Gradient Methods**

**Tags**:: #rl
**Links**:: [David Silver RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

Agent
Policy: gives the probability of taking actions in any state
Goal: determines the policy function to achieve high reward for whatever the environment will be


## Proximal Policy Optimization (PPO)
---
**Probability Ratio**: the probability of taking an action in the current policy $\pi_{\theta}$ to that of the old policy $\pi_{\theta_{old}}$
$$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
it measures how much the policy has changed between updates


**Advantage Function**: advantage function $A^{\pi}(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action $a$ in state $s$, over randomly selecting an action according to $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after. Mathematically, the advantage function is defined by
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

**PPO (Clipping)**: PPO uses a clipping function that modifies the objective function to restrict the size of policy updates:
$$L^{CLIP}(θ)=E_t​[min(r_t​(θ)A_t​,clip(r_t​(θ), 1−ϵ, 1+ϵ)A_t​)]$$

![[RLHF.png]]




## Direct Policy Optimization (DPO)
---

![[policy_gradient_methods_01.png]]



## Group Relative Policy Optimization (GRPO)
---
### Loss Function of GRPO Algorithm

+ **Goal**: to maximize rewards for agents received over time

+ **Requires**: Diversity in responses and diversity of rewards

+ **Algorithm**:
	![[policy_gradient_methods_02.png]]
	
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
		![[policy_gradient_methods_03.png]]



### Reinforce Algorithm
+ Functional form of the policy: $\pi(a|s, \mathbf{\theta})$
+ Initialization of parameter $\mathbf\theta$
+ Step size $\alpha$

	For episode $m=1,\dots,M$:
		sample trajectory $(s_0^m, a_0^m, r_1^m, s_1^m, \dots, s_{T_m-1}^m, a_{T_m-1}^m, r_{T_m}^m)$
		For $t=0, \cdots, T_m -1$:
			$g_t^m \leftarrow r_{t+1}^m + \gamma r_{t+2}^m + \cdots$
			$\theta \leftarrow \theta + \alpha \cdot \gamma^t g_t^m \cdot \frac{\nabla \pi(a_t^m|s_t^m, \theta)}{\pi(a_t^m|s_t^m, \theta)}$ or $\theta \leftarrow \theta + \alpha \cdot \gamma^t g_t^m \cdot \nabla \ln \pi(a_t^m|s_t^m, \theta)$
	

### Reinforce with Baseline Algorithm
+ Functional form of the policy: $\pi(a|s, \mathbf{\theta})$
+ Functional form of the value: $\hat{v}(s, \omega)$
+ Initialization of parameter $\mathbf\theta, \omega$
+ Step size $\alpha^\theta, \alpha^\omega$

	For episode $m=1,\dots,M$:
		sample trajectory $(s_0^m, a_0^m, r_1^m, s_1^m, \dots, s_{T_m-1}^m, a_{T_m-1}^m, r_{T_m}^m)$
		For $t=0, \cdots, T_m -1$:
			$g_t^m \leftarrow r_{t+1}^m + \gamma r_{t+2}^m + \cdots$
			$\delta \leftarrow g_t^m - \hat{v}(s_t^m, \omega)$
			$\omega \leftarrow \omega+ \alpha^\omega \delta \nabla \hat{v} (s_t^m, \omega)$
			$\theta \leftarrow \theta + \alpha \cdot \gamma^t g_t^m \cdot \frac{\nabla \pi(a_t^m|s_t^m, \theta)}{\pi(a_t^m|s_t^m, \theta)}$ or $\theta \leftarrow \theta + \alpha \cdot \gamma^t g_t^m \cdot \nabla \ln \pi(a_t^m|s_t^m, \theta)$
	
