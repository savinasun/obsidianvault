### [[dl_moc]]  
## **Policy Gradient Methods**
**Tags**::  
**Links**:: [David Silver RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

Proximal Policy Gradient

Policy: gives the probability of taking actions in any state
Goal: determines the policy function to achieve high reward for whatever the environment will be


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
	
