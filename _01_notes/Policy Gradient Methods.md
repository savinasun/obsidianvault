### [[dl_moc]]
# **Policy Gradient Methods**

**Tags**:: #rl #llm
**Links**:: [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] · [[PPO]] · [[DPO]] · [[GRPO]] · [[RLHF]] · [[Behavior Cloning]] · [[KL Divergence]] · [[LLMs]] · [[LLaMA]] · [[CoT Prompting]]


## Proximal Policy Optimization (PPO)
---
**Proximal Policy Optimization (PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is a first-order policy-gradient method that keeps each update close to the data-collecting policy by clipping the probability ratio, so one on-policy batch can be reused for several epochs of minibatch SGD. It is the RL stage of classic RLHF ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) for InstructGPT, [Touvron et al., 2023](https://arxiv.org/abs/2307.09288) for Llama 2-Chat). Full treatment — GAE, the combined objective, the RLHF objective with per-token KL, the lineage from Christiano et al. 2017 to Llama 2-Chat, and why the critic is costly for LLMs — in [[PPO]].

**Probability Ratio**: the ratio of the probability of taking an action under the current policy $\pi_{\theta}$ to that under the old (data-collecting) policy $\pi_{\theta_{old}}$, so that $r_t(\theta_{old}) = 1$
$$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
it measures how much the policy has changed between updates

**Advantage Function**: advantage function $A^{\pi}(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action $a$ in state $s$, over randomly selecting an action according to $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after. Mathematically, the advantage function is defined by
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

PPO estimates it as $\hat A_t$ with generalized advantage estimation: $\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}$ with $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ (see [[PPO#Clipped Surrogate Objective]]).

**PPO (Clipping)**: PPO uses a clipping function that modifies the objective function to restrict the size of policy updates ($\epsilon = 0.2$ in the paper, *PPO Eq. 7*):
$$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]$$
The $\min$ makes this a pessimistic lower bound on $r_t\hat A_t$: the ratio is clipped only when moving it further would *improve* the objective, so a step in the wrong direction is never clipped.

**PPO in LLMs**: the LLM is the policy ($\pi_\theta(y_t|x,y_{<t})$ over next tokens), the reward-model score arrives at the last token, and a per-token KL penalty $-\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ to the frozen SFT model $\pi_\text{ref}$ is folded into the reward — a different KL from the trust-region one that the clip approximates. PPO therefore holds four models (policy, critic, reference, reward model); [[DPO]] removes the reward model and the RL loop, and [[GRPO]] removes the critic. See [[PPO#PPO in LLM Post-Training (RLHF)]].


## Direct Preference Optimization (DPO)
---
**Direct Preference Optimization (DPO)** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) optimizes the same KL-regularized objective as RLHF without a reward model, a critic, or sampling during training: the reward is reparameterized through the policy, $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$, and the Bradley–Terry preference loss over pairs $(x,y_w,y_l)$ (winner, loser) becomes a classification loss on the policy (*DPO Eq. 7*):
$$\mathcal{L}_\text{DPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big]$$
Its gradient raises the likelihood of $y_w$ and lowers that of $y_l$, weighted by $\sigma(\hat r_\theta(y_l)-\hat r_\theta(y_w))$, which is larger when the implicit reward mis-orders the pair. DPO trains offline on fixed preference data, and it replaced the PPO stage in recipes such as Zephyr and Llama 3 (which found it cheaper at scale, and better on instruction-following benchmarks such as IFEval, than the PPO it tried). Derivation, gradient analysis, the paper's results, and LLM adoption: [[DPO]].


## Group Relative Policy Optimization (GRPO)
---
**Group Relative Policy Optimization (GRPO)** ([Shao et al., 2024](https://arxiv.org/abs/2402.03300), DeepSeekMath; [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948), DeepSeek-R1) is PPO without the critic: for each prompt $q$ it samples a group of $G$ outputs $\{o_i\}$ from $\pi_{\theta_{old}}$ and uses the group's reward statistics as the baseline — [[#Reinforce with Baseline Algorithm|REINFORCE with a baseline]] whose baseline is the group mean, wrapped in PPO's clipped ratio and a KL term to the reference model. Notation (DeepSeek papers): prompt $q$ and outputs $o_i$ (= $x$, $y_i$ in [[PPO]] and [[DPO]]); $\rho_i$ is PPO's ratio, renamed because $r_i$ is the reward here.
$$A_i = \frac{r_i - mean(r_1, r_2, \dots, r_G)}{std(r_1, r_2, \dots, r_G)}, $$
where $A_i$ is the advantage of the $i$-th completion and $r_i$ is the reward assigned to the $i$-th completion; in outcome supervision every token of $o_i$ shares $\hat A_{i,t}=A_i$. The objective (*DeepSeek-R1 Eq. 1–2*, sequence-level shorthand) is
$$\mathcal{J}_\text{GRPO}(\theta)=\mathbb{E}\Big[\frac1G\sum_{i=1}^G\Big(\min\big(\rho_iA_i,\ \text{clip}(\rho_i,1-\varepsilon,1+\varepsilon)A_i\big)-\beta\,\mathbb{D}_\text{KL}[\pi_\theta\|\pi_\text{ref}]\Big)\Big],\qquad \rho_i=\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$$
with the per-token estimator $\mathbb{D}_\text{KL}[\pi_\theta\|\pi_\text{ref}]=\frac{\pi_\text{ref}}{\pi_\theta}-\log\frac{\pi_\text{ref}}{\pi_\theta}-1$, added to the loss rather than to the reward as in PPO. GRPO is agnostic to the reward source: DeepSeekMath-RL used a trained reward model, while DeepSeek-R1-Zero used rule-based verifiable rewards on a base model with no SFT. Token-level form, KL-estimator analysis, the PPO comparison, hyperparameters, and known issues (Dr. GRPO, DAPO): [[GRPO]].


## REINFORCE and Baselines
---
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
			$\theta \leftarrow \theta + \alpha^\theta \cdot \gamma^t \delta \cdot \frac{\nabla \pi(a_t^m|s_t^m, \theta)}{\pi(a_t^m|s_t^m, \theta)}$ or $\theta \leftarrow \theta + \alpha^\theta \cdot \gamma^t \delta \cdot \nabla \ln \pi(a_t^m|s_t^m, \theta)$
	

## PPO vs DPO vs GRPO
---

| Aspect | [[PPO]] (RLHF) | [[DPO]] | [[GRPO]] |
|---|---|---|---|
| Original paper | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347); LLM form: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) | [Shao et al., 2024](https://arxiv.org/abs/2402.03300); at scale: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948) |
| Training data | on-policy rollouts from $\pi_\theta$ | fixed preference pairs $(x,y_w,y_l)$, off-policy | on-policy: $G$ rollouts per prompt |
| Reward signal | learned reward model, scalar at the last token | implicit, $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$, from pairwise labels | reward model or rule-based verifier |
| Credit assignment | GAE with a learned critic | sequence-level; pairs weighted by $\sigma(\hat r_\theta(y_l)-\hat r_\theta(y_w))$ | group-normalized outcome reward, same for every token (or process-level) |
| Models held in training | policy, critic, reference, reward model | policy, reference | policy, reference, reward model (none for rule-based rewards) |
| KL to reference | per-token penalty in the reward | implicit, through $\beta$ | explicit $k_3$ term in the loss |
| Used for | InstructGPT, Llama 2-Chat | Zephyr, Llama 3 | DeepSeekMath-RL, DeepSeek-R1(-Zero) |
| Known weak spots | critic cost; sensitive to GAE $\lambda$ | pairs go stale as the policy drifts; needs stabilizers (Llama 3 masks format tokens and adds an NLL term) | needs reward variance within each group; length and difficulty biases |

**Unified view.** DeepSeekMath §5.2.1 writes SFT, RFT, DPO, PPO, and GRPO as one gradient, $\nabla_\theta\mathcal{J}=\mathbb{E}_{(q,o)\sim\mathcal D}\big[\frac1{|o|}\sum_t GC(q,o,t)\,\nabla_\theta\log\pi_\theta(o_t|q,o_{<t})\big]$, where the methods differ only in the *data source*, the *reward*, and the **gradient coefficient** $GC$ — how strongly each token is reinforced or penalized:

| Method | Data source | Reward | Gradient coefficient $GC$ |
|---|---|---|---|
| SFT | SFT data | – | $1$ |
| RFT | $o\sim\pi_\text{sft}$ | rule | $\mathbb{I}(o)$ (1 if the answer is correct) |
| DPO | $o^+,o^-\sim\pi_\text{sft}$ | rule / human preference | $\sigma\big(\beta\log\frac{\pi_\theta(o^-_t)}{\pi_\text{ref}(o^-_t)}-\beta\log\frac{\pi_\theta(o^+_t)}{\pi_\text{ref}(o^+_t)}\big)$, positive on $o^+$ and negative on $o^-$ |
| Online RFT | $o\sim\pi_\theta$ | rule | $\mathbb{I}(o)$ |
| PPO | $o\sim\pi_\theta$ | reward model | $A_t$ (GAE) |
| GRPO | $\{o_i\}\sim\pi_\theta$ | reward model | $\hat A_{i,t}+\beta\big(\frac{\pi_\text{ref}(o_{i,t})}{\pi_\theta(o_{i,t})}-1\big)$ |

DeepSeekMath's experiments with this lens (on DeepSeekMath-Instruct 1.3B): online sampling beats offline (Online RFT > RFT); graded coefficients beat binary ones (GRPO > Online RFT, since RFT never penalizes wrong answers and reinforces every correct one equally); and step-level coefficients beat outcome-level ones (GRPO with process supervision > with outcome supervision).

In the open recipes cited here, human-preference chat alignment uses PPO (InstructGPT, Llama 2-Chat) or DPO (Zephyr, Llama 3), and verifiable-reward reasoning RL uses GRPO (DeepSeek-R1).

## Relationship to Other Notes
---
- [[PPO]] · [[DPO]] · [[GRPO]] — one note per algorithm; this note keeps a digest of each, the REINFORCE foundations they build on, and the cross-method comparison and unified-gradient view.
- [[RLHF]] — the three-stage pipeline (SFT → reward model → PPO) and its reward-hacking and KL discussion; [[PPO]], [[DPO]], and [[GRPO]] hold the algorithms behind that third stage, its replacement, and the critic-free alternative.
- [[KL Divergence]] — the reference-model penalty $\beta\,D_\text{KL}(\pi_\theta\|\pi_\text{ref})$ in RLHF and GRPO, the $k_3$ estimator, and TRPO's KL trust region that PPO's clip approximates.
- [[Behavior Cloning]] — SFT is the usual starting point and frozen reference $\pi_\text{ref}$ in RLHF-style PPO and in DPO (R1-Zero skips it and starts from the base model), and is the $GC=1$ special case in the unified gradient view.
- [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] — the on-policy actor-critic setting PPO comes from; the REINFORCE sections above are the policy-gradient foundation GRPO's group baseline builds on.
- [[LLMs]] · [[LLaMA]] — where these methods are applied: Llama 2-Chat (rejection sampling + PPO) and Llama 3 (rejection sampling + DPO, PPO explored and set aside).
- [[CoT Prompting]] — DeepSeek-R1-Zero uses GRPO with verifiable rewards to reinforce reasoning traces that reach correct answers, without human-labeled thought chains.
- [[Elo Score]] — the Bradley–Terry pairwise-preference model behind the reward-model loss that DPO reparameterizes.
