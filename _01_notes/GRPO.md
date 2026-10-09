### [[dl_moc]]  
# **Group Relative Policy Optimization (GRPO)**

**Tags**:: #rl #llm
**Links**:: [[Policy Gradient Methods]] · [[PPO]] · [[DPO]] · [[RLHF]] · [[KL Divergence]] · [[CoT Prompting]] · [[LLMs]]

**Group Relative Policy Optimization (GRPO)** is a PPO variant that drops the learned critic: for each prompt it samples a group of $G$ responses and uses the group's own reward statistics as the baseline, so a response's advantage is how much better it scored than its siblings ([Shao et al., 2024](https://arxiv.org/abs/2402.03300), DeepSeekMath). Read against the REINFORCE sections of [[Policy Gradient Methods]], it is [[Policy Gradient Methods#Reinforce with Baseline Algorithm|REINFORCE with a baseline]] whose baseline is the group-mean reward (additionally divided by the group std), wrapped in PPO's clipped ratio and an explicit KL term to a reference model. DeepSeekMath introduced it for math RL — DeepSeekMath-RL 7B improves GSM8K from 82.9% to 88.2% and MATH from 46.8% to 51.7% over its SFT starting point — and [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948) adopted it to train DeepSeek-R1-Zero and DeepSeek-R1, using rule-based, verifiable rewards on reasoning tasks (see [[CoT Prompting]]); it has since become a common base for open reasoning-RL recipes (see [[#Known Issues and Extensions]]).

Primary sources: DeepSeekMath §4 and App. A.1 (algorithm, gradient analysis); DeepSeek-R1 §2 and App. A.3 (large-scale use, comparison with PPO).

**Notation**: these papers write the prompt as $q$ and the sampled outputs as $o_i$ (= $x$, $y_i$ in [[PPO]] and [[DPO]]). The formulas below keep their symbols so they match the screenshots, and PPO's ratio $r_t(\theta)$ is renamed $\rho_{i,t}$ because $r_i$ denotes the reward here.

## Loss Function of GRPO Algorithm
---

+ **Goal**: to maximize rewards for agents received over time

+ **Requires**: Diversity in responses and diversity of rewards — if all $G$ rewards in a group are equal, every advantage is $0$ and the prompt contributes no policy gradient ([DAPO](https://arxiv.org/abs/2503.14476) filters such prompts out with *dynamic sampling*)

+ **Algorithm**:
	![[policy_gradient_methods_02.png]]
	One iteration (DeepSeekMath Algorithm 1): sample $G$ outputs per prompt from $\pi_{\theta_{old}}$, score them, compute group-relative advantages, and maximize the objective below. In the *iterative* variant the reference model is reset to the current policy, and the reward model is retrained on fresh samples, each iteration.
	
+ **An on-policy algorithm**: this means the agent learns from its own experiences, updating the policy it uses to act based on the outcomes of those actions. In DeepSeekMath-RL the policy takes a single update per exploration stage, so $\pi_{\theta_{old}}=\pi_\theta$ at update time and the clip never binds. DeepSeek-R1-Zero splits each rollout of 8,192 outputs into 16 minibatches for one inner epoch, so only the first minibatch is exactly on-policy; and R1's first RL stage sets $\varepsilon=10$, a range of $[-9, 11]$ on an always-positive ratio, so clipping is almost never active.

+ **Loss Function**: 
	![[policy_gradient_grpo_loss.png]]
	*DeepSeek-R1 Eq. (1)–(3): the sequence-level shorthand of the token-level objective given below.*
	
+ **Advantage**: Normalized rewards scores
$$A_i = \frac{r_i - mean(r_1, r_2, \dots, r_G)}{std(r_1, r_2, \dots, r_G)}, $$
	where $A_i$ is the advantage of the $i$-th completion and $r_i$ is the reward assigned to $i$-th completion
	In **outcome supervision** (used by R1) every token of $o_i$ gets the same $\hat A_{i,t}=A_i$. **Process supervision** scores each reasoning step with a process reward model, normalizes over all steps in the group, and sets $\hat A_{i,t}$ to the sum of the normalized rewards of the steps ending at or after token $t$ (DeepSeekMath §4.1.3). Group-relative normalization suits reward models, which are trained on comparisons between outputs for the same question.
	
+ **Components**:
	1. Ratio of Token Probability Distributions of the Updated Policy $\pi_\theta$ and the Old Policy $\pi_{\theta_{old}}$ (the frozen reference model enters only through the KL term)
	2. Advantages
	3. Ratio Clipping
	4. [[KL Divergence]]: make sure the new model does not deviate too much from the baseline model during update
		![[policy_gradient_methods_03.png]]

**Token-level form.** The screenshot above is DeepSeek-R1's sequence-level shorthand ($\pi_\theta(o_i|q)$); DeepSeekMath Eq. 3 applies ratio, clip, and KL per token and averages over each response's length:
$$\mathcal{J}_\text{GRPO}(\theta)=\mathbb{E}_{q,\,\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}}\ \frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big(\rho_{i,t}\hat A_{i,t},\ \text{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)-\beta\,\mathbb{D}_\text{KL}[\pi_\theta\|\pi_\text{ref}]\Big\},\qquad \rho_{i,t}=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$$

**KL estimator.** GRPO does not fold the KL penalty into the reward as PPO does; it adds a per-token KL estimate directly to the loss, which leaves $\hat A_{i,t}$ untouched. The estimator is Schulman's unbiased, always non-negative "$k_3$" ([Schulman, 2020](http://joschu.net/blog/kl-approx.html); DeepSeekMath Eq. 4), with $\Delta=\log\pi_\theta-\log\pi_\text{ref}$ as on the plot's x-axis:
$$\mathbb{D}_\text{KL}[\pi_\theta\|\pi_\text{ref}]=\frac{\pi_\text{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_\text{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-1=e^{-\Delta}+\Delta-1$$
It estimates $D_\text{KL}(\pi_\theta\|\pi_\text{ref})$ even though the plot labels it $KL(\pi_\text{ref}\|\pi)$: with $u=\pi_\text{ref}/\pi_\theta$ and tokens drawn from $\pi_\theta$, $\mathbb{E}[u-1]=0$ and $\mathbb{E}[-\log u]=D_\text{KL}(\pi_\theta\|\pi_\text{ref})$. The penalty grows exponentially when the policy becomes *less* likely than the reference ($\Delta<0$, "come back to the reference model") but only linearly when it becomes more confident ($\Delta>0$).

One GRPO iteration (sketch):

```python
ys, logp_old = policy.sample(x, n=G)                 # G responses per prompt
r = reward(x, ys)                                    # (G,) rule-based check or reward-model score
A = ((r - r.mean()) / (r.std() + 1e-6))[:, None]     # group-relative advantage, shared by all tokens of y_i
for _ in range(mu):
    logp = policy.logprobs(x, ys)                    # (G, T) token log-probs
    ratio = (logp - logp_old).exp()
    d = ref_logp - logp                              # log(pi_ref / pi_theta)
    kl = d.exp() - d - 1                             # k3 estimator
    obj = torch.min(ratio * A, ratio.clamp(1 - eps, 1 + eps) * A) - beta * kl
    loss = -((obj * mask).sum(1) / mask.sum(1)).mean()   # 1/|o_i| per response, then mean over the group
```

## GRPO vs PPO
---

GRPO keeps [[PPO]]'s clipped surrogate and changes where the advantage and the KL come from ([DeepSeekMath §4.1.1](https://arxiv.org/abs/2402.03300); [DeepSeek-R1 App. A.3](https://arxiv.org/abs/2501.12948)):

+ **Critic removed.** PPO's $\hat A_t$ comes from GAE over a learned value model about the size of the policy; GRPO's comes from the group of sampled responses, saving that model's memory and compute. R1 argues that a token-level critic is especially ill-suited to long chain-of-thought (see [[PPO#PPO in LLM Post-Training (RLHF)|the PPO note]]).
+ **KL moved into the loss.** PPO adds a per-token KL penalty to the reward, so what is optimized is the *cumulative* KL; the R1 authors argue this can implicitly penalize response length and keep long reasoning from emerging. R1 also resets the reference model to the latest policy every 400 steps, since over thousands of steps the policy legitimately drifts far from its initialization.
+ **Evidence.** On MATH with DeepSeek-Coder-V2-Lite (16B MoE, 2.4B active), PPO with the common GAE default $\lambda=0.95$ is considerably worse than GRPO, and tuning to $\lambda=1.0$ brings it close. R1's conclusion: PPO can match GRPO when tuned, at the price of extra hyperparameter search and the critic's memory.

## GRPO in LLM Training
---

| Setting | DeepSeekMath-RL 7B | DeepSeek-R1-Zero (latest arXiv version) |
|---|---|---|
| Starting model | DeepSeekMath-Instruct 7B (SFT) | DeepSeek-V3-Base, no SFT |
| Reward | learned reward model (outcome or process variant) | rule-based: accuracy + format |
| Group size $G$ | 64 | 16 |
| Learning rate | $10^{-6}$ | $3\times10^{-6}$ |
| KL coefficient $\beta$ | 0.04 | 0.001 |
| Reference model | reset to the current policy each iteration | replaced by the latest policy every 400 steps |
| Max response length | 1,024 | 32,768, then 65,536 after step 8.2k |
| Headline result | GSM8K 82.9→88.2%, MATH 46.8→51.7% | AIME 2024 pass@1 15.6→77.9% (v1 reported 71.0%) |

**Reward source is a separate choice.** GRPO only needs a scalar score per sampled response, so it is agnostic to where that score comes from: DeepSeekMath used a trained reward model, while R1-Zero deliberately avoids neural reward models for reasoning because they are susceptible to reward hacking in large-scale RL and need retraining. Instead an *accuracy* reward (final answer checked by rule, e.g. inside a box; code checked by a compiler against test cases) and a *format* reward (reasoning inside `<think>` tags) are added with equal weight. From the base model with this signal alone, R1-Zero's AIME 2024 pass@1 rises from 15.6% to 77.9% (86.7% with self-consistency decoding), and longer responses with self-verification and reflection emerge without being taught (DeepSeek-R1 §2.2–2.3). Group sampling is what turns these sparse correctness scores into usable advantages without a critic.

## Known Issues and Extensions
---

+ **Length and difficulty bias** ([Liu et al., 2025](https://arxiv.org/abs/2503.20783), "Dr. GRPO"): dividing by $|o_i|$ gives short correct responses larger gradients and penalizes long incorrect ones less, favoring length among wrong answers; dividing the centered reward by the group std up-weights questions that are too easy or too hard (rewards nearly all 1 or all 0). Dr. GRPO removes both normalizations, which curbs the artificial growth in response length (especially of incorrect outputs) while maintaining reasoning performance; their minimalist R1-Zero recipe built on it reaches 43.3% on AIME 2024 with a 7B base model.
+ **DAPO** ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)): an open recipe reaching 50 points on AIME 2024 with Qwen2.5-32B. Four changes to GRPO: *Clip-Higher* (decoupled $\varepsilon_\text{low}$/$\varepsilon_\text{high}$ to avoid entropy collapse), *Dynamic Sampling* (over-sample and drop prompts whose group accuracy is 0 or 1), a *token-level* policy-gradient loss (normalize by the batch's total token count rather than per response), and *Overlong Reward Shaping*. It also drops the KL term, arguing that a long-CoT policy is meant to move far from its initialization.

## Relationship to Other Notes
---
- [[PPO]] — GRPO is PPO without the critic: it keeps the clipped ratio, replaces GAE with a group-relative advantage, and moves the reference-model KL from the reward into the loss.
- [[Policy Gradient Methods]] — the group-mean baseline makes GRPO a REINFORCE-with-baseline method, and the unified-gradient table places its gradient coefficient next to PPO's and DPO's.
- [[DPO]] — the other alternative to PPO: DPO is offline on preference pairs, GRPO stays online and scores sampled groups with a reward model or a verifier.
- [[RLHF]] — GRPO takes the place of the PPO stage; because it only needs a scalar score per response, that score can come from a reward model or from rule-based verification.
- [[KL Divergence]] — the $k_3$ per-token estimator of $D_\text{KL}(\pi_\theta\|\pi_\text{ref})$ and its asymmetric penalty shape.
- [[CoT Prompting]] — DeepSeek-R1-Zero uses GRPO with verifiable rewards to reinforce reasoning traces that reach correct answers, without human-labeled thought chains.
- [[LLMs]] — GRPO is the RL stage of reasoning-model post-training (DeepSeekMath, DeepSeek-R1).
