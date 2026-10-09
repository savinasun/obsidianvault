### [[dl_moc]]  
# **Proximal Policy Optimization (PPO)**

**Tags**:: #rl #llm
**Links**:: [[Policy Gradient Methods]] · [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] · [[RLHF]] · [[KL Divergence]] · [[DPO]] · [[GRPO]] · [[LLMs]] · [[LLaMA]]

**Proximal Policy Optimization (PPO)** is a first-order policy-gradient method that keeps every update close to the policy that collected the data by clipping the probability ratio, so one batch of on-policy samples can be reused for several epochs of minibatch SGD ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). It was designed to keep the stable improvement of TRPO's hard KL trust region ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) with only first-order optimization. PPO became the RL stage of classic RLHF: [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593) applied it to language models with a reward learned from human preferences, and [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) (InstructGPT) and [Touvron et al., 2023](https://arxiv.org/abs/2307.09288) (Llama 2-Chat) used it to build instruction-following chat models. [[DPO]] and [[GRPO]] are the two main responses to PPO's cost in LLM training, and [[Policy Gradient Methods]] compares all three.

Primary sources: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) for the algorithm; [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) (InstructGPT) for its LLM form.

## Clipped Surrogate Objective
---
**Probability Ratio**: the ratio of the probability of taking an action under the current policy $\pi_{\theta}$ to that under the old (data-collecting) policy $\pi_{\theta_{old}}$, so that $r_t(\theta_{old}) = 1$
$$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
it measures how much the policy has changed between updates

**Advantage Function**: advantage function $A^{\pi}(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action $a$ in state $s$, over randomly selecting an action according to $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after. Mathematically, the advantage function is defined by
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

In practice $A^\pi$ is unknown and PPO uses an estimate $\hat A_t$ built from a learned value function $V_\psi$ with **generalized advantage estimation** ([GAE, Schulman et al., 2015](https://arxiv.org/abs/1506.02438)); *PPO Eq. 11–12*:
$$\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$
($\lambda=1$ gives the discounted return minus a baseline, $\lambda=0$ the one-step TD error; the sum is truncated at the end of the rollout segment.)

**PPO (Clipping)**: PPO uses a clipping function that modifies the objective function to restrict the size of policy updates ($\epsilon = 0.2$ in the paper, *PPO Eq. 7*):
$$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]$$
The $\min$ makes $L^{CLIP}$ a pessimistic lower bound on the unclipped surrogate $L^{CPI}=\hat{\mathbb{E}}_t[r_t\hat A_t]$ (the two agree to first order at $\theta_{old}$): the ratio is clipped only when moving it further would *improve* the objective. For $\hat A_t>0$ the incentive to raise $r_t$ disappears above $1+\epsilon$; for $\hat A_t<0$ the incentive to lower it disappears below $1-\epsilon$; a step in the wrong direction is never clipped, so its gradient still pulls the policy back.

PPO's alternative, a KL *penalty* $-\beta\,\mathrm{KL}[\pi_{\theta_{old}},\pi_\theta]$ with fixed or adaptive $\beta$, scored lower than clipping in the paper's continuous-control comparison (best KL variant 0.74 vs. 0.82 for $\epsilon=0.2$; *PPO Table 1*). When policy and value share a network, the full objective adds a value loss and an entropy bonus (*PPO Eq. 9*):
$$L^{CLIP+VF+S}_t(\theta)=\hat{\mathbb{E}}_t\big[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)\big],\qquad L^{VF}_t=(V_\theta(s_t)-V_t^{\text{targ}})^2$$
Each iteration collects rollouts under $\pi_{\theta_{old}}$, computes $\hat A_t$, runs $K$ epochs of minibatch SGD on this objective, then sets $\theta_{old}\leftarrow\theta$ (*PPO Algorithm 1*) — on-policy, but with data reuse.

## PPO in LLM Post-Training (RLHF)
---
**The LLM as a policy.** Let $x$ be the prompt and $y=(y_1,\dots,y_T)$ the response ([[GRPO]] follows the DeepSeek papers in writing $q$ and $o$). The state is $s_t=(x,y_{<t})$, the action is the next token $a_t=y_t$, and $\pi_\theta(y_t|x,y_{<t})$ is simply the LLM's next-token distribution; the transition (append the token) is deterministic. A response is one episode, and the reward model scores only the finished response, so the reward is sparse and arrives at the last token. InstructGPT describes this as a *bandit* environment (one prompt → one response → one reward, episode ends), whereas [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325) run PPO with each BPE token as a time step; the per-token view is what the KL penalty and the value function below operate on.

**RLHF objective.** InstructGPT maximizes the learned reward minus a KL penalty to the frozen SFT model, optionally mixed with a pretraining loss to avoid regressions on public NLP benchmarks (the "PPO-ptx" variant; *InstructGPT Eq. 2*):
$$\mathcal{J}(\theta) = \mathbb{E}_{(x,y)\sim\pi_\theta}\Big[r_\phi(x,y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_\text{SFT}(y|x)}\Big] + \gamma_\text{ptx}\,\mathbb{E}_{x\sim\mathcal{D}_\text{pretrain}}\big[\log\pi_\theta(x)\big]$$
Ouyang et al. write $\phi$ for the policy and $\theta$ for the reward model — swapped here to match [[RLHF]] — and $\gamma$ for the pretraining coefficient, renamed $\gamma_\text{ptx}$ to avoid a clash with the discount $\gamma$. The KL term is applied **per token and folded into the reward**: $r_t=-\beta\log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_\text{ref}(y_t|x,y_{<t})}$, with the reward-model score added at the last token ($\pi_\text{ref}=\pi_\text{SFT}$; the per-token form is written out in DeepSeekMath Eq. 2). This is a *different* KL from PPO's trust-region penalty above: it anchors $\pi_\theta$ to the fixed reference to limit reward hacking, whereas the trust region compares $\pi_\theta$ with $\pi_{\theta_{old}}$ and is already handled by the clip.

![[rlhf_pipeline.png]]

One PPO iteration for an LLM (sketch):

```python
y, logp_old = policy.sample(x)                     # rollout under pi_old; log-probs are frozen
rew = -beta * (logp_old - ref.logprobs(x, y))      # per-token KL penalty to the reference
rew[:, -1] += reward_model(x, y)                   # sparse reward-model score on the last token
A, V_targ = gae(rew, critic(x, y), gamma=1.0, lam=0.95)
for _ in range(K):                                 # K epochs of minibatch SGD on the same rollouts
    ratio = (policy.logprobs(x, y) - logp_old).exp()
    pg = -torch.min(ratio * A, ratio.clamp(1 - eps, 1 + eps) * A).mean()
    loss = pg + c1 * (critic(x, y) - V_targ).pow(2).mean()
```

## Lineage to LLMs
---
RLHF with PPO was built up in steps, each reusing the previous one's recipe:

| Paper | Setting | What it established |
|---|---|---|
| [Christiano et al., 2017](https://arxiv.org/abs/1706.03741) | Atari, simulated robot locomotion | RL from pairwise human preferences between trajectory segments, with feedback on <1% of the agent's interactions |
| [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593) | LM fine-tuning: stylistic continuation, TL;DR and CNN/DM summarization | PPO against a learned reward with a KL penalty $\beta\log\frac{\pi}{\rho}$ to the pretrained LM $\rho$ |
| [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325) | Reddit TL;DR summarization | Summaries that outperform human references and much larger SFT-only models |
| [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) | InstructGPT on API prompts | The SFT → reward model → PPO(-ptx) recipe; labelers prefer the 1.3B InstructGPT over the 175B GPT-3 |
| [Touvron et al., 2023](https://arxiv.org/abs/2307.09288) | Llama 2-Chat, 7B–70B | PPO applied on top of a rejection-sampling checkpoint, with separate helpfulness and safety reward models |

**Practical details.** InstructGPT initializes the value function from the reward model and uses a single 6B reward model for policies of all sizes: a 175B reward model trained less stably, made a poorer value-function initialization, and greatly raised PPO's compute. Llama 2-Chat uses the safety reward model for prompts flagged as unsafe (or when the safety score is below 0.15) and the helpfulness model otherwise, whitens the logit of the final score so it balances against the KL term, and trains with clip threshold 0.2, batch size 512, and $\beta=0.01$ (7B/13B) or $0.005$ (34B/70B).

## Why PPO Is Heavy for LLMs
---
Training holds four models: the policy and the critic (trained) plus the reference and the reward model (frozen) ([DeepSeekMath, Fig. 4](https://arxiv.org/abs/2402.03300)). The critic is usually as large as the policy, and learning an accurate per-token value is hard when only the last token is rewarded — harder still for long chain-of-thought, where text generated early may be revised later, so a partial response says little about the final reward ([DeepSeek-R1, App. A.3](https://arxiv.org/abs/2501.12948)). [[DPO]] removes the reward model and the RL loop; [[GRPO]] removes the critic. [[GRPO#GRPO vs PPO]] has the DeepSeek-R1 evidence that PPO's result on MATH hinges on the GAE $\lambda$.

## Relationship to Other Notes
---
- [[Policy Gradient Methods]] — the hub: REINFORCE with a baseline is the foundation that PPO's clipped surrogate and GAE build on; the cross-method comparison and the unified gradient view of PPO, DPO, and GRPO live there.
- [[RLHF]] — PPO is the third stage of the SFT → reward model → PPO pipeline; this note details that stage's objective, per-token KL, and practical settings, while RLHF covers reward-model training and reward hacking.
- [[DPO]] — replaces the PPO stage with a classification loss on preference pairs, removing both the reward model and the RL loop.
- [[GRPO]] — keeps PPO's clipped ratio but replaces the critic and GAE with a group-relative baseline, and moves the reference-model KL from the reward into the loss.
- [[KL Divergence]] — two different KLs meet here: the trust-region KL between $\pi_{\theta_{old}}$ and $\pi_\theta$ (which the clip approximates) and the KL penalty to the frozen reference model in RLHF.
- [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] — PPO is the standard on-policy actor-critic method there, in contrast to off-policy methods such as SAC.
- [[Behavior Cloning]] — the SFT stage that precedes PPO in RLHF is behavior cloning on demonstrations, and the SFT model becomes the reference $\pi_\text{ref}$.
- [[LLMs]] · [[LLaMA]] — Llama 2-Chat is post-trained with PPO on top of rejection sampling; Llama 3 later moved to DPO.
