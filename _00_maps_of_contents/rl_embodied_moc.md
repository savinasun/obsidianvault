### [[dl_moc]]
# **RL & Embodied AI**

**Tags**:: #rl
**Links**:: [[math_moc]] · [[learning_methods_moc]] · [[sequence_llms_moc]] · [[vision_ssl_moc]] · [[generative_moc]]

Policy optimization, imitation learning, world models, and human feedback — the research direction connecting deep learning to autonomous agents and embodied robotics.

## Foundations
---
- **[[Reinforcement Learning]]** — MDP formulation $(S, A, R, P, \gamma)$; value functions $V^\pi(s)$ and $Q^\pi(s,a)$; Bellman equations; model-free vs. model-based; on-policy vs. off-policy; Q-learning, SARSA, Actor-Critic taxonomy
- **[[Deep Reinforcement Learning]]** — DQN: replay buffer, target network, $\epsilon$-greedy; A2C/A3C; PPO (clipped surrogate objective); SAC (maximum entropy RL); reward shaping; sparse reward challenges

## Policy Optimization
---
- **[[Policy Gradient Methods]]** — REINFORCE (Williams, 1992); the policy gradient theorem $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log\pi_\theta(a|s) \cdot R]$; variance reduction via baselines; PPO and TRPO as stable high-performance variants
- **[[RLHF]]** — Reinforcement Learning from Human Feedback: reward model training on preference data, PPO fine-tuning against the reward model; Constitutional AI; DPO as an RLHF alternative that skips explicit RL; central to aligning LLMs

## Imitation & Offline Learning
---
- **[[Behavior Cloning]]** — Supervised learning on expert demonstrations; covariate shift / distributional shift problem; DAgger algorithm (interactive data collection); connection to offline RL and the imitation learning spectrum

## Model-Based & World Models
---
- **[[World Models]]** — Learning a latent dynamics model of the environment; Dreamer/DreamerV3: RSSM (Recurrent State Space Model), latent imagination for policy training; model-based RL reduces real-world sample complexity

## Cross-References
---
- **[[RLHF]]** bridges this section and [[sequence_llms_moc]] — it is the primary method for aligning LLMs to human preferences
- **[[Contrastive Learning]]** (→ [[vision_ssl_moc]]) provides the representation backbone for many visual RL agents
- **[[VAE]]** / **[[DDPM]]** (→ [[generative_moc]]) are used as world model components in latent imagination methods

## Dataview
---
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "rl")
SORT file.name ASC
```
