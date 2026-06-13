### [[dl_moc]]  
# **World Models**

**Tags**:: #rl
**Links**:: [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] · [[DDPM]] · [[Behavior Cloning]]


A **world model** is a learned model of the environment dynamics: given the current state and action, it predicts the next state and reward. Model-based RL uses world models to **plan** or **generate synthetic experience** rather than requiring all learning to come from real environment interactions — dramatically improving sample efficiency.

## Why World Models?
---
Model-free RL (PPO, SAC) requires millions of environment steps to learn complex tasks because every gradient update relies on real experience. World models break this constraint:

- **Planning**: instead of acting, simulate thousands of rollouts in the learned model and choose the best action
- **Data augmentation**: train the policy on imagined trajectories inside the world model
- **Transfer**: a good world model can support many downstream tasks

**Tradeoff**: model error compounds — a slightly wrong model produces increasingly wrong predictions over long horizons. Methods must be robust to this.

## Recurrent State Space Model (RSSM)
---
The RSSM (Hafner et al., 2019 — DreamerV1) is the core latent dynamics model used in the Dreamer family. It factors the latent state into:

- **Stochastic component** $z_t \sim p_\theta(z_t \mid h_t)$: captures uncertainty
- **Deterministic component** $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$: GRU hidden state, carries information across time

Together $(h_t, z_t)$ forms the latent state. The full model consists of:

| Component | Function | Distribution |
|-----------|----------|-------------|
| **Representation model** | $q_\phi(z_t \mid h_t, x_t)$ | Encoder: infer $z_t$ from current observation |
| **Transition model** | $p_\theta(z_t \mid h_t)$ | Prior: predict next $z_t$ from GRU state |
| **Observation model** | $p_\theta(x_t \mid h_t, z_t)$ | Decoder: reconstruct observation |
| **Reward model** | $p_\theta(r_t \mid h_t, z_t)$ | Predict reward |
| **Recurrent model** | $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$ | GRU transition |

Training uses the ELBO / RSSM objective, balancing reconstruction and KL regularization.

## Dreamer (DreamerV1)
---
Hafner et al. (2019) — **learns latent dynamics, then trains the policy entirely in imagination**:

1. **World model training**: fit RSSM to observed sequences $(x_t, a_t, r_t)$
2. **Behavior learning**: unroll imagined trajectories $\{h_\tau, z_\tau, a_\tau\}$ inside the RSSM for $H$ steps; train policy $\pi_\theta$ and value function $V_\psi$ on imagined rollouts using backpropagation through the model

The actor-critic update uses **latent-space value targets** via truncated backprop through time (TBPTT), avoiding the need to run real environment steps during policy learning.

## DreamerV2 & V3
---
**DreamerV2** (2020): replaces continuous stochastic latents with **categorical latents** (32 categories × 32 variables → 1024-bit latent), trained with straight-through gradients. Achieves superhuman performance on Atari.

**DreamerV3** (2023): the most general version. Key innovations:
- **Symlog predictions**: applies $\text{symlog}(x) = \text{sign}(x)\ln(|x|+1)$ to targets, handling reward scales that vary across tasks without reward normalization
- **KL balancing**: separately scales the KL terms for the stochastic and deterministic parts
- **Regularized returns**: uses a percentile-based normalization of returns
- **Universal**: solves tasks across Atari, DMControl, Minecraft, and robotics from pixels in a single training run with the same hyperparameters

## MuZero
---
MuZero (Schrittwieser et al., 2020 — DeepMind) learns a world model for **planning with MCTS** rather than imagination rollouts:

- **Representation function**: $h_t = f(o_1, \ldots, o_t)$ — encode observation history
- **Dynamics function**: $h_{t+1}, r_t = g(h_t, a_t)$ — predict next latent + reward
- **Prediction function**: $\pi_t, v_t = p(h_t)$ — predict policy and value

At each step, run **Monte Carlo Tree Search (MCTS)** using the dynamics function to simulate future states and select actions. Achieves state-of-the-art on Go, Chess, Shogi, and Atari — without knowledge of game rules.

## Comparison: Model-Free vs. World Models
---

| | Model-Free (PPO/SAC) | Dreamer | MuZero |
|---|---|---|---|
| **Sample efficiency** | Low | High | High |
| **Planning** | No | Latent imagination | MCTS |
| **Observation type** | State/pixels | Pixels | State/pixels |
| **Compounding error** | N/A | Medium risk | Medium risk |
| **Compute (training)** | Lower | Higher | Higher |
| **Generalization** | Task-specific | Can transfer | Task-specific |

## World Models for Robotics
---
In the robotics context, world models offer:
- **Data efficiency**: real-world robot data is expensive; simulate policy training in the model
- **Safe exploration**: test dangerous actions in simulation
- **Multi-task**: a single world model can support multiple downstream tasks

**Challenges specific to robotics**:
- **Partial observability**: cameras don't capture full state (contact forces, hidden joints)
- **Non-stationarity**: real-world physics changes (object mass, friction) — model must generalize
- **Long horizons**: robot manipulation tasks require many steps; compounding error is severe

**Current directions**: video prediction models (e.g., Genie, UniSim) as scalable world models trained on internet-scale video data, then used for robot policy training.
