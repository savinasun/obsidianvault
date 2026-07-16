### [[dl_moc]]  
# **Elo Score**

**Tags**:: #math #llm
**Links**:: [[RLHF]] · [[Benchmark Datasets]] · [[LLMs]] · [[Optimization Algorithms]] · [[Entropy]] · [[Deep Reinforcement Learning]]


The **Elo rating system** assigns each player a scalar rating $R$ such that rating *differences* predict expected game outcomes, and updates ratings after each game by how much the result deviated from expectation. Devised by **Arpad Elo** (1903–1992), a Hungarian-American physics professor at Marquette and chess master — "Elo" is his surname, not an acronym. Adopted by the USCF in 1960 (replacing the flawed Harkness system) and by FIDE in 1970. In ML it is the standard language for **pairwise-comparison evaluation**: LLM arena leaderboards, self-play curves in game-playing RL, and (via Bradley-Terry) the loss behind [[RLHF]] reward models.

## The Model
---
**Expected score** of player A against player B:
$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \qquad E_A + E_B = 1
$$

The 400 scale is an odds convention: $\frac{E_A}{1-E_A} = 10^{(R_A - R_B)/400}$ — every 400 points of advantage multiply the odds by 10. A +100 gap gives $E \approx 0.64$; +200 gives $E \approx 0.76$.

**Update rule** after a game with actual score $S_A \in \{1, \tfrac12, 0\}$ (win / draw / loss):
$$
R'_A = R_A + K\,(S_A - E_A)
$$

- $K$ controls responsiveness (FIDE: 40 for new/junior players, 20 below 2400, 10 once 2400 is reached — permanently; US Chess instead uses $K = 800/(N' + m)$ with $N'$ an effective-games uncertainty proxy)
- Draws are just $S = \tfrac12$; with draws possible, $E$ is an expected *score*, not a win probability ($E = P(\text{win}) + \tfrac12 P(\text{draw})$)
- Updates are **zero-sum** when both players share the same $K$: $\Delta R_A + \Delta R_B = 0$ (real federations break exact conservation via K tiers, rating floors, and bonus points — deliberate anti-deflation mechanisms)

## Statistical Foundations
---
**Gaussian vs. logistic.** Elo's original model: each player's single-game performance is $\mathcal{N}(R, \sigma^2)$ with "class interval" $\sigma = 200$, so the performance *difference* has $\sigma_d = 200\sqrt{2} \approx 283$ and $E_A = \Phi\big((R_A - R_B)/\sigma_d\big)$. The USCF later switched to the logistic curve (it fit results better, especially for lower-rated players) — that is the ubiquitous $1/(1+10^{-D/400})$ formula. **FIDE's official conversion table, however, is still Elo's normal-distribution table** (it matches $\Phi(D / \tfrac{2000}{7})$ almost exactly, and diverges from the logistic in the tails). Saying "FIDE uses the logistic formula" is a common error.

**Bradley-Terry equivalence.** Setting $\pi_i = 10^{R_i/400}$, the logistic Elo expected score is exactly the **Bradley-Terry model** (Bradley & Terry, 1952) for paired comparisons:
$$
\Pr(A \succ B) = \frac{\pi_A}{\pi_A + \pi_B} = \sigma\big(\theta_A - \theta_B\big), \qquad \theta_i = \frac{\ln 10}{400} R_i
$$
One natural-logit unit ≈ $400/\ln 10 \approx 173.7$ Elo points. The maximum-likelihood idea predates both: Zermelo (1929) fit exactly this model to chess tournaments. BT ratings are identifiable only **up to an additive constant** — only differences mean anything, and ratings from different pools (FIDE vs. USCF vs. engine lists vs. LLM arenas, each with its own anchor) are **not comparable**.

## Elo Update = Online SGD on Cross-Entropy
---
Treat a game as a binary label with prediction $E_A = \sigma\big(c(R_A - R_B)\big)$, $c = \ln 10/400$. The [[Entropy|cross-entropy]] loss $L = -[S_A \ln E_A + (1-S_A)\ln(1-E_A)]$ has gradient
$$
\frac{\partial L}{\partial R_A} = -(S_A - E_A)\,\frac{\ln 10}{400}
$$
so one SGD step $R_A \leftarrow R_A - \eta\, \partial L/\partial R_A$ **is precisely the Elo update** with $K = \eta \ln 10/400$ — the K-factor is a learning rate (K=20 ↔ natural-scale lr ≈ 0.115). Elo is logistic regression fit online, "a one-layer neural network" (Király & Qian, [arXiv 1701.08055](https://arxiv.org/abs/1701.08055)).

Consequences familiar from [[Optimization Algorithms]]:
- **Constant step size never converges** — ratings fluctuate around the (possibly drifting) truth with noise ∝ K; K trades adaptation speed against steady-state variance
- The final rating **depends on game order** (it's a sequential estimator)
- For *static* players, the right estimator is batch **BT maximum likelihood** over the full match matrix (Zermelo's iteration / MM algorithms, Hunter 2004) — order-independent, and requiring a strongly connected comparison graph to exist

## Elo for LLM Leaderboards: Chatbot Arena
---
**Chatbot Arena / LMArena** ([Chiang et al., 2024](https://arxiv.org/abs/2403.04132)) ranks LLMs from crowdsourced, anonymized pairwise battles (~240K human votes across 50+ models by early 2024). The evolution is a case study in the SGD-vs-MLE distinction above:

1. **May 2023**: online Elo updates with a deliberately small $K = 4$ (models are static — large K over-weights recent battles), initialized at 1000
2. **Dec 2023**: switched to **Bradley-Terry MLE** over the full battle history — same underlying logistic model, but order-independent, with proper uncertainty (bootstrap / sandwich robust standard errors → the leaderboard's confidence intervals)
3. **Aug 2024**: **style control** — augment the BT logistic regression with style covariates (answer length, markdown headers/bold/lists as normalized differences); length is the dominant coefficient, and verbose models drop once style is regressed out

The displayed "Elo score" (now "**Arena Score**") is therefore a BT coefficient on an Elo-like scale, not sequential Elo. Arena numbers are on their own anchor (init 1000) and are not comparable to chess ratings. See [[Benchmark Datasets]] for the companion MT-Bench data.

## Connection to RLHF Reward Models
---
The preference model in [[RLHF]] **is Bradley-Terry, i.e. Elo's model**. Christiano et al. ([2017](https://arxiv.org/abs/1706.03741)) made the analogy explicit: "just as the difference in Elo points of two chess players estimates the probability of one player defeating the other... the difference in predicted reward of two trajectory segments estimates the probability that one is chosen over the other." InstructGPT's reward-model loss
$$
\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}}\,\mathbb{E}_{(x, y_w, y_l)}\big[\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\big]
$$
is exactly the BT negative log-likelihood — a reward model learns a per-response Elo rating, amortized by a network. (The $1/\binom{K}{2}$ matters: labelers ranked $K{=}4$–$9$ responses per prompt; treating the induced pairs as independent examples caused overfitting.)

## Elo in Game-Playing RL
---
DeepMind's AlphaGo Zero / AlphaZero training curves report Elo fit by **BayesElo** (Bayesian logistic regression, $c_{\text{elo}} = 1/400$) over tournaments between training checkpoints plus an anchor baseline. AlphaGo Zero: 5185 vs. AlphaGo Master 4858 vs. AlphaGo Lee 3739 (and 100–0 head-to-head vs. Lee). **Caveat**: these are ratings within a closed pool of siblings — self-play Elo can be inflated and need not transfer to unseen opponents; the head-to-head match results are the harder evidence.

## Limitations
---
- **Transitivity**: a single scalar assumes $A > B > C \Rightarrow A > C$. In cyclic games it fails completely — for rock-paper-scissors, Elo converges to $\hat{p} = \tfrac12$ for every pair, "no predictive power" ([Balduzzi et al., 2018](https://arxiv.org/abs/1806.02643)). Their **mElo$_{2k}$** adds low-rank antisymmetric structure: $\hat{p}_{ij} = \sigma(r_i - r_j + \mathbf{c}_i^\top \Omega\, \mathbf{c}_j)$. Real games are "spinning tops" — non-transitivity peaks at mid-skill ([Czarnecki et al., 2020](https://arxiv.org/abs/2004.09468))
- **Additivity**: even in transitive games, win probability need not be a fixed function of the rating *difference* alone ([Bertrand et al., 2022](https://arxiv.org/abs/2206.12301))
- **No uncertainty**: a point estimate, whether from 10 games or 1000
- **Order dependence and the K dilemma**: responsiveness vs. noise (LLM eval inherits this — [Boubdir et al., 2023](https://arxiv.org/abs/2311.17295) document Elo volatility and axiom violations in LLM comparisons)
- **Stationarity**: assumes skill drifts slowly relative to 1/K games
- Draws carry an implicit model (vanilla Elo silently corresponds to ~50% draws between equals; Szczecinski & Djebbi)

## Successors
---
| System | Key idea | Extra state |
|---|---|---|
| **Glicko** (Glickman, 1995) | rating deviation RD = uncertainty; inactivity inflates RD, games shrink it; 95% CI ≈ $r \pm 2\,\text{RD}$ | $(r, \text{RD})$ |
| **Glicko-2** | adds per-player **volatility** $\sigma$ (expected skill fluctuation; controls RD inflation); used by Lichess | $(r, \text{RD}, \sigma)$ |
| **TrueSkill** ([Herbrich et al., NIPS 2006](https://papers.nips.cc/paper_files/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)) | Bayesian: skill $\sim \mathcal{N}(\mu, \sigma^2)$, factor graph + expectation propagation; teams (performance sums), multiplayer, draw margin; Xbox Live displays conservative $\mu - 3\sigma$ | $(\mu, \sigma)$ |
| **TrueSkill 2** (2018) | + per-player statistics, squads, experience; 52% → 68% match prediction on Halo 5 | — |
| **BayesElo** (Coulom) | batch MAP Elo with explicit draw + first-move-advantage parameters | — |
| **WHR** (Coulom, 2008) | whole-history MAP: Bradley-Terry + Wiener-process prior on each rating trajectory; order-independent, principled uncertainty growth $\propto \sqrt{t}$ | full history |

On a 2.3M-game KGS Go test set, prediction accuracy ordered: Elo 55.12% < Glicko 55.52% < TrueSkill 55.54% < BayesElo 55.67% < WHR 55.79% — small-looking gaps, but mostly statistically significant (Coulom's 95%-confidence threshold is a 0.091% test-set difference, which every adjacent gap clears except Glicko → TrueSkill at 0.014%).

## Chess Anchors (for intuition)
---
- Magnus Carlsen's record official peak: **2882** (May 2014); GM title requires reaching 2500 once + three norms
- Top engines: Stockfish ≈ 3650 on CCRL — but on an engine-pool anchor, not the FIDE scale
- FIDE floor: ratings below 1400 go unpublished; in March 2024 FIDE applied a one-time anti-deflation compression $\Delta R = 0.4\,(2000 - R)$ to all sub-2000 ratings

## References
---
- [Elo, 1978 — *The Rating of Chess Players, Past and Present*](https://en.wikipedia.org/wiki/Arpad_Elo)
- [Bradley & Terry, 1952 — Rank Analysis of Incomplete Block Designs](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091)
- [Glickman & Jones, 1999 — Rating the Chess Rating System](https://glicko.net/research/chance.pdf)
- [Király & Qian, 2017 — Modelling Competitive Sports: Bradley-Terry-Élő Models](https://arxiv.org/abs/1701.08055)
- [Chiang et al., 2024 — Chatbot Arena](https://arxiv.org/abs/2403.04132) · [LMSYS BT switch blog (Dec 2023)](https://lmsys.org/blog/2023-12-07-leaderboard/) · [Style control blog (Aug 2024)](https://lmsys.org/blog/2024-08-28-style-control/)
- [Christiano et al., 2017 — Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Balduzzi et al., 2018 — Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643)
- [Herbrich, Minka & Graepel, 2006 — TrueSkill](https://papers.nips.cc/paper_files/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)
- [Coulom, 2008 — Whole-History Rating](https://www.remi-coulom.fr/WHR/WHR.pdf) · [BayesElo](https://www.remi-coulom.fr/Bayesian-Elo/)
- [Glickman — Glicko](http://www.glicko.net/glicko/glicko.pdf) · [Glicko-2](http://www.glicko.net/glicko/glicko2.pdf)
- [FIDE Rating Regulations](https://handbook.fide.com/chapter/B022024)
