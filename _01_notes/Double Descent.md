### [[dl_moc]]  
# **Double Descent**

**Tags**:: #dl_theo
**Links**:: [[Scaling Laws]] · [[Manifold Hypothesis]] · [[Neural Network Initialization]] · [[Optimization Algorithms]]


**Double descent** describes the phenomenon where test error, as a function of model capacity, follows a surprising U-then-U pattern: it first decreases, then increases (classical overfitting), then *decreases again* as the model is overparameterized. This challenges the classical bias-variance tradeoff picture.

## Classical Regime: Bias-Variance Tradeoff
---
In the classical ML view, test error follows a single U-shaped curve as a function of model capacity:

$$
\text{Test Error} = \underbrace{\text{Bias}^2}_{\text{underfitting}} + \underbrace{\text{Variance}}_{\text{overfitting}} + \text{Noise}
$$

- **Underfitting** (high bias, low variance): the model is too simple to capture the data structure
- **Optimal capacity**: the sweet spot that minimizes test error
- **Overfitting** (low bias, high variance): the model memorizes training data, test error rises

**Classical prescription**:
- High Bias → Bigger network / train longer / different architecture
- High Variance → More data / regularization / early stopping

This prescription holds within the **classical (underparameterized) regime**.

## The Interpolation Threshold
---
The **interpolation threshold** is the critical model capacity at which the model can *exactly fit* (interpolate) the training data. Near this threshold:
- Training loss → 0
- The model is at maximum tension between memorization and generalization
- Test error peaks at its worst

This is the most dangerous point: the model has just enough capacity to memorize noise.

## Modern (Overparameterized) Regime: Double Descent
---
[Belkin et al., 2019](https://arxiv.org/abs/1812.11118) showed that beyond the interpolation threshold, test error *decreases again* as capacity continues to grow:

```
Test
Error
  |          classical regime  | modern regime
  |              ↓             |
  |       *                    |
  |      * *                   |
  |    *     *                 |         *
  |   *         **             |       **
  |  *             ****        |     **
  | *                  **** *  |   **
  |*                          *|***
  +——————————————————————————————————→ Model capacity
                              ↑
                    interpolation threshold
```

**Why does error decrease again in the overparameterized regime?**

1. **Implicit regularization by SGD**: among the many models that perfectly fit the training data, gradient descent finds the minimum-norm solution — which tends to generalize well. This is a form of implicit regularization that has no analogue in the classical regime.

2. **Benign overfitting**: overparameterized models can fit the noisy training labels while still learning the correct underlying function, because the extra capacity is used to fit noise in directions that don't affect test predictions (orthogonal to the data manifold).

3. **Minimum-norm interpolation**: in linear regression with $n$ samples and $p \gg n$ features, the minimum-norm least-squares solution $\hat\theta = X^\top(XX^\top)^{-1}y$ provably generalizes well under certain conditions on the data distribution.

## Empirical Evidence
---
[Nakkiran et al., 2019](https://arxiv.org/abs/1912.02292) confirmed double descent empirically across:
- Model size (number of parameters)
- Training epochs (epoch-wise double descent: even for fixed-size models, test error peaks early then decreases with more training)
- Dataset size (for fixed model, adding more data can initially *hurt* near the interpolation threshold)

![[Pasted image 20250513110154.png]]

Key findings:
- Occurs in ResNets, CNNs, and Transformers
- Occurs for both label noise and clean data
- The peak near the interpolation threshold is sharper with higher label noise

## Connection to Sample Complexity
---
Classical statistical learning theory (Vapnik-Chervonenkis, Rademacher complexity) predicts monotonic degradation beyond the interpolation threshold — it cannot explain double descent. This has driven renewed interest in **benign overfitting theory** and **neural tangent kernel (NTK) analysis**.

Under the NTK regime (infinite-width networks at initialization), overparameterized networks behave as kernel machines with a specific kernel, and minimum-norm interpolation provably generalizes.

## Connection to Scaling Laws
---
The modern regime's second descent is related to [[Scaling Laws]]: as both model size and dataset size grow jointly, test loss decreases monotonically. The double descent peak is avoided by scaling $N$ and $D$ together (as in [[Scaling Laws#Chinchilla Scaling Laws (Hoffmann et al., 2022)|Chinchilla]]) rather than fixing $D$ and only increasing $N$.

## Practical Implications
---
- **Don't stop at the first descent**: a model that overfits on the training set may improve further with more capacity or more training
- **Epoch double descent**: if training loss is still decreasing after many epochs, continuing to train may recover generalization (rather than worsening it as classical intuition would suggest)
- **Data near the interpolation threshold is worst**: if you have a medium-sized dataset and a medium-sized model that just barely overfits, you may be at the worst possible operating point — adding more data or more parameters can both help
