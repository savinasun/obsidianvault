### [[dl_moc]]  
# **Low Rank Matrix**

**Tags**::  
**Links**::


A matrix is “numerically low-rank” if its singular values drop off very quickly. For example, if  $σ_1≥σ_2≥⋯≥σ_k≫σ_{k+1}≈0$, then truncating at $k$ introduces little error $(∥A−A_k∥^2=σ_{k+1})$.

In some applications (e.g. image compression, PCA), one inspects the singular‐value “elbow” to pick $k$.

Below is a more in‐depth look at **why** the “elbow method” (or “scree‐plot elbow”) is commonly used to decide whether a matrix is numerically low‐rank. We cover both the **theoretical rationale** (via the Eckart–Young theorem and perturbation considerations) and the **practical/statistical viewpoint** (variance explained, noise floor, and diminishing returns). Citations are provided from recent pedagogical sources.


## Eckart–Young Theorem and Best Low‐Rank Approximation
---
1. **Eckart–Young (Frobenius‐norm) Optimality**  
    Suppose
    
    $$A \in \mathbb{R}^{m\times n},\ A=U \Sigma V^T  =\sum_{i=1}^r \sigma_i u_i v_i^T,\ A \;\in\; \mathbb{R}^{m\times n},\quad A = U\,\Sigma\,V^T \;=\; \sum_{i=1}^{r} \sigma_i\,u_i\,v_i^T$$,
    
    where σ1≥σ2≥⋯≥σr>0\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r>0. If one wants a **rank-kk** approximation AkA_k (with k<rk < r) that minimizes the Frobenius norm ∥A−B∥F\|A - B\|_F over all rank-kk matrices BB, then
    
    Ak  =  ∑i=1kσi ui viT A_k \;=\; \sum_{i=1}^k \sigma_i\,u_i\,v_i^T
    
    is optimal. Moreover,
    
    ∥A−Ak∥F2  =  ∑i=k+1rσi2. \|A - A_k\|_F^2 \;=\; \sum_{i=k+1}^{r} \sigma_i^2.
    
    In other words, **keeping the top kk singular values** yields the smallest possible error in Frobenius norm among all rank-kk approximations. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Low-rank_approximation?utm_source=chatgpt.com "Low-rank approximation"))
    
2. **Implication for Selecting kk:**
    
    - Notice that each additional singular value σk+1\sigma_{k+1} “fills in” exactly σk+12\sigma_{k+1}^2 of the remaining squared error.
        
    - If σk+1≪σk\sigma_{k+1} \ll \sigma_k, then adding the (k+1)(k+1)-st term only reduces the Frobenius‐error by a very small amount. In other words, **beyond some index kk**, the returns for adding more singular‐vector components diminish sharply.
        

Therefore, whenever the singular‐value spectrum has a **clear break**—i.e., σ1≥⋯≥σk≫σk+1≈…\sigma_1\ge\dots\ge\sigma_k \gg \sigma_{k+1}\approx\dots— then truncating at that kk makes ∥A−Ak∥≈σk+1\|A - A_k\| \approx \sigma_{k+1} very small. That point kk is the natural cutoff for a “numerically low‐rank” representation—since anything beyond σk+1\sigma_{k+1} belongs largely to noise or negligible modes.

---

## 2. The “Elbow” Heuristic in a Scree Plot

1. **What Is a Scree Plot?**
    
    - A **scree plot** graphs the singular values (σi)(\sigma_i) (or sometimes the percentage of variance explained by each singular value) on the vertical axis versus the index ii on the horizontal axis.
        
    - Typically you see a fairly **steep drop** in the first few σi\sigma_i, then a **“knee” or “elbow”**, then a gentler tail.
        
2. **Why the Elbow Marks the Rank Cutoff**
    
    - At indices i≤ki \le k, σi\sigma_i are significantly larger, meaning each mode contributes substantially to the matrix’s structure.
        
    - At i>ki > k, σi\sigma_i either flatten out (noise floor) or decay slowly—so each additional singular vector only explains a tiny fraction of the total.
        
    - The “elbow” is precisely where the gain from adding another singular component falls off rapidly. Thus, if you choose kk at that elbow, you capture most of the “meaningful energy” with as few modes as possible. ([web.stanford.edu](https://web.stanford.edu/class/cs168/l/l9.pdf?utm_source=chatgpt.com "[PDF] The Singular Value Decomposition (SVD) and Low-Rank Matrix ..."), [bioconductor.org](https://bioconductor.org/books/3.17/OSCA.advanced/dimensionality-reduction-redux.html?utm_source=chatgpt.com "Chapter 4 Dimensionality reduction, redux - Bioconductor"))
        
3. **Relation to Variance Explained (PCA Viewpoint)**
    
    - In principal component analysis (PCA), one often plots the **cumulative percentage** of variance explained by the top ii singular values (i.e.\ ∑j=1iσj2/∑j=1rσj2\sum_{j=1}^i \sigma_j^2 \big/ \sum_{j=1}^r \sigma_j^2).
        
    - A sharp elbow in that plot likewise indicates that going from ii to i+1i+1 yields only marginal new variance explained. Hence, it is a standard rule of thumb to say “keep components up to the elbow” when you want a small set of PCs that still explain—say—90–95% of the total variance. ([bioconductor.org](https://bioconductor.org/books/3.17/OSCA.advanced/dimensionality-reduction-redux.html?utm_source=chatgpt.com "Chapter 4 Dimensionality reduction, redux - Bioconductor"))
        

---

## 3. Theoretical Justification via Perturbation and Noise Models

1. **Signal vs. Noise Separation**
    
    - In a data‐plus‐noise model, one often has
        
        A  =  L  +  N, A \;=\; L \;+\; N,
        
        where LL is a true low‐rank signal (e.g.\ rank rtruer_{\rm true}) and NN is noise (often modeled as random i.i.d.\ or white noise).
        
    - For large random‐noise matrices NN, one expects their singular values to cluster near zero or follow a Marchenko–Pastur distribution (depending on dimensions). In particular, above some threshold τ\tau, any singular value σi(A)>τ\sigma_i(A) > \tau must come from the signal part LL, while those σj(A)≤τ\sigma_j(A)\le\tau are plausibly noise. ([arxiv.org](https://arxiv.org/pdf/1305.5870?utm_source=chatgpt.com "[PDF] The Optimal Hard Threshold for Singular Values is 4/ - arXiv"), [en.wikipedia.org](https://en.wikipedia.org/wiki/Low-rank_approximation?utm_source=chatgpt.com "Low-rank approximation"))
        
2. **Optimal Hard Threshold (Gavish–Donoho)**
    
    - In the asymptotic regime (large m, n with fixed aspect ratio), it can be shown that the **optimal threshold** λ∗\lambda^* for singular‐value hard‐thresholding is a constant multiple of the noise level, e.g.\ λ∗≈4/3 σnoise\lambda^*\approx 4/\sqrt{3}\,\sigma_{\rm noise}. Any σi(A)<λ∗\sigma_i(A) < \lambda^* is set to zero, and those σi(A)>λ∗\sigma_i(A) > \lambda^* are kept.
        
    - In practice, this often corresponds to the “elbow” of the empirical plot—below that point, singular values fluctuate around the noise floor. ([arxiv.org](https://arxiv.org/pdf/1305.5870?utm_source=chatgpt.com "[PDF] The Optimal Hard Threshold for Singular Values is 4/ - arXiv"), [en.wikipedia.org](https://en.wikipedia.org/wiki/Singular_value_decomposition?utm_source=chatgpt.com "Singular value decomposition"))
        
3. **Perturbation Theory (Weyl’s Inequalities)**
    
    - Suppose A=L+NA = L + N where rank(L)=rtrue\mathrm{rank}(L) = r_{\rm true}. Then Weyl’s inequality tells us
        
        ∣σi(A)  −  σi(L)∣  ≤  ∥N∥2. \bigl|\sigma_i(A) \;-\; \sigma_i(L)\bigr| \;\le\;\|N\|_2.
    - If the noise operator norm ∥N∥2\|N\|_2 is small compared to the top singular values of LL, then the top rtruer_{\rm true} singular values of AA remain far above the noise level, while the rest sink near ≤∥N∥2\le\|N\|_2.
        
    - Graphically, that separation appears as an elbow: the first rtruer_{\rm true} values lie in a plateau above ∥N∥2\|N\|_2, then a gap, and the tail near ∥N∥2\|N\|_2. Thus, the elbow index recovers rtruer_{\rm true}. ([web.stanford.edu](https://web.stanford.edu/class/cs168/l/l9.pdf?utm_source=chatgpt.com "[PDF] The Singular Value Decomposition (SVD) and Low-Rank Matrix ..."), [arxiv.org](https://arxiv.org/pdf/1305.5870?utm_source=chatgpt.com "[PDF] The Optimal Hard Threshold for Singular Values is 4/ - arXiv"))
        

---

## 4. Practical Considerations: Why the Elbow Method Is Popular

1. **Data‐Driven and Model‐Agnostic**
    
    - You do _not_ need to know the exact noise distribution. Simply plotting σi\sigma_i vs.\ ii (or cumulative variance) lets you **visually** identify where the curve “flattens out.” That is usually where signal ends and noise begins.
        
    - It works in a wide variety of settings: image compression, recommendation systems, bioinformatics, etc. ([medium.com](https://medium.com/%40hiroshi.wayama/linear-algebra-singular-value-decomposition-and-low-rank-approximation-3c83a4359b64?utm_source=chatgpt.com "Singular value decomposition and low-rank approximation - Medium"), [web.stanford.edu](https://web.stanford.edu/class/cs168/l/l9.pdf?utm_source=chatgpt.com "[PDF] The Singular Value Decomposition (SVD) and Low-Rank Matrix ..."))
        
2. **Avoiding Overfitting / Overparameterization**
    
    - In many machine‐learning pipelines (e.g.\ low-rank factor models, PCA preprocessing), choosing too large a rank kk leads to overfitting. At the elbow, you strike a balance:
        
        - **Most of the structure** is captured by the first kk modes.
            
        - **Additional modes** past kk only model noise or minor fluctuations.
            
    - Empirically, this often yields better generalization than choosing kk by arbitrary criteria (e.g.\ “retain 99% energy” may pick too many, while “retain 90%” might be too few). ([linkedin.com](https://www.linkedin.com/advice/0/how-do-you-choose-optimal-rank-low-rank-approximation?utm_source=chatgpt.com "How do you choose the optimal rank for low-rank approximation?"), [bioconductor.org](https://bioconductor.org/books/3.17/OSCA.advanced/dimensionality-reduction-redux.html?utm_source=chatgpt.com "Chapter 4 Dimensionality reduction, redux - Bioconductor"))
        
3. **Computational Efficiency**
    
    - When you see a sharp elbow at i=ki=k, you can cost-effectively compute just the top kk singular values and vectors (using truncated SVD methods). There is no need to compute a full SVD of size min⁡(m,n)\min(m,n). ([sjsu.edu](https://www.sjsu.edu/faculty/guangliang.chen/Math203S18/svdnotes.pdf?utm_source=chatgpt.com "[PDF] Some notes on SVD, dimensionality reduction, and clustering"), [web.lums.edu.pk](https://web.lums.edu.pk/~imdad/pdfs/CS5312_Notes/CS5312_Notes-17-SVD.pdf?utm_source=chatgpt.com "[PDF] Singular Value Decomposition - Index of /"))
        
    - In large‐scale settings, randomized or Lanczos algorithms will approximate σ1,…,σk\sigma_1,\dots,\sigma_k quickly—once you detect an elbow, you save on computing the tail.
        
4. **Variance Explained and Downstream Tasks**
    
    - If you want to choose a subspace dimension for subsequent tasks (clustering/PCA regression/latent‐factor modeling), picking kk at the elbow often preserves the bulk of **“meaningful variance”** while discarding spurious dimensions.
        
    - In genomics or single-cell RNA‐seq analysis, people literally run a “scree plot” and use the elbow to decide how many PCs to keep (so that biological signal is not thrown away). ([bioconductor.org](https://bioconductor.org/books/3.17/OSCA.advanced/dimensionality-reduction-redux.html?utm_source=chatgpt.com "Chapter 4 Dimensionality reduction, redux - Bioconductor"))
        

---

## 5. What If No Clear Elbow Appears?

1. **Gradual Decay vs. Sharp Drop**
    
    - In some applications, the singular‐value curve decays **smoothly**—no sudden knee. This means either:
        
        - The data **lacks a strong low-rank structure** (eigenvalues taper slowly, so every component still carries some signal), or
            
        - The noise is not white/rank‐one but has large correlated components, smearing out the separation.
            
    - In that case, one may resort to other rules:
        
        - **Energy‐cutoff** (e.g.\ pick smallest kk so that ∑i=1kσi2≥0.95∑i=1rσi2\sum_{i=1}^k \sigma_i^2 \ge 0.95 \sum_{i=1}^r \sigma_i^2). ([bioconductor.org](https://bioconductor.org/books/3.17/OSCA.advanced/dimensionality-reduction-redux.html?utm_source=chatgpt.com "Chapter 4 Dimensionality reduction, redux - Bioconductor"))
            
        - **Cross‐validation**: Split the matrix (or its entries) into training/test sets, compute low-rank fits for various kk, and pick the kk minimizing reconstruction error on held‐out entries. ([linkedin.com](https://www.linkedin.com/advice/0/how-do-you-choose-optimal-rank-low-rank-approximation?utm_source=chatgpt.com "How do you choose the optimal rank for low-rank approximation?"))
            
2. **Information‐Criterion Approaches**
    
    - Instead of a purely visual elbow, one can use methods like **AIC** or **BIC** on a latent‐factor model (e.g.\ probabilistic PCA). These automatically trade off model complexity (rank) versus likelihood (reconstruction accuracy).
        
    - In many cases, however, these criteria give a similar numeric answer to the elbow, but they require specifying a noise‐model likelihood.
        

---

## 6. Summarizing the Core Reasons

- **Optimality** (Eckart–Young): A rank-kk SVD is guaranteed to minimize error ∥A−Ak∥\|A - A_k\|. Hence, beyond the elbow, extra singular‐vector additions reduce error by only σk+1\sigma_{k+1} (which is very small). ([en.wikipedia.org](https://en.wikipedia.org/wiki/Low-rank_approximation?utm_source=chatgpt.com "Low-rank approximation"), [web.stanford.edu](https://web.stanford.edu/class/cs168/l/l9.pdf?utm_source=chatgpt.com "[PDF] The Singular Value Decomposition (SVD) and Low-Rank Matrix ..."))
    
- **Signal/Noise Separation** (Perturbation Theory + Random Matrix Intuition): If A=L+NA = L + N with rank(L)=rtrue\mathrm{rank}(L)=r_{\rm true} and noise NN, then only the top rtruer_{\rm true} singular values exceed the noise floor ∥N∥2\|N\|_2. The elbow is where σi≈∥N∥2\sigma_{i}\approx\|N\|_2 for i>rtruei>r_{\rm true}. ([arxiv.org](https://arxiv.org/pdf/1305.5870?utm_source=chatgpt.com "[PDF] The Optimal Hard Threshold for Singular Values is 4/ - arXiv"), [en.wikipedia.org](https://en.wikipedia.org/wiki/Singular_value_decomposition?utm_source=chatgpt.com "Singular value decomposition"))
    
- **Variance‐Explained Heuristic** (PCA/Scree‐Plot): The elbow shows where extra components explain diminishing amounts of variance—ideal when you want to keep only the “important” directions. ([bioconductor.org](https://bioconductor.org/books/3.17/OSCA.advanced/dimensionality-reduction-redux.html?utm_source=chatgpt.com "Chapter 4 Dimensionality reduction, redux - Bioconductor"), [medium.com](https://medium.com/%40hiroshi.wayama/linear-algebra-singular-value-decomposition-and-low-rank-approximation-3c83a4359b64?utm_source=chatgpt.com "Singular value decomposition and low-rank approximation - Medium"))
    
- **Computational Savings**: Identifying the elbow early means you need not compute the entire spectrum; truncated‐SVD methods can find σ1,…,σk\sigma_1,\dots,\sigma_k quickly and reliably. ([sjsu.edu](https://www.sjsu.edu/faculty/guangliang.chen/Math203S18/svdnotes.pdf?utm_source=chatgpt.com "[PDF] Some notes on SVD, dimensionality reduction, and clustering"), [web.lums.edu.pk](https://web.lums.edu.pk/~imdad/pdfs/CS5312_Notes/CS5312_Notes-17-SVD.pdf?utm_source=chatgpt.com "[PDF] Singular Value Decomposition - Index of /"))
    
- **Practical Success**: Over decades of applications (e.g.\ image compression, recommender systems, genomics), the elbow method consistently balances accuracy versus model complexity, leading to robust low-rank tradeoffs. ([researchgate.net](https://www.researchgate.net/figure/Scree-plot-the-characteristic-elbow-can-be-identified-at-i-i-3-in-this-case_fig1_364626263?utm_source=chatgpt.com "Scree plot: the characteristic elbow can be identified at í µí±˜ = 3"), [linkedin.com](https://www.linkedin.com/advice/0/how-do-you-choose-optimal-rank-low-rank-approximation?utm_source=chatgpt.com "How do you choose the optimal rank for low-rank approximation?"))
    

---

### Final Takeaway

By plotting σi\sigma_i versus ii, the “elbow” pinpoints exactly where the **marginal benefit** of retaining one more singular direction drops off. Mathematically, that means the next singular value σk+1\sigma_{k+1} is so small that

∥A−Ak∥F  =  ∑i=k+1rσi2 \|A - A_k\|_F \;=\; \sqrt{\sum_{i=k+1}^r \sigma_i^2}

is already negligible. In effect, you’ve captured all of the useful structure—anything beyond the elbow is “mostly noise.” Hence, **the elbow method provides a data‐driven, theoretically justified, and computationally efficient rule** for deciding when a matrix is (numerically) low‐rank.