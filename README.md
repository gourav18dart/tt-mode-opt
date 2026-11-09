# Optimizing Mode Order in Tensor Train Decomposition: Beam Search Evaluation

This repository provides an independent implementation, experimental evaluation, and beam-search extension of the research paper:

> Petr Tichavsky and Ondrej Straka, "Optimizing the Order of Modes in Tensor Train Decomposition," *IEEE Signal Processing Letters*, vol. 32, pp. 1361-1365, 2025.

---

## 1. Mathematical Formulation

### 1.1 Tensor Train Decomposition
Let $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times \dots \times I_N}$ be an order-$N$ tensor. The Tensor Train (TT) decomposition represents individual tensor elements as a 1D contraction of core tensors:

$$\mathcal{T}(i_1, i_2, \dots, i_N) = \sum_{b_1=1}^{r_1} \sum_{b_2=1}^{r_2} \dots \sum_{b_{N-1}=1}^{r_{N-1}} \mathbf{w}_1(i_1, b_1) \mathbf{w}_2(b_1, i_2, b_2) \dots \mathbf{w}_N(b_{N-1}, i_N)$$

where:
- $\mathbf{w}_1 \in \mathbb{R}^{I_1 \times r_1}$ and $\mathbf{w}_N \in \mathbb{R}^{r_{N-1} \times I_N}$ are boundary matrix cores ($r_0 = r_N = 1$).
- $\mathbf{w}_k \in \mathbb{R}^{r_{k-1} \times I_k \times r_k}$ for $k = 2, \dots, N-1$ are 3-way internal core tensors.
- $r_1, r_2, \dots, r_{N-1} \in \mathbb{N}$ denote the TT bond dimensions (ranks).

### 1.2 Model Complexity
The memory complexity is given by the total number of stored parameters:

$$\text{Parameters} = \sum_{k=1}^N r_{k-1} I_k r_k, \quad \text{with } r_0 = r_N = 1$$

The bond dimension $r_k$ at cut $k$ equals the numerical rank of the $k$-th unfolding matrix $\mathbf{X}_{(k)}$ of size $(I_1 \cdots I_k) \times (I_{k+1} \cdots I_N)$.

---

## 2. Mode Ordering and Limitation of Greedy Heuristics

### 2.1 Mode-Order Dependence
Unlike Canonical Polyadic (CP) decomposition, TT bond dimensions depend strictly on the permutation of the tensor modes $\pi = (\pi(1), \dots, \pi(N))$. An arbitrary permutation can cause the maximum bond dimension $\max_k r_k$ to increase substantially, leading to parameter explosion.

Finding the optimal permutation among all $N!$ possibilities is combinatorial (NP-hard).

### 2.2 Limitation: Myopic Greedy Search
To avoid combinatorial search, Tichavsky and Straka (2025) proposed greedy heuristics:
- **Algorithm 3 (`ttsvdopt`)**: Left-to-right greedy search for full tensors.
- **Algorithm 4 (`ttcrossopt`)**: Left-to-right greedy search for black-box implicit functions using cross-approximation.

Both methods operate with **beam width $K=1$**:
1. Test placing each remaining mode at position $k$.
2. Choose the candidate minimizing the immediate cut rank $r_k$, using the bond value $v_k = S(r_k, r_k)$ as a tie-breaker.
3. Fix mode $k$ permanently and move to step $k+1$.

**Theoretical Defect**:
Because the choice is evaluated only on the local boundary cut $r_k$, greedy search has zero lookahead regarding future cuts. For structured tensors (such as block-diagonal systems), an immediate choice that minimizes early ranks frequently causes variables from different blocks to become interleaved. Once interleaved, middle bond dimensions explode to 8, 9, or 10.

**Published Findings in the Paper**:
- **Table I (Block-Diagonal $Q$, $N=20$)**: Over 1,000 random permutations in MATLAB `tt-toolbox`, Algorithm 4 achieved the optimal maximum bond dimension ($7$) in only **2 out of 1,000 trials (0.2%)**, remaining trapped in ranks 8, 9, or 10 in 99.8% of runs.
- **Table II (Tridiagonal $Q$, $N=20$)**: Algorithm 4 achieved a minimum of **1,512.0** parameters and an average of **2,397.8** parameters, failing to achieve the optimal rank in 74.2% of trials.

---

## 3. Proposed Novelty: Multi-Candidate Beam Search

To mitigate greedy entrapment, we implement **Multi-Candidate Beam Search Mode Ordering**:
- For full tensors: `tt_beam_search`
- For black-box implicit functions: `tt_cross_beam_search`

Instead of retaining only a single candidate path ($K=1$), Beam Search maintains a beam of the top-$K$ candidate partial orderings simultaneously.

### 3.1 Algorithm Description
1. **Hypothesis State**: At step $k$, each hypothesis in beam $\mathcal{B}_k$ contains:
   $$\mathcal{H} = \left( \pi_{1:k},\; \mathcal{R}_k,\; \mathcal{C}_k,\; [r_1, \dots, r_k],\; [v_1, \dots, v_k] \right)$$
   where $\pi_{1:k}$ is the ordered prefix, $\mathcal{R}_k$ is the set of remaining unassigned modes, and $\mathcal{C}_k$ is the contracted remaining core state.
2. **Expansion via Cyclic Shifts**: Every hypothesis $\mathcal{H} \in \mathcal{B}_k$ evaluates candidate modes $m \in \mathcal{R}_k$ using cyclic shifts matching Algorithm 4 line 8.
3. **Canonical Gauge-Invariant Bond Values (Algorithm 2)**: Truncated TT-cross rounding computes the canonical bond value $v_k = S(r_k, r_k)$ via right-to-left QR orthogonalization followed by left-to-right SVD (`get_canonical_bond_values`). This ensures singular values reflect the true global tensor matricization $\mathbf{X}_{(k)}$ without local core gauge bias.
4. **Multi-Criteria Hierarchical Pruning**: Candidates are ranked lexicographically:
   - For $K=1$: $\text{Cost}(\mathcal{H}) = (r_k, v_k)$ (strictly equivalent to Algorithm 4).
   - For $K \ge 2$: $\text{Cost}(\mathcal{H}) = (\max_{j \le k} r_j,\; \sum_{j \le k} r_j,\; r_k,\; v_k)$.
5. **Exact $K=1$ Equivalence**: When $K=1$, Beam Search produces the **exact same sequence of decisions, permutations, and bond ranks as Algorithm 4** down to machine precision across all random seeds (verified by unit test `test_beam_width_1_exact_match_algorithm4`).

### 3.2 Computational Complexity
- Outer loop runs $N - 1$ steps.
- At step $k$, evaluating at most $K \cdot (N - k)$ candidates requires $K \cdot \frac{N(N-1)}{2}$ evaluations.
- Time complexity is $\mathcal{O}(K \cdot N^2 \cdot \text{cost(eval)})$.
- For any constant beam width $K$, the algorithm retains the **exact same asymptotic complexity $\mathcal{O}(N^2)$ as the paper's greedy algorithm**.

---

## 4. Empirical Evaluation

For a consolidated comparison of all saved benchmark runs, rank-validation
notes, presentation-ready conclusions, and anticipated Q&A, see
[PRESENTATION_RESULTS_FAQ.md](./PRESENTATION_RESULTS_FAQ.md).

### 4.1 100-Trial Python/Teneva TT-Cross Benchmark ($N=20$)

> **Qualified finding**: In the paired 100-trial varying-Q experiment, exact cut-rank re-evaluation found 13 beam wins, 84 ties, and 3 losses for maximum bond rank (paired sign test on non-ties: $p=0.021$). The mean rank difference was small, and its 95% confidence interval includes zero. This is evidence on this sampled problem family, not proof of universal superiority or a direct reproduction of the paper.

Quadratic forms $V(x) = x^T Q x$ were discretized on a grid of 9 points per mode ($-2:0.5:2$), representing $9^{20} \approx 1.21 \times 10^{19}$ potential values. Each trial used a new Q and a shared initial permutation for greedy and beam. The existing generator averages a full random block with its transpose; this does not match the paper's stated lower-triangle-then-mirror generation with variance-one entries.

Teneva TT-cross selected each ordering ($\delta=10^{-4}$). The final ordering's exact ranks were independently recomputed from Q using $r_k=\operatorname{rank}(Q_{\pi(1:k),\pi(k+1:N)})+2$. This corrects one beam outlier from reported rank 11 to exact rank 12. The varying-Q result file does not contain held-out accuracy measurements.

| Exact Maximum Bond Dimension | Published Table I (1,000 trials, MATLAB)* | Greedy (100 paired varying-Q trials, `teneva`) | Beam Search ($K=2$, same trials) |
| :---: | :---: | :---: | :---: |
| Rank 7 | 2 (0.2%) | 74 (74.0%) | 82 (82.0%) |
| Rank 8 | 223 (22.3%) | 17 (17.0%) | 13 (13.0%) |
| Rank 9 | 749 (74.9%) | 9 (9.0%) | 4 (4.0%) |
| Rank 10 | 26 (2.6%) | 0 | 0 |
| Rank 11 | 0 | 0 | 0 |
| Rank 12 | — | 0 | 1 (1.0%) |
| Mean maximum rank | — | 7.35 | 7.26 |
| Mean exact parameters | 5,673.9* | 4,237.2 | 4,165.2 |
| Paired rank outcomes | — | — | 13 wins / 84 ties / 3 losses |
| Mean parameter difference (beam - greedy) | — | — | -72 (-1.7%); 95% CI [-236, 92] |

*\*The paper's reported Table I average and the Python results are not directly comparable: the paper does not publish its Q matrix or seeds, uses MATLAB TT-Toolbox, and our varying-Q generator differs from its stated entry-generation procedure. The 100-trial Python result uses independently generated Q matrices, not one fixed problem instance.*

See [PRESENTATION_RESULTS_FAQ.md](./PRESENTATION_RESULTS_FAQ.md) for all saved benchmark comparisons, confidence intervals, limitations, and presentation Q&A.

### TT-Cross Package Comparison

The N=8 black-box benchmark can run the same greedy and beam-search trials with either Teneva or TTpy. Both runs use the same seed and generated starting permutations; each package uses its own cross implementation, so runtime and approximation ranks are package-level results rather than a claim of identical internal algorithms. The aggregate output includes maximum-bond counts, optimal-hit rates, parameter counts, and runtime, with TTpy-minus-Teneva deltas.

The TTpy dependency is pinned to the upstream pure-Python implementation. This avoids PyPI's older `ttpy` 1.x source distribution, whose API differs from the current TTpy package.

```bash
python experiments/run_tt_package_comparison.py --trials 5 --beam-width 3
```

Increase `--trials` for a more stable comparison. Results are written to `experiments/results/tt_package_comparison.json`; separate backend results are also saved as `ttcross_benchmark_results.json` and `ttcross_ttpy_benchmark_results.json`.

The checked-in five-trial run used $N=8$, beam width 3, tolerance $10^{-6}$, and seed 42. Both backends selected the same ranks and parameter counts on these trials; TTpy was faster in this run:

| Backend | Greedy max bond / params | Greedy sec/trial | Beam max bond / params | Beam sec/trial |
| :--- | :---: | :---: | :---: | :---: |
| Teneva 0.14.11 | 5 / 846 | 2.735 | 4 / 594 | 6.077 |
| TTpy 2.0.0rc3 | 5 / 846 | 1.063 | 4 / 594 | 2.494 |

TTpy's average runtime was 61.1% lower for greedy and 59.0% lower for beam search on this machine. These are small-sample, hardware-dependent measurements, not general performance guarantees. Raw outputs: [`tt_package_comparison.json`](experiments/results/tt_package_comparison.json), [`ttcross_ttpy_benchmark_results.json`](experiments/results/ttcross_ttpy_benchmark_results.json).

#### Varying-Q Benchmark Paired Outcomes
- **Beam lower exact maximum rank** in 13 trials, tied in 84, and higher in 3. The exact-rank win trials are `[3, 7, 12, 13, 16, 24, 26, 47, 53, 63, 71, 80, 96]`; the losses are `[0, 2, 59]`.
- **Paired sign test** on the 16 non-tied trials: $p=0.021$. This is evidence for this benchmark sample, not proof of universal superiority.
- **Validation limitation**: this varying-Q result file does not record held-out errors. Do not attribute the fixed-Q Teneva held-out errors shown elsewhere to these 100 trials.
- Raw trial data: [table1_100trials_reproduction.json](./experiments/results/table1_100trials_reproduction.json); exact-rank audit and presentation guidance: [PRESENTATION_RESULTS_FAQ.md](./PRESENTATION_RESULTS_FAQ.md).

---

### 4.2 Analytical Rank Oracle Study ($N=20$, 1,000 Independent Trials)
To evaluate mode ordering combinatorics in isolation from cross-interpolation sampling noise, we implemented an analytical rank oracle based on submatrix ranks:
$$r_k = \text{rank}(Q_{\pi(1:k), \pi(k+1:N)}) + 2$$

#### Block-Diagonal $Q$ ($N=20$, two $10 \times 10$ blocks)
- Theoretical natural optimum: Max Bond = 7, Total Parameters = 3,906.
- Under noiseless algebraic cuts, both Greedy ($K=1$) and Beam Search ($K=5$) cleanly partition the blocks in 100.0% of trials (1,000/1,000).

#### Tridiagonal $Q$ ($N=20$, Bandwidth 1)
- Theoretical natural optimum: Max Bond = 4, Total Parameters = 1,818.
- Published paper baseline (Algorithm 4): Minimum Parameters = 1,512.0, Average Parameters = 2,397.8.

| Method | Mean Max Bond | Mean Parameters | Min Parameters | Parameter Savings vs Greedy | Mean Time / Trial |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 9.51 | 7,877.0 | 5,616 | - | - |
| Greedy ($K=1$, Oracle) | 4.00 | 2,196.0 | 2,196 | Baseline | 4.8 ms |
| **Beam Search ($K=3$, Proposed)** | **4.00** | **1,818.0** | **1,818** | **-378 parameters / model** | 13.0 ms |
| **Beam Search ($K=5$, Proposed)** | **4.00** | **1,818.0** | **1,818** | **-378 parameters / model** | 23.4 ms |

---

### 4.3 Full-Tensor Dense SVD Benchmark ($N=8$, 50 Trials)
- Evaluated on dense full tensors ($N=8$, two $4 \times 4$ blocks, grid size 3).
- Theoretical optimum: Max Bond = 4, Total Parameters = 330.

| Method | Mean Max Bond | Optimal Rank Hit Rate ($\le 4$) | Mean Parameters | Parameter Reduction | Mean Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 5.52 | 0.0% | 590.4 | 0.0% | - |
| Greedy ($K=1$, Paper Alg 3) | 5.00 | **0.0%** (0/50) | 470.0 | 20.4% | 228 ms |
| Beam Search ($K=2$) | 5.00 | 0.0% | 470.0 | 20.4% | 362 ms |
| Beam Search ($K=3$) | 5.00 | 0.0% | 470.0 | 20.4% | 513 ms |
| **Beam Search ($K=5$, Proposed)** | **4.00** | **100.0%** (50/50) | **330.0** | **44.1%** | 719 ms |

In all trials, reconstruction error satisfied $\|\mathcal{T} - \mathcal{T}_{\text{rec}}\|_F / \|\mathcal{T}\|_F < 2 \times 10^{-14}$, verifying machine precision.

---

## 5. Repository Structure

```
TT_decomposition/
├── src/
│   ├── __init__.py           # Package exports
│   ├── tt_core.py            # TT reconstruction, parameter counting, canonical TT-rounding (Alg 2)
│   ├── algorithms.py         # Alg 1 (tt_svd), Alg 3 (tt_svd_opt), Alg 4 (tt_cross_opt), and Beam Search
│   └── models.py             # Paper-consistent quadratic form generators (Gaussian below diagonal)
├── tests/
│   └── test_correctness.py   # 6 automated unit tests validating SVD, rounding, K=1 match, and optimization
├── experiments/
│   ├── run_table1_reproduction.py     # Checkpointed TT-cross benchmark with held-out error evaluation
│   ├── run_analytical_oracle_study.py # 1000-trial analytical rank oracle study (N=20)
│   ├── run_benchmarks.py              # 50-trial full-tensor dense benchmark (N=8)
│   ├── run_ttcross_benchmark.py       # Functional black-box TT-cross benchmark (N=8)
│   ├── run_tt_package_comparison.py   # Paired Teneva vs TTpy TT-cross benchmark
│   └── results/
│       ├── table1_100trials_reproduction.json # Committed raw trial data for N=20 benchmark
│       ├── analytical_oracle_results.json     # Committed raw results for N=20 oracle study
│       ├── benchmark_results.json             # Committed raw results for N=8 dense benchmark
│       ├── ttcross_benchmark_results.json     # N=8 Teneva TT-cross results
│       ├── ttcross_ttpy_benchmark_results.json # N=8 TTpy TT-cross results
│       └── tt_package_comparison.json          # Paired package comparison and deltas
├── requirements.txt          # Dependencies (numpy, scipy, teneva, TTpy)
├── EXPLANATION_AND_DEFENSE_GUIDE.md  # Comprehensive project report and viva defense guide
└── README.md
```

---

## 6. Execution Instructions

### 6.1 Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### 6.2 Run Automated Test Suite
Executes 6 unit tests validating exact reconstruction, TT-rounding fidelity, Algorithm 3, K=1 exact equivalence, and TT-cross optimization:
```bash
python tests/test_correctness.py
```

### 6.3 Run Paired Teneva/TTpy TT-Cross Comparison ($N=8$)
Runs the same random permutations with both cross backends, comparing greedy and beam search:
```bash
python experiments/run_tt_package_comparison.py --trials 5 --beam-width 3
```

### 6.4 Run TT-Cross Benchmark ($N=20$ Modes, Checkpointed)
Executes the parallel black-box TT-cross benchmark with held-out test evaluation:
```bash
# Run 20-trial pilot benchmark
python experiments/run_table1_reproduction.py --trials 20 --workers 10

# Run full 100-trial benchmark
python experiments/run_table1_reproduction.py --trials 100 --workers 10
```

### 6.5 Run Analytical Oracle Study ($N=20$, 1,000 Trials)
Evaluates algebraic permutation ranks across 1,000 trials:
```bash
python experiments/run_analytical_oracle_study.py
```

### 6.6 Run Fixed-Q TTpy DMRG-Cross Benchmark ($N=20$)
Runs paired greedy and beam searches with TTpy `dmrg_cross` on the same fixed
matrix and permutations. Every candidate TT is rounded and checked on 1,000
held-out points; deterministic seed, pivoting, and fixed-rank retries are used
when a candidate is non-finite or inaccurate. The run records TTpy version,
accepted settings, retries, and validation errors. The pre-existing unvalidated
result file is intentionally not reused:
```bash
python experiments/run_ttpy_dmrg_table1.py --trials 100 --workers 10 \
  --output experiments/results/table1_100trials_ttpy_dmrg_cross_validated.json
```

### 6.7 Run Dense Full-Tensor Benchmark ($N=8$, 50 Trials)
Runs the dense TT-SVD benchmark comparing Greedy vs. Beam Search across beam widths $K \in \{1, 2, 3, 5\}$:
```bash
python experiments/run_benchmarks.py
```
