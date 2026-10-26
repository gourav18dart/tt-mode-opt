# Optimizing Mode Order in Tensor Train Decomposition: Beam Search Enhancement

This repository provides an implementation, replication, and extension of the research paper:

> Petr Tichavsky and Ondrej Straka, "Optimizing the Order of Modes in Tensor Train Decomposition," *IEEE Signal Processing Letters*, vol. 32, pp. 1361-1365, 2025.

---

## 1. Mathematical Formulation

### 1.1 Tensor Train Decomposition
Let $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times \dots \times I_N}$ be an order-$N$ tensor. The Tensor Train (TT) decomposition represents individual tensor elements as a contraction of low-order core tensors:

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

## 2. Mode Ordering and Limitation of Existing Literature

### 2.1 Mode-Order Dependence
Unlike Canonical Polyadic (CP) decomposition, TT bond dimensions depend strictly on the permutation of the tensor modes $\pi = (\pi(1), \dots, \pi(N))$. An arbitrary permutation can cause the maximum bond dimension $\max_k r_k$ to increase substantially, leading to parameter explosion.

Finding the optimal permutation among all $N!$ possibilities is combinatorial (NP-hard).

### 2.2 Limitation: Failure of Myopic Greedy Optimization
To avoid combinatorial search, Tichavsky and Straka (2025) proposed greedy heuristics:
- **Algorithm 3 (`ttsvdopt`)**: Left-to-right greedy search for full tensors.
- **Algorithm 4 (`ttcrossopt`)**: Left-to-right greedy search for black-box implicit functions using cross-approximation.

Both methods operate with **beam width $K=1$**:
1. Test placing each remaining mode at position $k$.
2. Choose the candidate minimizing the immediate cut rank $r_k$, using the bond value $v_k = S(r_k, r_k)$ as a tie-breaker.
3. Fix mode $k$ permanently and move to step $k+1$.

**Theoretical Defect**:
Because the choice is evaluated only on the local boundary cut $r_k$, greedy search has zero lookahead regarding future cuts. For structured tensors (such as block-diagonal systems), an immediate choice that minimizes early ranks frequently causes variables from different blocks to become interleaved. Once interleaved, middle bond dimensions explode to 8, 9, or 10.

**Empirical Evidence in the Published Paper (Table I)**:
In Section VI of the paper, when evaluated on a block-diagonal quadratic system ($N=20$) over 1,000 independent random permutations:
- The optimal maximum bond dimension ($7$) was achieved in only **2 out of 1,000 trials (0.2%)**.
- In **99.8%** of trials, the greedy approach became trapped in suboptimal permutations with maximum bond dimensions of 8, 9, or 10.

---

## 3. Proposed Novelty: Multi-Candidate Beam Search

To eliminate greedy entrapment, we developed **Multi-Candidate Beam Search Mode Ordering**:
- For full tensors: `tt_beam_search`
- For black-box implicit functions: `tt_cross_beam_search`

Instead of retaining only a single candidate path ($K=1$), Beam Search maintains a beam of the top-$K$ candidate partial orderings simultaneously.

### 3.1 Algorithm Description
1. **Hypothesis State**: At step $k$, each hypothesis in beam $\mathcal{B}_k$ contains:
   $$\mathcal{H} = \left( \pi_{1:k},\; \mathcal{R}_k,\; \mathcal{C}_k,\; [r_1, \dots, r_k],\; [v_1, \dots, v_k] \right)$$
   where $\pi_{1:k}$ is the ordered prefix, $\mathcal{R}_k$ is the set of remaining unassigned modes, and $\mathcal{C}_k$ is the contracted remaining core state.
2. **Expansion**: Every hypothesis $\mathcal{H} \in \mathcal{B}_k$ expands candidate modes $m \in \mathcal{R}_k$. Truncated SVD or TT-cross rounding computes $r_k$ and the bond value $v_k$.
3. **Multi-Criteria Hierarchical Pruning**: Candidates are ranked lexicographically by:
   $$\text{Cost}(\mathcal{H}) = \left( \max_{j \le k} r_j,\; \sum_{j \le k} r_j,\; v_k \right)$$
   - Priority 1: Minimizes the peak bottleneck TT rank.
   - Priority 2: Minimizes cumulative rank (surrogate for total parameter count).
   - Priority 3: Minimizes the residual singular value at the current cut (paper's tie-breaker).
4. **Beam Pruning**: The top-$K$ diverse candidates are preserved for step $k+1$.

### 3.2 Computational Complexity
- Outer loop runs $N - 1$ steps.
- At step $k$, evaluating at most $K \cdot (N - k)$ candidates requires $K \cdot \frac{N(N-1)}{2}$ evaluations.
- Time complexity is $\mathcal{O}(K \cdot N^2 \cdot \text{cost(SVD/cross)})$.
- For any constant beam width $K$, the algorithm retains the **exact same asymptotic complexity $\mathcal{O}(N^2)$ as the paper's greedy algorithm**.

---

## 4. Empirical Evaluation

### 4.1 Recreated Table I Benchmark ($N=20$, 100 Independent Trials, TT-Cross)
Following Section VI of the paper, we benchmarked the black-box quadratic form $V(x) = x^T Q x$ discretized on a grid of 9 points per mode ($-2:0.5:2$), representing $9^{20} \approx 1.21 \times 10^{19}$ potential values.

$Q$ has two independent $10 \times 10$ Gaussian positive-definite blocks ($Q_1, Q_2$). We ran 100 independent trials across 10 parallel CPU workers, comparing faithful Algorithm 4 (including bond-value tie-breaking) against Beam Search ($K=2$):

| Maximum Bond Dimension ($r_{\max}$) | Published Table I (1,000 trials, MATLAB) | Paper Algorithm 4 (100 trials, `teneva`) | Our Beam Search ($K=2$) (100 trials, `teneva`) |
| :---: | :---: | :---: | :---: |
| **Rank 7 (Optimal)** | **2 (0.2%)** | **74 (74.0%)** | **82 (82.0%)** |
| **Rank 8** | 223 (22.3%) | 17 (17.0%) | 13 (13.0%) |
| **Rank 9** | 749 (74.9%) | 9 (9.0%) | 4 (4.0%) |
| **Rank 10** | 26 (2.6%) | 0 (0.0%) | 0 (0.0%) |
| **Rank 11** | 0 (0.0%) | 0 (0.0%) | 1 (1.0%) |
| **Optimal Hit Rate ($r=7$)** | **0.2%** | **74.0%** | **82.0%** *(+8.0% gain)* |
| **Severe Degradation ($r \ge 9$)** | **77.5%** | **9.0%** | **5.0%** *(reduced by ~45%)* |
| **Average Parameters** | **5,673.9** | **4,235.9** $\pm$ 615.1 | **4,150.6** $\pm$ 669.0 *(saving 85.3 params)* |
| **Average Time per Trial** | — | **217.1 s** (~3.6 min) | **398.1 s** (~6.6 min) |

**Trial-by-Trial Breakdown**:
- In **13 trials**, Beam Search strictly beat Algorithm 4 in maximum bond dimension (e.g., reaching optimal rank 7 while Algorithm 4 was trapped at rank 8 or 9).
- In **84 trials**, both algorithms tied.
- In **3 trials**, Algorithm 4 had lower maximum bond dimension.
- Raw trial data: [`experiments/results/table1_100trials_reproduction.json`](experiments/results/table1_100trials_reproduction.json).

---

### 4.2 Analytical Rank Oracle Study ($N=20$, 1,000 Independent Trials)
To evaluate the combinatorial permutation search without TT-cross sampling noise, we evaluated mode ordering on the exact $N=20$ matrices using the algebraic rank oracle:
$$r_k = \text{rank}(Q_{\pi(1:k), \pi(k+1:N)}) + 2$$

#### Block-Diagonal $Q$ ($N=20$, two $10 \times 10$ blocks)
- Theoretical natural optimum: Max Bond = 7, Total Parameters = 3,906.

| Method | Mean Max Bond | Optimal Rank Hit Rate ($\le 7$) | Mean Parameters | Min Parameters | Mean Time / Trial |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 11.01 | 0.0% | 9,399.5 | 7,308 | - |
| Greedy ($K=1$, Oracle) | 7.00 | 100.0% (1000/1000) | 3,906.0 | 3,906 | 5.2 ms |
| **Beam Search ($K=5$, Proposed)** | **7.00** | **100.0% (1000/1000)** | **3,906.0** | **3,906** | 21.6 ms |

*Scientific Note: Under the noiseless analytical matrix rank oracle, both methods find the optimal block partition. This study isolates the permutation combinatorics from cross-approximation sampling error.*

#### Tridiagonal $Q$ ($N=20$, Bandwidth 1)
- Theoretical natural optimum: Max Bond = 4, Minimum Parameters = 1,818.
- Published paper baseline (Algorithm 4): Mean Parameters = 2,525.0.

| Method | Mean Max Bond | Mean Parameters | Min Parameters | Parameter Savings vs Greedy | Mean Time / Trial |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 9.51 | 7,877.0 | 5,616 | - | - |
| Greedy ($K=1$, Oracle) | 4.00 | 2,196.0 | 2,196 | Baseline | 4.8 ms |
| **Beam Search ($K=3$, Proposed)** | **4.00** | **1,818.0** | **1,818** | **-378 parameters / model** | 13.0 ms |
| **Beam Search ($K=5$, Proposed)** | **4.00** | **1,818.0** | **1,818** | **-378 parameters / model** | 23.4 ms |

Beam Search achieves the theoretical global minimum of **1,818 parameters** across all 1,000 trials, saving an additional 378 parameters per model over Greedy.

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
│   ├── tt_core.py            # TT reconstruction, parameter counting, TT-rounding (Alg 2)
│   ├── algorithms.py         # Alg 1 (tt_svd), Alg 3 (tt_svd_opt), Alg 4 (tt_cross_opt), and Beam Search
│   └── models.py             # Quadratic form generators with explicit random seeds
├── tests/
│   └── test_correctness.py   # 5 automated unit tests validating SVD, rounding, and optimization
├── experiments/
│   ├── run_table1_reproduction.py     # 100-trial parallel Table I reproduction (N=20 TT-cross)
│   ├── run_analytical_oracle_study.py # 1000-trial analytical rank oracle study (N=20)
│   ├── run_benchmarks.py              # 50-trial full-tensor dense benchmark (N=8)
│   ├── run_ttcross_benchmark.py       # Functional black-box TT-cross benchmark
│   └── results/
│       ├── table1_100trials_reproduction.json # Committed raw results for 100-trial N=20 benchmark
│       ├── analytical_oracle_results.json     # Committed raw results for N=20 oracle study
│       ├── benchmark_results.json             # Committed raw results for N=8 dense benchmark
│       └── ttcross_benchmark_results.json     # Committed raw results for N=8 cross benchmark
├── requirements.txt          # Dependencies (numpy, scipy, teneva)
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
Executes 5 unit tests validating exact reconstruction, TT-rounding fidelity, Algorithm 3, and TT-cross optimization:
```bash
python tests/test_correctness.py
```

### 6.3 Run 100-Trial Table I Reproduction Benchmark ($N=20$ Modes, TT-Cross)
Executes the parallel 100-trial black-box TT-cross benchmark:
```bash
python experiments/run_table1_reproduction.py
```

### 6.4 Run Analytical Oracle Study ($N=20$, 1,000 Trials)
Evaluates algebraic permutation ranks across 1,000 trials:
```bash
python experiments/run_analytical_oracle_study.py
```

### 6.5 Run Dense Full-Tensor Benchmark ($N=8$, 50 Trials)
Runs the dense TT-SVD benchmark comparing Greedy vs. Beam Search across beam widths $K \in \{1, 2, 3, 5\}$:
```bash
python experiments/run_benchmarks.py
```
