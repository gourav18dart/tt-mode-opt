# Optimizing Mode Order in Tensor Train Decomposition: Beam Search Enhancement

This repository provides an implementation and extension of the research paper:

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

Finding the optimal permutation among all $N!$ possibilities is NP-hard.

### 2.2 Limitation No. 1: Failure of Myopic Greedy Optimization
To avoid combinatorial search, Tichavsky and Straka (2025) proposed Algorithm 3 (`ttsvdopt`), a left-to-right greedy heuristic ($K=1$). At each step $k$:
1. It tests placing each remaining mode at position $k$.
2. It chooses the candidate minimizing the immediate cut rank $r_k$, using the bond value $v_k = S(r_k, r_k)$ as a tie-breaker.
3. It fixes mode $k$ permanently and moves to step $k+1$.

**Theoretical Defect**:
Because the choice is irreversible and evaluated only on the local boundary cut $r_k$, Algorithm 3 has no lookahead regarding future cuts. For structured tensors (such as block-diagonal systems), an immediate choice that minimizes $r_1$ frequently forces strongly coupled variables from different blocks to interleave later in the train. This causes the bottleneck bond dimensions in the middle of the train to inflate.

**Empirical Evidence in the Paper**:
In Table I of the paper, when evaluated on a block-diagonal quadratic system ($N=20$) over 1,000 independent random permutations:
- The optimal maximum bond dimension ($7$) was achieved in only **2 out of 1,000 trials (0.2%)**.
- In **99.8%** of trials, the greedy approach became trapped in suboptimal permutations with maximum bond dimensions of 8, 9, or 10.

---

## 3. Proposed Novelty: Multi-Candidate Beam Search

To resolve Limitation No. 1, we propose **Beam Search Mode Ordering** (`tt_beam_search`).

Instead of retaining only a single candidate path ($K=1$), Beam Search maintains a beam of the top-$K$ candidate partial orderings simultaneously.

### 3.1 Algorithm Description
1. **Hypothesis State**: At step $k$, each hypothesis in beam $\mathcal{B}_k$ contains:
   $$\mathcal{H} = \left( \pi_{1:k},\; \mathcal{R}_k,\; \mathcal{C}_k,\; [r_1, \dots, r_k],\; [v_1, \dots, v_k] \right)$$
   where $\pi_{1:k}$ is the ordered prefix, $\mathcal{R}_k$ is the set of remaining unassigned modes, and $\mathcal{C}_k$ is the contracted remaining tensor.
2. **Expansion**: Every hypothesis $\mathcal{H} \in \mathcal{B}_k$ expands all candidate modes $m \in \mathcal{R}_k$. Truncated singular value decomposition computes $r_k$ and the bond value $v_k = S(r_k, r_k)$.
3. **Multi-Criteria Hierarchical Pruning**: Candidates are ranked lexicographically by:
   $$\text{Cost}(\mathcal{H}) = \left( \max_{j \le k} r_j,\; \sum_{j \le k} r_j,\; v_k \right)$$
   - Priority 1: Minimizes the peak bottleneck TT rank.
   - Priority 2: Minimizes cumulative rank (surrogate for total parameter count).
   - Priority 3: Minimizes the residual singular value at the current cut (the paper's tie-breaker).
4. **Beam Pruning**: The top-$K$ diverse candidates are preserved for step $k+1$.

### 3.2 Theoretical Properties
- When $K=1$, the algorithm is strictly equivalent to the paper's Algorithm 3.
- When $K > 1$, alternative branches are explored in parallel, preventing premature entrapment in local minima.
- Time complexity is $O(K \cdot N^2 \cdot \text{cost(SVD)})$, maintaining polynomial scaling while substantially improving solution quality.

---

## 4. Empirical Evaluation

### 4.1 Benchmark Setup
Experiments were conducted following Section VI of Tichavsky & Straka (2025) using quadratic forms $V(x) = x^T Q x$ sampled over hyper-rectangular grids. Per Section III of the paper, grids with $\ge 3$ points preserve polynomial degree-2 ranks identically to denser grids.

We evaluate across 50 independent Monte Carlo trials under random input mode permutations with fixed random seeds for deterministic reproducibility.

### 4.2 Benchmark Results

#### Benchmark A: Block-Diagonal $Q$ ($N=8$, two $4 \times 4$ blocks)
- Theoretical natural optimum: Max Bond Dimension = 4, Total Parameters = 330.

| Method | Mean Max Bond | Optimal Rank Hit Rate ($\le 4$) | Mean Parameters | Parameter Reduction | Mean Runtime / Trial |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 5.52 | 0.0% | 590.4 | 0.0% | - |
| Greedy ($K=1$, Paper) | 5.00 | **0.0%** (0/50) | 470.0 | 20.4% | 228 ms |
| Beam Search ($K=2$) | 5.00 | 0.0% | 470.0 | 20.4% | 362 ms |
| Beam Search ($K=3$) | 5.00 | 0.0% | 470.0 | 20.4% | 513 ms |
| **Beam Search ($K=5$, Proposed)** | **4.00** | **100.0%** (50/50) | **330.0** | **44.1%** | 719 ms |

**Key Finding**: The paper's Algorithm 3 ($K=1$) achieved a 0.0% success rate, remaining trapped at maximum bond dimension 5 in all trials. Beam Search with $K=5$ achieved a **100.0% success rate**, recovering the global mathematical optimum in every trial within 719 ms.

#### Benchmark B: Tridiagonal $Q$ ($N=8$)
- Theoretical natural optimum: Max Bond Dimension = 3, Total Parameters = 260.

| Method | Mean Max Bond | Optimal Rank Hit Rate ($\le 3$) | Mean Parameters | Parameter Reduction | Mean Runtime / Trial |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 4.92 | 0.0% | 417.3 | 0.0% | - |
| Greedy ($K=1$, Paper) | 3.00 | 100.0% | 260.0 | 37.7% | 177 ms |
| **Beam Search ($K=5$, Proposed)** | **3.00** | **100.0%** | **260.0** | **37.7%** | 485 ms |

In all trials, the relative reconstruction error between the permuted tensor and the reconstructed TT decomposition was bounded by $2 \times 10^{-14}$, verifying machine-precision numerical exactness.

---

## 5. Repository Structure

```
TT_decomposition/
├── src/
│   ├── __init__.py           # Package exports
│   ├── tt_core.py            # TT reconstruction, parameter counting, TT-rounding
│   ├── algorithms.py         # Algorithm 1 (tt_svd), Algorithm 3 (tt_svd_opt), Novelty (tt_beam_search)
│   └── models.py             # Quadratic form generators with explicit random seeds
├── tests/
│   └── test_correctness.py   # Unit tests validating numerical precision and rank properties
├── experiments/
│   ├── run_benchmarks.py     # Reproducible Monte Carlo benchmarking suite
│   └── results/
│       └── benchmark_results.json  # Raw evaluation metrics
├── legacy/                   # Archived preliminary test files
├── requirements.txt          # Minimal dependency requirements
├── tt_core.py                # Backward-compatibility facade
└── README.md
```

---

## 6. Execution Instructions

### 6.1 Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### 6.2 Run Correctness Unit Tests
Executes unit tests validating TT decomposition, exact tensor reconstruction, and mathematical rank bounds:
```bash
python tests/test_correctness.py
```

### 6.3 Run Monte Carlo Benchmarks
Executes the full comparative benchmark suite across 50 random trials and outputs summary tables:
```bash
python experiments/run_benchmarks.py
```
