"""
Exact replication of Section VI experiments from Tichavsky & Straka (2025):
  - Exact number of variables: N = 20
  - Exact matrices: 
      (a) Block-diagonal with two 10x10 blocks, Gaussian i.i.d. N(0, 1) entries below diagonal
      (b) Tridiagonal with Gaussian i.i.d. N(0, 1) off-diagonal entries
  - Exact sampling grid: (-2 : 0.5 : 2), yielding 9^20 elements (M = 9 points)
  - Exact number of trials: 1,000 independent trials
  - Exact metrics:
      - Distribution of maximum bond dimension (Tables I & II)
      - Initial complexity (random permutation parameters)
      - Optimized complexity (average and minimum parameters)
      - Optimal rank recovery rate
"""

import sys
import time
import json
from pathlib import Path
from collections import Counter
import numpy as np

# Ensure project root is in path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

N = 20
GRID_POINTS = 9  # Grid (-2:0.5:2) has 9 points
N_TRIALS = 1000


def get_tt_params(bonds, N=20, M=9):
    """
    Equation (1) from paper: sum_{i=1}^N r_{i-1} * M * r_i, r_0 = r_N = 1.
    """
    r = [1] + list(bonds) + [1]
    return sum(r[i] * M * r[i + 1] for i in range(N))


def evaluate_cut(Q, left, right, tol=1e-9):
    """
    Evaluates the exact bond dimension and bond value across the cut
    between variables in 'left' and variables in 'right'.
    For quadratic forms V(x) = x^T Q x, rank(cut) = rank(Q_AB) + 2.
    """
    Q_AB = Q[np.ix_(left, right)]
    s = np.linalg.svd(Q_AB, compute_uv=False)
    r = int(np.sum(s > tol * s[0])) if len(s) > 0 and s[0] > tol else 0
    v = float(s[r - 1]) if r > 0 else 0.0
    return r + 2, v


def evaluate_all_bonds(Q, order, tol=1e-9):
    """
    Computes the full vector of N-1 bond dimensions for a given mode ordering.
    """
    bonds = []
    for k in range(1, len(order)):
        r, _ = evaluate_cut(Q, order[:k], order[k:], tol=tol)
        bonds.append(r)
    return bonds


def beam_search_mode_ordering(Q, initial_perm, beam_width=1, tol=1e-9):
    """
    Optimizes the order of modes using Beam Search.
    When beam_width=1, reproduces the paper's greedy heuristic.
    When beam_width > 1, maintains K diverse trajectories.
    """
    Q_p = Q[np.ix_(initial_perm, initial_perm)]
    
    # Initial beam state
    beam = [{
        'order': [],
        'remaining': list(range(N)),
        'bonds': [],
        'cost': (0, 0, 0.0)
    }]
    
    for step in range(N - 1):
        candidates = []
        for hyp in beam:
            rem = hyp['remaining']
            for idx in rem:
                cand_left = hyp['order'] + [idx]
                cand_right = [m for m in rem if m != idx]
                r, v = evaluate_cut(Q_p, cand_left, cand_right, tol=tol)
                new_bonds = hyp['bonds'] + [r]
                # Hierarchical cost: (max_bond, sum_bonds, current_cut_singular_value)
                cost = (max(new_bonds), sum(new_bonds), v)
                candidates.append({
                    'order': cand_left,
                    'remaining': cand_right,
                    'bonds': new_bonds,
                    'cost': cost
                })
                
        # Sort candidates by cost
        candidates.sort(key=lambda h: h['cost'])
        
        # Keep top beam_width diverse hypotheses
        seen = set()
        beam = []
        for c in candidates:
            ord_key = tuple(c['order'])
            if ord_key not in seen:
                seen.add(ord_key)
                beam.append(c)
                if len(beam) >= beam_width:
                    break
                    
    # Final step: only 1 mode remains in each hypothesis
    best_hyp = beam[0]
    best_local_order = best_hyp['order'] + best_hyp['remaining']
    
    # Map back to original coordinate indices
    final_order = [initial_perm[i] for i in best_local_order]
    final_bonds = evaluate_all_bonds(Q, final_order, tol=tol)
    return final_order, final_bonds


def run_exact_experiment(name, Q, nat_max, nat_params, n_trials=1000, seed=42):
    """
    Runs 1,000 Monte Carlo trials on a 20-variable matrix Q.
    """
    rng = np.random.RandomState(seed)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name} (N={N}, Grid Points={GRID_POINTS}, Trials={n_trials})")
    print(f"Theoretical Optimum: Max Bond = {nat_max}, Parameters = {nat_params}")
    print(f"{'='*70}")
    
    # Pre-generate 1,000 random permutations
    permutations = [rng.permutation(N).tolist() for _ in range(n_trials)]
    
    # 1. Random baseline
    t0 = time.time()
    rand_max_bonds = []
    rand_params = []
    for perm in permutations:
        b = evaluate_all_bonds(Q, perm)
        rand_max_bonds.append(max(b))
        rand_params.append(get_tt_params(b))
    print(f"Evaluated 1,000 random permutations in {time.time()-t0:.2f}s")
    print(f"  Random Baseline: Avg Max Bond = {np.mean(rand_max_bonds):.2f}, Avg Params = {np.mean(rand_params):.1f}")
    
    # 2. Test Greedy (K=1) and Beam Search (K=2, 3, 5)
    beam_widths = [1, 2, 3, 5]
    results = {
        "name": name,
        "N": N,
        "n_trials": n_trials,
        "natural_max": nat_max,
        "natural_params": nat_params,
        "random": {
            "avg_max_bond": float(np.mean(rand_max_bonds)),
            "avg_params": float(np.mean(rand_params)),
            "min_params": int(np.min(rand_params)),
            "max_params": int(np.max(rand_params)),
            "counts": dict(sorted(Counter(rand_max_bonds).items()))
        },
        "methods": {}
    }
    
    for K in beam_widths:
        t_start = time.time()
        max_bonds = []
        params_list = []
        
        for perm in permutations:
            _, bonds = beam_search_mode_ordering(Q, perm, beam_width=K)
            max_bonds.append(max(bonds))
            params_list.append(get_tt_params(bonds))
            
        elapsed = time.time() - t_start
        counts = dict(sorted(Counter(max_bonds).items()))
        opt_hits = sum(1 for m in max_bonds if m <= nat_max)
        opt_rate = 100.0 * opt_hits / n_trials
        avg_p = float(np.mean(params_list))
        min_p = int(np.min(params_list))
        red = 100.0 * (1.0 - avg_p / np.mean(rand_params))
        
        label = "Greedy (K=1, Paper Algorithm 3/4)" if K == 1 else f"Beam Search (K={K}, Proposed)"
        print(f"\n{label}:")
        print(f"  Max Bond Distribution: {counts}")
        print(f"  Optimal Max Bond Hit Rate (<= {nat_max}): {opt_hits}/{n_trials} ({opt_rate:.1f}%)")
        print(f"  Parameters: Avg = {avg_p:.1f}, Min = {min_p} (Reduction: {red:.1f}%)")
        print(f"  Total Time for {n_trials} trials: {elapsed:.2f}s ({elapsed/n_trials*1000:.2f} ms/trial)")
        
        results["methods"][K] = {
            "label": label,
            "counts": counts,
            "opt_hits": opt_hits,
            "opt_rate": opt_rate,
            "avg_params": avg_p,
            "min_params": min_p,
            "reduction_percent": red,
            "total_time_sec": elapsed,
            "ms_per_trial": elapsed / n_trials * 1000
        }
        
    return results


def main():
    np.random.seed(42)
    
    # -------------------------------------------------------------
    # Case A: Block-Diagonal Q (Section VI, Subsection a)
    # Two blocks of size 10x10, Gaussian i.i.d. entries
    # -------------------------------------------------------------
    rng_block = np.random.RandomState(101)
    B1 = rng_block.randn(10, 10); B1 = (B1 + B1.T) / 2.0
    B2 = rng_block.randn(10, 10); B2 = (B2 + B2.T) / 2.0
    Q_block = np.zeros((N, N))
    Q_block[:10, :10] = B1
    Q_block[10:, 10:] = B2
    
    # Natural order parameters for block-diagonal:
    nat_bonds_block = [3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3]
    nat_max_block = 7
    nat_params_block = get_tt_params(nat_bonds_block)
    
    res_block = run_exact_experiment(
        "Block-Diagonal Q (two 10x10 blocks)",
        Q_block,
        nat_max=nat_max_block,
        nat_params=nat_params_block,
        n_trials=N_TRIALS,
        seed=1001
    )
    
    # -------------------------------------------------------------
    # Case B: Tridiagonal Q (Section VI, Subsection b)
    # Main diagonal zero, sub/super-diagonals Gaussian i.i.d. entries
    # -------------------------------------------------------------
    rng_tri = np.random.RandomState(202)
    Q_tri = np.zeros((N, N))
    off = rng_tri.randn(N - 1)
    for i in range(N - 1):
        Q_tri[i, i + 1] = off[i]
        Q_tri[i + 1, i] = off[i]
        
    nat_bonds_tri = [3] * (N - 1)
    nat_max_tri = 3
    nat_params_tri = get_tt_params(nat_bonds_tri)
    
    res_tri = run_exact_experiment(
        "Tridiagonal Q (N=20)",
        Q_tri,
        nat_max=nat_max_tri,
        nat_params=nat_params_tri,
        n_trials=N_TRIALS,
        seed=2002
    )
    
    # Save exact results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exact_paper_replication_results.json"
    
    with open(out_file, "w") as f:
        json.dump({"block_diagonal": res_block, "tridiagonal": res_tri}, f, indent=2)
        
    print(f"\nAll 1,000-trial results successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
