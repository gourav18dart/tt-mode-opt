"""
Monte Carlo Benchmark Suite comparing Natural, Random, Greedy (Algorithm 3),
and Beam Search (Novelty) mode ordering on structured tensors.
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

from src.models import (
    build_quadratic_tensor,
    create_block_diagonal_Q,
    create_tridiagonal_Q
)
from src.tt_core import (
    tt_num_params,
    tt_rel_error
)
from src.algorithms import (
    tt_svd,
    tt_beam_search
)


def run_single_benchmark(
    name,
    Q,
    grid,
    n_trials=50,
    beam_widths=(1, 2, 3, 5),
    tol=1e-9,
    seed=42
):
    """
    Run Monte Carlo trials for a specific coupling matrix Q across multiple beam widths.
    """
    rng = np.random.RandomState(seed)
    N = Q.shape[0]
    shape = [len(grid)] * N
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name} (N={N}, Grid points={len(grid)})")
    print(f"Trials: {n_trials}, Beam Widths: {beam_widths}, Tol: {tol}")
    print(f"{'='*60}")
    
    # 1. Natural Tensor (Theoretical Lower Bound)
    t0 = time.time()
    T_nat = build_quadratic_tensor(Q, grid)
    build_time = time.time() - t0
    print(f"Built full tensor: shape={T_nat.shape}, elements={T_nat.size:,} in {build_time:.2f}s")
    
    cores_nat, nat_bonds, _ = tt_svd(T_nat, tol=tol)
    nat_max = max(nat_bonds)
    nat_params = tt_num_params(nat_bonds, shape)
    print(f"Natural Ordering: bonds={nat_bonds}, max={nat_max}, params={nat_params}")
    
    benchmark_data = {
        "name": name,
        "N": N,
        "grid_points": len(grid),
        "natural": {
            "bonds": nat_bonds,
            "max": nat_max,
            "params": nat_params
        },
        "random": {"max_bonds": [], "params": []},
        "methods": {k: {"max_bonds": [], "params": [], "times": [], "rel_errors": []} for k in beam_widths}
    }
    
    for trial in range(n_trials):
        # Generate random permutation of modes
        perm = rng.permutation(N).tolist()
        T_shuffled = np.transpose(T_nat, axes=perm)
        
        # 1. Random baseline (no optimization)
        cores_rand, b_rand, _ = tt_svd(T_shuffled, tol=tol)
        p_rand = tt_num_params(b_rand, shape)
        benchmark_data["random"]["max_bonds"].append(max(b_rand))
        benchmark_data["random"]["params"].append(p_rand)
        
        # 2. Test each beam width K
        for K in beam_widths:
            t_start = time.time()
            order, cores, bonds, _ = tt_beam_search(T_shuffled, tol=tol, beam_width=K)
            elapsed = time.time() - t_start
            
            p = tt_num_params(bonds, [shape[i] for i in order])
            rel_err = tt_rel_error(T_shuffled, cores, order=order)
            
            benchmark_data["methods"][K]["max_bonds"].append(max(bonds))
            benchmark_data["methods"][K]["params"].append(p)
            benchmark_data["methods"][K]["times"].append(elapsed)
            benchmark_data["methods"][K]["rel_errors"].append(rel_err)
            
        if (trial + 1) % 10 == 0 or (trial + 1) == n_trials:
            print(f"  Completed trial {trial + 1}/{n_trials}...")
            
    # Print summary
    rand_max = np.array(benchmark_data["random"]["max_bonds"])
    rand_par = np.array(benchmark_data["random"]["params"])
    print(f"\n--- Results Summary for {name} ---")
    print(f"Random Baseline: Avg Max Bond = {rand_max.mean():.2f}, Avg Params = {rand_par.mean():.1f}")
    
    for K in beam_widths:
        m_arr = np.array(benchmark_data["methods"][K]["max_bonds"])
        p_arr = np.array(benchmark_data["methods"][K]["params"])
        t_arr = np.array(benchmark_data["methods"][K]["times"])
        err_arr = np.array(benchmark_data["methods"][K]["rel_errors"])
        
        counts = Counter(m_arr)
        opt_rate = 100.0 * np.sum(m_arr <= nat_max) / n_trials
        red = 100.0 * (1.0 - p_arr.mean() / rand_par.mean())
        
        label = "Greedy (K=1, Paper)" if K == 1 else f"Beam Search (K={K}, Ours)"
        print(f"\n{label}:")
        print(f"  Avg Max Bond: {m_arr.mean():.2f} (Counts: {dict(sorted(counts.items()))})")
        print(f"  Optimal Max Bond Hit Rate (<= {nat_max}): {opt_rate:.1f}%")
        print(f"  Avg Parameters: {p_arr.mean():.1f} (Reduction vs Random: {red:.1f}%)")
        print(f"  Avg Time per trial: {t_arr.mean()*1000:.1f} ms")
        print(f"  Max Reconstruction Error: {err_arr.max():.2e}")
        
    return benchmark_data


def main():
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # 5 points: exact polynomial rank
    
    # ---------------------------------------------------------
    # Case 1: Block-Diagonal Q (4 + 4 = 8 modes)
    # ---------------------------------------------------------
    N1 = 8
    Q_block = create_block_diagonal_Q(N1, [4, 4], seed=101)
    res_block = run_single_benchmark(
        "Block-Diagonal Q (4+4)",
        Q_block,
        grid,
        n_trials=50,
        beam_widths=(1, 2, 3, 5),
        tol=1e-9
    )
    
    # ---------------------------------------------------------
    # Case 2: Tridiagonal Q (N = 8 modes)
    # ---------------------------------------------------------
    N2 = 8
    Q_tri = create_tridiagonal_Q(N2, seed=202)
    res_tri = run_single_benchmark(
        "Tridiagonal Q (N=8)",
        Q_tri,
        grid,
        n_trials=50,
        beam_widths=(1, 2, 3, 5),
        tol=1e-9
    )
    
    # Save combined results to experiments/results/
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_results.json"
    
    # Helper to convert numpy ints/floats for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_file, "w") as f:
        json.dump({"block": res_block, "tri": res_tri}, f, indent=2, default=convert_for_json)
        
    print(f"\nAll benchmark results successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
