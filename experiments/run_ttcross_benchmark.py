"""
TT-Cross Mode Ordering Benchmark:
Compares Algorithm 4 (Paper's TT-Cross Greedy Heuristic) against
TT-Cross Beam Search (Our Novelty) on black-box functional tensors.
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

from src.models import create_block_diagonal_Q
from src.algorithms import tt_cross_opt, tt_cross_beam_search
from src.tt_core import tt_num_params


def run_ttcross_experiment(N=8, n_trials=20, beam_width=3, tol=1e-6, seed=42):
    rng = np.random.RandomState(seed)
    grid = np.arange(-2, 2.01, 0.5)  # 9 points
    shape = [len(grid)] * N
    
    Q = create_block_diagonal_Q(N, [N // 2, N // 2], seed=101)
    
    def f_coords(X):
        return np.einsum('si,ij,sj->s', X, Q, X)
        
    print(f"\n{'='*70}")
    print(f"TT-CROSS BENCHMARK: Block-Diagonal Q ({N//2}+{N//2}={N} modes)")
    print(f"Trials: {n_trials}, Beam Width: {beam_width}, Tol: {tol}")
    print(f"{'='*70}")
    
    # Natural order baseline
    nat_order = list(range(N))
    _, nat_bonds, _ = tt_cross_opt(f_coords, N, grid, initial_perm=nat_order, tol=tol, seed=seed)
    nat_max = max(nat_bonds)
    nat_params = tt_num_params(nat_bonds, shape)
    print(f"Natural Ordering: bonds={nat_bonds}, max={nat_max}, params={nat_params}")
    
    # Pre-generate random permutations
    permutations = [rng.permutation(N).tolist() for _ in range(n_trials)]
    
    algo4_max_bonds = []
    algo4_params = []
    algo4_times = []
    
    beam_max_bonds = []
    beam_params = []
    beam_times = []
    
    for idx, perm in enumerate(permutations):
        # 1. Algorithm 4 (Paper's Greedy TT-Cross)
        t0 = time.time()
        ord_4, bonds_4, _ = tt_cross_opt(f_coords, N, grid, initial_perm=perm, tol=tol, seed=seed)
        t_4 = time.time() - t0
        algo4_max_bonds.append(max(bonds_4))
        algo4_params.append(tt_num_params(bonds_4, shape))
        algo4_times.append(t_4)
        
        # 2. TT-Cross Beam Search (Our Novelty)
        t0 = time.time()
        ord_b, bonds_b, _ = tt_cross_beam_search(f_coords, N, grid, initial_perm=perm, beam_width=beam_width, tol=tol, seed=seed)
        t_b = time.time() - t0
        beam_max_bonds.append(max(bonds_b))
        beam_params.append(tt_num_params(bonds_b, shape))
        beam_times.append(t_b)
        
        print(f"  Trial {idx+1}/{n_trials}: Algo4 max={max(bonds_4)} ({t_4:.2f}s) | Beam max={max(bonds_b)} ({t_b:.2f}s)")
        
    res = {
        "N": N,
        "n_trials": n_trials,
        "natural_max": nat_max,
        "natural_params": nat_params,
        "algo4_paper": {
            "counts": {str(k): int(v) for k, v in sorted(Counter(algo4_max_bonds).items())},
            "opt_hits": int(sum(1 for m in algo4_max_bonds if m <= nat_max)),
            "opt_rate": float(100.0 * sum(1 for m in algo4_max_bonds if m <= nat_max) / n_trials),
            "avg_params": float(np.mean(algo4_params)),
            "min_params": int(np.min(algo4_params)),
            "avg_time_sec": float(np.mean(algo4_times))
        },
        "beam_search_ours": {
            "counts": {str(k): int(v) for k, v in sorted(Counter(beam_max_bonds).items())},
            "opt_hits": int(sum(1 for m in beam_max_bonds if m <= nat_max)),
            "opt_rate": float(100.0 * sum(1 for m in beam_max_bonds if m <= nat_max) / n_trials),
            "avg_params": float(np.mean(beam_params)),
            "min_params": int(np.min(beam_params)),
            "avg_time_sec": float(np.mean(beam_times))
        }
    }
    
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ttcross_benchmark_results.json"
    
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_file, "w") as f:
        json.dump(res, f, indent=2, default=convert_for_json)
        
    print(f"\n--- TT-Cross Benchmark Summary ---")
    print(f"Algorithm 4 (Paper): Max Bond Counts = {res['algo4_paper']['counts']}, Optimal Hit Rate = {res['algo4_paper']['opt_rate']:.1f}%, Avg Params = {res['algo4_paper']['avg_params']:.1f}")
    print(f"Beam Search (Ours):  Max Bond Counts = {res['beam_search_ours']['counts']}, Optimal Hit Rate = {res['beam_search_ours']['opt_rate']:.1f}%, Avg Params = {res['beam_search_ours']['avg_params']:.1f}")
    print(f"Saved results to {out_file}")
    return res


if __name__ == "__main__":
    run_ttcross_experiment(N=8, n_trials=5, beam_width=3, tol=1e-6, seed=42)
