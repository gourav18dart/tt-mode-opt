"""
Benchmark reproduction study for Table I from Tichavsky & Straka (2025):
Direct head-to-head comparison of Algorithm 4 (ttcrossopt) vs Multi-Candidate
TT-Cross Beam Search on N=20 Block-Diagonal Potential across 100 independent trials.
"""

import sys
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Ensure repository root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models import create_block_diagonal_Q
from src.tt_core import tt_num_params
from src.algorithms import tt_cross_opt, tt_cross_beam_search


def evaluate_trial(args):
    trial_id, N, block_sizes, grid, perm = args
    # Reproducible matrix generation per trial
    Q = create_block_diagonal_Q(N, block_sizes, seed=1000 + trial_id)
    
    def f_coords(X):
        return np.einsum('si,ij,sj->s', X, Q, X)
        
    shape = [len(grid)] * N
    
    # 1. Run Algorithm 4 (Paper's ttcrossopt with faithful tie-breaking)
    t0 = time.time()
    ord_alg4, bonds_alg4, _ = tt_cross_opt(
        f_coords, N, grid, initial_perm=perm, tol=1e-4, seed=trial_id
    )
    t_alg4 = time.time() - t0
    params_alg4 = tt_num_params(bonds_alg4, [len(grid)] * N)
    
    # 2. Run TT-Cross Beam Search (Novelty, K=2)
    t0 = time.time()
    ord_beam, bonds_beam, _ = tt_cross_beam_search(
        f_coords, N, grid, initial_perm=perm, beam_width=2, tol=1e-4, seed=trial_id
    )
    t_beam = time.time() - t0
    params_beam = tt_num_params(bonds_beam, [len(grid)] * N)
    
    result = {
        'trial_id': int(trial_id),
        'perm': [int(p) for p in perm],
        'alg4': {
            'order': [int(o) for o in ord_alg4],
            'bonds': [int(b) for b in bonds_alg4],
            'max_bond': int(max(bonds_alg4)),
            'params': int(params_alg4),
            'time_sec': float(round(t_alg4, 2))
        },
        'beam': {
            'order': [int(o) for o in ord_beam],
            'bonds': [int(b) for b in bonds_beam],
            'max_bond': int(max(bonds_beam)),
            'params': int(params_beam),
            'time_sec': float(round(t_beam, 2))
        }
    }
    return result


def main():
    N = 20
    block_sizes = [10, 10]
    grid = np.arange(-2, 2.01, 0.5)  # 9 points per mode as in Section VI
    num_trials = 100
    num_workers = min(10, os.cpu_count() or 4)
    
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'table1_100trials_reproduction.json')
    
    print(f"============================================================")
    print(f"TABLE I REPRODUCTION BENCHMARK (N={N}, {num_trials} Trials)")
    print(f"Workers: {num_workers} parallel CPU processes")
    print(f"Grid: {len(grid)} points per mode ({len(grid)}^{N} total points)")
    print(f"Tolerance: delta = 1e-4 (exact as Section VI)")
    print(f"============================================================")
    
    # Generate 100 distinct initial permutations
    tasks = []
    rng_master = np.random.default_rng(2025)
    for t in range(num_trials):
        perm = rng_master.permutation(N).tolist()
        tasks.append((t, N, block_sizes, grid, perm))
        
    all_results = []
    start_total = time.time()
    
    # ProcessPoolExecutor for parallel execution across CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_trial, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            t_id = futures[future]
            try:
                res = future.result()
                all_results.append(res)
                
                # Sort by trial_id for clean bookkeeping
                all_results.sort(key=lambda x: x['trial_id'])
                
                # Save incremental progress
                with open(out_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
                alg4_r = res['alg4']['max_bond']
                beam_r = res['beam']['max_bond']
                elapsed = time.time() - start_total
                
                print(f"[{len(all_results):3d}/{num_trials}] Trial {t_id:2d} finished | "
                      f"Alg4: rank {alg4_r} ({res['alg4']['time_sec']}s) | "
                      f"Beam: rank {beam_r} ({res['beam']['time_sec']}s) | "
                      f"Total Elapsed: {elapsed/60:.1f}m")
            except Exception as e:
                print(f"[ERROR] Trial {t_id} failed: {e}")
                
    total_elapsed = time.time() - start_total
    print(f"\nBenchmark completed in {total_elapsed/60:.2f} minutes.")
    
    # Compile statistics
    alg4_ranks = [r['alg4']['max_bond'] for r in all_results]
    beam_ranks = [r['beam']['max_bond'] for r in all_results]
    
    alg4_params = [r['alg4']['params'] for r in all_results]
    beam_params = [r['beam']['params'] for r in all_results]
    
    all_possible_ranks = sorted(list(set(alg4_ranks + beam_ranks + [7, 8, 9, 10, 11])))
    
    print("\n" + "="*60)
    print("FINAL RESULTS TABLE (100 INDEPENDENT TRIALS)")
    print("="*60)
    print(f"{'Max Bond Dimension':<20} | {'Paper Alg 4 Count':<18} | {'Our Beam Search (K=2)':<22}")
    print("-" * 66)
    for r in all_possible_ranks:
        c_alg4 = alg4_ranks.count(r)
        c_beam = beam_ranks.count(r)
        print(f"Rank {r:<15} | {c_alg4:<18} | {c_beam:<22}")
    print("-" * 66)
    print(f"{'Optimal Hit Rate (r=7)':<20} | {alg4_ranks.count(7)/len(all_results)*100:.1f}%{'':<13} | {beam_ranks.count(7)/len(all_results)*100:.1f}%")
    print(f"{'Average Parameters':<20} | {np.mean(alg4_params):<18.1f} | {np.mean(beam_params):<22.1f}")
    print("="*60)


if __name__ == '__main__':
    main()
