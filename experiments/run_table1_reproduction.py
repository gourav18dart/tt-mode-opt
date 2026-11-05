"""
Benchmark reproduction study for Table I from Tichavsky & Straka (2025):
Direct head-to-head comparison of Algorithm 4 (ttcrossopt) vs Multi-Candidate
TT-Cross Beam Search on N=20 Block-Diagonal Potential across 100 independent trials.
"""

import sys
import os
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import teneva

# Ensure repository root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models import create_block_diagonal_Q
from src.tt_core import tt_num_params
from src.algorithms import tt_cross_opt, tt_cross_beam_search


def evaluate_trial(args):
    trial_id, N, Q, grid, perm = args
    
    def f_coords(X):
        return np.einsum('si,ij,sj->s', X, Q, X)
        
    shape = [len(grid)] * N
    
    # Generate 1,000 held-out evaluation points
    rng_test = np.random.default_rng(5000 + trial_id)
    I_test = rng_test.integers(0, len(grid), size=(1000, N))
    X_test = grid[I_test]
    y_true = f_coords(X_test)
    y_norm = float(np.linalg.norm(y_true))
    
    # 1. Run Algorithm 4 (Greedy, K=1)
    t0 = time.time()
    ord_alg4, bonds_alg4, tt_alg4 = tt_cross_opt(
        f_coords, N, grid, initial_perm=perm, tol=1e-4, seed=trial_id
    )
    t_alg4 = time.time() - t0
    params_alg4 = tt_num_params(bonds_alg4, [len(grid)] * N)
    
    # Evaluate held-out accuracy for Algorithm 4
    I_alg4 = I_test[:, ord_alg4]
    y_pred_alg4 = teneva.get(tt_alg4, I_alg4)
    rel_err_alg4 = float(np.linalg.norm(y_true - y_pred_alg4) / y_norm) if y_norm > 0 else 0.0
    
    # 2. Run TT-Cross Beam Search (Novelty, K=2)
    t0 = time.time()
    ord_beam, bonds_beam, tt_beam = tt_cross_beam_search(
        f_coords, N, grid, initial_perm=perm, beam_width=2, tol=1e-4, seed=trial_id
    )
    t_beam = time.time() - t0
    params_beam = tt_num_params(bonds_beam, [len(grid)] * N)
    
    # Evaluate held-out accuracy for Beam Search
    I_beam = I_test[:, ord_beam]
    y_pred_beam = teneva.get(tt_beam, I_beam)
    rel_err_beam = float(np.linalg.norm(y_true - y_pred_beam) / y_norm) if y_norm > 0 else 0.0
    
    result = {
        'trial_id': int(trial_id),
        'perm': [int(p) for p in perm],
        'alg4': {
            'order': [int(o) for o in ord_alg4],
            'bonds': [int(b) for b in bonds_alg4],
            'max_bond': int(max(bonds_alg4)),
            'params': int(params_alg4),
            'rel_err': float(rel_err_alg4),
            'time_sec': float(round(t_alg4, 2))
        },
        'beam': {
            'order': [int(o) for o in ord_beam],
            'bonds': [int(b) for b in bonds_beam],
            'max_bond': int(max(bonds_beam)),
            'params': int(params_beam),
            'rel_err': float(rel_err_beam),
            'time_sec': float(round(t_beam, 2))
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run N=20 TT-cross benchmark with checkpointing.")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials (e.g. 20 for pilot, 100 or 1000).")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers.")
    args = parser.parse_args()
    
    N = 20
    block_sizes = [10, 10]
    grid = np.arange(-2, 2.01, 0.5)  # 9 points per mode as in Section VI
    num_trials = args.trials
    num_workers = min(args.workers, os.cpu_count() or 4)
    
    # Fixed problem instance Q as in Section VI of the paper
    Q_fixed = create_block_diagonal_Q(N, block_sizes, seed=42)
    
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f'table1_{num_trials}trials_checkpoint.json')
    
    print(f"============================================================")
    print(f"TT-CROSS BENCHMARK (N={N}, {num_trials} Trials, Checkpointed)")
    print(f"Workers: {num_workers} parallel CPU processes")
    print(f"Fixed Q: Symmetric block-diagonal (Gaussian below diagonal, seed 42)")
    print(f"Grid: {len(grid)} points per mode ({len(grid)}^{N} total points)")
    print(f"Held-out Test: 1,000 points per trial")
    print(f"Tolerance: delta = 1e-4")
    print(f"============================================================")
    
    # Check if checkpoint already exists to resume
    all_results = []
    completed_ids = set()
    if os.path.exists(out_file):
        try:
            with open(out_file, 'r') as f:
                all_results = json.load(f)
            completed_ids = {r['trial_id'] for r in all_results}
            print(f"Resuming from existing checkpoint: {len(completed_ids)}/{num_trials} trials already completed.")
        except Exception as e:
            print(f"Could not load existing checkpoint: {e}")
            
    # Generate distinct initial permutations evaluated on the fixed Q
    tasks = []
    rng_master = np.random.default_rng(2025)
    for t in range(num_trials):
        perm = rng_master.permutation(N).tolist()
        if t not in completed_ids:
            tasks.append((t, N, Q_fixed, grid, perm))
        
    print(f"Remaining trials to execute: {len(tasks)}")
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
    
    alg4_errs = [r['alg4']['rel_err'] for r in all_results]
    beam_errs = [r['beam']['rel_err'] for r in all_results]
    
    all_possible_ranks = sorted(list(set(alg4_ranks + beam_ranks + [7, 8, 9, 10, 11])))
    
    print("\n" + "="*66)
    print(f"FINAL RESULTS TABLE ({len(all_results)} PERMUTATION TRIALS ON FIXED Q)")
    print("="*66)
    print(f"{'Max Bond Dimension':<20} | {'Greedy (K=1) Count':<18} | {'Beam Search (K=2)':<22}")
    print("-" * 66)
    for r in all_possible_ranks:
        c_alg4 = alg4_ranks.count(r)
        c_beam = beam_ranks.count(r)
        print(f"Rank {r:<15} | {c_alg4:<18} | {c_beam:<22}")
    print("-" * 66)
    print(f"{'Optimal Hit Rate (r=7)':<20} | {alg4_ranks.count(7)/len(all_results)*100:.1f}%{'':<13} | {beam_ranks.count(7)/len(all_results)*100:.1f}%")
    print(f"{'Average Parameters':<20} | {np.mean(alg4_params):<18.1f} | {np.mean(beam_params):<22.1f}")
    print(f"{'Held-out Rel Error':<20} | {np.mean(alg4_errs):<18.2e} | {np.mean(beam_errs):<22.2e}")
    print("="*66)


if __name__ == '__main__':
    main()
