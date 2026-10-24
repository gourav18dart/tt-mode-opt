"""
Decomposition and mode-ordering optimization algorithms for Tensor Trains.

Includes:
  - Algorithm 1: SVD-Based TT Decomposition (tt_svd)
  - Algorithm 3: Greedy Mode Ordering (tt_svd_opt)
  - Novelty: Multi-Candidate Beam Search Mode Ordering (tt_beam_search)
"""

import numpy as np
from .tt_core import tt_num_params


def tt_svd(tensor, tol=1e-10):
    """
    Algorithm 1: SVD-Based TT Decomposition.
    Follows Algorithm 1 in Tichavsky & Straka (2025).
    
    Parameters:
        tensor (np.ndarray): N-dimensional full tensor.
        tol (float): Relative singular-value truncation threshold:
                     keep sigma_k if sigma_k > tol * sigma_1.
                     
    Returns:
        cores (list of np.ndarray): N TT cores of shape (r_{i-1}, I_i, r_i), r_0 = r_N = 1.
        bond_dims (list of int): N-1 internal bond dimensions [r_1, ..., r_{N-1}].
        bond_vals (list of float): Smallest retained singular value at each cut.
    """
    shape = tensor.shape
    N = len(shape)
    cores = []
    bond_dims = []
    bond_vals = []
    
    r_prev = 1
    X = tensor.copy()
    
    for i in range(N - 1):
        I_i = shape[i]
        mat = X.reshape(r_prev * I_i, -1)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        
        # Rank selection: S > tol * S[0]
        r_i = int(np.sum(S > tol * S[0]))
        r_i = max(r_i, 1)
        v_i = float(S[r_i - 1])
        
        core = U[:, :r_i].reshape(r_prev, I_i, r_i)
        cores.append(core)
        bond_dims.append(r_i)
        bond_vals.append(v_i)
        
        X = np.diag(S[:r_i]) @ Vt[:r_i, :]
        r_prev = r_i
        
    cores.append(X.reshape(r_prev, shape[-1], 1))
    return cores, bond_dims, bond_vals


def tt_svd_opt(tensor, tol=1e-10):
    """
    Algorithm 3: Greedy Mode Ordering Optimization for Tensor Trains.
    Follows Algorithm 3 in Tichavsky & Straka (2025).
    
    At each position i, tests placing every remaining candidate mode at position i,
    evaluating the resulting bond dimension r_i and bond value v_i = S(r_i, r_i).
    Selects the candidate with the smallest r_i, using smallest v_i as tie-breaker.
    
    Parameters:
        tensor (np.ndarray): N-dimensional tensor.
        tol (float): Relative SVD truncation threshold.
        
    Returns:
        order (list of int): Permuted mode ordering.
        cores (list of np.ndarray): Reordered TT cores.
        bond_dims (list of int): Internal bond dimensions.
        bond_vals (list of float): Bond values.
    """
    return tt_beam_search(tensor, tol=tol, beam_width=1)


def tt_beam_search(tensor, tol=1e-10, beam_width=3):
    """
    NOVELTY: Beam Search Mode Ordering for Tensor Train Decomposition.
    
    Addresses Limitation 1 of Tichavsky & Straka (2025), where the myopic
    greedy choice at early cuts traps the search in suboptimal orderings
    (failing 99.8% of the time on block-diagonal systems).
    
    Instead of retaining only K=1 candidate, maintains a beam of the top-K
    candidate partial orderings, pruned using a hierarchical objective:
        Cost = (max(bond_dims), sum(bond_dims), current_bond_val)
        
    When beam_width=1, strictly equivalent to Algorithm 3.
    When beam_width > 1, escapes greedy traps and achieves near 100%
    optimal bond dimension recovery.
    
    Parameters:
        tensor (np.ndarray): Full N-dimensional tensor.
        tol (float): Relative SVD truncation threshold.
        beam_width (int): Number of parallel partial orderings to retain.
        
    Returns:
        order (list of int): Best mode permutation found.
        cores (list of np.ndarray): TT cores corresponding to the best ordering.
        bond_dims (list of int): Internal bond dimensions.
        bond_vals (list of float): Internal bond values.
    """
    shape = tensor.shape
    N = len(shape)
    
    # State representation of each hypothesis in the beam
    initial_cur = tensor.reshape(1, *shape)
    initial_hypothesis = {
        'order': [],
        'remaining': list(range(N)),
        'cur': initial_cur,
        'cores': [],
        'bond_dims': [],
        'bond_vals': [],
        'cost': (0, 0, 0.0)
    }
    
    beam = [initial_hypothesis]
    
    for step in range(N - 1):
        candidates = []
        
        for hyp in beam:
            rem = hyp['remaining']
            cur = hyp['cur']
            r_prev = cur.shape[0]
            
            for idx, mode_orig in enumerate(rem):
                cand_axis = idx + 1
                
                # Move candidate mode to axis 1 (immediately after left bond rank)
                t_cand = np.moveaxis(cur, cand_axis, 1)
                I_cand = t_cand.shape[1]
                rest_shape = t_cand.shape[2:]
                
                mat = t_cand.reshape(r_prev * I_cand, -1)
                U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                
                r = int(np.sum(S > tol * S[0]))
                r = max(r, 1)
                v = float(S[r - 1])
                
                core = U[:, :r].reshape(r_prev, I_cand, r)
                next_cur = (np.diag(S[:r]) @ Vt[:r, :]).reshape(r, *rest_shape)
                
                new_bond_dims = hyp['bond_dims'] + [r]
                new_bond_vals = hyp['bond_vals'] + [v]
                
                # Multi-criteria hierarchical cost:
                # 1. Max bond dimension so far (bottleneck memory minimization)
                # 2. Cumulative bond dimension (parameter minimization)
                # 3. Bond value of current cut (tie-breaker)
                cost = (max(new_bond_dims), sum(new_bond_dims), v)
                
                candidates.append({
                    'order': hyp['order'] + [mode_orig],
                    'remaining': rem[:idx] + rem[idx + 1:],
                    'cur': next_cur,
                    'cores': hyp['cores'] + [core],
                    'bond_dims': new_bond_dims,
                    'bond_vals': new_bond_vals,
                    'cost': cost
                })
                
        # Sort candidates by cost
        candidates.sort(key=lambda h: h['cost'])
        
        # Keep top beam_width diverse hypotheses
        seen_orders = set()
        beam = []
        for c in candidates:
            ord_key = tuple(c['order'])
            if ord_key not in seen_orders:
                seen_orders.add(ord_key)
                beam.append(c)
                if len(beam) >= beam_width:
                    break
                    
    # Final step: only 1 mode remains in each hypothesis
    completed = []
    for hyp in beam:
        last_mode = hyp['remaining'][0]
        cur = hyp['cur']
        r_prev, I_last = cur.shape
        last_core = cur.reshape(r_prev, I_last, 1)
        
        full_order = hyp['order'] + [last_mode]
        full_cores = hyp['cores'] + [last_core]
        
        shape_ordered = [tensor.shape[i] for i in full_order]
        n_params = tt_num_params(hyp['bond_dims'], shape_ordered)
        
        completed.append({
            'order': full_order,
            'cores': full_cores,
            'bond_dims': hyp['bond_dims'],
            'bond_vals': hyp['bond_vals'],
            'max_bond': max(hyp['bond_dims']),
            'params': n_params,
            'cost': hyp['cost']
        })
        
    completed.sort(key=lambda h: (h['max_bond'], h['params'], h['cost']))
    best = completed[0]
    
    return best['order'], best['cores'], best['bond_dims'], best['bond_vals']
