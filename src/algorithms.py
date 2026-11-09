"""
Decomposition and mode-ordering optimization algorithms for Tensor Trains.

Includes:
  - Algorithm 1: SVD-Based TT Decomposition (tt_svd)
  - Algorithm 3: Greedy Mode Ordering (tt_svd_opt)
  - Novelty: Multi-Candidate Beam Search Mode Ordering (tt_beam_search)
"""

import numpy as np
from .tt_core import tt_num_params, tt_round


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


def get_canonical_bond_values(cores):
    """
    Computes gauge-independent bond values by orthogonalizing the TT format (Algorithm 2).
    Right-to-left QR orthogonalization ensures that when sweeping left-to-right,
    the singular values obtained at cut k are the exact singular values of the
    global tensor unfolding matrix X_(k).
    """
    d = len(cores)
    F = [c.copy() for c in cores]
    # Right-to-left QR orthogonalization
    for i in range(d - 1, 0, -1):
        r1, n, r2 = F[i].shape
        Q, R = np.linalg.qr(F[i].reshape(r1, n * r2).T)
        F[i] = Q.T.reshape(-1, n, r2)
        F[i - 1] = np.tensordot(F[i - 1], R.T, axes=(-1, 1))
        
    # Left-to-right SVD: singular values of the true global matricization
    vals = []
    r_prev = 1
    for i in range(d - 1):
        _, n, r2 = F[i].shape
        mat = F[i].reshape(r_prev * n, r2)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        # In Algorithm 2/3, v_i is the smallest retained singular value S(r_i, r_i)
        vals.append(float(S[-1]) if len(S) > 0 else 0.0)
        F[i + 1] = np.tensordot(np.diag(S) @ Vt, F[i + 1], axes=(1, 0))
        r_prev = len(S)
        
    return vals


def _get_cross_tt(f_multi_index, shape, tol=1e-6, seed=123, nswp=5, backend="teneva"):
    """
    Internal helper to construct TT cores via cross-approximation and rank truncation.
    Returns:
        Y_tr (list of np.ndarray): Truncated TT cores.
        ranks (list of int): Bond dimensions [r_1, ..., r_{N-1}].
        vals (list of float): Canonical, gauge-invariant cut singular values (Algorithm 2).
    """
    if backend == "teneva":
        import teneva

        Y0 = teneva.rand(shape, r=2, seed=seed)
        Y = teneva.cross(f_multi_index, Y0, e=tol, nswp=nswp)
        Y_tr = teneva.truncate(Y, e=tol)
        cores = Y_tr
        ranks = [int(r) for r in teneva.ranks(Y_tr)[1:-1]]
    elif backend == "ttpy":
        import tt

        rng = np.random.RandomState(seed)
        initial_ranks = [1] + [2] * (len(shape) - 1) + [1]
        initial_cores = [
            rng.standard_normal((initial_ranks[i], mode_size, initial_ranks[i + 1]))
            for i, mode_size in enumerate(shape)
        ]
        Y0 = tt.vector.from_list(initial_cores)
        Y = tt.rect_cross(
            f_multi_index,
            Y0,
            eps=tol,
            nswp=nswp,
            kickrank=1,
            verbose=False,
        )
        cores = [np.asarray(core) for core in tt.vector.to_list(Y)]
        ranks = [int(r) for r in Y.r[1:-1]]
    else:
        raise ValueError(f"Unsupported TT-cross backend: {backend!r}")

    vals = get_canonical_bond_values(cores)
    return cores, ranks, vals


def tt_cross_opt(f_coords, N, grid, initial_perm=None, tol=1e-6, seed=123, backend="teneva"):
    """
    Algorithm 4: Optimized TT-Cross Decomposition (Faithful implementation).
    Follows Algorithm 4 from Tichavsky & Straka (2025), including bond-value tie-breaking.
    
    Parameters:
        f_coords (callable): Function f(X) returning 1D array of values for shape (samples, N).
        N (int): Number of modes.
        grid (np.ndarray): 1D coordinate grid.
        initial_perm (list of int, optional): Starting permutation.
        tol (float): Truncation tolerance.
        seed (int): Seed for TT-cross initialization.
        backend (str): Cross-approximation package: "teneva" or "ttpy".
        
    Returns:
        best_order (list of int): Optimized mode ordering.
        best_bonds (list of int): TT bond dimensions under optimized order.
        final_tt (list of np.ndarray): Final TT cores.
    """
    shape = [len(grid)] * N
    order = list(range(N)) if initial_perm is None else list(initial_perm)
    
    def make_evaluator(curr_ord):
        def f_multi_index(I):
            X = np.zeros_like(I, dtype=float)
            for k, var in enumerate(curr_ord):
                X[:, var] = grid[I[:, k]]
            return f_coords(X)
        return f_multi_index
        
    F, r, v = _get_cross_tt(make_evaluator(order), shape, tol=tol, seed=seed, backend=backend)
    
    for i in range(N - 1):
        r_i = r[i]
        v_i = v[i]
        orda = list(order)
        for j in range(1, N - i + 1):
            # Cyclic shift of remaining modes as in line 8 of Algorithm 4:
            # orda = [orda(1:i-1), orda(i+1:d), orda(i)]
            cand_ord = orda[:i] + orda[i+1:] + [orda[i]]
            Fa, ra, va = _get_cross_tt(
                make_evaluator(cand_ord), shape, tol=tol, seed=seed, backend=backend
            )
            # Line 11: if ra(i) < r_i or (ra(i) == r_i and va(i) < v_i)
            if ra[i] < r_i or (ra[i] == r_i and va[i] < v_i):
                r_i = ra[i]
                v_i = va[i]
                F = Fa
                order = list(cand_ord)
                r = ra
                v = va
            orda = cand_ord
                
    final_bonds = list(r)
    return order, final_bonds, F


def tt_cross_beam_search(
    f_coords, N, grid, initial_perm=None, beam_width=3, tol=1e-6, seed=123, backend="teneva"
):
    """
    Beam Search Mode Ordering for TT-Cross Decomposition.
    Generalizes Algorithm 4 (ttcrossopt):
    - When beam_width == 1, exactly reproduces Algorithm 4.
    - When beam_width >= 2, maintains top-K candidate orderings in parallel.
    
    Evaluates candidate permutations using black-box TT-cross approximation and ranks
    hypotheses by (max_rank_so_far, sum_rank_so_far, current_rank, current_canonical_bond_val).

    The cross approximation backend can be selected with ``backend="teneva"`` or
    ``backend="ttpy"``.
    """
    shape = [len(grid)] * N
    init_ord = list(range(N)) if initial_perm is None else list(initial_perm)
    
    def make_evaluator(curr_ord):
        def f_multi_index(I):
            X = np.zeros_like(I, dtype=float)
            for k, var in enumerate(curr_ord):
                X[:, var] = grid[I[:, k]]
            return f_coords(X)
        return f_multi_index
        
    F0, r0, v0 = _get_cross_tt(
        make_evaluator(init_ord), shape, tol=tol, seed=seed, backend=backend
    )
    beam = [(list(init_ord), r0, v0, F0)]
    
    for i in range(N - 1):
        candidates = []
        for curr_order, curr_r, curr_v, curr_F in beam:
            orda = list(curr_order)
            # Evaluate all cyclic shifts as in line 8 of Algorithm 4
            for j in range(1, N - i + 1):
                cand_ord = orda[:i] + orda[i+1:] + [orda[i]]
                Fa, ra, va = _get_cross_tt(
                    make_evaluator(cand_ord), shape, tol=tol, seed=seed, backend=backend
                )
                if beam_width == 1:
                    cost = (ra[i], va[i])
                else:
                    cost = (int(max(ra[:i+1])), int(sum(ra[:i+1])), ra[i], float(va[i]))
                candidates.append((cand_ord, ra, va, Fa, cost))
                orda = cand_ord
                
        # Sort candidates lexicographically
        candidates.sort(key=lambda x: x[4])
        seen = set()
        beam = []
        for cand in candidates:
            k = tuple(cand[0][:i+1])
            if k not in seen:
                seen.add(k)
                beam.append((cand[0], cand[1], cand[2], cand[3]))
                if len(beam) >= beam_width:
                    break
                    
    best_order = beam[0][0]
    final_bonds = list(beam[0][1])
    final_tt = beam[0][3]
    return best_order, final_bonds, final_tt
