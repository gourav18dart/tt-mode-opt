"""
Core Tensor Train data structures, parameter counters, and reconstruction utilities.
"""

import numpy as np


def tt_to_full(cores):
    """
    Contract TT cores back to a full multi-dimensional dense tensor.
    
    Parameters:
        cores (list of np.ndarray): TT cores of shapes (r_{i-1}, I_i, r_i),
                                    with r_0 = r_N = 1.
                                    
    Returns:
        np.ndarray: Full reconstructed tensor of shape (I_0, I_1, ..., I_{N-1}).
    """
    res = cores[0]  # Shape: (1, I_0, r_1)
    for c in cores[1:]:
        # res: (1, I_0, ..., I_{k-1}, r_k)
        # c:   (r_k, I_k, r_{k+1})
        res = np.tensordot(res, c, axes=(-1, 0))
    # res has shape (1, I_0, ..., I_{N-1}, 1)
    return res.squeeze(axis=0).squeeze(axis=-1)


def tt_num_params(bond_dims, shape):
    """
    Compute the total number of parameters in the Tensor Train format:
        #params = sum_{i=1}^N (r_{i-1} * I_i * r_i), with r_0 = r_N = 1.
        
    Parameters:
        bond_dims (list of int): List of N-1 internal bond dimensions [r_1, ..., r_{N-1}].
        shape (list or tuple of int): Physical dimensions [I_0, ..., I_{N-1}].
        
    Returns:
        int: Total number of scalar entries stored in all TT cores.
    """
    r = [1] + list(bond_dims) + [1]
    return sum(r[i] * shape[i] * r[i + 1] for i in range(len(shape)))


def tt_rel_error(tensor, cores, order=None):
    """
    Compute the relative Frobenius reconstruction error:
        || T_perm - T_hat ||_F / || T_perm ||_F
        
    Parameters:
        tensor (np.ndarray): Original full tensor.
        cores (list of np.ndarray): Reconstructed TT cores.
        order (list of int, optional): Mode permutation applied before decomposition.
        
    Returns:
        float: Relative Frobenius reconstruction error.
    """
    t_target = np.transpose(tensor, axes=order) if order is not None else tensor
    rec = tt_to_full(cores)
    norm_target = np.linalg.norm(t_target)
    if norm_target == 0.0:
        return float(np.linalg.norm(t_target - rec))
    return float(np.linalg.norm(t_target - rec) / norm_target)


def tt_round(cores, tol=1e-10):
    """
    Numerically stable TT-rounding (re-compression) of an existing TT.
    Performs right-to-left QR orthogonalization followed by left-to-right SVD.
    
    Parameters:
        cores (list of np.ndarray): Input TT cores.
        tol (float): Relative truncation threshold.
        
    Returns:
        new_cores (list of np.ndarray): Rounded TT cores with reduced bond dimensions.
        bond_dims (list of int): New bond dimensions.
    """
    N = len(cores)
    F = [c.copy() for c in cores]
    
    # Step 1: Right-to-left QR orthogonalization
    for i in range(N - 1, 0, -1):
        r_prev, I_i, r_next = F[i].shape
        # Reshape to (r_prev, I_i * r_next)
        mat = F[i].reshape(r_prev, I_i * r_next)
        # QR of transpose gives LQ decomposition
        Q, R = np.linalg.qr(mat.T)
        # Q has shape (I_i * r_next, r_prev_new), R has shape (r_prev_new, r_prev)
        # Right core becomes Q.T
        r_new = R.shape[0]
        F[i] = Q.T.reshape(r_new, I_i, r_next)
        # Contract R into previous core F[i-1]
        F[i - 1] = np.tensordot(F[i - 1], R.T, axes=(-1, 1))
        
    # Step 2: Left-to-right SVD truncation
    new_cores = []
    bond_dims = []
    r_prev = 1
    
    for i in range(N - 1):
        _, I_i, r_next = F[i].shape
        mat = F[i].reshape(r_prev * I_i, r_next)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        
        r_i = int(np.sum(S > tol * S[0]))
        r_i = max(r_i, 1)
        
        core = U[:, :r_i].reshape(r_prev, I_i, r_i)
        new_cores.append(core)
        bond_dims.append(r_i)
        
        # Propagate S * Vt to next core
        F[i + 1] = np.tensordot(np.diag(S[:r_i]) @ Vt[:r_i, :], F[i + 1], axes=(1, 0))
        r_prev = r_i
        
    new_cores.append(F[-1].reshape(r_prev, F[-1].shape[1], 1))
    return new_cores, bond_dims
