"""
Data models and test problem generators for Tensor Train analysis.
Implements multivariate quadratic forms and structured coupling matrices.
"""

import numpy as np


def build_quadratic_tensor(Q, grid):
    """
    Build the full multidimensional tensor for the quadratic form:
        V(x) = x^T Q x
    sampled on a 1D grid for each of the N coordinates.
    
    Parameters:
        Q (np.ndarray): Symmetric N x N matrix defining the couplings.
        grid (np.ndarray): 1D array of sampled coordinate points.
        
    Returns:
        np.ndarray: N-dimensional tensor of shape (M, M, ..., M), where M = len(grid).
    """
    N = Q.shape[0]
    mesh = np.meshgrid(*([grid] * N), indexing='ij')
    # Stack along the final axis to obtain shape (M, ..., M, N)
    X = np.stack(mesh, axis=-1)
    # Tensor contraction x^T Q x = sum_{i,j} X[..., i] * Q[i, j] * X[..., j]
    T = np.einsum('...i,ij,...j->...', X, Q, X)
    return T


def create_block_diagonal_Q(N, block_sizes, seed=None):
    """
    Create a symmetric block-diagonal matrix Q consisting of independent dense blocks.
    
    Parameters:
        N (int): Dimension of the matrix (must equal sum(block_sizes)).
        block_sizes (list of int): Size of each diagonal block.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Symmetric N x N block-diagonal matrix.
    """
    assert sum(block_sizes) == N, f"Sum of block sizes {sum(block_sizes)} != N ({N})"
    rng = np.random.RandomState(seed)
        
    Q = np.zeros((N, N))
    start = 0
    for sz in block_sizes:
        end = start + sz
        B = rng.randn(sz, sz)
        B = (B + B.T) / 2.0
        Q[start:end, start:end] = B
        start = end
        
    return Q


def create_tridiagonal_Q(N, seed=None):
    """
    Create a symmetric tridiagonal matrix Q with random off-diagonal entries
    as described in Section VI of Tichavsky & Straka (2025).
    
    Parameters:
        N (int): Dimension of the matrix.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Symmetric N x N tridiagonal matrix.
    """
    rng = np.random.RandomState(seed)
        
    Q = np.zeros((N, N))
    off = rng.randn(N - 1)
    for i in range(N - 1):
        Q[i, i + 1] = off[i]
        Q[i + 1, i] = off[i]
        
    return Q


def create_random_Q(N, seed=None):
    """
    Create a fully dense symmetric random matrix Q.
    
    Parameters:
        N (int): Dimension of the matrix.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Symmetric N x N matrix with Gaussian i.i.d. entries.
    """
    rng = np.random.RandomState(seed)
    Q = rng.randn(N, N)
    return (Q + Q.T) / 2.0
