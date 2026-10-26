"""
Comprehensive mathematical and numerical validation tests for Tensor Train algorithms.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Ensure project root is in path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.models import (
    build_quadratic_tensor,
    create_block_diagonal_Q,
    create_tridiagonal_Q,
    create_random_Q
)
from src.tt_core import (
    tt_to_full,
    tt_num_params,
    tt_rel_error,
    tt_round
)
from src.algorithms import (
    tt_svd,
    tt_svd_opt,
    tt_beam_search
)


class TestTensorTrain(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # 5 points
        
    def test_tt_svd_reconstruction(self):
        """Test Algorithm 1 reconstructs full tensor to numerical machine precision."""
        N = 5
        Q = create_random_Q(N, seed=123)
        T = build_quadratic_tensor(Q, self.grid)
        
        cores, bond_dims, bond_vals = tt_svd(T, tol=1e-10)
        self.assertEqual(len(cores), N)
        self.assertEqual(len(bond_dims), N - 1)
        
        # Relative error check
        rel_err = tt_rel_error(T, cores)
        self.assertLess(rel_err, 1e-12, f"Reconstruction error {rel_err:.2e} too high!")
        
    def test_tt_round_fidelity(self):
        """Test TT-rounding (QR + SVD) maintains exact tensor values."""
        N = 5
        Q = create_random_Q(N, seed=456)
        T = build_quadratic_tensor(Q, self.grid)
        
        cores, _, _ = tt_svd(T, tol=1e-10)
        cores_rounded, bonds_rounded, _ = tt_round(cores, tol=1e-10)
        
        rel_err = tt_rel_error(T, cores_rounded)
        self.assertLess(rel_err, 1e-12, f"TT-round error {rel_err:.2e} too high!")
        
    def test_paper_block_diagonal_properties(self):
        """
        Verify the mathematical property from Section III of the paper:
        A block-diagonal Q of size (4+4) has an internal neck bond dimension of 2
        between the two independent blocks.
        """
        N = 8
        Q = create_block_diagonal_Q(N, [4, 4], seed=789)
        T = build_quadratic_tensor(Q, self.grid)
        
        cores, bond_dims, _ = tt_svd(T, tol=1e-9)
        # Bond between block 1 (modes 0,1,2,3) and block 2 (modes 4,5,6,7) is index 3
        neck_bond = bond_dims[3]
        self.assertEqual(neck_bond, 2, f"Expected neck bond 2 between blocks, got {neck_bond}")
        self.assertLessEqual(max(bond_dims), 4, f"Expected max bond <= 4, got {max(bond_dims)}")
        
    def test_beam_search_correctness_and_superiority(self):
        """
        Verify that Beam Search (K=5) escapes greedy traps on a shuffled block-diagonal
        tensor and produces an exact TT decomposition.
        """
        N = 8
        Q = create_block_diagonal_Q(N, [4, 4], seed=101)
        T_nat = build_quadratic_tensor(Q, self.grid)
        
        # Permutation where greedy gets trapped at max bond 5 while beam search finds optimal 4
        shuffled_perm = [1, 5, 0, 7, 2, 4, 3, 6]
        T_shuffled = np.transpose(T_nat, axes=shuffled_perm)
        
        # Test Greedy (K=1, Algorithm 3)
        ord_g, cores_g, bonds_g, _ = tt_svd_opt(T_shuffled, tol=1e-9)
        err_g = tt_rel_error(T_shuffled, cores_g, order=ord_g)
        self.assertLess(err_g, 1e-12, f"Greedy reconstruction error too high: {err_g:.2e}")
        
        # Test Beam Search (K=5, Proposed Novelty)
        ord_b, cores_b, bonds_b, _ = tt_beam_search(T_shuffled, tol=1e-9, beam_width=5)
        err_b = tt_rel_error(T_shuffled, cores_b, order=ord_b)
        self.assertLess(err_b, 1e-12, f"Beam search reconstruction error too high: {err_b:.2e}")
        
        # Rigorously verify beam search achieves strictly lower max bond rank than greedy
        print(f"\n[Test] Greedy (K=1) Max Bond: {max(bonds_g)}, Bonds: {bonds_g}")
        print(f"[Test] Beam   (K=5) Max Bond: {max(bonds_b)}, Bonds: {bonds_b}")
        self.assertEqual(max(bonds_b), 4, f"Beam search should achieve optimal rank 4, got {max(bonds_b)}")
        self.assertEqual(max(bonds_g), 5, f"Greedy should get trapped at rank 5, got {max(bonds_g)}")
        self.assertLess(max(bonds_b), max(bonds_g))

    def test_tt_cross_optimization(self):
        """
        Verify that both TT-Cross mode optimizers (Algorithm 4 and TT-Cross Beam Search)
        successfully execute and optimize a shuffled block-diagonal function.
        """
        from src.algorithms import tt_cross_opt, tt_cross_beam_search
        N = 8
        Q = create_block_diagonal_Q(N, [4, 4], seed=101)
        grid = np.arange(-2, 2.01, 0.5)
        
        def f_coords(X):
            return np.einsum('si,ij,sj->s', X, Q, X)
            
        shuffled_perm = [1, 5, 0, 7, 2, 4, 3, 6]
        
        # 1. Run Algorithm 4 (Paper's TT-cross greedy optimizer)
        ord_algo4, bonds_algo4, _ = tt_cross_opt(f_coords, N, grid, initial_perm=shuffled_perm, tol=1e-6, seed=123)
        
        # 2. Run TT-Cross Beam Search (Proposed Novelty, K=3)
        ord_beam, bonds_beam, _ = tt_cross_beam_search(f_coords, N, grid, initial_perm=shuffled_perm, beam_width=3, tol=1e-6, seed=123)
        
        print(f"\n[TT-Cross Test] Algorithm 4 (Paper) Final Order: {ord_algo4}, Bonds: {bonds_algo4}, Max: {max(bonds_algo4)}")
        print(f"[TT-Cross Test] Beam Search (Ours)  Final Order: {ord_beam}, Bonds: {bonds_beam}, Max: {max(bonds_beam)}")
        
        self.assertLessEqual(max(bonds_algo4), 5)
        self.assertLessEqual(max(bonds_beam), 4)
        self.assertEqual(len(ord_algo4), N)
        self.assertEqual(len(ord_beam), N)


if __name__ == "__main__":
    unittest.main()
