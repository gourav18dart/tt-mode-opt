"""
Tensor Train Mode Optimization Package
Implementing TT-SVD, Greedy Mode Ordering, and Beam Search Mode Ordering.
"""

from .tt_core import (
    tt_to_full,
    tt_num_params,
    tt_rel_error,
    tt_round
)

from .models import (
    build_quadratic_tensor,
    create_block_diagonal_Q,
    create_tridiagonal_Q,
    create_random_Q
)

from .algorithms import (
    tt_svd,
    tt_svd_opt,
    tt_beam_search
)

__all__ = [
    'tt_to_full',
    'tt_num_params',
    'tt_rel_error',
    'tt_round',
    'build_quadratic_tensor',
    'create_block_diagonal_Q',
    'create_tridiagonal_Q',
    'create_random_Q',
    'tt_svd',
    'tt_svd_opt',
    'tt_beam_search'
]
