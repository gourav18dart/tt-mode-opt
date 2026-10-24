import numpy as np
import time
import json

from tt_core import (
    build_quadratic_tensor,
    max_bond_dim_for_order,
    greedy_reorder,
    tt_num_params
)


# ============================================================
# SETTINGS
# ============================================================

np.random.seed(42)

# Paper grid:
# -2 : 0.5 : 2
#
# Gives:
# [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
grid = np.arange(
    -2,
    2.001,
    0.5
)

N = 6

# Small-scale demonstration
N_TRIALS = 200

TOL = 1e-9


# ============================================================
# RUN ONE EXPERIMENTAL CASE
# ============================================================

def run_case(name, Q):

    # --------------------------------------------------------
    # Build full quadratic tensor
    # --------------------------------------------------------

    T = build_quadratic_tensor(
        Q,
        grid
    )

    # --------------------------------------------------------
    # Natural ordering
    # --------------------------------------------------------

    natural_bonds, natural_max, natural_params = (
        max_bond_dim_for_order(
            T,
            list(range(N)),
            tol=TOL
        )
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    random_maxbonds = []
    random_params = []

    greedy_maxbonds = []
    greedy_params = []

    # --------------------------------------------------------
    # Start timing
    # --------------------------------------------------------

    t0 = time.time()

    # ========================================================
    # TRIALS
    # ========================================================

    for trial in range(N_TRIALS):

        # ----------------------------------------------------
        # Random permutation
        # ----------------------------------------------------

        perm = np.random.permutation(N).tolist()

        # Shuffle tensor modes
        Tp = np.transpose(
            T,
            axes=perm
        )

        # ----------------------------------------------------
        # RANDOM ORDER BASELINE
        # ----------------------------------------------------

        _, bmax_r, params_r = (
            max_bond_dim_for_order(
                Tp,
                list(range(N)),
                tol=TOL
            )
        )

        random_maxbonds.append(
            bmax_r
        )

        random_params.append(
            params_r
        )

        # ----------------------------------------------------
        # GREEDY REORDERING
        # ----------------------------------------------------

        order = greedy_reorder(
            Tp,
            tol=TOL
        )

        _, bmax_g, params_g = (
            max_bond_dim_for_order(
                Tp,
                order,
                tol=TOL
            )
        )

        greedy_maxbonds.append(
            bmax_g
        )

        greedy_params.append(
            params_g
        )

    # --------------------------------------------------------
    # Elapsed time
    # --------------------------------------------------------

    elapsed = time.time() - t0

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {
        "name": name,

        "natural_bonds": natural_bonds,
        "natural_max": natural_max,
        "natural_params": natural_params,

        "random_maxbonds": random_maxbonds,
        "random_params": random_params,

        "greedy_maxbonds": greedy_maxbonds,
        "greedy_params": greedy_params,

        "elapsed": elapsed
    }

    return result


# ============================================================
# CASE A: BLOCK-DIAGONAL Q
# ============================================================

# Two 3 x 3 symmetric blocks
B1 = np.random.randn(3, 3)
B1 = (B1 + B1.T) / 2

B2 = np.random.randn(3, 3)
B2 = (B2 + B2.T) / 2

Qblock = np.zeros(
    (N, N)
)

Qblock[:3, :3] = B1
Qblock[3:, 3:] = B2


# ============================================================
# CASE B: TRIDIAGONAL Q
# ============================================================

Qtri = np.zeros(
    (N, N)
)

off = np.random.randn(
    N - 1
)

for i in range(N - 1):

    Qtri[i, i + 1] = off[i]
    Qtri[i + 1, i] = off[i]


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = {}


# ------------------------------------------------------------
# Block diagonal
# ------------------------------------------------------------

results["block"] = run_case(
    "Block-diagonal (3+3)",
    Qblock
)

print(
    f"Block-diagonal done in "
    f"{results['block']['elapsed']:.1f}s"
)


# ------------------------------------------------------------
# Tridiagonal
# ------------------------------------------------------------

results["tri"] = run_case(
    "Tridiagonal",
    Qtri
)

print(
    f"Tridiagonal done in "
    f"{results['tri']['elapsed']:.1f}s"
)


# ============================================================
# SUMMARY
# ============================================================

def summarize(r):

    print(
        f"\n=== {r['name']} ==="
    )

    # --------------------------------------------------------
    # Natural ordering
    # --------------------------------------------------------

    print(
        f"Natural-order bond dims: "
        f"{r['natural_bonds']} "
        f"(max={r['natural_max']}, "
        f"params={r['natural_params']})"
    )

    # --------------------------------------------------------
    # Convert to numpy arrays
    # --------------------------------------------------------

    rm = np.array(
        r["random_maxbonds"]
    )

    rp = np.array(
        r["random_params"]
    )

    gm = np.array(
        r["greedy_maxbonds"]
    )

    gp = np.array(
        r["greedy_params"]
    )

    # --------------------------------------------------------
    # Random results
    # --------------------------------------------------------

    print(
        f"RANDOM ordering over {len(rm)} trials: "
        f"avg max bond={rm.mean():.2f}, "
        f"avg params={rp.mean():.1f}, "
        f"min params={rp.min()}, "
        f"max params={rp.max()}"
    )

    # --------------------------------------------------------
    # Greedy results
    # --------------------------------------------------------

    print(
        f"GREEDY ordering over {len(gm)} trials: "
        f"avg max bond={gm.mean():.2f}, "
        f"avg params={gp.mean():.1f}, "
        f"min params={gp.min()}, "
        f"max params={gp.max()}"
    )

    # --------------------------------------------------------
    # Parameter reduction
    # --------------------------------------------------------

    reduction = (
        100
        * (1 - gp.mean() / rp.mean())
    )

    print(
        f"Average parameter reduction "
        f"from greedy reordering: "
        f"{reduction:.1f}%"
    )

    # --------------------------------------------------------
    # How many trials achieved natural max bond
    # --------------------------------------------------------

    opt_hits = int(
        np.sum(
            gm <= r["natural_max"]
        )
    )

    print(
        f"Trials where greedy matched/beat "
        f"the natural-order max bond dim: "
        f"{opt_hits}/{len(gm)}"
    )


# ============================================================
# PRINT SUMMARIES
# ============================================================

summarize(
    results["block"]
)

summarize(
    results["tri"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "demo_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    "\nSaved demo_results.json"
)