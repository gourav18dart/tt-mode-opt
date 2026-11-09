"""Focused TTpy DMRG-cross correctness and failure diagnosis."""

import argparse
import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tt
from tt.algs.cross import element

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_ttpy_dmrg_table1 import (
    DMRG_MAX_SWEEPS,
    DMRG_PIVOTING,
    GRID,
    N,
    PERMUTATION_SEED,
    Q_SEED,
    TOL,
    _make_evaluator,
    _decompose_order_once,
    _seed_for_order,
    build_paper_block_diagonal_q,
)


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _run_cross(fun, shape, *, eps, pivoting, seed, check_seed):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tt.dmrg_cross(
            fun,
            shape,
            eps=eps,
            pivoting=pivoting,
            nswp=DMRG_MAX_SWEEPS,
            seed=seed,
            n_check=1000,
            check_seed=check_seed,
            verbose=False,
        )

    finite_cores = all(
        np.isfinite(np.asarray(core)).all() for core in result.cores
    )
    pivot_history = [sweep["pivotmax"] for sweep in result.history.sweeps]
    return result, {
        "converged": result.history.converged,
        "stop_reason": result.history.stop_reason,
        "sweeps": len(result.history.sweeps),
        "ranks": [int(rank) for rank in result.r[1:-1]],
        "finite_cores": finite_cores,
        "nonfinite_pivot_sweeps": [
            index + 1
            for index, pivot in enumerate(pivot_history)
            if not np.isfinite(pivot)
        ],
        "held_out_relative_error": _json_number(result.history.err_check),
        "held_out_infinity_relative_error": _json_number(
            result.history.err_check_inf
        ),
        "warnings": [str(item.message) for item in caught],
    }


def _small_exact_check(matrix):
    order = list(range(4))
    small_matrix = matrix[:4, :4]
    _, fun = _make_evaluator(small_matrix, order)
    result, report = _run_cross(
        fun,
        [len(GRID)] * len(order),
        eps=TOL,
        pivoting=DMRG_PIVOTING,
        seed=1,
        check_seed=7,
    )

    indices = np.indices([len(GRID)] * len(order)).reshape(len(order), -1).T
    exact = fun(indices)
    predicted = np.asarray(element(result, indices)).reshape(-1)
    denominator = np.linalg.norm(exact)
    report["exhaustive_relative_error"] = _json_number(
        np.linalg.norm(exact - predicted) / denominator
    )
    report["full_grid_points_checked"] = int(len(indices))
    report["shape"] = [len(GRID)] * len(order)
    return report


def _find_greedy_failure(matrix, initial_order, trial_id):
    cache = {}

    def evaluate(order):
        key = tuple(order)
        if key not in cache:
            try:
                cache[key] = _decompose_order_once(
                    matrix,
                    order,
                    _seed_for_order(trial_id, order),
                    attempt=1,
                    pivoting=DMRG_PIVOTING,
                )
            except Exception as error:
                cache[key] = error
        outcome = cache[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    try:
        current = evaluate(initial_order)
        for step in range(N - 1):
            best = current
            candidate_order = list(current["order"])
            for _ in range(N - step):
                candidate_order = (
                    candidate_order[:step]
                    + candidate_order[step + 1:]
                    + [candidate_order[step]]
                )
                candidate = evaluate(candidate_order)
                if (
                    candidate["ranks"][step] < best["ranks"][step]
                    or (
                        candidate["ranks"][step] == best["ranks"][step]
                        and candidate["values"][step] < best["values"][step]
                    )
                ):
                    best = candidate
            current = best
        return {
            "found": False,
            "candidate_decompositions": len(cache),
        }
    except Exception as error:
        failed_order = next(
            order for order in cache
            if isinstance(cache[order], Exception)
        )
        return {
            "found": True,
            "trial_id": trial_id,
            "candidate_decompositions": len(cache),
            "failed_order": list(failed_order),
            "baseline_seed": _seed_for_order(trial_id, failed_order),
            "error": f"{type(error).__name__}: {error}",
        }


def _probe_failure(matrix, failure):
    order = failure["failed_order"]
    _, fun = _make_evaluator(matrix, order)
    seed = failure["baseline_seed"]
    cases = [
        ("baseline", TOL, DMRG_PIVOTING, seed),
        ("different_seed", TOL, DMRG_PIVOTING, seed + 1),
        ("pivoting_zero", TOL, 0, seed),
        ("pivoting_one", TOL, 1, seed),
        ("tighter_eps", 1e-8, DMRG_PIVOTING, seed),
    ]
    reports = []
    for name, eps, pivoting, case_seed in cases:
        _, report = _run_cross(
            fun,
            [len(GRID)] * N,
            eps=eps,
            pivoting=pivoting,
            seed=case_seed,
            check_seed=Q_SEED + failure["trial_id"],
        )
        reports.append(
            {
                "case": name,
                "eps": eps,
                "pivoting": pivoting,
                "seed": case_seed,
                **report,
            }
        )
    return reports


def run_diagnostic(output_path):
    matrix = build_paper_block_diagonal_q(Q_SEED)
    permutations = np.random.default_rng(PERMUTATION_SEED)
    permutations = [permutations.permutation(N).tolist() for _ in range(2)]

    failure = _find_greedy_failure(matrix, permutations[1], trial_id=1)
    if not failure["found"]:
        raise RuntimeError(
            "The deterministic trial-1 failure was not reproduced; "
            "the configured TTpy behavior may have changed."
        )

    report = {
        "metadata": {
            "experiment": "Focused TTpy DMRG-cross correctness and failure diagnosis",
            "ttpy_version": version("ttpy"),
            "N": N,
            "grid": GRID.tolist(),
            "Q_seed": Q_SEED,
            "Q_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "tolerance": TOL,
            "baseline_pivoting": DMRG_PIVOTING,
            "maximum_sweeps": DMRG_MAX_SWEEPS,
            "held_out_points": 1000,
        },
        "small_exact_check": _small_exact_check(matrix),
        "reproduced_failure": failure,
        "controlled_failure_probes": _probe_failure(matrix, failure),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "ttpy_dmrg_diagnostic.json",
    )
    args = parser.parse_args()
    result = run_diagnostic(args.output)
    print(json.dumps(result, indent=2))
