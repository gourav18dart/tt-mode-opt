"""Paired fixed-Q mode-ordering benchmark using TTpy's DMRG-cross."""

import argparse
import hashlib
import json
import math
import os
import platform
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import tt
from tt.algs.cross import element

from src.algorithms import get_canonical_bond_values
from src.tt_core import tt_num_params


N = 20
BLOCK_SIZES = (10, 10)
GRID = np.arange(-2, 2.01, 0.5)
TOL = 1e-4
BEAM_WIDTH = 2
Q_SEED = 42
PERMUTATION_SEED = 2025
DMRG_PIVOTING = 2
DMRG_RETRY_PIVOTINGS = (2, 1, 0)
DMRG_RETRIES_PER_PIVOTING = 5
DMRG_FIXED_RANK_CAPS = (16, 20, 24)
DMRG_FIXED_RANK_SEEDS_PER_CAP = 5
DMRG_RETRY_ATTEMPTS = (
    len(DMRG_RETRY_PIVOTINGS) * DMRG_RETRIES_PER_PIVOTING
    + len(DMRG_FIXED_RANK_CAPS) * DMRG_FIXED_RANK_SEEDS_PER_CAP
)
DMRG_MAX_SWEEPS = 100
HELD_OUT_SIZE = 1000
HELD_OUT_SEED = 5000
MAX_HELD_OUT_ERROR = 10 * TOL


def build_paper_block_diagonal_q(seed):
    """Generate symmetric blocks with N(0, 1) diagonal and lower entries."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((N, N), dtype=float)
    offset = 0
    for block_size in BLOCK_SIZES:
        lower = np.tril(rng.standard_normal((block_size, block_size)))
        block = lower + np.tril(lower, -1).T
        matrix[offset:offset + block_size, offset:offset + block_size] = block
        offset += block_size
    return matrix


def _seed_for_order(trial_id, order):
    payload = f"{trial_id}:" + ",".join(str(mode) for mode in order)
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _make_evaluator(matrix, order):
    def f_coords(coordinates):
        return np.einsum("si,ij,sj->s", coordinates, matrix, coordinates)

    def f_multi_index(indices):
        coordinates = np.zeros_like(indices, dtype=float)
        for position, mode in enumerate(order):
            coordinates[:, mode] = GRID[indices[:, position]]
        return f_coords(coordinates)

    return f_coords, f_multi_index


def _decompose_order_once(matrix, order, seed, attempt, pivoting, rmax=None):
    """Run and validate one seeded TTpy DMRG-cross approximation."""
    f_coords, f_multi_index = _make_evaluator(matrix, order)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_tt = tt.dmrg_cross(
            f_multi_index,
            [len(GRID)] * N,
            eps=TOL if rmax is None else None,
            pivoting=pivoting,
            rmax=rmax,
            nswp=DMRG_MAX_SWEEPS,
            seed=seed,
            verbose=False,
        )

    if not raw_tt.history.converged and not (
        rmax is not None and raw_tt.history.stop_reason == "rmax"
    ):
        raise RuntimeError(
            "TTpy DMRG-cross did not converge: "
            f"{raw_tt.history.stop_reason}, sweeps={len(raw_tt.history.sweeps)}, "
            f"pivotmax={raw_tt.history.sweeps[-1]['pivotmax']:.6g}"
        )
    if not all(np.isfinite(np.asarray(core)).all() for core in raw_tt.cores):
        raise FloatingPointError("TTpy DMRG-cross returned non-finite core values")

    rounded_tt = raw_tt.round(TOL)
    cores = [np.asarray(core) for core in tt.vector.to_list(rounded_tt)]
    if not all(np.isfinite(core).all() for core in cores):
        raise FloatingPointError("TTpy rounding returned non-finite core values")

    check_seed = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
    check_indices = np.random.default_rng(check_seed).integers(
        0, len(GRID), size=(HELD_OUT_SIZE, N)
    )
    exact = f_multi_index(check_indices)
    predicted = np.asarray(element(rounded_tt, check_indices))
    exact_norm = float(np.linalg.norm(exact))
    held_out_rel_error = (
        float(np.linalg.norm(exact - predicted) / exact_norm)
        if exact_norm > 0
        else float(np.linalg.norm(exact - predicted))
    )
    if not np.isfinite(held_out_rel_error):
        raise FloatingPointError("TTpy held-out relative error is non-finite")
    if held_out_rel_error > MAX_HELD_OUT_ERROR:
        raise ValueError(
            "TTpy held-out relative error "
            f"{held_out_rel_error:.6g} exceeds {MAX_HELD_OUT_ERROR:.6g}"
        )

    ranks = [int(rank) for rank in rounded_tt.r[1:-1]]
    values = get_canonical_bond_values(cores)
    return {
        "order": list(order),
        "ranks": ranks,
        "values": values,
        "tensor": rounded_tt,
        "seed": seed,
        "attempt": attempt,
        "pivoting": pivoting,
        "rmax": rmax,
        "held_out_rel_error": held_out_rel_error,
        "fun_eval": int(raw_tt.history.fun_eval),
        "sweeps": len(raw_tt.history.sweeps),
        "warnings": [str(item.message) for item in caught],
    }


def _decompose_order(matrix, order, trial_id):
    """Retry adaptive and fixed-rank runs until finite and accurate."""
    base_seed = _seed_for_order(trial_id, order)
    failures = []
    attempt = 0
    retry_configs = [
        (pivoting, None, seed_offset)
        for seed_offset in range(DMRG_RETRIES_PER_PIVOTING)
        for pivoting in DMRG_RETRY_PIVOTINGS
    ]
    retry_configs.extend(
        (DMRG_PIVOTING, rmax, seed_offset)
        for rmax in DMRG_FIXED_RANK_CAPS
        for seed_offset in range(DMRG_FIXED_RANK_SEEDS_PER_CAP)
    )
    for pivoting, rmax, seed_offset in retry_configs:
        attempt += 1
        seed = (base_seed + seed_offset) & 0xFFFFFFFF
        try:
            return _decompose_order_once(
                matrix, order, seed, attempt, pivoting, rmax=rmax
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failures.append(
                f"pivoting={pivoting}, rmax={rmax}, seed={seed}: "
                f"{type(error).__name__}: {error}"
            )
    raise RuntimeError(
        f"TTpy DMRG-cross failed validation for ordering {list(order)} after "
        f"{DMRG_RETRY_ATTEMPTS} deterministic attempts: "
        + " | ".join(failures)
    )


def _run_greedy(matrix, initial_order, trial_id):
    cache = {}

    def evaluate(order):
        key = tuple(order)
        if key not in cache:
            cache[key] = _decompose_order(matrix, order, trial_id)
        return cache[key]

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
            try:
                candidate = evaluate(candidate_order)
            except RuntimeError:
                continue
            if (
                candidate["ranks"][step] < best["ranks"][step]
                or (
                    candidate["ranks"][step] == best["ranks"][step]
                    and candidate["values"][step] < best["values"][step]
                )
            ):
                best = candidate
        current = best

    return _finish_method(current, cache, matrix, trial_id)


def _run_beam(matrix, initial_order, trial_id):
    cache = {}

    def evaluate(order):
        key = tuple(order)
        if key not in cache:
            cache[key] = _decompose_order(matrix, order, trial_id)
        return cache[key]

    beam = [evaluate(initial_order)]
    for step in range(N - 1):
        candidates = []
        for hypothesis in beam:
            candidate_order = list(hypothesis["order"])
            for _ in range(N - step):
                candidate_order = (
                    candidate_order[:step]
                    + candidate_order[step + 1:]
                    + [candidate_order[step]]
                )
                try:
                    candidate = evaluate(candidate_order)
                except RuntimeError:
                    continue
                cost = (
                    max(candidate["ranks"][:step + 1]),
                    sum(candidate["ranks"][:step + 1]),
                    candidate["ranks"][step],
                    candidate["values"][step],
                )
                candidates.append((candidate, cost))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[1])
        selected = []
        seen_prefixes = set()
        for candidate, _cost in candidates:
            prefix = tuple(candidate["order"][:step + 1])
            if prefix not in seen_prefixes:
                selected.append(candidate)
                seen_prefixes.add(prefix)
                if len(selected) == BEAM_WIDTH:
                    break
        beam = selected

    return _finish_method(beam[0], cache, matrix, trial_id)


def _finish_method(best, cache, matrix, trial_id):
    rng = np.random.default_rng(HELD_OUT_SEED + trial_id)
    held_out_indices = rng.integers(0, len(GRID), size=(HELD_OUT_SIZE, N))
    coordinates = GRID[held_out_indices]
    exact = np.einsum("si,ij,sj->s", coordinates, matrix, coordinates)
    ordered_indices = held_out_indices[:, best["order"]]
    predicted = np.asarray(element(best["tensor"], ordered_indices))
    exact_norm = float(np.linalg.norm(exact))
    rel_error = (
        float(np.linalg.norm(exact - predicted) / exact_norm)
        if exact_norm > 0
        else float(np.linalg.norm(exact - predicted))
    )
    ranks = best["ranks"]
    successful = [
        result for result in cache.values()
        if not isinstance(result, Exception)
    ]
    failed = [
        {
            "order": list(order),
            "error": f"{type(error).__name__}: {error}",
        }
        for order, error in cache.items()
        if isinstance(error, Exception)
    ]
    return {
        "order": best["order"],
        "bonds": ranks,
        "max_bond": max(ranks),
        "params": int(tt_num_params(ranks, [len(GRID)] * N)),
        "held_out_rel_error": rel_error,
        "accuracy_valid": rel_error <= MAX_HELD_OUT_ERROR,
        "held_out_points": HELD_OUT_SIZE,
        "cross_calls": len(cache),
        "valid_cross_calls": len(successful),
        "failed_cross_calls": len(failed),
        "failed_candidates": failed,
        "retry_attempts": sum(result["attempt"] for result in successful),
        "retried_cross_calls": sum(
            result["attempt"] > 1 for result in successful
        ),
        "maximum_attempts_for_one_cross": max(
            result["attempt"] for result in successful
        ),
        "maximum_candidate_held_out_rel_error": max(
            result["held_out_rel_error"] for result in successful
        ),
        "function_evaluations": sum(
            result["fun_eval"] for result in successful
        ),
        "total_sweeps": sum(result["sweeps"] for result in successful),
        "dmrg_seeds": [result["seed"] for result in successful],
        "dmrg_attempts": [
            {
                "attempt": result["attempt"],
                "pivoting": result["pivoting"],
                "rmax": result["rmax"],
                "seed": result["seed"],
                "held_out_rel_error": result["held_out_rel_error"],
            }
            for result in successful
        ],
        "warnings": [
            message
            for result in successful
            for message in result["warnings"]
        ],
    }


def _evaluate_trial(args):
    trial_id, matrix, permutation = args
    record = {
        "trial_id": trial_id,
        "initial_permutation": permutation,
    }
    for method, runner in (("greedy", _run_greedy), ("beam", _run_beam)):
        started = time.perf_counter()
        try:
            result = runner(matrix, permutation, trial_id)
            result["time_sec"] = time.perf_counter() - started
            record[method] = result
        except Exception as error:
            record[method] = {
                "error": f"{type(error).__name__}: {error}",
                "time_sec": time.perf_counter() - started,
            }
    record["status"] = (
        "ok"
        if "error" not in record["greedy"] and "error" not in record["beam"]
        else "failed"
    )
    return record


def _paired_sign_test(wins, losses):
    total = wins + losses
    if total == 0:
        return None
    tail = sum(math.comb(total, k) for k in range(wins, total + 1))
    other_tail = sum(math.comb(total, k) for k in range(losses, total + 1))
    return min(1.0, 2.0 * min(tail, other_tail) / (2 ** total))


def _summarize(trials):
    completed = [
        trial for trial in trials
        if trial["status"] == "ok"
    ]
    summary = {
        "requested_trials": len(trials),
        "completed_trials": len(completed),
        "failed_trials": len(trials) - len(completed),
        "accuracy_threshold_rel_error": MAX_HELD_OUT_ERROR,
    }
    for method in ("greedy", "beam"):
        results = [trial[method] for trial in completed]
        accurate = [result for result in results if result["accuracy_valid"]]
        if not results:
            summary[method] = {}
            continue
        ranks = [result["max_bond"] for result in results]
        params = [result["params"] for result in results]
        summary[method] = {
            "computed_trials": len(results),
            "accuracy_valid_trials": len(accurate),
            "accuracy_invalid_trials": len(results) - len(accurate),
            "max_bond_counts": {
                str(rank): count
                for rank, count in sorted(Counter(ranks).items())
            },
            "rank_7_or_better_hits": sum(rank <= 7 for rank in ranks),
            "rank_7_or_better_rate_percent": (
                100.0 * sum(rank <= 7 for rank in ranks) / len(ranks)
            ),
            "average_max_bond": float(np.mean(ranks)),
            "average_parameters": float(np.mean(params)),
            "minimum_parameters": int(np.min(params)),
            "average_time_sec": float(
                np.mean([result["time_sec"] for result in results])
            ),
            "average_held_out_rel_error": float(
                np.mean([result["held_out_rel_error"] for result in results])
            ),
            "maximum_held_out_rel_error": float(
                np.max([result["held_out_rel_error"] for result in results])
            ),
            "trials_over_1e-4_held_out_error": sum(
                result["held_out_rel_error"] > TOL for result in results
            ),
            "accuracy_valid_average_max_bond": (
                float(np.mean([result["max_bond"] for result in accurate]))
                if accurate
                else None
            ),
            "accuracy_valid_average_parameters": (
                float(np.mean([result["params"] for result in accurate]))
                if accurate
                else None
            ),
        }

    paired = [
        trial for trial in completed
        if trial["greedy"]["accuracy_valid"] and trial["beam"]["accuracy_valid"]
    ]
    rank_wins = sum(
        trial["beam"]["max_bond"] < trial["greedy"]["max_bond"]
        for trial in paired
    )
    rank_losses = sum(
        trial["beam"]["max_bond"] > trial["greedy"]["max_bond"]
        for trial in paired
    )
    param_wins = sum(
        trial["beam"]["params"] < trial["greedy"]["params"]
        for trial in paired
    )
    param_losses = sum(
        trial["beam"]["params"] > trial["greedy"]["params"]
        for trial in paired
    )
    summary["paired_comparison"] = {
        "accuracy_valid_paired_trials": len(paired),
        "excluded_for_accuracy_or_computation": len(trials) - len(paired),
        "beam_lower_max_bond": rank_wins,
        "equal_max_bond": len(paired) - rank_wins - rank_losses,
        "beam_higher_max_bond": rank_losses,
        "exact_two_sided_sign_test_p": _paired_sign_test(rank_wins, rank_losses),
        "beam_fewer_parameters": param_wins,
        "equal_parameters": len(paired) - param_wins - param_losses,
        "beam_more_parameters": param_losses,
        "average_parameter_delta_beam_minus_greedy": (
            float(np.mean([
                trial["beam"]["params"] - trial["greedy"]["params"]
                for trial in paired
            ]))
            if paired
            else None
        ),
    }
    return summary


def _save_checkpoint(path, metadata, trials):
    payload = {
        "metadata": metadata,
        "summary": _summarize(trials),
        "trials": sorted(trials, key=lambda trial: trial["trial_id"]),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
    temp_path.replace(path)


def run_benchmark(num_trials=100, workers=10, output_path=None):
    output_path = (
        Path(output_path)
        if output_path
        else Path(__file__).resolve().parent
        / "results"
        / f"table1_{num_trials}trials_ttpy_dmrg_cross.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = build_paper_block_diagonal_q(Q_SEED)
    matrix_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
    metadata = {
        "experiment": "Fixed-Q paired mode-ordering benchmark with TTpy DMRG-cross",
        "N": N,
        "block_sizes": list(BLOCK_SIZES),
        "grid": GRID.tolist(),
        "grid_points_per_mode": len(GRID),
        "full_tensor_entries_not_materialized": int(len(GRID) ** N),
        "Q_seed": Q_SEED,
        "Q_generation": (
            "Two 10x10 symmetric blocks. Each diagonal and lower-triangular "
            "entry is an independent N(0,1) draw; lower entries are mirrored. "
            "Off-block entries are zero."
        ),
        "Q_sha256": matrix_hash,
        "Q_matrix": matrix.tolist(),
        "permutation_seed": PERMUTATION_SEED,
        "num_trials": num_trials,
        "same_fixed_Q_for_all_trials": True,
        "same_permutation_and_dmrg_seeds_for_greedy_and_beam": True,
        "ttpy_version": version("ttpy"),
        "ttpy_cross_method": "tt.dmrg_cross",
        "ttpy_dmrg_settings": {
            "eps": TOL,
            "pivoting": DMRG_PIVOTING,
            "nswp_max": DMRG_MAX_SWEEPS,
            "retry_attempts": DMRG_RETRY_ATTEMPTS,
            "retry_pivotings": list(DMRG_RETRY_PIVOTINGS),
            "retries_per_pivoting": DMRG_RETRIES_PER_PIVOTING,
            "fixed_rank_fallback_caps": list(DMRG_FIXED_RANK_CAPS),
            "fixed_rank_fallback_seeds_per_cap": (
                DMRG_FIXED_RANK_SEEDS_PER_CAP
            ),
            "candidate_held_out_points": HELD_OUT_SIZE,
            "candidate_maximum_accepted_held_out_relative_error": (
                MAX_HELD_OUT_ERROR
            ),
            "inaccurate_candidate_policy": (
                "Exclude after deterministic retries; greedy retains its "
                "current valid ordering if no insertion candidate passes, and "
                "beam stops expanding if no candidate passes."
            ),
            "seed_strategy": (
                "Stable BLAKE2b-derived base seed from trial ID and candidate "
                "ordering, then consecutive uint32 seeds on retry; the same "
                "candidate ordering uses the same sequence in both methods."
            ),
            "round_after_cross": TOL,
        },
        "greedy_method": "Algorithm 4, local rank/bond-value comparison",
        "beam_width": BEAM_WIDTH,
        "beam_cost": ["max_rank_so_far", "sum_rank_so_far", "current_rank", "bond_value"],
        "held_out_points_per_trial": HELD_OUT_SIZE,
        "maximum_accepted_held_out_relative_error": MAX_HELD_OUT_ERROR,
        "held_out_seed_strategy": "numpy.default_rng(5000 + trial_id)",
        "python_version": platform.python_version(),
        "workers": workers,
    }

    trials = []
    completed_ids = set()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError(
                f"Existing output {output_path} has different experiment metadata; "
                "choose a new --output path rather than mixing runs."
            )
        trials = existing.get("trials", [])
        completed_ids = {trial["trial_id"] for trial in trials}
        print(f"Resuming {len(completed_ids)}/{num_trials} recorded trials.")

    rng = np.random.default_rng(PERMUTATION_SEED)
    tasks = []
    for trial_id in range(num_trials):
        permutation = rng.permutation(N).tolist()
        if trial_id not in completed_ids:
            tasks.append((trial_id, matrix, permutation))

    print(
        f"TTpy {metadata['ttpy_version']} dmrg_cross: N={N}, "
        f"fixed Q SHA256={matrix_hash[:12]}..., trials={num_trials}, "
        f"beam width={BEAM_WIDTH}, tol={TOL}, workers={workers}"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_trial, task): task[0] for task in tasks}
        for future in as_completed(futures):
            trial = future.result()
            trials.append(trial)
            trials.sort(key=lambda item: item["trial_id"])
            _save_checkpoint(output_path, metadata, trials)
            greedy = trial["greedy"]
            beam = trial["beam"]
            if trial["status"] == "ok":
                print(
                    f"[{len(trials):3d}/{num_trials}] Trial {trial['trial_id']:3d}: "
                    f"greedy rank={greedy['max_bond']} params={greedy['params']} "
                    f"err={greedy['held_out_rel_error']:.2e}"
                    f"{'' if greedy['accuracy_valid'] else ' INVALID'} "
                    f"({greedy['time_sec']:.1f}s); "
                    f"beam rank={beam['max_bond']} params={beam['params']} "
                    f"err={beam['held_out_rel_error']:.2e}"
                    f"{'' if beam['accuracy_valid'] else ' INVALID'} "
                    f"({beam['time_sec']:.1f}s)"
                )
            else:
                print(
                    f"[{len(trials):3d}/{num_trials}] Trial {trial['trial_id']:3d} "
                    f"FAILED: greedy={greedy.get('error', 'ok')}; "
                    f"beam={beam.get('error', 'ok')}"
                )

    _save_checkpoint(output_path, metadata, trials)
    summary = _summarize(trials)
    print(f"\nFinished in {(time.perf_counter() - started) / 60:.1f} minutes.")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a paired fixed-Q TTpy DMRG-cross benchmark."
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()
    run_benchmark(
        num_trials=arguments.trials,
        workers=arguments.workers,
        output_path=arguments.output,
    )
