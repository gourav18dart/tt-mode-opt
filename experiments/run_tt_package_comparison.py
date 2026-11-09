"""Run the same TT-cross benchmark with Teneva and TTpy and compare results."""

import argparse
import json
from pathlib import Path

from run_ttcross_benchmark import run_ttcross_experiment


def run_comparison(N=8, n_trials=5, beam_width=3, tol=1e-6, seed=42):
    results = {
        backend: run_ttcross_experiment(
            N=N,
            n_trials=n_trials,
            beam_width=beam_width,
            tol=tol,
            seed=seed,
            backend=backend,
        )
        for backend in ("teneva", "ttpy")
    }

    comparison = {}
    for method in ("algo4_paper", "beam_search_ours"):
        teneva_method = results["teneva"][method]
        ttpy_method = results["ttpy"][method]
        teneva_params = teneva_method["avg_params"]
        teneva_time = teneva_method["avg_time_sec"]
        comparison[method] = {
            "optimal_rate_delta_percentage_points": (
                ttpy_method["opt_rate"] - teneva_method["opt_rate"]
            ),
            "average_parameters_delta": ttpy_method["avg_params"] - teneva_params,
            "average_parameters_delta_percent": (
                100.0 * (ttpy_method["avg_params"] - teneva_params) / teneva_params
                if teneva_params
                else 0.0
            ),
            "average_time_delta_sec": ttpy_method["avg_time_sec"] - teneva_time,
            "average_time_delta_percent": (
                100.0 * (ttpy_method["avg_time_sec"] - teneva_time) / teneva_time
                if teneva_time
                else 0.0
            ),
        }

    output = {
        "N": N,
        "n_trials": n_trials,
        "beam_width": beam_width,
        "tolerance": tol,
        "seed": seed,
        "same_seed_and_initial_permutations": True,
        "results_by_backend": results,
        "ttpy_minus_teneva": comparison,
    }
    out_file = Path(__file__).resolve().parent / "results" / "tt_package_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as result_file:
        json.dump(output, result_file, indent=2)
    print(f"\nSaved paired backend comparison to {out_file}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Teneva and TTpy TT-cross results.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_comparison(n_trials=args.trials, beam_width=args.beam_width, seed=args.seed)
