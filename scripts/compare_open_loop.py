"""Compare JAX plans with scripts/reference_open_loop.jl output; save generated data outside Git."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from drone_racing_jax.solver import solve_game
from drone_racing_jax.splines import evaluate
from drone_racing_jax.tracks import make_track


def compare(reference):
    track = make_track()
    context = jnp.asarray(reference["context"])
    sample_error = {}
    for name, spline in [
        ("path", track.path),
        ("tangent", track.tangent),
        ("qc", track.contour_weight),
    ]:
        actual = jax.vmap(lambda s: evaluate(spline, s))(
            jnp.array([sample["s"] for sample in reference["samples"]])
        )
        expected = np.array([sample[name] for sample in reference["samples"]])
        sample_error[name] = float(np.max(np.abs(np.asarray(actual) - expected)))
    rows = []
    for case in reference["cases"]:
        x0 = jnp.asarray(case["x0"])
        horizon = case["horizon"]
        start = perf_counter()
        result = solve_game(x0, context, track, horizon=horizon)
        result.us.block_until_ready()
        compilation_and_first_solve = perf_counter() - start
        if not bool(result.converged):
            raise RuntimeError(f"{case['name']}: JAX did not converge, residual={result.residual}")
        timings = []
        for _ in range(10):
            start = perf_counter()
            result = solve_game(x0, context, track, horizon=horizon)
            result.us.block_until_ready()
            timings.append(perf_counter() - start)
        state_gap = np.asarray(result.xs) - np.array(case["xs"])
        control_gap = np.asarray(result.us) - np.array(case["us"])
        row = dict(
            name=case["name"],
            horizon=horizon,
            julia_residual=case["residual"],
            jax_residual=float(result.residual),
            max_state_gap=float(np.max(np.abs(state_gap))),
            max_position_gap=float(np.max(np.abs(state_gap[..., :3]))),
            max_control_gap=float(np.max(np.abs(control_gap))),
            julia_median_ms=case["median_seconds"] * 1000,
            jax_median_ms=median(timings) * 1000,
            compilation_and_first_solve_seconds=compilation_and_first_solve,
            jax_iterations=int(result.iterations),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
        julia_version=reference["julia_version"],
        source_commit=reference["source_commit"],
        track_length_gap=abs(float(track.length) - reference["track_length"]),
        sample_max_errors=sample_error,
        cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/reference_open_loop.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/open_loop_comparison.json"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = compare(json.loads(args.reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Track comparison:", result["sample_max_errors"])


if __name__ == "__main__":
    main()
