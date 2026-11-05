"""Re-run Julia's closed-loop inputs in JAX; report trajectory and role agreement."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from drone_racing_jax.controller import run_closed_loop
from drone_racing_jax.tracks import make_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("outputs/reference_closed_loop.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/closed_loop_comparison.json"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    reference = json.loads(args.reference.read_text())
    track, rows = make_track(), []
    for case in reference["cases"]:
        result = run_closed_loop(
            jnp.asarray(case["x0"]),
            jnp.asarray(reference["context"]),
            track,
            steps=case["steps"],
            horizon=case["horizon"],
        )
        states = np.asarray(result.states) - np.asarray(case["states"])
        controls = np.asarray(result.us) - np.asarray(case["us"])
        row = dict(
            name=case["name"],
            steps=case["steps"],
            max_position_gap=float(np.max(np.abs(states[..., :3]))),
            max_physical_state_gap=float(np.max(np.abs(states))),
            max_input_gap=float(np.max(np.abs(controls))),
            role_mismatches=int(np.sum(np.asarray(result.attackers) != case["attackers"])),
            role_switches=int(jnp.sum(result.attackers[1:] != result.attackers[:-1])),
            jax_worst_residual=float(result.residuals.max()),
            julia_worst_residual=float(np.max(case["residuals"])),
            jax_median_warm_step_ms=float(jnp.median(result.step_seconds[1:])) * 1000,
            julia_median_warm_step_ms=float(np.median(case["step_seconds"][1:])) * 1000,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output.parent / f"{case['name']}_jax.npz", **result._asdict())
    args.output.write_text(
        json.dumps(
            dict(jax_version=jax.__version__, julia_version=reference["julia_version"], cases=rows),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
