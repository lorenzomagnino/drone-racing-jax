"""Run a seeded, synchronous MPG simulation without training."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from drone_racing_jax.controller import run_closed_loop
from drone_racing_jax.game import default_context
from drone_racing_jax.splines import evaluate
from drone_racing_jax.tracks import make_track


def initial_state(track, seed=42):
    """Original standard starts, with independent Cartesian offsets uniform in ±0.15 m."""
    progress = jnp.array([track.length - 1, track.length - 2.5])
    positions = jax.vmap(lambda s: evaluate(track.path, s))(progress)
    offsets = np.random.default_rng(seed).uniform(-0.15, 0.15, size=(2, 3))
    return jnp.zeros((2, 11)).at[:, :3].set(positions + offsets).at[:, 9].set(progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/race"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    track = make_track()
    result = run_closed_loop(
        initial_state(track, args.seed), default_context(), track, steps=args.steps
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "trajectory.npz", **result._asdict())
    summary = dict(
        steps=args.steps,
        seed=args.seed,
        dt=0.05,
        horizon=15,
        worst_kkt_residual=float(result.residuals.max()),
        role_switches=int(jnp.sum(result.attackers[1:] != result.attackers[:-1])),
        first_step_seconds=float(result.step_seconds[0]),
        median_warm_step_ms=(
            float(jnp.median(result.step_seconds[1:])) * 1000 if args.steps > 1 else None
        ),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = jax.vmap(lambda s: evaluate(track.path, s))(jnp.linspace(0, track.length, 500))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1, label="Track")
    for player in range(2):
        ax.plot(result.states[:, player, 0], result.states[:, player, 1], label=f"Drone {player+1}")
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal", title="MPG closed loop, no training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "trajectory.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
