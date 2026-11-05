"""Paired nominal/SR racing checks, with feasible full-horizon opponent deviations."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from drone_racing_jax.controller import controller_step, estimate_progress, initialize_controller
from drone_racing_jax.game import (
    default_context,
    dynamics,
    rollout,
    stage_cost,
    state_constraints,
    trajectory_cost,
)
from drone_racing_jax.race import initial_state
from drone_racing_jax.robust import solve_robust_game, worst_response
from drone_racing_jax.solver import solve_game
from drone_racing_jax.splines import evaluate
from drone_racing_jax.tracks import make_track


def simulate(x0, context, track, radius, threat, steps=100, horizon=15):
    """Apply either rational nominal play or the current attacker's worst opponent plan.

    The stress policy re-maximizes at every step with the same threat radius for all
    design radii. Each complete adversarial plan is checked before its first input
    is applied. A failed solve stops the case and is reported, not scored as a win.
    """
    if steps < 1 or radius < 0 or threat < 0 or not np.isfinite(radius + threat):
        raise ValueError("Need positive steps and finite nonnegative design/threat radii")
    memory = initialize_controller(x0[:, 9], float(context[4]), horizon)
    measured = x0[:, :9]
    states, us, costs, attackers, residuals, seconds, planning = [measured], [], [], [], [], [], []
    budgets, slacks, failure = [], [], None
    for step in range(steps):
        start = perf_counter()
        result = controller_step(measured, memory, context, track, radius=radius)
        result.us.block_until_ready()
        if not bool(jnp.all(result.plans.converged)):
            failure = f"controller at step {step}: {result.plans.residual}"
            break
        memory = result.controller
        attacker = int(memory.attacker) - 1
        applied = result.us
        if threat > 0:
            order = jnp.array([attacker, 1 - attacker])
            center = result.plans.us[attacker]
            adversary = worst_response(
                center, result.planning_state[order], context.at[4].set(1.0), track, radius=threat
            )
            budget = float(jnp.linalg.norm(adversary.us - center[:, 1]))
            slack = float(
                jnp.min(
                    state_constraints(
                        rollout(result.planning_state[1 - attacker], adversary.us[:-1])[1:]
                    )
                )
            )
            if not bool(adversary.converged) or budget > threat + 1e-6 or slack < -1e-7:
                failure = f"adversary at step {step}: residual={adversary.residual}"
                break
            budgets.append(budget)
            slacks.append(slack)
            applied = applied.at[1 - attacker].set(adversary.us[0])
        applied.block_until_ready()
        seconds.append(perf_counter() - start)
        cost = stage_cost(result.planning_state, applied, context.at[4].set(memory.attacker), track)
        measured = dynamics(result.planning_state, applied)[:, :9]
        states.append(measured)
        us.append(applied)
        costs.append(cost)
        attackers.append(attacker)
        residuals.append(result.plans.residual)
        planning.append(result.planning_state)
    states = np.asarray(jnp.stack(states))
    separation = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=-1)
    complete = len(us) == steps
    record = dict(
        radius=radius,
        threat=threat,
        requested_steps=steps,
        completed_steps=len(us),
        complete=complete,
        failure=failure,
        min_separation=float(separation.min()),
        collision_steps=int(np.sum(separation[1:] < 0.35)),
        max_adversary_deviation=max(budgets, default=0.0),
        min_adversary_slack=min(slacks, default=0.0),
        worst_controller_residual=float(jnp.max(jnp.array(residuals))) if residuals else None,
        median_warm_step_ms=float(np.median(seconds[1:])) * 1000 if len(seconds) > 1 else None,
    )
    if complete:
        costs = np.asarray(jnp.stack(costs))
        progress = jax.vmap(estimate_progress, in_axes=(0, 0, None))(
            measured[:, :3], memory.progress, track
        )
        record.update(
            cost_per_player=costs.sum(axis=0).tolist(),
            attacking_cost_per_tick=float(costs[np.arange(steps), attackers].mean()),
            progress_gain=np.asarray(progress - planning[0][:, 9]).tolist(),
            role_switches=int(np.sum(np.diff(attackers) != 0)),
        )
    return record, states


def open_loop_check(track):
    progress = jnp.array([4.2, 4.7])
    x = jnp.zeros((2, 11)).at[:, 9].set(progress)
    x = x.at[:, :3].set(jax.vmap(lambda s: evaluate(track.path, s))(progress))
    c = default_context()
    nominal = solve_game(x, c, track)
    rows = []
    for radius in [0.0, 1.0, 5.0, 10.0]:
        robust = solve_robust_game(x, c, track, radius=radius)
        before = worst_response(nominal.us, x, c, track, radius=radius)
        after = worst_response(robust.us, x, c, track, radius=radius)
        assert bool(robust.converged & before.converged & after.converged)
        nominal_play = trajectory_cost(robust.xs, robust.us, c, track)[1]
        rows.append(
            dict(
                radius=radius,
                nominal_plan_worst_cost=float(before.cost),
                robust_plan_worst_cost=float(after.cost),
                robust_plan_nominal_cost=float(nominal_play),
                residual=float(robust.residual),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("outputs/robust_study"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    track, c = make_track(), default_context()
    report = dict(
        steps=args.steps,
        seeds=args.seeds,
        radii=[0.0, 5.0, 10.0],
        threats=[0.0, 5.0],
        open_loop=open_loop_check(track),
        closed_loop=[],
    )
    for threat in report["threats"]:
        for radius in report["radii"]:
            for seed in args.seeds:
                record, states = simulate(
                    initial_state(track, seed), c, track, radius, threat, steps=args.steps
                )
                record["seed"] = seed
                report["closed_loop"].append(record)
                np.save(args.output / f"states_e{radius:g}_d{threat:g}_s{seed}.npy", states)
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    plot(report, args.output / "comparison.png")


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for threat in report["threats"]:
        distances, costs = [], []
        for radius in report["radii"]:
            rows = [
                r for r in report["closed_loop"] if r["radius"] == radius and r["threat"] == threat
            ]
            complete = all(r["complete"] for r in rows)
            distances.append(np.mean([r["min_separation"] for r in rows]) if complete else np.nan)
            costs.append(
                np.mean([r["attacking_cost_per_tick"] for r in rows]) if complete else np.nan
            )
        label = "Nominal opponent" if threat == 0 else f"Adversarial opponent, radius {threat:g}"
        axes[0].plot(report["radii"], distances, "o-", label=label)
        axes[1].plot(report["radii"], costs, "o-", label=label)
    axes[0].axhline(0.35, color="black", linestyle="--", linewidth=1, label="Collision threshold")
    axes[0].set(ylabel="Mean minimum separation (m)")
    axes[1].set(ylabel="Mean attacking cost per tick (lower is better)")
    for ax in axes:
        ax.set_xlabel("Design radius epsilon (whole-horizon input L2)")
        ax.legend(fontsize=8)
    axes[0].legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
