"""Synchronous, decentralized MPG with measured-state feedback and shifted warm starts."""

from functools import partial
from math import isfinite
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp

from drone_racing_jax.game import dynamics
from drone_racing_jax.robust import RobustSolution, solve_robust_game
from drone_racing_jax.solver import Solution, solve_game
from drone_racing_jax.splines import evaluate


class ControllerState(NamedTuple):
    progress: jax.Array
    attacker: jax.Array
    guesses: jax.Array  # ego player, time, player in that ego's game, control


class Step(NamedTuple):
    controller: ControllerState
    planning_state: jax.Array
    us: jax.Array
    plans: Solution | RobustSolution


class ClosedLoop(NamedTuple):
    states: jax.Array  # measured physical states, including the initial state
    planning_states: jax.Array
    us: jax.Array
    attackers: jax.Array
    residuals: jax.Array
    iterations: jax.Array
    step_seconds: jax.Array


def estimate_progress(position, previous, track):
    """Optim.jl 1.13.3 bounded Brent search on previous±2.5, abs/rel tolerance 1e-5."""

    def objective(s):
        return jnp.sum((evaluate(track.path, s) - position) ** 2)

    ratio = (3 - jnp.sqrt(5.0)) / 2
    a, b = previous - 2.5, previous + 2.5
    x = a + ratio * (b - a)
    fx = objective(x)

    def condition(state):
        a, b, x, _, _, _, _, _, _, _, iteration = state
        tol = 1e-5 * jnp.abs(x) + 1e-5
        return (iteration < 1000) & (jnp.abs(x - (a + b) / 2) > 2 * tol - (b - a) / 2)

    def step(state):
        a, b, x, w, v, fx, fw, fv, d, e, iteration = state
        tol = 1e-5 * jnp.abs(x) + 1e-5
        midpoint = (a + b) / 2
        r, q = (x - w) * (fx - fv), (x - v) * (fx - fw)
        p, q = (x - v) * q - (x - w) * r, 2 * (q - r)
        p, q = jnp.where(q > 0, -p, p), jnp.abs(q)
        parabolic = (
            (jnp.abs(e) > tol)
            & (jnp.abs(p) < jnp.abs(q * e / 2))
            & (p < q * (b - x))
            & (p < q * (x - a))
        )
        candidate = p / jnp.where(q == 0, 1, q)
        near_bound = ((x + candidate - a) < 2 * tol) | ((b - x - candidate) < 2 * tol)
        candidate = jnp.where(near_bound, jnp.where(x < midpoint, tol, -tol), candidate)
        e = jnp.where(parabolic, d, jnp.where(x < midpoint, b - x, a - x))
        d = jnp.where(parabolic, candidate, ratio * e)
        u = x + jnp.where(jnp.abs(d) >= tol, d, jnp.where(d > 0, tol, -tol))
        fu = objective(u)
        better = fu < fx
        next_a = jnp.where(better, jnp.where(u < x, a, x), jnp.where(u < x, u, a))
        next_b = jnp.where(better, jnp.where(u < x, x, b), jnp.where(u < x, b, u))
        replace_w = (fu <= fw) | (w == x)
        replace_v = (fu <= fv) | (v == x) | (v == w)
        next_v = jnp.where(better | replace_w, w, jnp.where(replace_v, u, v))
        next_fv = jnp.where(better | replace_w, fw, jnp.where(replace_v, fu, fv))
        return (
            next_a,
            next_b,
            jnp.where(better, u, x),
            jnp.where(better, x, jnp.where(replace_w, u, w)),
            next_v,
            jnp.where(better, fu, fx),
            jnp.where(better, fx, jnp.where(replace_w, fu, fw)),
            next_fv,
            d,
            e,
            iteration + 1,
        )

    return jax.lax.while_loop(condition, step, (a, b, x, x, x, fx, fx, fx, 0.0, 0.0, 0))[2]


def augment_measurement(measured, progress, track):
    """Rebuild theta and nonnegative projected vtheta from physical (p,v,a) measurements."""
    tangents = jax.vmap(lambda s: evaluate(track.tangent, s))(progress)
    projection = (jnp.sum(measured[:, 3:6] * tangents, axis=-1) / jnp.sum(tangents**2, axis=-1))[
        :, None
    ] * tangents
    speed = jnp.linalg.norm(projection, axis=-1)
    return jnp.concatenate((measured, progress[:, None], speed[:, None]), axis=-1)


def update_attacker(progress, attacker):
    """Switch only after the current attacker leads by strictly more than 0.75 m."""
    difference = progress[0] - progress[1]
    return jnp.where(
        (attacker == 1) & (difference > 0.75),
        2,
        jnp.where((attacker == 2) & (difference < -0.75), 1, attacker),
    )


def shift_controls(us):
    """Advance a plan one step, appending zero (Julia's unused final control is also zero)."""
    return jnp.concatenate((us[1:], jnp.zeros_like(us[:1])), axis=0)


def initialize_controller(progress, attacker=2, horizon=15):
    if progress.shape != (2,) or horizon < 2 or attacker not in (1, 2):
        raise ValueError("Need two progress estimates, attacker 1/2, and horizon>=2")
    return ControllerState(
        progress, jnp.asarray(attacker, dtype=jnp.int32), jnp.zeros((2, horizon, 2, 4))
    )


@partial(jax.jit, static_argnames=("dt", "max_velocity", "tolerance", "max_iterations", "radius"))
def controller_step(
    measured,
    state,
    context,
    track,
    dt=0.05,
    max_velocity=3.0,
    tolerance=1e-8,
    max_iterations=100,
    radius=0.0,
):
    """Re-solve both ego games from measured (2,9) physical states. Check plans.converged.

    A positive fixed radius selects local OL-SRNE candidates; zero retains nominal MPG.
    Failed solves yield NaN actions and retain the previous controller state. The
    host simulation raises before applying any such action.
    """
    if measured.shape != (2, 9):
        raise ValueError("Expected measured physical state shape (2,9)")
    if not isfinite(radius) or radius < 0:
        raise ValueError("The ambiguity radius must be nonnegative")
    horizon = state.guesses.shape[1]
    progress = jax.vmap(estimate_progress, in_axes=(0, 0, None))(
        measured[:, :3], state.progress, track
    )
    augmented = augment_measurement(measured, progress, track)
    attacker = update_attacker(progress, state.attacker)
    ego_states = jnp.stack((augmented, augmented[::-1]))
    contexts = jnp.stack((context.at[4].set(attacker), context.at[4].set(3 - attacker)))
    solver = partial(solve_robust_game, radius=radius) if radius > 0 else solve_game
    plans = jax.vmap(
        lambda x, c, guess: solver(
            x,
            c,
            track,
            horizon=horizon,
            dt=dt,
            max_velocity=max_velocity,
            initial_guess=guess,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )(ego_states, contexts, state.guesses)
    valid = jnp.all(plans.converged)
    guesses = jax.vmap(shift_controls)(plans.us)
    next_state = ControllerState(
        jnp.where(valid, progress, state.progress),
        jnp.where(valid, attacker, state.attacker),
        jnp.where(valid, guesses, state.guesses),
    )
    return Step(next_state, augmented, jnp.where(valid, plans.us[:, 0, 0], jnp.nan), plans)


def run_closed_loop(x0, context, track, steps=100, horizon=15, dt=0.05, radius=0.0):
    """Synchronous noiseless point-mass simulation. No training or race termination rules.

    x0 is (2,11); theta supplies the initial projection guess, while vtheta is remeasured.
    Each step solves independent ego games, applies each ego's first input, then measures
    the resulting physical state. Nonconvergence raises before advancing the plant.
    """
    if x0.shape != (2, 11) or steps < 1:
        raise ValueError("Need initial state shape (2,11) and at least one step")
    state = initialize_controller(x0[:, 9], float(context[4]), horizon)
    measured = x0[:, :9]
    states, planning_states, controls, attackers, residuals, iterations, times = (
        [measured],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index in range(steps):
        start = perf_counter()
        result = controller_step(measured, state, context, track, dt=dt, radius=radius)
        result.us.block_until_ready()
        times.append(perf_counter() - start)
        if not bool(jnp.all(result.plans.converged)):
            raise RuntimeError(
                f"MPG solve failed at step {index}: residuals={result.plans.residual}"
            )
        state = result.controller
        measured = dynamics(result.planning_state, result.us, dt)[:, :9]
        states.append(measured)
        planning_states.append(result.planning_state)
        controls.append(result.us)
        attackers.append(state.attacker)
        residuals.append(result.plans.residual)
        iterations.append(result.plans.iterations)
    return ClosedLoop(
        *(
            jnp.stack(values)
            for values in (states, planning_states, controls, attackers, residuals, iterations)
        ),
        jnp.asarray(times),
    )
