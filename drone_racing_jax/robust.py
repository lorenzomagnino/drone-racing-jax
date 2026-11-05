"""Local OL-SRNE candidates for the default attacker-only racing interaction.

The ambiguity is a full-horizon L2 input ball intersected with opponent feasibility.
No linear sensitivity penalty or uncertainty on own dynamics/constraints is substituted.
"""

from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from drone_racing_jax.game import collision_penalty, rollout, state_constraints, trajectory_cost
from drone_racing_jax.solver import _game_gradient, _solve_kkt, solve_game


class OpponentResponse(NamedTuple):
    us: jax.Array
    cost: jax.Array
    residual: jax.Array
    converged: jax.Array
    probe_gap: jax.Array


class RobustSolution(NamedTuple):
    xs: jax.Array
    us: jax.Array
    multipliers: jax.Array
    residual: jax.Array
    iterations: jax.Array
    converged: jax.Array
    adversary_us: jax.Array
    adversary_multipliers: jax.Array
    inner_residual: jax.Array
    probe_gap: jax.Array


def _opponent_controls(w, defender, radius):
    # Opponent path acceleration and the unused last input cannot affect ego's cost.
    return defender.at[:-1, :3].add(radius * w)


def _physical_slacks(us, x0, dt, max_velocity):
    xs = rollout(x0, us[:-1], dt)[1:]
    # vtheta is unchanged from the feasible nominal center; omit its redundant rows.
    indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
    return state_constraints(xs, max_velocity)[:, indices].reshape(-1)


def _interaction(a, w, defender, x0, context, radius, dt):
    own = rollout(x0[0], a[:-1], dt)
    opponent = rollout(x0[1], _opponent_controls(w, defender, radius)[:-1], dt)
    return context[5] * collision_penalty(own[:, :3], opponent[:, :3], context[6]).sum()


def _inner_slacks(w, defender, x0, radius, dt, max_velocity):
    physical = _physical_slacks(_opponent_controls(w, defender, radius), x0[1], dt, max_velocity)
    return jnp.concatenate((physical, (1 - jnp.sum(w**2))[None]))


def _inner_residual(w, multipliers, a, defender, x0, context, radius, dt, max_velocity):
    slacks, pullback = jax.vjp(
        lambda w: _inner_slacks(w, defender, x0, radius, dt, max_velocity), w
    )
    gradient = jax.grad(_interaction, argnums=1)(a, w, defender, x0, context, radius, dt)
    stationarity = -gradient - pullback(multipliers)[0]
    return jnp.concatenate((stationarity.reshape(-1), jnp.minimum(slacks, multipliers)))


def _residual(z, x0, context, defender, track, radius, dt, max_velocity):
    h = defender.shape[0]
    n, m, q = h * 4, (h - 1) * 3, (h - 1) * 14
    a, w = z[:n].reshape(h, 4), z[n : n + m].reshape(h - 1, 3)
    own_duals, inner_duals = z[n + m : n + m + q], z[n + m + q :]
    joint = jnp.stack((a, _opponent_controls(w, defender, radius)), axis=1)
    gradient = _game_gradient(joint, x0, context, track, dt).reshape(h, 2, 4)[:, 0]
    slacks, pullback = jax.vjp(
        lambda a: state_constraints(rollout(x0[0], a[:-1], dt)[1:], max_velocity).reshape(-1), a
    )
    own_stationarity = gradient - pullback(own_duals)[0]
    inner = _inner_residual(w, inner_duals, a, defender, x0, context, radius, dt, max_velocity)
    return jnp.concatenate(
        (own_stationarity.reshape(-1), inner[:m], jnp.minimum(slacks, own_duals), inner[m:])
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _root(x0, context, defender, track, guess, radius, dt, max_velocity, tolerance, max_iterations):
    residual = partial(
        _residual,
        x0=x0,
        context=context,
        defender=defender,
        track=track,
        radius=radius,
        dt=dt,
        max_velocity=max_velocity,
    )
    return _solve_kkt(residual, guess, defender.shape[0] * 7 - 3, tolerance, max_iterations)


def _forward(
    x0, context, defender, track, guess, radius, dt, max_velocity, tolerance, max_iterations
):
    result = _root(
        x0, context, defender, track, guess, radius, dt, max_velocity, tolerance, max_iterations
    )
    return result, (result[0], x0, context, defender, track)


def _backward(radius, dt, max_velocity, tolerance, max_iterations, saved, cotangent):
    z, x0, context, defender, track = saved
    residual = partial(_residual, track=track, radius=radius, dt=dt, max_velocity=max_velocity)
    matrix = jax.jacfwd(residual)(z, x0, context, defender)
    adjoint = jnp.linalg.solve(matrix.T, cotangent[0])
    _, back = jax.vjp(lambda x, c, d: residual(z, x, c, d), x0, context, defender)
    gradients = back(-adjoint)
    valid = jnp.max(jnp.abs(residual(z, x0, context, defender))) <= tolerance
    return (*(jnp.where(valid, g, jnp.nan) for g in gradients), None, jnp.zeros_like(z))


_root.defvjp(_forward, _backward)


def _seeds(a, defender, x0, radius, dt):
    own, opponent = rollout(x0[0], a[:-1], dt), rollout(x0[1], defender[:-1], dt)
    h = defender.shape[0]
    mapping = jax.jacfwd(
        lambda w: rollout(x0[1], _opponent_controls(w, defender, 1.0)[:-1], dt)[1:, :3]
    )(jnp.zeros((h - 1, 3)))
    reach = radius * jnp.linalg.norm(mapping[:, 0], axis=(1, 2))
    distances = jnp.linalg.norm(own[1:, :3] - opponent[1:, :3], axis=-1)
    critical = jnp.argmin(distances - reach)
    towards = jnp.einsum(
        "ijk,i->jk", mapping[critical], own[critical + 1, :3] - opponent[critical + 1, :3]
    )
    sideways = mapping[critical, 0] + mapping[critical, 1]
    directions = jnp.stack((towards, towards + 0.3 * sideways, towards - 0.3 * sideways))
    directions /= jnp.maximum(jnp.linalg.norm(directions, axis=(1, 2), keepdims=True), 1e-15)
    return directions, jnp.min(distances - reach)


def _guess(a, w, defender, x0, context, radius, dt):
    gradient = jax.grad(_interaction, argnums=1)(a, w, defender, x0, context, radius, dt)
    eta = jnp.maximum(1e-6, jnp.sum(gradient * w) / 2)
    return jnp.concatenate(
        (a.reshape(-1), w.reshape(-1), jnp.zeros((a.shape[0] - 1) * 26), eta[None])
    )


@partial(
    jax.jit,
    static_argnames=("radius", "horizon", "dt", "max_velocity", "tolerance", "max_iterations"),
)
def solve_robust_game(
    x0,
    context,
    track,
    radius,
    horizon=15,
    dt=0.05,
    max_velocity=3.0,
    initial_guess=None,
    tolerance=1e-8,
    max_iterations=100,
):
    """Local min-max KKT candidates with three adversarial starts. Check converged.

    Supports the default game (blocking weight zero, nonnegative collision weight).
    The radius is in raw full-horizon input L2 units and is fixed across planning calls.
    Exactly recovers solve_game at radius=0. No global inner-maximum certificate is claimed.
    """
    if not isfinite(radius) or radius < 0:
        raise ValueError("The ambiguity radius must be nonnegative")
    nominal = solve_game(
        x0,
        context,
        track,
        horizon=horizon,
        dt=dt,
        max_velocity=max_velocity,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    attacker = (context[4] - 1).astype(jnp.int32)
    order = jnp.array([attacker, 1 - attacker])
    canonical_x0 = x0[order]
    canonical_context = context.at[4].set(1.0)
    a, defender = nominal.us[:, attacker], nominal.us[:, 1 - attacker]
    empty_duals = jnp.zeros((horizon - 1) * 12 + 1)

    def unchanged():
        return RobustSolution(*nominal, defender, empty_duals, jnp.array(0.0), jnp.array(0.0))

    if radius == 0:
        return unchanged()
    seeds, minimum_distance = _seeds(a, defender, canonical_x0, radius, dt)

    def optimize():
        guesses = jax.vmap(
            lambda w: _guess(a, w, defender, canonical_x0, canonical_context, radius, dt)
        )(seeds)
        roots, iterations = jax.vmap(
            lambda z: _root(
                canonical_x0,
                canonical_context,
                defender,
                track,
                z,
                radius,
                dt,
                max_velocity,
                tolerance,
                max_iterations,
            )
        )(guesses)

        def assess(z):
            n, m, q = horizon * 4, (horizon - 1) * 3, (horizon - 1) * 14
            own, w = z[:n].reshape(horizon, 4), z[n : n + m].reshape(horizon - 1, 3)
            residual = _residual(
                z,
                canonical_x0,
                canonical_context,
                defender,
                track,
                radius,
                dt,
                max_velocity,
            )
            error = jnp.max(jnp.abs(residual))
            inner_error = jnp.max(
                jnp.abs(jnp.concatenate((residual[n : n + m], residual[n + m + q :])))
            )
            probes, _ = _seeds(own, defender, canonical_x0, radius, dt)
            probes = jnp.concatenate((probes, jnp.zeros_like(probes[:1])), axis=0)
            values = jax.vmap(
                lambda p: _interaction(
                    own, p, defender, canonical_x0, canonical_context, radius, dt
                )
            )(probes)
            feasible = jax.vmap(
                lambda p: jnp.min(
                    _inner_slacks(p, defender, canonical_x0, radius, dt, max_velocity)
                )
                >= -1e-7
            )(probes)
            current = _interaction(own, w, defender, canonical_x0, canonical_context, radius, dt)
            gap = jnp.maximum(0.0, jnp.max(jnp.where(feasible, values, -jnp.inf)) - current)
            controls = nominal.us.at[:, attacker].set(own)
            worst = _opponent_controls(w, defender, radius)
            cost = trajectory_cost(
                rollout(canonical_x0, jnp.stack((own, worst), axis=1)[:-1], dt),
                jnp.stack((own, worst), axis=1),
                canonical_context,
                track,
            )[0]
            return (
                controls,
                worst,
                z[n + m : n + m + q],
                z[n + m + q :],
                error,
                inner_error,
                gap,
                cost,
            )

        controls, worst, own_duals, inner_duals, errors, inner_errors, gaps, costs = jax.vmap(
            assess
        )(roots)
        valid = (errors <= tolerance) & (gaps <= 1e-7) & jnp.isfinite(costs)
        choice = jnp.where(
            jnp.any(valid), jnp.argmin(jnp.where(valid, costs, jnp.inf)), jnp.argmin(errors)
        )
        us = controls[choice]
        duals = (
            nominal.multipliers.reshape(horizon - 1, 2, 14)
            .at[:, attacker]
            .set(own_duals[choice].reshape(horizon - 1, 14))
            .reshape(-1)
        )
        residual = jnp.maximum(errors[choice], nominal.residual)
        return RobustSolution(
            rollout(x0, us[:-1], dt),
            us,
            duals,
            residual,
            nominal.iterations + iterations[choice],
            nominal.converged & valid[choice],
            worst[choice],
            inner_duals[choice],
            inner_errors[choice],
            gaps[choice],
        )

    result = jax.lax.cond((minimum_distance >= context[6]) | (context[5] == 0), unchanged, optimize)
    supported = (context[11] == 0) & (context[5] >= 0) & ((context[4] == 1) | (context[4] == 2))
    return result._replace(
        converged=result.converged & supported,
        residual=jnp.where(supported, result.residual, jnp.inf),
    )


@partial(jax.jit, static_argnames=("radius", "dt", "max_velocity", "tolerance", "max_iterations"))
def worst_response(
    us, x0, context, track, radius, dt=0.05, max_velocity=3.0, tolerance=1e-8, max_iterations=100
):
    """Independent three-start inner maximization for a fixed feasible plan.

    A validation oracle, not differentiated through. Cost is the attacker's full
    trajectory cost. Check converged; no global-maximum certificate is claimed.
    """
    if not isfinite(radius) or radius < 0:
        raise ValueError("The ambiguity radius must be nonnegative")
    attacker = (context[4] - 1).astype(jnp.int32)
    order = jnp.array([attacker, 1 - attacker])
    x, c = x0[order], context.at[4].set(1.0)
    a, defender = us[:, attacker], us[:, 1 - attacker]
    m = (us.shape[0] - 1) * 3

    def cost(opponent):
        joint = jnp.stack((a, opponent), axis=1)
        return trajectory_cost(rollout(x, joint[:-1], dt), joint, c, track)[0]

    def unchanged():
        return OpponentResponse(
            defender, cost(defender), jnp.array(0.0), jnp.array(True), jnp.array(0.0)
        )

    if radius == 0:
        return unchanged()
    seeds, minimum_distance = _seeds(a, defender, x, radius, dt)

    def optimize():
        def residual(z):
            return _inner_residual(
                z[:m].reshape(-1, 3), z[m:], a, defender, x, c, radius, dt, max_velocity
            )

        def solve(w):
            full_guess = _guess(a, w, defender, x, c, radius, dt)
            n, q = a.size, (a.shape[0] - 1) * 14
            guess = jnp.concatenate((full_guess[n : n + m], full_guess[n + m + q :]))
            z, _ = _solve_kkt(residual, guess, m, tolerance, max_iterations)
            return z, jnp.max(jnp.abs(residual(z)))

        roots, errors = jax.vmap(solve)(seeds)
        opponents = jax.vmap(lambda z: _opponent_controls(z[:m].reshape(-1, 3), defender, radius))(
            roots
        )
        values = jax.vmap(cost)(opponents)
        feasible = errors <= tolerance
        choice = jnp.where(
            jnp.any(feasible), jnp.argmax(jnp.where(feasible, values, -jnp.inf)), jnp.argmin(errors)
        )
        probes = jnp.concatenate((seeds, jnp.zeros_like(seeds[:1])), axis=0)
        probe_values = jax.vmap(lambda w: cost(_opponent_controls(w, defender, radius)))(probes)
        probe_feasible = jax.vmap(
            lambda w: jnp.min(_inner_slacks(w, defender, x, radius, dt, max_velocity)) >= -1e-7
        )(probes)
        gap = jnp.maximum(
            0.0, jnp.max(jnp.where(probe_feasible, probe_values, -jnp.inf)) - values[choice]
        )
        return OpponentResponse(
            opponents[choice], values[choice], errors[choice], feasible[choice] & (gap <= 1e-7), gap
        )

    return jax.lax.cond((minimum_distance >= context[6]) | (context[5] == 0), unchanged, optimize)
