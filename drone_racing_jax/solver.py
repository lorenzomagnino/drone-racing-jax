"""Condensed Nash KKT solve with semismooth Newton and implicit differentiation."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from drone_racing_jax.game import _stage_cost, inequality_constraints, rollout
from drone_racing_jax.splines import evaluate_polynomial


class Solution(NamedTuple):
    xs: jax.Array
    us: jax.Array
    multipliers: jax.Array
    residual: jax.Array
    iterations: jax.Array
    converged: jax.Array


def _states(us, x0, dt):
    # The Julia MPG's last control is costed but has no successor state.
    return rollout(x0, us[:-1], dt)


def _slacks(us, x0, dt, max_velocity):
    return inequality_constraints(_states(us, x0, dt), max_velocity)


def _stage_gradient(x, u, context, track):
    theta = x[:, 9]
    splines = (track.path, track.tangent, track.contour_weight)
    values = [jax.vmap(lambda t: evaluate_polynomial(spline, t))(theta) for spline in splines]
    derivatives = [
        jax.vmap(lambda t: evaluate_polynomial(spline, t, 1))(theta) for spline in splines
    ]
    gx, gu, goal, tangent, qc = jax.jacrev(_stage_cost, argnums=(0, 1, 3, 4, 5))(
        x, u, context, *values
    )
    players = jnp.arange(2)
    gx, gu = gx[players, players], gu[players, players]
    theta_gradient = (
        jnp.sum(goal[players, players] * derivatives[0], axis=-1)
        + jnp.sum(tangent[players, players] * derivatives[1], axis=-1)
        + qc[players, players] * derivatives[2]
    )
    return gx.at[:, 9].add(theta_gradient), gu


def _game_gradient(us, x0, context, track, dt):
    xs, pullback = jax.vjp(lambda u: _states(u, x0, dt), us)
    gx, gu = jax.vmap(_stage_gradient, in_axes=(0, 0, None, None))(xs, us, context, track)
    # Keep Julia's cached first derivatives, then linearize their actual polynomials.
    return (pullback(gx)[0] + gu).reshape(-1)


def kkt_residual(us, multipliers, x0, context, track, dt=0.05, max_velocity=3.0):
    """Stationarity and min(slack, multiplier); zero means feasible complementarity.

    Dynamics and initial-state equalities hold by construction. Multipliers follow
    game.inequality_constraints order, with L_i = J_i - lambda_i @ slack_i.
    """
    slacks, pullback = jax.vjp(lambda u: _slacks(u, x0, dt, max_velocity), us)
    stationarity = _game_gradient(us, x0, context, track, dt) - pullback(multipliers)[0].reshape(-1)
    return jnp.concatenate((stationarity, jnp.minimum(slacks, multipliers)))


def _residual(z, x0, context, track, horizon, dt, max_velocity):
    n = horizon * 8
    return kkt_residual(z[:n].reshape(horizon, 2, 4), z[n:], x0, context, track, dt, max_velocity)


def _newton(
    x0, context, track, initial_guess, horizon, dt, max_velocity, tolerance, max_iterations
):
    n = horizon * 8
    z = jnp.concatenate((initial_guess.reshape(-1), jnp.zeros((horizon - 1) * 28)))

    def residual(z):
        return _residual(z, x0, context, track, horizon, dt, max_velocity)

    return _solve_kkt(residual, z, n, tolerance, max_iterations)


def _solve_kkt(residual, z, n, tolerance, max_iterations):
    """Shared damped Newton solve for nominal and strategically robust KKT systems."""

    def condition(state):
        z, iteration, moving = state
        error = jnp.max(jnp.abs(residual(z)))
        return (iteration < max_iterations) & (error > tolerance) & jnp.isfinite(error) & moving

    def step(state):
        z, iteration, _ = state
        value = residual(z)
        matrix = jax.jacfwd(residual)(z)
        # A small primal regularizer stabilizes nearly singular local KKT systems.
        matrix = matrix.at[jnp.arange(n), jnp.arange(n)].add(1e-9)
        merit = jnp.sum(value**2)

        def take_step(direction):
            def backtrack(state):
                alpha, count = state
                trial = residual(z + alpha * direction)
                insufficient = jnp.sum(trial**2) > (1 - 1e-4 * alpha) * merit
                return (count < 20) & (insufficient | ~jnp.all(jnp.isfinite(trial)))

            alpha, _ = jax.lax.while_loop(
                backtrack, lambda state: (state[0] / 2, state[1] + 1), (jnp.array(1.0), 0)
            )
            trial_z = z + alpha * direction
            improvement = jnp.all(jnp.isfinite(trial_z)) & (jnp.sum(residual(trial_z) ** 2) < merit)
            return trial_z, improvement

        trial_z, improvement = take_step(jnp.linalg.solve(matrix, -value))

        def safeguard(_):
            # A least-squares step can leave a stalled Newton direction near a kink.
            normal = matrix.T @ matrix + 1e-5 * jnp.eye(z.size)
            return take_step(jnp.linalg.solve(normal, -matrix.T @ value))

        trial_z, improvement = jax.lax.cond(
            improvement, lambda _: (trial_z, improvement), safeguard, operand=None
        )
        return jnp.where(improvement, trial_z, z), iteration + 1, improvement

    z, iterations, _ = jax.lax.while_loop(condition, step, (z, 0, jnp.array(True)))
    return z, iterations


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _root(x0, context, track, initial_guess, horizon, dt, max_velocity, tolerance, max_iterations):
    return _newton(
        x0, context, track, initial_guess, horizon, dt, max_velocity, tolerance, max_iterations
    )


def _root_forward(
    x0, context, track, initial_guess, horizon, dt, max_velocity, tolerance, max_iterations
):
    result = _newton(
        x0, context, track, initial_guess, horizon, dt, max_velocity, tolerance, max_iterations
    )
    return result, (result[0], x0, context, track)


def _root_backward(horizon, dt, max_velocity, tolerance, max_iterations, saved, cotangent):
    z, x0, context, track = saved

    def residual(z, x0, context):
        return _residual(z, x0, context, track, horizon, dt, max_velocity)

    matrix = jax.jacfwd(residual)(z, x0, context)
    adjoint = jnp.linalg.solve(matrix.T, cotangent[0])
    _, pullback = jax.vjp(lambda x0, context: residual(z, x0, context), x0, context)
    grad_x0, grad_context = pullback(-adjoint)
    converged = jnp.max(jnp.abs(residual(z, x0, context))) <= tolerance
    grad_x0 = jnp.where(converged, grad_x0, jnp.nan)
    grad_context = jnp.where(converged, grad_context, jnp.nan)
    return grad_x0, grad_context, None, jnp.zeros((horizon, 2, 4), dtype=z.dtype)


_root.defvjp(_root_forward, _root_backward)


@partial(jax.jit, static_argnames=("horizon", "dt", "max_velocity", "tolerance", "max_iterations"))
def solve_game(
    x0,
    context,
    track,
    horizon=15,
    dt=0.05,
    max_velocity=3.0,
    initial_guess=None,
    tolerance=1e-8,
    max_iterations=100,
):
    """Solve the original finite-horizon game, differentiable in context and initial state.

    Check result.converged before using a plan. Failed solves have NaN implicit gradients.
    An optional (H,2,4) initial_guess supplies controls; track data/config are fixed.
    The hard speed bound is independent of the soft cost's context entries, as in Julia.
    """
    if horizon < 2 or dt <= 0 or max_velocity <= 0 or tolerance <= 0 or max_iterations < 0:
        raise ValueError("Need H>=2, positive dt/speed/tolerance, and nonnegative iteration limit")
    if x0.shape != (2, 11) or context.shape != (12,):
        raise ValueError("Expected x0 shape (2,11) and context shape (12,)")
    if initial_guess is None:
        initial_guess = jnp.zeros((horizon, 2, 4), dtype=x0.dtype)
    if initial_guess.shape != (horizon, 2, 4):
        raise ValueError("Initial guess must have shape (horizon,2,4)")
    z, iterations = _root(
        x0, context, track, initial_guess, horizon, dt, max_velocity, tolerance, max_iterations
    )
    us, multipliers = z[: horizon * 8].reshape(horizon, 2, 4), z[horizon * 8 :]
    residual = jax.lax.stop_gradient(
        jnp.max(jnp.abs(kkt_residual(us, multipliers, x0, context, track, dt, max_velocity)))
    )
    return Solution(
        _states(us, x0, dt),
        us,
        multipliers,
        residual,
        iterations,
        jnp.isfinite(residual) & (residual <= tolerance),
    )
