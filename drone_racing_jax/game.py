"""Default two-player jerk game from dynamics.jl, game.jl and Server/server_utils.jl."""

import jax
import jax.numpy as jnp

from drone_racing_jax.splines import evaluate


def dynamics(x, u, dt=0.05):
    """Exact RK4/ZOH step: x=(p[3],v[3],a[3],theta,vtheta), u=(jerk[3],atheta).

    Leading batch/player dimensions are supported. Mass is unused in the Julia jerk model.
    """
    p, v, a = x[..., :3], x[..., 3:6], x[..., 6:9]
    jerk, atheta = u[..., :3], u[..., 3:4]
    return jnp.concatenate(
        (
            p + dt * v + dt**2 / 2 * a + dt**3 / 6 * jerk,
            v + dt * a + dt**2 / 2 * jerk,
            a + dt * jerk,
            x[..., 9:10] + dt * x[..., 10:11] + dt**2 / 2 * atheta,
            x[..., 10:11] + dt * atheta,
        ),
        axis=-1,
    )


def rollout(x0, us, dt=0.05):
    """Open-loop dynamics rollout, including x0 (H inputs -> H+1 states)."""

    def step(x, u):
        next_x = dynamics(x, u, dt)
        return next_x, next_x

    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate((x0[None], xs), axis=0)


def default_context():
    """MPG medium-speed context, in Julia construct_context order; attacker IDs are 1/2."""
    return jnp.array([3.0, 1.5, 1.5, 0.001, 2.0, 1.5, 1.0, 0.075, 2.0, 3.0, 1.5, 0.0])


def stage_cost(x, u, context, track):
    """Return both players' costs for joint x=(2,11), u=(2,4).

    The last context entry enables the original MPCCGB blocking variant (zero for MPG/LMPG).
    """
    theta = x[:, 9]
    goal = jax.vmap(evaluate, in_axes=(None, 0))(track.path, theta)
    tangent = jax.vmap(evaluate, in_axes=(None, 0))(track.tangent, theta)
    qc = jax.vmap(evaluate, in_axes=(None, 0))(track.contour_weight, theta)
    return _stage_cost(x, u, context, goal, tangent, qc)


def _stage_cost(x, u, context, goal, tangent, qc):
    """Cost algebra with independent path features; shared with the KKT assembly."""
    error = x[:, :3] - goal
    lag = jnp.sum(tangent * error, axis=-1, keepdims=True) * tangent
    contour = error - lag
    attacker = jnp.array([2 - context[4], context[4] - 1])
    max_velocity = (1 - attacker) * context[8] + attacker * context[9]
    speed_gap = max_velocity**2 - jnp.sum(x[:, 3:6] ** 2, axis=-1)
    velocity_error = (speed_gap - jnp.abs(speed_gap)) ** 2
    collision_error = collision_penalty(x[0, :3], x[1, :3], context[6])
    return (
        context[0] * jnp.sum(lag**2, axis=-1)
        + (context[1] * qc + context[2]) * jnp.sum(contour**2, axis=-1)
        + context[3] * jnp.sum(u**2, axis=-1)
        + context[7] * velocity_error
        + (attacker * context[5] - (1 - attacker) * context[11]) * collision_error
        - context[10] * x[:, 10]
    )


def collision_penalty(first, second, radius):
    """Original soft collision penalty; leading time/batch dimensions are supported."""
    separation = jnp.sum((first - second) ** 2, axis=-1)
    return jnp.where(separation >= radius**2, 0.0, (jnp.abs(separation) - radius**2) ** 2)


def trajectory_cost(xs, us, context, track):
    """Julia sum(zip(xs,us)): no dt scaling or additional terminal cost.

    MPG: H states, H controls (last control has a cost but no successor state).
    LMPG: H+1 states, H controls (final state has no cost).
    """
    horizon = min(xs.shape[0], us.shape[0])
    return jax.vmap(stage_cost, in_axes=(0, 0, None, None))(
        xs[:horizon], us[:horizon], context, track
    ).sum(axis=0)


def state_constraints(x, max_velocity=3.0):
    """Nonnegative feasible slacks: world box ±20, v_xyz and vtheta ±attacker speed.

    No hard collision, acceleration, progress, track-corridor, or input constraints.
    """
    bounded_velocity = x[..., jnp.array([3, 4, 5, 10])]
    return jnp.concatenate(
        (
            20 + x[..., :3],
            20 - x[..., :3],
            max_velocity + bounded_velocity,
            max_velocity - bounded_velocity,
        ),
        axis=-1,
    )


def equality_constraints(xs, us, x0, dt=0.05):
    """MPG equality residuals; also accepts a standard H+1-state rollout."""
    transitions = xs[1:] - dynamics(xs[:-1], us[: xs.shape[0] - 1], dt)
    return jnp.concatenate((transitions.reshape(-1), (xs[0] - x0).reshape(-1)))


def inequality_constraints(xs, max_velocity=3.0):
    """MPG excludes the fixed initial state from private inequalities."""
    return state_constraints(xs[1:], max_velocity).reshape(-1)
