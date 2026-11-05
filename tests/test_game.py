import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing_jax.game import (
    default_context,
    dynamics,
    equality_constraints,
    inequality_constraints,
    rollout,
    stage_cost,
    state_constraints,
    trajectory_cost,
)
from drone_racing_jax.splines import periodic_spline
from drone_racing_jax.tracks import Track


@pytest.fixture
def straight_track():
    times = np.array([0.0, 1.0, 2.0])
    path = periodic_spline(np.zeros((3, 3)), times)
    tangent = periodic_spline(np.tile([0.6, 0.8, 0.0], (3, 1)), times)
    weight = periodic_spline(np.full(3, 0.25), times)
    return Track(path, tangent, weight, jnp.array(2.0), jnp.zeros((2, 3)))


def test_dynamics_matches_julia_rk4_matrices():
    rng = np.random.default_rng(42)
    x, u = rng.normal(size=(2, 11)), rng.normal(size=(2, 4))
    a, b = np.zeros((11, 11)), np.zeros((11, 4))
    a[np.arange(6), np.arange(3, 9)] = 1
    a[9, 10] = 1
    b[np.arange(6, 9), np.arange(3)] = 1
    b[10, 3] = 1
    dt = 0.05
    ad = np.eye(11) + dt * a + dt**2 / 2 * a @ a + dt**3 / 6 * a @ a @ a
    bd = (dt * np.eye(11) + dt**2 / 2 * a + dt**3 / 6 * a @ a) @ b
    expected = x @ ad.T + u @ bd.T
    assert_allclose(jax.jit(dynamics)(x, u), expected, atol=1e-14)
    assert_allclose(jax.jit(jax.vmap(dynamics))(x, u), expected, atol=1e-14)


@pytest.mark.parametrize("attacker", [1.0, 2.0])
@pytest.mark.parametrize("blocking", [0.0, 0.5])
def test_cost_matches_literal_julia_formula(straight_track, attacker, blocking):
    x = np.array(
        [[0.1, 0.2, 0.3, 3, 2, 1, 0, 0, 0, 0.4, 0.7], [0.2, 0.4, 0.3, 1, 0, 0, 0, 0, 0, 0.5, 0.9]]
    )
    u = np.array([[1, 2, 3, 4], [0.2, -1, 0.4, 0.1]])
    context = default_context().at[4].set(attacker).at[11].set(blocking)
    expected = []
    for i in range(2):
        tangent = np.array([0.6, 0.8, 0.0])
        lag = (tangent @ x[i, :3]) * tangent
        contour = x[i, :3] - lag
        attacking = float(i + 1 == attacker)
        max_speed = 2 + attacking
        speed_gap = max_speed**2 - x[i, 3:6] @ x[i, 3:6]
        separation = np.sum((x[0, :3] - x[1, :3]) ** 2)
        collision = 0 if separation >= 1 else (abs(separation) - 1) ** 2
        expected.append(
            3 * (lag @ lag)
            + (1.5 * 0.25 + 1.5) * (contour @ contour)
            + 0.001 * (u[i] @ u[i])
            + 0.075 * (speed_gap - abs(speed_gap)) ** 2
            + (attacking * 1.5 - (1 - attacking) * blocking) * collision
            - 1.5 * x[i, 10]
        )
    assert_allclose(jax.jit(stage_cost)(x, u, context, straight_track), expected, atol=1e-12)
    swapped = stage_cost(
        x[::-1].copy(), u[::-1].copy(), context.at[4].set(3 - attacker), straight_track
    )
    assert_allclose(swapped, np.array(expected)[::-1], atol=1e-12)


@pytest.mark.parametrize("distance", [0.5, 1.0, 1.5])
def test_collision_hinge_and_role(straight_track, distance):
    x = jnp.zeros((2, 11)).at[1, 0].set(distance)
    context = jnp.zeros(12).at[4].set(1).at[5].set(2).at[6].set(1)
    expected = [2 * max(1 - distance**2, 0) ** 2, 0]
    assert_allclose(stage_cost(x, jnp.zeros((2, 4)), context, straight_track), expected)


def test_horizon_cost_and_constraints(straight_track):
    x0 = jnp.zeros((2, 11)).at[:, 3].set(0.2)
    us = jnp.full((4, 2, 4), 0.1)
    xs = jax.jit(rollout)(x0, us)
    assert_allclose(jax.jit(equality_constraints)(xs, us, x0), 0, atol=1e-14)
    assert_allclose(equality_constraints(xs[:-1], us, x0), 0, atol=1e-14)
    assert inequality_constraints(xs[:-1]).shape == (3 * 2 * 14,)
    assert np.all(inequality_constraints(xs[:-1]) > 0)
    context = default_context()
    baseline = trajectory_cost(xs[:-1], us, context, straight_track)
    assert_allclose(jax.jit(trajectory_cost)(xs, us, context, straight_track), baseline)
    changed_terminal = xs.at[-1].set(1e6)
    assert_allclose(trajectory_cost(changed_terminal, us, context, straight_track), baseline)
    changed_last_u = us.at[-1].add(1)
    assert np.all(trajectory_cost(xs[:-1], changed_last_u, context, straight_track) > baseline)
    invalid = x0.at[0, 0].set(21).at[1, 10].set(-4)
    assert np.sum(np.asarray(state_constraints(invalid)) < 0) == 2


def test_cost_parameter_gradient(straight_track):
    x = jnp.ones((2, 11))
    u = jnp.ones((2, 4))
    context = default_context()
    loss = jax.jit(lambda c: stage_cost(x, u, c, straight_track).sum())
    gradient = jax.grad(loss)(context)
    for index in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
        delta = jnp.zeros(12).at[index].set(1e-5)
        finite_difference = (loss(context + delta) - loss(context - delta)) / 2e-5
        assert_allclose(gradient[index], finite_difference, atol=1e-7)
