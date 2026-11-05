import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing_jax.controller import run_closed_loop
from drone_racing_jax.game import default_context, rollout, state_constraints, trajectory_cost
from drone_racing_jax.robust import _inner_residual, solve_robust_game, worst_response
from drone_racing_jax.robust_study import simulate
from drone_racing_jax.solver import solve_game
from drone_racing_jax.splines import evaluate
from drone_racing_jax.tracks import make_track


@pytest.fixture(scope="module")
def problem():
    track = make_track()
    progress = jnp.array([4.2, 4.7])
    x = jnp.zeros((2, 11)).at[:, 9].set(progress)
    x = x.at[:, :3].set(jax.vmap(lambda s: evaluate(track.path, s))(progress))
    return x, default_context(), track


def test_zero_radius_is_exact_nominal_solver(problem):
    x, c, track = problem
    nominal = solve_game(x, c, track, horizon=4)
    robust = solve_robust_game(x, c, track, radius=0.0, horizon=4)
    for expected, actual in zip(nominal, robust[:6]):
        assert_allclose(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("attacker", [1, 2])
def test_minmax_kkt_ball_and_opponent_feasibility(problem, attacker):
    x, c, track = problem
    c = c.at[4].set(float(attacker))
    radius = 5.0
    result = solve_robust_game(x, c, track, radius=radius)
    nominal = solve_game(x, c, track)
    assert result.converged, (result.residual, result.probe_gap)
    a, d = attacker - 1, 2 - attacker
    assert_allclose(result.us[:, d], nominal.us[:, d], atol=0, rtol=0)
    assert np.max(np.abs(result.us[:, a] - nominal.us[:, a])) > 1e-4
    assert np.min(state_constraints(result.xs[1:])) >= -1e-8
    deviation = result.adversary_us - result.us[:, d]
    assert float(jnp.linalg.norm(deviation)) <= radius + 1e-6
    assert_allclose(deviation[:, 3], 0, atol=0)
    assert_allclose(deviation[-1], 0, atol=0)
    opponent_states = rollout(x[d], result.adversary_us[:-1])
    assert np.min(state_constraints(opponent_states[1:])) >= -1e-8
    w = deviation[:-1, :3] / radius
    residual = _inner_residual(
        w,
        result.adversary_multipliers,
        result.us[:, a],
        result.us[:, d],
        x[jnp.array([a, d])],
        c.at[4].set(1.0),
        radius,
        0.05,
        3.0,
    )
    assert_allclose(residual, 0, atol=1e-8)
    assert_allclose(result.inner_residual, jnp.max(jnp.abs(residual)), atol=1e-12)
    oracle = worst_response(result.us, x, c, track, radius=radius)
    assert oracle.converged
    joint = result.us.at[:, d].set(result.adversary_us)
    own_gradient = jax.grad(
        lambda controls: trajectory_cost(rollout(x, controls[:-1]), controls, c, track)[a]
    )(joint)[:, a]
    _, back = jax.vjp(
        lambda own: state_constraints(rollout(x[a], own[:-1])[1:]).reshape(-1), result.us[:, a]
    )
    own_duals = result.multipliers.reshape(-1, 2, 14)[:, a].reshape(-1)
    assert_allclose(own_gradient - back(own_duals)[0], 0, atol=1e-8)
    cost = trajectory_cost(rollout(x, joint[:-1]), joint, c, track)[a]
    assert_allclose(oracle.cost, cost, atol=2e-7)
    nominal_worst = worst_response(nominal.us, x, c, track, radius=radius)
    assert nominal_worst.converged
    assert float(oracle.cost) < float(nominal_worst.cost) - 1e-4


@pytest.mark.parametrize("velocity_limited", [False, True])
def test_inner_maximum_matches_exact_one_transition_geometry(problem, velocity_limited):
    _, c, track = problem
    x = jnp.zeros((2, 11)).at[:, 2].set(1.9).at[1, 0].set(0.7).at[:, 9].set(4.2)
    if velocity_limited:
        x = x.at[0, 3].set(2.99)
    us = jnp.zeros((2, 2, 4))
    radius, dt = 1000.0, 0.05
    result = worst_response(us, x, c, track, radius=radius)
    assert result.converged, (result.residual, result.probe_gap)
    center = rollout(x, us[:-1])
    distance = jnp.linalg.norm(center[1, 0, :3] - center[1, 1, :3])
    max_jerk = (3 - 2.99) / (dt**2 / 2) if velocity_limited else radius
    closest = distance - dt**3 / 6 * max_jerk
    nominal = trajectory_cost(center, us, c, track)[1]
    expected = nominal + c[5] * ((c[6] ** 2 - closest**2) ** 2 - (c[6] ** 2 - distance**2) ** 2)
    assert_allclose(result.cost, expected, atol=1e-8)
    assert float(jnp.linalg.norm(result.us - us[:, 0])) <= radius + 1e-5
    assert np.min(state_constraints(rollout(x[0], result.us[:-1])[1:])) >= -1e-8


def test_robust_implicit_gradients(problem):
    x, c, track = problem
    loss = jax.jit(
        lambda c: solve_robust_game(x, c, track, radius=5.0, horizon=4, tolerance=1e-10).us[0, 1, 0]
    )
    gradient = jax.jit(jax.grad(loss))(c)
    for index in [0, 3, 5, 6, 10]:
        epsilon = 1e-5 * max(abs(float(c[index])), 1e-3)
        delta = jnp.zeros(12).at[index].set(epsilon)
        finite_difference = (loss(c + delta) - loss(c - delta)) / (2 * epsilon)
        assert_allclose(gradient[index], finite_difference, atol=2e-6, rtol=2e-5)


def test_no_interaction_recovers_nominal(problem):
    x, c, track = problem
    c = c.at[5].set(0.0)
    robust = solve_robust_game(x, c, track, radius=10.0, horizon=4)
    nominal = solve_game(x, c, track, horizon=4)
    assert robust.converged
    assert_allclose(robust.us, nominal.us, atol=0, rtol=0)


def test_short_robust_loop_is_reproducible(problem):
    x, c, track = problem
    first = run_closed_loop(x, c, track, radius=5.0, steps=3, horizon=4)
    second = run_closed_loop(x, c, track, radius=5.0, steps=3, horizon=4)
    assert_allclose(first.us, second.us, atol=0, rtol=0)
    assert np.max(first.residuals) <= 1e-8


def test_invalid_radius_and_unsupported_blocking(problem):
    x, c, track = problem
    with pytest.raises(ValueError):
        solve_robust_game(x, c, track, radius=-1.0)
    result = solve_robust_game(x, c.at[11].set(0.5), track, radius=1.0, horizon=4)
    assert not result.converged
    assert np.isinf(result.residual)


def test_adversarial_study_checks_feasible_full_plan_before_execution(problem):
    x, c, track = problem
    record, states = simulate(x, c, track, radius=1.0, threat=1.0, steps=2, horizon=4)
    assert record["complete"], record["failure"]
    assert record["max_adversary_deviation"] <= 1 + 1e-6
    assert record["min_adversary_slack"] >= -1e-7
    assert states.shape == (3, 2, 9)
