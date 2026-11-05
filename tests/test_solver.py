import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing_jax.game import (
    default_context,
    equality_constraints,
    inequality_constraints,
    rollout,
    trajectory_cost,
)
from drone_racing_jax.solver import _game_gradient, kkt_residual, solve_game
from drone_racing_jax.splines import evaluate
from drone_racing_jax.tracks import make_track


@pytest.fixture(scope="module")
def problem():
    track = make_track()
    progress = jnp.array([4.2, 4.7])
    x0 = jnp.zeros((2, 11)).at[:, :3].set(jax.vmap(lambda s: evaluate(track.path, s))(progress))
    return x0.at[:, 9].set(progress), default_context(), track


@pytest.mark.parametrize("horizon", [4, 15])
def test_kkt_and_original_game_gradient(problem, horizon):
    x0, context, track = problem
    result = solve_game(x0, context, track, horizon=horizon)
    assert result.converged, (result.residual, result.iterations)
    assert float(result.residual) < 1e-8
    assert_allclose(equality_constraints(result.xs, result.us, x0), 0, atol=1e-13)
    slack = inequality_constraints(result.xs)
    assert np.min(slack) > -1e-8
    assert np.min(result.multipliers) > -1e-8
    assert np.max(np.abs(slack * result.multipliers)) < 1e-7
    assert_allclose(result.us[-1], 0, atol=1e-10)
    original_gradient = jax.jacrev(
        lambda u: trajectory_cost(rollout(x0, u[:-1]), u, context, track)
    )(result.us)
    expected = jnp.stack((original_gradient[0, :, 0], original_gradient[1, :, 1]), axis=1)
    assert_allclose(
        _game_gradient(result.us, x0, context, track, 0.05), expected.reshape(-1), atol=1e-12
    )


@pytest.mark.parametrize("active", [False, True])
def test_implicit_gradient_matches_finite_differences(problem, active):
    x0, context, track = problem
    if active:
        x0 = x0.at[:, 10].set(2.9)
    else:
        context = context.at[10].set(0.1)
    weights = jnp.asarray(np.random.default_rng(42).normal(size=(4, 2, 4)))

    def loss(context):
        result = solve_game(x0, context, track, horizon=4, tolerance=1e-10)
        return jnp.sum(result.us * weights) + 0.1 * result.multipliers.sum()

    result = solve_game(x0, context, track, horizon=4, tolerance=1e-10)
    assert result.converged
    assert bool(jnp.any(result.multipliers > 1e-5)) == active
    gradient = jax.jit(jax.grad(loss))(context)
    loss = jax.jit(loss)
    for index in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
        epsilon = 1e-5 * max(abs(float(context[index])), 1e-3)
        delta = jnp.zeros(12).at[index].set(epsilon)
        finite_difference = (loss(context + delta) - loss(context - delta)) / (2 * epsilon)
        assert_allclose(
            gradient[index],
            finite_difference,
            rtol=2e-5,
            atol=2e-5,
            err_msg=f"context[{index}], active={active}",
        )


def test_kkt_linearization_differentiates_actual_cached_polynomials(problem):
    x0, context, track = problem
    us = jnp.asarray(np.random.default_rng(7).normal(size=(4, 2, 4)))
    multipliers = jnp.zeros(3 * 28)
    residual = jax.jit(lambda u: kkt_residual(u, multipliers, x0, context, track))
    direction = jnp.asarray(np.random.default_rng(8).normal(size=us.shape))
    _, derivative = jax.jvp(residual, (us,), (direction,))
    epsilon = 1e-4
    finite_difference = (
        residual(us + epsilon * direction) - residual(us - epsilon * direction)
    ) / (2 * epsilon)
    assert_allclose(derivative, finite_difference, rtol=1e-6, atol=1e-9)


def test_initial_state_gradient_and_warm_start(problem):
    x0, context, track = problem
    context = context.at[10].set(0.1)
    result = solve_game(x0, context, track, horizon=4, tolerance=1e-10)
    warmed = solve_game(x0, context, track, horizon=4, initial_guess=result.us, tolerance=1e-10)
    assert warmed.converged
    assert int(warmed.iterations) <= int(result.iterations)
    assert_allclose(warmed.us, result.us, atol=1e-7)
    loss = jax.jit(lambda x: solve_game(x, context, track, horizon=4, tolerance=1e-10).us[0, 1, 0])
    derivative = jax.jit(jax.grad(loss))(x0)[0, 0]
    delta = jnp.zeros_like(x0).at[0, 0].set(1e-5)
    finite_difference = (loss(x0 + delta) - loss(x0 - delta)) / 2e-5
    assert_allclose(derivative, finite_difference, rtol=1e-5, atol=1e-6)


def test_failure_is_explicit_and_has_no_usable_gradient(problem):
    x0, context, track = problem
    result = solve_game(x0, context, track, horizon=4, max_iterations=0)
    assert not result.converged
    assert int(result.iterations) == 0
    gradient = jax.grad(lambda c: solve_game(x0, c, track, horizon=4, max_iterations=0).us.sum())(
        context
    )
    assert np.all(np.isnan(gradient))
    bad_state = x0.at[0, 0].set(jnp.nan)
    assert not solve_game(bad_state, context, track, horizon=4).converged


def test_seeded_batch_is_reproducible(problem):
    x0, context, track = problem
    perturbations = jax.random.normal(jax.random.key(42), (2, 2, 11)) * 0.001
    batch_solve = jax.jit(jax.vmap(lambda x: solve_game(x, context, track, horizon=3)))
    first, second = batch_solve(x0 + perturbations), batch_solve(x0 + perturbations)
    assert np.all(first.converged)
    assert_allclose(first.us, second.us, atol=0, rtol=0)


def test_globalization_handles_moving_state(problem):
    _, context, track = problem
    progress = jnp.array([19.98427417, 18.81716934])
    positions = jax.vmap(lambda s: evaluate(track.path, s))(progress)
    tangents = jax.vmap(lambda s: evaluate(track.tangent, s))(progress)
    speed = jnp.array([1.8, 2.5])
    x0 = jnp.zeros((2, 11)).at[:, :3].set(positions).at[:, 3:6].set(tangents * speed[:, None])
    x0 = x0.at[:, 9].set(progress).at[:, 10].set(speed)
    result = solve_game(x0, context, track)
    assert result.converged, (result.residual, result.iterations)


@pytest.mark.parametrize(
    "kwargs", [{"horizon": 1}, {"dt": 0}, {"tolerance": 0}, {"max_iterations": -1}]
)
def test_invalid_solver_configuration(problem, kwargs):
    with pytest.raises(ValueError):
        solve_game(*problem, **kwargs)
