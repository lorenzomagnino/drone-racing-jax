import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing_jax.controller import (
    augment_measurement,
    controller_step,
    estimate_progress,
    initialize_controller,
    run_closed_loop,
    shift_controls,
    update_attacker,
)
from drone_racing_jax.game import default_context, dynamics
from drone_racing_jax.splines import evaluate
from drone_racing_jax.tracks import make_track


@pytest.fixture(scope="module")
def problem():
    track = make_track()
    progress = jnp.array([track.length - 1, track.length - 2.5])
    positions = jax.vmap(lambda s: evaluate(track.path, s))(progress)
    x0 = jnp.zeros((2, 11)).at[:, :3].set(positions).at[:, 9].set(progress)
    return x0, default_context(), track


def test_progress_is_unwrapped_and_uses_local_track_projection(problem):
    _, _, track = problem
    target = track.length + 0.8
    position = evaluate(track.path, target)
    actual = jax.jit(estimate_progress)(position, target - 0.3, track)
    assert_allclose(actual, target, atol=5e-4)
    assert actual > track.length


def test_progress_speed_is_norm_of_velocity_projection(problem):
    x0, _, track = problem
    tangent = jax.vmap(lambda s: evaluate(track.tangent, s))(x0[:, 9])
    physical = x0[:, :9].at[:, 3:6].set(-2 * tangent)
    augmented = augment_measurement(physical, x0[:, 9], track)
    assert_allclose(augmented[:, :9], physical)
    assert_allclose(augmented[:, 10], 2 * jnp.linalg.norm(tangent, axis=-1), atol=1e-12)


@pytest.mark.parametrize(
    "difference,attacker,expected",
    [
        (0.75, 1, 1),
        (0.75001, 1, 2),
        (2.0, 2, 2),
        (-0.75, 2, 2),
        (-0.75001, 2, 1),
        (-2.0, 1, 1),
    ],
)
def test_role_switch_requires_overtake_threshold(difference, attacker, expected):
    assert update_attacker(jnp.array([difference, 0.0]), attacker) == expected


def test_shift_advances_every_player_and_appends_zero():
    us = jnp.arange(4 * 2 * 4, dtype=float).reshape(4, 2, 4)
    shifted = jax.jit(shift_controls)(us)
    assert_allclose(shifted[:-1], us[1:])
    assert_allclose(shifted[-1], 0)


def test_step_uses_measured_state_and_each_egos_first_action(problem):
    x0, context, track = problem
    state = initialize_controller(x0[:, 9], horizon=4)
    first = controller_step(x0[:, :9], state, context, track)
    assert np.all(first.plans.converged)
    assert_allclose(first.us, first.plans.us[:, 0, 0])
    assert_allclose(first.controller.guesses, jax.vmap(shift_controls)(first.plans.us))
    assert_allclose(first.plans.xs[1, 0], first.planning_state[::-1])
    physical = dynamics(first.planning_state, first.us)[:, :9].at[0, 1].add(0.01)
    second = controller_step(physical, first.controller, context, track)
    assert np.all(second.plans.converged)
    assert_allclose(second.planning_state[:, :9], physical)
    assert_allclose(second.plans.xs[:, 0, 0, :9], physical)
    assert not np.allclose(second.planning_state[0, :3], first.plans.xs[0, 1, 0, :3])


def test_short_closed_loop_is_reproducible_with_seed(problem):
    x0, context, track = problem
    positions = jax.random.uniform(jax.random.key(42), (2, 3), minval=-0.01, maxval=0.01)
    x0 = x0.at[:, :3].add(positions)
    first = run_closed_loop(x0, context, track, steps=5, horizon=4)
    second = run_closed_loop(x0, context, track, steps=5, horizon=4)
    assert first.states.shape == (6, 2, 9)
    assert_allclose(first.states, second.states, atol=0, rtol=0)
    assert_allclose(first.us, second.us, atol=0, rtol=0)
    assert np.max(first.residuals) < 1e-8
    for t in range(5):
        expected = dynamics(first.planning_states[t], first.us[t])[:, :9]
        assert_allclose(first.states[t + 1], expected, atol=1e-13)
        assert_allclose(first.planning_states[t, :, :9], first.states[t])


def test_failed_step_preserves_state_and_returns_no_action(problem):
    x0, context, track = problem
    state = initialize_controller(x0[:, 9], horizon=4)
    result = controller_step(x0[:, :9], state, context, track, max_iterations=0)
    assert not np.all(result.plans.converged)
    assert np.all(np.isnan(result.us))
    for old, new in zip(state, result.controller):
        assert_allclose(old, new)


def test_simulation_raises_before_applying_failed_plan(problem):
    x0, context, track = problem
    with pytest.raises(RuntimeError, match="failed at step 0"):
        run_closed_loop(x0.at[0, 0].set(jnp.nan), context, track, steps=2, horizon=4)


def test_invalid_inputs(problem):
    x0, context, track = problem
    with pytest.raises(ValueError):
        initialize_controller(x0[:, 9], attacker=3)
    with pytest.raises(ValueError):
        controller_step(x0, initialize_controller(x0[:, 9]), context, track)
    with pytest.raises(ValueError):
        run_closed_loop(x0, context, track, steps=0)
    with pytest.raises(ValueError):
        run_closed_loop(x0, context.at[4].set(1.5), track, steps=1)
