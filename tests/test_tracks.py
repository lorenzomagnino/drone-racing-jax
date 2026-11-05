import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from drone_racing_jax.splines import evaluate, periodic_spline
from drone_racing_jax.tracks import make_track, position


def test_natural_cubic_and_julia_cached_derivatives():
    spline = periodic_spline([0, 1, 0], [0, 1, 2])
    assert_allclose(evaluate(spline, 0.5), 0.6875)
    assert_allclose(evaluate(spline, -1.5), 0.6875)
    assert_allclose(evaluate(spline, 2.5), 0.6875)
    assert_allclose(evaluate(spline, 0.5, 1), 0.75)
    assert_allclose(evaluate(spline, 0.5, 2), -1.5)
    assert_allclose(evaluate(spline, 0.5, 3), 0)
    first = jax.grad(lambda t: evaluate(spline, t))
    assert_allclose(jax.jit(first)(0.5), 0.75)
    assert_allclose(jax.grad(first)(0.5), -1.5)
    assert_allclose(jax.grad(jax.grad(first))(0.5), 0)
    # The source's cached derivative deliberately differs from the cubic's true derivative.
    finite_difference = (evaluate(spline, 0.50001) - evaluate(spline, 0.49999)) / 0.00002
    assert_allclose(finite_difference, 1.125, atol=1e-9)


@pytest.mark.parametrize(
    "times,values",
    [([1, 2, 3], [0, 1, 0]), ([0, 1, 1], [0, 1, 0]), ([0, 1, 2], [0, float("nan"), 0])],
)
def test_invalid_spline_data(times, values):
    with pytest.raises(ValueError):
        periodic_spline(values, times)


@pytest.mark.parametrize(
    "name,gate_count", [("lemniscate", 2), ("eight", 2), ("lissajous", 4), ("circle", 2)]
)
def test_track_setup_and_jit_vmap(name, gate_count):
    track = make_track(name)
    assert track.gates.shape == (gate_count, 3)
    assert track.path.times.shape == (31,)
    assert track.contour_weight.times.shape == (200,)
    assert float(track.length) > 10
    assert_allclose(evaluate(track.path, 0), position(0, name), atol=1e-12)
    assert_allclose(evaluate(track.path, track.length), position(0, name), atol=1e-12)
    ss = jnp.linspace(0, track.length, 50)
    values = jax.jit(jax.vmap(evaluate, in_axes=(None, 0)))(track.path, ss)
    assert values.shape == (50, 3)
    assert np.all(np.isfinite(values))
    for s in np.asarray(ss)[1:-1]:
        # Interior knots interpolate the analytic path at the bisection-selected parameters.
        assert np.all(np.isfinite(jax.jacfwd(lambda t: evaluate(track.path, t))(s)))


def test_unknown_track():
    with pytest.raises(ValueError, match="Unknown track"):
        position(0.0, "unknown")
