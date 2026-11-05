"""Default track geometry and arc-length setup from trajectories.jl and server_utils.jl."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drone_racing_jax.splines import Spline, evaluate, periodic_spline


class Track(NamedTuple):
    path: Spline
    tangent: Spline
    contour_weight: Spline
    length: jax.Array  # original analytic path length, not the fitted spline's length
    gates: jax.Array


def position(t, name="lemniscate"):
    """Analytic default geometry, before the Julia arc-length reparameterization."""
    if name == "lemniscate":
        return jnp.array([3.635 * jnp.sin(t) + 0.78, 4 * jnp.sin(t) * jnp.cos(t), 1.9])
    if name == "eight":
        return jnp.array([3 * jnp.cos(t) + 1, 3.5 * jnp.sin(t) * jnp.cos(t), jnp.sin(t) + 2.5])
    if name == "lissajous":
        return jnp.array([3 * jnp.sin(3 * t) + 1, 3.5 * jnp.sin(t), 3 * jnp.sin(2 * t) + 5])
    if name == "circle":
        angle = jnp.deg2rad(15.0)
        return jnp.array(
            [
                3 * jnp.sin(t) * jnp.cos(angle) + 1,
                3.5 * jnp.cos(t),
                -3 * jnp.sin(t) * jnp.sin(angle) + 2,
            ]
        )
    raise ValueError(f"Unknown track: {name}")


def make_track(name="lemniscate"):
    """Host setup: 30 arc-length segments, 200 contour-weight knots, covariance 0.1 I.

    Uses composite Gauss-Legendre quadrature instead of Julia QuadGK; bisection retains
    the original 1e-5 stopping tolerance. Enable JAX float64 before calling.
    """
    path_position = jax.jit(lambda t: position(t, name))
    path_velocity = jax.jit(jax.vmap(jax.jacfwd(path_position)))
    nodes, weights = np.polynomial.legendre.leggauss(32)

    def arc_length(end):
        edges = np.linspace(0, end, 33)
        mid, half = (edges[:-1] + edges[1:]) / 2, np.diff(edges) / 2
        ts = (mid[:, None] + half[:, None] * nodes).reshape(-1)
        speed = np.linalg.norm(np.asarray(path_velocity(jnp.asarray(ts))), axis=-1).reshape(32, 32)
        return np.sum(speed * weights * half[:, None])

    length = arc_length(2 * np.pi)
    ts = [0.0]
    for target in np.linspace(0, length, 31)[1:]:
        left, right = 0.0, 2 * np.pi
        while right - left > 1e-5:
            middle = (left + right) / 2
            if arc_length(middle) < target:
                left = middle
            else:
                right = middle
        ts.append((left + right) / 2)
    ss = np.linspace(0, length, 31)
    path = periodic_spline(jax.vmap(path_position)(jnp.asarray(ts)), ss)
    derivative = jax.vmap(lambda s: evaluate(path, s, 1))(jnp.asarray(ss))
    tangent = derivative / jnp.sqrt(jnp.sum(derivative**2, axis=-1, keepdims=True) + 1e-5)
    tangent = periodic_spline(tangent, ss)
    fractions = {
        "lemniscate": [0.25, 0.75],
        "circle": [0.25, 0.75],
        "eight": [0.0, 0.5],
        "lissajous": [0.2, 0.4, 0.6, 0.8],
    }[name]
    gates = jax.vmap(lambda s: evaluate(path, s))(jnp.asarray(fractions) * length)
    contour_times = np.linspace(0, length, 200)
    points = jax.vmap(lambda s: evaluate(path, s))(jnp.asarray(contour_times))
    difference = points[:, None, :] - gates[None, :, :]
    contour_values = jnp.exp(-5 * jnp.sum(difference**2, axis=-1)).sum(axis=-1)
    return Track(
        path, tangent, periodic_spline(contour_values, contour_times), jnp.asarray(length), gates
    )
