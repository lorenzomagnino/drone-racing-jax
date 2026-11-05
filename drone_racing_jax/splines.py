"""Julia PeriodicSpline values and cached derivatives, with fixed track data.

Construction runs on the host. Evaluation and time derivatives are pure JAX.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Spline(NamedTuple):
    times: jax.Array
    coefficients: jax.Array  # derivative order, interval, ascending power, value dimensions


def _polynomial(times, coefficients, t, derivative=0):
    index = np.clip(np.searchsorted(times, t, side="left") - 1, 0, len(times) - 2)
    delta = t - times[index]
    c = coefficients[index]
    if derivative:
        return c[1] + delta * (2 * c[2] + delta * 3 * c[3])
    return c[0] + delta * (c[1] + delta * (c[2] + delta * c[3]))


def _quadratic_basis(knots, t):
    basis = ((knots[:-1] <= t) & (t < knots[1:])).astype(float)
    for degree in (1, 2):
        size = len(knots) - degree - 1
        left = knots[degree : degree + size] - knots[:size]
        right = knots[degree + 1 : degree + size + 1] - knots[1 : size + 1]
        basis = (t - knots[:size]) / np.where(left == 0, 1, left) * basis[:size] + (
            knots[degree + 1 : degree + size + 1] - t
        ) / np.where(right == 0, 1, right) * basis[1 : size + 1]
    if t == knots[-1]:
        basis[-1] = 1
    return basis


def periodic_spline(values, times):
    """Natural cubic with endpoint replacement, quadratic/linear/constant derivative caches.

    This mirrors DataInterpolations 8.3.1, including its quadratic knot convention.
    Times must start at zero, as in the original's modulo implementation.
    """
    times, values = np.asarray(times, dtype=float), np.array(values, dtype=float, copy=True)
    if times.ndim != 1 or len(times) < 3 or times[0] != 0:
        raise ValueError("Need at least three 1D knots starting at zero")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("Knots must be finite and strictly increasing")
    if values.shape[0] != len(times) or not np.all(np.isfinite(values)):
        raise ValueError("Need a finite value for each knot")
    values[-1] = values[0]
    width = np.diff(times)
    h = width.reshape((-1,) + (1,) * (values.ndim - 1))
    slopes = np.diff(values, axis=0) / h
    matrix = np.eye(len(times))
    rows = np.arange(1, len(times) - 1)
    matrix[rows, rows - 1] = width[:-1]
    matrix[rows, rows] = 2 * (width[:-1] + width[1:])
    matrix[rows, rows + 1] = width[1:]
    rhs = np.zeros_like(values)
    rhs[1:-1] = 6 * np.diff(slopes, axis=0)
    second = np.linalg.solve(matrix, rhs)
    cubic = np.stack(
        (
            values[:-1],
            slopes - h * (2 * second[:-1] + second[1:]) / 6,
            second[:-1] / 2,
            np.diff(second, axis=0) / (6 * h),
        ),
        axis=1,
    )
    first_values = np.array([_polynomial(times, cubic, t, 1) for t in times])
    knots = np.concatenate((np.repeat(times[0], 3), times[1:-2], np.repeat(times[-1], 3)))
    basis = np.array([_quadratic_basis(knots, t) for t in times])
    control_points = np.linalg.solve(basis, first_values)
    midpoint_values = np.array(
        [_quadratic_basis(knots, t) @ control_points for t in (times[:-1] + times[1:]) / 2]
    )
    alpha = 2 * (first_values[1:] + first_values[:-1]) - 4 * midpoint_values
    beta = 4 * (midpoint_values - first_values[:-1]) - np.diff(first_values, axis=0)
    zero = np.zeros_like(alpha)
    quadratic = np.stack((first_values[:-1], beta / h, alpha / h**2, zero), axis=1)
    second_values = np.array([_polynomial(times, quadratic, t, 1) for t in times])
    linear_slopes = np.diff(second_values, axis=0) / h
    linear = np.stack((second_values[:-1], linear_slopes, zero, zero), axis=1)
    # DI's derivative of LinearInterpolation uses the interval to the left at a knot.
    third_values = np.concatenate((linear_slopes[:1], linear_slopes), axis=0)
    constant = np.stack((third_values[:-1], zero, zero, zero), axis=1)
    return Spline(jnp.asarray(times), jnp.asarray(np.stack((cubic, quadratic, linear, constant))))


def evaluate_polynomial(spline, t, order=0):
    """Evaluate a cache's actual polynomial, for linearizing numerical KKT equations."""
    if order > 3:
        return jnp.zeros_like(spline.coefficients[0, 0, 0])
    t = jnp.mod(t, spline.times[-1])
    index = jnp.clip(
        jnp.searchsorted(spline.times, t, side="right") - 1, 0, spline.times.shape[0] - 2
    )
    delta = t - spline.times[index]
    c = spline.coefficients[order, index]
    return c[0] + delta * (c[1] + delta * (c[2] + delta * c[3]))


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def evaluate(spline, t, order=0):
    """Evaluate at scalar t; vmap for arrays. Only t is differentiable, like Julia.

    Cached derivative interpolants intentionally define AD, rather than differentiating
    the cubic polynomial directly. Track coefficients and knots are fixed data.
    """
    return evaluate_polynomial(spline, t, order)


@evaluate.defjvp
def _evaluate_jvp(order, primals, tangents):
    spline, t = primals
    _, dt = tangents
    return evaluate(spline, t, order), evaluate(spline, t, order + 1) * dt
