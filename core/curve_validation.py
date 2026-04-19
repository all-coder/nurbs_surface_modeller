"""Validation and knot helpers for Bezier, B-spline, and NURBS curves."""

from __future__ import annotations

import numpy as np

from config import EPSILON
from core.bspline import generate_clamped_uniform_knot_vector
from core.validation import validate_degree, validate_knot_vector


def validate_curve_control_points(control_points: np.ndarray, min_points: int = 2) -> None:
    """Validate curve control point array dimensions and numeric integrity.

    Inputs:
        control_points: Curve control points expected with shape (n, 3).
        min_points: Minimum number of control points required.

    Outputs:
        None.

    Behavior:
        Raises ValueError when shape is invalid, point count is too small,
        or values contain NaN/inf.
    """
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError("curve control_points must have shape (n, 3)")
    if control_points.shape[0] < min_points:
        raise ValueError(f"curve requires at least {min_points} control points")
    if not np.all(np.isfinite(control_points)):
        raise ValueError("curve control_points must be finite numeric values")


def validate_curve_degree(degree: int, control_count: int) -> None:
    """Validate curve degree against number of control points.

    Inputs:
        degree: Candidate spline degree.
        control_count: Number of control points.

    Outputs:
        None.

    Behavior:
        Delegates to shared degree validator and raises ValueError if invalid.
    """
    validate_degree(int(degree), int(control_count), "curve")


def validate_curve_knot_vector(knots: np.ndarray, degree: int, control_count: int) -> None:
    """Validate knot vector for a curve.

    Inputs:
        knots: Knot vector values.
        degree: Curve degree.
        control_count: Number of curve control points.

    Outputs:
        None.

    Behavior:
        Delegates to shared knot validator and raises ValueError if invalid.
    """
    validate_knot_vector(np.asarray(knots, dtype=float), int(degree), int(control_count), "curve")


def validate_curve_weights(weights: np.ndarray, control_count: int) -> None:
    """Validate NURBS curve weights.

    Inputs:
        weights: Weight vector expected with shape (n,).
        control_count: Number of curve control points.

    Outputs:
        None.

    Behavior:
        Ensures one strictly positive finite weight per control point.
    """
    if weights.ndim != 1 or weights.shape[0] != control_count:
        raise ValueError("curve weights must have shape (n,) matching control points")
    if not np.all(np.isfinite(weights)):
        raise ValueError("curve weights must be finite numeric values")
    if np.any(weights <= 0.0):
        raise ValueError("all curve weights must be strictly positive")


def validate_knot_mode(mode: str) -> str:
    """Validate and normalize curve knot preset mode.

    Inputs:
        mode: Knot mode string.

    Outputs:
        Normalized lowercase mode string.

    Behavior:
        Supports "uniform" and "non-uniform" modes.
    """
    normalized = str(mode).strip().lower()
    if normalized not in {"uniform", "non-uniform"}:
        raise ValueError('knot mode must be either "uniform" or "non-uniform"')
    return normalized


def generate_nonuniform_knot_vector(control_points: np.ndarray, degree: int) -> np.ndarray:
    """Generate a clamped non-uniform knot vector using chord-length spacing.

    Inputs:
        control_points: Curve control points with shape (n, 3).
        degree: Curve degree.

    Outputs:
        Knot vector with length n + degree + 1.

    Behavior:
        Uses chord-length parameterization and averaging to produce interior
        knot spacing that follows control polygon segment lengths.
    """
    points = np.asarray(control_points, dtype=float)
    validate_curve_control_points(points, min_points=2)
    validate_curve_degree(int(degree), points.shape[0])

    control_count = points.shape[0]
    knot_count = control_count + degree + 1
    knots = np.zeros(knot_count, dtype=float)
    knots[control_count:] = 1.0

    interior_count = control_count - degree - 1
    if interior_count <= 0:
        return knots

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])

    if total < EPSILON:
        # Degenerate control polygons fallback to stable uniform behavior.
        return generate_clamped_uniform_knot_vector(control_count, degree)

    parameter_values = cumulative / total

    for j in range(1, interior_count + 1):
        start = j
        end = j + degree
        knots[degree + j] = float(np.sum(parameter_values[start:end]) / degree)

    return knots


def build_knots_for_mode(control_points: np.ndarray, degree: int, mode: str) -> np.ndarray:
    """Build a curve knot vector from a preset mode.

    Inputs:
        control_points: Curve control points with shape (n, 3).
        degree: Curve degree.
        mode: Knot mode, either "uniform" or "non-uniform".

    Outputs:
        Knot vector with shape (n + degree + 1,).

    Behavior:
        Routes to clamped uniform or chord-length non-uniform generation.
    """
    normalized_mode = validate_knot_mode(mode)
    count = int(np.asarray(control_points, dtype=float).shape[0])

    if normalized_mode == "uniform":
        return generate_clamped_uniform_knot_vector(count, degree)

    return generate_nonuniform_knot_vector(control_points, degree)
