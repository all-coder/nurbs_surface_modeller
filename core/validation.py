"""Validation helpers for NURBS dimensions, degrees, knot vectors, and parameters."""

from __future__ import annotations

import numpy as np

from config import EPSILON


def validate_degree(degree: int, control_count: int, axis_name: str) -> None:
    """Validate spline degree for a given axis.

    Inputs:
        degree: Requested polynomial degree.
        control_count: Number of control points in the axis.
        axis_name: Axis label used in error messages (for example 'u' or 'v').

    Outputs:
        None.

    Behavior:
        Raises ValueError when the degree is negative or not strictly lower than
        the number of control points.
    """
    if degree < 1:
        raise ValueError(f"degree_{axis_name} must be >= 1")
    if control_count <= degree:
        raise ValueError(
            f"degree_{axis_name} must be smaller than control points along {axis_name}"
        )


def validate_knot_vector(knots: np.ndarray, degree: int, control_count: int, axis_name: str) -> None:
    """Validate knot vector monotonicity and size constraints.

    Inputs:
        knots: Knot vector values.
        degree: Degree associated with the knot vector.
        control_count: Number of control points in the axis.
        axis_name: Axis label used in error messages.

    Outputs:
        None.

    Behavior:
        Checks vector length and non-decreasing order. Raises ValueError when
        constraints are not satisfied.
    """
    expected_length = control_count + degree + 1
    if len(knots) != expected_length:
        raise ValueError(
            f"knot vector for axis {axis_name} must have length {expected_length}"
        )

    diffs = np.diff(knots)
    if np.any(diffs < -EPSILON):
        raise ValueError(f"knot vector for axis {axis_name} must be non-decreasing")


def validate_control_net_and_weights(control_net: np.ndarray, weights: np.ndarray) -> None:
    """Validate control net and weights tensor dimensions.

    Inputs:
        control_net: Cartesian control net with shape (m, n, 3).
        weights: Weight matrix with shape (m, n).

    Outputs:
        None.

    Behavior:
        Ensures shape compatibility and strictly positive weights. Raises
        ValueError if constraints are violated.
    """
    if control_net.ndim != 3 or control_net.shape[2] != 3:
        raise ValueError("control_net must have shape (m, n, 3)")
    if weights.shape != control_net.shape[:2]:
        raise ValueError("weights shape must match control_net first two dimensions")
    if np.any(weights <= 0.0):
        raise ValueError("all NURBS weights must be strictly positive")


def validate_parameter(parameter: float, start_value: float, end_value: float, axis_name: str) -> None:
    """Validate that a parameter lies inside a closed domain interval.

    Inputs:
        parameter: Parameter value to validate.
        start_value: Lower bound of the valid domain.
        end_value: Upper bound of the valid domain.
        axis_name: Axis label used in error messages.

    Outputs:
        None.

    Behavior:
        Accepts small floating noise of EPSILON at both ends and raises
        ValueError for clear out-of-range values.
    """
    if parameter < start_value - EPSILON or parameter > end_value + EPSILON:
        raise ValueError(
            f"parameter {axis_name}={parameter} outside [{start_value}, {end_value}]"
        )
