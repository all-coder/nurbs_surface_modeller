"""B-spline basis evaluation utilities used for tensor-product NURBS surfaces."""

from __future__ import annotations

import numpy as np

from config import EPSILON


def generate_clamped_uniform_knot_vector(control_count: int, degree: int) -> np.ndarray:
    """Generate a clamped uniform knot vector.

    Inputs:
        control_count: Number of control points along one parametric axis.
        degree: Polynomial degree along that axis.

    Outputs:
        Knot vector with length control_count + degree + 1.

    Behavior:
        Generates degree+1 repeated values at both ends and uniformly spaced
        internal knots in [0, 1]. Raises ValueError for invalid dimensions.
    """
    if control_count <= degree:
        raise ValueError("control_count must be greater than degree")

    knot_count = control_count + degree + 1
    knots = np.zeros(knot_count, dtype=float)

    end_start = knot_count - degree - 1
    knots[end_start:] = 1.0

    interior_count = control_count - degree - 1
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2, endpoint=True)[1:-1]
        knots[degree + 1 : degree + 1 + interior_count] = interior

    return knots


def find_span(control_count: int, degree: int, parameter: float, knots: np.ndarray) -> int:
    """Locate the knot span index containing the parameter.

    Inputs:
        control_count: Number of control points in the current axis.
        degree: Polynomial degree in that axis.
        parameter: Parameter value to locate.
        knots: Knot vector array.

    Outputs:
        Span index k such that knots[k] <= parameter < knots[k+1], with boundary handling.

    Behavior:
        Performs binary search as described in Piegl and Tiller. Parameters at
        the upper domain endpoint map to the last valid span.
    """
    n = control_count - 1
    if parameter >= knots[n + 1] - EPSILON:
        return n
    if parameter <= knots[degree] + EPSILON:
        return degree

    low = degree
    high = n + 1
    mid = (low + high) // 2

    while parameter < knots[mid] or parameter >= knots[mid + 1]:
        if parameter < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_functions(span: int, parameter: float, degree: int, knots: np.ndarray) -> np.ndarray:
    """Evaluate non-zero B-spline basis functions at a parameter value.

    Inputs:
        span: Span index containing the parameter.
        parameter: Parameter value.
        degree: Degree of basis functions.
        knots: Knot vector used for evaluation.

    Outputs:
        Array of shape (degree + 1,) with basis values in local span order.

    Behavior:
        Uses the Cox-de Boor dynamic programming approach for stable evaluation.
        The i-th return value corresponds to global basis index span-degree+i.
    """
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)
    values = np.zeros(degree + 1, dtype=float)
    values[0] = 1.0

    for j in range(1, degree + 1):
        left[j] = parameter - knots[span + 1 - j]
        right[j] = knots[span + j] - parameter
        saved = 0.0

        for r in range(j):
            denom = right[r + 1] + left[j - r]
            term = 0.0 if abs(denom) < EPSILON else values[r] / denom
            values[r] = saved + right[r + 1] * term
            saved = left[j - r] * term
        values[j] = saved

    return values


def basis_first_derivatives(span: int, parameter: float, degree: int, knots: np.ndarray) -> np.ndarray:
    """Evaluate first derivatives of non-zero B-spline basis functions.

    Inputs:
        span: Span index containing the parameter.
        parameter: Parameter value.
        degree: Degree of basis functions.
        knots: Knot vector used for evaluation.

    Outputs:
        Array of shape (degree + 1,) containing first derivatives in local span order.

    Behavior:
        Computes derivative values from the degree-1 basis functions using the
        closed-form derivative relation. Degree zero returns zeros.
    """
    derivatives = np.zeros(degree + 1, dtype=float)
    if degree == 0:
        return derivatives

    lower_basis = basis_functions(span, parameter, degree - 1, knots)

    for local_index in range(degree + 1):
        global_index = span - degree + local_index

        left_denom = knots[global_index + degree] - knots[global_index]
        right_denom = knots[global_index + degree + 1] - knots[global_index + 1]

        left_term = 0.0
        right_term = 0.0

        if local_index > 0 and abs(left_denom) >= EPSILON:
            left_term = degree * lower_basis[local_index - 1] / left_denom

        if local_index < degree and abs(right_denom) >= EPSILON:
            right_term = degree * lower_basis[local_index] / right_denom

        derivatives[local_index] = left_term - right_term

    return derivatives
