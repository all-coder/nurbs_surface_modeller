"""Curve entities and evaluators for Bezier, B-spline, and NURBS workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EPSILON
from core.bspline import basis_first_derivatives, basis_functions, find_span
from core.curve_validation import (
    build_knots_for_mode,
    validate_curve_control_points,
    validate_curve_degree,
    validate_curve_knot_vector,
    validate_curve_weights,
)
from core.validation import validate_parameter
from utils.geometry import linspace_inclusive, normalize_vector


@dataclass(frozen=True)
class CurveEvaluation:
    """Container for first-order curve differential quantities.

    Inputs:
        point: Cartesian curve point C(t).
        tangent: First derivative vector C'(t).

    Outputs:
        Immutable evaluation result object.

    Behavior:
        Bundles point and tangent for rendering and interaction workflows.
    """

    point: np.ndarray
    tangent: np.ndarray


class BezierCurve:
    """Represent a non-rational Bezier curve in 3D."""

    def __init__(self, control_points: np.ndarray) -> None:
        points = np.asarray(control_points, dtype=float)
        validate_curve_control_points(points, min_points=2)
        self.control_points = points.copy()
        self.degree = self.control_points.shape[0] - 1

    def parameter_range(self) -> tuple[float, float]:
        """Return parameter domain for Bezier curve."""
        return (0.0, 1.0)

    def evaluate_point(self, t_value: float) -> np.ndarray:
        """Evaluate Bezier point C(t) using de Casteljau recursion."""
        validate_parameter(float(t_value), 0.0, 1.0, "t")

        t_float = float(t_value)
        work = self.control_points.copy()

        for level in range(1, self.degree + 1):
            upper = self.degree - level + 1
            work[:upper] = (1.0 - t_float) * work[:upper] + t_float * work[1 : upper + 1]

        return work[0]

    def evaluate_derivatives(self, t_value: float) -> CurveEvaluation:
        """Evaluate C(t) and C'(t) for Bezier curve."""
        point = self.evaluate_point(t_value)

        if self.degree == 0:
            return CurveEvaluation(point=point, tangent=np.zeros(3, dtype=float))

        derivative_control_points = self.degree * np.diff(self.control_points, axis=0)
        derivative_curve = BezierCurve(derivative_control_points)
        tangent = derivative_curve.evaluate_point(t_value)

        return CurveEvaluation(point=point, tangent=tangent)

    def evaluate_tangent(self, t_value: float) -> np.ndarray:
        """Evaluate unit tangent vector at C(t)."""
        evaluation = self.evaluate_derivatives(t_value)
        return normalize_vector(evaluation.tangent, fallback=np.array([1.0, 0.0, 0.0]))

    def sample_points(self, sample_count: int) -> np.ndarray:
        """Sample Bezier points over parameter domain."""
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")

        t_values = linspace_inclusive(0.0, 1.0, sample_count)
        return np.array([self.evaluate_point(float(t_value)) for t_value in t_values], dtype=float)

    def update_control_point_inplace(self, index: int, new_point: np.ndarray) -> None:
        """Update one Bezier control point in-place."""
        if not (0 <= index < self.control_points.shape[0]):
            raise IndexError("control point index out of bounds")

        point = np.asarray(new_point, dtype=float)
        if point.shape != (3,):
            raise ValueError("new_point must have shape (3,)")

        self.control_points[index] = point


class BSplineCurve:
    """Represent a non-rational B-spline curve in 3D."""

    def __init__(
        self,
        control_points: np.ndarray,
        degree: int,
        knots: np.ndarray,
    ) -> None:
        points = np.asarray(control_points, dtype=float)
        validate_curve_control_points(points, min_points=2)
        validate_curve_degree(int(degree), points.shape[0])

        knot_values = np.asarray(knots, dtype=float)
        validate_curve_knot_vector(knot_values, int(degree), points.shape[0])

        self.control_points = points.copy()
        self.degree = int(degree)
        self.knots = knot_values.copy()

    def parameter_range(self) -> tuple[float, float]:
        """Return valid parameter range [t_min, t_max]."""
        count = self.control_points.shape[0]
        return float(self.knots[self.degree]), float(self.knots[count])

    def evaluate_derivatives(self, t_value: float) -> CurveEvaluation:
        """Evaluate C(t) and C'(t) for B-spline curve."""
        t_min, t_max = self.parameter_range()
        validate_parameter(float(t_value), t_min, t_max, "t")

        count = self.control_points.shape[0]
        span = find_span(count, self.degree, float(t_value), self.knots)
        basis = basis_functions(span, float(t_value), self.degree, self.knots)
        deriv = basis_first_derivatives(span, float(t_value), self.degree, self.knots)

        point = np.zeros(3, dtype=float)
        tangent = np.zeros(3, dtype=float)

        for local_index in range(self.degree + 1):
            global_index = span - self.degree + local_index
            control_point = self.control_points[global_index]
            point += basis[local_index] * control_point
            tangent += deriv[local_index] * control_point

        return CurveEvaluation(point=point, tangent=tangent)

    def evaluate_point(self, t_value: float) -> np.ndarray:
        """Evaluate B-spline point C(t)."""
        return self.evaluate_derivatives(t_value).point

    def evaluate_tangent(self, t_value: float) -> np.ndarray:
        """Evaluate unit tangent vector at C(t)."""
        evaluation = self.evaluate_derivatives(t_value)
        return normalize_vector(evaluation.tangent, fallback=np.array([1.0, 0.0, 0.0]))

    def sample_points(self, sample_count: int) -> np.ndarray:
        """Sample B-spline points over parameter domain."""
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")

        t_min, t_max = self.parameter_range()
        t_values = linspace_inclusive(t_min, t_max, sample_count)
        return np.array([self.evaluate_point(float(t_value)) for t_value in t_values], dtype=float)

    def update_control_point_inplace(self, index: int, new_point: np.ndarray) -> None:
        """Update one B-spline control point in-place."""
        if not (0 <= index < self.control_points.shape[0]):
            raise IndexError("control point index out of bounds")

        point = np.asarray(new_point, dtype=float)
        if point.shape != (3,):
            raise ValueError("new_point must have shape (3,)")

        self.control_points[index] = point

    def reset_knots_from_mode(self, mode: str) -> None:
        """Regenerate knot vector from a preset mode."""
        self.knots = build_knots_for_mode(self.control_points, self.degree, mode)


class NURBSCurve:
    """Represent a rational NURBS curve in 3D."""

    def __init__(
        self,
        control_points: np.ndarray,
        degree: int,
        knots: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        points = np.asarray(control_points, dtype=float)
        validate_curve_control_points(points, min_points=2)
        validate_curve_degree(int(degree), points.shape[0])

        knot_values = np.asarray(knots, dtype=float)
        validate_curve_knot_vector(knot_values, int(degree), points.shape[0])

        weight_values = np.asarray(weights, dtype=float)
        validate_curve_weights(weight_values, points.shape[0])

        self.control_points = points.copy()
        self.degree = int(degree)
        self.knots = knot_values.copy()
        self.weights = weight_values.copy()

        self._homogeneous = self._build_homogeneous_control_points()

    def _build_homogeneous_control_points(self) -> np.ndarray:
        """Build homogeneous control points [xw, yw, zw, w]."""
        weighted_xyz = self.control_points * self.weights[:, None]
        return np.column_stack((weighted_xyz, self.weights))

    def parameter_range(self) -> tuple[float, float]:
        """Return valid parameter range [t_min, t_max]."""
        count = self.control_points.shape[0]
        return float(self.knots[self.degree]), float(self.knots[count])

    def evaluate_derivatives(self, t_value: float) -> CurveEvaluation:
        """Evaluate C(t) and C'(t) for NURBS curve."""
        t_min, t_max = self.parameter_range()
        validate_parameter(float(t_value), t_min, t_max, "t")

        count = self.control_points.shape[0]
        span = find_span(count, self.degree, float(t_value), self.knots)
        basis = basis_functions(span, float(t_value), self.degree, self.knots)
        deriv = basis_first_derivatives(span, float(t_value), self.degree, self.knots)

        curve_w = np.zeros(4, dtype=float)
        tangent_w = np.zeros(4, dtype=float)

        for local_index in range(self.degree + 1):
            global_index = span - self.degree + local_index
            control_h = self._homogeneous[global_index]
            curve_w += basis[local_index] * control_h
            tangent_w += deriv[local_index] * control_h

        weight = float(curve_w[3])
        if abs(weight) < EPSILON:
            raise ValueError("homogeneous weight is near zero at the given parameter")

        point = curve_w[:3] / weight
        tangent = (tangent_w[:3] - point * tangent_w[3]) / weight
        return CurveEvaluation(point=point, tangent=tangent)

    def evaluate_point(self, t_value: float) -> np.ndarray:
        """Evaluate NURBS point C(t)."""
        return self.evaluate_derivatives(t_value).point

    def evaluate_tangent(self, t_value: float) -> np.ndarray:
        """Evaluate unit tangent vector at C(t)."""
        evaluation = self.evaluate_derivatives(t_value)
        return normalize_vector(evaluation.tangent, fallback=np.array([1.0, 0.0, 0.0]))

    def sample_points(self, sample_count: int) -> np.ndarray:
        """Sample NURBS points over parameter domain."""
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")

        t_min, t_max = self.parameter_range()
        t_values = linspace_inclusive(t_min, t_max, sample_count)
        return np.array([self.evaluate_point(float(t_value)) for t_value in t_values], dtype=float)

    def update_control_point_inplace(
        self,
        index: int,
        new_point: np.ndarray,
        new_weight: float | None = None,
    ) -> None:
        """Update one NURBS control point and optional weight in-place."""
        if not (0 <= index < self.control_points.shape[0]):
            raise IndexError("control point index out of bounds")

        point = np.asarray(new_point, dtype=float)
        if point.shape != (3,):
            raise ValueError("new_point must have shape (3,)")

        self.control_points[index] = point

        if new_weight is not None:
            if new_weight <= 0.0:
                raise ValueError("new_weight must be strictly positive")
            self.weights[index] = float(new_weight)

        self._homogeneous[index, :3] = self.control_points[index] * self.weights[index]
        self._homogeneous[index, 3] = self.weights[index]

    def reset_knots_from_mode(self, mode: str) -> None:
        """Regenerate knot vector from a preset mode."""
        self.knots = build_knots_for_mode(self.control_points, self.degree, mode)


CurveObject = BezierCurve | BSplineCurve | NURBSCurve


def convert_curve_to_nurbs(curve: CurveObject) -> NURBSCurve:
    """Convert any supported curve object to canonical NURBS representation."""
    if isinstance(curve, NURBSCurve):
        return NURBSCurve(
            control_points=curve.control_points.copy(),
            degree=curve.degree,
            knots=curve.knots.copy(),
            weights=curve.weights.copy(),
        )

    if isinstance(curve, BSplineCurve):
        weights = np.ones(curve.control_points.shape[0], dtype=float)
        return NURBSCurve(
            control_points=curve.control_points.copy(),
            degree=curve.degree,
            knots=curve.knots.copy(),
            weights=weights,
        )

    if isinstance(curve, BezierCurve):
        control_count = curve.control_points.shape[0]
        degree = curve.degree
        knots = np.concatenate(
            (
                np.zeros(degree + 1, dtype=float),
                np.ones(control_count, dtype=float),
            )
        )
        weights = np.ones(control_count, dtype=float)
        return NURBSCurve(
            control_points=curve.control_points.copy(),
            degree=degree,
            knots=knots,
            weights=weights,
        )

    raise TypeError("unsupported curve type for conversion to NURBS")
