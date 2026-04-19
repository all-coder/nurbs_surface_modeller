"""Tensor-product NURBS surface evaluation and derivative computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EPSILON
from core.bspline import basis_first_derivatives, basis_functions, find_span
from core.validation import (
    validate_control_net_and_weights,
    validate_degree,
    validate_knot_vector,
    validate_parameter,
)
from utils.geometry import linspace_inclusive, normalize_vector


@dataclass(frozen=True)
class SurfaceEvaluation:
    """Container for first-order NURBS surface differential quantities.

    Inputs:
        point: Cartesian surface point B(u, v).
        du: First partial derivative with respect to u.
        dv: First partial derivative with respect to v.

    Outputs:
        Immutable object containing the three vectors above.

    Behavior:
        Serves as a typed result object for downstream consumers such as normal
        computation and toolpath generation.
    """

    point: np.ndarray
    du: np.ndarray
    dv: np.ndarray


class NURBSSurface:
    """Represent a tensor-product NURBS surface defined by a control net.

    Inputs:
        control_net: Cartesian control points with shape (m, n, 3).
        weights: Positive weight matrix with shape (m, n).
        knot_u: Knot vector for u axis.
        knot_v: Knot vector for v axis.
        degree_u: Degree along u.
        degree_v: Degree along v.

    Outputs:
        Instance exposing evaluation and sampling methods.

    Behavior:
        Validates all dimensions and spline settings during construction and then
        evaluates rational tensor-product surfaces in homogeneous coordinates.
    """

    def __init__(
        self,
        control_net: np.ndarray,
        weights: np.ndarray,
        knot_u: np.ndarray,
        knot_v: np.ndarray,
        degree_u: int,
        degree_v: int,
    ) -> None:
        """Initialize and validate a tensor-product NURBS surface.

        Inputs:
            control_net: Control net with shape (m, n, 3).
            weights: Weight matrix with shape (m, n).
            knot_u: Knot vector in u direction.
            knot_v: Knot vector in v direction.
            degree_u: Degree in u direction.
            degree_v: Degree in v direction.

        Outputs:
            None.

        Behavior:
            Stores validated arrays and precomputes homogeneous control points for
            efficient repeated evaluations.
        """
        self.control_net = np.asarray(control_net, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.knot_u = np.asarray(knot_u, dtype=float)
        self.knot_v = np.asarray(knot_v, dtype=float)
        self.degree_u = int(degree_u)
        self.degree_v = int(degree_v)

        validate_control_net_and_weights(self.control_net, self.weights)

        u_count, v_count = self.control_net.shape[:2]
        validate_degree(self.degree_u, u_count, "u")
        validate_degree(self.degree_v, v_count, "v")
        validate_knot_vector(self.knot_u, self.degree_u, u_count, "u")
        validate_knot_vector(self.knot_v, self.degree_v, v_count, "v")

        self._homogeneous = self._build_homogeneous_control_net()

    def parameter_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return valid parameter ranges for u and v.

        Inputs:
            None.

        Outputs:
            Tuple of two intervals: ((u_min, u_max), (v_min, v_max)).

        Behavior:
            Uses clamped knot-domain conventions based on degree and control count.
        """
        u_count, v_count = self.control_net.shape[:2]
        u_min = float(self.knot_u[self.degree_u])
        u_max = float(self.knot_u[u_count])
        v_min = float(self.knot_v[self.degree_v])
        v_max = float(self.knot_v[v_count])
        return (u_min, u_max), (v_min, v_max)

    def update_control_point_inplace(
        self,
        i_u: int,
        i_v: int,
        new_point: np.ndarray,
        new_weight: float,
    ) -> None:
        """Update one control point and weight without rebuilding the surface object.

        Inputs:
            i_u: Control-point row index in the u direction.
            i_v: Control-point column index in the v direction.
            new_point: New Cartesian position with shape (3,).
            new_weight: New strictly positive NURBS weight.

        Outputs:
            None.

        Behavior:
            Mutates the in-memory control net and weight matrix, then refreshes
            the corresponding homogeneous coordinate only for the changed index.
            Raises IndexError for invalid indices and ValueError for invalid
            point shape or non-positive weights.
        """
        u_count, v_count = self.control_net.shape[:2]
        if not (0 <= i_u < u_count and 0 <= i_v < v_count):
            raise IndexError("control point index out of bounds")

        point_array = np.asarray(new_point, dtype=float)
        if point_array.shape != (3,):
            raise ValueError("new_point must have shape (3,)")
        if new_weight <= 0.0:
            raise ValueError("new_weight must be strictly positive")

        self.control_net[i_u, i_v] = point_array
        self.weights[i_u, i_v] = float(new_weight)

        weighted_xyz = point_array * float(new_weight)
        self._homogeneous[i_u, i_v, :3] = weighted_xyz
        self._homogeneous[i_u, i_v, 3] = float(new_weight)

    def evaluate_point(self, u_value: float, v_value: float) -> np.ndarray:
        """Evaluate the NURBS surface point B(u, v).

        Inputs:
            u_value: Parameter in u domain.
            v_value: Parameter in v domain.

        Outputs:
            Cartesian point array with shape (3,).

        Behavior:
            Evaluates homogeneous weighted sums and performs rational division by
            the scalar homogeneous coordinate.
        """
        result = self.evaluate_derivatives(u_value, v_value)
        return result.point

    def evaluate_derivatives(self, u_value: float, v_value: float) -> SurfaceEvaluation:
        """Evaluate B(u, v), Bu(u, v), and Bv(u, v).

        Inputs:
            u_value: Parameter in u domain.
            v_value: Parameter in v domain.

        Outputs:
            SurfaceEvaluation containing point, du, and dv vectors.

        Behavior:
            Computes first-order rational derivatives using homogeneous partials
            and quotient-rule conversion to Cartesian space.
        """
        (u_min, u_max), (v_min, v_max) = self.parameter_ranges()
        validate_parameter(u_value, u_min, u_max, "u")
        validate_parameter(v_value, v_min, v_max, "v")

        u_count, v_count = self.control_net.shape[:2]
        span_u = find_span(u_count, self.degree_u, u_value, self.knot_u)
        span_v = find_span(v_count, self.degree_v, v_value, self.knot_v)

        basis_u = basis_functions(span_u, u_value, self.degree_u, self.knot_u)
        basis_v = basis_functions(span_v, v_value, self.degree_v, self.knot_v)
        deriv_u = basis_first_derivatives(span_u, u_value, self.degree_u, self.knot_u)
        deriv_v = basis_first_derivatives(span_v, v_value, self.degree_v, self.knot_v)

        surface_w, du_w, dv_w = self._evaluate_homogeneous_partials(
            span_u,
            span_v,
            basis_u,
            basis_v,
            deriv_u,
            deriv_v,
        )

        weight = float(surface_w[3])
        if abs(weight) < EPSILON:
            raise ValueError("homogeneous weight is near zero at the given parameter")

        point = surface_w[:3] / weight
        du = (du_w[:3] - point * du_w[3]) / weight
        dv = (dv_w[:3] - point * dv_w[3]) / weight

        return SurfaceEvaluation(point=point, du=du, dv=dv)

    def evaluate_normal(self, u_value: float, v_value: float) -> np.ndarray:
        """Evaluate the unit surface normal vector at B(u, v).

        Inputs:
            u_value: Parameter in u domain.
            v_value: Parameter in v domain.

        Outputs:
            Unit normal vector with shape (3,).

        Behavior:
            Computes the cross product Bu x Bv and normalizes it. When the cross
            product is numerically degenerate, a stable fallback axis is returned.
        """
        result = self.evaluate_derivatives(u_value, v_value)
        cross_value = np.cross(result.du, result.dv)
        return normalize_vector(cross_value, fallback=np.array([0.0, 0.0, 1.0]))

    def evaluate_grid(
        self,
        u_samples: int,
        v_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample surface points and normals on a regular parameter grid.

        Inputs:
            u_samples: Number of sample points in u direction.
            v_samples: Number of sample points in v direction.

        Outputs:
            Tuple containing points, normals, u_grid, and v_grid arrays.

        Behavior:
            Evaluates derivatives at each grid node to produce both geometry and
            normals for rendering and toolpath preview.
        """
        (u_min, u_max), (v_min, v_max) = self.parameter_ranges()
        u_values = linspace_inclusive(u_min, u_max, u_samples)
        v_values = linspace_inclusive(v_min, v_max, v_samples)

        points = np.zeros((u_samples, v_samples, 3), dtype=float)
        normals = np.zeros_like(points)

        for i_u, u_value in enumerate(u_values):
            for i_v, v_value in enumerate(v_values):
                eval_result = self.evaluate_derivatives(float(u_value), float(v_value))
                points[i_u, i_v] = eval_result.point
                normals[i_u, i_v] = normalize_vector(
                    np.cross(eval_result.du, eval_result.dv),
                    fallback=np.array([0.0, 0.0, 1.0]),
                )

        u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="ij")
        return points, normals, u_grid, v_grid

    def _build_homogeneous_control_net(self) -> np.ndarray:
        """Build homogeneous control points Pw = [xw, yw, zw, w].

        Inputs:
            None.

        Outputs:
            Homogeneous control net array with shape (m, n, 4).

        Behavior:
            Multiplies Cartesian coordinates by weights and stores the weight as
            the fourth component for rational derivative calculations.
        """
        weighted_xyz = self.control_net * self.weights[:, :, None]
        return np.dstack((weighted_xyz, self.weights))

    def _evaluate_homogeneous_partials(
        self,
        span_u: int,
        span_v: int,
        basis_u: np.ndarray,
        basis_v: np.ndarray,
        deriv_u: np.ndarray,
        deriv_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate homogeneous position and first partial derivatives.

        Inputs:
            span_u: Active span index in u direction.
            span_v: Active span index in v direction.
            basis_u: Non-zero basis values in u.
            basis_v: Non-zero basis values in v.
            deriv_u: Non-zero first derivatives in u.
            deriv_v: Non-zero first derivatives in v.

        Outputs:
            Tuple (S, Su, Sv) where each item has shape (4,).

        Behavior:
            Performs local tensor accumulation over the active knot spans only,
            improving efficiency compared with full-net summation.
        """
        surface_w = np.zeros(4, dtype=float)
        du_w = np.zeros(4, dtype=float)
        dv_w = np.zeros(4, dtype=float)

        for local_u in range(self.degree_u + 1):
            global_u = span_u - self.degree_u + local_u
            for local_v in range(self.degree_v + 1):
                global_v = span_v - self.degree_v + local_v

                control_h = self._homogeneous[global_u, global_v]
                coeff = basis_u[local_u] * basis_v[local_v]
                coeff_du = deriv_u[local_u] * basis_v[local_v]
                coeff_dv = basis_u[local_u] * deriv_v[local_v]

                surface_w += coeff * control_h
                du_w += coeff_du * control_h
                dv_w += coeff_dv * control_h

        return surface_w, du_w, dv_w
