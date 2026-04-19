"""Unit tests for Bezier, B-spline, NURBS curves and loft generation."""

from __future__ import annotations

import unittest

import numpy as np

from core.curve_validation import build_knots_for_mode, generate_nonuniform_knot_vector
from core.curves import BSplineCurve, BezierCurve, NURBSCurve, convert_curve_to_nurbs
from surface.providers.extrusion import extrude_surface_from_curve
from surface.providers.loft import loft_surface_from_curves


class TestCurveValidationAndEvaluation(unittest.TestCase):
    """Validate curve construction and evaluation behavior."""

    def test_bezier_interpolates_endpoints(self) -> None:
        """Bezier curve should pass through first and last control points."""
        control_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 15.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        curve = BezierCurve(control_points)

        np.testing.assert_allclose(curve.evaluate_point(0.0), control_points[0], atol=1e-8)
        np.testing.assert_allclose(curve.evaluate_point(1.0), control_points[-1], atol=1e-8)

    def test_bspline_nonuniform_knots_are_monotonic(self) -> None:
        """Generated non-uniform knot vectors should remain non-decreasing."""
        control_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 2.0, 0.0],
                [12.0, 6.0, 0.0],
                [30.0, 9.0, 0.0],
                [50.0, 10.0, 0.0],
            ],
            dtype=float,
        )
        degree = 3
        knots = generate_nonuniform_knot_vector(control_points, degree)

        self.assertEqual(len(knots), control_points.shape[0] + degree + 1)
        self.assertTrue(np.all(np.diff(knots) >= -1e-12))

    def test_nurbs_weight_changes_geometry(self) -> None:
        """Adjusting NURBS weights should alter evaluated points."""
        control_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 20.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 20.0, 0.0],
                [40.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        degree = 3
        knots = build_knots_for_mode(control_points, degree, "uniform")

        baseline = NURBSCurve(control_points, degree, knots, np.ones(control_points.shape[0], dtype=float))
        weighted = NURBSCurve(
            control_points,
            degree,
            knots,
            np.array([1.0, 1.0, 6.0, 1.0, 1.0], dtype=float),
        )

        p_baseline = baseline.evaluate_point(0.5)
        p_weighted = weighted.evaluate_point(0.5)

        self.assertGreater(float(np.linalg.norm(p_weighted - p_baseline)), 1e-6)


class TestCurveLoftGeneration(unittest.TestCase):
    """Validate surface loft generation from mixed curve inputs."""

    def test_loft_surface_from_mixed_curve_types(self) -> None:
        """Loft provider should accept mixed curve classes via NURBS conversion."""
        bezier = BezierCurve(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 5.0, 5.0],
                    [20.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        )

        bspline_points = np.array(
            [
                [0.0, 10.0, 3.0],
                [8.0, 12.0, 8.0],
                [16.0, 8.0, 6.0],
                [24.0, 10.0, 2.0],
            ],
            dtype=float,
        )
        bspline_degree = 3
        bspline_knots = build_knots_for_mode(bspline_points, bspline_degree, "non-uniform")
        bspline = BSplineCurve(bspline_points, bspline_degree, bspline_knots)

        nurbs_points = np.array(
            [
                [0.0, 20.0, 0.0],
                [8.0, 22.0, 4.0],
                [16.0, 18.0, 8.0],
                [24.0, 20.0, 0.0],
            ],
            dtype=float,
        )
        nurbs_degree = 3
        nurbs_knots = build_knots_for_mode(nurbs_points, nurbs_degree, "uniform")
        nurbs_weights = np.array([1.0, 1.8, 1.8, 1.0], dtype=float)
        nurbs = NURBSCurve(nurbs_points, nurbs_degree, nurbs_knots, nurbs_weights)

        surface = loft_surface_from_curves([bezier, bspline, nurbs], samples_per_curve=30)
        sample = surface.evaluate_point(0.5, 0.5)

        self.assertEqual(sample.shape, (3,))
        self.assertTrue(np.all(np.isfinite(sample)))

    def test_convert_curve_to_nurbs_returns_compatible_type(self) -> None:
        """All supported curves should convert to NURBSCurve instances."""
        bezier = BezierCurve(np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=float))
        converted = convert_curve_to_nurbs(bezier)
        self.assertIsInstance(converted, NURBSCurve)

    def test_extrude_surface_from_curve_produces_expected_shape(self) -> None:
        """Extrusion provider should build valid control-net dimensions."""
        curve = BezierCurve(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [8.0, 6.0, 0.0],
                    [16.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        )

        surface = extrude_surface_from_curve(
            curve,
            direction=np.array([0.0, 0.0, 1.0], dtype=float),
            height=24.0,
            layer_count=6,
            samples_along_curve=20,
        )

        self.assertEqual(surface.control_net.shape, (6, 20, 3))
        np.testing.assert_allclose(surface.control_net[-1, :, 2] - surface.control_net[0, :, 2], 24.0, atol=1e-8)

    def test_extrude_surface_rejects_invalid_direction(self) -> None:
        """Extrusion provider should reject zero-magnitude direction vectors."""
        curve = BezierCurve(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float))
        with self.assertRaises(ValueError):
            extrude_surface_from_curve(
                curve,
                direction=np.array([0.0, 0.0, 0.0], dtype=float),
                height=10.0,
                layer_count=3,
                samples_along_curve=10,
            )


if __name__ == "__main__":
    unittest.main()
