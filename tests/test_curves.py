"""Unit tests for Bezier, B-spline, NURBS curves and loft generation."""

from __future__ import annotations

import unittest

import numpy as np

from core.curve_validation import build_knots_for_mode, generate_nonuniform_knot_vector
from core.curves import BSplineCurve, BezierCurve, NURBSCurve, convert_curve_to_nurbs
from gui.app import reduce_curve_selection_state, sanitize_curve_selection_state
from surface.providers.extrusion import extrude_surface_from_curve
from surface.providers.loft import loft_surface_from_curves
from surface.providers.surface_body_extrusion import build_surface_extrusion_mesh
from surface.providers.surface_normal_extrusion import extrude_surface_along_center_normal


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

    def test_surface_normal_extrusion_offsets_control_net(self) -> None:
        """Center-normal extrusion should offset the full control net consistently."""
        surface = loft_surface_from_curves(
            [
                BezierCurve(np.array([[0.0, 0.0, 0.0], [8.0, 3.0, 2.0], [16.0, 0.0, 0.0]], dtype=float)),
                BezierCurve(np.array([[0.0, 12.0, 2.0], [8.0, 14.0, 5.0], [16.0, 12.0, 2.0]], dtype=float)),
            ],
            samples_per_curve=12,
        )

        distance = 7.5
        normal = surface.evaluate_normal(0.5, 0.5)
        extruded = extrude_surface_along_center_normal(surface, distance=distance)
        expected_offset = np.broadcast_to(
            distance * normal[None, None, :],
            extruded.control_net.shape,
        )

        np.testing.assert_allclose(
            extruded.control_net - surface.control_net,
            expected_offset,
            atol=1e-7,
        )
        np.testing.assert_allclose(extruded.weights, surface.weights, atol=1e-12)

    def test_surface_normal_extrusion_rejects_non_positive_distance(self) -> None:
        """Center-normal extrusion should enforce positive distances."""
        surface = loft_surface_from_curves(
            [
                BezierCurve(np.array([[0.0, 0.0, 0.0], [6.0, 3.0, 0.0], [12.0, 0.0, 0.0]], dtype=float)),
                BezierCurve(np.array([[0.0, 8.0, 1.0], [6.0, 10.0, 2.0], [12.0, 8.0, 1.0]], dtype=float)),
            ],
            samples_per_curve=10,
        )
        with self.assertRaises(ValueError):
            extrude_surface_along_center_normal(surface, distance=0.0)

    def test_surface_body_extrusion_generates_closed_solid_mesh(self) -> None:
        """Solid mode should create finite, index-valid triangulated body mesh."""
        surface = loft_surface_from_curves(
            [
                BezierCurve(np.array([[0.0, 0.0, 0.0], [6.0, 4.0, 1.0], [12.0, 0.0, 0.0]], dtype=float)),
                BezierCurve(np.array([[0.0, 9.0, 2.0], [6.0, 12.0, 3.0], [12.0, 9.0, 2.0]], dtype=float)),
            ],
            samples_per_curve=10,
        )

        vertices, triangles = build_surface_extrusion_mesh(surface, distance=6.0, mode="solid", u_samples=16, v_samples=14)
        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(triangles.ndim, 2)
        self.assertEqual(triangles.shape[1], 3)
        self.assertTrue(np.all(np.isfinite(vertices)))
        self.assertGreater(vertices.shape[0], 0)
        self.assertGreater(triangles.shape[0], 0)
        self.assertGreaterEqual(int(np.min(triangles)), 0)
        self.assertLess(int(np.max(triangles)), vertices.shape[0])

    def test_surface_body_extrusion_generates_closed_shell_mesh(self) -> None:
        """Shell mode should create finite, index-valid triangulated hollow body mesh."""
        surface = loft_surface_from_curves(
            [
                BezierCurve(np.array([[0.0, 0.0, 0.0], [8.0, 3.0, 2.0], [16.0, 0.0, 0.0]], dtype=float)),
                BezierCurve(np.array([[0.0, 11.0, 1.0], [8.0, 14.0, 4.0], [16.0, 11.0, 1.0]], dtype=float)),
            ],
            samples_per_curve=12,
        )

        vertices, triangles = build_surface_extrusion_mesh(surface, distance=4.0, mode="shell", u_samples=12, v_samples=12)
        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(triangles.ndim, 2)
        self.assertEqual(triangles.shape[1], 3)
        self.assertTrue(np.all(np.isfinite(vertices)))
        self.assertGreater(vertices.shape[0], 0)
        self.assertGreater(triangles.shape[0], 0)
        self.assertGreaterEqual(int(np.min(triangles)), 0)
        self.assertLess(int(np.max(triangles)), vertices.shape[0])

    def test_surface_body_extrusion_rejects_invalid_mode(self) -> None:
        """Body extrusion should reject unknown mode values."""
        surface = loft_surface_from_curves(
            [
                BezierCurve(np.array([[0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [8.0, 0.0, 0.0]], dtype=float)),
                BezierCurve(np.array([[0.0, 6.0, 1.0], [4.0, 8.0, 2.0], [8.0, 6.0, 1.0]], dtype=float)),
            ],
            samples_per_curve=8,
        )
        with self.assertRaises(ValueError):
            build_surface_extrusion_mesh(surface, distance=3.0, mode="invalid", u_samples=10, v_samples=10)


class TestCurveSelectionReducer(unittest.TestCase):
    """Validate deterministic additive curve-selection state transitions."""

    def test_plain_click_selects_single_curve(self) -> None:
        """Plain click should replace selection with exactly one curve."""
        selected, active, last = reduce_curve_selection_state(
            valid_curve_ids=[10, 20, 30],
            selected_curve_ids={10, 30},
            active_curve_id=30,
            last_interacted_curve_id=30,
            clicked_curve_id=20,
            shift_pressed=False,
        )

        self.assertEqual(selected, {20})
        self.assertEqual(active, 20)
        self.assertEqual(last, 20)

    def test_shift_click_toggles_membership(self) -> None:
        """Shift+click should add then remove target curve deterministically."""
        first_selected, first_active, first_last = reduce_curve_selection_state(
            valid_curve_ids=[10, 20, 30],
            selected_curve_ids={10},
            active_curve_id=10,
            last_interacted_curve_id=10,
            clicked_curve_id=20,
            shift_pressed=True,
        )
        self.assertEqual(first_selected, {10, 20})
        self.assertEqual(first_active, 20)
        self.assertEqual(first_last, 20)

        second_selected, second_active, second_last = reduce_curve_selection_state(
            valid_curve_ids=[10, 20, 30],
            selected_curve_ids=first_selected,
            active_curve_id=first_active,
            last_interacted_curve_id=first_last,
            clicked_curve_id=20,
            shift_pressed=True,
        )
        self.assertEqual(second_selected, {10})
        self.assertEqual(second_active, 10)
        self.assertEqual(second_last, 20)

    def test_empty_click_clears_selection(self) -> None:
        """Clicking empty space should clear selected and active curves."""
        selected, active, last = reduce_curve_selection_state(
            valid_curve_ids=[1, 2, 3],
            selected_curve_ids={1, 2},
            active_curve_id=2,
            last_interacted_curve_id=2,
            clicked_curve_id=None,
            shift_pressed=True,
        )

        self.assertEqual(selected, set())
        self.assertIsNone(active)
        self.assertIsNone(last)

    def test_sanitize_removes_deleted_curve_ids(self) -> None:
        """Sanitizer should drop deleted ids and keep deterministic active curve."""
        selected, active, last = sanitize_curve_selection_state(
            valid_curve_ids=[5, 7],
            selected_curve_ids={5, 6, 7},
            active_curve_id=6,
            last_interacted_curve_id=6,
        )

        self.assertEqual(selected, {5, 7})
        self.assertEqual(active, 5)
        self.assertIsNone(last)


if __name__ == "__main__":
    unittest.main()
