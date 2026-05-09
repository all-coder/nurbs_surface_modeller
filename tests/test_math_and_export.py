"""Unit tests for core NURBS math and G-code export pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from core.bspline import (
    basis_first_derivatives,
    basis_functions,
    find_span,
    generate_clamped_uniform_knot_vector,
)
from gcode.exporter import GCodeSettings, generate_gcode_program
from machining.toolpath import generate_zigzag_toolpath
from surface.factory import create_default_surface, create_surface_from_arrays


class TestBSplineBasis(unittest.TestCase):
    """Validate local B-spline basis behavior for consistency and correctness."""

    def test_partition_of_unity(self) -> None:
        """Check that non-zero basis functions sum to one for interior parameters."""
        control_count = 6
        degree = 3
        knots = generate_clamped_uniform_knot_vector(control_count, degree)
        u_value = 0.43
        span = find_span(control_count, degree, u_value, knots)
        values = basis_functions(span, u_value, degree, knots)
        self.assertAlmostEqual(float(np.sum(values)), 1.0, places=9)

    def test_derivative_sum_zero(self) -> None:
        """Check that first derivatives of basis functions sum to zero."""
        control_count = 6
        degree = 3
        knots = generate_clamped_uniform_knot_vector(control_count, degree)
        u_value = 0.51
        span = find_span(control_count, degree, u_value, knots)
        derivs = basis_first_derivatives(span, u_value, degree, knots)
        self.assertAlmostEqual(float(np.sum(derivs)), 0.0, places=8)


class TestSurfaceAndToolpath(unittest.TestCase):
    """Validate surface normal computations, toolpath generation, and G-code output."""

    def test_inplace_control_point_update_changes_surface_point(self) -> None:
        """Ensure in-place control-point updates propagate to surface evaluation."""
        surface = create_default_surface(rows=4, cols=4)
        baseline = surface.evaluate_point(0.5, 0.5)

        old_point = surface.control_net[1, 1].copy()
        moved_point = old_point + np.array([0.0, 0.0, 12.0])
        old_weight = float(surface.weights[1, 1])

        surface.update_control_point_inplace(1, 1, moved_point, old_weight)
        updated = surface.evaluate_point(0.5, 0.5)

        self.assertGreater(float(np.linalg.norm(updated - baseline)), 1e-6)

    def test_planar_surface_normal(self) -> None:
        """Ensure planar control nets produce near-vertical unit normals."""
        control_net = np.zeros((4, 4, 3), dtype=float)
        x_values = np.linspace(0.0, 30.0, 4)
        y_values = np.linspace(0.0, 30.0, 4)

        for i_u, x_value in enumerate(x_values):
            for i_v, y_value in enumerate(y_values):
                control_net[i_u, i_v] = np.array([x_value, y_value, 0.0])

        weights = np.ones((4, 4), dtype=float)
        surface = create_surface_from_arrays(control_net, weights, degree_u=3, degree_v=3)
        normal = surface.evaluate_normal(0.5, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=8)
        self.assertGreater(abs(float(normal[2])), 0.9)

    def test_toolpath_and_gcode_contains_required_commands(self) -> None:
        """Check generated G-code for mandatory startup and termination commands."""
        surface = create_default_surface(rows=4, cols=4)
        passes = generate_zigzag_toolpath(surface, stepover_mm=1.0, tool_radius_mm=2.0)
        self.assertGreater(len(passes), 1)

        gcode_text = generate_gcode_program(passes, GCodeSettings())
        for token in ("G21", "G90", "M3", "M8", "G0", "G1", "M30"):
            self.assertIn(token, gcode_text)

    def test_toolpath_is_3d_and_offsets_by_radius(self) -> None:
        """Ensure CL points vary in Z and keep the tool radius offset."""
        tool_radius = 2.0
        surface = create_default_surface(rows=6, cols=6)
        passes = generate_zigzag_toolpath(surface, stepover_mm=1.0, tool_radius_mm=tool_radius)
        self.assertGreater(len(passes), 1)

        points = [item for tool_pass in passes for item in tool_pass.points]
        self.assertGreater(len(points), 10)

        z_values = [float(item.cl_point[2]) for item in points]
        self.assertGreater(max(z_values) - min(z_values), 0.1)

        for item in points[: min(30, len(points))]:
            self.assertAlmostEqual(float(np.linalg.norm(item.normal)), 1.0, places=6)
            self.assertAlmostEqual(
                float(np.linalg.norm(item.cl_point - item.contact_point)),
                tool_radius,
                places=5,
            )

        repeat = generate_zigzag_toolpath(surface, stepover_mm=1.0, tool_radius_mm=tool_radius)
        points_repeat = [item for tool_pass in repeat for item in tool_pass.points]
        self.assertEqual(len(points), len(points_repeat))
        cl_points = np.array([item.cl_point for item in points], dtype=float)
        cl_points_repeat = np.array([item.cl_point for item in points_repeat], dtype=float)
        self.assertTrue(np.allclose(cl_points, cl_points_repeat))

    def test_surface_link_points_follow_constant_u(self) -> None:
        """Verify surface link samples stay on a constant-u boundary."""
        surface = create_default_surface(rows=5, cols=5)
        passes = generate_zigzag_toolpath(
            surface,
            stepover_mm=2.0,
            tool_radius_mm=1.5,
            link_passes=True,
        )
        self.assertGreater(len(passes), 1)

        link_counts = [len(tool_pass.link_points) for tool_pass in passes[:-1]]
        self.assertTrue(all(count >= 2 for count in link_counts))

        first_link = passes[0].link_points
        self.assertGreaterEqual(len(first_link), 2)
        u_values = {round(float(item.u_value), 7) for item in first_link}
        self.assertEqual(len(u_values), 1)


if __name__ == "__main__":
    unittest.main()
