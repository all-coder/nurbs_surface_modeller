"""Generate zig-zag CL toolpaths from NURBS surfaces with chord filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    COLLINEAR_TOLERANCE_MM,
    DEFAULT_TOOLPATH_SAMPLES_U,
    EPSILON,
)
from surface.nurbs_surface import NURBSSurface
from utils.geometry import normalize_vector, point_line_deviation


@dataclass(frozen=True)
class ToolpathPoint:
    """Represent one tool-contact sample and corresponding cutter-location sample.

    Inputs:
        contact_point: Surface contact point in Cartesian coordinates.
        cl_point: Cutter location point offset by tool radius along normal.
        normal: Unit normal vector used for offset.
        u_value: Surface parameter u at the sample point.
        v_value: Surface parameter v at the sample point.

    Outputs:
        Immutable sample object consumed by filtering and G-code exporting.

    Behavior:
        Stores both geometric and parameter metadata so generated G-code can be
        traced back to original surface locations.
    """

    contact_point: np.ndarray
    cl_point: np.ndarray
    normal: np.ndarray
    u_value: float
    v_value: float


@dataclass(frozen=True)
class ToolpathPass:
    """Represent one zig-zag scanline pass at fixed v parameter.

    Inputs:
        v_parameter: Constant v parameter for this pass.
        direction: Pass travel direction string, either "forward" or "reverse".
        points: Ordered toolpath points for this pass.

    Outputs:
        Immutable pass object grouping pass metadata and points.

    Behavior:
        Enables explicit pass-level operations such as rapid linking and pass
        statistics during G-code generation.
    """

    v_parameter: float
    direction: str
    points: list[ToolpathPoint]


def estimate_v_step_from_mm(
    surface: NURBSSurface,
    stepover_mm: float,
    sample_count: int = 120,
) -> float:
    """Estimate parameter step in v matching an approximate metric stepover.

    Inputs:
        surface: NURBS surface to sample.
        stepover_mm: Desired stepover in millimeters.
        sample_count: Number of samples for arc-length approximation.

    Outputs:
        Estimated v-parameter step value.

    Behavior:
        Samples the mid-u isoparametric curve to approximate arc length and
        converts physical stepover to parameter step size.
    """
    if stepover_mm <= 0.0:
        raise ValueError("stepover_mm must be greater than zero")

    (u_min, u_max), (v_min, v_max) = surface.parameter_ranges()
    u_mid = 0.5 * (u_min + u_max)
    v_values = np.linspace(v_min, v_max, sample_count, endpoint=True)

    points = np.array([surface.evaluate_point(float(u_mid), float(v)) for v in v_values])
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = float(np.sum(segments))

    param_range = v_max - v_min
    if arc_length < EPSILON or param_range < EPSILON:
        return param_range / 20.0

    mm_per_param = arc_length / param_range
    step = stepover_mm / mm_per_param

    min_step = param_range / 300.0
    max_step = param_range / 2.0
    return float(max(min_step, min(step, max_step)))


def filter_collinear_toolpath_points(
    points: list[ToolpathPoint],
    tolerance_mm: float = COLLINEAR_TOLERANCE_MM,
) -> list[ToolpathPoint]:
    """Remove nearly collinear toolpath points using CL geometry deviation.

    Inputs:
        points: Ordered toolpath samples from one pass.
        tolerance_mm: Maximum deviation considered collinear.

    Outputs:
        Reduced list of toolpath points preserving geometric fidelity.

    Behavior:
        Keeps first and last points and removes interior points whose cutter
        locations are closer than tolerance to neighboring line segments.
    """
    if len(points) <= 2:
        return points[:]

    filtered: list[ToolpathPoint] = [points[0]]
    for index in range(1, len(points) - 1):
        prev_point = filtered[-1]
        curr_point = points[index]
        next_point = points[index + 1]

        deviation = point_line_deviation(
            curr_point.cl_point,
            prev_point.cl_point,
            next_point.cl_point,
        )

        if deviation >= tolerance_mm:
            filtered.append(curr_point)

    filtered.append(points[-1])
    return filtered


def generate_zigzag_toolpath(
    surface: NURBSSurface,
    stepover_mm: float,
    tool_radius_mm: float,
    u_samples: int = DEFAULT_TOOLPATH_SAMPLES_U,
    tolerance_mm: float = COLLINEAR_TOLERANCE_MM,
) -> list[ToolpathPass]:
    """Generate zig-zag passes in u direction and offset points to CL positions.

    Inputs:
        surface: Surface model used for evaluation and normals.
        stepover_mm: Desired stepover distance between adjacent scanlines.
        tool_radius_mm: Tool radius used for normal offset.
        u_samples: Number of points sampled per pass before filtering.
        tolerance_mm: Chord tolerance for collinearity filtering.

    Outputs:
        List of ToolpathPass objects covering the v domain.

    Behavior:
        Creates scanlines at increasing v values and alternates u traversal
        direction per pass. Each contact point is offset by R*n_hat to create
        cutter-location points and then simplified by tolerance filtering.
    """
    if tool_radius_mm <= 0.0:
        raise ValueError("tool_radius_mm must be greater than zero")
    if u_samples < 2:
        raise ValueError("u_samples must be at least 2")

    (u_min, u_max), (v_min, v_max) = surface.parameter_ranges()
    v_step = estimate_v_step_from_mm(surface, stepover_mm)

    v_values = [float(v_min)]
    current_v = float(v_min)
    while current_v + v_step < v_max:
        current_v += v_step
        v_values.append(float(current_v))
    if v_values[-1] < v_max - EPSILON:
        v_values.append(float(v_max))

    passes: list[ToolpathPass] = []
    for pass_index, v_value in enumerate(v_values):
        forward = pass_index % 2 == 0
        direction = "forward" if forward else "reverse"

        u_values = np.linspace(u_min, u_max, u_samples, endpoint=True)
        if not forward:
            u_values = u_values[::-1]

        raw_points: list[ToolpathPoint] = []
        for u_value in u_values:
            eval_result = surface.evaluate_derivatives(float(u_value), float(v_value))
            normal = normalize_vector(
                np.cross(eval_result.du, eval_result.dv),
                fallback=np.array([0.0, 0.0, 1.0]),
            )
            cl_point = eval_result.point + tool_radius_mm * normal
            raw_points.append(
                ToolpathPoint(
                    contact_point=eval_result.point,
                    cl_point=cl_point,
                    normal=normal,
                    u_value=float(u_value),
                    v_value=float(v_value),
                )
            )

        filtered_points = filter_collinear_toolpath_points(raw_points, tolerance_mm)
        passes.append(ToolpathPass(v_parameter=float(v_value), direction=direction, points=filtered_points))

    return passes
