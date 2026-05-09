"""Generate zig-zag CL toolpaths from NURBS surfaces with chord filtering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import (
    COLLINEAR_TOLERANCE_MM,
    DEFAULT_LINK_SAMPLES,
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
    link_points: list[ToolpathPoint] = field(default_factory=list)


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
        Contact-point deviation is also checked to preserve curvature detail.
    """
    if len(points) <= 2:
        return points[:]

    filtered: list[ToolpathPoint] = [points[0]]
    for index in range(1, len(points) - 1):
        prev_point = filtered[-1]
        curr_point = points[index]
        next_point = points[index + 1]

        deviation_cl = point_line_deviation(
            curr_point.cl_point,
            prev_point.cl_point,
            next_point.cl_point,
        )
        deviation_contact = point_line_deviation(
            curr_point.contact_point,
            prev_point.contact_point,
            next_point.contact_point,
        )

        if deviation_cl >= tolerance_mm or deviation_contact >= tolerance_mm:
            filtered.append(curr_point)

    filtered.append(points[-1])
    return filtered


def _stable_normal(
    du: np.ndarray,
    dv: np.ndarray,
    previous_normal: np.ndarray | None,
) -> np.ndarray:
    """Compute a stable unit normal, reusing prior direction when needed."""
    cross_value = np.cross(du, dv)
    cross_norm = float(np.linalg.norm(cross_value))
    if cross_norm < EPSILON:
        if previous_normal is not None:
            return previous_normal
        return np.array([0.0, 0.0, 1.0], dtype=float)

    normal = cross_value / cross_norm
    if previous_normal is not None and float(np.dot(normal, previous_normal)) < 0.0:
        normal = -normal
    return normal


def _orient_mesh_normal(
    normal: np.ndarray,
    tool_axis: np.ndarray,
    previous_normal: np.ndarray | None,
) -> np.ndarray:
    """Orient a mesh normal to face the tool axis and stay consistent."""
    oriented = normalize_vector(normal, fallback=tool_axis)
    if float(np.dot(oriented, tool_axis)) < 0.0:
        oriented = -oriented
    if previous_normal is not None and float(np.dot(oriented, previous_normal)) < 0.0:
        oriented = -oriented
    return oriented


def _point_in_triangle_xy(
    point_xy: tuple[float, float],
    tri_xy: np.ndarray,
) -> bool:
    """Check whether XY point lies inside the projected triangle."""
    px, py = point_xy
    x0, y0 = float(tri_xy[0, 0]), float(tri_xy[0, 1])
    x1, y1 = float(tri_xy[1, 0]), float(tri_xy[1, 1])
    x2, y2 = float(tri_xy[2, 0]), float(tri_xy[2, 1])

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < EPSILON:
        return False

    inv = 1.0 / denom
    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv
    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv
    c = 1.0 - a - b
    return (
        a >= -EPSILON
        and b >= -EPSILON
        and c >= -EPSILON
        and a <= 1.0 + EPSILON
        and b <= 1.0 + EPSILON
        and c <= 1.0 + EPSILON
    )


def _collect_mesh_intersections(
    x_value: float,
    y_value: float,
    tri_vertices: np.ndarray,
    tri_normals: np.ndarray,
    tri_min_xy: np.ndarray,
    tri_max_xy: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Return Z intersections and normals for a vertical ray at (x, y)."""
    candidates = (
        (x_value >= tri_min_xy[:, 0])
        & (x_value <= tri_max_xy[:, 0])
        & (y_value >= tri_min_xy[:, 1])
        & (y_value <= tri_max_xy[:, 1])
    )
    if not np.any(candidates):
        return []

    hits: list[tuple[float, np.ndarray]] = []
    candidate_indices = np.nonzero(candidates)[0]
    for tri_index in candidate_indices:
        normal = tri_normals[tri_index]
        if abs(float(normal[2])) < EPSILON:
            continue

        tri = tri_vertices[tri_index]
        d_value = float(np.dot(normal, tri[0]))
        z_value = (d_value - normal[0] * x_value - normal[1] * y_value) / normal[2]

        if not np.isfinite(z_value):
            continue

        if not _point_in_triangle_xy((x_value, y_value), tri[:, :2]):
            continue

        hits.append((float(z_value), normal))

    return hits


def generate_zigzag_toolpath(
    surface: NURBSSurface,
    stepover_mm: float,
    tool_radius_mm: float,
    u_samples: int = DEFAULT_TOOLPATH_SAMPLES_U,
    tolerance_mm: float = COLLINEAR_TOLERANCE_MM,
    link_passes: bool = False,
    link_samples: int = DEFAULT_LINK_SAMPLES,
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
        Optional surface-link moves are sampled along constant u between passes.
    """
    if tool_radius_mm <= 0.0:
        raise ValueError("tool_radius_mm must be greater than zero")
    if u_samples < 2:
        raise ValueError("u_samples must be at least 2")
    if link_samples < 2:
        raise ValueError("link_samples must be at least 2")

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
    previous_normal: np.ndarray | None = None
    for pass_index, v_value in enumerate(v_values):
        forward = pass_index % 2 == 0
        direction = "forward" if forward else "reverse"

        u_values = np.linspace(u_min, u_max, u_samples, endpoint=True)
        if not forward:
            u_values = u_values[::-1]

        raw_points: list[ToolpathPoint] = []
        for u_value in u_values:
            eval_result = surface.evaluate_derivatives(float(u_value), float(v_value))
            normal = _stable_normal(eval_result.du, eval_result.dv, previous_normal)
            previous_normal = normal
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
        link_points: list[ToolpathPoint] = []
        if link_passes and pass_index < len(v_values) - 1:
            next_v = float(v_values[pass_index + 1])
            u_link = float(u_max if forward else u_min)
            link_v_values = np.linspace(v_value, next_v, link_samples, endpoint=True)
            for link_v in link_v_values:
                eval_result = surface.evaluate_derivatives(u_link, float(link_v))
                normal = _stable_normal(eval_result.du, eval_result.dv, previous_normal)
                previous_normal = normal
                cl_point = eval_result.point + tool_radius_mm * normal
                link_points.append(
                    ToolpathPoint(
                        contact_point=eval_result.point,
                        cl_point=cl_point,
                        normal=normal,
                        u_value=u_link,
                        v_value=float(link_v),
                    )
                )

        passes.append(
            ToolpathPass(
                v_parameter=float(v_value),
                direction=direction,
                points=filtered_points,
                link_points=link_points,
            )
        )

    return passes


def generate_mesh_zigzag_toolpath(
    vertices: np.ndarray,
    triangles: np.ndarray,
    stepover_mm: float,
    tool_radius_mm: float,
    u_samples: int = DEFAULT_TOOLPATH_SAMPLES_U,
    tolerance_mm: float = COLLINEAR_TOLERANCE_MM,
    link_passes: bool = False,
    link_samples: int = DEFAULT_LINK_SAMPLES,
    envelope_mode: str = "top",
) -> list[ToolpathPass]:
    """Generate zig-zag toolpaths from a triangulated mesh envelope."""
    if tool_radius_mm <= 0.0:
        raise ValueError("tool_radius_mm must be greater than zero")
    if stepover_mm <= 0.0:
        raise ValueError("stepover_mm must be greater than zero")
    if u_samples < 2:
        raise ValueError("u_samples must be at least 2")
    if link_samples < 2:
        raise ValueError("link_samples must be at least 2")

    normalized_mode = str(envelope_mode).strip().lower()
    if normalized_mode not in {"top", "full", "bottom"}:
        raise ValueError("envelope_mode must be 'top', 'bottom', or 'full'")

    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(triangles, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    if tris.size == 0:
        return []
    if np.min(tris) < 0 or np.max(tris) >= verts.shape[0]:
        raise ValueError("triangle indices are out of bounds")

    tri_vertices = verts[tris]
    tri_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    tri_norms = np.linalg.norm(tri_normals, axis=1)
    valid = tri_norms > EPSILON
    if not np.any(valid):
        return []

    tri_vertices = tri_vertices[valid]
    tri_normals = (tri_normals[valid].T / tri_norms[valid]).T

    tri_min_xy = np.min(tri_vertices[:, :, :2], axis=1)
    tri_max_xy = np.max(tri_vertices[:, :, :2], axis=1)

    x_min = float(np.min(verts[:, 0]))
    x_max = float(np.max(verts[:, 0]))
    y_min = float(np.min(verts[:, 1]))
    y_max = float(np.max(verts[:, 1]))

    y_values = [float(y_min)]
    current_y = float(y_min)
    while current_y + stepover_mm < y_max:
        current_y += stepover_mm
        y_values.append(float(current_y))
    if y_values[-1] < y_max - EPSILON:
        y_values.append(float(y_max))

    tool_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    passes: list[ToolpathPass] = []

    for pass_index, y_value in enumerate(y_values):
        forward = pass_index % 2 == 0
        direction = "forward" if forward else "reverse"

        x_values = np.linspace(x_min, x_max, u_samples, endpoint=True)
        if not forward:
            x_values = x_values[::-1]

        def build_segments(use_bottom: bool) -> list[list[ToolpathPoint]]:
            segments: list[list[ToolpathPoint]] = []
            current: list[ToolpathPoint] = []
            previous_normal: np.ndarray | None = None

            for x_value in x_values:
                hits = _collect_mesh_intersections(
                    float(x_value),
                    float(y_value),
                    tri_vertices,
                    tri_normals,
                    tri_min_xy,
                    tri_max_xy,
                )

                if not hits:
                    if current:
                        segments.append(current)
                        current = []
                    previous_normal = None
                    continue

                if use_bottom:
                    z_value, normal = min(hits, key=lambda item: item[0])
                else:
                    z_value, normal = max(hits, key=lambda item: item[0])

                contact = np.array([float(x_value), float(y_value), float(z_value)], dtype=float)
                normal = _orient_mesh_normal(normal, tool_axis, previous_normal)
                previous_normal = normal
                cl_point = contact + tool_radius_mm * normal

                current.append(
                    ToolpathPoint(
                        contact_point=contact,
                        cl_point=cl_point,
                        normal=normal,
                        u_value=float(x_value),
                        v_value=float(y_value),
                    )
                )

            if current:
                segments.append(current)
            return segments

        top_segments = build_segments(use_bottom=False)
        bottom_segments: list[list[ToolpathPoint]] = []
        if normalized_mode in {"bottom", "full"}:
            bottom_segments = build_segments(use_bottom=True)

        last_top_pass_index: int | None = None
        for segment in top_segments:
            filtered = filter_collinear_toolpath_points(segment, tolerance_mm)
            if len(filtered) < 2:
                continue
            passes.append(
                ToolpathPass(
                    v_parameter=float(y_value),
                    direction=direction,
                    points=filtered,
                    link_points=[],
                )
            )
            last_top_pass_index = len(passes) - 1

        for segment in bottom_segments:
            filtered = filter_collinear_toolpath_points(segment, tolerance_mm)
            if len(filtered) < 2:
                continue
            passes.append(
                ToolpathPass(
                    v_parameter=float(y_value),
                    direction=direction,
                    points=filtered,
                    link_points=[],
                )
            )

        if link_passes and len(top_segments) == 1 and last_top_pass_index is not None:
            next_index = pass_index + 1
            if next_index < len(y_values):
                next_y = float(y_values[next_index])
                x_link = float(x_max if forward else x_min)

                last_pass = passes[last_top_pass_index]
                if last_pass.points:
                    end_point = last_pass.points[-1].contact_point
                    if abs(float(end_point[0]) - x_link) < max(1e-6, stepover_mm * 0.05):
                        link_values = np.linspace(y_value, next_y, link_samples, endpoint=True)
                        link_points: list[ToolpathPoint] = []
                        previous_normal: np.ndarray | None = None
                        for link_y in link_values:
                            hits = _collect_mesh_intersections(
                                x_link,
                                float(link_y),
                                tri_vertices,
                                tri_normals,
                                tri_min_xy,
                                tri_max_xy,
                            )
                            if not hits:
                                continue
                            z_value, normal = max(hits, key=lambda item: item[0])
                            contact = np.array([x_link, float(link_y), float(z_value)], dtype=float)
                            normal = _orient_mesh_normal(normal, tool_axis, previous_normal)
                            previous_normal = normal
                            cl_point = contact + tool_radius_mm * normal
                            link_points.append(
                                ToolpathPoint(
                                    contact_point=contact,
                                    cl_point=cl_point,
                                    normal=normal,
                                    u_value=float(x_link),
                                    v_value=float(link_y),
                                )
                            )
                        if link_points:
                            passes[last_top_pass_index] = ToolpathPass(
                                v_parameter=last_pass.v_parameter,
                                direction=last_pass.direction,
                                points=last_pass.points,
                                link_points=link_points,
                            )

    return passes
