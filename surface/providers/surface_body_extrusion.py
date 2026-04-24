"""Triangulated shell/solid body generation from NURBS surfaces."""

from __future__ import annotations

import numpy as np

from config import EPSILON
from surface.nurbs_surface import NURBSSurface


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"solid", "shell"}:
        raise ValueError("mode must be either 'solid' or 'shell'")
    return normalized


def _triangulate_grid(
    rows: int,
    cols: int,
    index_offset: int,
    reverse_winding: bool,
) -> np.ndarray:
    triangles: list[list[int]] = []
    for i_row in range(rows - 1):
        for i_col in range(cols - 1):
            a = index_offset + i_row * cols + i_col
            b = index_offset + (i_row + 1) * cols + i_col
            c = index_offset + (i_row + 1) * cols + (i_col + 1)
            d = index_offset + i_row * cols + (i_col + 1)

            if reverse_winding:
                triangles.append([a, c, b])
                triangles.append([a, d, c])
            else:
                triangles.append([a, b, c])
                triangles.append([a, c, d])
    return np.asarray(triangles, dtype=np.int64)


def _boundary_loop_indices(rows: int, cols: int) -> list[int]:
    loop: list[int] = []

    for i_col in range(cols):
        loop.append(i_col)
    for i_row in range(1, rows):
        loop.append(i_row * cols + (cols - 1))
    for i_col in range(cols - 2, -1, -1):
        loop.append((rows - 1) * cols + i_col)
    for i_row in range(rows - 2, 0, -1):
        loop.append(i_row * cols)

    if len(loop) < 4:
        raise ValueError("surface boundary is too small to build side walls")
    return loop


def _build_side_triangles(
    boundary_loop: list[int],
    lower_offset: int,
    upper_offset: int,
) -> np.ndarray:
    triangles: list[list[int]] = []
    loop_count = len(boundary_loop)
    for i_edge in range(loop_count):
        lower_a = lower_offset + boundary_loop[i_edge]
        lower_b = lower_offset + boundary_loop[(i_edge + 1) % loop_count]
        upper_a = upper_offset + boundary_loop[i_edge]
        upper_b = upper_offset + boundary_loop[(i_edge + 1) % loop_count]

        triangles.append([lower_a, lower_b, upper_b])
        triangles.append([lower_a, upper_b, upper_a])
    return np.asarray(triangles, dtype=np.int64)


def build_surface_extrusion_mesh(
    surface: NURBSSurface,
    distance: float,
    mode: str,
    u_samples: int = 45,
    v_samples: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """Build triangulated shell/solid extrusion mesh from one source surface.

    Inputs:
        surface: Source surface used to sample the base skin.
        distance: Positive extrusion distance (or shell thickness).
        mode: Either "solid" or "shell".
        u_samples: Sampling resolution in u direction.
        v_samples: Sampling resolution in v direction.

    Outputs:
        Tuple `(vertices, triangles)` where:
        - vertices has shape `(N, 3)`
        - triangles has shape `(M, 3)` with integer vertex indices.

    Behavior:
        Uses center normal `evaluate_normal(0.5, 0.5)` as extrusion direction,
        samples the source surface grid, then builds a closed triangulated body.
    """
    normalized_mode = _validate_mode(mode)
    if distance <= 0.0:
        raise ValueError("distance must be strictly positive")
    if u_samples < 2 or v_samples < 2:
        raise ValueError("u_samples and v_samples must be at least 2")

    points, _, _, _ = surface.evaluate_grid(u_samples=u_samples, v_samples=v_samples)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("sampled surface points must have shape (u, v, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("sampled surface points must be finite")

    center_normal = np.asarray(surface.evaluate_normal(0.5, 0.5), dtype=float)
    if center_normal.shape != (3,):
        raise ValueError("center normal must have shape (3,)")
    if not np.all(np.isfinite(center_normal)):
        raise ValueError("center normal must be finite")

    normal_norm = float(np.linalg.norm(center_normal))
    if normal_norm < EPSILON:
        raise ValueError("center normal magnitude is too small for extrusion")
    unit_normal = center_normal / normal_norm

    rows, cols = points.shape[:2]
    boundary_loop = _boundary_loop_indices(rows, cols)

    if normalized_mode == "solid":
        bottom_points = points
        top_points = points + float(distance) * unit_normal[None, None, :]

        bottom_vertices = bottom_points.reshape(-1, 3)
        top_vertices = top_points.reshape(-1, 3)
        vertices = np.vstack((bottom_vertices, top_vertices))

        base_count = rows * cols
        bottom_triangles = _triangulate_grid(rows, cols, index_offset=0, reverse_winding=False)
        top_triangles = _triangulate_grid(rows, cols, index_offset=base_count, reverse_winding=True)
        side_triangles = _build_side_triangles(boundary_loop, lower_offset=0, upper_offset=base_count)
        triangles = np.vstack((bottom_triangles, top_triangles, side_triangles))
        return vertices, triangles

    half_offset = 0.5 * float(distance) * unit_normal
    outer_points = points + half_offset[None, None, :]
    inner_points = points - half_offset[None, None, :]

    outer_vertices = outer_points.reshape(-1, 3)
    inner_vertices = inner_points.reshape(-1, 3)
    vertices = np.vstack((outer_vertices, inner_vertices))

    base_count = rows * cols
    outer_triangles = _triangulate_grid(rows, cols, index_offset=0, reverse_winding=False)
    inner_triangles = _triangulate_grid(rows, cols, index_offset=base_count, reverse_winding=True)
    boundary_triangles = _build_side_triangles(boundary_loop, lower_offset=0, upper_offset=base_count)
    triangles = np.vstack((outer_triangles, inner_triangles, boundary_triangles))
    return vertices, triangles
