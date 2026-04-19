"""Factory helpers to create NURBS surface instances from common inputs."""

from __future__ import annotations

import numpy as np

from config import DEFAULT_DEGREE_U, DEFAULT_DEGREE_V
from core.bspline import generate_clamped_uniform_knot_vector
from surface.nurbs_surface import NURBSSurface


def create_surface_from_arrays(
    control_net: np.ndarray,
    weights: np.ndarray,
    degree_u: int,
    degree_v: int,
    knot_u: np.ndarray | None = None,
    knot_v: np.ndarray | None = None,
) -> NURBSSurface:
    """Construct a NURBSSurface from explicit arrays and optional knot vectors.

    Inputs:
        control_net: Control net array with shape (m, n, 3).
        weights: Positive weight matrix with shape (m, n).
        degree_u: Spline degree in u direction.
        degree_v: Spline degree in v direction.
        knot_u: Optional knot vector in u; generated when omitted.
        knot_v: Optional knot vector in v; generated when omitted.

    Outputs:
        NURBSSurface instance.

    Behavior:
        Generates clamped uniform knots when knot vectors are not supplied and
        returns a fully validated surface object.
    """
    u_count, v_count = control_net.shape[:2]
    knot_u_value = (
        np.asarray(knot_u, dtype=float)
        if knot_u is not None
        else generate_clamped_uniform_knot_vector(u_count, degree_u)
    )
    knot_v_value = (
        np.asarray(knot_v, dtype=float)
        if knot_v is not None
        else generate_clamped_uniform_knot_vector(v_count, degree_v)
    )
    return NURBSSurface(
        control_net=np.asarray(control_net, dtype=float),
        weights=np.asarray(weights, dtype=float),
        knot_u=knot_u_value,
        knot_v=knot_v_value,
        degree_u=degree_u,
        degree_v=degree_v,
    )


def create_default_surface(rows: int = 4, cols: int = 4) -> NURBSSurface:
    """Create a default demo surface suitable for GUI startup and testing.

    Inputs:
        rows: Number of control points along u direction.
        cols: Number of control points along v direction.

    Outputs:
        NURBSSurface instance with smooth initial shape.

    Behavior:
        Builds a regular control lattice with a mild sinusoidal elevation and
        unit weights so users can immediately inspect surface behavior.
    """
    x_coords = np.linspace(0.0, 60.0, rows)
    y_coords = np.linspace(0.0, 60.0, cols)

    control_net = np.zeros((rows, cols, 3), dtype=float)
    for i_u, x_value in enumerate(x_coords):
        for i_v, y_value in enumerate(y_coords):
            z_value = 8.0 * np.sin(i_u / max(rows - 1, 1) * np.pi) * np.cos(
                i_v / max(cols - 1, 1) * np.pi
            )
            control_net[i_u, i_v] = np.array([x_value, y_value, z_value], dtype=float)

    weights = np.ones((rows, cols), dtype=float)
    return create_surface_from_arrays(
        control_net=control_net,
        weights=weights,
        degree_u=min(DEFAULT_DEGREE_U, rows - 1),
        degree_v=min(DEFAULT_DEGREE_V, cols - 1),
    )
