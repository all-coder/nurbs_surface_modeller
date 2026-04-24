"""Surface generation by offsetting a source surface along its center normal."""

from __future__ import annotations

import numpy as np

from config import EPSILON
from surface.factory import create_surface_from_arrays
from surface.nurbs_surface import NURBSSurface


def extrude_surface_along_center_normal(
    surface: NURBSSurface,
    distance: float,
) -> NURBSSurface:
    """Create a derived surface by offsetting control points along center normal.

    Inputs:
        surface: Source surface instance to derive from.
        distance: Positive offset distance along the center normal.

    Outputs:
        New NURBSSurface instance with translated control net.

    Behavior:
        Evaluates the source surface normal at (u=0.5, v=0.5) and translates all
        control points by `distance * normal` while preserving degree, knots,
        and weights.
    """
    if distance <= 0.0:
        raise ValueError("distance must be strictly positive")

    normal = np.asarray(surface.evaluate_normal(0.5, 0.5), dtype=float)
    if normal.shape != (3,):
        raise ValueError("surface normal must have shape (3,)")
    if not np.all(np.isfinite(normal)):
        raise ValueError("surface normal must be finite")

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < EPSILON:
        raise ValueError("surface normal magnitude is too small")
    unit_normal = normal / normal_norm

    offset = float(distance) * unit_normal
    control_net = np.asarray(surface.control_net, dtype=float) + offset[None, None, :]

    return create_surface_from_arrays(
        control_net=control_net,
        weights=np.asarray(surface.weights, dtype=float).copy(),
        degree_u=int(surface.degree_u),
        degree_v=int(surface.degree_v),
        knot_u=np.asarray(surface.knot_u, dtype=float).copy(),
        knot_v=np.asarray(surface.knot_v, dtype=float).copy(),
    )
