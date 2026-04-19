"""Surface generation by extruding a source curve along a direction vector."""

from __future__ import annotations

import numpy as np

from config import DEFAULT_DEGREE_U, DEFAULT_DEGREE_V, EPSILON
from core.curves import CurveObject, convert_curve_to_nurbs
from surface.factory import create_surface_from_arrays
from surface.nurbs_surface import NURBSSurface


def extrude_surface_from_curve(
    curve: CurveObject,
    direction: np.ndarray,
    height: float,
    layer_count: int,
    samples_along_curve: int = 40,
) -> NURBSSurface:
    """Create a NURBS surface by extruding one curve over multiple layers.

    Inputs:
        curve: Source curve to extrude.
        direction: Extrusion direction vector with shape (3,).
        height: Total extrusion height in model units.
        layer_count: Number of control rows generated along extrusion direction.
        samples_along_curve: Number of sampled control points along the source curve.

    Outputs:
        Generated NURBSSurface instance.

    Behavior:
        Samples the source curve into one base row and copies it across layers
        using equally spaced offsets along a normalized direction vector.
    """
    if height <= 0.0:
        raise ValueError("height must be strictly positive")
    if layer_count < 2:
        raise ValueError("layer_count must be at least 2")
    if samples_along_curve < 2:
        raise ValueError("samples_along_curve must be at least 2")

    direction_vector = np.asarray(direction, dtype=float)
    if direction_vector.shape != (3,):
        raise ValueError("direction must have shape (3,)")

    direction_norm = float(np.linalg.norm(direction_vector))
    if direction_norm < EPSILON:
        raise ValueError("direction must have non-zero magnitude")
    unit_direction = direction_vector / direction_norm

    source_curve = convert_curve_to_nurbs(curve)
    sample_count = max(int(samples_along_curve), int(source_curve.control_points.shape[0]))
    base_profile = source_curve.sample_points(sample_count)

    layer_offsets = np.linspace(0.0, float(height), int(layer_count), endpoint=True)

    control_net = np.zeros((int(layer_count), sample_count, 3), dtype=float)
    for layer_index, layer_offset in enumerate(layer_offsets):
        control_net[layer_index] = base_profile + float(layer_offset) * unit_direction[None, :]

    weights = np.ones((int(layer_count), sample_count), dtype=float)

    degree_u = min(DEFAULT_DEGREE_U, int(layer_count) - 1)
    degree_v = min(DEFAULT_DEGREE_V, sample_count - 1)

    return create_surface_from_arrays(
        control_net=control_net,
        weights=weights,
        degree_u=degree_u,
        degree_v=degree_v,
    )
