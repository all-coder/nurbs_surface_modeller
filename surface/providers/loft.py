"""Loft-style surface generation from a collection of supported curves."""

from __future__ import annotations

import numpy as np

from config import DEFAULT_DEGREE_U, DEFAULT_DEGREE_V
from core.curves import CurveObject, convert_curve_to_nurbs
from surface.factory import create_surface_from_arrays
from surface.nurbs_surface import NURBSSurface


def loft_surface_from_curves(
    curves: list[CurveObject],
    samples_per_curve: int = 40,
) -> NURBSSurface:
    """Generate a NURBS surface by lofting through an ordered curve list.

    Inputs:
        curves: Ordered input curves to loft through.
        samples_per_curve: Number of samples taken on each source curve.

    Outputs:
        Generated NURBSSurface instance.

    Behavior:
        Converts all supported curve types to canonical NURBS representation,
        samples each curve into a profile line, and stacks sampled profiles as
        a control net before constructing the final surface.
    """
    if len(curves) < 2:
        raise ValueError("at least two curves are required to loft a surface")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    nurbs_curves = [convert_curve_to_nurbs(curve) for curve in curves]
    sample_count = max(
        int(samples_per_curve),
        max(curve.control_points.shape[0] for curve in nurbs_curves),
    )

    control_net = np.zeros((len(nurbs_curves), sample_count, 3), dtype=float)
    for curve_index, curve in enumerate(nurbs_curves):
        control_net[curve_index] = curve.sample_points(sample_count)

    weights = np.ones((len(nurbs_curves), sample_count), dtype=float)

    degree_u = min(DEFAULT_DEGREE_U, len(nurbs_curves) - 1)
    degree_v = min(DEFAULT_DEGREE_V, sample_count - 1)

    return create_surface_from_arrays(
        control_net=control_net,
        weights=weights,
        degree_u=degree_u,
        degree_v=degree_v,
    )
