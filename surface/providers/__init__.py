"""Surface generation providers built from higher-level geometry workflows."""

from surface.providers.extrusion import extrude_surface_from_curve
from surface.providers.loft import loft_surface_from_curves
from surface.providers.surface_body_extrusion import build_surface_extrusion_mesh
from surface.providers.surface_normal_extrusion import extrude_surface_along_center_normal

__all__ = [
    "extrude_surface_from_curve",
    "loft_surface_from_curves",
    "build_surface_extrusion_mesh",
    "extrude_surface_along_center_normal",
]
