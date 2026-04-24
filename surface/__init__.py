"""Surface package exports for NURBS model creation and evaluation."""

from surface.factory import create_default_surface, create_surface_from_arrays
from surface.nurbs_surface import NURBSSurface, SurfaceEvaluation
from surface.providers.surface_body_extrusion import build_surface_extrusion_mesh
from surface.providers.extrusion import extrude_surface_from_curve
from surface.providers.loft import loft_surface_from_curves
from surface.providers.surface_normal_extrusion import extrude_surface_along_center_normal

__all__ = [
	"NURBSSurface",
	"SurfaceEvaluation",
	"create_default_surface",
	"create_surface_from_arrays",
	"build_surface_extrusion_mesh",
	"extrude_surface_from_curve",
	"loft_surface_from_curves",
	"extrude_surface_along_center_normal",
]
