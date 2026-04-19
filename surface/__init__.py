"""Surface package exports for NURBS model creation and evaluation."""

from surface.factory import create_default_surface, create_surface_from_arrays
from surface.nurbs_surface import NURBSSurface, SurfaceEvaluation
from surface.providers.extrusion import extrude_surface_from_curve
from surface.providers.loft import loft_surface_from_curves

__all__ = [
	"NURBSSurface",
	"SurfaceEvaluation",
	"create_default_surface",
	"create_surface_from_arrays",
	"extrude_surface_from_curve",
	"loft_surface_from_curves",
]
