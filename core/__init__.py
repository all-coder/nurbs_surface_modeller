"""Core math and geometry entities for NURBS modelling workflows."""

from core.bspline import (
	basis_first_derivatives,
	basis_functions,
	find_span,
	generate_clamped_uniform_knot_vector,
)
from core.curve_validation import (
	build_knots_for_mode,
	generate_nonuniform_knot_vector,
	validate_curve_control_points,
	validate_curve_degree,
	validate_curve_knot_vector,
	validate_curve_weights,
	validate_knot_mode,
)
from core.curves import BSplineCurve, BezierCurve, CurveEvaluation, NURBSCurve, convert_curve_to_nurbs

__all__ = [
	"BSplineCurve",
	"BezierCurve",
	"CurveEvaluation",
	"NURBSCurve",
	"basis_first_derivatives",
	"basis_functions",
	"build_knots_for_mode",
	"convert_curve_to_nurbs",
	"find_span",
	"generate_clamped_uniform_knot_vector",
	"generate_nonuniform_knot_vector",
	"validate_curve_control_points",
	"validate_curve_degree",
	"validate_curve_knot_vector",
	"validate_curve_weights",
	"validate_knot_mode",
]
