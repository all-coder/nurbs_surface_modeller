"""Global configuration constants for the NURBS surface modeller project."""

APP_NAME = "NURBS Surface Modeller"

# Numerical stability and geometry tolerances (millimeters unless noted).
EPSILON = 1e-10
COLLINEAR_TOLERANCE_MM = 0.01

# Default machining parameters.
DEFAULT_TOOL_RADIUS_MM = 2.0
DEFAULT_STEPOVER_MM = 1.0
DEFAULT_FEED_RATE_MM_PER_MIN = 300.0
DEFAULT_PLUNGE_RATE_MM_PER_MIN = 180.0
DEFAULT_SAFE_Z_MM = 10.0
DEFAULT_SPINDLE_RPM = 8000

# Surface sampling parameters.
DEFAULT_SURFACE_SAMPLES_U = 45
DEFAULT_SURFACE_SAMPLES_V = 45
DEFAULT_TOOLPATH_SAMPLES_U = 120
DEFAULT_LINK_SAMPLES = 12

# Interactive control-net editing parameters.
DEFAULT_CONTROL_POINT_WIDGET_RADIUS = 1.6
DEFAULT_SELECTED_POINT_WIDGET_RADIUS = 2.2
DEFAULT_DRAG_SENSITIVITY = 1.0
DEFAULT_DRAG_UPDATE_INTERVAL_SEC = 0.016

# Default geometric setup.
DEFAULT_DEGREE_U = 3
DEFAULT_DEGREE_V = 3
