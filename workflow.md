# NURBS Surface Modeller - Technical Documentation

## Overview

The NURBS Surface Modeller is a PyQt5-based desktop application for interactive design, analysis, and CNC machining of tensor-product NURBS (Non-Uniform Rational B-Spline) surfaces. The application provides a complete workflow from curve creation through toolpath generation and G-code export for CNC machining.

**Core Technologies:**
- **NURBS Mathematics**: Tensor-product rational spline surfaces with weighted control points
- **B-spline Basis Functions**: Cox-de Boor recursion algorithm for stable evaluation
- **GUI Framework**: PyQt5 with PyVista for 3D visualization
- **Numerical Computation**: NumPy for vector and matrix operations

---

## Application Architecture

### High-Level Design

```
User Input (GUI)
    ↓
Curve Management (Bezier, B-spline, NURBS)
    ↓
Surface Generation (Control-net, Loft, Extrude)
    ↓
Surface Editing (Control point manipulation)
    ↓
Toolpath Generation (Zigzag scanning with normal offset)
    ↓
G-code Export (CNC-ready machine code)
```

### Module Organization

- **core/**: Curve representations and B-spline basis functions
  - `curves.py`: BezierCurve, BSplineCurve, NURBSCurve classes
  - `bspline.py`: Basis function evaluation (Cox-de Boor algorithm)
  - `curve_validation.py`: Input validation for curves
  - `validation.py`: Shared validation utilities

- **surface/**: NURBS surface representation and generation
  - `nurbs_surface.py`: NURBSSurface class (tensor-product evaluation)
  - `factory.py`: Surface construction helpers
  - `providers/`: Surface generation methods
    - `loft.py`: Interpolation through multiple curves
    - `extrusion.py`: Linear curve sweep
    - `surface_normal_extrusion.py`: Surface offsetting
    - `surface_body_extrusion.py`: Triangulated solid/shell bodies

- **machining/**: CNC toolpath generation
  - `toolpath.py`: Zigzag scanning, normal offset, filtering

- **gcode/**: G-code generation and export
  - `exporter.py`: ToolpathPass → G-code text conversion

- **gui/**: Interactive PyQt5 user interface
  - `app.py`: Main application window (2000+ lines)

- **utils/**: Geometric utilities
  - `geometry.py`: Vector operations, deviation calculation

---

## Curve Types and Mathematical Foundation

### Curve Representation

The application supports three curve types, each inheriting from a common spline interface:

#### 1. Bezier Curves

**Mathematical Basis:**
Bezier curves are defined by the de Casteljau algorithm, a recursive linear interpolation scheme:

$$B(t) = \sum_{i=0}^{n} B_i^n(t) \mathbf{P}_i$$

where $B_i^n(t)$ are Bernstein basis functions:

$$B_i^n(t) = \binom{n}{i}(1-t)^{n-i}t^i$$

**Implementation (`core/curves.py`):**
```python
class BezierCurve:
    def __init__(self, control_points: np.ndarray):
        self.control_points = control_points  # shape: (n, 3)
        self.degree = len(control_points) - 1
    
    def evaluate_point(self, t: float) -> np.ndarray:
        # De Casteljau recursion: iteratively interpolate between control points
        # Time complexity: O(n²)
```

**Key Properties:**
- Degree = number of control points - 1
- Always passes through first and last control points
- Derivative computation via control point differences

#### 2. B-spline Curves (Non-Rational)

**Mathematical Basis:**
B-spline curves use basis functions with local support (non-zero over limited parameter ranges):

$$C(u) = \sum_{i=0}^{n} N_{i,p}(u) \mathbf{P}_i$$

where $N_{i,p}(u)$ are B-spline basis functions of degree $p$.

**Cox-de Boor Recursion:**
```
N_{i,0}(u) = 1 if U[i] ≤ u < U[i+1], else 0
N_{i,p}(u) = (u - U[i])/(U[i+p] - U[i]) * N_{i,p-1}(u) 
           + (U[i+p+1] - u)/(U[i+p+1] - U[i+1]) * N_{i+1,p-1}(u)
```

**Implementation (`core/bspline.py`):**
- `find_span(u, knots, n)`: Binary search for knot interval containing parameter
- `basis_functions(u, p, U)`: Evaluates all non-zero basis functions
- `basis_first_derivatives(u, p, U)`: Computes basis function derivatives

**Knot Vector Construction:**
- **Uniform**: Equally-spaced knots with clamped endpoints
- **Non-uniform**: User-defined or auto-generated for specific shapes

#### 3. NURBS Curves (Rational Weighted)

**Mathematical Basis:**
NURBS curves extend B-splines by adding weights to control points, enabling exact representation of conic sections:

$$C(u) = \frac{\sum_{i=0}^{n} w_i N_{i,p}(u) \mathbf{P}_i}{\sum_{i=0}^{n} w_i N_{i,p}(u)}$$

**Homogeneous Coordinates Implementation:**
To efficiently evaluate NURBS, the algorithm uses homogeneous coordinates:
1. Convert control points to 4D: $\mathbf{P}_i^w = [w_i P_{i,x}, w_i P_{i,y}, w_i P_{i,z}, w_i]$
2. Evaluate as regular B-spline in homogeneous space
3. Divide XYZ by W: $C(u) = [X/W, Y/W, Z/W]$

**Advantages:**
- Weights > 1.0 attract curve toward control point
- Weights < 1.0 repel curve away
- Weight = 1.0 recovers non-rational B-spline behavior

**Implementation (`core/curves.py`):**
```python
class NURBSCurve:
    def __init__(self, control_points: np.ndarray, degree: int, 
                 knots: np.ndarray, weights: np.ndarray):
        self.control_points = control_points  # shape: (n, 3)
        self.weights = weights                # shape: (n,), all > 0
        self.degree = degree
        self.knots = knots
    
    def evaluate_point(self, t: float) -> np.ndarray:
        # Homogeneous coordinate evaluation
        basis_values = basis_functions(t, self.degree, self.knots)
        # Compute weighted sum and rational division
```

### Derivative Computation

**For Bezier curves:** Direct control point differentiation
```
dB/dt = n * Σ(P_{i+1} - P_i) * B_i^{n-1}(t)
```

**For B-spline and NURBS curves:** Quotient rule applied to basis functions
```
dC/du = Σ dN_{i,p}/du * P_i   (for B-splines)
dC/du = [numerator' * denom - numerator * denom'] / denom²  (for NURBS)
```

---

## NURBS Surfaces

### Tensor-Product Structure

A tensor-product NURBS surface is defined over a 2D parameter domain $(u, v)$:

$$S(u,v) = \frac{\sum_{i=0}^{m} \sum_{j=0}^{n} w_{i,j} N_{i,p}(u) N_{j,q}(v) \mathbf{P}_{i,j}}{\sum_{i=0}^{m} \sum_{j=0}^{n} w_{i,j} N_{i,p}(u) N_{j,q}(v)}$$

**Components:**
- **Control Net**: $m \times n$ array of 3D control points
- **Weights**: $m \times n$ array of positive scalar weights
- **Knot Vectors**: Two independent knot vectors (one for $u$, one for $v$)
- **Degrees**: Two degrees ($p$ for $u$ direction, $q$ for $v$ direction)

### Surface Evaluation

**File**: `surface/nurbs_surface.py`

```python
class NURBSSurface:
    def __init__(self, control_net: np.ndarray, weights: np.ndarray,
                 knot_u: np.ndarray, knot_v: np.ndarray,
                 degree_u: int, degree_v: int):
        self.control_net = control_net          # shape: (m, n, 3)
        self.weights = weights                  # shape: (m, n)
        self.knot_u = knot_u
        self.knot_v = knot_v
        self.degree_u = degree_u
        self.degree_v = degree_v
    
    def evaluate_point(self, u: float, v: float) -> np.ndarray:
        """
        Evaluate surface point at (u, v) using tensor-product basis functions.
        
        Steps:
        1. Compute non-zero basis functions in u direction at u
        2. Compute non-zero basis functions in v direction at v
        3. Form 2D basis function products: B[i,j](u,v) = N_i(u) * N_j(v)
        4. Apply weights and compute rational quotient
        5. Return 3D point
        """
    
    def evaluate_derivatives(self, u: float, v: float) -> tuple:
        """
        Compute surface point, partial derivatives ∂S/∂u and ∂S/∂v.
        
        Returns:
            (point, du_partial, dv_partial)
        
        Used for: normal computation, tangent vectors, toolpath orientation
        """
    
    def evaluate_normal(self, u: float, v: float) -> np.ndarray:
        """
        Compute unit surface normal via cross product of partials:
        n = (∂S/∂u × ∂S/∂v) / |∂S/∂u × ∂S/∂v|
        """
    
    def evaluate_grid(self, u_samples: int, v_samples: int) -> np.ndarray:
        """
        Sample surface into regular grid for visualization or toolpath.
        
        Returns:
            grid_points: shape (u_samples, v_samples, 3)
        
        Used for: mesh visualization, toolpath generation
        """
    
    def parameter_ranges(self) -> tuple:
        """Return valid parameter domain (u_min, u_max, v_min, v_max)"""
        # Clamped knot vectors: ranges = (0, 1) after normalization
```

### In-Place Control Point Updates

```python
def update_control_point_inplace(self, i_u: int, i_v: int, 
                                 new_point: np.ndarray, 
                                 new_weight: float) -> None:
    """Update one control point and its weight, preserving surface structure."""
    self.control_net[i_u, i_v] = new_point
    self.weights[i_u, i_v] = new_weight
```

---

## Surface Generation Methods

### 1. Control-Net Demo

**Purpose**: Create a surface from explicit row/column grid dimensions

**Implementation** (`surface/factory.py`):
```python
def create_default_surface(rows: int, cols: int) -> NURBSSurface:
    # Create sinusoidal elevated surface for demonstration
    u = np.linspace(0, 1, cols)
    v = np.linspace(0, 1, rows)
    uu, vv = np.meshgrid(u, v)
    
    # Elevation = sin(π*u) * sin(π*v) for interesting shape
    z = np.sin(np.pi * uu) * np.sin(np.pi * vv)
    
    # Build 3D control net
    control_net = np.stack([uu, vv, z], axis=-1)
    
    # Create uniform knots
    knots_u = generate_clamped_uniform_knot_vector(cols, degree=2)
    knots_v = generate_clamped_uniform_knot_vector(rows, degree=2)
    
    # All weights = 1.0 (non-rational case)
    weights = np.ones((rows, cols))
    
    return NURBSSurface(control_net, weights, knots_u, knots_v, 2, 2)
```

### 2. Curve Lofting

**Purpose**: Generate surface by interpolating through multiple curves

**Mathematical Concept:**
Stack sampled profiles from input curves as rows of a new control net, then create a surface through these profiles.

**Implementation** (`surface/providers/loft.py`):
```python
def loft_surface_from_curves(curves: list, samples_per_curve: int) -> NURBSSurface:
    """
    Generate surface by stacking profiles.
    
    Algorithm:
    1. For each input curve:
       a. Sample into `samples_per_curve` discrete points
       b. Store as one row of control net
    2. Create surface with:
       - Rows = number of curves
       - Columns = samples per curve
       - Degrees determined by curve types
    """
```

**Key Decisions:**
- All curves converted to NURBS for consistency
- Automatic knot vector generation
- Weights initialized to 1.0

### 3. Curve Extrusion

**Purpose**: Sweep a profile curve along a linear direction

**Implementation** (`surface/providers/extrusion.py`):
```python
def extrude_surface_from_curve(curve, direction: np.ndarray, height: float,
                              layer_count: int, 
                              samples_along_curve: int) -> NURBSSurface:
    """
    Linear extrusion of curve along direction vector.
    
    Algorithm:
    1. Normalize direction vector
    2. Sample curve at `samples_along_curve` points
    3. For each layer k in [0, layer_count-1]:
       a. Compute offset along direction: offset = (k/(layer_count-1)) * height
       b. Translate sampled points by offset * direction
       c. Store as row k of control net
    4. Create surface with layer_count rows
    """
```

**Parameters:**
- `direction`: X+, Y+, or Z+ axis (user selectable)
- `height`: Total extrusion distance (mm)
- `layer_count`: Number of profile layers (minimum 2)

### 4. Surface Normal Extrusion (Offset)

**Purpose**: Create offset surface by translating control points along surface normal

**Implementation** (`surface/providers/surface_normal_extrusion.py`):
```python
def extrude_surface_along_center_normal(surface: NURBSSurface, 
                                       distance: float) -> NURBSSurface:
    """
    Offset surface by translating all control points along center normal.
    
    Algorithm:
    1. Evaluate surface normal at parameter (0.5, 0.5) - center point
    2. Normalize to unit vector
    3. For each control point P_ij in control net:
       P'_ij = P_ij + distance * unit_normal
    4. Preserve degrees and knot vectors
    5. Preserve weights
    """
```

### 5. Solid and Shell Body Extrusion

**Purpose**: Generate closed triangulated meshes for CNC machining

**Implementation** (`surface/providers/surface_body_extrusion.py`):
```python
def build_surface_extrusion_mesh(surface: NURBSSurface, distance: float,
                                mode: str, u_samples: int, 
                                v_samples: int) -> tuple:
    """
    Generate triangulated mesh (vertices, triangles) for solid or shell body.
    
    Modes:
    - "solid": Closed bottom + top + side walls (volume enclosed)
    - "shell": Hollow with outer and inner surfaces
    
    Algorithm:
    1. Sample surface into u_samples × v_samples grid
    2. For "solid" mode:
       a. Bottom grid = surface samples
       b. Top grid = surface samples offset +distance along normal
       c. Side walls = triangles connecting boundary loops of top and bottom
    3. For "shell" mode:
       a. Outer grid = surface offset +distance/2
       b. Inner grid = surface offset -distance/2
       c. Side walls = connecting boundary loops
    """
```

**Output:**
- `vertices`: (N, 3) array of 3D coordinates
- `triangles`: (M, 3) array of vertex indices (counter-clockwise winding)

---

## Toolpath Generation

### Zigzag Scanning Algorithm

**Purpose**: Generate CNC-ready zig-zag passes across surface with normal-offset cutter locations

**File**: `machining/toolpath.py`

**Data Structures:**
```python
@dataclass
class ToolpathPoint:
    contact_point: np.ndarray      # (3,) - surface point (tool contact)
    cl_point: np.ndarray           # (3,) - cutter location (offset by radius)
    normal: np.ndarray             # (3,) - surface normal (unit)
    u_value: float                 # parameter
    v_value: float                 # parameter

@dataclass
class ToolpathPass:
    v_parameter: float             # constant v value for this pass
    direction: str                 # "forward" or "reverse"
    points: list[ToolpathPoint]    # ordered points along pass
    link_points: list[ToolpathPoint]  # optional interpolation to next pass
```

### Algorithm Steps

**Step 1: Parameter Space Scanning**
```
For v in [0, 1] with step size Δv:
    If pass_count is even:
        direction = "forward", u traverses 0 → 1
    Else:
        direction = "reverse", u traverses 1 → 0
    
    For u in [0, 1] with step Δu:
        Evaluate surface at (u, v)
        Create ToolpathPoint with:
            - contact_point = S(u, v)
            - normal = n(u, v)
            - cl_point = contact_point + tool_radius * normal
```

**Step 2: Parameter Step Calculation**
```python
def estimate_v_step_from_mm(surface, stepover_mm: float) -> float:
    """
    Convert machine stepover (mm) to surface parameter step.
    
    Algorithm:
    1. Evaluate surface at parameter center (0.5, 0.5)
    2. Sample nearby points to estimate local surface metric
    3. Compute Δv such that |S(u, 0.5+Δv) - S(u, 0.5)| ≈ stepover_mm
    """
```

**Step 3: Collinearity Filtering**
```python
def filter_collinear_toolpath_points(points: list, tolerance_mm: float) -> list:
    """
    Remove redundant nearly-collinear cutter-location points.
    
    Algorithm:
    For each triplet (P[i-1], P[i], P[i+1]):
        deviation = perpendicular distance from P[i] to line(P[i-1], P[i+1])
        if deviation < tolerance_mm:
            Remove P[i] (redundant for G-code)
    
    Benefit: Reduces G-code file size, fewer motion commands
    """
```

### Normal Offset for Cutter Compensation

**Mathematical Basis:**
The cutter location (CL) point is offset from surface contact point by tool radius along normal:

$$CL = S(u, v) + R \cdot \hat{n}(u, v)$$

where:
- $R$ = tool radius (mm)
- $\hat{n}(u, v)$ = unit surface normal at $(u, v)$

**Implementation Details:**
```python
def _stable_normal(surface, u: float, v: float, prior_normal=None) -> np.ndarray:
    """
    Compute surface normal with frame continuity.
    
    Problem: Cross product N = ∂S/∂u × ∂S/∂v is ambiguous up to sign
    Solution: If prior_normal available, choose sign to maintain continuity
    
    Algorithm:
    1. Compute n = ∂S/∂u × ∂S/∂v (unnormalized)
    2. If prior_normal available and dot(n, prior_normal) < 0:
        n = -n  (flip to maintain orientation)
    3. Return n / |n|
    """
```

### Mesh-Based Toolpath Generation

**Purpose**: Generate toolpath from triangulated bodies (shells, solids)

**Algorithm**:
```python
def generate_mesh_zigzag_toolpath(vertices: np.ndarray, triangles: np.ndarray,
                                  stepover_mm: float, tool_radius_mm: float,
                                  ...) -> list[ToolpathPass]:
    """
    Generate passes by ray-casting through mesh.
    
    Algorithm:
    1. For each scanline position (step by stepover_mm):
        a. Cast vertical ray at (x, y)
        b. Find all ray-triangle intersections
        c. For each hit:
            - Contact point = intersection
            - Normal = triangle normal (optionally flipped)
            - CL point = contact + radius * normal
    2. Group intersections into passes based on mesh topology
    """
```

---

## G-Code Generation and Export

### G-Code Structure

**File**: `gcode/exporter.py`

**Data Class**:
```python
@dataclass
class GCodeSettings:
    safe_z_mm: float                      # safe Z height for rapid moves
    feed_rate_mm_per_min: float           # cutting feed rate
    plunge_rate_mm_per_min: float         # Z-axis plunge rate
    spindle_rpm: int                      # spindle speed
    link_mode: str                        # "retract" (rapid) or "surface" (interpolated)
```

### G-Code Output Format

**Header Section:**
```gcode
G21                    ; Metric units (mm)
G90                    ; Absolute positioning
S5000 M3               ; Spindle speed 5000 RPM, spindle ON
M8                     ; Coolant ON
```

**Per-Pass Motion:**
```gcode
; --- Pass 1 (direction: forward) ---
G0 X... Y... Z...      ; Rapid to start position at safe height
G1 Z...                ; Plunge to first point (feed_rate)
G1 X... Y... Z... F... ; Feed moves through pass points
G1 X... Y... Z...      ; Continue feeding across pass

; Link to next pass:
; - If link_mode="retract":
G0 Z...                ; Rapid to safe Z
G0 X... Y...           ; Rapid to next pass start
G1 Z...                ; Plunge

; - If link_mode="surface":
G1 X... Y... Z... F... ; Interpolated move on surface (slower but smoother)
```

**Footer Section:**
```gcode
G0 Z...                ; Rapid to safe height
M9                     ; Coolant OFF
M5                     ; Spindle OFF
M30                    ; Program END
```

### G-Code Generation Algorithm

```python
def generate_gcode_program(passes: list[ToolpathPass], 
                          settings: GCodeSettings) -> str:
    """
    Convert ToolpathPass objects to G-code text.
    
    Algorithm:
    1. Write header (units, abs positioning, spindle commands)
    2. For each pass:
        a. Write rapid move to pass start at safe Z
        b. Write plunge (G1 at plunge_rate)
        c. For each ToolpathPoint in pass:
            Write: G1 X... Y... Z... F...
        d. Write linking move to next pass:
            - If link_mode="retract": rapid to safe Z, rapid XY, plunge
            - If link_mode="surface": interpolate through link_points
    3. Write footer (safe Z, spindle/coolant OFF, M30)
    """
```

### Format Details

**Coordinate Formatting** (`utils/geometry.py`):
```python
def format_xyz(x: float, y: float, z: float) -> str:
    """Return: 'X{x:.3f} Y{y:.3f} Z{z:.3f}'"""
    # Precision: 0.001 mm (3 decimal places)
```

**Typical Command:**
```gcode
G1 X45.123 Y-32.456 Z10.000 F300
```

---

## GUI Architecture

### Main Application Window

**File**: `gui/app.py` (2000+ lines)

**Class**: `NURBSSurfaceMainWindow(QtWidgets.QMainWindow)`

**Key Responsibilities:**
1. **Curve Management**: Create, edit, delete curves
2. **Surface Generation**: Generate surfaces from various sources
3. **Surface Editing**: Manipulate control points via GUI or viewport
4. **Toolpath Generation**: Compute zig-zag passes
5. **G-Code Export**: Write machine-ready code to file
6. **Rendering**: Visualize curves, surfaces, toolpaths, control points
7. **State Management**: Track active geometry, selections, UI state

### Major GUI Sections (Collapsible Groups)

#### 1. Curves Panel
**Controls:**
- Curve type selector (Bezier, B-spline, NURBS)
- Degree spinner (for B-spline/NURBS)
- Knot mode selector (Uniform, Non-uniform)
- Plane lock checkbox (constrain to Z=0)
- Interaction mode selector (Select, Add Points, Drag Handles)
- Curve list widget (multi-select)
- Control point list for active curve
- Point coordinate editors (X, Y, Z, Weight for NURBS)

**Workflow:**
1. Select curve type and parameters
2. Click "Create Curve" to instantiate with default points
3. Choose interaction mode:
   - **Select**: Click curves in viewport to select
   - **Add Points**: Click viewport to place new control points
   - **Drag Handles**: Click-drag control point handles for interactive editing
4. Edit points via:
   - XYZ coordinate spin boxes
   - Dragging sphere widgets in viewport
   - Apply button to commit changes

#### 2. Control Point Editor Panel
**Controls:**
- Row/Column index selectors
- X, Y, Z coordinate spinners
- Weight spinner (for NURBS surfaces)
- Drag sensitivity adjustment
- Show control net checkbox
- Apply Point Update button

**State:**
Disabled until surface exists. Allows editing of any control point in active surface.

#### 3. Surface Generation Panel
**Controls:**
- Source mode selector:
  - "Control Net Demo": Generate from explicit grid size
  - "Loft From Curves": Interpolate through curves
  - "Extrude Active Curve": Linear sweep
- Conditional inputs based on source mode:
  - Control-net: Rows, Columns spinners
  - Loft: Curve sampling rate
  - Extrude: Axis selector, Height, Layer count

**Workflow:**
1. Select source mode
2. Adjust relevant parameters
3. Click "Generate Surface" → creates NURBSSurface
4. Surface becomes active (displayed in viewport)

#### 4. Surface Operations Panel
**Controls:**
- Active surface selector (dropdown)
- Extrusion mode selector (Offset, Shell, Solid)
- Extrude distance spinner
- Replace checkbox (for offset mode)
- Extrude button

**Workflow:**
1. Select active surface
2. Choose extrusion mode
3. Set distance
4. Click "Extrude" → generates derived surface or mesh

#### 5. Toolpath Generation Panel
**Controls:**
- Toolpath source selector (Surface, Mesh)
- Scan axis selector (Auto, U/X, V/Y)
- Stepover (mm)
- Tool radius (mm)
- Chord tolerance (mm) for filtering
- Mesh envelope mode (Top, Full)
- Link passes checkbox
- Generate Toolpath button

**Workflow:**
1. Ensure surface exists
2. Set machine parameters (stepover, tool radius)
3. Choose source (parametric surface or mesh)
4. Click "Generate Toolpath" → computes ToolpathPass list
5. Passes displayed as colored scanlines in viewport

#### 6. G-Code Export Panel
**Controls:**
- Feed rate (mm/min)
- Plunge rate (mm/min)
- Safe Z (mm)
- Spindle RPM
- Export G-code button

**Workflow:**
1. Ensure toolpath passes exist
2. Set CNC machine parameters
3. Click "Export G-code"
4. File dialog → save to .nc file

### Section Management

**LRU Section Collapsing:**
```python
max_expanded_sections = 3  # Only 3 sections expanded at once

When user expands a new section:
    if expanded_sections.count > 3:
        Collapse least-recently-used expanded section
```

This prevents UI overflow when window is small.

### Viewport Interaction

**3D Visualization**: PyVista plotter embedded in central widget

**Rendered Elements:**
1. **Reference Plane**: XY grid at Z=0 (permanent)
2. **Surfaces**: Tessellated mesh with edges
3. **Control Net**: Wireframe of control point connections
4. **Control Point Widgets**: Draggable colored spheres
5. **Curves**: Polylines with control hulls and point markers
6. **Toolpath Overlays**: Colored scanlines with pass points
7. **Derived Bodies**: Semi-transparent triangulated meshes

**Interaction Modes:**
- **Rotate**: Middle mouse button drag
- **Pan**: Right mouse button drag
- **Zoom**: Mouse scroll wheel
- **Select Curve**: Left click on rendered curve (Select mode)
- **Add Point**: Left click on viewport plane (Add Points mode)
- **Drag Control Point**: Left click-drag on sphere widget (Drag mode)

### State Synchronization

**Problem**: Multiple UI controls must stay in sync (list widget, spin boxes, viewport widgets)

**Solution**: Centralized state synchronization methods:
```python
_sync_ui_enabled_state()          # Enable/disable controls based on workflow state
_refresh_curve_list_widget()      # Update list from curves array
_sync_curve_point_editor_values() # Update spin boxes from selected point
_refresh_scene()                  # Redraw all viewport elements
_sync_widget_positions_from_control_net()  # Move sphere widgets after edits
```

---

## Rendering Pipeline

### Scene Refresh Workflow

```python
def _refresh_scene(self, reset_camera=True, recompute_surface=True, 
                   redraw_toolpath=True):
    """
    Comprehensive scene update with selective re-computation.
    
    Steps:
    1. If reset_camera: fit all geometry in viewport
    2. If recompute_surface: rebuild surface mesh from control net
    3. If redraw_toolpath: redraw toolpath actors
    4. Draw curves, control nets, control widgets
    5. Draw toolpath overlays
    6. Draw derived bodies
    7. Render
    """
```

### Surface Mesh Generation

```python
def _refresh_surface_mesh():
    """Generate tessellated mesh from active surface for visualization."""
    
    # Sample surface into grid
    grid = surface.evaluate_grid(45, 45)  # Config: DEFAULT_SURFACE_SAMPLES
    
    # Create PyVista mesh
    vertices, faces = _grid_to_polydata(grid)
    
    # Render with edges
    actor = plotter.add_mesh(
        mesh, 
        color="#94a3b8",
        show_edges=True,
        edge_color="#0f172a"
    )
```

### Toolpath Visualization

```python
def _draw_toolpath_overlays():
    """Render toolpath passes as colored scanlines."""
    
    for pass_index, pass_obj in enumerate(generated_passes):
        color = _get_pass_color(pass_index)
        
        for point in pass_obj.points:
            # Draw CL point
            actor = plotter.add_mesh(
                sphere_at(point.cl_point),
                color=color,
                opacity=0.8
            )
            
            # Draw normal vector
            actor = plotter.add_mesh(
                line(point.contact_point, point.cl_point),
                color=color
            )
```

---

## Data Flow: Complete Workflow

### Curve → Surface → Toolpath → G-Code

**Example: Loft Surface from Two Curves**

```
1. User Interface
   - Creates Curve 1 (4 control points)
   - Creates Curve 2 (4 control points)
   - Selects "Loft From Curves" mode
   
2. Surface Generation (loft_surface_from_curves)
   - Sample Curve 1 at 40 points → row 0 of control net
   - Sample Curve 2 at 40 points → row 1 of control net
   - Create 2×40 control net
   - Generate uniform knots: degree 1 in v (2 rows), degree 3 in u (40 cols)
   - Return NURBSSurface
   
3. GUI Updates
   - Activates surface in scene
   - Disables curve controls, enables surface controls
   - Caches surface mesh for viewport
   
4. Viewport Rendering
   - Evaluates surface at 45×45 parameter grid
   - Generates 45×45 tessellated mesh
   - Renders with control net wireframe
   - Displays control point sphere widgets

5. Toolpath Generation (generate_zigzag_toolpath)
   - Scans v from 0 to 1 with step 0.1 mm → ~0.05 in parameter space
   - For each v:
       - u from 0→1 (even passes) or 1→0 (odd passes)
       - Sample at u step ~0.05
       - For each (u,v):
           * Evaluate surface: S(u,v)
           * Compute normal: n(u,v)
           * Compute CL: S(u,v) + 2mm * n(u,v)  [tool_radius=2mm]
           * Create ToolpathPoint
   - Filter collinear points: remove redundant CL points
   - Create ToolpathPass objects with direction and link points
   
6. Viewport Visualization
   - Render each pass as colored scanline
   - Display CL points as small spheres
   - Show surface normals as vectors

7. G-Code Export (generate_gcode_program)
   - Header: G21, G90, S5000 M3, M8
   - For each pass:
       - Rapid to start: G0 X... Y... Z...
       - Plunge: G1 Z... F300  [plunge_rate]
       - Feed through pass: G1 X... Y... Z... F300  [feed_rate]
       - Link to next pass: G0 Z... (retract mode)
   - Footer: G0 Z..., M9, M5, M30
   
8. File Export
   - Write G-code text to .nc file
   - Status message: "Exported to /path/file.nc"
```

---

## Mathematical Concepts and Algorithms

### Cox-de Boor Basis Function Evaluation

**Problem**: Efficiently compute all non-zero B-spline basis functions at parameter $u$

**Algorithm** (`core/bspline.py`):
```
find_span(u) → index k where U[k] ≤ u < U[k+1]

basis_functions(u, k, p):
    N[k, 0] = 1  (the span containing u)
    All other N[i, 0] = 0
    
    for degree d from 1 to p:
        for each i that could have N[i, d] ≠ 0:
            left_factor = (u - U[i]) / (U[i+d] - U[i])
            right_factor = (U[i+d+1] - u) / (U[i+d+1] - U[i+1])
            N[i, d] = left_factor * N[i, d-1] + right_factor * N[i+1, d-1]
    
    return N[*, p]
```

**Complexity**: $O(p^2)$ where $p$ is degree (typically 3-5)

**Stability**: Recursive formula avoids numerical catastrophe through local computation

### Surface Normal Computation

**From Partial Derivatives:**
$$\mathbf{n}(u, v) = \frac{\frac{\partial S}{\partial u} \times \frac{\partial S}{\partial v}}{|\frac{\partial S}{\partial u} \times \frac{\partial S}{\partial v}|}$$

**Steps:**
1. Evaluate $\frac{\partial S}{\partial u}$ using derivative basis functions
2. Evaluate $\frac{\partial S}{\partial v}$ using derivative basis functions
3. Compute cross product: tangent vectors define plane, normal is perpendicular
4. Normalize to unit length

**Implementation Considerations:**
- **Sign Ambiguity**: Cross product can flip sign; solved by tracking prior normal
- **Degenerate Cases**: If partials are parallel (flat region), normal is undefined
  - Handled by fallback to grid-based estimation

### Collinearity Detection Algorithm

**Purpose**: Identify redundant (nearly-collinear) toolpath points

**Algorithm**:
```
For each triplet (P[i-1], P[i], P[i+1]):
    line_vector = P[i+1] - P[i-1]
    from_start = P[i] - P[i-1]
    
    // Project from_start onto line vector
    parameter_t = dot(from_start, line_vector) / dot(line_vector, line_vector)
    
    // Find closest point on infinite line
    closest_on_line = P[i-1] + parameter_t * line_vector
    
    // Perpendicular distance
    deviation = |P[i] - closest_on_line|
    
    if deviation < COLLINEAR_TOLERANCE_MM:
        Mark P[i] for removal
```

**Complexity**: $O(n)$ single pass

**Benefit**: Reduces G-code file size by 20-40% typically, without affecting surface finish quality

---

## Configuration and Constants

**File**: `config.py`

```python
# Numerical tolerance
EPSILON = 1e-10

# Collinearity filtering threshold (mm)
COLLINEAR_TOLERANCE_MM = 0.05

# Default machine parameters (mm, RPM, mm/min)
DEFAULT_TOOL_RADIUS_MM = 2.0
DEFAULT_STEPOVER_MM = 1.0
DEFAULT_FEED_RATE_MM_PER_MIN = 300.0
DEFAULT_PLUNGE_RATE_MM_PER_MIN = 150.0
DEFAULT_SAFE_Z_MM = 10.0
DEFAULT_SPINDLE_RPM = 5000

# Surface sampling resolution (points per direction)
DEFAULT_SURFACE_SAMPLES_U = 45
DEFAULT_SURFACE_SAMPLES_V = 45

# GUI widget sizing
DEFAULT_WIDGET_RADIUS_MM = 2.0
SELECTED_WIDGET_RADIUS_MM = 3.0
CURVE_DRAG_PICK_RADIUS_MM = 5.0

# GUI animation
DRAG_UPDATE_INTERVAL_SEC = 0.016  # ~60 FPS
DRAG_SENSITIVITY = 1.0            # 1.0 = 1:1 mouse movement
```

---

## Dependencies and Requirements

**File**: `requirements.txt`

```
numpy==1.24.3              # Numerical computation
PyQt5==5.15.7              # GUI framework
pyvista==0.37.0            # 3D visualization (VTK wrapper)
pytest==7.3.1              # Unit testing
```

**Key Libraries:**
- **NumPy**: Array operations, linear algebra, matrix computations
- **PyQt5**: Widget framework, signals/slots, event handling
- **PyVista**: Wraps VTK for mesh visualization, rendering, interaction
- **Pytest**: Test discovery and execution (optional, for development)

---

## Application Entry Points

**File**: `main.py`

```python
def run_gui():
    """Launch interactive PyQt5 GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    window = NURBSSurfaceMainWindow()
    window.show()
    sys.exit(app.exec())

def run_demo_export():
    """Headless demonstration: create surface and export G-code."""
    # Create default surface
    surface = create_default_surface(rows=5, cols=5)
    
    # Generate toolpath
    passes = generate_zigzag_toolpath(surface, stepover_mm=1.0, 
                                     tool_radius_mm=2.0)
    
    # Export G-code
    settings = GCodeSettings(safe_z_mm=10.0, feed_rate_mm_per_min=300.0, ...)
    gcode_text = generate_gcode_program(passes, settings)
    write_gcode_file("output.nc", gcode_text)
```

**Usage:**
```bash
python main.py              # Launch GUI
python -m pytest tests/     # Run unit tests
```

---

## Summary of Key Workflows

### Creating and Editing Curves

1. **Create**: Type + Degree + Knot Mode → Click "Create Curve"
2. **Edit Points**: 
   - Via List: Select in list, edit XYZ spinners, click "Apply"
   - Via Viewport: Switch to "Drag Handles", click-drag spheres
3. **Delete**: Select curve in list → Click "Delete Curve"

### Generating Surfaces

1. **From Grid**: Set Rows/Cols → Click "Generate Surface"
2. **From Curves**: Create ≥2 curves, Loft mode → Click "Generate Surface"
3. **From Extrusion**: Select active curve, Extrude mode, set height/layers → Click "Generate Surface"

### Creating Toolpaths

1. **Generate Passes**: Set Stepover, Tool Radius → Click "Generate Zigzag Toolpath"
2. **View in Viewport**: Passes render as colored scanlines
3. **Export G-Code**: Set Feed/Plunge/Safe Z → Click "Export G-Code" → Save file

### Advanced: Mesh-Based Machining

1. **Create Solid/Shell**: Select surface, Extrusion mode → Click "Extrude Selected Surface"
2. **Generate from Mesh**: Toolpath source → "Derived Body Mesh", Envelope mode → Generate
3. **Inspect**: Viewport shows raycast intersections, normal vectors

---

## Performance Considerations

**Surface Evaluation**: O(degree² × degree²) per point ~ O(1) for fixed low degree
**Grid Sampling**: O(u_samples × v_samples × degree⁴) ~= 5-10ms for 45×45 grid
**Toolpath Generation**: O(v_samples × u_samples) ~= 50-100ms for typical stepover
**G-Code Export**: O(total_points) ~= <1ms text generation
**Viewport Rendering**: Bottleneck is VTK rasterization, not computation

**Optimization Strategies:**
- Cache surface mesh between edits
- Lazy re-computation (only when geometry changes)
- Frustum culling in viewport (handled by PyVista/VTK)
- Collinearity filtering reduces final G-code size

---

## Limitations and Future Work

**Current Limitations:**
1. Single-tool CNC (no tool changes)
2. Planar Z-level toolpath only (no 5-axis)
3. No collision detection
4. No tool path optimization (nearest-neighbor ordering)
5. Limited surface types (no ruled surfaces, sweeps)

**Potential Enhancements:**
1. Multi-tool library with automatic selection
2. 5-axis simultaneous CNC capability
3. Collision avoidance with boundaries
4. Adaptive feedrate based on surface curvature
5. Finishing passes with reduced stepover
6. STL/STEP file import/export
7. CAM simulation and verification

---

## Testing

**File**: `tests/`

**Test Coverage:**
- Curve evaluation and derivatives (Bezier, B-spline, NURBS)
- Surface evaluation grid sampling
- Toolpath generation algorithms
- G-code formatting and export

**Running Tests:**
```bash
pytest tests/ -v
pytest tests/test_math_and_export.py::test_nurbs_circle_exact
pytest tests/ --cov=core --cov=surface --cov=machining
```

---

## Conclusion

The NURBS Surface Modeller integrates advanced computational geometry (tensor-product NURBS), interactive 3D visualization (PyQt5 + PyVista), and CNC machining algorithms into a cohesive desktop application. The modular architecture separates mathematical computation (core, surface) from GUI logic (gui) and machine-specific operations (machining, gcode), enabling extensibility and maintenance. The complete workflow from curve creation through G-code export provides practitioners with a practical CAD/CAM tool for surface-based manufacturing.
