"""PyQt5 + PyVista based GUI application for NURBS surface modeling."""

# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from core.curve_validation import build_knots_for_mode
from core.curves import BSplineCurve, BezierCurve, CurveObject, NURBSCurve
from config import (
    COLLINEAR_TOLERANCE_MM,
    DEFAULT_CONTROL_POINT_WIDGET_RADIUS,
    DEFAULT_DRAG_SENSITIVITY,
    DEFAULT_DRAG_UPDATE_INTERVAL_SEC,
    DEFAULT_FEED_RATE_MM_PER_MIN,
    DEFAULT_PLUNGE_RATE_MM_PER_MIN,
    DEFAULT_SELECTED_POINT_WIDGET_RADIUS,
    DEFAULT_SAFE_Z_MM,
    DEFAULT_SPINDLE_RPM,
    DEFAULT_STEPOVER_MM,
    DEFAULT_SURFACE_SAMPLES_U,
    DEFAULT_SURFACE_SAMPLES_V,
    DEFAULT_TOOL_RADIUS_MM,
)
from gcode.exporter import (
    GCodeSettings,
    LINK_MODE_RETRACT,
    LINK_MODE_SURFACE,
    generate_gcode_program,
    write_gcode_file,
)
from machining.toolpath import ToolpathPass, generate_zigzag_toolpath
from surface.factory import create_default_surface
from surface.providers.extrusion import extrude_surface_from_curve
from surface.providers.loft import loft_surface_from_curves
from surface.providers.surface_body_extrusion import build_surface_extrusion_mesh
from surface.providers.surface_normal_extrusion import extrude_surface_along_center_normal

GUI_IMPORT_ERROR: Exception | None = None

try:
    from PyQt5 import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    import pyvista as pv
except Exception as import_error:  # pragma: no cover - import errors are runtime-environment specific.
    GUI_IMPORT_ERROR = import_error
    QtCore = None

    class _FallbackQtWidgets:
        """Fallback container that provides minimal QtWidgets symbols.

        Inputs:
            None.

        Outputs:
            Namespace-like object with placeholder QMainWindow.

        Behavior:
            Allows this module to be imported in non-GUI environments while
            run_gui_app raises a clear dependency error before window creation.
        """

        class QMainWindow:
            """Placeholder base class used when Qt is unavailable.

            Inputs:
                None.

            Outputs:
                None.

            Behavior:
                Acts only as a class-definition stub and is never instantiated
                in valid runtime flow because run_gui_app exits early.
            """

            pass

    QtWidgets = _FallbackQtWidgets
    QtInteractor = None
    pv = None


def sanitize_curve_selection_state(
    valid_curve_ids: list[int],
    selected_curve_ids: set[int],
    active_curve_id: int | None,
    last_interacted_curve_id: int | None,
) -> tuple[set[int], int | None, int | None]:
    """Drop invalid selection references and return deterministic active state."""
    valid_order = [int(curve_id) for curve_id in valid_curve_ids]
    valid_set = set(valid_order)
    sanitized_selected = {int(curve_id) for curve_id in selected_curve_ids if int(curve_id) in valid_set}

    sanitized_last = (
        int(last_interacted_curve_id)
        if last_interacted_curve_id is not None and int(last_interacted_curve_id) in valid_set
        else None
    )

    if not sanitized_selected:
        return set(), None, None

    sanitized_active = (
        int(active_curve_id)
        if active_curve_id is not None and int(active_curve_id) in sanitized_selected
        else None
    )
    if sanitized_active is None:
        if sanitized_last is not None and sanitized_last in sanitized_selected:
            sanitized_active = sanitized_last
        else:
            sanitized_active = next(
                (curve_id for curve_id in valid_order if curve_id in sanitized_selected),
                None,
            )

    return sanitized_selected, sanitized_active, sanitized_last


def reduce_curve_selection_state(
    valid_curve_ids: list[int],
    selected_curve_ids: set[int],
    active_curve_id: int | None,
    last_interacted_curve_id: int | None,
    clicked_curve_id: int | None,
    shift_pressed: bool,
) -> tuple[set[int], int | None, int | None]:
    """Apply one click action to curve-selection state."""
    valid_set = {int(curve_id) for curve_id in valid_curve_ids}

    if clicked_curve_id is None or int(clicked_curve_id) not in valid_set:
        return set(), None, None

    clicked_id = int(clicked_curve_id)
    if shift_pressed:
        updated_selected = set(int(curve_id) for curve_id in selected_curve_ids if int(curve_id) in valid_set)
        if clicked_id in updated_selected:
            updated_selected.remove(clicked_id)
        else:
            updated_selected.add(clicked_id)
        proposed_active = clicked_id if clicked_id in updated_selected else active_curve_id
        updated_last = clicked_id
    else:
        updated_selected = {clicked_id}
        proposed_active = clicked_id
        updated_last = clicked_id

    return sanitize_curve_selection_state(
        valid_curve_ids=valid_curve_ids,
        selected_curve_ids=updated_selected,
        active_curve_id=proposed_active,
        last_interacted_curve_id=updated_last,
    )


class NURBSSurfaceMainWindow(QtWidgets.QMainWindow):
    """Main desktop window that controls rendering and export actions.

    Inputs:
        None.

    Outputs:
        Configured Qt window instance.

    Behavior:
        Owns editable surface state, displays geometry, generates toolpaths, and
        exports G-code through side-panel controls.
    """

    def __init__(self) -> None:
        """Initialize UI state, surface model, and plot actors.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Starts with an empty modeling scene, builds widgets, and draws a
            permanent CAD-like reference plane for orientation.
        """
        super().__init__()
        self.setWindowTitle("NURBS Surface Modeller")
        self.resize(1400, 850)

        self.surface = None
        self.surface_entries: list[dict[str, object]] = []
        self.active_surface_id: int | None = None
        self._next_surface_id = 1
        self._rendered_surface_id: int | None = None
        self._derived_body_entries: list[dict[str, object]] = []
        self._derived_body_actors: list[object] = []
        self.generated_passes: list[ToolpathPass] = []

        self.curves: list[CurveObject] = []
        self.curve_ids: list[int] = []
        self.curve_knot_modes: list[str] = []
        self.selected_curve_ids: set[int] = set()
        self.active_curve_id: int | None = None
        self._last_interacted_curve_id: int | None = None
        self._next_curve_id = 1
        self.active_curve_index: int | None = None
        self.active_curve_point_index: int | None = None
        self._curve_interaction_mode = "select"
        self._is_curve_dragging = False
        self._curve_drag_point_index: int | None = None
        self._last_curve_drag_update_timestamp = 0.0
        self._curve_drag_pick_radius_mm = 12.0

        self._reference_plane_actor = None
        self._reference_plane_mesh = None
        self._reference_plane_extent = 0.0

        self._surface_actor = None
        self._surface_mesh = None
        self._control_net_actor = None
        self._control_net_mesh = None
        self._control_point_widgets: list[object] = []
        self._curve_point_widgets: list[object] = []
        self._toolpath_actors: list[object] = []
        self._curve_actors: list[object] = []
        self._curve_hull_actors: list[object] = []
        self._curve_point_actors: list[object] = []
        self._curve_actor_to_curve_id: dict[int, int] = {}
        self._is_syncing_curve_selection_ui = False
        self._major_sections: list[object] = []
        self._section_lru: list[object] = []
        self._max_expanded_sections = 3
        self._is_updating_section_state = False

        self._selected_control_flat_index: int | None = None
        self._selected_curve_point_widget_index: int | None = None
        self._active_widget_layer = "none"
        self._is_syncing_widgets = False
        self._is_syncing_curve_widgets = False
        self._control_net_visible = True
        self._drag_sensitivity = DEFAULT_DRAG_SENSITIVITY
        self._drag_update_interval_sec = DEFAULT_DRAG_UPDATE_INTERVAL_SEC
        self._last_drag_update_timestamp = 0.0

        self._default_widget_radius = DEFAULT_CONTROL_POINT_WIDGET_RADIUS
        self._selected_widget_radius = DEFAULT_SELECTED_POINT_WIDGET_RADIUS
        self._default_widget_color = (0.82, 0.29, 0.36)
        self._selected_widget_color = (0.96, 0.74, 0.32)

        self._build_ui()
        self._refresh_scene(reset_camera=True)
        self.status_label.setText("Ready: create control input and click Generate Surface")

    def _build_ui(self) -> None:
        """Build top-level Qt layout and connect interaction callbacks.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Creates a horizontal split with a PyVista viewport on the left and
            parameter controls on the right.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        self.plotter = QtInteractor(central)
        root_layout.addWidget(self.plotter.interactor, stretch=3)
        self.plotter.set_background("#f8fafc")
        self.plotter.show_axes()

        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setSpacing(8)
        root_layout.addWidget(panel, stretch=1)

        self._build_curve_group(panel_layout)
        self._build_control_point_group(panel_layout)
        self._build_surface_generation_group(panel_layout)
        self._build_surface_operations_group(panel_layout)
        self._build_toolpath_group(panel_layout)
        self._build_export_group(panel_layout)

        self._register_viewport_curve_interactions()

        self.status_label = QtWidgets.QLabel("Ready")
        panel_layout.addWidget(self.status_label)
        panel_layout.addStretch(1)
        self._enforce_max_expanded_sections(None)
        self._sync_ui_enabled_state()

    def _set_section_collapsed_state(self, section_group: object, expanded: bool) -> None:
        """Apply expanded/collapsed visual state to one major section group."""
        if QtWidgets is None:
            return

        title_height = max(28, int(section_group.fontMetrics().height()) + 12)
        if expanded:
            section_group.setMaximumHeight(16777215)
            section_group.setMinimumHeight(0)
        else:
            section_group.setMaximumHeight(title_height)
            section_group.setMinimumHeight(title_height)

    def _register_major_section(self, section_group: object, expanded_by_default: bool) -> None:
        """Register one major GUI section as collapsible and LRU-managed."""
        if QtWidgets is None:
            return

        section_group.setCheckable(True)
        section_group.setChecked(bool(expanded_by_default))
        self._set_section_collapsed_state(section_group, expanded=bool(expanded_by_default))
        section_group.toggled.connect(
            lambda checked, group=section_group: self._on_major_section_toggled(group, bool(checked))
        )

        self._major_sections.append(section_group)
        if expanded_by_default:
            self._section_lru.append(section_group)

    def _on_major_section_toggled(self, section_group: object, expanded: bool) -> None:
        """Track section usage and enforce maximum simultaneous expanded sections."""
        if self._is_updating_section_state:
            return

        self._set_section_collapsed_state(section_group, expanded=bool(expanded))
        if expanded:
            if section_group in self._section_lru:
                self._section_lru.remove(section_group)
            self._section_lru.append(section_group)
        else:
            self._section_lru = [group for group in self._section_lru if group is not section_group]

        self._enforce_max_expanded_sections(section_group if expanded else None)

    def _enforce_max_expanded_sections(self, newly_expanded_group: object | None) -> None:
        """Collapse least-recently-used expanded sections until limit is respected."""
        expanded_groups = [group for group in self._major_sections if group.isChecked()]
        if len(expanded_groups) <= self._max_expanded_sections:
            return

        focus_widget = None
        if QtWidgets is not None and hasattr(QtWidgets, "QApplication"):
            focus_widget = QtWidgets.QApplication.focusWidget()

        self._is_updating_section_state = True
        try:
            while len([group for group in self._major_sections if group.isChecked()]) > self._max_expanded_sections:
                candidate = None
                for group in self._section_lru:
                    if not group.isChecked():
                        continue
                    if newly_expanded_group is not None and group is newly_expanded_group:
                        continue
                    if focus_widget is not None and group.isAncestorOf(focus_widget):
                        continue
                    candidate = group
                    break

                if candidate is None:
                    for group in self._section_lru:
                        if not group.isChecked():
                            continue
                        if newly_expanded_group is not None and group is newly_expanded_group:
                            continue
                        candidate = group
                        break

                if candidate is None:
                    break

                candidate.setChecked(False)
                self._set_section_collapsed_state(candidate, expanded=False)
                self._section_lru = [group for group in self._section_lru if group is not candidate]
        finally:
            self._is_updating_section_state = False

    def _has_active_surface(self) -> bool:
        """Return True when an editable surface is currently available.

        Inputs:
            None.

        Outputs:
            Boolean indicating whether surface-dependent tools can run.

        Behavior:
            Centralizes surface existence checks for button and panel gating.
        """
        return self.surface is not None

    def _has_valid_surface_input(self) -> bool:
        """Return True when pending surface input widgets contain valid values.

        Inputs:
            None.

        Outputs:
            Boolean indicating whether surface generation is allowed.

        Behavior:
            Validates source-specific surface inputs from the generation panel.
        """
        source_mode = self.surface_source_combo.currentData()
        if source_mode == "control-net":
            return int(self.surface_rows_spin.value()) >= 2 and int(self.surface_cols_spin.value()) >= 2
        if source_mode == "curve-loft":
            return len(self.curves) >= 2
        if source_mode == "curve-extrude":
            return (
                self._get_active_curve() is not None
                and float(self.surface_extrude_height_spin.value()) > 0.0
                and int(self.surface_extrude_layers_spin.value()) >= 2
            )
        return False

    def _sync_ui_enabled_state(self) -> None:
        """Synchronize panel/button enabled states with current geometry state.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Keeps controls consistent with workflow: generate surface first,
            then edit/generate toolpath, and export only after passes exist.
        """
        required_widgets = (
            "control_point_group",
            "surface_source_combo",
            "surface_rows_spin",
            "surface_cols_spin",
            "surface_curve_samples_spin",
            "generate_surface_button",
            "generate_toolpath_button",
            "export_gcode_button",
            "create_curve_button",
            "delete_curve_button",
            "add_curve_point_button",
            "delete_curve_point_button",
            "apply_curve_point_button",
            "curve_point_weight_spin",
            "curve_interaction_mode_combo",
            "surface_extrude_axis_combo",
            "surface_extrude_height_spin",
            "surface_extrude_layers_spin",
            "surface_selector_combo",
            "surface_extrusion_mode_combo",
            "surface_normal_extrude_distance_spin",
            "surface_normal_replace_checkbox",
            "extrude_selected_surface_button",
        )
        if not all(hasattr(self, name) for name in required_widgets):
            return

        has_surface = self._has_active_surface()
        has_passes = bool(self.generated_passes)
        selected_curve = self._get_active_curve()
        has_curve_selection = selected_curve is not None
        has_curve_point_selection = (
            has_curve_selection
            and self.active_curve_point_index is not None
            and 0 <= int(self.active_curve_point_index) < self._curve_point_count(selected_curve)
        )

        selected_is_nurbs = isinstance(selected_curve, NURBSCurve)

        source_mode = self.surface_source_combo.currentData()
        use_control_net_source = source_mode == "control-net"
        use_extrude_source = source_mode == "curve-extrude"

        self.surface_rows_spin.setEnabled(use_control_net_source)
        self.surface_cols_spin.setEnabled(use_control_net_source)
        self.surface_curve_samples_spin.setEnabled(not use_control_net_source)
        self.surface_extrude_axis_combo.setEnabled(use_extrude_source)
        self.surface_extrude_height_spin.setEnabled(use_extrude_source)
        self.surface_extrude_layers_spin.setEnabled(use_extrude_source)
        extrusion_mode = str(self.surface_extrusion_mode_combo.currentData())
        self.surface_selector_combo.setEnabled(bool(self.surface_entries))
        self.surface_extrusion_mode_combo.setEnabled(has_surface)
        self.surface_normal_extrude_distance_spin.setEnabled(has_surface)
        self.surface_normal_replace_checkbox.setEnabled(has_surface and extrusion_mode == "offset")
        self.extrude_selected_surface_button.setEnabled(
            has_surface and float(self.surface_normal_extrude_distance_spin.value()) > 0.0
        )

        self.control_point_group.setEnabled(has_surface)
        self.generate_surface_button.setEnabled(self._has_valid_surface_input())
        self.generate_toolpath_button.setEnabled(has_surface)
        self.export_gcode_button.setEnabled(has_surface and has_passes)

        self.create_curve_button.setEnabled(True)
        self.delete_curve_button.setEnabled(has_curve_selection)
        self.add_curve_point_button.setEnabled(has_curve_selection)
        self.delete_curve_point_button.setEnabled(has_curve_point_selection)
        self.apply_curve_point_button.setEnabled(has_curve_point_selection)
        self.curve_point_weight_spin.setEnabled(has_curve_point_selection and selected_is_nurbs)

    def _configure_control_editor_ranges(self) -> None:
        """Configure row/column editor ranges from active surface dimensions.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Sets safe defaults when no surface exists and updates ranges after
            explicit surface generation.
        """
        if not self._has_active_surface():
            for widget in (self.row_spin, self.col_spin):
                blocked = widget.blockSignals(True)
                widget.setRange(0, 0)
                widget.setValue(0)
                widget.blockSignals(blocked)
            return

        max_u, max_v = np.array(self.surface.control_net.shape[:2]) - 1

        for widget, max_value in ((self.row_spin, int(max_u)), (self.col_spin, int(max_v))):
            blocked = widget.blockSignals(True)
            widget.setRange(0, max_value)
            widget.setValue(min(int(widget.value()), max_value))
            widget.blockSignals(blocked)

    def _build_curve_group(self, parent_layout) -> None:
        """Create widgets for interactive multi-curve creation and editing.

        Inputs:
            parent_layout: Layout receiving the curve controls group.

        Outputs:
            None.

        Behavior:
            Allows users to create Bezier, B-spline, and NURBS curves, select
            active curves, edit control points, and manage per-point weights.
        """
        group = QtWidgets.QGroupBox("Curves")
        self.curve_group = group
        layout = QtWidgets.QVBoxLayout(group)

        create_row = QtWidgets.QHBoxLayout()
        self.curve_type_combo = QtWidgets.QComboBox()
        self.curve_type_combo.addItems(["Bezier", "B-spline", "NURBS"])
        self.curve_type_combo.currentTextChanged.connect(self._on_curve_type_changed)

        self.curve_degree_spin = QtWidgets.QSpinBox()
        self.curve_degree_spin.setRange(1, 6)
        self.curve_degree_spin.setValue(3)

        self.curve_knot_mode_combo = QtWidgets.QComboBox()
        self.curve_knot_mode_combo.addItems(["Uniform", "Non-uniform"])

        self.create_curve_button = QtWidgets.QPushButton("Create Curve")
        self.create_curve_button.clicked.connect(self._on_create_curve)

        create_row.addWidget(QtWidgets.QLabel("Type"))
        create_row.addWidget(self.curve_type_combo)
        create_row.addWidget(QtWidgets.QLabel("Degree"))
        create_row.addWidget(self.curve_degree_spin)
        create_row.addWidget(QtWidgets.QLabel("Knots"))
        create_row.addWidget(self.curve_knot_mode_combo)
        create_row.addWidget(self.create_curve_button)
        layout.addLayout(create_row)

        self.curve_plane_lock_checkbox = QtWidgets.QCheckBox("Place points on XY reference plane (z=0)")
        self.curve_plane_lock_checkbox.setChecked(True)
        layout.addWidget(self.curve_plane_lock_checkbox)

        interaction_row = QtWidgets.QHBoxLayout()
        self.curve_interaction_mode_combo = QtWidgets.QComboBox()
        self.curve_interaction_mode_combo.addItem("Select", "select")
        self.curve_interaction_mode_combo.addItem("Add Points (Viewport Click)", "add")
        self.curve_interaction_mode_combo.addItem("Drag Handles (Viewport)", "drag")
        self.curve_interaction_mode_combo.currentIndexChanged.connect(self._on_curve_interaction_mode_changed)
        interaction_row.addWidget(QtWidgets.QLabel("Viewport mode"))
        interaction_row.addWidget(self.curve_interaction_mode_combo)
        layout.addLayout(interaction_row)

        self.curve_list_widget = QtWidgets.QListWidget()
        self.curve_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.curve_list_widget.currentRowChanged.connect(self._on_curve_selection_changed)
        self.curve_list_widget.itemSelectionChanged.connect(self._on_curve_list_item_selection_changed)
        layout.addWidget(self.curve_list_widget)

        curve_button_row = QtWidgets.QHBoxLayout()
        self.delete_curve_button = QtWidgets.QPushButton("Delete Curve")
        self.delete_curve_button.clicked.connect(self._on_delete_curve)
        self.add_curve_point_button = QtWidgets.QPushButton("Add Point")
        self.add_curve_point_button.clicked.connect(self._on_add_curve_point)
        self.delete_curve_point_button = QtWidgets.QPushButton("Delete Point")
        self.delete_curve_point_button.clicked.connect(self._on_delete_curve_point)

        curve_button_row.addWidget(self.delete_curve_button)
        curve_button_row.addWidget(self.add_curve_point_button)
        curve_button_row.addWidget(self.delete_curve_point_button)
        layout.addLayout(curve_button_row)

        self.curve_point_list_widget = QtWidgets.QListWidget()
        self.curve_point_list_widget.currentRowChanged.connect(self._on_curve_point_selection_changed)
        layout.addWidget(self.curve_point_list_widget)

        editor_form = QtWidgets.QFormLayout()
        self.curve_point_x_spin = QtWidgets.QDoubleSpinBox()
        self.curve_point_y_spin = QtWidgets.QDoubleSpinBox()
        self.curve_point_z_spin = QtWidgets.QDoubleSpinBox()
        self.curve_point_weight_spin = QtWidgets.QDoubleSpinBox()

        for spin in (self.curve_point_x_spin, self.curve_point_y_spin, self.curve_point_z_spin):
            spin.setRange(-500.0, 500.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.5)

        self.curve_point_weight_spin.setRange(0.05, 50.0)
        self.curve_point_weight_spin.setDecimals(3)
        self.curve_point_weight_spin.setSingleStep(0.1)

        self.apply_curve_point_button = QtWidgets.QPushButton("Apply Point Edit")
        self.apply_curve_point_button.clicked.connect(self._on_apply_curve_point)

        editor_form.addRow("X", self.curve_point_x_spin)
        editor_form.addRow("Y", self.curve_point_y_spin)
        editor_form.addRow("Z", self.curve_point_z_spin)
        editor_form.addRow("Weight", self.curve_point_weight_spin)
        editor_form.addRow(self.apply_curve_point_button)
        layout.addLayout(editor_form)

        parent_layout.addWidget(group)
        self._register_major_section(group, expanded_by_default=True)
        self._on_curve_type_changed(self.curve_type_combo.currentText())
        self._refresh_curve_list_widget()

    def _curve_type_name(self, curve: CurveObject) -> str:
        """Return UI curve-type label for one curve instance."""
        if isinstance(curve, BezierCurve):
            return "Bezier"
        if isinstance(curve, BSplineCurve):
            return "B-spline"
        return "NURBS"

    def _curve_index_from_id(self, curve_id: int | None) -> int | None:
        """Return curve index for one stable curve id."""
        if curve_id is None:
            return None
        for curve_index, known_id in enumerate(self.curve_ids):
            if known_id == int(curve_id):
                return curve_index
        return None

    def _curve_id_from_index(self, curve_index: int | None) -> int | None:
        """Return stable curve id from one curve index."""
        if curve_index is None:
            return None
        if 0 <= int(curve_index) < len(self.curve_ids):
            return int(self.curve_ids[int(curve_index)])
        return None

    def _allocate_curve_id(self) -> int:
        """Allocate and return one new unique curve id."""
        curve_id = int(self._next_curve_id)
        self._next_curve_id += 1
        return curve_id

    def _sync_active_curve_index_from_selection(self) -> None:
        """Mirror id-based curve selection state into legacy index fields."""
        previous_active_index = self.active_curve_index
        self.active_curve_index = self._curve_index_from_id(self.active_curve_id)
        active_curve = self._get_active_curve()
        if active_curve is None:
            self.active_curve_point_index = None
            return

        if previous_active_index != self.active_curve_index:
            self.active_curve_point_index = 0 if self._curve_point_count(active_curve) > 0 else None
            return

        if self.active_curve_point_index is None:
            self.active_curve_point_index = 0 if self._curve_point_count(active_curve) > 0 else None
            return

        if not (0 <= int(self.active_curve_point_index) < self._curve_point_count(active_curve)):
            self.active_curve_point_index = 0 if self._curve_point_count(active_curve) > 0 else None

    def _apply_curve_selection_action(self, clicked_curve_id: int | None, shift_pressed: bool) -> None:
        """Apply one viewport/list click action to curve selection state."""
        (
            self.selected_curve_ids,
            self.active_curve_id,
            self._last_interacted_curve_id,
        ) = reduce_curve_selection_state(
            valid_curve_ids=self.curve_ids,
            selected_curve_ids=self.selected_curve_ids,
            active_curve_id=self.active_curve_id,
            last_interacted_curve_id=self._last_interacted_curve_id,
            clicked_curve_id=clicked_curve_id,
            shift_pressed=bool(shift_pressed),
        )
        self._sync_active_curve_index_from_selection()

    def _sanitize_curve_selection_after_model_change(self) -> None:
        """Drop stale curve references from selection state after model changes."""
        (
            self.selected_curve_ids,
            self.active_curve_id,
            self._last_interacted_curve_id,
        ) = sanitize_curve_selection_state(
            valid_curve_ids=self.curve_ids,
            selected_curve_ids=self.selected_curve_ids,
            active_curve_id=self.active_curve_id,
            last_interacted_curve_id=self._last_interacted_curve_id,
        )
        self._sync_active_curve_index_from_selection()

    def _sync_curve_selection_ui_from_state(self) -> None:
        """Push internal curve selection state into the curve list widget."""
        if not hasattr(self, "curve_list_widget"):
            return

        self._is_syncing_curve_selection_ui = True
        try:
            blocked = self.curve_list_widget.blockSignals(True)
            for curve_index, curve_id in enumerate(self.curve_ids):
                item = self.curve_list_widget.item(curve_index)
                if item is None:
                    continue
                item.setSelected(curve_id in self.selected_curve_ids)

            if self.active_curve_index is None:
                self.curve_list_widget.setCurrentRow(-1)
            else:
                self.curve_list_widget.setCurrentRow(int(self.active_curve_index))
            self.curve_list_widget.blockSignals(blocked)
        finally:
            self._is_syncing_curve_selection_ui = False

    def _sync_curve_selection_state_from_list_widget(self) -> None:
        """Pull curve selection state from list widget selection/current row."""
        if self._is_syncing_curve_selection_ui:
            return

        selected_rows = sorted(
            {
                int(model_index.row())
                for model_index in self.curve_list_widget.selectedIndexes()
                if 0 <= int(model_index.row()) < len(self.curve_ids)
            }
        )

        selected_ids = {self.curve_ids[row] for row in selected_rows}
        current_row = int(self.curve_list_widget.currentRow())
        current_id = self._curve_id_from_index(current_row)

        if current_id is not None and current_id in selected_ids:
            active_id = current_id
        elif self._last_interacted_curve_id is not None and self._last_interacted_curve_id in selected_ids:
            active_id = self._last_interacted_curve_id
        else:
            active_id = next((self.curve_ids[row] for row in selected_rows), None)

        self.selected_curve_ids = selected_ids
        self.active_curve_id = active_id
        if active_id is not None:
            self._last_interacted_curve_id = active_id

        self._sanitize_curve_selection_after_model_change()
        self._sync_curve_selection_ui_from_state()

    def _get_active_curve(self) -> CurveObject | None:
        """Return currently selected curve instance, if any."""
        if self.active_curve_index is None:
            return None
        if not (0 <= self.active_curve_index < len(self.curves)):
            return None
        return self.curves[self.active_curve_index]

    def _curve_point_count(self, curve: CurveObject | None) -> int:
        """Return number of control points in one curve."""
        if curve is None:
            return 0
        return int(curve.control_points.shape[0])

    def _default_curve_points(self, count: int, curve_slot: int) -> np.ndarray:
        """Create initial control points for a new curve on the reference plane."""
        x_values = np.linspace(-30.0, 30.0, count)
        y_value = 20.0 * float(curve_slot)
        y_values = np.full(count, y_value, dtype=float)
        z_values = np.zeros(count, dtype=float)
        return np.column_stack((x_values, y_values, z_values))

    def _refresh_surface_selector_widget(self) -> None:
        """Refresh active-surface selector entries from surface registry state."""
        if not hasattr(self, "surface_selector_combo"):
            return

        blocked = self.surface_selector_combo.blockSignals(True)
        self.surface_selector_combo.clear()
        for entry in self.surface_entries:
            name = str(entry.get("name", "Surface"))
            surface_id = int(entry["surface_id"])
            self.surface_selector_combo.addItem(name, surface_id)

        if self.active_surface_id is not None:
            active_index = self.surface_selector_combo.findData(int(self.active_surface_id))
            if active_index >= 0:
                self.surface_selector_combo.setCurrentIndex(int(active_index))
        self.surface_selector_combo.blockSignals(blocked)

    def _set_active_surface(self, surface_id: int | None) -> bool:
        """Set one active surface from registry and sync compatibility bridge."""
        previous_surface_id = self.active_surface_id
        if surface_id is None:
            self.active_surface_id = None
            self.surface = None
            self._invalidate_active_surface_render_cache()
            self._refresh_surface_selector_widget()
            self._configure_control_editor_ranges()
            self._sync_spin_boxes_from_state()
            self._sync_ui_enabled_state()
            return True

        for entry in self.surface_entries:
            if int(entry["surface_id"]) != int(surface_id):
                continue
            self.active_surface_id = int(surface_id)
            self.surface = entry["surface"]
            if previous_surface_id != self.active_surface_id:
                self._invalidate_active_surface_render_cache()
            self._refresh_surface_selector_widget()
            self._configure_control_editor_ranges()
            self._sync_spin_boxes_from_state()
            self._sync_ui_enabled_state()
            return True
        return False

    def _invalidate_active_surface_render_cache(self) -> None:
        """Drop cached active-surface render objects so next refresh rebuilds them."""
        self._rendered_surface_id = None
        if self._surface_actor is not None:
            self.plotter.remove_actor(self._surface_actor)
            self._surface_actor = None
        self._surface_mesh = None

        if self._control_net_actor is not None:
            self.plotter.remove_actor(self._control_net_actor)
            self._control_net_actor = None
        self._control_net_mesh = None

    def _register_surface_entry(
        self,
        surface_object: object,
        name: str,
        operation_tag: str,
        source_surface_id: int | None = None,
        set_active: bool = True,
    ) -> int:
        """Register one surface in scene registry and optionally activate it."""
        surface_id = int(self._next_surface_id)
        self._next_surface_id += 1

        entry = {
            "surface_id": surface_id,
            "name": str(name),
            "surface": surface_object,
            "source_surface_id": source_surface_id,
            "operation_tag": str(operation_tag),
        }
        self.surface_entries.append(entry)

        if set_active:
            self._set_active_surface(surface_id)
        else:
            self._refresh_surface_selector_widget()
        return surface_id

    def _clear_surface_registry(self) -> None:
        """Clear all registered surfaces and active-surface compatibility state."""
        self.surface_entries.clear()
        self._set_active_surface(None)
        self._rendered_surface_id = None

    def _register_derived_body_entry(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        name: str,
        mode: str,
        source_surface_id: int | None,
    ) -> None:
        """Store one derived extrusion body for persistent viewport rendering."""
        self._derived_body_entries.append(
            {
                "name": str(name),
                "mode": str(mode),
                "source_surface_id": source_surface_id,
                "vertices": np.asarray(vertices, dtype=float).copy(),
                "triangles": np.asarray(triangles, dtype=np.int64).copy(),
            }
        )

    def _clear_derived_body_actors(self) -> None:
        """Remove all rendered derived-body actors from viewport."""
        for actor in self._derived_body_actors:
            self.plotter.remove_actor(actor)
        self._derived_body_actors.clear()

    def _draw_derived_body_overlays(self) -> None:
        """Render all stored shell/solid extrusion bodies as triangulated meshes."""
        self._clear_derived_body_actors()

        for body_entry in self._derived_body_entries:
            vertices = np.asarray(body_entry["vertices"], dtype=float)
            triangles = np.asarray(body_entry["triangles"], dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
                continue

            mesh = self._create_polydata_from_triangles(vertices, triangles)
            if mesh is None:
                continue

            color = "#38bdf8" if str(body_entry["mode"]) == "solid" else "#34d399"
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.42,
                show_edges=True,
                edge_color="#0f172a",
                line_width=1,
            )
            self._derived_body_actors.append(actor)

    def _invalidate_generated_geometry_from_curve_edit(self) -> None:
        """Invalidate dependent generated geometry when source curves change."""
        self.generated_passes = []
        self._clear_toolpath_actors()
        self._derived_body_entries.clear()
        self._clear_derived_body_actors()

        if self.surface_entries or self._has_active_surface():
            self._clear_surface_registry()

    def _create_curve_from_inputs(self, curve_type: str) -> CurveObject:
        """Create a new curve from current curve panel settings."""
        normalized_type = curve_type.strip()
        degree_value = int(self.curve_degree_spin.value())
        knot_mode = self.curve_knot_mode_combo.currentText().strip().lower()

        if normalized_type == "Bezier":
            points = self._default_curve_points(4, len(self.curves))
            return BezierCurve(points)

        control_count = max(degree_value + 1, 4)
        points = self._default_curve_points(control_count, len(self.curves))
        knots = build_knots_for_mode(points, degree_value, knot_mode)

        if normalized_type == "B-spline":
            return BSplineCurve(points, degree=degree_value, knots=knots)

        weights = np.ones(control_count, dtype=float)
        return NURBSCurve(points, degree=degree_value, knots=knots, weights=weights)

    def _rebuild_curve_from_components(
        self,
        curve: CurveObject,
        points: np.ndarray,
        weights: np.ndarray | None,
        knot_mode: str,
    ) -> CurveObject:
        """Reconstruct a curve object from edited components."""
        if isinstance(curve, BezierCurve):
            return BezierCurve(points)

        if isinstance(curve, BSplineCurve):
            degree = min(curve.degree, points.shape[0] - 1)
            knots = build_knots_for_mode(points, degree, knot_mode)
            return BSplineCurve(points, degree=degree, knots=knots)

        degree = min(curve.degree, points.shape[0] - 1)
        if weights is None:
            raise ValueError("weights are required when rebuilding a NURBS curve")
        knots = build_knots_for_mode(points, degree, knot_mode)
        return NURBSCurve(points, degree=degree, knots=knots, weights=weights)

    def _refresh_curve_list_widget(self) -> None:
        """Refresh curve list widget contents from current curve collection."""
        self._sanitize_curve_selection_after_model_change()

        blocked = self.curve_list_widget.blockSignals(True)
        self.curve_list_widget.clear()

        for curve_index, curve in enumerate(self.curves):
            curve_id = self.curve_ids[curve_index] if curve_index < len(self.curve_ids) else -1
            label = (
                f"{curve_index + 1}: {self._curve_type_name(curve)} "
                f"({self._curve_point_count(curve)} pts) [id={curve_id}]"
            )
            self.curve_list_widget.addItem(label)
        self.curve_list_widget.blockSignals(blocked)

        self._sync_curve_selection_ui_from_state()

        self._refresh_curve_point_list_widget()

    def _refresh_curve_point_list_widget(self) -> None:
        """Refresh active-curve control-point list widget contents."""
        blocked = self.curve_point_list_widget.blockSignals(True)
        self.curve_point_list_widget.clear()

        curve = self._get_active_curve()
        if curve is not None:
            for point_index, point in enumerate(curve.control_points):
                label = f"P{point_index}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
                self.curve_point_list_widget.addItem(label)

            if self.active_curve_point_index is not None and 0 <= self.active_curve_point_index < self._curve_point_count(curve):
                self.curve_point_list_widget.setCurrentRow(self.active_curve_point_index)

        self.curve_point_list_widget.blockSignals(blocked)
        self._sync_curve_point_editor_values()

    def _sync_curve_point_editor_values(self) -> None:
        """Synchronize curve point editor spin-box values from selection state."""
        curve = self._get_active_curve()
        if curve is None or self.active_curve_point_index is None:
            defaults = (
                (self.curve_point_x_spin, 0.0),
                (self.curve_point_y_spin, 0.0),
                (self.curve_point_z_spin, 0.0),
                (self.curve_point_weight_spin, 1.0),
            )
            for widget, value in defaults:
                blocked = widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(blocked)
            return

        if not (0 <= self.active_curve_point_index < self._curve_point_count(curve)):
            return

        point = curve.control_points[self.active_curve_point_index]
        weight = 1.0
        if isinstance(curve, NURBSCurve):
            weight = float(curve.weights[self.active_curve_point_index])

        values = (
            (self.curve_point_x_spin, point[0]),
            (self.curve_point_y_spin, point[1]),
            (self.curve_point_z_spin, point[2]),
            (self.curve_point_weight_spin, weight),
        )

        for widget, value in values:
            blocked = widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(blocked)

    def _on_curve_type_changed(self, curve_type: str) -> None:
        """Update curve creation controls when type selection changes."""
        normalized_type = curve_type.strip()
        is_bezier = normalized_type == "Bezier"

        self.curve_degree_spin.setEnabled(not is_bezier)
        self.curve_knot_mode_combo.setEnabled(not is_bezier)
        self._sync_ui_enabled_state()

    def _on_curve_interaction_mode_changed(self, _index: int) -> None:
        """Update active viewport interaction mode for curve editing."""
        self._curve_interaction_mode = str(self.curve_interaction_mode_combo.currentData())
        if self._curve_interaction_mode != "drag":
            self._is_curve_dragging = False
            self._curve_drag_point_index = None
        else:
            self.status_label.setText("Drag mode active: drag visible curve handles in viewport")

        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)

    def _register_viewport_curve_interactions(self) -> None:
        """Register viewport callbacks used for curve click and drag workflows."""
        self.plotter.track_click_position(
            callback=self._on_viewport_left_click,
            side="left",
            double=False,
            viewport=False,
        )
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_viewport_left_button_release)

    def _display_to_world_point(self, x_position: float, y_position: float, depth: float) -> np.ndarray | None:
        """Convert one display coordinate and depth value to world coordinates."""
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(float(x_position), float(y_position), float(depth))
        renderer.DisplayToWorld()
        world_h = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if world_h.shape != (4,) or abs(float(world_h[3])) < 1e-6:
            return None

        world_point = world_h[:3] / float(world_h[3])
        if np.any(np.isnan(world_point)) or np.any(np.isinf(world_point)):
            return None
        return world_point

    def _intersect_display_with_plane(
        self,
        display_position: tuple[float, float],
        plane_z: float,
    ) -> np.ndarray | None:
        """Intersect cursor ray from display coordinates with plane z=constant."""
        x_position, y_position = display_position
        near_point = self._display_to_world_point(float(x_position), float(y_position), 0.0)
        far_point = self._display_to_world_point(float(x_position), float(y_position), 1.0)
        if near_point is None or far_point is None:
            return None

        ray_direction = far_point - near_point
        if abs(float(ray_direction[2])) < 1e-7:
            return None

        t_value = (float(plane_z) - float(near_point[2])) / float(ray_direction[2])
        if abs(float(t_value)) > 1000.0:
            return None
        return near_point + t_value * ray_direction

    def _estimate_world_units_per_screen_pixel(self) -> float:
        """Estimate world-space distance represented by one screen pixel."""
        try:
            camera = self.plotter.renderer.GetActiveCamera()
            view_angle_deg = float(camera.GetViewAngle())
            camera_distance = float(camera.GetDistance())
            window_size = self.plotter.ren_win.GetSize()
            viewport_height_px = max(float(window_size[1]), 1.0)

            world_view_height = 2.0 * camera_distance * np.tan(np.deg2rad(0.5 * view_angle_deg))
            if not np.isfinite(world_view_height) or world_view_height <= 0.0:
                return 1.0
            return max(float(world_view_height / viewport_height_px), 1e-6)
        except Exception:
            return 1.0

    def _nearest_active_curve_point_index(self, world_point: np.ndarray) -> int | None:
        """Return nearest active-curve point index when within picking threshold."""
        curve = self._get_active_curve()
        if curve is None or curve.control_points.shape[0] == 0:
            return None

        distances = np.linalg.norm(curve.control_points - world_point[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        pixel_world_size = self._estimate_world_units_per_screen_pixel()
        pick_radius = max(
            self._curve_drag_pick_radius_mm,
            16.0 * pixel_world_size,
        )
        if float(distances[nearest_index]) > pick_radius:
            return None
        return nearest_index

    def _desired_widget_layer(self) -> str:
        """Return active sphere-widget layer based on current workflow state."""
        if self._has_active_surface():
            return "surface"
        if self._curve_interaction_mode == "drag" and self._get_active_curve() is not None:
            return "curve"
        return "none"

    def _ensure_active_widget_layer(self) -> None:
        """Ensure only one widget layer is active and synchronized."""
        desired_layer = self._desired_widget_layer()

        if desired_layer != self._active_widget_layer:
            self.plotter.clear_sphere_widgets()
            self._control_point_widgets.clear()
            self._curve_point_widgets.clear()
            self._selected_control_flat_index = None
            self._selected_curve_point_widget_index = None
            self._active_widget_layer = desired_layer

        if desired_layer == "surface":
            if not self._has_active_surface():
                return

            expected_widget_count = int(np.prod(self.surface.control_net.shape[:2]))
            if len(self._control_point_widgets) != expected_widget_count:
                self.plotter.clear_sphere_widgets()
                self._control_point_widgets.clear()
                self._curve_point_widgets.clear()
                self._selected_control_flat_index = None
                self._selected_curve_point_widget_index = None
                self._active_widget_layer = "surface"
                self._initialize_control_point_widgets()
            else:
                self._sync_widget_positions_from_control_net()
            return

        if desired_layer == "curve":
            curve = self._get_active_curve()
            if curve is None:
                return

            if len(self._curve_point_widgets) != self._curve_point_count(curve):
                self.plotter.clear_sphere_widgets()
                self._control_point_widgets.clear()
                self._curve_point_widgets.clear()
                self._selected_curve_point_widget_index = None
                self._active_widget_layer = "curve"
                self._initialize_curve_point_widgets()
            else:
                self._sync_curve_widget_positions_from_curve()

    def _make_curve_drag_callback(self, point_index: int):
        """Create and return a drag callback for one curve control point."""

        def _callback(center: tuple[float, float, float], widget: object) -> None:
            self._on_curve_point_widget_drag(point_index, np.asarray(center, dtype=float), widget)

        return _callback

    def _initialize_curve_point_widgets(self) -> None:
        """Create draggable sphere widgets for active-curve control points."""
        curve = self._get_active_curve()
        if curve is None:
            self._curve_point_widgets.clear()
            self._selected_curve_point_widget_index = None
            return

        self._curve_point_widgets.clear()

        for point_index, point in enumerate(curve.control_points):
            widget = self.plotter.add_sphere_widget(
                callback=self._make_curve_drag_callback(point_index),
                center=point.tolist(),
                radius=max(1.8, self._default_widget_radius * 1.15),
                color=(0.99, 0.52, 0.16),
                selected_color=self._selected_widget_color,
                pass_widget=True,
                test_callback=False,
                interaction_event="always",
            )
            self._curve_point_widgets.append(widget)

        selected_index = self.active_curve_point_index
        if selected_index is None or not (0 <= selected_index < len(self._curve_point_widgets)):
            selected_index = 0 if self._curve_point_widgets else None
        self.active_curve_point_index = selected_index
        self._set_selected_curve_point_widget(selected_index)

    def _set_selected_curve_point_widget(self, point_index: int | None) -> None:
        """Highlight selected curve-point widget and de-highlight others."""
        self._selected_curve_point_widget_index = point_index

        for widget_index, widget in enumerate(self._curve_point_widgets):
            prop = widget.GetSphereProperty()
            if point_index is not None and widget_index == point_index:
                prop.SetColor(*self._selected_widget_color)
                widget.SetRadius(max(self._selected_widget_radius, self._default_widget_radius * 1.2))
            else:
                prop.SetColor(0.99, 0.52, 0.16)
                widget.SetRadius(max(1.8, self._default_widget_radius * 1.15))

    def _sync_curve_widget_positions_from_curve(self) -> None:
        """Move curve-point widget centers to match active curve control points."""
        curve = self._get_active_curve()
        if curve is None or not self._curve_point_widgets:
            return

        if len(self._curve_point_widgets) != self._curve_point_count(curve):
            self._initialize_curve_point_widgets()
            return

        self._is_syncing_curve_widgets = True
        try:
            for widget, point in zip(self._curve_point_widgets, curve.control_points):
                widget.SetCenter(float(point[0]), float(point[1]), float(point[2]))
        finally:
            self._is_syncing_curve_widgets = False

        self._set_selected_curve_point_widget(self.active_curve_point_index)

    def _on_curve_point_widget_drag(
        self,
        point_index: int,
        new_center: np.ndarray,
        widget: object,
    ) -> None:
        """Handle active-curve control-point drag events from sphere widgets."""
        if self._is_syncing_curve_widgets:
            return

        curve = self._get_active_curve()
        if curve is None or not (0 <= point_index < self._curve_point_count(curve)):
            return

        current_time = time.perf_counter()
        if current_time - self._last_curve_drag_update_timestamp < self._drag_update_interval_sec:
            return
        self._last_curve_drag_update_timestamp = current_time

        if not self._is_curve_dragging:
            self._invalidate_generated_geometry_from_curve_edit()
            self._clear_surface_visuals()

        self._is_curve_dragging = True
        self._curve_drag_point_index = point_index

        current_point = curve.control_points[point_index].copy()
        adjusted_point = current_point + self._drag_sensitivity * (new_center - current_point)

        if self.curve_plane_lock_checkbox.isChecked():
            adjusted_point[2] = 0.0

        if isinstance(curve, NURBSCurve):
            curve.update_control_point_inplace(point_index, adjusted_point, float(curve.weights[point_index]))
        else:
            curve.update_control_point_inplace(point_index, adjusted_point)

        if self._drag_sensitivity != 1.0:
            self._is_syncing_curve_widgets = True
            try:
                widget.SetCenter(
                    float(adjusted_point[0]),
                    float(adjusted_point[1]),
                    float(adjusted_point[2]),
                )
            finally:
                self._is_syncing_curve_widgets = False

        self.active_curve_point_index = point_index
        active_curve_id = self._curve_id_from_index(self.active_curve_index)
        if active_curve_id is not None:
            self.active_curve_id = active_curve_id
            self.selected_curve_ids.add(active_curve_id)
            self._last_interacted_curve_id = active_curve_id
            self._sync_active_curve_index_from_selection()
        self._set_selected_curve_point_widget(point_index)

        if self.generated_passes:
            self.generated_passes = []
            self._clear_toolpath_actors()
            self._sync_ui_enabled_state()

        self._sync_curve_point_editor_values()
        self._draw_curve_overlays()
        self.status_label.setText(f"Dragging curve point P{point_index}")
        self.plotter.render()

    def _is_shift_modifier_active(self) -> bool:
        """Return True when shift modifier is active for current interactor event."""
        try:
            if QtWidgets is not None and QtCore is not None and hasattr(QtWidgets, "QApplication"):
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if int(modifiers & QtCore.Qt.ShiftModifier):
                    return True

            interactor = getattr(self.plotter.iren, "interactor", None)
            if interactor is not None and hasattr(interactor, "GetShiftKey"):
                return bool(interactor.GetShiftKey())
            if hasattr(self.plotter.iren, "GetShiftKey"):
                return bool(self.plotter.iren.GetShiftKey())
        except Exception:
            return False
        return False

    def _pick_curve_id_at_click(self, display_position: tuple[float, float]) -> int | None:
        """Pick one rendered curve actor and return mapped curve id if available."""
        if not self._curve_actor_to_curve_id:
            return None

        if pv is None:
            return None

        try:
            picker = pv._vtk.vtkPropPicker()
            x_pos, y_pos = display_position
            picked = int(picker.Pick(float(x_pos), float(y_pos), 0.0, self.plotter.renderer))
            if picked != 1:
                return None

            picked_actor = picker.GetActor()
            if picked_actor is not None:
                mapped_id = self._curve_actor_to_curve_id.get(id(picked_actor))
                if mapped_id is not None:
                    return mapped_id

            picked_prop = picker.GetViewProp()
            if picked_prop is not None:
                return self._curve_actor_to_curve_id.get(id(picked_prop))
        except Exception:
            return None
        return None

    def _on_viewport_left_click(self, display_position: tuple[float, float]) -> None:
        """Handle left-click actions for curve selection and add-point workflows."""
        if self._curve_interaction_mode == "select":
            curve_id = self._pick_curve_id_at_click(display_position)
            shift_pressed = self._is_shift_modifier_active()
            self._apply_curve_selection_action(curve_id, shift_pressed=shift_pressed)
            self._refresh_curve_list_widget()
            self._sync_ui_enabled_state()
            self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
            if curve_id is None:
                self.status_label.setText("Curve selection cleared")
            elif self.active_curve_id is not None:
                self.status_label.setText(f"Selected curve id={self.active_curve_id}")
            return

        if self._curve_interaction_mode != "add":
            return

        curve = self._get_active_curve()
        if curve is None:
            self.status_label.setText("Select a curve before placing viewport points")
            return

        plane_z = 0.0 if self.curve_plane_lock_checkbox.isChecked() else float(np.mean(curve.control_points[:, 2]))
        world_point = self._intersect_display_with_plane(display_position, plane_z=plane_z)
        if world_point is None:
            self.status_label.setText("Could not map click to workspace plane")
            return

        if self.curve_plane_lock_checkbox.isChecked():
            world_point[2] = 0.0

        self._append_point_to_active_curve(world_point, status_text="Curve point placed from viewport click")

    def _on_viewport_left_button_press(self, *_args) -> None:
        """Start curve-handle drag workflow when drag mode is active."""
        if self._curve_interaction_mode != "drag":
            return

        curve = self._get_active_curve()
        if curve is None:
            return

        display_position = self.plotter.iren.get_event_position()
        plane_z = 0.0 if self.curve_plane_lock_checkbox.isChecked() else float(np.mean(curve.control_points[:, 2]))
        world_point = self._intersect_display_with_plane(display_position, plane_z=plane_z)
        if world_point is None:
            return

        nearest_index = self._nearest_active_curve_point_index(world_point)
        if nearest_index is None:
            return

        self._is_curve_dragging = True
        self._curve_drag_point_index = nearest_index
        self.active_curve_point_index = nearest_index
        active_curve_id = self._curve_id_from_index(self.active_curve_index)
        if active_curve_id is not None:
            self.active_curve_id = active_curve_id
            self.selected_curve_ids.add(active_curve_id)
            self._last_interacted_curve_id = active_curve_id

        self._invalidate_generated_geometry_from_curve_edit()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText(f"Dragging curve point P{nearest_index}")

    def _on_viewport_mouse_move(self, *_args) -> None:
        """Update active curve control point while dragging in viewport."""
        if self._curve_interaction_mode != "drag" or not self._is_curve_dragging:
            return

        curve = self._get_active_curve()
        point_index = self._curve_drag_point_index
        if curve is None or point_index is None or not (0 <= point_index < self._curve_point_count(curve)):
            return

        current_time = time.perf_counter()
        if current_time - self._last_curve_drag_update_timestamp < self._drag_update_interval_sec:
            return
        self._last_curve_drag_update_timestamp = current_time

        display_position = self.plotter.iren.get_event_position()
        plane_z = 0.0 if self.curve_plane_lock_checkbox.isChecked() else float(curve.control_points[point_index, 2])
        new_point = self._intersect_display_with_plane(display_position, plane_z=plane_z)
        if new_point is None:
            return

        if self.curve_plane_lock_checkbox.isChecked():
            new_point[2] = 0.0

        if isinstance(curve, NURBSCurve):
            curve.update_control_point_inplace(point_index, new_point, float(curve.weights[point_index]))
        else:
            curve.update_control_point_inplace(point_index, new_point)

        self._sync_curve_point_editor_values()
        self._draw_curve_overlays()
        self.plotter.render()

    def _on_viewport_left_button_release(self, *_args) -> None:
        """Finish curve-handle drag workflow and synchronize panel widgets."""
        if not self._is_curve_dragging:
            return

        self._is_curve_dragging = False
        self._curve_drag_point_index = None
        self._refresh_curve_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText("Curve handle drag completed")

    def _on_create_curve(self) -> None:
        """Create and select a new curve from current curve panel inputs."""
        try:
            curve = self._create_curve_from_inputs(self.curve_type_combo.currentText())
        except ValueError as error:
            self.status_label.setText(f"Curve creation failed: {error}")
            return

        knot_mode = self.curve_knot_mode_combo.currentText().strip().lower()
        if isinstance(curve, BezierCurve):
            knot_mode = "uniform"

        curve_id = self._allocate_curve_id()
        self.curves.append(curve)
        self.curve_ids.append(curve_id)
        self.curve_knot_modes.append(knot_mode)
        self._apply_curve_selection_action(curve_id, shift_pressed=False)
        self.active_curve_point_index = 0 if self._curve_point_count(curve) > 0 else None

        self._invalidate_generated_geometry_from_curve_edit()
        self._refresh_curve_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText("Curve created")

    def _on_delete_curve(self) -> None:
        """Delete currently selected curve."""
        if self.active_curve_index is None or not (0 <= self.active_curve_index < len(self.curves)):
            self.status_label.setText("Select a curve to delete")
            return

        del self.curves[self.active_curve_index]
        del self.curve_ids[self.active_curve_index]
        del self.curve_knot_modes[self.active_curve_index]
        self._sanitize_curve_selection_after_model_change()

        self._invalidate_generated_geometry_from_curve_edit()
        self._refresh_curve_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText("Curve deleted")

    def _on_curve_list_item_selection_changed(self) -> None:
        """Handle multi-selection updates initiated from curve list widget."""
        self._sync_curve_selection_state_from_list_widget()
        self._sync_curve_selection_ui_from_state()
        self._refresh_curve_point_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)

    def _on_curve_selection_changed(self, row: int) -> None:
        """Handle active curve selection changes from list widget."""
        if self._is_syncing_curve_selection_ui:
            return

        if row < 0 or row >= len(self.curves):
            self.active_curve_id = None
            self._sanitize_curve_selection_after_model_change()
        else:
            row_curve_id = self._curve_id_from_index(int(row))
            if row_curve_id is not None and row_curve_id in self.selected_curve_ids:
                self.active_curve_id = row_curve_id
                self._last_interacted_curve_id = row_curve_id
                self._sanitize_curve_selection_after_model_change()
            else:
                self._sync_curve_selection_state_from_list_widget()

        self._sync_curve_selection_ui_from_state()
        self._refresh_curve_point_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)

    def _on_curve_point_selection_changed(self, row: int) -> None:
        """Handle active control-point selection changes for active curve."""
        curve = self._get_active_curve()
        if curve is None or row < 0 or row >= self._curve_point_count(curve):
            self.active_curve_point_index = None
        else:
            self.active_curve_point_index = int(row)

        self._sync_curve_point_editor_values()
        self._sync_ui_enabled_state()

        if self._curve_interaction_mode == "drag":
            self._set_selected_curve_point_widget(self.active_curve_point_index)
            self.plotter.render()

    def _on_add_curve_point(self) -> None:
        """Append one control point to the active curve."""
        curve = self._get_active_curve()
        if curve is None:
            self.status_label.setText("Select a curve before adding points")
            return

        points = curve.control_points.copy()
        if points.shape[0] == 0:
            new_point = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            new_point = points[-1] + np.array([10.0, 0.0, 0.0], dtype=float)

        if self.curve_plane_lock_checkbox.isChecked():
            new_point[2] = 0.0

        self._append_point_to_active_curve(new_point, status_text="Curve point added")

    def _append_point_to_active_curve(self, new_point: np.ndarray, status_text: str) -> bool:
        """Append one control point to active curve and refresh dependent state."""
        curve = self._get_active_curve()
        if curve is None:
            self.status_label.setText("Select a curve before adding points")
            return False

        points = np.vstack((curve.control_points.copy(), np.asarray(new_point, dtype=float)))

        weights = None
        if isinstance(curve, NURBSCurve):
            weights = np.append(curve.weights, 1.0)

        knot_mode = self.curve_knot_modes[self.active_curve_index]
        self.curves[self.active_curve_index] = self._rebuild_curve_from_components(
            curve,
            points,
            weights,
            knot_mode,
        )

        self.active_curve_point_index = points.shape[0] - 1
        self._invalidate_generated_geometry_from_curve_edit()
        self._refresh_curve_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText(status_text)
        return True

    def _on_delete_curve_point(self) -> None:
        """Delete selected control point from active curve when valid."""
        curve = self._get_active_curve()
        if curve is None or self.active_curve_point_index is None:
            self.status_label.setText("Select a curve point to delete")
            return

        point_index = int(self.active_curve_point_index)
        if not (0 <= point_index < self._curve_point_count(curve)):
            self.status_label.setText("Select a valid curve point to delete")
            return

        minimum_count = 2
        if isinstance(curve, (BSplineCurve, NURBSCurve)):
            minimum_count = curve.degree + 1

        if self._curve_point_count(curve) <= minimum_count:
            self.status_label.setText("Cannot delete point: curve would become invalid")
            return

        points = np.delete(curve.control_points, point_index, axis=0)

        weights = None
        if isinstance(curve, NURBSCurve):
            weights = np.delete(curve.weights, point_index)

        knot_mode = self.curve_knot_modes[self.active_curve_index]
        self.curves[self.active_curve_index] = self._rebuild_curve_from_components(
            curve,
            points,
            weights,
            knot_mode,
        )

        self.active_curve_point_index = min(point_index, points.shape[0] - 1)
        self._invalidate_generated_geometry_from_curve_edit()
        self._refresh_curve_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText("Curve point deleted")

    def _on_apply_curve_point(self) -> None:
        """Apply coordinate and optional weight edits to active curve point."""
        curve = self._get_active_curve()
        if curve is None or self.active_curve_point_index is None:
            self.status_label.setText("Select a curve point to edit")
            return

        point_index = int(self.active_curve_point_index)
        if not (0 <= point_index < self._curve_point_count(curve)):
            self.status_label.setText("Select a valid curve point to edit")
            return

        new_point = np.array(
            [
                self.curve_point_x_spin.value(),
                self.curve_point_y_spin.value(),
                self.curve_point_z_spin.value(),
            ],
            dtype=float,
        )
        if self.curve_plane_lock_checkbox.isChecked():
            new_point[2] = 0.0

        if isinstance(curve, NURBSCurve):
            new_weight = float(self.curve_point_weight_spin.value())
            curve.update_control_point_inplace(point_index, new_point, new_weight)
        else:
            curve.update_control_point_inplace(point_index, new_point)

        self._invalidate_generated_geometry_from_curve_edit()
        self._refresh_curve_list_widget()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=False, redraw_toolpath=False)
        self.status_label.setText("Curve point updated")

    def _build_control_point_group(self, parent_layout) -> None:
        """Create widgets for control point coordinate and weight editing.

        Inputs:
            parent_layout: Layout receiving the controls group.

        Outputs:
            None.

        Behavior:
            Adds row/column selectors and numeric spin boxes, then binds updates
            to rebuild the NURBS surface.
        """
        group = QtWidgets.QGroupBox("Control Point Editor")
        self.control_point_group = group
        layout = QtWidgets.QFormLayout(group)

        self.row_spin = QtWidgets.QSpinBox()
        self.row_spin.setRange(0, 0)
        self.col_spin = QtWidgets.QSpinBox()
        self.col_spin.setRange(0, 0)

        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin = QtWidgets.QDoubleSpinBox()

        for spin in (self.x_spin, self.y_spin, self.z_spin):
            spin.setRange(-500.0, 500.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.5)

        self.weight_spin.setRange(0.05, 50.0)
        self.weight_spin.setDecimals(3)
        self.weight_spin.setSingleStep(0.1)

        self.drag_sensitivity_spin = QtWidgets.QDoubleSpinBox()
        self.drag_sensitivity_spin.setRange(0.1, 2.0)
        self.drag_sensitivity_spin.setDecimals(2)
        self.drag_sensitivity_spin.setSingleStep(0.05)
        self.drag_sensitivity_spin.setValue(self._drag_sensitivity)
        self.drag_sensitivity_spin.valueChanged.connect(self._on_drag_sensitivity_changed)

        self.show_control_net_checkbox = QtWidgets.QCheckBox("Show control net lines")
        self.show_control_net_checkbox.setChecked(self._control_net_visible)
        self.show_control_net_checkbox.toggled.connect(self._on_toggle_control_net)

        self.apply_point_button = QtWidgets.QPushButton("Apply Point Update")
        self.apply_point_button.clicked.connect(self._on_apply_control_point)
        self.row_spin.valueChanged.connect(self._sync_spin_boxes_from_state)
        self.col_spin.valueChanged.connect(self._sync_spin_boxes_from_state)

        layout.addRow("Row index", self.row_spin)
        layout.addRow("Column index", self.col_spin)
        layout.addRow("X", self.x_spin)
        layout.addRow("Y", self.y_spin)
        layout.addRow("Z", self.z_spin)
        layout.addRow("Weight", self.weight_spin)
        layout.addRow("Drag sensitivity", self.drag_sensitivity_spin)
        layout.addRow(self.show_control_net_checkbox)
        layout.addRow(self.apply_point_button)

        parent_layout.addWidget(group)
        self._register_major_section(group, expanded_by_default=True)
        self._configure_control_editor_ranges()
        self._sync_spin_boxes_from_state()

    def _build_surface_generation_group(self, parent_layout) -> None:
        """Create widgets for explicit surface generation from input settings.

        Inputs:
            parent_layout: Layout receiving the controls group.

        Outputs:
            None.

        Behavior:
            Supports first-pass generation from a control-net size and keeps
            surface creation explicit through a dedicated action button.
        """
        group = QtWidgets.QGroupBox("Surface Generation")
        self.surface_generation_group = group
        layout = QtWidgets.QFormLayout(group)

        self.surface_source_combo = QtWidgets.QComboBox()
        self.surface_source_combo.addItem("Control Net Demo", "control-net")
        self.surface_source_combo.addItem("Loft From Curves", "curve-loft")
        self.surface_source_combo.addItem("Extrude Active Curve", "curve-extrude")
        self.surface_source_combo.currentIndexChanged.connect(self._on_surface_source_changed)

        self.surface_rows_spin = QtWidgets.QSpinBox()
        self.surface_rows_spin.setRange(2, 25)
        self.surface_rows_spin.setValue(4)

        self.surface_cols_spin = QtWidgets.QSpinBox()
        self.surface_cols_spin.setRange(2, 25)
        self.surface_cols_spin.setValue(4)

        self.surface_curve_samples_spin = QtWidgets.QSpinBox()
        self.surface_curve_samples_spin.setRange(10, 200)
        self.surface_curve_samples_spin.setValue(40)

        self.surface_extrude_axis_combo = QtWidgets.QComboBox()
        self.surface_extrude_axis_combo.addItem("X+", "x+")
        self.surface_extrude_axis_combo.addItem("Y+", "y+")
        self.surface_extrude_axis_combo.addItem("Z+", "z+")
        self.surface_extrude_axis_combo.setCurrentIndex(2)

        self.surface_extrude_height_spin = QtWidgets.QDoubleSpinBox()
        self.surface_extrude_height_spin.setRange(1.0, 500.0)
        self.surface_extrude_height_spin.setValue(30.0)
        self.surface_extrude_height_spin.setDecimals(3)

        self.surface_extrude_layers_spin = QtWidgets.QSpinBox()
        self.surface_extrude_layers_spin.setRange(2, 80)
        self.surface_extrude_layers_spin.setValue(8)

        self.generate_surface_button = QtWidgets.QPushButton("Generate Surface")
        self.generate_surface_button.clicked.connect(self._on_generate_surface)

        for widget in (
            self.surface_rows_spin,
            self.surface_cols_spin,
            self.surface_curve_samples_spin,
            self.surface_extrude_height_spin,
            self.surface_extrude_layers_spin,
        ):
            widget.valueChanged.connect(lambda _value: self._sync_ui_enabled_state())
        self.surface_extrude_axis_combo.currentIndexChanged.connect(self._sync_ui_enabled_state)

        layout.addRow("Source", self.surface_source_combo)
        layout.addRow("Rows", self.surface_rows_spin)
        layout.addRow("Columns", self.surface_cols_spin)
        layout.addRow("Curve samples", self.surface_curve_samples_spin)
        layout.addRow("Extrude axis", self.surface_extrude_axis_combo)
        layout.addRow("Extrude height", self.surface_extrude_height_spin)
        layout.addRow("Extrude layers", self.surface_extrude_layers_spin)
        layout.addRow(self.generate_surface_button)

        parent_layout.addWidget(group)
        self._register_major_section(group, expanded_by_default=True)

    def _on_surface_source_changed(self, _index: int) -> None:
        """React to surface-source mode changes in generation controls."""
        self._sync_ui_enabled_state()

    def _build_surface_operations_group(self, parent_layout) -> None:
        """Create widgets for non-destructive active-surface operations."""
        group = QtWidgets.QGroupBox("Surface Operations")
        self.surface_operations_group = group
        layout = QtWidgets.QFormLayout(group)

        self.surface_selector_combo = QtWidgets.QComboBox()
        self.surface_selector_combo.currentIndexChanged.connect(self._on_surface_selector_changed)

        self.surface_extrusion_mode_combo = QtWidgets.QComboBox()
        self.surface_extrusion_mode_combo.addItem("Surface Offset (Legacy)", "offset")
        self.surface_extrusion_mode_combo.addItem("Shell Extrusion (Hollow)", "shell")
        self.surface_extrusion_mode_combo.addItem("Full Solid Extrusion", "solid")
        self.surface_extrusion_mode_combo.currentIndexChanged.connect(self._sync_ui_enabled_state)

        self.surface_normal_extrude_distance_spin = QtWidgets.QDoubleSpinBox()
        self.surface_normal_extrude_distance_spin.setRange(0.1, 500.0)
        self.surface_normal_extrude_distance_spin.setValue(12.0)
        self.surface_normal_extrude_distance_spin.setDecimals(3)
        self.surface_normal_extrude_distance_spin.valueChanged.connect(
            lambda _value: self._sync_ui_enabled_state()
        )

        self.surface_normal_replace_checkbox = QtWidgets.QCheckBox("Replace active surface (offset mode only)")
        self.surface_normal_replace_checkbox.setChecked(False)

        self.extrude_selected_surface_button = QtWidgets.QPushButton("Extrude Selected Surface")
        self.extrude_selected_surface_button.clicked.connect(self._on_extrude_selected_surface)

        layout.addRow("Active surface", self.surface_selector_combo)
        layout.addRow("Extrusion mode", self.surface_extrusion_mode_combo)
        layout.addRow("Extrude distance", self.surface_normal_extrude_distance_spin)
        layout.addRow(self.surface_normal_replace_checkbox)
        layout.addRow(self.extrude_selected_surface_button)

        parent_layout.addWidget(group)
        self._register_major_section(group, expanded_by_default=False)
        self._refresh_surface_selector_widget()

    def _on_surface_selector_changed(self, _index: int) -> None:
        """Set active surface from selector and refresh dependent viewport/UI."""
        selected_surface_id = self.surface_selector_combo.currentData()
        if selected_surface_id is None:
            return
        if not self._set_active_surface(int(selected_surface_id)):
            return

        self.generated_passes = []
        self._clear_toolpath_actors()
        self._refresh_scene(reset_camera=False, recompute_surface=True, redraw_toolpath=False)

    def _on_extrude_selected_surface(self) -> None:
        """Generate derived geometry from active surface with selected extrusion mode."""
        if not self._has_active_surface():
            self.status_label.setText("Select or generate a surface before extrusion")
            return

        distance = float(self.surface_normal_extrude_distance_spin.value())
        if distance <= 0.0:
            self.status_label.setText("Extrusion distance must be positive")
            return

        source_surface_id = self.active_surface_id
        source_surface = self.surface
        if source_surface is None:
            self.status_label.setText("No active surface available for extrusion")
            return

        extrusion_mode = str(self.surface_extrusion_mode_combo.currentData())
        try:
            if extrusion_mode == "offset":
                derived_surface = extrude_surface_along_center_normal(source_surface, distance=distance)
                if self.surface_normal_replace_checkbox.isChecked() and source_surface_id is not None:
                    for entry in self.surface_entries:
                        if int(entry["surface_id"]) != int(source_surface_id):
                            continue
                        entry["surface"] = derived_surface
                        entry["operation_tag"] = "surface-normal-extrude-replace"
                        entry["name"] = f"Surface {source_surface_id} (normal extrude)"
                        break
                    self._invalidate_active_surface_render_cache()
                    self._set_active_surface(source_surface_id)
                else:
                    base_name = "Surface normal extrusion"
                    self._register_surface_entry(
                        surface_object=derived_surface,
                        name=f"{base_name} {self._next_surface_id}",
                        operation_tag="surface-normal-extrude",
                        source_surface_id=source_surface_id,
                        set_active=True,
                    )
                status_text = "Derived offset surface generated from active surface normal"
            else:
                vertices, triangles = build_surface_extrusion_mesh(
                    source_surface,
                    distance=distance,
                    mode=extrusion_mode,
                    u_samples=DEFAULT_SURFACE_SAMPLES_U,
                    v_samples=DEFAULT_SURFACE_SAMPLES_V,
                )
                self._register_derived_body_entry(
                    vertices=vertices,
                    triangles=triangles,
                    name=f"{extrusion_mode.title()} Body {len(self._derived_body_entries) + 1}",
                    mode=extrusion_mode,
                    source_surface_id=source_surface_id,
                )
                status_text = f"Generated {extrusion_mode} extrusion body from active surface"
        except ValueError as error:
            self.status_label.setText(f"Surface extrusion failed: {error}")
            return

        self.generated_passes = []
        self._clear_toolpath_actors()
        self._refresh_scene(reset_camera=False, recompute_surface=True, redraw_toolpath=False)
        self.status_label.setText(status_text)

    def _build_toolpath_group(self, parent_layout) -> None:
        """Create widgets for zig-zag toolpath generation parameters.

        Inputs:
            parent_layout: Layout receiving the controls group.

        Outputs:
            None.

        Behavior:
            Adds controls for stepover and tool radius and binds button actions
            that generate and draw cutter-location passes.
        """
        group = QtWidgets.QGroupBox("Toolpath")
        self.toolpath_group = group
        layout = QtWidgets.QFormLayout(group)

        self.stepover_spin = QtWidgets.QDoubleSpinBox()
        self.stepover_spin.setRange(0.1, 20.0)
        self.stepover_spin.setValue(DEFAULT_STEPOVER_MM)
        self.stepover_spin.setDecimals(3)

        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 50.0)
        self.radius_spin.setValue(DEFAULT_TOOL_RADIUS_MM)
        self.radius_spin.setDecimals(3)

        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 5.0)
        self.tolerance_spin.setValue(COLLINEAR_TOLERANCE_MM)
        self.tolerance_spin.setDecimals(3)

        self.link_passes_check = QtWidgets.QCheckBox("Link passes on surface (G1)")
        self.link_passes_check.setChecked(False)

        self.generate_toolpath_button = QtWidgets.QPushButton("Generate Zig-Zag Toolpath")
        self.generate_toolpath_button.clicked.connect(self._on_generate_toolpath)

        layout.addRow("Stepover (mm)", self.stepover_spin)
        layout.addRow("Tool radius (mm)", self.radius_spin)
        layout.addRow("Chord tolerance (mm)", self.tolerance_spin)
        layout.addRow("Pass linking", self.link_passes_check)
        layout.addRow(self.generate_toolpath_button)

        parent_layout.addWidget(group)
        self._register_major_section(group, expanded_by_default=False)

    def _build_export_group(self, parent_layout) -> None:
        """Create widgets for G-code settings and file export.

        Inputs:
            parent_layout: Layout receiving the controls group.

        Outputs:
            None.

        Behavior:
            Captures feed/safe-height settings and allows writing generated
            toolpaths to a user-selected G-code file.
        """
        group = QtWidgets.QGroupBox("G-code Export")
        self.export_group = group
        layout = QtWidgets.QFormLayout(group)

        self.feed_spin = QtWidgets.QDoubleSpinBox()
        self.feed_spin.setRange(10.0, 5000.0)
        self.feed_spin.setValue(DEFAULT_FEED_RATE_MM_PER_MIN)
        self.feed_spin.setDecimals(1)

        self.plunge_spin = QtWidgets.QDoubleSpinBox()
        self.plunge_spin.setRange(10.0, 5000.0)
        self.plunge_spin.setValue(DEFAULT_PLUNGE_RATE_MM_PER_MIN)
        self.plunge_spin.setDecimals(1)

        self.safe_z_spin = QtWidgets.QDoubleSpinBox()
        self.safe_z_spin.setRange(0.1, 500.0)
        self.safe_z_spin.setValue(DEFAULT_SAFE_Z_MM)
        self.safe_z_spin.setDecimals(3)

        self.spindle_spin = QtWidgets.QSpinBox()
        self.spindle_spin.setRange(100, 40000)
        self.spindle_spin.setValue(DEFAULT_SPINDLE_RPM)

        self.export_gcode_button = QtWidgets.QPushButton("Export G-code")
        self.export_gcode_button.clicked.connect(self._on_export_gcode)

        layout.addRow("Feed (mm/min)", self.feed_spin)
        layout.addRow("Plunge (mm/min)", self.plunge_spin)
        layout.addRow("Safe Z (mm)", self.safe_z_spin)
        layout.addRow("Spindle RPM", self.spindle_spin)
        layout.addRow(self.export_gcode_button)

        parent_layout.addWidget(group)
        self._register_major_section(group, expanded_by_default=False)

    def _on_generate_surface(self) -> None:
        """Create a surface explicitly from the configured input dimensions.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Creates a surface from the selected source mode (control-net demo
            or curve loft), resets stale passes, and redraws scene geometry.
        """
        if not self._has_valid_surface_input():
            self.status_label.setText("Provide valid source geometry before generating a surface")
            return

        source_mode = self.surface_source_combo.currentData()

        try:
            if source_mode == "control-net":
                generated_surface = create_default_surface(
                    rows=int(self.surface_rows_spin.value()),
                    cols=int(self.surface_cols_spin.value()),
                )
                surface_name = f"Control Net Surface {self._next_surface_id}"
                operation_tag = "control-net"
            elif source_mode == "curve-loft":
                generated_surface = loft_surface_from_curves(
                    self.curves,
                    samples_per_curve=int(self.surface_curve_samples_spin.value()),
                )
                surface_name = f"Loft Surface {self._next_surface_id}"
                operation_tag = "curve-loft"
            else:
                active_curve = self._get_active_curve()
                if active_curve is None:
                    raise ValueError("select an active curve before extrusion")

                axis_key = str(self.surface_extrude_axis_combo.currentData())
                axis_map = {
                    "x+": np.array([1.0, 0.0, 0.0], dtype=float),
                    "y+": np.array([0.0, 1.0, 0.0], dtype=float),
                    "z+": np.array([0.0, 0.0, 1.0], dtype=float),
                }
                axis_direction = axis_map.get(axis_key, np.array([0.0, 0.0, 1.0], dtype=float))

                generated_surface = extrude_surface_from_curve(
                    active_curve,
                    direction=axis_direction,
                    height=float(self.surface_extrude_height_spin.value()),
                    layer_count=int(self.surface_extrude_layers_spin.value()),
                    samples_along_curve=int(self.surface_curve_samples_spin.value()),
                )
                surface_name = f"Curve Extrusion Surface {self._next_surface_id}"
                operation_tag = "curve-extrude"
        except ValueError as error:
            self.status_label.setText(f"Surface generation failed: {error}")
            return

        self._register_surface_entry(
            surface_object=generated_surface,
            name=surface_name,
            operation_tag=operation_tag,
            source_surface_id=None,
            set_active=True,
        )

        if self._curve_interaction_mode == "drag":
            select_index = self.curve_interaction_mode_combo.findData("select")
            if select_index >= 0:
                blocked = self.curve_interaction_mode_combo.blockSignals(True)
                self.curve_interaction_mode_combo.setCurrentIndex(int(select_index))
                self.curve_interaction_mode_combo.blockSignals(blocked)
            self._curve_interaction_mode = "select"
            self._is_curve_dragging = False
            self._curve_drag_point_index = None

        self.generated_passes = []
        self._clear_toolpath_actors()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=True, recompute_surface=True, redraw_toolpath=False)

        if source_mode == "control-net":
            self.status_label.setText("Surface generated from configured control net")
        elif source_mode == "curve-loft":
            self.status_label.setText("Surface generated by lofting selected curves")
        else:
            self.status_label.setText("Surface generated by extruding active curve")

    def _sync_spin_boxes_from_state(self) -> None:
        """Synchronize editor spin boxes with selected control point state.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Reads currently selected control point and updates coordinate and
            weight widgets without triggering surface recomputation.
        """
        if not self._has_active_surface():
            for widget, value in (
                (self.x_spin, 0.0),
                (self.y_spin, 0.0),
                (self.z_spin, 0.0),
                (self.weight_spin, 1.0),
            ):
                blocked = widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(blocked)
            self._set_selected_control_point(None)
            return

        i_u = int(self.row_spin.value())
        i_v = int(self.col_spin.value())

        u_count, v_count = self.surface.control_net.shape[:2]
        i_u = max(0, min(i_u, int(u_count) - 1))
        i_v = max(0, min(i_v, int(v_count) - 1))

        self._set_spinboxes_for_control_point(
            i_u,
            i_v,
            update_index_selectors=True,
        )
        self._set_selected_control_point(self._uv_to_flat_index(i_u, i_v))

    def _set_spinboxes_for_control_point(
        self,
        i_u: int,
        i_v: int,
        update_index_selectors: bool,
    ) -> None:
        """Synchronize row/column and numeric editor controls for one point.

        Inputs:
            i_u: Control-point row index.
            i_v: Control-point column index.
            update_index_selectors: Whether to overwrite row/column selector values.

        Outputs:
            None.

        Side effects:
            Updates Qt spin-box values while blocking emitted signals to avoid
            recursive callbacks.
        """
        if not self._has_active_surface():
            return

        u_count, v_count = self.surface.control_net.shape[:2]
        safe_i_u = max(0, min(int(i_u), int(u_count) - 1))
        safe_i_v = max(0, min(int(i_v), int(v_count) - 1))

        if update_index_selectors or safe_i_u != int(i_u) or safe_i_v != int(i_v):
            for widget, value in ((self.row_spin, safe_i_u), (self.col_spin, safe_i_v)):
                blocked = widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(blocked)

        point = self.surface.control_net[safe_i_u, safe_i_v]
        weight = self.surface.weights[safe_i_u, safe_i_v]

        for widget, value in (
            (self.x_spin, point[0]),
            (self.y_spin, point[1]),
            (self.z_spin, point[2]),
            (self.weight_spin, weight),
        ):
            blocked = widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(blocked)

    def _on_apply_control_point(self) -> None:
        """Apply edited coordinates and weight to the selected control point.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Applies edits to the selected control point in-place, clears stale
            toolpath overlays, and triggers a full surface refresh.
        """
        if not self._has_active_surface():
            self.status_label.setText("Generate a surface before editing control points")
            return

        i_u = int(self.row_spin.value())
        i_v = int(self.col_spin.value())

        u_count, v_count = self.surface.control_net.shape[:2]
        if not (0 <= i_u < int(u_count) and 0 <= i_v < int(v_count)):
            self.status_label.setText(f"Invalid control-point index ({i_u}, {i_v}) for current surface")
            return

        new_point = np.array(
            [self.x_spin.value(), self.y_spin.value(), self.z_spin.value()],
            dtype=float,
        )
        new_weight = float(self.weight_spin.value())
        try:
            self.surface.update_control_point_inplace(i_u, i_v, new_point, new_weight)
        except (IndexError, ValueError) as error:
            self.status_label.setText(f"Control-point update failed: {error}")
            return

        flat_index = self._uv_to_flat_index(i_u, i_v)
        self._set_selected_control_point(flat_index)

        self.generated_passes = []
        self._clear_toolpath_actors()
        self._sync_ui_enabled_state()
        self._refresh_scene(reset_camera=False, recompute_surface=True, redraw_toolpath=False)
        self.status_label.setText("Updated selected control point")

    def _uv_to_flat_index(self, i_u: int, i_v: int) -> int:
        """Convert control-net (u, v) indices to flattened point index.

        Inputs:
            i_u: Row index in u direction.
            i_v: Column index in v direction.

        Outputs:
            Flattened index in row-major order.

        Side effects:
            None.
        """
        if not self._has_active_surface():
            raise RuntimeError("cannot compute control-point index without active surface")
        _, v_count = self.surface.control_net.shape[:2]
        return i_u * v_count + i_v

    def _flat_index_to_uv(self, flat_index: int) -> tuple[int, int]:
        """Convert flattened control-point index back to (u, v) indices.

        Inputs:
            flat_index: Flattened row-major index.

        Outputs:
            Tuple (i_u, i_v) representing control-net coordinates.

        Side effects:
            None.
        """
        if not self._has_active_surface():
            raise RuntimeError("cannot compute control-point coordinates without active surface")
        _, v_count = self.surface.control_net.shape[:2]
        return divmod(flat_index, v_count)

    def _make_control_drag_callback(self, i_u: int, i_v: int):
        """Create and return a sphere-widget drag callback bound to one point.

        Inputs:
            i_u: Row index in the control net.
            i_v: Column index in the control net.

        Outputs:
            Callable accepted by PyVista sphere-widget API.

        Side effects:
            None at creation time. The returned callback mutates control points
            and scene geometry while dragging.
        """

        def _callback(center: tuple[float, float, float], widget: object) -> None:
            self._on_control_point_drag(i_u, i_v, np.asarray(center, dtype=float), widget)

        return _callback

    def _initialize_control_point_widgets(self) -> None:
        """Create draggable sphere widgets for all control points.

        Inputs:
            None.

        Outputs:
            None.

        Side effects:
            Clears existing sphere widgets from the plotter and re-adds one
            draggable marker per control point, each wired to drag callbacks.
        """
        if not self._has_active_surface():
            self._control_point_widgets.clear()
            self._selected_control_flat_index = None
            return

        self._control_point_widgets.clear()

        control_points = self.surface.control_net.reshape(-1, 3)
        for flat_index, point in enumerate(control_points):
            i_u, i_v = self._flat_index_to_uv(flat_index)
            widget = self.plotter.add_sphere_widget(
                callback=self._make_control_drag_callback(i_u, i_v),
                center=point.tolist(),
                radius=self._default_widget_radius,
                color=self._default_widget_color,
                selected_color=self._selected_widget_color,
                pass_widget=True,
                test_callback=False,
                interaction_event="always",
            )
            self._control_point_widgets.append(widget)

        if self._selected_control_flat_index is None and self._control_point_widgets:
            self._selected_control_flat_index = 0
        self._set_selected_control_point(self._selected_control_flat_index)

    def _set_selected_control_point(self, flat_index: int | None) -> None:
        """Highlight the active control-point widget and de-highlight others.

        Inputs:
            flat_index: Selected flattened index, or None to clear selection.

        Outputs:
            None.

        Side effects:
            Updates sphere-widget color and radius properties in the viewport.
        """
        self._selected_control_flat_index = flat_index
        for widget_index, widget in enumerate(self._control_point_widgets):
            prop = widget.GetSphereProperty()
            if flat_index is not None and widget_index == flat_index:
                prop.SetColor(*self._selected_widget_color)
                widget.SetRadius(self._selected_widget_radius)
            else:
                prop.SetColor(*self._default_widget_color)
                widget.SetRadius(self._default_widget_radius)

    def _sync_widget_positions_from_control_net(self) -> None:
        """Move sphere-widget centers to match current control-point arrays.

        Inputs:
            None.

        Outputs:
            None.

        Side effects:
            Programmatically updates widget centers while suppressing recursive
            drag callback handling.
        """
        if not self._has_active_surface() or not self._control_point_widgets:
            return

        self._is_syncing_widgets = True
        try:
            control_points = self.surface.control_net.reshape(-1, 3)
            for widget, point in zip(self._control_point_widgets, control_points):
                widget.SetCenter(float(point[0]), float(point[1]), float(point[2]))
        finally:
            self._is_syncing_widgets = False

    def _on_control_point_drag(
        self,
        i_u: int,
        i_v: int,
        new_center: np.ndarray,
        widget: object,
    ) -> None:
        """Handle interactive sphere-widget drag updates for one control point.

        Inputs:
            i_u: Row index of the dragged control point.
            i_v: Column index of the dragged control point.
            new_center: New 3D center reported by the widget.
            widget: Widget instance generating the callback.

        Outputs:
            None.

        Side effects:
            Mutates control-point geometry, updates surface and control-net mesh
            geometry in real time, clears stale toolpath overlays, and refreshes
            UI controls and status text.
        """
        if not self._has_active_surface() or self._is_syncing_widgets:
            return

        u_count, v_count = self.surface.control_net.shape[:2]
        if not (0 <= i_u < int(u_count) and 0 <= i_v < int(v_count)):
            return

        current_time = time.perf_counter()
        if current_time - self._last_drag_update_timestamp < self._drag_update_interval_sec:
            return
        self._last_drag_update_timestamp = current_time

        current_point = self.surface.control_net[i_u, i_v].copy()
        adjusted_point = current_point + self._drag_sensitivity * (new_center - current_point)

        current_weight = float(self.surface.weights[i_u, i_v])
        try:
            self.surface.update_control_point_inplace(i_u, i_v, adjusted_point, current_weight)
        except (IndexError, ValueError):
            return

        if self._drag_sensitivity != 1.0:
            self._is_syncing_widgets = True
            try:
                widget.SetCenter(
                    float(adjusted_point[0]),
                    float(adjusted_point[1]),
                    float(adjusted_point[2]),
                )
            finally:
                self._is_syncing_widgets = False

        flat_index = self._uv_to_flat_index(i_u, i_v)
        self._set_selected_control_point(flat_index)
        self._set_spinboxes_for_control_point(i_u, i_v, update_index_selectors=True)

        if self.generated_passes:
            self.generated_passes = []
            self._clear_toolpath_actors()
            self._sync_ui_enabled_state()

        self._update_surface_mesh_geometry()
        self._update_control_net_geometry()
        self.status_label.setText(f"Dragging control point ({i_u}, {i_v})")
        self.plotter.render()

    def _clear_curve_actors(self) -> None:
        """Remove all rendered curve, hull, and curve-point actors."""
        self._curve_actor_to_curve_id.clear()

        for actor in self._curve_actors:
            self.plotter.remove_actor(actor)
        self._curve_actors.clear()

        for actor in self._curve_hull_actors:
            self.plotter.remove_actor(actor)
        self._curve_hull_actors.clear()

        for actor in self._curve_point_actors:
            self.plotter.remove_actor(actor)
        self._curve_point_actors.clear()

    def _draw_curve_overlays(self) -> None:
        """Render all curves with control polygons and control points."""
        self._clear_curve_actors()

        for curve_index, curve in enumerate(self.curves):
            curve_id = self.curve_ids[curve_index] if curve_index < len(self.curve_ids) else curve_index
            control_points = np.asarray(curve.control_points, dtype=float)
            if control_points.shape[0] < 2:
                continue

            is_active = self.active_curve_id == curve_id
            is_selected = curve_id in self.selected_curve_ids
            if is_active:
                hull_color = "#f59e0b"
                curve_color = "#dc2626"
                point_color = "#f97316"
            elif is_selected:
                hull_color = "#a78bfa"
                curve_color = "#6d28d9"
                point_color = "#7c3aed"
            else:
                hull_color = "#9ca3af"
                curve_color = "#2563eb"
                point_color = "#475569"

            hull_mesh = pv.lines_from_points(control_points, close=False)
            hull_actor = self.plotter.add_mesh(hull_mesh, color=hull_color, line_width=2)
            self._curve_hull_actors.append(hull_actor)
            self._curve_actor_to_curve_id[id(hull_actor)] = int(curve_id)

            sampled_points = curve.sample_points(120)
            curve_mesh = pv.lines_from_points(sampled_points, close=False)
            curve_actor = self.plotter.add_mesh(curve_mesh, color=curve_color, line_width=4)
            self._curve_actors.append(curve_actor)
            self._curve_actor_to_curve_id[id(curve_actor)] = int(curve_id)

            points_actor = self.plotter.add_points(
                control_points,
                color=point_color,
                point_size=16,
                render_points_as_spheres=True,
            )
            self._curve_point_actors.append(points_actor)
            self._curve_actor_to_curve_id[id(points_actor)] = int(curve_id)

    def _clear_surface_visuals(self) -> None:
        """Remove all surface-derived actors/widgets while keeping scene helpers.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Clears surface mesh, control net, and control-point widgets when the
            application is in no-active-surface mode.
        """
        if self._surface_actor is not None:
            self.plotter.remove_actor(self._surface_actor)
            self._surface_actor = None
        self._surface_mesh = None

        if self._control_net_actor is not None:
            self.plotter.remove_actor(self._control_net_actor)
            self._control_net_actor = None
        self._control_net_mesh = None

        self._selected_control_flat_index = None
        self._rendered_surface_id = None

    def _estimate_reference_plane_extent(self) -> float:
        """Return a half-extent that keeps the XY reference plane usable.

        Inputs:
            None.

        Outputs:
            Positive half-extent for a square XY plane.

        Behavior:
            Uses a fixed baseline and expands to comfortably include active
            surface control points when available.
        """
        extent = 120.0

        if self.curves:
            all_curve_xy = np.vstack([curve.control_points[:, :2] for curve in self.curves])
            max_curve_axis = float(np.max(np.abs(all_curve_xy)))
            extent = max(extent, max_curve_axis + 30.0)

        if self._has_active_surface():
            xy_coords = self.surface.control_net[:, :, :2].reshape(-1, 2)
            max_axis = float(np.max(np.abs(xy_coords)))
            extent = max(extent, max_axis + 30.0)
        return extent

    def _ensure_reference_plane_geometry(self) -> None:
        """Create or update the permanent XY reference plane actor.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Keeps a non-modeling CAD-style plane visible at Z=0 for spatial
            orientation before and after surface creation.
        """
        desired_extent = self._estimate_reference_plane_extent()
        needs_rebuild = (
            self._reference_plane_mesh is None
            or abs(desired_extent - self._reference_plane_extent) > 1e-6
        )

        if needs_rebuild:
            self._reference_plane_extent = desired_extent
            self._reference_plane_mesh = pv.Plane(
                center=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, 1.0),
                i_size=2.0 * desired_extent,
                j_size=2.0 * desired_extent,
                i_resolution=30,
                j_resolution=30,
            )

            if self._reference_plane_actor is not None:
                self.plotter.remove_actor(self._reference_plane_actor)

            self._reference_plane_actor = self.plotter.add_mesh(
                self._reference_plane_mesh,
                color="#dbeafe",
                opacity=0.30,
                show_edges=True,
                edge_color="#94a3b8",
                line_width=1,
                pickable=False,
            )

        if self._reference_plane_actor is not None:
            self._reference_plane_actor.SetVisibility(1)

    def _create_control_net_mesh(self) -> object:
        """Build PolyData for control-net grid lines from current control points.

        Inputs:
            None.

        Outputs:
            PyVista PolyData containing control-net line connectivity.

        Side effects:
            None.
        """
        if not self._has_active_surface():
            raise RuntimeError("cannot create control-net mesh without active surface")

        control_points = self.surface.control_net
        u_count, v_count = control_points.shape[:2]

        points_flat = control_points.reshape(-1, 3).copy()
        lines: list[int] = []

        for i_u in range(u_count):
            for i_v in range(v_count - 1):
                start = self._uv_to_flat_index(i_u, i_v)
                end = self._uv_to_flat_index(i_u, i_v + 1)
                lines.extend([2, start, end])

        for i_v in range(v_count):
            for i_u in range(u_count - 1):
                start = self._uv_to_flat_index(i_u, i_v)
                end = self._uv_to_flat_index(i_u + 1, i_v)
                lines.extend([2, start, end])

        mesh = pv.PolyData()
        mesh.points = points_flat
        if lines:
            mesh.lines = np.asarray(lines, dtype=np.int64)
        return mesh

    def _create_polydata_from_triangles(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> object | None:
        """Build one PyVista PolyData from explicit triangle-index geometry."""
        if pv is None:
            return None

        verts = np.asarray(vertices, dtype=float)
        tris = np.asarray(triangles, dtype=np.int64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            return None
        if tris.ndim != 2 or tris.shape[1] != 3:
            return None
        if tris.size == 0:
            return None
        if np.min(tris) < 0 or np.max(tris) >= verts.shape[0]:
            return None

        faces = np.hstack(
            (
                np.full((tris.shape[0], 1), 3, dtype=np.int64),
                tris,
            )
        ).ravel()
        return pv.PolyData(verts, faces)

    def _update_surface_mesh_geometry(self) -> None:
        """Recompute and apply surface geometry on the existing mesh actor.

        Inputs:
            None.

        Outputs:
            None.

        Side effects:
            Evaluates the NURBS grid and mutates surface mesh points in-place,
            creating the actor only once on first call.
        """
        if not self._has_active_surface():
            if self._surface_actor is not None:
                self.plotter.remove_actor(self._surface_actor)
                self._surface_actor = None
            self._surface_mesh = None
            self._rendered_surface_id = None
            return

        if self._rendered_surface_id != self.active_surface_id:
            if self._surface_actor is not None:
                self.plotter.remove_actor(self._surface_actor)
                self._surface_actor = None
            self._surface_mesh = None

        points, _, _, _ = self.surface.evaluate_grid(
            u_samples=DEFAULT_SURFACE_SAMPLES_U,
            v_samples=DEFAULT_SURFACE_SAMPLES_V,
        )

        if self._surface_mesh is None:
            x_grid = points[:, :, 0]
            y_grid = points[:, :, 1]
            z_grid = points[:, :, 2]
            self._surface_mesh = pv.StructuredGrid(x_grid, y_grid, z_grid)
            self._surface_actor = self.plotter.add_mesh(
                self._surface_mesh,
                color="#7eb6ff",
                opacity=0.85,
                show_edges=True,
                edge_color="#2f4f6f",
            )
            self._rendered_surface_id = self.active_surface_id
            return

        flat_points = np.column_stack(
            (
                points[:, :, 0].ravel(order="F"),
                points[:, :, 1].ravel(order="F"),
                points[:, :, 2].ravel(order="F"),
            )
        )
        self._surface_mesh.points = flat_points
        self._surface_mesh.Modified()
        self._rendered_surface_id = self.active_surface_id

    def _update_control_net_geometry(self) -> None:
        """Update control-net line geometry and visibility state.

        Inputs:
            None.

        Outputs:
            None.

        Side effects:
            Creates control-net actor once and then updates PolyData points
            in-place as control points move.
        """
        if not self._has_active_surface():
            if self._control_net_actor is not None:
                self.plotter.remove_actor(self._control_net_actor)
                self._control_net_actor = None
            self._control_net_mesh = None
            return

        if self._control_net_mesh is None:
            self._control_net_mesh = self._create_control_net_mesh()
            self._control_net_actor = self.plotter.add_mesh(
                self._control_net_mesh,
                color="#1d3557",
                line_width=2,
            )
        else:
            self._control_net_mesh.points = self.surface.control_net.reshape(-1, 3)
            self._control_net_mesh.Modified()

        if self._control_net_actor is not None:
            self._control_net_actor.SetVisibility(1 if self._control_net_visible else 0)

    def _on_drag_sensitivity_changed(self, value: float) -> None:
        """Update drag sensitivity factor used during widget movement.

        Inputs:
            value: New sensitivity multiplier in range [0.1, 2.0].

        Outputs:
            None.

        Side effects:
            Changes how far control points move relative to raw widget movement.
        """
        self._drag_sensitivity = float(value)

    def _on_toggle_control_net(self, enabled: bool) -> None:
        """Toggle visibility of control-net grid-line actor.

        Inputs:
            enabled: True to show control-net lines, False to hide them.

        Outputs:
            None.

        Side effects:
            Updates actor visibility and refreshes the viewport.
        """
        self._control_net_visible = bool(enabled)
        if self._control_net_actor is not None:
            self._control_net_actor.SetVisibility(1 if enabled else 0)
            self.plotter.render()

    def _refresh_scene(
        self,
        reset_camera: bool,
        recompute_surface: bool = True,
        redraw_toolpath: bool = True,
    ) -> None:
        """Redraw surface mesh, control points, and optional toolpath overlays.

        Inputs:
            reset_camera: Whether the viewport camera should reset.
            recompute_surface: Whether to recompute and apply surface geometry.
            redraw_toolpath: Whether to rebuild toolpath overlay actors.

        Outputs:
            None.

        Behavior:
            Updates mesh geometries without recreating the plotter window. The
            surface and control net are updated in-place, while optional overlays
            can be rebuilt as needed.
        """
        self._ensure_reference_plane_geometry()

        if self._has_active_surface():
            if recompute_surface:
                self._update_surface_mesh_geometry()

            self._update_control_net_geometry()
        else:
            self._clear_surface_visuals()
            self.generated_passes = []
            self._clear_toolpath_actors()

        self._draw_curve_overlays()
        self._draw_derived_body_overlays()
        self._ensure_active_widget_layer()

        if redraw_toolpath:
            self._clear_toolpath_actors()
            self._draw_toolpath_overlay()

        self._sync_ui_enabled_state()

        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def _on_generate_toolpath(self) -> None:
        """Generate zig-zag toolpath from current surface and render overlay.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Computes scanline passes using current machining settings and updates
            visual overlays and status text with pass statistics.
        """
        if not self._has_active_surface():
            self.status_label.setText("Generate a surface before generating toolpath")
            return

        self.generated_passes = generate_zigzag_toolpath(
            surface=self.surface,
            stepover_mm=float(self.stepover_spin.value()),
            tool_radius_mm=float(self.radius_spin.value()),
            tolerance_mm=float(self.tolerance_spin.value()),
            link_passes=self.link_passes_check.isChecked(),
        )
        self._refresh_scene(reset_camera=False)
        self._sync_ui_enabled_state()

        total_points = sum(len(item.points) for item in self.generated_passes)
        self.status_label.setText(
            f"Generated {len(self.generated_passes)} passes with {total_points} CL points"
        )

    def _draw_toolpath_overlay(self) -> None:
        """Draw generated toolpath passes as line segments in the 3D viewport.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Converts each pass into connected line segments and adds them as
            lightweight colored actors for visual verification.
        """
        if not self.generated_passes:
            return

        for tool_pass in self.generated_passes:
            points = np.array([item.cl_point for item in tool_pass.points], dtype=float)
            if len(points) < 2:
                continue

            line_mesh = pv.lines_from_points(points, close=False)
            actor = self.plotter.add_mesh(line_mesh, color="#2a9d8f", line_width=3)
            self._toolpath_actors.append(actor)

            if tool_pass.link_points:
                link_points = np.array(
                    [item.cl_point for item in tool_pass.link_points],
                    dtype=float,
                )
                if len(link_points) >= 2:
                    link_mesh = pv.lines_from_points(link_points, close=False)
                    actor = self.plotter.add_mesh(link_mesh, color="#e76f51", line_width=2)
                    self._toolpath_actors.append(actor)

    def _clear_toolpath_actors(self) -> None:
        """Remove currently displayed toolpath actors from the viewport.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Iterates over cached actor references, removes each from the plotter,
            and clears internal actor storage.
        """
        for actor in self._toolpath_actors:
            self.plotter.remove_actor(actor)
        self._toolpath_actors.clear()

    def _on_export_gcode(self) -> None:
        """Export generated passes to a user-selected G-code output file.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Opens a save-file dialog, generates CNC code from current passes and
            machine settings, writes output, and updates status text.
        """
        if not self._has_active_surface():
            self.status_label.setText("Generate a surface before exporting G-code")
            return

        if not self.generated_passes:
            self.status_label.setText("Generate toolpath before exporting G-code")
            return

        output_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save G-code",
            "toolpath.nc",
            "G-code (*.nc *.gcode *.txt)",
        )
        if not output_file:
            self.status_label.setText("G-code export canceled")
            return

        settings = GCodeSettings(
            safe_z_mm=float(self.safe_z_spin.value()),
            feed_rate_mm_per_min=float(self.feed_spin.value()),
            plunge_rate_mm_per_min=float(self.plunge_spin.value()),
            spindle_rpm=int(self.spindle_spin.value()),
            link_mode=(
                LINK_MODE_SURFACE if self.link_passes_check.isChecked() else LINK_MODE_RETRACT
            ),
        )
        gcode_text = generate_gcode_program(self.generated_passes, settings)
        written_path = write_gcode_file(Path(output_file), gcode_text)
        self.status_label.setText(f"Exported G-code to {written_path}")


def run_gui_app() -> int:
    """Start the GUI application loop and return process exit code.

    Inputs:
        None.

    Outputs:
        Integer process exit code from Qt event loop.

    Behavior:
        Validates runtime GUI dependencies, creates QApplication, launches the
        main window, and blocks until the user closes the application.
    """
    if GUI_IMPORT_ERROR is not None:
        raise ImportError(
            "GUI dependencies are missing. Install PyQt5, pyvista, and pyvistaqt."
        ) from GUI_IMPORT_ERROR

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = NURBSSurfaceMainWindow()
    window.show()
    return int(app.exec_())
