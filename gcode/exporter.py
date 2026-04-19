"""Generate CNC G-code from surface toolpath data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import (
    DEFAULT_FEED_RATE_MM_PER_MIN,
    DEFAULT_PLUNGE_RATE_MM_PER_MIN,
    DEFAULT_SAFE_Z_MM,
    DEFAULT_SPINDLE_RPM,
)
from machining.toolpath import ToolpathPass
from utils.geometry import format_xyz


@dataclass(frozen=True)
class GCodeSettings:
    """Store output settings controlling motion and machine state commands.

    Inputs:
        safe_z_mm: Clearance height used for rapid moves.
        feed_rate_mm_per_min: Feed rate used for G1 cutting moves.
        plunge_rate_mm_per_min: Feed rate used when plunging from safe height.
        spindle_rpm: Spindle speed command value.

    Outputs:
        Immutable settings object consumed by G-code generation.

    Behavior:
        Encapsulates defaults and allows user overrides in one place.
    """

    safe_z_mm: float = DEFAULT_SAFE_Z_MM
    feed_rate_mm_per_min: float = DEFAULT_FEED_RATE_MM_PER_MIN
    plunge_rate_mm_per_min: float = DEFAULT_PLUNGE_RATE_MM_PER_MIN
    spindle_rpm: int = DEFAULT_SPINDLE_RPM


def generate_gcode_program(
    passes: list[ToolpathPass],
    settings: GCodeSettings,
) -> str:
    """Generate a complete 3-axis G-code program from toolpath passes.

    Inputs:
        passes: Ordered zig-zag toolpath passes.
        settings: G-code motion and machine settings.

    Outputs:
        Multi-line G-code program string.

    Behavior:
        Writes mandatory startup commands (G21, G90, M3, M8), inserts safe
        rapid transitions between passes, emits G1 feed moves on surface points,
        and closes with M30.
    """
    lines: list[str] = []

    lines.append("(NURBS Surface Toolpath)")
    lines.append("G21")
    lines.append("G90")
    lines.append(f"S{settings.spindle_rpm}")
    lines.append("M3")
    lines.append("M8")
    lines.append(f"G0 Z{settings.safe_z_mm:.3f}")

    if not passes:
        lines.append("M9")
        lines.append("M5")
        lines.append("M30")
        return "\n".join(lines) + "\n"

    for pass_index, tool_pass in enumerate(passes):
        if not tool_pass.points:
            continue

        start = tool_pass.points[0].cl_point
        lines.append(f"(Pass {pass_index + 1}, v={tool_pass.v_parameter:.6f}, {tool_pass.direction})")
        lines.append(f"G0 X{start[0]:.3f} Y{start[1]:.3f}")
        lines.append(
            f"G1 Z{start[2]:.3f} F{settings.plunge_rate_mm_per_min:.1f}"
        )

        for point in tool_pass.points:
            lines.append(
                f"G1 {format_xyz(point.cl_point[0], point.cl_point[1], point.cl_point[2])} "
                f"F{settings.feed_rate_mm_per_min:.1f}"
            )

        lines.append(f"G0 Z{settings.safe_z_mm:.3f}")

    lines.append("M9")
    lines.append("M5")
    lines.append("M30")
    return "\n".join(lines) + "\n"


def write_gcode_file(output_path: str | Path, gcode_text: str) -> Path:
    """Write G-code text to disk and return resolved file path.

    Inputs:
        output_path: Destination file path where G-code will be stored.
        gcode_text: Generated G-code content.

    Outputs:
        Resolved pathlib.Path to the created file.

    Behavior:
        Creates parent directories when needed and writes UTF-8 text content.
    """
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(gcode_text, encoding="utf-8")
    return output
