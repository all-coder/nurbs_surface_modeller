"""Application entrypoint for the modular NURBS surface modeller project."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_STEPOVER_MM, DEFAULT_TOOL_RADIUS_MM
from gcode.exporter import GCodeSettings, generate_gcode_program, write_gcode_file
from machining.toolpath import generate_zigzag_toolpath
from surface.factory import create_default_surface


def run_demo_export(output_path: str | Path, rows: int, cols: int) -> Path:
    """Generate a demo surface toolpath and export G-code without launching GUI.

    Inputs:
        output_path: Destination file path for the generated G-code.
        rows: Number of control points in u direction for demo surface.
        cols: Number of control points in v direction for demo surface.

    Outputs:
        Resolved path to the exported G-code file.

    Behavior:
        Creates a default NURBS surface, generates zig-zag CL passes, emits
        standards-compliant G-code, and writes the program to disk.
    """
    surface = create_default_surface(rows=rows, cols=cols)
    passes = generate_zigzag_toolpath(
        surface=surface,
        stepover_mm=DEFAULT_STEPOVER_MM,
        tool_radius_mm=DEFAULT_TOOL_RADIUS_MM,
    )
    gcode_text = generate_gcode_program(passes, GCodeSettings())
    return write_gcode_file(output_path, gcode_text)


def run_gui() -> int:
    """Launch the interactive GUI application.

    Inputs:
        None.

    Outputs:
        Integer process exit code.

    Behavior:
        Imports GUI modules lazily so command-line workflows remain usable when
        GUI dependencies are unavailable.
    """
    from gui.app import run_gui_app

    return run_gui_app()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line parser for GUI and non-GUI workflows.

    Inputs:
        None.

    Outputs:
        Configured argparse.ArgumentParser instance.

    Behavior:
        Provides switches to run headless export mode or launch the GUI.
    """
    parser = argparse.ArgumentParser(description="NURBS Surface Modeller")
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run in headless mode and export a demo G-code file.",
    )
    parser.add_argument(
        "--output",
        default="demo_toolpath.nc",
        help="Output file path used by --nogui mode.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Control net rows for demo generation.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Control net columns for demo generation.",
    )
    return parser


def main() -> int:
    """Execute the application according to command-line arguments.

    Inputs:
        None.

    Outputs:
        Integer exit code.

    Behavior:
        Uses GUI mode by default. In headless mode, exports a demo G-code file
        and prints the destination path.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.nogui:
        output_path = run_demo_export(args.output, rows=args.rows, cols=args.cols)
        print(f"G-code exported to: {output_path}")
        return 0

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
