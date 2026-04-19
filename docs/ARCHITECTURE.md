# Architecture

This repository uses a flat root layout for maintainability:

- core: B-spline basis and validation logic.
- surface: Tensor-product NURBS surface evaluation and factories.
- machining: Zig-zag toolpath generation and chord filtering.
- gcode: CNC G-code generation and file export.
- gui: Interactive desktop application.
- utils: Shared geometry and formatting helpers.
- tests: Unit tests for math and export pipeline.

Main entrypoint: main.py
