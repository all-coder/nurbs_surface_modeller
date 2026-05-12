# Project 10 - NURBS Surface Modeler

A comprehensive CAD/CAM application for designing NURBS surfaces and generating CNC G-code with interactive 3D visualization.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### 1. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

From the project root directory, run:

```bash
pip install -r requirements.txt
```

## Running the Project

All commands should be executed from the project root directory.

```bash
python main.py
```

## Project Structure

```
NURBS_SurfaceModeller/
├── README.md
├── main.py
├── config.py
├── requirements.txt
├── CADCAM_Project_10_Report.pdf
│
├── core/
│   ├── __init__.py
│   ├── bspline.py
│   ├── curve_validation.py
│   ├── curves.py
│   └── validation.py
│
├── surface/
│   ├── __init__.py
│   ├── nurbs_surface.py
│   ├── factory.py
│   └── providers/
│       ├── __init__.py
│       ├── extrusion.py
│       ├── loft.py
│       ├── surface_body_extrusion.py
│       └── surface_normal_extrusion.py
│
├── machining/
│   ├── __init__.py
│   └── toolpath.py
│
├── gcode/
│   ├── __init__.py
│   └── exporter.py
│
├── gui/
│   ├── __init__.py
│   └── app.py
│
├── utils/
│   ├── __init__.py
│   └── geometry.py
│
├── tests/
│   ├── test_curves.py
│   └── test_math_and_export.py
│
├── docs/
│   └── ARCHITECTURE.md
│
├── report/
│   ├── report.tex
│   └── workflow.md
│
└── assets/
```
