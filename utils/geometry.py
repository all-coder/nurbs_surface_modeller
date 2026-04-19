"""Reusable geometric helper functions used by multiple project modules."""

from __future__ import annotations

import numpy as np

from config import EPSILON


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize a 3D vector while protecting against near-zero magnitude.

    Inputs:
        vector: Vector to normalize with shape (3,).
        fallback: Optional unit-like vector returned when magnitude is too small.

    Outputs:
        A normalized vector with shape (3,).

    Behavior:
        Returns vector / ||vector|| for valid magnitudes. If magnitude is below
        EPSILON, returns fallback when provided, otherwise returns [0, 0, 1].
    """
    norm = float(np.linalg.norm(vector))
    if norm < EPSILON:
        if fallback is not None:
            return np.asarray(fallback, dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


def point_line_deviation(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Compute shortest distance from a point to a line segment.

    Inputs:
        point: Query point with shape (3,).
        start: Segment start point with shape (3,).
        end: Segment end point with shape (3,).

    Outputs:
        Euclidean deviation distance in millimeters.

    Behavior:
        Projects point onto segment start-end. If the segment is degenerate,
        the function falls back to point-to-start distance.
    """
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < EPSILON:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def format_xyz(x_value: float, y_value: float, z_value: float, precision: int = 3) -> str:
    """Format XYZ coordinates for deterministic G-code output.

    Inputs:
        x_value: X coordinate in millimeters.
        y_value: Y coordinate in millimeters.
        z_value: Z coordinate in millimeters.
        precision: Decimal precision for formatting.

    Outputs:
        A G-code friendly string in the form "X.. Y.. Z..".

    Behavior:
        Uses fixed-point formatting to make outputs predictable and easy to diff.
    """
    return (
        f"X{x_value:.{precision}f} "
        f"Y{y_value:.{precision}f} "
        f"Z{z_value:.{precision}f}"
    )


def linspace_inclusive(start_value: float, end_value: float, count: int) -> np.ndarray:
    """Create an inclusive linear parameter array.

    Inputs:
        start_value: Start scalar value.
        end_value: End scalar value.
        count: Number of samples to generate.

    Outputs:
        NumPy array of shape (count,) including both endpoints.

    Behavior:
        Raises ValueError when count is smaller than 2 to avoid malformed ranges.
    """
    if count < 2:
        raise ValueError("count must be at least 2 for inclusive sampling")
    return np.linspace(start_value, end_value, count, endpoint=True)
