"""
shapes.py — a small library of wireframe primitives.

Each function returns a PART as a list of (N, 3) float32 edge segments, in local
space, centered on its own origin, +Y up. A "part" is just any list of polyline
segments — viewport.py's make_demo_assembly() below composes several of these
into a small mechanical assembly; wiring in real CAD geometry later means
replacing that one function with something that returns the same shape of data.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

Segments = List[np.ndarray]


def box(half: Tuple[float, float, float]) -> Segments:
    """12 edges of an axis-aligned box, centered at the origin."""
    hx, hy, hz = half
    corners = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    v = np.array([(sx * hx, sy * hy, sz * hz) for sx, sy, sz in corners], np.float32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [v[[a, b]] for a, b in edges]


def cylinder(radius: float, half_height: float, n: int = 16, teeth: float = 0.0) -> Segments:
    """Top/bottom rings plus 4 vertical struts, standing on the Y axis.
    `teeth` (0-1): every other ring vertex pokes outward — a cheap gear look."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r = radius * (1.0 + teeth * 0.18 * (np.arange(n) % 2))

    def ring(y: float) -> np.ndarray:
        return np.stack((r * np.cos(a), np.full(n, y, np.float32), r * np.sin(a)), axis=1).astype(np.float32)

    top, bot = ring(half_height), ring(-half_height)
    segs = [np.vstack([top, top[:1]]), np.vstack([bot, bot[:1]])]      # closed rings
    for i in range(0, n, max(n // 4, 1)):                              # a few vertical struts
        segs.append(np.array([top[i], bot[i]], np.float32))
    return segs


def sphere(radius: float, n: int = 20) -> Segments:
    """Three orthogonal great-circle rings — a cheap, recognizable wire sphere."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=True).astype(np.float32)
    c, s = np.cos(a), np.sin(a)
    z0 = np.zeros(n, np.float32)
    xy = np.stack((radius * c, radius * s, z0), axis=1)
    xz = np.stack((radius * c, z0, radius * s), axis=1)
    yz = np.stack((z0, radius * c, radius * s), axis=1)
    return [xy, xz, yz]


def translate(segments: Segments, offset: Tuple[float, float, float]) -> Segments:
    off = np.array(offset, np.float32)
    return [seg + off for seg in segments]
