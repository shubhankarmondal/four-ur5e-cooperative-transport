"""Grasp matrix and weighted minimum-norm wrench allocation over n grasps.

Sign convention (used by every module): ``(f_i, m_i)`` is the wrench
applied BY gripper ``i`` ON the payload, expressed in the world frame, with the
moment ``m_i`` taken about the grasp point ``r_i`` (the ``handle_{i}`` site,
i.e. the desired pinch point).  The resulting wrench on the payload about the
reference point ``p_obj`` is::

    W = sum_i [ f_i ; m_i + (r_i - p_obj) x f_i ] = G w,   w = [f_1; m_1; ...; f_n; m_n]

    G = [G_1 ... G_n],   G_i = [[ I3,                 0  ],
                                [ [r_i - p_obj]_x,    I3 ]]

Allocation solves ``min_w  w^T Q w  s.t.  G w = W_des`` (regularised by ``delta``)::

    w = Q^-1 G^T (G Q^-1 G^T + delta I6)^-1 W_des,
    Q = blockdiag_i(force_weight I3, moment_weight I3)

With ``delta = 0`` and at least three non-collinear grasp points ``G`` has full
row rank, the reconstruction is exact and ``w`` contains no internal (null-space
of ``G``) component in the ``Q`` metric: in particular no internal squeeze.

Choice of ``moment_weight``.  A pinch grasp transmits a moment only through the
pad contact patch, i.e. a moment ``m`` costs roughly as much as a force ``m / l``
at the pad edge, with ``l`` the pad length scale.  Weighting the moments by
``force_weight / l^2`` makes the cost dimensionally consistent and, for grasp
points at distance ``d`` from the object centre, sends a fraction
``l^2 / (d^2 + l^2)`` of a pure yaw torque into gripper moments (the rest into
tangential forces).  Default ``l = 0.02 m`` (2F-85 pad scale) gives
``moment_weight = 2500``; with the scene's ``d = 0.275 m`` that fraction is 0.5 %.
In the symmetric weight-only case the moments are exactly zero for any weight.
"""

from __future__ import annotations

import numpy as np

PAD_LENGTH_SCALE = 0.02  # [m] pinch-pad length scale used to weight moments
DEFAULT_MOMENT_WEIGHT = 1.0 / PAD_LENGTH_SCALE**2  # = 2500, relative to force_weight = 1


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: ``skew(a) @ b == np.cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def grasp_matrix(p_obj: np.ndarray, grasp_points: np.ndarray) -> np.ndarray:
    """Grasp matrix ``G`` (6, 6n) mapping grasp wrenches to the object wrench.

    ``p_obj`` (3,) is the object reference point (payload frame origin/COM) and
    ``grasp_points`` (n, 3) the world positions ``r_i``.  See the module docstring.
    """
    p_obj = np.asarray(p_obj, dtype=float).reshape(3)
    pts = np.asarray(grasp_points, dtype=float).reshape(-1, 3)
    n = pts.shape[0]
    G = np.zeros((6, 6 * n))
    for i, r in enumerate(pts):
        c = 6 * i
        G[0:3, c : c + 3] = np.eye(3)
        G[3:6, c : c + 3] = skew(r - p_obj)
        G[3:6, c + 3 : c + 6] = np.eye(3)
    return G


def allocate(
    G: np.ndarray,
    W_des: np.ndarray,
    force_weight: float = 1.0,
    moment_weight: float = DEFAULT_MOMENT_WEIGHT,
    damping: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Weighted minimum-norm grasp wrenches ``w`` (6n,) with ``G w ~= W_des``.

    ``w = Q^-1 G^T (G Q^-1 G^T + damping I)^-1 W_des`` with
    ``Q = blockdiag(force_weight I3, moment_weight I3)`` per grasp.  Returns
    ``(w, residual)`` where ``residual = ||G w - W_des||`` (zero up to round-off
    when ``damping = 0`` and ``G`` has full row rank).  ``w`` is ordered
    ``[f_1; m_1; ...; f_n; m_n]``: wrench BY each gripper ON the payload.
    """
    if force_weight <= 0.0 or moment_weight <= 0.0:
        raise ValueError("force_weight and moment_weight must be > 0")
    if damping < 0.0:
        raise ValueError("damping must be >= 0")
    G = np.asarray(G, dtype=float)
    W_des = np.asarray(W_des, dtype=float).reshape(6)
    n = G.shape[1] // 6
    q_inv = np.tile(np.r_[np.full(3, 1.0 / force_weight), np.full(3, 1.0 / moment_weight)], n)
    GQi = G * q_inv  # G Q^-1 (column scaling)
    M = GQi @ G.T + damping * np.eye(6)
    w = GQi.T @ np.linalg.solve(M, W_des)
    residual = float(np.linalg.norm(G @ w - W_des))
    return w, residual
