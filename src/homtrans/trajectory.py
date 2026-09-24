"""Desired payload trajectory: grip -> lift -> hold -> circle (+ Z sinusoid) -> settle.

All quantities are the DESIRED pose/twist of the payload body frame in the world
frame (SI units).  The orientation is held fixed at ``R0`` throughout, so
``omega = alpha = 0``.  Velocities and accelerations are exact analytic
derivatives (no finite differences); the whole trajectory is C^2 (position,
velocity and acceleration continuous at every phase boundary) because every
blend is the quintic smoothstep, whose first and second derivatives vanish at
both ends.

Quintic smoothstep on u in [0, 1]::

    s(u)   = 10 u^3 - 15 u^4 + 6 u^5          s(0) = 0,  s(1) = 1
    s'(u)  = 30 u^2 (1 - u)^2                  s'(0) = s'(1) = 0
    s''(u) = 60 u (1 - u)(1 - 2u)              s''(0) = s''(1) = 0

Phases (half-open intervals ``[start, end)``; ``t < 0`` is reported as ``grip``
and ``t >= duration`` as ``settle``)::

    grip    [0, t_grip)          p = p_rest                     (grippers close)
    lift    T_lift               p = p_rest + h_lift s(tau/T_lift) e_z
    hold    t_hold               p = p_lift = p_rest + h_lift e_z
    circle  T_circle             p = c + radius [cos th, sin th, 0] + z(tau) e_z
    settle  t_settle             p = p_lift

Circle phase, local time ``tau`` in ``[0, T_circle]``.  The circle is CONCENTRIC
with the lift point (the payload never moves more than ``radius`` from where it
was lifted, so no gripper is carried over the stand).  The angle advances at the
constant rate ``omega_c = 2 pi / period`` and the radius and the Z amplitude share
one smooth envelope ``e(tau)``, so the payload spirals out from ``p_lift``, makes
``n_circles`` full turns at full radius, and spirals back in::

    th(tau) = omega_c tau
    e(tau)  = s(tau / T_ramp)                       tau in [0, T_ramp]
            = 1                                     tau in [T_ramp, T_circle - T_ramp]
            = s((T_circle - tau) / T_ramp)          tau in [T_circle - T_ramp, T_circle]
    T_circle = n_circles period + 2 T_ramp

    p_xy = p_lift,xy + r [cos th, sin th],            r = radius e
    v_xy = r' [cos th, sin th] + r omega_c [-sin th, cos th]
    a_xy = r'' [cos th, sin th] + 2 r' omega_c [-sin th, cos th] - r omega_c^2 [cos th, sin th]

Z sinusoid, ``omega_z = k omega_c`` (integer ``k``)::

    z(tau)  = A_z e(tau) sin(omega_z tau)
    z'      = A_z [e' sin + e omega_z cos]
    z''     = A_z [e'' sin + 2 e' omega_z cos - e omega_z^2 sin]

Because ``e, e', e''`` vanish at both ends, position offset, velocity and
acceleration are all zero at the start and end of the circle phase (C^2 joins).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PHASES = ("grip", "lift", "hold", "circle", "settle")


def smoothstep(u: float) -> tuple[float, float, float]:
    """Quintic smoothstep ``(s, ds/du, d2s/du2)`` at ``u``, clamped to ``[0, 1]``."""
    u = min(max(u, 0.0), 1.0)
    s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    ds = 30.0 * u * u * (1.0 - u) * (1.0 - u)
    dds = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)
    return s, ds, dds


@dataclass(frozen=True)
class TrajectorySample:
    """Desired payload state at time ``t`` (world frame).

    ``p, v, a``: position, linear velocity, linear acceleration of the payload
    frame origin; ``R``: orientation (3x3); ``omega, alpha``: angular velocity and
    acceleration (world frame); ``phase``: one of :data:`PHASES`.
    """

    t: float
    phase: str
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    R: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray


@dataclass
class PayloadTrajectory:
    """Grip / lift / hold / circle / settle reference for the payload.

    Parameters (SI): ``p_rest`` rest position of the payload frame; ``h_lift``
    lift height; ``t_grip, T_lift, t_hold, t_settle`` phase durations;
    ``radius`` circle radius; ``period`` = ``2 pi / omega_c`` (plateau period);
    ``n_circles`` full turns (positive integer); ``A_z`` Z-sinusoid amplitude;
    ``k`` Z harmonic (``omega_z = k omega_c``, non-negative integer); ``T_ramp``
    angular-rate and Z-envelope ramp time; ``R0`` fixed desired orientation.
    """

    p_rest: tuple[float, float, float] = (0.0, 0.0, 0.40)
    h_lift: float = 0.08
    t_grip: float = 1.0
    T_lift: float = 2.0
    t_hold: float = 1.0
    radius: float = 0.15
    period: float = 10.0
    n_circles: int = 1
    A_z: float = 0.04
    k: int = 2
    T_ramp: float = 2.0
    t_settle: float = 1.5
    R0: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        self.p_rest = np.asarray(self.p_rest, dtype=float).reshape(3)
        self.R0 = np.asarray(self.R0, dtype=float).reshape(3, 3)
        if not np.allclose(self.R0 @ self.R0.T, np.eye(3), atol=1e-9) or np.linalg.det(self.R0) < 0:
            raise ValueError("R0 must be a rotation matrix")
        for name in ("t_grip", "T_lift", "T_ramp", "period"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        for name in ("t_hold", "t_settle", "radius", "h_lift"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.n_circles) != self.n_circles or self.n_circles < 1:
            raise ValueError("n_circles must be a positive integer")
        if int(self.k) != self.k or self.k < 0:
            raise ValueError("k must be a non-negative integer")

    # ------------------------------------------------------------ derived values
    @property
    def circle_rate(self) -> float:
        """Plateau angular rate ``omega_c = 2 pi / period`` [rad/s]."""
        return 2.0 * np.pi / self.period

    @property
    def omega_z(self) -> float:
        """Z-sinusoid angular frequency ``k omega_c`` [rad/s]."""
        return self.k * self.circle_rate

    @property
    def T_plateau(self) -> float:
        """Full-radius time ``n_circles period`` [s] (exactly n full turns)."""
        return self.n_circles * self.period

    @property
    def T_circle(self) -> float:
        """Circle phase duration ``T_plateau + 2 T_ramp`` [s]."""
        return self.T_plateau + 2.0 * self.T_ramp

    @property
    def p_lift(self) -> np.ndarray:
        return self.p_rest + np.array([0.0, 0.0, self.h_lift])

    @property
    def circle_center(self) -> np.ndarray:
        """The circle is concentric with the lift point."""
        return self.p_lift

    @property
    def phase_boundaries(self) -> list[tuple[str, float, float]]:
        """``[(phase, t_start, t_end), ...]`` in time order."""
        durations = (self.t_grip, self.T_lift, self.t_hold, self.T_circle, self.t_settle)
        out, t0 = [], 0.0
        for name, d in zip(PHASES, durations):
            out.append((name, t0, t0 + d))
            t0 += d
        return out

    @property
    def duration(self) -> float:
        return self.phase_boundaries[-1][2]

    # -------------------------------------------------------- circle primitives
    def envelope(self, tau: float) -> tuple[float, float, float]:
        """``(e, e', e'')`` of the shared radius / Z envelope at circle time ``tau``."""
        Tr, Tc = self.T_ramp, self.T_circle
        tau = min(max(tau, 0.0), Tc)
        if tau < Tr:
            e, de, dde = smoothstep(tau / Tr)
            return e, de / Tr, dde / Tr**2
        if tau <= Tc - Tr:
            return 1.0, 0.0, 0.0
        e, de, dde = smoothstep((Tc - tau) / Tr)
        return e, -de / Tr, dde / Tr**2

    def circle_angle(self, tau: float) -> tuple[float, float, float]:
        """``(theta, theta_dot, theta_ddot)`` at circle-local time ``tau`` (constant rate)."""
        tau = min(max(tau, 0.0), self.T_circle)
        return self.circle_rate * tau, self.circle_rate, 0.0

    def z_offset(self, tau: float) -> tuple[float, float, float]:
        """``(z, z', z'')`` of the Z sinusoid at circle-local time ``tau``."""
        wz = self.omega_z
        tau = min(max(tau, 0.0), self.T_circle)
        e, de, dde = self.envelope(tau)
        sn, cs = np.sin(wz * tau), np.cos(wz * tau)
        z = self.A_z * e * sn
        dz = self.A_z * (de * sn + e * wz * cs)
        ddz = self.A_z * (dde * sn + 2.0 * de * wz * cs - e * wz * wz * sn)
        return z, dz, ddz

    # ------------------------------------------------------------------ sample
    def sample(self, t: float) -> TrajectorySample:
        """Desired payload pose, twist and acceleration at time ``t``."""
        p = self.p_rest.copy()
        v = np.zeros(3)
        a = np.zeros(3)
        phase = PHASES[-1]
        if t < 0.0:
            phase = PHASES[0]
        else:
            for name, t0, t1 in self.phase_boundaries:
                if t < t1:
                    phase, tau = name, t - t0
                    break
        if phase == "lift":
            s, ds, dds = smoothstep(tau / self.T_lift)
            p[2] += self.h_lift * s
            v[2] = self.h_lift * ds / self.T_lift
            a[2] = self.h_lift * dds / self.T_lift**2
        elif phase == "circle":
            th, w, _ = self.circle_angle(tau)
            e, de, dde = self.envelope(tau)
            z, dz, ddz = self.z_offset(tau)
            c, sn = np.cos(th), np.sin(th)
            r, dr, ddr = self.radius * e, self.radius * de, self.radius * dde
            p = self.circle_center + np.array([r * c, r * sn, z])
            v = np.array([dr * c - r * w * sn, dr * sn + r * w * c, dz])
            a = np.array(
                [
                    ddr * c - 2.0 * dr * w * sn - r * w * w * c,
                    ddr * sn + 2.0 * dr * w * c - r * w * w * sn,
                    ddz,
                ]
            )
        elif phase in ("hold", "settle"):
            p = self.p_lift
        return TrajectorySample(
            t=float(t),
            phase=phase,
            p=p,
            v=v,
            a=a,
            R=self.R0.copy(),
            omega=np.zeros(3),
            alpha=np.zeros(3),
        )
