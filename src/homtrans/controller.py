"""Cooperative controller: object PD -> wrench allocation -> per-arm Cartesian impedance.

Pipeline at every control step (world frame, SI):

1. Desired payload state from :class:`~homtrans.trajectory.PayloadTrajectory`.
2. Desired object wrench (about the payload frame origin = its COM)::

       F   = m [a_d + Kp (p_d - p) + Kd (v_d - v)] + m g e_z
       tau = I_w [alpha_d + Kr e_R + Kw (omega_d - omega)] + omega x (I_w omega)
       I_w = R I_body R^T,     e_R = vee(log(R_d R^T))

   ``Kp, Kd, Kr, Kw`` are per-axis bandwidth gains [1/s^2, 1/s] (scalars or
   3-vectors, world axes), scaled by the mass ``m`` and the world inertia
   ``I_w``, so the closed-loop object dynamics are ``e'' + Kd e' + Kp e = 0`` per
   axis independently of the payload's mass properties.
3. Weighted minimum-norm allocation (:mod:`homtrans.allocation`) of ``W_des``
   over the four ``handle_{i}`` site points: ``(f_i, m_i)`` = wrench BY gripper
   ``i`` ON the payload, moment about the ``handle_{i}`` site.
4. Arm ``i`` joint torques (6 arm dofs)::

       tau_i = bias_i + J_i^T [ K_p (p_d - p) + K_d (v_d - v)          + f_i  ;
                                K_r vee(log(R_d R^T)) + K_w (w_d - w)   + m_i' ]

   with ``J_i = [jacp; jacr]`` of site ``ur{i}_pinch`` restricted to the arm dofs,
   ``bias_i = qfrc_bias`` (gravity + Coriolis of the arm and its gripper) on those
   dofs, ``m_i' = m_i + (r_i - p_pinch) x f_i`` the allocated moment shifted to
   the pinch site, and the desired pinch pose/twist taken from the desired
   payload pose and the fixed handle offset ``(r_b, R_b)``::

       p_d = p_obj,d + R_obj,d r_b,   R_d = R_obj,d R_b,
       v_d = v_obj,d + w_obj,d x (R_obj,d r_b),   w_d = w_obj,d.

   ``tau = J^T F`` realises the wrench ``F`` exerted BY the pinch site ON its
   environment (static equilibrium with the reaction ``-F``), so the allocated
   feed-forward ``f_i`` enters with a plus sign.  Torques are clipped to
   ``±(150, 150, 150, 28, 28, 28)`` N·m and saturation is reported.

The controller reads the MuJoCo state; ``data`` must hold kinematics consistent
with ``qpos``/``qvel`` (``site_xpos``, ``cdof``, ``qfrc_bias``): call
``mujoco.mj_step1`` (or ``mj_forward``) before :meth:`CooperativeController.compute`
and ``mujoco.mj_step2`` after writing ``data.ctrl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from homtrans.allocation import DEFAULT_MOMENT_WEIGHT, allocate, grasp_matrix, skew
from homtrans.trajectory import PayloadTrajectory, TrajectorySample, smoothstep

TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
N_ARMS = 4
E_Z = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ SO(3) tools
def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: ``exp([w]_x) = I + sin(th) K + (1 - cos(th)) K^2``, ``K = [w/th]_x``."""
    w = np.asarray(w, dtype=float).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Exact rotation vector ``vee(log(R))``, angle in ``[0, pi]``.

    ``th = arccos((tr R - 1) / 2)``; generic case ``th / (2 sin th) vee(R - R^T)``;
    small-angle series ``(1/2)(1 + th^2/6) vee(R - R^T)``; near ``pi`` the axis
    comes from the symmetric part ``(R + R^T)/2 = cos th I + (1 - cos th) a a^T``
    with its sign fixed by ``vee(R - R^T) = 2 sin(th) a``.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * (1.0 + th * th / 6.0) * w
    if th < np.pi - 1e-3:
        return th / (2.0 * np.sin(th)) * w
    aat = (0.5 * (R + R.T) - c * np.eye(3)) / (1.0 - c)
    k = int(np.argmax(np.diag(aat)))
    a = aat[:, k] / np.sqrt(max(aat[k, k], 1e-300))
    if a @ w < 0.0:
        a = -a
    return th * a / np.linalg.norm(a)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a (not necessarily normalised) quaternion ``(w, x, y, z)``."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# ------------------------------------------------------------------------ gains
@dataclass
class ObjectGains:
    """Object-level PD bandwidths (scalar or per-world-axis 3-vector).

    ``kp`` [1/s^2] and ``kd`` [1/s] multiply ``m``; ``kr`` [1/s^2] and ``kw``
    [1/s] multiply ``I_w = R I_body R^T``.  Defaults: ``sqrt(40) = 6.3 rad/s``,
    damping ratio ``12 / (2 sqrt 40) = 0.95``.
    """

    kp: float | np.ndarray = 40.0
    kd: float | np.ndarray = 12.0
    kr: float | np.ndarray = 20.0
    kw: float | np.ndarray = 6.0


@dataclass
class ArmGains:
    """Per-arm Cartesian impedance at the pinch site (scalar or 3-vector, world axes).

    ``kp`` [N/m], ``kd`` [N·s/m], ``kr`` [N·m/rad], ``kw`` [N·m·s/rad].
    """

    kp: float | np.ndarray = 1500.0
    kd: float | np.ndarray = 80.0
    kr: float | np.ndarray = 60.0
    kw: float | np.ndarray = 4.0


@dataclass
class ControllerGains:
    object: ObjectGains = field(default_factory=ObjectGains)
    arm: ArmGains = field(default_factory=ArmGains)


@dataclass
class AllocationParams:
    """Weights of :func:`homtrans.allocation.allocate` (see its docstring)."""

    force_weight: float = 1.0
    moment_weight: float = DEFAULT_MOMENT_WEIGHT
    damping: float = 0.0


@dataclass
class GripperSchedule:
    """Gripper command ``closed_value * s((t - close_start) / close_duration)``.

    ``s`` is the quintic smoothstep; ctrl 0 = open, 255 = closed.
    ``close_duration=None`` uses the trajectory's ``t_grip``.
    """

    closed_value: float = 255.0
    close_start: float = 0.0
    close_duration: float | None = None


# ------------------------------------------------------------ control laws
def object_wrench(
    m: float,
    I_body: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    R: np.ndarray,
    omega_world: np.ndarray,
    sample: TrajectorySample,
    gains: ObjectGains,
    g: float = 9.81,
    feedback: bool = True,
) -> np.ndarray:
    """Desired payload wrench ``W_des = [F; tau]`` (world frame, about the COM).

    ``F = m [a_d + Kp (p_d - p) + Kd (v_d - v)] + m g e_z``;
    ``tau = I_w [alpha_d + Kr e_R + Kw (omega_d - omega)] + omega x (I_w omega)``,
    ``I_w = R I_body R^T``, ``e_R = vee(log(R_d R^T))``.  With ``feedback=False``
    the PD terms are dropped (pure feed-forward).  This is the wrench the
    grippers must apply ON the payload in total.
    """
    I_w = R @ np.asarray(I_body, dtype=float) @ R.T
    lin = np.array(sample.a, dtype=float)
    ang = np.array(sample.alpha, dtype=float)
    if feedback:
        lin = lin + gains.kp * (sample.p - p) + gains.kd * (sample.v - v)
        e_R = so3_log(sample.R @ R.T)
        ang = ang + gains.kr * e_R + gains.kw * (sample.omega - omega_world)
    F = m * lin + m * g * E_Z
    tau = I_w @ ang + np.cross(omega_world, I_w @ omega_world)
    return np.r_[F, tau]


def arm_torque(
    J: np.ndarray,
    bias: np.ndarray,
    p: np.ndarray,
    R: np.ndarray,
    v_lin: np.ndarray,
    omega: np.ndarray,
    p_des: np.ndarray,
    R_des: np.ndarray,
    v_des: np.ndarray,
    omega_des: np.ndarray,
    f_ff: np.ndarray,
    gains: ArmGains,
    tau_max: np.ndarray = TORQUE_LIMITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint torques of one arm: Cartesian impedance + allocated feed-forward.

    ``tau = bias + J^T [K_p (p_des - p) + K_d (v_des - v) + f_ff[:3] ;
    K_r vee(log(R_des R^T)) + K_w (omega_des - omega) + f_ff[3:]]``.

    ``J`` (6, 6) = ``[jacp; jacr]`` of the pinch site (world frame) on the arm
    dofs; ``bias`` (6,) = ``qfrc_bias`` on those dofs; ``f_ff`` (6,) = wrench the
    gripper must apply ON the payload (moment about the pinch site).  Returns
    ``(tau_clipped, saturated)`` with ``saturated[j] = |tau_j| > tau_max[j]``.
    """
    F = np.empty(6)
    F[:3] = gains.kp * (p_des - p) + gains.kd * (v_des - v_lin) + f_ff[:3]
    F[3:] = gains.kr * so3_log(R_des @ R.T) + gains.kw * (omega_des - omega) + f_ff[3:]
    tau = bias + J.T @ F
    saturated = np.abs(tau) > tau_max
    return np.clip(tau, -tau_max, tau_max), saturated


# ------------------------------------------------------------ MuJoCo controller
class CooperativeController:
    """Four-arm cooperative carrying controller on ``models/four_ur5e_transport.xml``.

    Model element names are those of :mod:`homtrans.scene`.  ``feedforward_enabled=False``
    disables the allocated wrench (pure impedance to the desired pinch poses);
    ``payload_feedback_enabled=False`` drops the object-level PD from ``W_des``.
    ``load_ramp=(t0, t1)`` optionally scales ``W_des`` by ``s((t - t0)/(t1 - t0))``
    (gradual load transfer after the jaws close); ``None`` = no scaling.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        trajectory: PayloadTrajectory,
        gains: ControllerGains | None = None,
        allocation: AllocationParams | None = None,
        gripper: GripperSchedule | None = None,
        feedforward_enabled: bool = True,
        payload_feedback_enabled: bool = True,
        load_ramp: tuple[float, float] | None = None,
    ) -> None:
        self.trajectory = trajectory
        self.gains = gains or ControllerGains()
        self.allocation = allocation or AllocationParams()
        self.gripper = gripper or GripperSchedule()
        self.feedforward_enabled = feedforward_enabled
        self.payload_feedback_enabled = payload_feedback_enabled
        self.load_ramp = load_ramp
        self.nu, self.nv = model.nu, model.nv
        self.g = float(-model.opt.gravity[2])

        # Payload: free joint, mass properties about the body origin.
        body = model.body("payload")
        self.payload_body = body.id
        jnt = model.joint("payload_free")
        if model.jnt_type[jnt.id] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[jnt.id] != body.id:
            raise ValueError("payload_free must be the free joint of body 'payload'")
        self.qadr = int(model.jnt_qposadr[jnt.id])
        self.dadr = int(model.jnt_dofadr[jnt.id])
        self.mass = float(model.body_mass[body.id])
        if abs(model.body_subtreemass[body.id] - self.mass) > 1e-9:
            raise ValueError("payload has massive child bodies; composite inertia not supported")
        if np.linalg.norm(model.body_ipos[body.id]) > 1e-6:
            raise ValueError("payload COM must coincide with the payload frame origin")
        Ri = quat_to_mat(model.body_iquat[body.id])
        self.I_body = Ri @ np.diag(model.body_inertia[body.id]) @ Ri.T

        # Handle offsets (desired pinch frames in the payload frame) and arm indices.
        self.handle_pos = np.zeros((N_ARMS, 3))
        self.handle_rot = np.zeros((N_ARMS, 3, 3))
        self.arm_dofs = np.zeros((N_ARMS, 6), dtype=int)
        self.arm_acts = np.zeros((N_ARMS, 6), dtype=int)
        self.pinch_sites = np.zeros(N_ARMS, dtype=int)
        self.gripper_acts = np.zeros(N_ARMS, dtype=int)
        for k in range(N_ARMS):
            i = k + 1
            site = model.site(f"handle_{i}")
            if model.site_bodyid[site.id] != body.id:
                raise ValueError(f"handle_{i} must be a site of body 'payload'")
            self.handle_pos[k] = model.site_pos[site.id]
            self.handle_rot[k] = quat_to_mat(model.site_quat[site.id])
            for j, (jname, aname) in enumerate(zip(ARM_JOINTS, ARM_ACTUATORS)):
                jid = model.joint(f"ur{i}_{jname}").id
                aid = model.actuator(f"ur{i}_{aname}").id
                if (
                    model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                    or model.actuator_trnid[aid, 0] != jid
                ):
                    raise ValueError(f"actuator ur{i}_{aname} must drive joint ur{i}_{jname}")
                self.arm_dofs[k, j] = model.jnt_dofadr[jid]
                self.arm_acts[k, j] = aid
            self.pinch_sites[k] = model.site(f"ur{i}_pinch").id
            self.gripper_acts[k] = model.actuator(f"ur{i}_fingers_actuator").id
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    # ------------------------------------------------------------------ state
    def payload_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(p, R, v, omega_world)`` of the payload from ``payload_free``.

        MuJoCo free joint: ``qvel[0:3]`` = linear velocity of the body origin in
        the world frame, ``qvel[3:6]`` = angular velocity in the BODY frame, so
        ``omega_world = R qvel[3:6]``.
        """
        a, d = self.qadr, self.dadr
        p = data.qpos[a : a + 3].copy()
        R = quat_to_mat(data.qpos[a + 3 : a + 7])
        v = data.qvel[d : d + 3].copy()
        omega = R @ data.qvel[d + 3 : d + 6]
        return p, R, v, omega

    def gripper_command(self, t: float) -> float:
        dur = self.gripper.close_duration or self.trajectory.t_grip
        s, _, _ = smoothstep((t - self.gripper.close_start) / dur)
        return self.gripper.closed_value * s

    def load_scale(self, t: float) -> float:
        if self.load_ramp is None:
            return 1.0
        t0, t1 = self.load_ramp
        return smoothstep((t - t0) / max(t1 - t0, 1e-9))[0]

    # ---------------------------------------------------------------- compute
    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> tuple[np.ndarray, dict]:
        """Control vector ``ctrl`` (nu,) and diagnostics at time ``t``.

        Requires up-to-date kinematics in ``data`` (see module docstring).
        """
        s = self.trajectory.sample(t)
        p, R, v, omega = self.payload_state(data)

        # Steps 2-3: object wrench, allocated over the actual handle-site points.
        lam = self.load_scale(t)
        W_des = lam * object_wrench(
            self.mass, self.I_body, p, v, R, omega, s, self.gains.object, self.g,
            feedback=self.payload_feedback_enabled,
        )
        grasp_pts = p + self.handle_pos @ R.T
        if self.feedforward_enabled:
            G = grasp_matrix(p, grasp_pts)
            ap = self.allocation
            w, residual = allocate(G, W_des, ap.force_weight, ap.moment_weight, ap.damping)
            f = w.reshape(N_ARMS, 6)
        else:
            f, residual = np.zeros((N_ARMS, 6)), 0.0

        # Step 4: per-arm impedance to the desired pinch pose + feed-forward.
        ctrl = np.zeros(self.nu)
        pinch_pos_err = np.zeros((N_ARMS, 3))
        pinch_rot_err = np.zeros((N_ARMS, 3))
        tau_all = np.zeros((N_ARMS, 6))
        sat_all = np.zeros((N_ARMS, 6), dtype=bool)
        for k in range(N_ARMS):
            r_d = s.R @ self.handle_pos[k]
            p_des = s.p + r_d
            R_des = s.R @ self.handle_rot[k]
            v_des = s.v + np.cross(s.omega, r_d)
            sid = self.pinch_sites[k]
            p_pin = data.site_xpos[sid].copy()
            R_pin = data.site_xmat[sid].reshape(3, 3).copy()
            mujoco.mj_jacSite(model, data, self._jacp, self._jacr, sid)
            v_pin = self._jacp @ data.qvel
            w_pin = self._jacr @ data.qvel
            dofs = self.arm_dofs[k]
            J = np.vstack((self._jacp[:, dofs], self._jacr[:, dofs]))
            f_ff = np.r_[f[k, :3], f[k, 3:] + np.cross(grasp_pts[k] - p_pin, f[k, :3])]
            tau, sat = arm_torque(
                J, data.qfrc_bias[dofs], p_pin, R_pin, v_pin, w_pin,
                p_des, R_des, v_des, s.omega, f_ff, self.gains.arm,
            )
            ctrl[self.arm_acts[k]] = tau
            tau_all[k], sat_all[k] = tau, sat
            pinch_pos_err[k] = p_des - p_pin
            pinch_rot_err[k] = so3_log(R_des @ R_pin.T)

        grip = self.gripper_command(t)
        ctrl[self.gripper_acts] = grip

        diag = {
            "t": float(t),
            "phase": s.phase,
            "p_des": s.p,
            "p": p,
            "payload_pos_err": s.p - p,
            "payload_rot_err": so3_log(s.R @ R.T),
            "load_scale": lam,
            "W_des": W_des,
            "f": f,
            "alloc_residual": residual,
            "pinch_pos_err": pinch_pos_err,
            "pinch_rot_err": pinch_rot_err,
            "tau": tau_all,
            "saturated": sat_all,
            "any_saturated": bool(sat_all.any()),
            "gripper_ctrl": grip,
        }
        return ctrl, diag
