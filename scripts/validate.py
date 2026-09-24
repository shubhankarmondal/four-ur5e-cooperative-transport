#!/usr/bin/env python
"""Validation suite of the four-UR5e cooperative plate transport.

    pixi run validate

Options: ``--jobs N`` (parallel simulations, default 4), ``--skip-determinism``
(no fresh-process demo reruns), ``--skip-robustness``, ``--skip-causality``,
``--skip-stress`` (INFO-only stress / timestep / ablation runs).

Prints a PASS/FAIL/INFO/SKIP table and writes
``outputs/validation/validation_report.json``.  Exit status 1 if any check FAILs.

Every number comes from a simulation this script runs itself, or from demo runs
it launches in fresh processes under ``outputs/runs/validation_det_{a,b}``.  The
in-process harness replicates ``scripts/run_demo.py`` step for step (defaults are
read from that file's argparse calls with ``ast``), and check 2b verifies that
its payload path is bitwise equal to the fresh-process demo log.

Checks
  1  model integrity (arms, grippers, the one free joint, equality constraints,
     payload mass/inertia, torque actuators, no hidden forces on the payload)
  2  clean-start determinism (two fresh ``run_demo.py`` processes + the harness)
  3  full-run metrics (tracking, lift-off, grasp forces, unwanted contacts,
     saturation, joint velocity/limits, full circle, Z bobs, grasp slip)
  4  causality: release (open all grippers, then back the arms off), zero arm
     torques; reported only: open all grippers (the open lower fingers still
     support the plate, see ``checks_causality``) and open gripper 1 alone
  5  frame/sign audit (grasp matrix, W_des law, Newton-Euler on the payload,
     pinch Jacobian vs finite differences, allocated vs realised grasp forces)
  6  robustness (+50 % mass, period 8 s, +-0.01 rad initial perturbation)
  7  video correspondence (sidecar/record/model sha, frame count, states)
  8  self-containment (no absolute user paths, no imports beyond the standard
     library, the declared dependencies and ``homtrans``; meshes inside the repo)

The bars are the constants in ``BARS``: acceptance thresholds of this validator
for the demonstration, not physical requirements.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import mujoco  # noqa: E402

from homtrans.controller import CooperativeController  # noqa: E402
from homtrans.kinematics import initial_configuration  # noqa: E402
from homtrans.scene import MODEL_XML, build_spec, portable_xml  # noqa: E402
from homtrans.trajectory import PayloadTrajectory  # noqa: E402

N_ARMS = 4
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_ACTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])  # UR5e joint torque limits [N·m]
UR5E_MAX_JOINT_SPEED = np.pi  # rad/s: UR5e datasheet, 180 deg/s on every joint
OUT_DIR = HERE / "outputs" / "validation"
REPORT = OUT_DIR / "validation_report.json"
VIDEO = HERE / "outputs" / "videos" / "four_ur5e_cooperative_transport.mp4"  # scripts/render.py
# sha256 of the model files in assets/, byte-identical to MuJoCo Menagerie
# revision 8161bba264d7fa7c99ca301e91e7fb44737676ad
MENAGERIE_SHA256 = {
    "ur5e.xml": "ffd0a7b336d3c2d27c35a15cc0d4aaa95196769aca190ac8d6522d3318c345a0",
    "2f85.xml": "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de",
}
# third-party packages declared in pixi.toml, by import name
DEPENDENCIES = {"mujoco", "numpy", "PIL", "imageio_ffmpeg", "pytest"}

BARS = {
    "pos_err_max_m": 0.010,      # payload position error while carried
    "pos_err_rms_m": 0.005,
    "rot_err_max_rad": 0.05,     # ~2.9 deg
    "circle_radius_frac": 0.90,  # swept angle counted only at >= 90 % of R
    "circle_sweep_rad": 2.0 * np.pi,
    "z_excursion_frac": 0.80,    # of A_z, both up and down
    "release_drop_m": 0.10,      # within 1 s of opening all grippers
    "sag_drop_m": 0.10,          # within 1 s of zeroing arm torques
    "joint_margin_rad": 0.05,
    "determinism_tol": 1e-12,
    # Newton-Euler: the dynamic residual m a - (F_c + m g) equals the constraint solver's
    # remaining gradient on the payload rows (the Newton solver stops at opt.tolerance,
    # often after 1 iteration), so it is not zero: contacts must explain >= 99 % of the
    # weight-scale dynamics.  ne_exact is the algebraic identity (no generalized force
    # on the payload other than gravity, gyroscopic and contact terms).
    "newton_euler_rel": 1e-2,    # |dynamic residual| / (m g)
    "ne_exact_rel": 1e-9,        # |qfrc_smooth + qfrc_constraint - contacts - gravity - gyro| / (m g)
    "jacobian_fd_tol": 1e-6,
    "grasp_matrix_tol": 1e-9,
    "wdes_law_tol": 1e-9,
    "slip_mm": 5.0,              # pinch-vs-handle drift while carried (reported, bar)
}
EVENT_T = 8.0          # release / single-release time [s]
ZERO_TORQUE_T = 3.5    # zero-arm-torque time: middle of the hold, payload lifted and still
FORCE_EPS = 1e-6       # N; contact counted as "loaded" above this


# ---- demo defaults (read from scripts/run_demo.py so the harness cannot drift)
def demo_defaults() -> dict:
    src = (HERE / "scripts" / "run_demo.py").read_text()
    tree = ast.parse(src)
    out = {"radius": 0.15, "period": 10.0, "az": 0.04, "k": 2, "lift": 0.08,
           "load_ramp": (0.5, 1.0), "parsed": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value.lstrip("-").replace("-", "_")
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            out[name] = ast.literal_eval(kw.value)
                            out["parsed"].append(name)
                        except ValueError:
                            pass
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "CooperativeController":
            for kw in node.keywords:
                if kw.arg == "load_ramp":
                    out["load_ramp"] = ast.literal_eval(kw.value)
                    out["parsed"].append("load_ramp")
    return out


DEMO = demo_defaults()


def demo_trajectory(**over) -> PayloadTrajectory:
    kw = dict(radius=DEMO["radius"], period=DEMO["period"], A_z=DEMO["az"], k=DEMO["k"],
              h_lift=DEMO["lift"])
    kw.update(over)
    return PayloadTrajectory(**kw)


# --------------------------------------------------------- result bookkeeping
@dataclass
class Check:
    id: str
    name: str
    status: str  # PASS / FAIL / INFO / SKIP
    value: str = ""
    metrics: dict = field(default_factory=dict)
    note: str = ""


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else str(x)
    return x


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------- model lookups
CAT_NONE, CAT_FLOOR, CAT_STAND, CAT_PLATE, CAT_HANDLE, CAT_PAD, CAT_FINGER, CAT_LINK = range(8)


@dataclass
class ModelInfo:
    payload: int
    stand_geom: int
    geom_cat: np.ndarray
    geom_arm: np.ndarray
    geom_side: np.ndarray  # pads: 0 = left, 1 = right
    geom_handle: np.ndarray
    arm_qadr: np.ndarray  # (4, 6)
    arm_dadr: np.ndarray
    arm_acts: np.ndarray
    grip_acts: np.ndarray
    driver_qadr: np.ndarray  # (4,) right driver joint qpos adr (opening indicator)
    pinch_sites: np.ndarray
    handle_sites: np.ndarray
    pay_qadr: int
    pay_dadr: int
    jnt_lo: np.ndarray  # (4, 6)
    jnt_hi: np.ndarray

    @classmethod
    def build(cls, m: mujoco.MjModel) -> "ModelInfo":
        pay = m.body("payload").id
        cat = np.zeros(m.ngeom, dtype=int)
        arm = np.zeros(m.ngeom, dtype=int)
        side = np.full(m.ngeom, -1, dtype=int)
        hnd = np.zeros(m.ngeom, dtype=int)
        for g in range(m.ngeom):
            name = m.geom(g).name
            b = int(m.geom_bodyid[g])
            bname = m.body(b).name
            if name == "floor":
                cat[g] = CAT_FLOOR
            elif name == "stand":
                cat[g] = CAT_STAND
            elif b == pay:
                mt = re.fullmatch(r"handle_(\d)_geom", name)
                if mt:
                    cat[g], hnd[g] = CAT_HANDLE, int(mt.group(1))
                else:
                    cat[g] = CAT_PLATE
            else:
                mt = re.match(r"ur(\d)_", bname)
                if mt:
                    arm[g] = int(mt.group(1))
                    if re.fullmatch(rf"ur{arm[g]}_g_(left|right)_pad", bname):
                        cat[g] = CAT_PAD
                        side[g] = 0 if "_left_" in bname else 1
                    elif bname.startswith(f"ur{arm[g]}_g_"):
                        cat[g] = CAT_FINGER
                    else:
                        cat[g] = CAT_LINK
        jq = np.array([[m.jnt_qposadr[m.joint(f"ur{i}_{j}").id] for j in ARM_JOINTS]
                       for i in range(1, N_ARMS + 1)])
        jd = np.array([[m.jnt_dofadr[m.joint(f"ur{i}_{j}").id] for j in ARM_JOINTS]
                       for i in range(1, N_ARMS + 1)])
        jr = np.array([[m.jnt_range[m.joint(f"ur{i}_{j}").id] for j in ARM_JOINTS]
                       for i in range(1, N_ARMS + 1)])
        acts = np.array([[m.actuator(f"ur{i}_{a}").id for a in ARM_ACTS] for i in range(1, N_ARMS + 1)])
        fj = m.joint("payload_free").id
        return cls(
            payload=pay, stand_geom=m.geom("stand").id, geom_cat=cat, geom_arm=arm,
            geom_side=side, geom_handle=hnd, arm_qadr=jq, arm_dadr=jd, arm_acts=acts,
            grip_acts=np.array([m.actuator(f"ur{i}_fingers_actuator").id for i in range(1, N_ARMS + 1)]),
            driver_qadr=np.array([m.jnt_qposadr[m.joint(f"ur{i}_g_right_driver_joint").id]
                                  for i in range(1, N_ARMS + 1)]),
            pinch_sites=np.array([m.site(f"ur{i}_pinch").id for i in range(1, N_ARMS + 1)]),
            handle_sites=np.array([m.site(f"handle_{i}").id for i in range(1, N_ARMS + 1)]),
            pay_qadr=int(m.jnt_qposadr[fj]), pay_dadr=int(m.jnt_dofadr[fj]),
            jnt_lo=jr[..., 0], jnt_hi=jr[..., 1],
        )


def quat2mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (independent implementation via quaternions)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-15:
        return 2.0 * q[1:]
    return 2.0 * np.arctan2(s, q[0]) * q[1:] / s


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------- contact analysis (vectorised over data.contact)
def contact_wrenches(m: mujoco.MjModel, d: mujoco.MjData):
    """Per contact: geoms (n,2), world force ON geom2 (n,3), world torque ON geom2 (n,3),
    position (n,3), normal force (n,)."""
    n = d.ncon
    if n == 0:
        z = np.zeros((0, 3))
        return np.zeros((0, 2), dtype=int), z, z, z, np.zeros(0)
    con = d.contact
    geom = np.array(con.geom[:n], dtype=int)
    adr = np.array(con.efc_address[:n], dtype=int)
    dim = np.array(con.dim[:n], dtype=int)
    frame = np.array(con.frame[:n]).reshape(n, 3, 3)
    pos = np.array(con.pos[:n])
    f6 = np.zeros((n, 6))
    for dd in np.unique(dim):
        sel = (dim == dd) & (adr >= 0)
        if not sel.any():
            continue
        if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC or dd == 1:
            f6[sel, :dd] = d.efc_force[adr[sel, None] + np.arange(dd)]
        else:  # pyramidal: fall back to mj_contactForce
            buf = np.zeros(6)
            for k in np.flatnonzero(sel):
                mujoco.mj_contactForce(m, d, int(k), buf)
                f6[k] = buf
    fw = np.einsum("nji,nj->ni", frame, f6[:, :3])  # frame rows are axes: world = frame^T f
    tw = np.einsum("nji,nj->ni", frame, f6[:, 3:])
    return geom, fw, tw, pos, f6[:, 0]


# --------------------------------- scenarios and the instrumented closed loop
@dataclass
class Scenario:
    name: str
    traj: dict = field(default_factory=dict)
    duration: float | None = None
    mass_scale: float = 1.0
    controller_knows_mass: bool = True
    perturb: float = 0.0
    seed: int = 0
    events: tuple = ()  # (t, action), action: open_all / open_1 / zero_arm / no_stand / retract:<m>
    full_log: bool = False
    audit_times: tuple = ()
    frame_fps: float = 0.0  # >0: store qpos at the FrameRecorder schedule
    response_t: float | None = None  # time the causality response is measured from
    timestep: float | None = None  # override model.opt.timestep
    feedforward: bool = True
    clearance_every: float = 0.0  # >0: geometric clearance audit every this many seconds


def load_model(mass_scale: float = 1.0) -> mujoco.MjModel:
    if mass_scale == 1.0:
        return mujoco.MjModel.from_xml_path(str(MODEL_XML))
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    for g in spec.body("payload").geoms:
        g.mass = g.mass * mass_scale
    return spec.compile()


_QPOS0_CACHE: dict = {}


def nominal_qpos0() -> np.ndarray:
    if "q" not in _QPOS0_CACHE:
        _QPOS0_CACHE["q"] = initial_configuration(mujoco.MjModel.from_xml_path(str(MODEL_XML)))
    return _QPOS0_CACHE["q"].copy()


def simulate(scn: Scenario, qpos0: np.ndarray | None = None) -> dict:
    """Run one closed loop exactly like run_demo/simulation.run, with instrumentation."""
    wall0 = time.perf_counter()
    nominal = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    model = load_model(scn.mass_scale)
    traj = demo_trajectory(**scn.traj)
    ctrl_model = model if scn.controller_knows_mass else nominal
    controller = CooperativeController(ctrl_model, traj, load_ramp=tuple(DEMO["load_ramp"]),
                                       feedforward_enabled=scn.feedforward, payload_feedback_enabled=True)
    if scn.timestep is not None:
        model.opt.timestep = scn.timestep
    info = ModelInfo.build(model)
    q0 = nominal_qpos0() if qpos0 is None else qpos0.copy()
    if scn.perturb > 0:
        rng = np.random.default_rng(scn.seed)
        q0[info.arm_qadr.ravel()] += rng.uniform(-scn.perturb, scn.perturb, info.arm_qadr.size)
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    duration = scn.duration if scn.duration is not None else traj.duration
    dt = model.opt.timestep
    n = int(round(duration / dt))
    mass = float(model.body_mass[info.payload])
    Ri = quat2mat(model.body_iquat[info.payload])
    I_b = Ri @ np.diag(model.body_inertia[info.payload]) @ Ri.T
    gvec = model.opt.gravity.copy()
    events = sorted(scn.events)
    done_events: set = set()

    L = {  # per-step log (post-step state; contact forces of the step just taken)
        "t": np.zeros(n), "p": np.zeros((n, 3)), "quat": np.zeros((n, 4)),
        "p_des": np.zeros((n, 3)), "phase": np.zeros(n, dtype=int),
        "rot_err": np.zeros(n), "pos_err": np.zeros(n),
        "qarm": np.zeros((n, 4, 6)), "dqarm": np.zeros((n, 4, 6)), "tau": np.zeros((n, 4, 6)),
        "sat_pre": np.zeros((n, 4, 6), dtype=bool), "grip_ctrl": np.zeros((n, 4)),
        "driver": np.zeros((n, 4)),
        "pad_fn": np.zeros((n, 4, 2)), "pad_fz": np.zeros((n, 4)), "pad_plate_fn": np.zeros((n, 4)),
        "pad_wrong_handle_fn": np.zeros((n, 4)), "finger_pay_fn": np.zeros((n, 4)),
        "finger_pay_n": np.zeros((n, 4), dtype=int), "link_pay_n": np.zeros((n, 4), dtype=int),
        "stand_fn": np.zeros(n), "pay_floor_n": np.zeros(n, dtype=int),
        "armarm_n": np.zeros(n, dtype=int), "robot_floor_n": np.zeros(n, dtype=int),
        "robot_stand_n": np.zeros(n, dtype=int), "self_n": np.zeros((n, 4), dtype=int),
        "grip_net_f": np.zeros((n, 4, 3)), "f_alloc": np.zeros((n, 4, 6)), "W_des": np.zeros((n, 6)),
        "ne_lin": np.zeros((n, 3)), "ne_ang": np.zeros((n, 3)), "ne_lin_pads": np.zeros((n, 3)),
        "ne_exact": np.zeros((n, 6)), "ne_solver": np.zeros((n, 6)), "solver_niter": np.zeros(n, dtype=int),
        "slip": np.zeros((n, 4, 3)), "limit_active_n": np.zeros(n, dtype=int),
        "tamper": np.zeros(n, dtype=bool), "pad_pen_max": np.zeros(n),
    }
    phase_names = ("grip", "lift", "hold", "circle", "settle")
    audit_steps = {int(round(ta / dt)) - 1: ta for ta in scn.audit_times}
    audits = []
    frames_t, frames_q = [], []
    frame_k = 0
    if scn.frame_fps > 0:
        frames_t.append(data.time)
        frames_q.append(data.qpos.copy())
        frame_k = 1
    arm_dofs_all = info.arm_dadr.ravel()
    clear = []
    cs = clearance_sets(model, info) if scn.clearance_every > 0 else None
    clear_every = int(round(scn.clearance_every / dt)) if cs else 0
    status, reason = "completed", ""
    last = -1
    for step in range(n):
        t_pre = data.time
        # ---- events that change the model (applied before the step's collision)
        for (te, action) in events:
            if t_pre < te or (te, action) in done_events:
                continue
            if action == "no_stand":
                model.geom_contype[info.stand_geom] = 0
                model.geom_conaffinity[info.stand_geom] = 0
                done_events.add((te, action))
            elif action.startswith("retract"):
                # move every desired pinch point radially outward (off the handle end)
                dist = float(action.split(":")[1])
                hp = controller.handle_pos
                radial = hp.copy()
                radial[:, 2] = 0.0
                radial /= np.linalg.norm(radial, axis=1, keepdims=True)
                controller.handle_pos = hp + dist * radial
                done_events.add((te, action))
        qvel_pre = data.qvel.copy()
        mujoco.mj_step1(model, data)
        snap = (data.qpos.copy(), data.qvel.copy(), data.qfrc_applied.copy(), data.xfrc_applied.copy())
        ctrl, diag = controller.compute(model, data, data.time)
        tamper = not (np.array_equal(snap[0], data.qpos) and np.array_equal(snap[1], data.qvel)
                      and not snap[2].any() and not snap[3].any()
                      and not data.qfrc_applied.any() and not data.xfrc_applied.any())
        ctrl = np.array(ctrl, dtype=float)
        for (te, action) in events:
            if t_pre >= te:
                if action == "open_all":
                    ctrl[info.grip_acts] = 0.0
                elif action == "open_1":
                    ctrl[info.grip_acts[0]] = 0.0
                elif action == "zero_arm":
                    ctrl[info.arm_acts.ravel()] = 0.0
        if step in audit_steps:
            audits.append(frame_audit(model, data, controller, diag, info, traj, audit_steps[step]))
        data.ctrl[:] = ctrl
        mujoco.mj_step2(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            status, reason = "diverged", f"non-finite state at t={data.time:.4f}"
            break
        last = step
        # ---- log
        s = traj.sample(data.time)
        p = data.qpos[info.pay_qadr:info.pay_qadr + 3].copy()
        q = data.qpos[info.pay_qadr + 3:info.pay_qadr + 7].copy()
        R = quat2mat(q)
        L["t"][step] = data.time
        L["p"][step] = p
        L["quat"][step] = q
        L["p_des"][step] = s.p
        L["phase"][step] = phase_names.index(s.phase)
        L["pos_err"][step] = np.linalg.norm(s.p - p)
        L["rot_err"][step] = rot_angle(s.R.T @ R)
        L["qarm"][step] = data.qpos[info.arm_qadr]
        L["dqarm"][step] = data.qvel[info.arm_dadr]
        L["tau"][step] = data.actuator_force[info.arm_acts]
        L["sat_pre"][step] = np.asarray(diag.get("saturated", np.zeros((4, 6), bool)))
        L["grip_ctrl"][step] = ctrl[info.grip_acts]
        L["driver"][step] = data.qpos[info.driver_qadr]
        L["W_des"][step] = np.asarray(diag.get("W_des", np.full(6, np.nan)))
        L["f_alloc"][step] = np.asarray(diag.get("f", np.full((4, 6), np.nan))).reshape(4, 6)
        L["tamper"][step] = tamper
        # contacts: geometry/forces belong to the pre-integration state of this step
        geom, fw, tw, cpos, fn = contact_wrenches(model, data)
        c1, c2 = info.geom_cat[geom[:, 0]], info.geom_cat[geom[:, 1]]
        a1, a2 = info.geom_arm[geom[:, 0]], info.geom_arm[geom[:, 1]]
        pay1 = (c1 == CAT_PLATE) | (c1 == CAT_HANDLE)
        pay2 = (c2 == CAT_PLATE) | (c2 == CAT_HANDLE)
        on_pay = pay1 | pay2
        sign = np.where(pay2, 1.0, -1.0)  # force ON the payload
        other = np.where(pay2, geom[:, 0], geom[:, 1])
        pgeom = np.where(pay2, geom[:, 1], geom[:, 0])
        oc, oa = info.geom_cat[other], info.geom_arm[other]
        f_pay = fw * sign[:, None]
        t_pay = tw * sign[:, None]
        com = data.xipos[info.payload]
        for k in np.flatnonzero(on_pay):
            i = oa[k] - 1
            if oc[k] == CAT_PAD:
                if info.geom_cat[pgeom[k]] == CAT_PLATE:
                    L["pad_plate_fn"][step, i] += fn[k]
                elif info.geom_handle[pgeom[k]] != i + 1:
                    L["pad_wrong_handle_fn"][step, i] += fn[k]
                L["pad_fn"][step, i, info.geom_side[other[k]]] += fn[k]
                L["pad_fz"][step, i] += f_pay[k, 2]
                L["grip_net_f"][step, i] += f_pay[k]
            elif oc[k] == CAT_FINGER:
                L["finger_pay_fn"][step, i] += fn[k]
                L["finger_pay_n"][step, i] += 1
                L["grip_net_f"][step, i] += f_pay[k]
            elif oc[k] == CAT_LINK:
                L["link_pay_n"][step, i] += 1
            elif oc[k] == CAT_STAND:
                L["stand_fn"][step] += fn[k]
            elif oc[k] == CAT_FLOOR:
                L["pay_floor_n"][step] += 1
        rob1, rob2 = a1 > 0, a2 > 0
        L["armarm_n"][step] = int(np.sum(rob1 & rob2 & (a1 != a2)))
        L["self_n"][step] = [int(np.sum(rob1 & rob2 & (a1 == a2) & (a1 == i))) for i in range(1, 5)]
        L["robot_floor_n"][step] = int(np.sum((rob1 & (c2 == CAT_FLOOR)) | (rob2 & (c1 == CAT_FLOOR))))
        L["robot_stand_n"][step] = int(np.sum((rob1 & (c2 == CAT_STAND)) | (rob2 & (c1 == CAT_STAND))))
        # Newton-Euler on the payload with this step's (pre-integration) quantities
        F_c = f_pay[on_pay].sum(axis=0)
        T_c = (np.cross(cpos[on_pay] - com, f_pay[on_pay]) + t_pay[on_pay]).sum(axis=0)
        pads = on_pay & (oc == CAT_PAD)
        F_pads = f_pay[pads].sum(axis=0)
        if pads.any():
            L["pad_pen_max"][step] = float(-np.min(np.array(data.contact.dist[:data.ncon])[pads]))
        acc = data.qacc[info.pay_dadr:info.pay_dadr + 6]
        Rp = data.xmat[info.payload].reshape(3, 3)  # pre-integration (step2 does not refresh kinematics)
        wb = qvel_pre[info.pay_dadr + 3:info.pay_dadr + 6]
        L["ne_lin"][step] = mass * acc[:3] - (F_c + mass * gvec)
        L["ne_lin_pads"][step] = mass * acc[:3] - (F_pads + mass * gvec)
        L["ne_ang"][step] = I_b @ acc[3:] + np.cross(wb, I_b @ wb) - Rp.T @ T_c
        # exact identity: every generalized force on the payload = gravity + gyroscopic + contacts
        qf = data.qfrc_smooth[info.pay_dadr:info.pay_dadr + 6] + data.qfrc_constraint[info.pay_dadr:info.pay_dadr + 6]
        L["ne_exact"][step] = qf - np.r_[F_c + mass * gvec, Rp.T @ T_c - np.cross(wb, I_b @ wb)]
        L["ne_solver"][step] = np.r_[mass * acc[:3], I_b @ acc[3:]] - qf
        L["solver_niter"][step] = int(data.solver_niter[0])
        # grasp slip: pinch position in the handle frame (pre-integration kinematics)
        for i in range(N_ARMS):
            hs, ps = info.handle_sites[i], info.pinch_sites[i]
            L["slip"][step, i] = data.site_xmat[hs].reshape(3, 3).T @ (data.site_xpos[ps] - data.site_xpos[hs])
        # active joint-limit constraints on arm dofs
        if data.nefc:
            ty = data.efc_type[:data.nefc]
            ids = data.efc_id[:data.nefc]
            lim = ty == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
            if lim.any():
                dofs = model.jnt_dofadr[ids[lim]]
                L["limit_active_n"][step] = int(np.isin(dofs, arm_dofs_all).sum())
        if cs is not None and step % clear_every == 0:
            cl = clearance_sample(model, data, cs)
            cl["t"] = float(data.time)
            cl["phase"] = int(L["phase"][step])
            clear.append(cl)
        if scn.frame_fps > 0:
            while data.time >= frame_k / scn.frame_fps - 1e-9:
                frames_t.append(data.time)
                frames_q.append(data.qpos.copy())
                frame_k += 1
    L = {k: v[: last + 1] for k, v in L.items()}
    out = {
        "name": scn.name, "status": status, "reason": reason, "t_end": float(data.time),
        "duration": duration, "wall_s": time.perf_counter() - wall0, "mass": mass,
        "final_qpos": data.qpos.copy(), "audits": audits, "p_path": L["p"].copy(),
        "traj": {"p_lift": traj.p_lift, "p_rest": traj.p_rest, "radius": traj.radius,
                 "A_z": traj.A_z, "k": traj.k, "n_circles": traj.n_circles,
                 "T_ramp": traj.T_ramp, "period": traj.period,
                 "phase_boundaries": traj.phase_boundaries},
    }
    out["metrics"] = run_metrics(L, out)
    if clear:
        keys = [k for k in clear[0] if k not in ("t", "phase", "pairs")]
        carried = [c for c in clear if c["phase"] >= 2]
        out["clearance"] = {
            "min_whole_run_m": {k: min(c[k] for c in clear) for k in keys},
            "min_carried_m": {k: min(c[k] for c in carried) for k in keys} if carried else {},
            "argmin_t": {k: min(clear, key=lambda c: c[k])["t"] for k in keys},
            "argmin_pair": {k: min(clear, key=lambda c: c[k])["pairs"][k] for k in keys},
            "argmin_pair_carried": {k: min(carried, key=lambda c: c[k])["pairs"][k] for k in keys} if carried else {},
            "samples": len(clear),
        }
    out["event_response"] = event_response(L, scn)
    if scn.full_log:
        out["log"] = L
    if scn.frame_fps > 0:
        out["frames_t"] = np.array(frames_t)
        out["frames_q"] = np.array(frames_q)
    return out


# ---------- geometric clearance (near misses that contact counts cannot show)
def clearance_sets(m: mujoco.MjModel, info: ModelInfo) -> dict:
    col = (m.geom_contype | m.geom_conaffinity) != 0
    robot = [np.flatnonzero(col & (info.geom_arm == i)) for i in range(1, N_ARMS + 1)]
    nonpad = [g[info.geom_cat[g] != CAT_PAD] for g in robot]
    pay = np.flatnonzero(col & np.isin(info.geom_cat, [CAT_PLATE, CAT_HANDLE]))
    return {"robot": robot, "nonpad": nonpad, "payload": pay,
            "stand": np.array([info.stand_geom]), "floor": np.flatnonzero(info.geom_cat == CAT_FLOOR)}


def min_distance(m, d, A: np.ndarray, B: np.ndarray, distmax: float = 0.05) -> tuple[float, int, int]:
    """Smallest signed distance between geom sets A and B (capped at distmax) and the pair."""
    if A.size == 0 or B.size == 0:
        return distmax, -1, -1
    pa, pb = d.geom_xpos[A], d.geom_xpos[B]
    ra, rb = m.geom_rbound[A], m.geom_rbound[B]
    gap = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2) - ra[:, None] - rb[None, :]
    # planes have rbound 0: always test them
    gap[:, m.geom_type[B] == mujoco.mjtGeom.mjGEOM_PLANE] = -1.0
    gap[m.geom_type[A] == mujoco.mjtGeom.mjGEOM_PLANE, :] = -1.0
    best, pair = distmax, (-1, -1)
    fromto = np.zeros(6)
    for ia, ib in zip(*np.nonzero(gap < distmax)):
        dist = mujoco.mj_geomDistance(m, d, int(A[ia]), int(B[ib]), distmax, fromto)
        if dist < best:
            best, pair = float(dist), (int(A[ia]), int(B[ib]))
    return best, pair[0], pair[1]


def _geom_label(m, g: int) -> str:
    if g < 0:
        return "-"
    return m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}#{g}"


def clearance_sample(m, d, cs: dict) -> dict:
    res = {
        "arm_arm": min(min_distance(m, d, cs["robot"][i], cs["robot"][j])
                       for i in range(N_ARMS) for j in range(i + 1, N_ARMS)),
        "nonpad_payload": min(min_distance(m, d, g, cs["payload"]) for g in cs["nonpad"]),
        "robot_stand": min(min_distance(m, d, g, cs["stand"]) for g in cs["robot"]),
        "robot_floor": min(min_distance(m, d, g, cs["floor"], 0.2) for g in cs["robot"]),
        "payload_stand": min_distance(m, d, cs["payload"], cs["stand"], 0.2),
    }
    out = {k: v[0] for k, v in res.items()}
    out["pairs"] = {k: f"{_geom_label(m, v[1])} / {_geom_label(m, v[2])}" for k, v in res.items()}
    return out


# ------------------------ trajectory audit: analytic derivatives and C2 joins
def trajectory_audit(traj: PayloadTrajectory) -> dict:
    h = 1e-6
    ts = np.linspace(0.0, traj.duration, 3901)[1:-1]
    bounds = [b[1] for b in traj.phase_boundaries[1:]] + [traj.duration]
    ts = np.array([t for t in ts if min(abs(t - b) for b in bounds) > 3 * h])
    ev, ea = 0.0, 0.0
    for t in ts:
        sm, s0, sp = traj.sample(t - h), traj.sample(t), traj.sample(t + h)
        ev = max(ev, float(np.max(np.abs((sp.p - sm.p) / (2 * h) - s0.v))))
        ea = max(ea, float(np.max(np.abs((sp.v - sm.v) / (2 * h) - s0.a))))
    jumps = {}
    eps = 1e-9
    for name, t0, _ in traj.phase_boundaries[1:]:
        a, b = traj.sample(t0 - eps), traj.sample(t0 + eps)
        jumps[name] = [float(np.max(np.abs(b.p - a.p))), float(np.max(np.abs(b.v - a.v))),
                       float(np.max(np.abs(b.a - a.a)))]
    a, b = traj.sample(traj.duration - eps), traj.sample(traj.duration + eps)
    jumps["end"] = [float(np.max(np.abs(b.p - a.p))), float(np.max(np.abs(b.v - a.v))), float(np.max(np.abs(b.a - a.a)))]
    peak_a = max(float(np.linalg.norm(traj.sample(t).a)) for t in ts)
    peak_v = max(float(np.linalg.norm(traj.sample(t).v)) for t in ts)
    return {"fd_vel_err": ev, "fd_acc_err": ea, "boundary_jumps_p_v_a": jumps, "peak_acc": peak_a, "peak_vel": peak_v}


# ---- frame / sign audit at one instant (called between mj_step1 and mj_step2)
def frame_audit(m, d, controller, diag, info: ModelInfo, traj, t_nominal) -> dict:
    res = {"t": float(d.time), "t_nominal": t_nominal}
    # (a) grasp matrix rebuilt from MuJoCo's own site/COM positions
    com = d.xipos[info.payload].copy()
    r = d.site_xpos[info.handle_sites].copy()
    G = np.zeros((6, 24))
    for i in range(N_ARMS):
        G[:3, 6 * i:6 * i + 3] = np.eye(3)
        G[3:, 6 * i:6 * i + 3] = skew(r[i] - com)
        G[3:, 6 * i + 3:6 * i + 6] = np.eye(3)
    w = np.asarray(diag["f"]).reshape(24)
    W = np.asarray(diag["W_des"])
    res["grasp_matrix_residual"] = float(np.linalg.norm(G @ w - W))
    res["W_des"] = W.copy()
    res["f_alloc"] = w.reshape(4, 6).copy()
    # (b) W_des law recomputed independently (controller's gains, MuJoCo state)
    s = traj.sample(d.time)
    mass = float(m.body_mass[info.payload])
    Ri = quat2mat(m.body_iquat[info.payload])
    I_b = Ri @ np.diag(m.body_inertia[info.payload]) @ Ri.T
    R = d.xmat[info.payload].reshape(3, 3).copy()
    v = d.qvel[info.pay_dadr:info.pay_dadr + 3]
    wbody = d.qvel[info.pay_dadr + 3:info.pay_dadr + 6]
    ww = R @ wbody
    g = controller.gains.object
    lam = controller.load_scale(d.time)
    F = mass * (s.a + g.kp * (s.p - d.xpos[info.payload]) + g.kd * (s.v - v)) - mass * m.opt.gravity
    I_w = R @ I_b @ R.T
    tau = I_w @ (s.alpha + g.kr * so3_log(s.R @ R.T) + g.kw * (s.omega - ww)) + np.cross(ww, I_w @ ww)
    W_ind = lam * np.r_[F, tau]
    res["W_des_law_residual"] = float(np.linalg.norm(W_ind - W))
    res["load_scale"] = lam
    # (c) pinch-site Jacobian vs central finite differences of the site pose
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    mujoco.mj_kinematics(m, d2)
    mujoco.mj_comPos(m, d2)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    errs = []
    h = 1e-6
    for i in range(N_ARMS):
        sid = info.pinch_sites[i]
        mujoco.mj_jacSite(m, d2, jp, jr, sid)
        J = np.vstack([jp[:, info.arm_dadr[i]], jr[:, info.arm_dadr[i]]])
        Jfd = np.zeros((6, 6))
        for j in range(6):
            qa = info.arm_qadr[i, j]
            poses = []
            for sgn in (1.0, -1.0):
                d2.qpos[:] = d.qpos
                d2.qpos[qa] += sgn * h
                mujoco.mj_kinematics(m, d2)
                poses.append((d2.site_xpos[sid].copy(), d2.site_xmat[sid].reshape(3, 3).copy()))
            (pp, Rp), (pm, Rm) = poses
            Jfd[:3, j] = (pp - pm) / (2 * h)
            Jfd[3:, j] = so3_log(Rp @ Rm.T) / (2 * h)  # world-frame angular velocity
        errs.append(float(np.max(np.abs(J - Jfd))))
    res["jacobian_fd_max_abs_err"] = errs
    return res


# -------------------------------------------------------------------- metrics
def run_metrics(L: dict, out: dict) -> dict:
    t = L["t"]
    if t.size == 0:
        return {"empty": True}
    tr = out["traj"]
    ph = L["phase"]
    carried = ph >= 2  # hold, circle, settle
    circle = ph == 3
    mt: dict = {"status": out["status"], "t_end": out["t_end"], "finite": out["status"] == "completed"}

    def mx(x, mask=None):
        x = np.asarray(x)
        if mask is not None:
            x = x[mask]
        return float(np.max(x)) if x.size else float("nan")

    mt["pos_err_max_carried_m"] = mx(L["pos_err"], carried)
    mt["pos_err_rms_carried_m"] = float(np.sqrt(np.mean(L["pos_err"][carried] ** 2))) if carried.any() else float("nan")
    mt["pos_err_max_all_m"] = mx(L["pos_err"])
    mt["rot_err_max_carried_rad"] = mx(L["rot_err"], carried)
    mt["rot_err_max_all_rad"] = mx(L["rot_err"])
    # lift-off
    z = L["p"][:, 2]
    zr = float(tr["p_rest"][2])
    mt["stand_fn_max_carried_N"] = mx(L["stand_fn"], carried)
    mt["stand_contact_samples_carried"] = int(np.sum(L["stand_fn"][carried] > 0))
    lift = ph == 1
    liftoff = np.flatnonzero(lift & (L["stand_fn"] <= 0))
    mt["liftoff_t"] = float(t[liftoff[0]]) if liftoff.size else float("nan")
    mt["z_min_above_rest_carried_m"] = float(np.min(z[carried]) - zr) if carried.any() else float("nan")
    # grasp forces
    pad_tot = L["pad_fn"].sum(axis=2)
    mt["pad_fn_min_carried_per_arm_N"] = pad_tot[carried].min(axis=0) if carried.any() else np.full(4, np.nan)
    mt["pad_fn_min_carried_per_pad_N"] = L["pad_fn"][carried].min(axis=0) if carried.any() else np.full((4, 2), np.nan)
    mt["pad_fn_mean_carried_per_arm_N"] = pad_tot[carried].mean(axis=0) if carried.any() else np.full(4, np.nan)
    mt["pad_fz_mean_carried_per_arm_N"] = L["pad_fz"][carried].mean(axis=0) if carried.any() else np.full(4, np.nan)
    mt["pad_fz_min_carried_per_arm_N"] = L["pad_fz"][carried].min(axis=0) if carried.any() else np.full(4, np.nan)
    mt["grasp_loss_samples"] = int(np.sum(np.any(pad_tot[carried] <= FORCE_EPS, axis=1)))
    # unwanted contacts over the WHOLE run
    unw = {
        "arm_arm": int(np.sum(L["armarm_n"] > 0)),
        "robot_floor": int(np.sum(L["robot_floor_n"] > 0)),
        "robot_stand": int(np.sum(L["robot_stand_n"] > 0)),
        "payload_floor": int(np.sum(L["pay_floor_n"] > 0)),
        "arm_link_payload": int(np.sum(L["link_pay_n"].sum(axis=1) > 0)),
        "gripper_nonpad_payload": int(np.sum(L["finger_pay_n"].sum(axis=1) > 0)),
        "pad_on_plate": int(np.sum(L["pad_plate_fn"].sum(axis=1) > 0)),
        "pad_on_wrong_handle": int(np.sum(L["pad_wrong_handle_fn"].sum(axis=1) > 0)),
    }
    mt["unwanted_contact_steps"] = unw
    first = {}
    for key, arr in (("arm_arm", L["armarm_n"]), ("robot_floor", L["robot_floor_n"]),
                     ("robot_stand", L["robot_stand_n"]), ("payload_floor", L["pay_floor_n"]),
                     ("arm_link_payload", L["link_pay_n"].sum(axis=1)),
                     ("gripper_nonpad_payload", L["finger_pay_n"].sum(axis=1))):
        idx = np.flatnonzero(arr > 0)
        if idx.size:
            first[key] = float(t[idx[0]])
    mt["unwanted_first_t"] = first
    mt["gripper_nonpad_payload_fn_max_N"] = mx(L["finger_pay_fn"])
    mt["self_collision_steps_per_arm"] = (L["self_n"] > 0).sum(axis=0)
    # actuation
    frac = np.abs(L["tau"]) / TORQUE_LIMITS
    mt["torque_frac_max"] = mx(frac)
    mt["torque_frac_max_per_joint"] = frac.max(axis=(0, 1))
    mt["saturated_step_fraction"] = float(np.mean(np.any(frac >= 0.999, axis=(1, 2))))
    mt["saturated_pre_clip_step_fraction"] = float(np.mean(np.any(L["sat_pre"], axis=(1, 2))))
    mt["joint_speed_max_rad_s"] = mx(np.abs(L["dqarm"]))
    mt["joint_speed_max_per_joint"] = np.abs(L["dqarm"]).max(axis=(0, 1))
    mt["_qarm_min"] = L["qarm"].min(axis=0)
    mt["_qarm_max"] = L["qarm"].max(axis=0)
    mt["limit_active_steps"] = int(np.sum(L["limit_active_n"] > 0))
    # circle geometry of the ACTUAL payload path
    c = np.asarray(tr["p_lift"])
    R = float(tr["radius"])
    if circle.any():
        xy = L["p"][circle, :2] - c[:2]
        rad = np.linalg.norm(xy, axis=1)
        ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        dth = np.diff(ang)
        ok = (rad[1:] >= BARS["circle_radius_frac"] * R) & (rad[:-1] >= BARS["circle_radius_frac"] * R)
        mt["swept_angle_at_0.9R_rad"] = float(np.sum(dth[ok]))
        mt["swept_angle_total_rad"] = float(ang[-1] - ang[0])
        mt["radius_max_m"] = float(rad.max())
        full = rad >= BARS["circle_radius_frac"] * R
        mt["radius_mean_at_plateau_m"] = float(rad[full].mean()) if full.any() else float("nan")
        # least-squares circle centre of the plateau samples
        if full.sum() > 10:
            A = np.c_[2 * xy[full], np.ones(full.sum())]
            sol, *_ = np.linalg.lstsq(A, (xy[full] ** 2).sum(axis=1), rcond=None)
            mt["fitted_centre_offset_m"] = float(np.linalg.norm(sol[:2]))
            mt["fitted_radius_m"] = float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))
        dz = L["p"][circle, 2] - c[2]
        mt["z_excursion_up_m"] = float(dz.max())
        mt["z_excursion_down_m"] = float(dz.min())
        thr = BARS["z_excursion_frac"] * float(tr["A_z"])
        # bobs per circle: Z peaks above 0.8 A_z inside the full-radius window, which is
        # exactly n_circles periods long (one full turn per period)
        t_c0 = [b for b in tr["phase_boundaries"] if b[0] == "circle"][0][1]
        w0 = t_c0 + tr["T_ramp"]
        w1 = w0 + tr["n_circles"] * tr["period"]
        win = (t[circle] >= w0) & (t[circle] < w1)
        # hysteresis: a bob starts above +thr and ends when z falls back below 0
        def count(x, hi):
            n_b, armed = 0, True
            for v in x:
                if armed and v > hi:
                    n_b, armed = n_b + 1, False
                elif not armed and v < 0.0:
                    armed = True
            return n_b
        mt["z_bobs_per_circle_window"] = count(dz[win], thr)
        mt["z_dips_per_circle_window"] = count(-dz[win], thr)
        mt["z_bobs_whole_circle_phase"] = count(dz, thr)
        mt["circle_window_s"] = [float(w0), float(w1)]
        xyw = xy[win]
        angw = np.unwrap(np.arctan2(xyw[:, 1], xyw[:, 0]))
        mt["swept_angle_in_circle_window_rad"] = float(angw[-1] - angw[0]) if angw.size else float("nan")
    # slip: drift of the pinch point in the handle frame while carried
    if carried.any():
        sl = L["slip"][carried]
        mt["slip_drift_max_mm"] = 1e3 * np.linalg.norm(sl - sl[0], axis=2).max(axis=0)
        mt["pinch_handle_offset_start_carry_mm"] = 1e3 * np.linalg.norm(sl[0], axis=1)
    # Newton-Euler residual (whole run, all contacts / pads only)
    mg = out["mass"] * 9.81
    mt["newton_euler_lin_max_N"] = mx(np.linalg.norm(L["ne_lin"], axis=1))
    mt["newton_euler_ang_max_Nm"] = mx(np.linalg.norm(L["ne_ang"], axis=1))
    mt["newton_euler_lin_rel"] = mt["newton_euler_lin_max_N"] / mg
    mt["newton_euler_lin_pads_only_max_N_carried"] = mx(np.linalg.norm(L["ne_lin_pads"], axis=1), carried)
    mt["newton_euler_exact_lin_max_N"] = mx(np.linalg.norm(L["ne_exact"][:, :3], axis=1))
    mt["newton_euler_exact_ang_max_Nm"] = mx(np.linalg.norm(L["ne_exact"][:, 3:], axis=1))
    mt["solver_residual_lin_max_N"] = mx(np.linalg.norm(L["ne_solver"][:, :3], axis=1))
    mt["solver_residual_ang_max_Nm"] = mx(np.linalg.norm(L["ne_solver"][:, 3:], axis=1))
    mt["solver_niter_max"] = int(L["solver_niter"].max())
    mt["solver_niter_median"] = float(np.median(L["solver_niter"]))
    mt["controller_tampered_steps"] = int(L["tamper"].sum())
    mt["pad_penetration_max_carried_mm"] = 1e3 * mx(L["pad_pen_max"], carried)
    # load sharing: allocated vs realised vertical force per gripper while carried
    if carried.any():
        mt["alloc_fz_mean_carried_per_arm_N"] = L["f_alloc"][carried, :, 2].mean(axis=0)
        mt["realised_grip_fz_mean_carried_per_arm_N"] = L["grip_net_f"][carried, :, 2].mean(axis=0)
        mt["realised_grip_f_mean_carried_per_arm_N"] = L["grip_net_f"][carried].mean(axis=0)
    mt["driver_angle_range"] = [float(L["driver"].min()), float(L["driver"].max())]
    return mt


def event_response(L: dict, scn: Scenario) -> dict:
    """Payload drop etc. after the first scenario event (causality tests)."""
    if not scn.events or L["t"].size == 0:
        return {}
    if scn.response_t is not None:
        te = scn.response_t
    else:
        te = min(t for t, a in scn.events if a != "no_stand") if any(a != "no_stand" for _, a in scn.events) else min(t for t, _ in scn.events)
    t = L["t"]
    i0 = int(np.searchsorted(t, te))
    if i0 >= t.size:
        return {"note": "event after end of run"}
    z0 = float(L["p"][i0, 2])
    res = {"t_event": te, "z_at_event": z0, "p_at_event": L["p"][i0]}
    for horizon in (0.25, 0.5, 1.0, 2.0):
        sel = (t >= te) & (t <= te + horizon)
        if sel.any() and t[-1] >= te + horizon - 1e-9:
            res[f"z_drop_{horizon}s_m"] = float(z0 - L["p"][sel, 2].min())
            res[f"pos_err_max_{horizon}s_m"] = float(L["pos_err"][sel].max())
            res[f"rot_err_max_{horizon}s_rad"] = float(L["rot_err"][sel].max())
    post = t >= te
    res["pad_fn_end_per_arm_N"] = L["pad_fn"][-1].sum(axis=1)
    res["finger_fn_end_per_arm_N"] = L["finger_pay_fn"][-1]
    res["stand_fn_end_N"] = float(L["stand_fn"][-1])
    res["stand_contact_after_event"] = bool(np.any(L["stand_fn"][post] > 0))
    res["pay_floor_after_event"] = bool(np.any(L["pay_floor_n"][post] > 0))
    res["driver_angle_end"] = L["driver"][-1]
    res["z_end"] = float(L["p"][-1, 2])
    sel1 = (t >= te) & (t <= te + 1.0)
    res["z_min_1s"] = float(L["p"][sel1, 2].min()) if sel1.any() else float("nan")
    res["pad_fz_end_per_arm_N"] = L["pad_fz"][-1]
    res["rot_err_end_rad"] = float(L["rot_err"][-1])
    res["pos_err_end_m"] = float(L["pos_err"][-1])
    res["grasp_loss_arms_end"] = [i + 1 for i in range(4) if L["pad_fn"][-1, i].sum() <= FORCE_EPS]
    return res


# --------------------------------------------------- check 1: model integrity
def check_model(model: mujoco.MjModel) -> list[Check]:
    checks = []
    names_b = [model.body(b).name for b in range(model.nbody)]
    arms = sorted({int(mt.group(1)) for n in names_b if (mt := re.match(r"ur(\d+)_base$", n))})
    grippers = sorted({int(mt.group(1)) for n in names_b if (mt := re.match(r"ur(\d+)_g_base$", n))})
    prefixes = sorted({mt.group(1) for n in names_b if (mt := re.match(r"(ur\d+)_", n))})
    arm_joints_ok = all(model.joint(f"ur{i}_{j}").id >= 0 for i in range(1, 5) for j in ARM_JOINTS)
    ok = arms == [1, 2, 3, 4] and grippers == [1, 2, 3, 4] and prefixes == ["ur1", "ur2", "ur3", "ur4"] and arm_joints_ok
    checks.append(Check("1a", "4 UR5e arms + 4 2F-85 grippers (by name prefix)", "PASS" if ok else "FAIL",
                        f"arms={arms} grippers={grippers}", {"prefixes": prefixes, "nbody": model.nbody}))
    free = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    fb = model.jnt_bodyid[model.joint("payload_free").id] if "payload_free" in free else -1
    ok = free == ["payload_free"] and fb == model.body("payload").id and model.body_parentid[fb] == 0
    checks.append(Check("1b", "exactly one free joint = payload_free (child of world)", "PASS" if ok else "FAIL",
                        f"free={free}"))
    # equality constraints: every one inside ONE gripper
    eqs, bad = [], []
    for e in range(model.neq):
        ty = int(model.eq_type[e])
        o1, o2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if ty in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
            objtype = int(model.eq_objtype[e])
            if objtype == int(mujoco.mjtObj.mjOBJ_SITE):
                n1 = model.site(o1).name
                n2 = model.site(o2).name if o2 >= 0 else "world"
            else:
                n1 = model.body(o1).name
                n2 = model.body(o2).name if o2 >= 0 else "world"
        elif ty == int(mujoco.mjtEq.mjEQ_JOINT):
            n1 = model.joint(o1).name
            n2 = model.joint(o2).name if o2 >= 0 else "(none)"
        elif ty == int(mujoco.mjtEq.mjEQ_TENDON):
            n1 = model.tendon(o1).name
            n2 = model.tendon(o2).name if o2 >= 0 else "(none)"
        else:
            n1, n2 = str(o1), str(o2)
        rec = {"id": e, "type": mujoco.mjtEq(ty).name, "obj1": n1, "obj2": n2, "active0": bool(model.eq_active0[e])}
        eqs.append(rec)
        m1 = re.match(r"(ur\d+)_g_", n1)
        m2 = re.match(r"(ur\d+)_g_", n2)
        if not (m1 and m2 and m1.group(1) == m2.group(1)):
            bad.append(rec)
    checks.append(Check("1c", "every equality constraint is gripper-internal (no payload/world weld)",
                        "PASS" if not bad and model.neq == 12 else "FAIL",
                        f"neq={model.neq}, non-gripper={len(bad)}", {"equalities": eqs, "bad": bad}))
    # payload mass / inertia vs an independent box computation from the geoms
    pay = model.body("payload").id
    Ic = np.zeros((3, 3))
    msum = 0.0
    # MuJoCo does not store geom masses in mjModel: take the geom mass attributes from the MJCF
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    for gs in spec.body("payload").geoms:
        mg = float(gs.mass)
        a, b, c = 2 * np.asarray(gs.size[:3])
        Ib = mg / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        Rg = quat2mat(np.asarray(gs.quat))
        pg = np.asarray(gs.pos)
        Ic += Rg @ Ib @ Rg.T + mg * (pg @ pg * np.eye(3) - np.outer(pg, pg))
        msum += mg
    Ri = quat2mat(model.body_iquat[pay])
    I_model = Ri @ np.diag(model.body_inertia[pay]) @ Ri.T
    mass = float(model.body_mass[pay])
    ok = (abs(mass - 1.0) < 1e-9 and abs(msum - mass) < 1e-9 and np.allclose(I_model, Ic, atol=1e-9)
          and np.linalg.norm(model.body_ipos[pay]) < 1e-9 and np.all(model.body_inertia[pay] > 0))
    checks.append(Check("1d", "payload mass 1.0 kg, inertia = independent box sum, COM at origin",
                        "PASS" if ok else "FAIL",
                        f"m={mass:.6f} kg, diag(I)={np.round(np.diag(I_model), 5).tolist()}",
                        {"mass": mass, "mass_from_geoms": msum, "I_model": I_model, "I_indep": Ic,
                         "ipos": model.body_ipos[pay]}))
    # torque actuators
    bad = []
    for i in range(1, 5):
        for a, lim in zip(ARM_ACTS, TORQUE_LIMITS):
            aid = model.actuator(f"ur{i}_{a}").id
            jid = model.joint(f"ur{i}_{a}_joint").id
            good = (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[aid, 0] == jid
                    and model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
                    and model.actuator_gainprm[aid, 0] == 1.0 and not model.actuator_gainprm[aid, 1:].any()
                    and model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_NONE
                    and model.actuator_gear[aid, 0] == 1.0
                    and bool(model.actuator_ctrllimited[aid]) and bool(model.actuator_forcelimited[aid])
                    and np.allclose(model.actuator_ctrlrange[aid], [-lim, lim])
                    and np.allclose(model.actuator_forcerange[aid], [-lim, lim]))
            if not good:
                bad.append(f"ur{i}_{a}")
    grip_ok = all(model.actuator(f"ur{i}_fingers_actuator").id >= 0 for i in range(1, 5))
    checks.append(Check("1e", "arm actuators are pure torque, ctrl=force range ±(150,150,150,28,28,28)",
                        "PASS" if not bad and grip_ok and model.nu == 28 else "FAIL",
                        f"nu={model.nu}, bad={bad}"))
    # hidden forces on the payload / hidden arm gravity compensation
    pdofs = np.arange(model.jnt_dofadr[model.joint("payload_free").id], model.jnt_dofadr[model.joint("payload_free").id] + 6)
    act_on_payload = []
    for a in range(model.nu):
        tt, tid = model.actuator_trntype[a], model.actuator_trnid[a, 0]
        if tt in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT) and model.jnt_bodyid[tid] == pay:
            act_on_payload.append(model.actuator(a).name)
        if tt == mujoco.mjtTrn.mjTRN_SITE and model.site_bodyid[tid] == pay:
            act_on_payload.append(model.actuator(a).name)
        if tt == mujoco.mjtTrn.mjTRN_BODY and tid == pay:
            act_on_payload.append(model.actuator(a).name)
    tendon_on_payload = []
    for w in range(model.nwrap):
        wt, oid = model.wrap_type[w], model.wrap_objid[w]
        if wt == mujoco.mjtWrap.mjWRAP_SITE and model.site_bodyid[oid] == pay:
            tendon_on_payload.append(w)
        if wt == mujoco.mjtWrap.mjWRAP_JOINT and model.jnt_bodyid[oid] == pay:
            tendon_on_payload.append(w)
    hidden = {
        "gravity": model.opt.gravity.tolist(),
        "payload_dof_damping": model.dof_damping[pdofs].tolist(),
        "payload_dof_armature": model.dof_armature[pdofs].tolist(),
        "payload_dof_frictionloss": model.dof_frictionloss[pdofs].tolist(),
        "body_gravcomp_nonzero": [model.body(b).name for b in range(model.nbody) if model.body_gravcomp[b] != 0],
        "jnt_actgravcomp_nonzero": [model.joint(j).name for j in range(model.njnt) if model.jnt_actgravcomp[j]],
        "nmocap": model.nmocap, "density": model.opt.density, "viscosity": model.opt.viscosity,
        "wind": model.opt.wind.tolist(), "actuators_on_payload": act_on_payload,
        "tendon_wraps_on_payload": tendon_on_payload, "npair": model.npair,
    }
    ok = (np.allclose(model.opt.gravity, [0, 0, -9.81]) and not np.any(model.dof_damping[pdofs])
          and not np.any(model.dof_armature[pdofs]) and not np.any(model.dof_frictionloss[pdofs])
          and not hidden["body_gravcomp_nonzero"] and not hidden["jnt_actgravcomp_nonzero"]
          and model.nmocap == 0 and model.opt.density == 0 and model.opt.viscosity == 0
          and not np.any(model.opt.wind) and not act_on_payload and not tendon_on_payload)
    checks.append(Check("1f", "no hidden forces: gravity -9.81, no payload damping/actuator/tendon/gravcomp/mocap/fluid",
                        "PASS" if ok else "FAIL", "", hidden))
    # model file reproducible from scene.py, asset XMLs unmodified
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        fresh = portable_xml(build_spec())
    same = fresh == MODEL_XML.read_text()
    asset_now = {"ur5e.xml": sha256(HERE / "assets" / "ur5e" / "ur5e.xml"),
                 "2f85.xml": sha256(HERE / "assets" / "robotiq_2f85" / "2f85.xml")}
    assets_ok = asset_now == MENAGERIE_SHA256
    checks.append(Check("1g", "model XML == fresh build from scene.py; asset XMLs == Menagerie sha256",
                        "PASS" if same and assets_ok else "FAIL",
                        f"xml_reproducible={same}, assets_match={assets_ok}",
                        {"model_sha256": sha256(MODEL_XML), "assets": asset_now, "expected": MENAGERIE_SHA256}))
    return checks


def check_initial_state(model: mujoco.MjModel, q0: np.ndarray) -> Check:
    """At t=0: payload rests on the stand only, no robot touches anything, pads clear of the handle."""
    d = mujoco.MjData(model)
    d.qpos[:] = q0
    mujoco.mj_forward(model, d)
    info = ModelInfo.build(model)
    geom, fw, tw, pos, fn = contact_wrenches(model, d)
    pairs = []
    ok = True
    for k in range(geom.shape[0]):
        g1, g2 = geom[k]
        c1, c2 = info.geom_cat[g1], info.geom_cat[g2]
        cats = {c1, c2}
        if cats <= {CAT_PLATE, CAT_STAND} and CAT_STAND in cats:
            continue
        ok = False
        pairs.append(f"{model.geom(g1).name or model.body(model.geom_bodyid[g1]).name}-"
                     f"{model.geom(g2).name or model.body(model.geom_bodyid[g2]).name}")
    # pad clearance to the handle (signed distance, positive = gap)
    clear = []
    fromto = np.zeros(6)
    for i in range(1, N_ARMS + 1):
        hg = model.geom(f"handle_{i}_geom").id
        dmin = np.inf
        for g in range(model.ngeom):
            if info.geom_cat[g] == CAT_PAD and info.geom_arm[g] == i:
                dmin = min(dmin, mujoco.mj_geomDistance(model, d, g, hg, 0.05, fromto))
        clear.append(float(dmin))
    ok = ok and all(c > 0 for c in clear)
    return Check("1h", "initial state: only plate-stand contact, pads clear of the handles",
                 "PASS" if ok else "FAIL",
                 f"other contacts={pairs or 'none'}; pad-handle clearance per arm={np.round(np.array(clear) * 1e3, 2).tolist()} mm",
                 {"clearance_m": clear, "contacts": pairs})


# ------------------------------------- check 2: determinism (fresh processes)
def run_demo_subprocess(name: str) -> dict:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl")
    t0 = time.perf_counter()
    cp = subprocess.run([sys.executable, str(HERE / "scripts" / "run_demo.py"), "--name", name],
                        cwd=str(HERE), env=env, capture_output=True, text=True)
    return {"name": name, "returncode": cp.returncode, "wall_s": time.perf_counter() - t0,
            "stderr_tail": cp.stderr[-800:], "stdout_tail": cp.stdout[-400:]}


def compare_runs(a: Path, b: Path) -> dict:
    out = {"identical": True, "max_abs_diff": 0.0, "arrays": {}}
    for fname in ("log.npz", "frames.npz"):
        with np.load(a / fname) as za, np.load(b / fname) as zb:
            for k in za.files:
                x, y = za[k], zb[k]
                if x.dtype.kind in "fc" and x.shape == y.shape:
                    diff = float(np.max(np.abs(x - y))) if x.size else 0.0
                    same = bool(np.array_equal(x, y, equal_nan=True))
                else:
                    same = bool(x.shape == y.shape and np.array_equal(x, y))
                    diff = 0.0 if same else float("inf")
                if k == "model_path":
                    continue
                out["arrays"][f"{fname}:{k}"] = diff
                out["identical"] &= same
                out["max_abs_diff"] = max(out["max_abs_diff"], diff)
    return out


# ---------------------------------------------- check 7: video correspondence
def count_video_frames(path: Path) -> int:
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cp = subprocess.run([exe, "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                        capture_output=True, text=True)
    frames = re.findall(r"frame=\s*(\d+)", cp.stderr)
    return int(frames[-1]) if frames else -1


def check_video(main: dict | None) -> list[Check]:
    side = VIDEO.with_suffix(".json")
    unverified = VIDEO.with_name(VIDEO.stem + ".UNVERIFIED.mp4")
    if not VIDEO.exists() and unverified.exists():
        uside = unverified.with_suffix(".json")
        probs = json.loads(uside.read_text()).get("problems") if uside.exists() else "no sidecar"
        return [Check("7", "video correspondence", "FAIL",
                      f"only {unverified.name} exists (render_video's own verification failed): {probs}")]
    if not VIDEO.exists() or not side.exists():
        return [Check("7", "video correspondence", "SKIP",
                      f"{VIDEO.relative_to(HERE)} or its .json sidecar does not exist (pixi run render)")]
    rep = json.loads(side.read_text())
    rec_path = Path(rep.get("record", ""))
    m: dict = {"sidecar": rep.get("video"), "record": str(rec_path), "verified_flag": rep.get("verified")}
    problems = []
    decoded = count_video_frames(VIDEO)
    m["decoded_frames"] = decoded
    m["sidecar_frames"] = int(rep.get("title_frames", 0)) + int(rep.get("sim_frames", 0))
    if decoded != m["sidecar_frames"]:
        problems.append(f"decoded {decoded} frames != sidecar {m['sidecar_frames']}")
    cur = sha256(MODEL_XML)
    m["model_sha256_now"] = cur
    m["model_sha256_sidecar"] = rep.get("model_sha256")
    if rep.get("model_sha256") != cur:
        problems.append("sidecar model sha256 != current model file")
    if rec_path.exists():
        with np.load(rec_path) as z:
            n_rec = int(z["t"].shape[0])
            rec_sha = str(z["model_sha256"]) if "model_sha256" in z.files else ""
            rec_q = np.array(z["qpos"])
            rec_t = np.array(z["t"])
            rec_fps = float(z["fps"])
            rec_dt = float(z["timestep"]) if "timestep" in z.files else 1e-3
        m.update(record_frames=n_rec, record_model_sha256=rec_sha, record_fps=rec_fps)
        # real time: recorded frame k sits at sim time t0 + k/fps (within one step), and the
        # video plays the sim frames at that same rate (no slow motion)
        lag = float(np.max(np.abs(rec_t - rec_t[0] - np.arange(n_rec) / rec_fps))) if n_rec else float("nan")
        m["record_frame_time_max_err_s"] = lag
        m["video_fps"] = rep.get("fps")
        if not lag <= rec_dt + 1e-9:
            problems.append(f"record frames are not at 1/fps sim-time spacing (max err {lag:.4f} s)")
        if rep.get("fps") is None or abs(float(rep["fps"]) - rec_fps) > 1e-9:
            problems.append(f"video fps {rep.get('fps')} != record fps {rec_fps} (not real time)")
        m["sim_seconds_in_video"] = (decoded - int(rep.get("title_frames", 0))) / float(rep.get("fps") or rec_fps)
        m["sim_seconds_recorded"] = float(rec_t[-1] - rec_t[0]) + 1.0 / rec_fps if n_rec else float("nan")
        if rec_sha != cur:
            problems.append("record model sha256 != current model file")
        if rep.get("fps") and abs(float(rep["fps"]) - rec_fps) < 1e-9 and n_rec != int(rep.get("sim_frames", -1)):
            problems.append(f"record has {n_rec} frames, sidecar sim_frames {rep.get('sim_frames')}")
        summ = rec_path.parent / "summary.json"
        if summ.exists():
            sj = json.loads(summ.read_text())
            m["record_run_status"] = sj.get("status")
            m["record_run_notes"] = sj.get("config", {}).get("notes")
            if sj.get("status") != "completed":
                problems.append(f"recorded run status {sj.get('status')}")
        if main is not None and "frames_q" in main:
            fq = main["frames_q"]
            k = min(len(fq), len(rec_q))
            m["state_max_abs_diff_vs_fresh_run"] = float(np.max(np.abs(fq[:k] - rec_q[:k]))) if k else float("nan")
            m["frames_compared"] = k
            if len(fq) != len(rec_q) or m["state_max_abs_diff_vs_fresh_run"] > 1e-9:
                problems.append("recorded states differ from a fresh default-demo run "
                                f"(max |dq| = {m['state_max_abs_diff_vs_fresh_run']:.3g}, "
                                f"frames {len(rec_q)} vs {len(fq)})")
    else:
        problems.append(f"record {rec_path} missing")
    return [Check("7", "video: frames, model sha, recorded states == fresh default run",
                  "PASS" if not problems else "FAIL", "; ".join(problems) or "consistent", m)]


# -------------------------------------------------- check 8: self-containment
def source_files() -> list[Path]:
    """Tracked text sources (``git ls-files``; without git, every file outside
    generated directories)."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True,
                             check=True).stdout
        files = [HERE / f for f in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:  # no git, or an untracked copy inside another work tree
        skip = {".git", ".pixi", "outputs", "tmp", "__pycache__"}
        files = [f for f in HERE.rglob("*") if not skip & set(f.relative_to(HERE).parts)]
    return [f for f in files
            if f.is_file() and f.suffix in (".py", ".sh", ".toml", ".ini", ".cfg", ".xml")]


def check_self_contained() -> list[Check]:
    """No absolute user paths, no imports beyond the standard library, the declared
    dependencies and ``homtrans``; every mesh of the model file inside the repository."""
    root = str(HERE.resolve())
    abs_path = re.compile(r"/(home|Users)/|[A-Za-z]:\\{1,2}Users|" + re.escape(root))
    allowed = set(sys.stdlib_module_names) | DEPENDENCIES | {"homtrans"}
    abs_hits, import_hits = [], []
    sources = source_files()
    for f in sources:
        rel = f.relative_to(HERE)
        text = f.read_text(errors="replace")
        for ln, line in enumerate(text.splitlines(), 1):
            if abs_path.search(line):
                abs_hits.append(f"{rel}:{ln}: {line.strip()[:120]}")
        if f.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:  # relative: package-internal
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".")[0] not in allowed:
                    import_hits.append(f"{rel}:{node.lineno}: {name}")
    meshes = re.findall(r'file="([^"]+)"', MODEL_XML.read_text())
    resolved = [(MODEL_XML.parent / m).resolve() for m in meshes]
    outside = [m for m, r in zip(meshes, resolved) if not r.is_relative_to(root)]
    missing = [m for m, r in zip(meshes, resolved) if not r.is_file()]
    ok = bool(meshes) and not abs_hits and not import_hits and not outside and not missing
    return [Check("8", "self-contained: no absolute user paths/foreign imports; meshes inside repo",
                  "PASS" if ok else "FAIL",
                  f"{len(sources)} source files: {len(abs_hits)} absolute paths, {len(import_hits)} foreign "
                  f"imports; {len(meshes)} model meshes: {len(outside)} outside, {len(missing)} missing",
                  {"absolute_paths": abs_hits, "foreign_imports": import_hits,
                   "meshes_outside": outside[:10], "meshes_missing": missing[:10]})]


# ------------------------------ assembling checks 3-6 from simulation results
def checks_full_run(r: dict) -> list[Check]:
    mt = r["metrics"]
    C = []
    C.append(Check("3a", "full demo run completes, finite state", "PASS" if mt["finite"] else "FAIL",
                   f"status={r['status']} t_end={r['t_end']:.3f}/{r['duration']} {r['reason']}"))
    ok = (mt["pos_err_max_carried_m"] < BARS["pos_err_max_m"] and mt["pos_err_rms_carried_m"] < BARS["pos_err_rms_m"])
    C.append(Check("3b", f"payload tracking while carried (max<{BARS['pos_err_max_m']*1e3:.0f} mm, rms<{BARS['pos_err_rms_m']*1e3:.0f} mm)",
                   "PASS" if ok else "FAIL",
                   f"max={mt['pos_err_max_carried_m']*1e3:.2f} mm rms={mt['pos_err_rms_carried_m']*1e3:.2f} mm (whole run max {mt['pos_err_max_all_m']*1e3:.2f} mm)"))
    C.append(Check("3c", f"orientation error while carried < {BARS['rot_err_max_rad']} rad",
                   "PASS" if mt["rot_err_max_carried_rad"] < BARS["rot_err_max_rad"] else "FAIL",
                   f"max={mt['rot_err_max_carried_rad']:.4f} rad ({np.degrees(mt['rot_err_max_carried_rad']):.2f} deg)"))
    ok = mt["stand_contact_samples_carried"] == 0 and mt["z_min_above_rest_carried_m"] > 0.0
    C.append(Check("3d", "lifted off the stand: 0 stand force while carried, z above rest", "PASS" if ok else "FAIL",
                   f"lift-off t={mt['liftoff_t']:.3f} s, stand samples carried={mt['stand_contact_samples_carried']}, "
                   f"min z-z_rest={mt['z_min_above_rest_carried_m']*1e3:.1f} mm"))
    pmin = np.asarray(mt["pad_fn_min_carried_per_arm_N"])
    ppad = np.asarray(mt["pad_fn_min_carried_per_pad_N"])
    ok = mt["grasp_loss_samples"] == 0 and np.all(pmin > FORCE_EPS)
    C.append(Check("3e", "every gripper's pad normal force > 0 throughout the carry", "PASS" if ok else "FAIL",
                   f"min per arm = {np.round(pmin, 1).tolist()} N; min per pad (L,R) = {np.round(ppad, 1).tolist()}",
                   {"mean_per_arm_N": mt["pad_fn_mean_carried_per_arm_N"],
                    "pad_fz_mean_per_arm_N": mt["pad_fz_mean_carried_per_arm_N"],
                    "pad_fz_min_per_arm_N": mt["pad_fz_min_carried_per_arm_N"]}))
    unw = mt["unwanted_contact_steps"]
    bad = {k: v for k, v in unw.items() if v}
    C.append(Check("3f", "no arm-arm / arm-floor / arm-stand / payload-floor / non-pad-payload / pad-plate contacts",
                   "PASS" if not bad else "FAIL",
                   ("none" if not bad else f"steps with contact: {bad}; first t: {mt['unwanted_first_t']}")
                   + f"; same-arm self-contact steps per arm {np.asarray(mt['self_collision_steps_per_arm']).tolist()} (info)",
                   {"steps": unw, "first_t": mt["unwanted_first_t"],
                    "gripper_nonpad_payload_fn_max_N": mt["gripper_nonpad_payload_fn_max_N"],
                    "self_collision_steps_per_arm": mt["self_collision_steps_per_arm"]}))
    C.append(Check("3g", "torque saturation", "PASS" if mt["saturated_step_fraction"] == 0 else "FAIL",
                   f"saturated steps={mt['saturated_step_fraction']*100:.2f} %, pre-clip={mt['saturated_pre_clip_step_fraction']*100:.2f} %, "
                   f"max |tau|/limit={mt['torque_frac_max']:.3f}",
                   {"per_joint_max_frac": mt["torque_frac_max_per_joint"]}))
    C.append(Check("3h", "max arm joint speed < UR5e 180 deg/s", "PASS" if mt["joint_speed_max_rad_s"] < UR5E_MAX_JOINT_SPEED else "FAIL",
                   f"max |qd|={mt['joint_speed_max_rad_s']:.3f} rad/s", {"per_joint": mt["joint_speed_max_per_joint"]}))
    lo, hi = r["jnt_lo"], r["jnt_hi"]
    margin = float(np.min(np.minimum(mt["_qarm_min"] - lo, hi - mt["_qarm_max"])))
    C.append(Check("3i", f"joint-limit margin > {BARS['joint_margin_rad']} rad, no limit constraint active",
                   "PASS" if margin > BARS["joint_margin_rad"] and mt["limit_active_steps"] == 0 else "FAIL",
                   f"min margin={margin:.3f} rad, limit-active steps={mt['limit_active_steps']}"))
    R = r["traj"]["radius"]
    ok = mt.get("swept_angle_at_0.9R_rad", 0) >= BARS["circle_sweep_rad"]
    C.append(Check("3j", "ACTUAL payload sweeps >= 2 pi about the lift point at >= 90 % R",
                   "PASS" if ok else "FAIL",
                   f"swept={mt.get('swept_angle_at_0.9R_rad', float('nan')):.3f} rad (2pi=6.283), "
                   f"mean r={mt.get('radius_mean_at_plateau_m', float('nan'))*1e3:.1f} mm (R={R*1e3:.0f}), "
                   f"fitted centre offset={mt.get('fitted_centre_offset_m', float('nan'))*1e3:.2f} mm",
                   {k: mt.get(k) for k in ("swept_angle_total_rad", "radius_max_m", "fitted_radius_m")}))
    Az = r["traj"]["A_z"]
    thr = BARS["z_excursion_frac"] * Az
    nb = r["traj"]["k"] * r["traj"]["n_circles"]
    ok = (mt.get("z_excursion_up_m", 0) >= thr and -mt.get("z_excursion_down_m", 0) >= thr
          and mt.get("z_bobs_per_circle_window") == nb)
    C.append(Check("3k", f"Z excursion >= 80 % A_z up and down, {nb} bobs per circle",
                   "PASS" if ok else "FAIL",
                   f"up={mt.get('z_excursion_up_m', float('nan'))*1e3:.1f} mm down={mt.get('z_excursion_down_m', float('nan'))*1e3:.1f} mm "
                   f"(A_z={Az*1e3:.0f}); bobs/dips in the {mt.get('circle_window_s')} s full-radius turn = "
                   f"{mt.get('z_bobs_per_circle_window')}/{mt.get('z_dips_per_circle_window')} "
                   f"(swept {mt.get('swept_angle_in_circle_window_rad', float('nan')):.3f} rad); whole circle phase incl. ramps: "
                   f"{mt.get('z_bobs_whole_circle_phase')} bobs"))
    sl = np.asarray(mt.get("slip_drift_max_mm", np.full(4, np.nan)))
    C.append(Check("3l", f"grasp slip: pinch drift in handle frame while carried < {BARS['slip_mm']} mm",
                   "PASS" if np.all(sl < BARS["slip_mm"]) else "FAIL",
                   f"max drift per arm={np.round(sl, 2).tolist()} mm",
                   {"offset_at_carry_start_mm": mt.get("pinch_handle_offset_start_carry_mm")}))
    C.append(Check("3m", "controller never writes qpos/qvel/qfrc_applied/xfrc_applied",
                   "PASS" if mt["controller_tampered_steps"] == 0 else "FAIL",
                   f"tampered steps={mt['controller_tampered_steps']}"))
    C.append(Check("3n", "pad-handle contact penetration while carried (report; soft-contact realism)", "INFO",
                   f"max penetration={mt['pad_penetration_max_carried_mm']:.3f} mm (handle thickness 30 mm)"))
    cl = r.get("clearance")
    if cl:
        w, c = cl["min_whole_run_m"], cl["min_carried_m"]
        C.append(Check("3o", "closest approach (report; near misses): arm-arm, gripper non-pad-payload, robot-stand, robot-floor",
                       "INFO",
                       f"carried: arm-arm {c['arm_arm']*1e3:.1f} mm, non-pad-payload {c['nonpad_payload']*1e3:.1f} mm, "
                       f"robot-stand {c['robot_stand']*1e3:.1f} mm, robot-floor {c['robot_floor']*1e3:.1f} mm, payload-stand "
                       f"{c['payload_stand']*1e3:.1f} mm; closest non-pad-payload pair {cl['argmin_pair_carried'].get('nonpad_payload')} "
                       f"(whole run {w['nonpad_payload']*1e3:.2f} mm at t={cl['argmin_t']['nonpad_payload']:.2f} s); robot-floor pair "
                       f"{cl['argmin_pair_carried'].get('robot_floor')}; distances capped at 50 mm (200 mm floor/stand)", cl))
    return C


def _drops(e: dict) -> str:
    return "/".join(f"{e.get(f'z_drop_{h}s_m', float('nan'))*1e3:.1f}" for h in (0.25, 0.5, 1.0, 2.0))


def checks_causality(res: dict) -> list[Check]:
    C = []
    z_rest = float(demo_trajectory().p_rest[2])
    # 4a/4a' are reported, not graded: the jaws close vertically across the handle, so
    # the opened lower fingers still support the plate from below (pad Fz ~ m g / 4 per
    # arm) and it settles onto them instead of falling.  4a'' is the graded release test.
    for key, cid, label in (("release_all", "4a", "stand on"),
                            ("release_all_nostand", "4a'", "stand off")):
        r = res.get(key)
        if not r:
            continue
        e = r["event_response"]
        d1 = e.get("z_drop_1.0s_m", float("nan"))
        C.append(Check(cid, f"open all grippers at t={EVENT_T}s, {label} (report: lower fingers still support)",
                       "INFO",
                       f"drop > {BARS['release_drop_m']} m in 1 s: {'yes' if d1 > BARS['release_drop_m'] else 'no (expected)'}; "
                       f"drop 0.25/0.5/1/2 s = {_drops(e)} mm; after 2 s: pad normal force per arm "
                       f"{np.round(e.get('pad_fn_end_per_arm_N', []), 2).tolist()} N (pad Fz {np.round(e.get('pad_fz_end_per_arm_N', []), 2).tolist()} N), "
                       f"jaws open (driver {np.round(e.get('driver_angle_end', []), 3).tolist()} rad), "
                       f"payload still following the path (pos err {e.get('pos_err_end_m', float('nan'))*1e3:.1f} mm)", e))
    r = res.get("release_retract")
    if r:
        e = r["event_response"]
        d1 = e.get("z_drop_1.0s_m", float("nan"))
        C.append(Check("4a''", f"open all at t={EVENT_T}s, then arms back off 8 cm radially at t={EVENT_T + 0.5}s (stand disabled): "
                       f"drop > {BARS['release_drop_m']} m in 1 s", "PASS" if d1 > BARS["release_drop_m"] else "FAIL",
                       f"drop 0.25/0.5/1/2 s after back-off = {_drops(e)} mm; payload-floor contact={e.get('pay_floor_after_event')}", e))
    r = res.get("zero_arm")
    if r:
        e = r["event_response"]
        zmin = e.get("z_min_1s", float("nan"))
        ok = e.get("stand_contact_after_event") and zmin <= z_rest + 0.005
        C.append(Check("4b", f"zero arm torques at t={ZERO_TORQUE_T}s (stand kept, only {DEMO['lift']*1e3:.0f} mm of fall available): "
                       "payload falls back onto the stand within 1 s", "PASS" if ok else "FAIL",
                       f"drop 0.25/0.5/1/2 s = {_drops(e)} mm, min z in 1 s = {zmin:.4f} m (rest {z_rest:.3f}), "
                       f"stand contact={e.get('stand_contact_after_event')}", e))
    r = res.get("zero_arm_nostand")
    if r:
        e = r["event_response"]
        d1 = e.get("z_drop_1.0s_m", float("nan"))
        C.append(Check("4b'", f"zero arm torques at t={ZERO_TORQUE_T}s (stand collision disabled): payload drops > {BARS['sag_drop_m']} m in 1 s",
                       "PASS" if d1 > BARS["sag_drop_m"] else "FAIL",
                       f"drop 0.25/0.5/1/2 s = {_drops(e)} mm; payload-floor contact={e.get('pay_floor_after_event')}", e))
    r1 = res.get("release_1")
    if r1:
        e = r1["event_response"]
        mt = r1["metrics"]
        C.append(Check("4c", f"open gripper 1 only at t={EVENT_T}s (report)", "INFO",
                       f"payload stays held by arms 2-4: max pos err in 2 s after {e.get('pos_err_max_2.0s_m', float('nan'))*1e3:.1f} mm, "
                       f"at end {e.get('pos_err_end_m', float('nan'))*1e3:.1f} mm; rot err max {mt['rot_err_max_all_rad']:.4f} rad; "
                       f"z drop {e.get('z_drop_2.0s_m', float('nan'))*1e3:.1f} mm; pad force at end {np.round(e.get('pad_fn_end_per_arm_N', []), 1).tolist()} N; "
                       f"unwanted contacts {({k: v for k, v in mt['unwanted_contact_steps'].items() if v}) or 'none'}; status={r1['status']}",
                       {"event": e, "metrics": _small(mt)}))
    return C


def _small(mt: dict) -> dict:
    keys = ("status", "t_end", "pos_err_max_carried_m", "pos_err_rms_carried_m", "rot_err_max_carried_rad",
            "stand_contact_samples_carried", "grasp_loss_samples", "pad_fn_min_carried_per_arm_N",
            "unwanted_contact_steps", "unwanted_first_t", "saturated_step_fraction", "torque_frac_max",
            "joint_speed_max_rad_s", "swept_angle_at_0.9R_rad", "z_excursion_up_m", "z_excursion_down_m",
            "z_bobs_per_circle_window", "slip_drift_max_mm", "gripper_nonpad_payload_fn_max_N", "limit_active_steps")
    return {k: mt.get(k) for k in keys}


def robust_verdict(r: dict) -> tuple[bool, list[str]]:
    mt = r["metrics"]
    fails = []
    if not mt["finite"]:
        fails.append(f"status {r['status']} {r['reason']}")
        return False, fails
    if mt["pos_err_max_carried_m"] >= BARS["pos_err_max_m"]:
        fails.append(f"pos err {mt['pos_err_max_carried_m']*1e3:.1f} mm")
    if mt["rot_err_max_carried_rad"] >= BARS["rot_err_max_rad"]:
        fails.append(f"rot err {mt['rot_err_max_carried_rad']:.3f} rad")
    if mt["grasp_loss_samples"]:
        fails.append(f"grasp loss samples {mt['grasp_loss_samples']}")
    if mt["stand_contact_samples_carried"]:
        fails.append("stand contact while carried")
    bad = {k: v for k, v in mt["unwanted_contact_steps"].items() if v}
    if bad:
        fails.append(f"unwanted contacts {bad}")
    if mt["saturated_step_fraction"] > 0:
        fails.append(f"saturation {mt['saturated_step_fraction']*100:.2f} %")
    if mt["joint_speed_max_rad_s"] >= UR5E_MAX_JOINT_SPEED:
        fails.append(f"joint speed {mt['joint_speed_max_rad_s']:.2f} rad/s")
    if mt.get("swept_angle_at_0.9R_rad", 0) < BARS["circle_sweep_rad"]:
        fails.append(f"swept {mt.get('swept_angle_at_0.9R_rad', 0):.2f} rad")
    return not fails, fails


def checks_robustness(res: dict) -> list[Check]:
    C = []
    labels = {
        "mass150_mismatch": ("6a", "payload mass +50 %, controller uses nominal 1.0 kg (model mismatch)", True),
        "mass150_known": ("6a'", "payload mass +50 %, controller knows 1.5 kg", True),
        "period8": ("6b", "circle period 8 s", True),
        "perturb": ("6c", "initial arm joints perturbed ±0.01 rad (seed 0)", True),
        "period5": ("6d", "stress (report): circle period 5 s", False),
        "period3": ("6d'", "stress (report): circle period 3 s", False),
        "period2": ("6d''", "stress (report): circle period 2 s", False),
        "dt05": ("6e", "numerics (report): timestep 0.5 ms instead of 1 ms", False),
        "no_ff": ("6f", "ablation (report): allocated feed-forward wrench disabled (pure impedance)", False),
    }
    for key, (cid, label, graded) in labels.items():
        r = res.get(key)
        if r is None:
            continue
        ok, fails = robust_verdict(r)
        mt = r["metrics"]
        status = ("PASS" if ok else "FAIL") if graded else "INFO"
        if not graded:
            label = label + (" -> clean" if ok else " -> NOT clean")
        C.append(Check(cid, label, status,
                       (f"pos max {mt['pos_err_max_carried_m']*1e3:.2f} mm, rot {mt['rot_err_max_carried_rad']:.4f} rad, "
                        f"min pad {np.round(np.asarray(mt['pad_fn_min_carried_per_arm_N']), 1).tolist()} N, "
                        f"tau {mt['torque_frac_max']:.2f}" if mt["finite"] else "") + ("" if ok else f" | FAIL: {'; '.join(fails)}"),
                       _small(mt)))
    return C


def checks_audit(main: dict) -> list[Check]:
    C = []
    ta = trajectory_audit(demo_trajectory())
    worst_jump = max(max(v) for v in ta["boundary_jumps_p_v_a"].values())
    ok = ta["fd_vel_err"] < 1e-6 and ta["fd_acc_err"] < 1e-5 and worst_jump < 1e-6
    C.append(Check("5f", "trajectory: analytic v, a == finite differences of p; p, v, a continuous at every phase join (C2 claim)",
                   "PASS" if ok else "FAIL",
                   f"|v - dp/dt|max={ta['fd_vel_err']:.1e}, |a - dv/dt|max={ta['fd_acc_err']:.1e}, worst join jump={worst_jump:.1e}; "
                   f"peak |v|={ta['peak_vel']:.3f} m/s, peak |a|={ta['peak_acc']:.3f} m/s^2 ({ta['peak_acc']/9.81*100:.1f} % g)", ta))
    au = main["audits"]
    if not au:
        return [Check("5", "frame/sign audit", "SKIP", "no audit instants")]
    gm = max(a["grasp_matrix_residual"] for a in au)
    wl = max(a["W_des_law_residual"] for a in au)
    jf = max(max(a["jacobian_fd_max_abs_err"]) for a in au)
    ts = [round(a["t"], 3) for a in au]
    C.append(Check("5a", "sum_i G_i w_i == W_des with G rebuilt from MuJoCo site/COM positions",
                   "PASS" if gm < BARS["grasp_matrix_tol"] else "FAIL", f"max residual={gm:.2e} N/N·m at t={ts}"))
    C.append(Check("5b", "W_des == m(a_d + Kp e + Kd de) - m g ; I(Kr e_R + Kw de_w) + w x Iw (recomputed)",
                   "PASS" if wl < BARS["wdes_law_tol"] else "FAIL", f"max residual={wl:.2e}"))
    mt = main["metrics"]
    mg = main["mass"] * 9.81
    ex_l, ex_a = mt["newton_euler_exact_lin_max_N"], mt["newton_euler_exact_ang_max_Nm"]
    ok = ex_l < BARS["ne_exact_rel"] * mg and ex_a < BARS["ne_exact_rel"] * mg
    C.append(Check("5c", "only gravity + contacts act on the payload: qfrc_smooth+qfrc_constraint == sum(mj_contactForce) + m g (every step)",
                   "PASS" if ok else "FAIL", f"max |res| lin={ex_l:.2e} N, ang={ex_a:.2e} N·m"))
    rel = mt["newton_euler_lin_rel"]
    ang = mt["newton_euler_ang_max_Nm"]
    ok = rel < BARS["newton_euler_rel"] and ang < BARS["newton_euler_rel"] * mg * 0.3
    C.append(Check("5c'", f"Newton-Euler m a == sum(contact forces) + m g, I dw + w x Iw == tau (every step, < {BARS['newton_euler_rel']:.0%} m g)",
                   "PASS" if ok else "FAIL",
                   f"max |lin res|={mt['newton_euler_lin_max_N']:.2e} N ({rel:.1e} m g), max |ang res|={ang:.2e} N·m; "
                   f"= Newton-solver residual (max {mt['solver_residual_lin_max_N']:.2e} N, niter median {mt['solver_niter_median']:.0f} max {mt['solver_niter_max']}); "
                   f"pads-only lin res while carried={mt['newton_euler_lin_pads_only_max_N_carried']:.2e} N"))
    C.append(Check("5d", "pinch-site Jacobian (mj_jacSite) == central finite differences of site pose",
                   "PASS" if jf < BARS["jacobian_fd_tol"] else "FAIL", f"max |J - J_fd|={jf:.2e} at t={ts}"))
    af = np.asarray(mt.get("alloc_fz_mean_carried_per_arm_N", np.full(4, np.nan)))
    rf = np.asarray(mt.get("realised_grip_fz_mean_carried_per_arm_N", np.full(4, np.nan)))
    C.append(Check("5e", "allocated vs realised vertical grasp force per gripper (mean while carried)", "INFO",
                   f"allocated={np.round(af, 2).tolist()} N, realised={np.round(rf, 2).tolist()} N (m g={main['mass']*9.81:.2f} N)",
                   {"audits": [{k: a[k] for k in ("t", "W_des", "f_alloc", "grasp_matrix_residual",
                                                  "W_des_law_residual", "jacobian_fd_max_abs_err", "load_scale")} for a in au],
                    "realised_mean_force_per_arm": mt.get("realised_grip_f_mean_carried_per_arm_N")}))
    return C


# ----------------------------------------------------------------------- main
def _worker(scn: Scenario, q0: np.ndarray) -> dict:
    r = simulate(scn, q0)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--skip-causality", action="store_true")
    ap.add_argument("--skip-stress", action="store_true", help="skip the INFO-only stress/numerics/ablation runs")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args()
    t_start = time.perf_counter()
    print(f"demo defaults parsed from run_demo.py: { {k: DEMO[k] for k in ('radius', 'period', 'az', 'k', 'lift', 'load_ramp')} }")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    checks: list[Check] = check_model(model)
    info = ModelInfo.build(model)
    q0 = nominal_qpos0()
    checks.append(check_initial_state(model, q0))

    audit_times = (0.8, 2.0, 5.0, 8.0, 12.0, 16.0)
    scenarios = [Scenario("main", full_log=False, audit_times=audit_times, frame_fps=30.0, clearance_every=0.05)]
    if not args.skip_causality:
        scenarios += [
            Scenario("release_all", events=((EVENT_T, "open_all"),), duration=EVENT_T + 2.0),
            Scenario("release_all_nostand", events=((EVENT_T, "open_all"), (EVENT_T, "no_stand")), duration=EVENT_T + 2.0),
            Scenario("release_retract", events=((EVENT_T, "open_all"), (EVENT_T, "no_stand"),
                                                (EVENT_T + 0.5, "retract:0.08")),
                     duration=EVENT_T + 2.0, response_t=EVENT_T + 0.5),
            Scenario("zero_arm", events=((ZERO_TORQUE_T, "zero_arm"),), duration=ZERO_TORQUE_T + 1.5),
            Scenario("zero_arm_nostand", events=((ZERO_TORQUE_T, "zero_arm"), (ZERO_TORQUE_T, "no_stand")),
                     duration=ZERO_TORQUE_T + 1.5),
            Scenario("release_1", events=((EVENT_T, "open_1"),)),
        ]
    if not args.skip_robustness:
        scenarios += [
            Scenario("mass150_mismatch", mass_scale=1.5, controller_knows_mass=False),
            Scenario("mass150_known", mass_scale=1.5, controller_knows_mass=True),
            Scenario("period8", traj={"period": 8.0}),
            Scenario("perturb", perturb=0.01, seed=0),
        ]
    if not args.skip_stress:
        scenarios += [
            Scenario("period5", traj={"period": 5.0}),
            Scenario("period3", traj={"period": 3.0}),
            Scenario("period2", traj={"period": 2.0}),
            Scenario("dt05", timestep=0.0005),
            Scenario("no_ff", feedforward=False),
        ]
    results: dict = {}
    det: dict = {}
    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=max(1, args.jobs), mp_context=ctx) as pool:
        det_futs = {}
        if not args.skip_determinism:
            for nm in ("validation_det_a", "validation_det_b"):
                det_futs[nm] = pool.submit(run_demo_subprocess, nm)
        futs = {pool.submit(_worker, s, q0): s.name for s in scenarios}
        for fu in cf.as_completed(futs):
            nm = futs[fu]
            try:
                results[nm] = fu.result()
            except Exception as exc:  # report, never hide
                results[nm] = {"name": nm, "status": "exception", "reason": repr(exc), "metrics": {"finite": False}}
            r = results[nm]
            print(f"  sim {nm:22s} {r['status']:10s} t_end={r.get('t_end', float('nan')):.2f} wall={r.get('wall_s', float('nan')):.1f}s", flush=True)
        for nm, fu in det_futs.items():
            det[nm] = fu.result()
            print(f"  demo subprocess {nm}: rc={det[nm]['returncode']} wall={det[nm]['wall_s']:.1f}s", flush=True)
    for r in results.values():
        r["jnt_lo"], r["jnt_hi"] = info.jnt_lo, info.jnt_hi

    main_r = results.get("main")
    # ---- check 2
    if args.skip_determinism:
        checks.append(Check("2", "clean-start determinism", "SKIP", "--skip-determinism"))
    else:
        rc_ok = all(d["returncode"] == 0 for d in det.values())
        runs = HERE / "outputs" / "runs"
        a, b = runs / "validation_det_a", runs / "validation_det_b"
        cmpab = compare_runs(a, b) if rc_ok else {"identical": False, "max_abs_diff": float("nan")}
        checks.append(Check("2a", "two fresh run_demo.py processes: log.npz + frames.npz identical",
                            "PASS" if rc_ok and cmpab["max_abs_diff"] <= BARS["determinism_tol"] else "FAIL",
                            f"bitwise identical={cmpab['identical']}, max |diff|={cmpab['max_abs_diff']:.3g}",
                            {"subprocesses": det, "compare": cmpab}))
        fid = {}
        if rc_ok and main_r and main_r["status"] == "completed":
            with np.load(a / "log.npz") as z:
                lp = np.array(z["p"])
            with np.load(a / "frames.npz") as z:
                fq, ft = np.array(z["qpos"]), np.array(z["t"])
            n_steps = int(round(main_r["duration"] / model.opt.timestep))
            idx = np.array(sorted(set(range(0, n_steps, 10)) | {n_steps - 1}))
            hp = main_r["frames_q"]
            fid["demo_log_samples"] = int(lp.shape[0])
            fid["final_qpos_max_abs_diff"] = float(np.max(np.abs(fq[-1] - main_r["final_qpos"])))
            fid["frames_max_abs_diff"] = float(np.max(np.abs(fq - hp))) if fq.shape == hp.shape else float("inf")
            fid["frame_count"] = [int(fq.shape[0]), int(hp.shape[0])]
            fid["final_time"] = [float(ft[-1]), main_r["t_end"]]
            fid["log_index_count_matches"] = bool(idx.size == lp.shape[0])
            hpath = main_r["p_path"]
            fid["payload_path_max_abs_diff"] = (float(np.max(np.abs(hpath[idx] - lp)))
                                                if idx.size == lp.shape[0] and idx[-1] < hpath.shape[0] else float("inf"))
            ok = (fid["final_qpos_max_abs_diff"] <= BARS["determinism_tol"]
                  and fid["frames_max_abs_diff"] <= BARS["determinism_tol"]
                  and fid["payload_path_max_abs_diff"] <= BARS["determinism_tol"])
        else:
            ok = False
        checks.append(Check("2b", "validator harness reproduces the demo (logged payload path, frame qpos, final qpos)",
                            "PASS" if ok else "FAIL",
                            f"max |dp| path={fid.get('payload_path_max_abs_diff', float('nan')):.3g}, max |dq| frames="
                            f"{fid.get('frames_max_abs_diff', float('nan')):.3g}, final={fid.get('final_qpos_max_abs_diff', float('nan')):.3g}",
                            fid))
    for nm, r in results.items():
        if r.get("status") == "exception":
            checks.append(Check({"main": "3"}.get(nm, "6" if nm.startswith(("mass", "period", "perturb", "dt", "no_")) else "4"),
                                f"scenario '{nm}' raised", "FAIL", r["reason"]))

    def section(fn, cid, label, *a):
        try:
            return fn(*a)
        except Exception as exc:  # a validator error is a FAIL, never a silent pass
            return [Check(cid, f"{label}: validator could not evaluate", "FAIL", repr(exc))]

    # ---- checks 3, 5
    if main_r and main_r.get("status") != "exception":
        checks += section(checks_full_run, "3", "full run", main_r)
        checks += section(checks_audit, "5", "frame/sign audit", main_r)
    # ---- check 4
    if args.skip_causality:
        checks.append(Check("4", "causality", "SKIP", "--skip-causality"))
    else:
        checks += section(checks_causality, "4", "causality",
                          {k: v for k, v in results.items() if v.get("status") != "exception"})
    # ---- check 6
    if args.skip_robustness:
        checks.append(Check("6", "robustness", "SKIP", "--skip-robustness"))
    else:
        checks += section(checks_robustness, "6", "robustness",
                          {k: v for k, v in results.items() if v.get("status") != "exception"})
    # ---- checks 7, 8
    checks += section(check_video, "7", "video", main_r)
    checks += section(check_self_contained, "8", "self-containment")

    order = {c: i for i, c in enumerate(["1", "2", "3", "4", "5", "6", "7", "8"])}
    checks.sort(key=lambda c: (order.get(c.id[0], 9), c.id))
    n_fail = sum(c.status == "FAIL" for c in checks)
    print()
    print(f"{'id':5s} {'status':6s} {'check':78s} value")
    print("-" * 150)
    for c in checks:
        print(f"{c.id:5s} {c.status:6s} {c.name[:78]:78s} {c.value}")
    print("-" * 150)
    counts = {s: sum(c.status == s for c in checks) for s in ("PASS", "FAIL", "INFO", "SKIP")}
    print(f"{counts}  wall {time.perf_counter() - t_start:.0f} s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_sha256": sha256(MODEL_XML),
        "demo_defaults": {k: DEMO[k] for k in ("radius", "period", "az", "k", "lift", "load_ramp")},
        "bars": BARS, "event_t": EVENT_T, "zero_torque_t": ZERO_TORQUE_T,
        "counts": counts,
        "checks": [c.__dict__ for c in checks],
        "scenario_metrics": {k: _small(v.get("metrics", {})) | {"event_response": v.get("event_response"),
                                                                "wall_s": v.get("wall_s")}
                             for k, v in results.items()},
        "command": " ".join(["python", "scripts/validate.py", *sys.argv[1:]]),
    }
    out.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    print(f"report: {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
