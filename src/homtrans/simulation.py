"""Closed-loop run of the four-arm transport: step, log, guard, summarise.

The payload is moved only by contact with the grippers.  Each step the
cooperative controller computes the 28 actuator commands from the current
MuJoCo state; nothing writes the payload state.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from homtrans.scene import MODEL_XML, N_ARMS, TORQUE_LIMITS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"


# -------------------------------------------------------- contact bookkeeping
@dataclass
class ContactMap:
    """Body ids needed to classify contacts."""

    payload: int
    stand_geom: int
    floor_geom: int
    pad_bodies: dict[int, int]  # body id -> arm index
    robot_bodies: dict[int, int]  # every body of arm i -> i

    @classmethod
    def build(cls, model: mujoco.MjModel) -> ContactMap:
        pad_bodies, robot_bodies = {}, {}
        for b in range(model.nbody):
            name = model.body(b).name
            for i in range(1, N_ARMS + 1):
                if name.startswith(f"ur{i}_"):
                    robot_bodies[b] = i
                    if name in (f"ur{i}_g_left_pad", f"ur{i}_g_right_pad"):
                        pad_bodies[b] = i
        return cls(
            payload=model.body("payload").id,
            stand_geom=model.geom("stand").id,
            floor_geom=model.geom("floor").id,
            pad_bodies=pad_bodies,
            robot_bodies=robot_bodies,
        )


def classify_contacts(model: mujoco.MjModel, data: mujoco.MjData, cmap: ContactMap) -> dict:
    """Pad-payload normal force per arm, stand contact, and unwanted contacts."""
    pad_force = np.zeros(N_ARMS)
    other_finger = np.zeros(N_ARMS)
    stand = 0.0
    unwanted: list[str] = []
    force = np.zeros(6)
    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = c.geom1, c.geom2
        b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
        mujoco.mj_contactForce(model, data, k, force)
        fn = abs(force[0])
        pair = {b1, b2}
        if cmap.payload in pair:
            other = b2 if b1 == cmap.payload else b1
            og = g2 if b1 == cmap.payload else g1
            if og == cmap.stand_geom:
                stand += fn
            elif other in cmap.pad_bodies:
                pad_force[cmap.pad_bodies[other] - 1] += fn
            elif other in cmap.robot_bodies:
                i = cmap.robot_bodies[other]
                name = model.body(other).name
                if "_g_" in name:
                    other_finger[i - 1] += fn
                else:
                    unwanted.append(f"payload-{name}")
            elif og != cmap.floor_geom:
                unwanted.append(f"payload-{model.body(other).name}")
            else:
                unwanted.append("payload-floor")
            continue
        arms = {cmap.robot_bodies.get(b1), cmap.robot_bodies.get(b2)} - {None}
        if len(arms) == 2:
            unwanted.append(f"{model.body(b1).name}-{model.body(b2).name}")
        elif len(arms) == 1 and (g1 in (cmap.floor_geom, cmap.stand_geom)
                                 or g2 in (cmap.floor_geom, cmap.stand_geom)):
            unwanted.append(f"robot-{'floor' if cmap.floor_geom in (g1, g2) else 'stand'}")
    return {
        "pad_force": pad_force,
        "other_finger_force": other_finger,
        "stand_force": stand,
        "unwanted": unwanted,
    }


# ------------------------------------------------------------------------ run
@dataclass
class RunConfig:
    name: str = "run"
    duration: float | None = None  # default: trajectory duration
    log_rate: float = 100.0
    record_fps: float = 30.0
    record_frames: bool = True
    max_payload_error: float = 0.10  # m, abort threshold (divergence guard)
    notes: dict = field(default_factory=dict)


def payload_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, ...]:
    """Payload position, rotation, world linear velocity, world angular velocity."""
    j = model.joint("payload_free")
    qa, va = model.jnt_qposadr[j.id], model.jnt_dofadr[j.id]
    p = data.qpos[qa : qa + 3].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[qa + 3 : qa + 7])
    R = R.reshape(3, 3)
    v = data.qvel[va : va + 3].copy()
    w = R @ data.qvel[va + 3 : va + 6]  # free-joint angular velocity is in the body frame
    return p, R, v, w


def _per_arm_norm(x) -> np.ndarray:
    if x is None:
        return np.full(N_ARMS, np.nan)
    x = np.asarray(x, dtype=float)
    return np.linalg.norm(x, axis=-1) if x.ndim == 2 else x


def rotation_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)))


def run(controller, trajectory, qpos0: np.ndarray, config: RunConfig,
        model: mujoco.MjModel | None = None, recorder=None) -> dict:
    """Run the closed loop and write ``outputs/runs/<name>/`` (log.npz, summary.json)."""
    model = model or mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    cmap = ContactMap.build(model)
    duration = config.duration if config.duration is not None else trajectory.duration
    dt = model.opt.timestep
    n_steps = int(round(duration / dt))
    log_every = max(1, int(round(1.0 / (config.log_rate * dt))))
    arm_act = np.array([[model.actuator(f"ur{i}_{a}").id for a in
                         ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
                        for i in range(1, N_ARMS + 1)])

    rows: dict[str, list] = {k: [] for k in (
        "t", "p", "p_des", "rot_err", "pos_err", "tau", "sat", "pad_force", "other_finger_force",
        "stand_force", "n_unwanted", "phase", "pinch_pos_err", "pinch_rot_err", "W_des", "f_alloc",
        "alloc_residual")}
    unwanted_seen: dict[str, float] = {}
    status, reason = "completed", ""
    wall0 = time.perf_counter()
    if recorder is not None:
        recorder.record(data.time, data)
    for step in range(n_steps):
        # split step: positions/velocities of this step are current when the
        # controller reads site poses, Jacobians and qfrc_bias
        mujoco.mj_step1(model, data)
        ctrl, diag = controller.compute(model, data, data.time)
        data.ctrl[:] = ctrl
        mujoco.mj_step2(model, data)
        if recorder is not None:
            recorder.record(data.time, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            status, reason = "diverged", f"non-finite state at t={data.time:.4f}"
            break
        if step % log_every == 0 or step == n_steps - 1:
            p, R, _, _ = payload_state(model, data)
            s = trajectory.sample(data.time)
            contacts = classify_contacts(model, data, cmap)
            for name in contacts["unwanted"]:
                unwanted_seen.setdefault(name, data.time)
            tau = data.actuator_force[arm_act]
            rows["t"].append(data.time)
            rows["p"].append(p)
            rows["p_des"].append(s.p)
            rows["pos_err"].append(float(np.linalg.norm(s.p - p)))
            rows["rot_err"].append(rotation_angle(s.R.T @ R))
            rows["tau"].append(tau)
            rows["sat"].append(np.abs(tau) >= 0.999 * TORQUE_LIMITS)
            rows["pad_force"].append(contacts["pad_force"])
            rows["other_finger_force"].append(contacts["other_finger_force"])
            rows["stand_force"].append(contacts["stand_force"])
            rows["n_unwanted"].append(len(contacts["unwanted"]))
            rows["phase"].append(getattr(s, "phase", ""))
            rows["pinch_pos_err"].append(_per_arm_norm(diag.get("pinch_pos_err")))
            rows["pinch_rot_err"].append(_per_arm_norm(diag.get("pinch_rot_err")))
            rows["W_des"].append(np.asarray(diag.get("W_des", np.full(6, np.nan))))
            rows["f_alloc"].append(np.asarray(diag.get("f", np.full(6 * N_ARMS, np.nan))).ravel())
            rows["alloc_residual"].append(float(diag.get("alloc_residual", np.nan)))
            if rows["pos_err"][-1] > config.max_payload_error:
                status = "diverged"
                reason = f"payload error {rows['pos_err'][-1]:.3f} m at t={data.time:.3f}"
                break
    wall = time.perf_counter() - wall0

    out = RUNS / config.name
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in rows.items() if k != "phase"}
    np.savez_compressed(out / "log.npz", phase=np.asarray(rows["phase"]), **arrays)
    if recorder is not None and config.record_frames:
        recorder.save(out / "frames.npz")
    summary = summarise(arrays, rows["phase"], status, reason, unwanted_seen, wall, data.time,
                        duration)
    summary["config"] = asdict(config)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def summarise(a: dict, phase: list, status: str, reason: str, unwanted: dict, wall: float,
              t_end: float, duration: float) -> dict:
    t = a["t"]
    phases = np.asarray(phase)
    moving = np.isin(phases, ["lift", "hold", "circle", "settle"]) if phases.size else t > 0
    carried = np.isin(phases, ["hold", "circle", "settle"]) if phases.size else t > 0
    def mx(x, mask=None):
        x = np.asarray(x)
        if mask is not None and x.shape[0] == mask.shape[0]:
            x = x[mask]
        return float(np.nanmax(x)) if x.size else float("nan")
    return {
        "status": status,
        "reason": reason,
        "t_end": float(t_end),
        "duration": float(duration),
        "wall_seconds": wall,
        "realtime_factor": float(t_end / wall) if wall > 0 else float("nan"),
        "max_pos_err_m": mx(a["pos_err"]),
        "rms_pos_err_m": float(np.sqrt(np.mean(np.square(a["pos_err"])))) if len(t) else float("nan"),
        "max_pos_err_while_carried_m": mx(a["pos_err"], carried),
        "max_rot_err_rad": mx(a["rot_err"]),
        "max_abs_torque_fraction": mx(np.abs(a["tau"]) / TORQUE_LIMITS) if len(t) else float("nan"),
        "saturated_fraction": float(np.mean(np.any(a["sat"], axis=(1, 2)))) if len(t) else float("nan"),
        "max_stand_force_while_carried_N": mx(a["stand_force"], carried),
        "min_pad_force_while_carried_N": float(np.nanmin(np.asarray(a["pad_force"])[carried]))
        if len(t) and carried.any() else float("nan"),
        "max_other_finger_force_N": mx(a["other_finger_force"]),
        "max_pinch_pos_err_m": mx(a["pinch_pos_err"], moving),
        "unwanted_contacts_first_seen": unwanted,
        "max_alloc_residual": mx(a["alloc_residual"]),
    }
