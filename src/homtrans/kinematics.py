"""Kinematics for the four-UR5e cooperative transport scene.

Grasp poses, damped-least-squares IK on the ``ur{i}_pinch`` sites, the initial
(pre-grasp, gripper open) configuration, a contact filter and a workspace check
along payload trajectories. Names and frames are those of :mod:`homtrans.scene`:

* arm i (1..4) joints ``ur{i}_shoulder_pan_joint .. ur{i}_wrist_3_joint``;
* grasp site ``ur{i}_pinch`` (+z approach, +-y finger closing axis);
* payload body ``payload`` (free joint ``payload_free``) carrying sites
  ``handle_{i}`` whose frame IS the desired ``ur{i}_pinch`` frame when grasped.

Poses are ``(p (3,), R (3, 3))`` in world coordinates; joint vectors are the 6
arm joints of one arm, in ``ARM_JOINTS`` order.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

N_ARMS = 4
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
PAYLOAD_BODY = "payload"
PAYLOAD_JOINT = "payload_free"

# 2F-85 pad bodies, e.g. ``ur1_g_right_pad`` (scene) or ``ur1_right_pad``.
_PAD_BODY_RE = re.compile(r"^ur\d+_(\w+_)?(left|right)_(silicone_)?pad$")


# -------------------------------------------------------------- Index helpers
def arm_joint_ids(model, i: int) -> np.ndarray:
    """Joint ids of arm i's 6 hinges, in ``ARM_JOINTS`` order."""
    return np.array([model.joint(f"ur{i}_{j}").id for j in ARM_JOINTS], dtype=int)


def arm_qpos_adr(model, i: int) -> np.ndarray:
    """qpos addresses of arm i's 6 hinges."""
    return model.jnt_qposadr[arm_joint_ids(model, i)].copy()


def arm_dof_adr(model, i: int) -> np.ndarray:
    """dof (qvel) addresses of arm i's 6 hinges."""
    return model.jnt_dofadr[arm_joint_ids(model, i)].copy()


def arm_limits(model, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) joint limits of arm i (inf where a joint is unlimited)."""
    jid = arm_joint_ids(model, i)
    lo = np.where(model.jnt_limited[jid] > 0, model.jnt_range[jid, 0], -np.inf)
    hi = np.where(model.jnt_limited[jid] > 0, model.jnt_range[jid, 1], np.inf)
    return lo, hi


def gripper_joint_ids(model, i: int) -> np.ndarray:
    """Ids of arm i's 2F-85 joints: every ``ur{i}_*`` joint that is not an arm joint."""
    arm = {f"ur{i}_{j}" for j in ARM_JOINTS}
    pre = f"ur{i}_"
    return np.array(
        [j for j in range(model.njnt)
         if model.joint(j).name.startswith(pre) and model.joint(j).name not in arm],
        dtype=int,
    )


def pinch_site_id(model, i: int) -> int:
    return model.site(f"ur{i}_pinch").id


def _payload_qpos_adr(model) -> int:
    return int(model.jnt_qposadr[model.joint(PAYLOAD_JOINT).id])


def _body_subtree(model, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


# ----------------------------------------------------------- Rotation helpers
def _quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def _mat_to_quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


def rotation_error(R_target, R_current) -> np.ndarray:
    """World-frame rotation vector w with exp([w]) R_current = R_target (log map)."""
    qt, qc = _mat_to_quat(R_target), _mat_to_quat(R_current)
    w_local = np.zeros(3)
    mujoco.mju_subQuat(w_local, qt, qc)  # qc * exp(w_local) = qt, w_local in current frame
    return np.asarray(R_current) @ w_local


# -------------------------------------------------------------- Grasp targets
def handle_local_pose(model, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Pose of site ``handle_{i}`` in the payload body frame.

    Composes the static body chain from the site's body up to ``payload``; there
    must be no joint in between (the handle is rigid on the plate).
    """
    sid = model.site(f"handle_{i}").id
    pay = model.body(PAYLOAD_BODY).id
    p = model.site_pos[sid].copy()
    R = _quat_to_mat(model.site_quat[sid])
    b = int(model.site_bodyid[sid])
    while b != pay:
        if model.body_jntnum[b] != 0 or b == 0:
            raise ValueError(f"handle_{i} is not rigidly attached to body '{PAYLOAD_BODY}'")
        Rb = _quat_to_mat(model.body_quat[b])
        p = model.body_pos[b] + Rb @ p
        R = Rb @ R
        b = int(model.body_parentid[b])
    return p, R


def pinch_targets_for_payload_pose(model, p_payload, R_payload) -> list[tuple[np.ndarray, np.ndarray]]:
    """Desired ``ur{i}_pinch`` poses (i = 1..4) for a given payload body pose."""
    p_payload = np.asarray(p_payload, dtype=float)
    R_payload = np.asarray(R_payload, dtype=float)
    out = []
    for i in range(1, N_ARMS + 1):
        ph, Rh = handle_local_pose(model, i)
        out.append((p_payload + R_payload @ ph, R_payload @ Rh))
    return out


def desired_pinch_pose(model, data, i: int, p_payload=None, R_payload=None):
    """Desired pinch pose of arm i = payload pose o handle_{i} site pose.

    Without an explicit payload pose, reads ``data.site_xpos/site_xmat`` of
    ``handle_{i}``: ``data`` must hold current kinematics (after mj_forward or
    mj_kinematics).
    """
    if p_payload is None:
        sid = model.site(f"handle_{i}").id
        return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()
    if R_payload is None:
        raise ValueError("R_payload is required when p_payload is given")
    ph, Rh = handle_local_pose(model, i)
    R_payload = np.asarray(R_payload, dtype=float)
    return np.asarray(p_payload, dtype=float) + R_payload @ ph, R_payload @ Rh


def set_payload_pose(model, data, p, R) -> None:
    """Write a payload pose into ``data.qpos`` (free joint)."""
    a = _payload_qpos_adr(model)
    data.qpos[a:a + 3] = p
    data.qpos[a + 3:a + 7] = _mat_to_quat(R)


def payload_pose_from_qpos(model, qpos) -> tuple[np.ndarray, np.ndarray]:
    a = _payload_qpos_adr(model)
    return np.array(qpos[a:a + 3], dtype=float), _quat_to_mat(qpos[a + 3:a + 7])


# ------------------------------------------------------------ Jacobian and IK
def pinch_jacobian(model, data, i: int) -> np.ndarray:
    """6x6 world-frame Jacobian [linear; angular] of ``ur{i}_pinch`` w.r.t. arm i's joints.

    Requires current ``mj_kinematics`` + ``mj_comPos`` (both done by mj_forward).
    """
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, pinch_site_id(model, i))
    dof = arm_dof_adr(model, i)
    return np.vstack([jp[:, dof], jr[:, dof]])


def sigma_min(J) -> float:
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def joint_margin(model, i: int, q) -> float:
    """Smallest distance (rad) of arm i's joints from their limits."""
    lo, hi = arm_limits(model, i)
    return float(np.min(np.minimum(np.asarray(q) - lo, hi - np.asarray(q))))


def _pose_error(model, data, sid, p_t, R_t):
    e_p = p_t - data.site_xpos[sid]
    e_r = rotation_error(R_t, data.site_xmat[sid].reshape(3, 3))
    return e_p, e_r


def solve_arm_ik(model, data, i: int, p_target, R_target, q_seed, *,
                 max_iters: int = 200, tol_pos: float = 1e-6, tol_rot: float = 1e-5,
                 rot_weight: float = 0.3, damping: float = 1e-3, max_step: float = 0.3):
    """Damped-least-squares (Levenberg-Marquardt) IK placing ``ur{i}_pinch`` at a pose.

    Only arm i's 6 qpos entries are changed: on return ``data.qpos`` holds the
    best iterate for arm i and ``data`` holds kinematics (mj_kinematics +
    mj_comPos) for that qpos. Joint limits are enforced by clipping and by
    freezing joints pinned at a limit. ``rot_weight`` (m/rad) scales the
    orientation rows in the least-squares metric only; convergence is judged on
    the unweighted errors ``|e_p| < tol_pos`` (m) and ``|e_r| < tol_rot`` (rad).

    Returns ``(q (6,), info)`` with info keys: converged, iters, pos_err,
    rot_err, sigma_min, joint_margin, within_limits.
    """
    p_t = np.asarray(p_target, dtype=float)
    R_t = np.asarray(R_target, dtype=float)
    adr = arm_qpos_adr(model, i)
    lo, hi = arm_limits(model, i)
    sid = pinch_site_id(model, i)
    W = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])

    def evaluate(q):
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        e_p, e_r = _pose_error(model, data, sid, p_t, R_t)
        return e_p, e_r, float(np.linalg.norm(np.concatenate([e_p, e_r]) * W))

    q = np.clip(np.asarray(q_seed, dtype=float).copy(), lo, hi)
    e_p, e_r, cost = evaluate(q)
    lam = damping
    it = 0
    converged = False
    for it in range(1, max_iters + 1):
        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
            converged = True
            it -= 1
            break
        J = pinch_jacobian(model, data, i) * W[:, None]
        e = np.concatenate([e_p, e_r]) * W
        accepted = False
        for _ in range(12):
            free = np.ones(6, dtype=bool)
            for _pass in range(3):  # freeze joints driven past a limit, re-solve
                Jf = J[:, free]
                dq = np.zeros(6)
                dq[free] = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam**2 * np.eye(6), e)
                n = np.linalg.norm(dq)
                if n > max_step:
                    dq *= max_step / n
                q_try = q + dq
                viol = ((q_try < lo) & (dq < 0)) | ((q_try > hi) & (dq > 0))
                if not viol.any():
                    break
                free &= ~viol
            q_try = np.clip(q + dq, lo, hi)
            e_p2, e_r2, cost2 = evaluate(q_try)
            if cost2 < cost:
                q, e_p, e_r, cost = q_try, e_p2, e_r2, cost2
                lam = max(lam * 0.3, 1e-6)
                accepted = True
                break
            lam *= 4.0
        if not accepted:  # stuck (local minimum or limit): restore best and stop
            e_p, e_r, cost = evaluate(q)
            break
    else:
        converged = np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot
    data.qpos[adr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    info = {
        "converged": bool(converged),
        "iters": int(it),
        "pos_err": float(np.linalg.norm(e_p)),
        "rot_err": float(np.linalg.norm(e_r)),
        "sigma_min": sigma_min(pinch_jacobian(model, data, i)),
        "joint_margin": joint_margin(model, i, q),
        "within_limits": bool(np.all(q >= lo) and np.all(q <= hi)),
    }
    return q.copy(), info


# ----------------------------------------------------- Contacts and clearance
def _geom_class(model):
    """Per-geom tags used by the contact filter."""
    pay_sub = _body_subtree(model, model.body(PAYLOAD_BODY).id)
    # the stand is either a body named "stand" or static geom(s) named "stand*"
    stand_sub = _body_subtree(model, model.body("stand").id) if _has_body(model, "stand") else set()
    arm_of_body = {}
    for b in range(model.nbody):
        mm = re.match(r"^ur(\d+)_", model.body(b).name)
        if mm:
            arm_of_body[b] = int(mm.group(1))
    base_bodies = {model.body(f"ur{i}_base").id for i in range(1, N_ARMS + 1)
                   if _has_body(model, f"ur{i}_base")}
    pay_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in pay_sub]
    handle_geoms = {g for g in pay_geoms if "handle" in model.geom(g).name}
    return {
        "payload": {g for g in pay_geoms},
        "handle": handle_geoms if handle_geoms else set(pay_geoms),
        "stand": {g for g in range(model.ngeom)
                  if model.geom_bodyid[g] in stand_sub or model.geom(g).name.startswith("stand")},
        "world": {g for g in range(model.ngeom)
                  if model.geom_bodyid[g] == 0 and not model.geom(g).name.startswith("stand")},
        "pad": {g for g in range(model.ngeom)
                if _PAD_BODY_RE.match(model.body(model.geom_bodyid[g]).name)},
        "base": {g for g in range(model.ngeom) if model.geom_bodyid[g] in base_bodies},
        "arm_of_geom": {g: arm_of_body[model.geom_bodyid[g]] for g in range(model.ngeom)
                        if model.geom_bodyid[g] in arm_of_body},
    }


def _has_body(model, name) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def _allowed(tags, g1, g2, allow_payload_stand=True) -> bool:
    for a, b in ((g1, g2), (g2, g1)):
        if a in tags["pad"] and b in tags["handle"]:
            return True  # gripper pad <-> payload handle
        if a in tags["world"] and b in tags["base"]:
            return True  # floor (static world) <-> robot base
        if allow_payload_stand and a in tags["payload"] and b in tags["stand"]:
            return True  # payload resting on the stand (environment, not a robot contact)
    return False


def unwanted_contacts(model, data, *, forward: bool = True, max_dist: float = 0.0,
                      allow_payload_stand: bool = True, _tags=None) -> list[dict]:
    """Contacts other than pad<->handle, floor<->robot base and payload<->stand.

    Runs ``mj_forward`` first unless ``forward=False``. Contacts whose distance
    exceeds ``max_dist`` (possible when geoms carry a margin) are ignored.
    ``allow_payload_stand=False`` also reports payload<->stand contacts (use it
    once the payload should have left the stand). Returns dicts
    ``{geom1, geom2, body1, body2, dist}`` (names, metres).
    """
    if forward:
        mujoco.mj_forward(model, data)
    tags = _tags if _tags is not None else _geom_class(model)
    out = []
    for k in range(data.ncon):
        c = data.contact[k]
        if c.dist > max_dist:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        if _allowed(tags, g1, g2, allow_payload_stand):
            continue
        out.append({
            "geom1": model.geom(g1).name or f"#{g1}",
            "geom2": model.geom(g2).name or f"#{g2}",
            "body1": model.body(model.geom_bodyid[g1]).name,
            "body2": model.body(model.geom_bodyid[g2]).name,
            "dist": float(c.dist),
        })
    return out


def _collides(model, g) -> bool:
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


def arm_clearance(model, data, i: int, *, distmax: float = 0.5, _tags=None) -> dict:
    """Minimum signed distances (m, capped at ``distmax``) from arm i to the scene.

    Keys (each with a ``*_pair`` naming the closest geoms):

    * ``floor``: forearm and everything distal (incl. gripper) vs the floor
      (base/shoulder/upper-arm heights are fixed by the mounting);
    * ``stand``: all arm-i geoms vs the stand;
    * ``payload``: UR5e link geoms vs the payload;
    * ``gripper_payload``: non-pad 2F-85 geoms vs the payload;
    * ``other_arms``: all arm-i geoms vs the other arms;
    * ``pad_handle``: pads vs handle geoms (with open jaws: the jaw clearance);
    * ``min``: min of floor, stand, payload, other_arms (the arm-safety number).

    Requires current kinematics (mj_forward / mj_kinematics).
    """
    tags = _tags if _tags is not None else _geom_class(model)
    mine = [g for g, a in tags["arm_of_geom"].items() if a == i and _collides(model, g)]
    grip_bodies = _gripper_bodies(model, i)
    distal = _body_subtree(model, model.body(f"ur{i}_forearm_link").id)
    link = [g for g in mine if model.geom_bodyid[g] not in grip_bodies]
    pads = [g for g in mine if g in tags["pad"]]
    grip_nonpad = [g for g in mine if model.geom_bodyid[g] in grip_bodies and g not in tags["pad"]]
    coll = [g for g in range(model.ngeom) if _collides(model, g)]
    other_arms = [g for g in coll if tags["arm_of_geom"].get(g, i) != i]
    payload = [g for g in coll if g in tags["payload"]]
    fromto = np.zeros(6)

    def dmin(A, B):
        best, pair = distmax, None
        for a in A:
            for b in B:
                dd = mujoco.mj_geomDistance(model, data, a, b, distmax, fromto)
                if dd < best:
                    best, pair = dd, (model.geom(a).name or f"#{a}", model.geom(b).name or f"#{b}")
        return float(best), pair

    out = {}
    for key, A, B in (
        ("floor", [g for g in mine if model.geom_bodyid[g] in distal], sorted(tags["world"])),
        ("stand", mine, sorted(tags["stand"])),
        ("payload", link, payload),
        ("gripper_payload", grip_nonpad, payload),
        ("other_arms", mine, other_arms),
        ("pad_handle", pads, sorted(tags["handle"])),
    ):
        out[key], out[key + "_pair"] = dmin([g for g in A if _collides(model, g)],
                                            [g for g in B if _collides(model, g)])
    out["min"] = min(out["floor"], out["stand"], out["payload"], out["other_arms"])
    return out


def _gripper_bodies(model, i: int) -> set[int]:
    """Bodies of arm i's 2F-85: the subtree below ``ur{i}_wrist_3_link`` minus that link."""
    w3 = model.body(f"ur{i}_wrist_3_link").id
    return _body_subtree(model, w3) - {w3}


# ------------------------------------------------------ Initial configuration
def _seeds() -> list[np.ndarray]:
    """Joint-space seeds covering the UR5e IK branches (shoulder, elbow, wrist)."""
    seeds = []
    for pan in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        for lift, elbow in ((-1.2, 1.6), (-2.0, -1.6), (-0.6, 1.0), (-2.6, -1.0)):
            for w1 in (-1.57, 0.0, -3.14):
                for w2 in (1.57, -1.57):
                    seeds.append(np.array([pan, lift, elbow, w1, w2, 0.0]))
    return seeds


def _wrap_pi(q, lo, hi) -> np.ndarray:
    """Map each joint to its 2*pi-equivalent closest to 0 that stays within limits."""
    q = np.array(q, dtype=float)
    for k in range(q.size):
        c = (q[k] + np.pi) % (2 * np.pi) - np.pi
        if lo[k] <= c <= hi[k]:
            q[k] = c
    return q


def _elbow_up(model, data, i) -> bool:
    """True when the elbow lies above the shoulder->wrist line (vertical-plane sense)."""
    s = data.xpos[model.body(f"ur{i}_upper_arm_link").id]
    e = data.xpos[model.body(f"ur{i}_forearm_link").id]
    w = data.xpos[model.body(f"ur{i}_wrist_1_link").id]
    u = (w - s) / max(np.linalg.norm(w - s), 1e-9)
    perp = (e - s) - np.dot(e - s, u) * u
    return bool(perp[2] > 0)


def _facing_error(model, i, q_pan, p_target) -> float:
    """Angle (rad) between arm i's horizontal reach direction at pan angle
    ``q_pan`` and the horizontal direction from its shoulder to ``p_target``.

    The reach direction at pan = 0 is taken from the all-zero pose (shoulder ->
    wrist_1), then rotated by ``q_pan`` about the pan axis; this picks the
    shoulder solution that faces the target over its mirror image."""
    d0 = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d0)  # arm joints at qpos0 = 0
    sh = d0.xpos[model.body(f"ur{i}_shoulder_link").id].copy()
    f0 = d0.xpos[model.body(f"ur{i}_wrist_1_link").id] - sh
    axis = d0.xaxis[model.joint(f"ur{i}_shoulder_pan_joint").id].copy()
    R = np.zeros(9)
    qrot = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qrot, axis, float(q_pan))
    mujoco.mju_quat2Mat(R, qrot)
    f = R.reshape(3, 3) @ f0
    g = np.asarray(p_target, dtype=float) - sh
    f[2] = g[2] = 0.0
    c = np.dot(f, g) / max(np.linalg.norm(f) * np.linalg.norm(g), 1e-12)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _arm_contacts(contacts, i):
    pre = f"ur{i}_"
    return [c for c in contacts if c["body1"].startswith(pre) or c["body2"].startswith(pre)]


def _track(model, data, i, q0, targets):
    """Follow arm i along pinch ``targets`` from ``q0`` (warm-started IK).

    Returns (all converged without branch change, min sigma_min, min joint
    margin). Leaves arm i's qpos at ``q0``."""
    adr = arm_qpos_adr(model, i)
    q, ok = np.array(q0, dtype=float), True
    smin, marg, br0 = np.inf, joint_margin(model, i, q0), _branch(q0)
    for p_t, R_t in targets:
        q, info = solve_arm_ik(model, data, i, p_t, R_t, q)
        ok &= info["converged"] and _branch(q) == br0
        smin, marg = min(smin, info["sigma_min"]), min(marg, info["joint_margin"])
    data.qpos[adr] = q0
    return bool(ok), float(smin), float(marg)


def arm_candidates(model, data, i, p_t, R_t, *, extra_seeds=(), probe_targets=(),
                   tags=None) -> list[dict]:
    """All distinct converged IK solutions for arm i at a target, best first.

    Each entry: q, sigma_min, joint_margin, elbow_up, wrist (|sin q5|),
    facing_err, contacts (unwanted, involving arm i), clearance, probe, score.
    ``data`` must already hold the rest of the scene (payload pose, other arms,
    grippers). Ranking is lexicographic: collision-free, elbow up, shoulder
    facing the target (not the mirror solution reaching back over the base),
    then a smooth score = sigma_min x saturating factors for joint margin
    (0.5 rad), clearance (5 cm), gripper/payload clearance (5 mm) and |sin q5|
    (0.5). If ``probe_targets`` (pinch poses of arm i along an intended motion)
    are given, sigma_min and joint margin are the minima along that motion
    tracked from the candidate, so the branch is chosen for the whole motion
    (the branch cannot change during a continuous motion).
    """
    tags = tags if tags is not None else _geom_class(model)
    lo, hi = arm_limits(model, i)
    adr = arm_qpos_adr(model, i)
    sols = []
    for seed in list(extra_seeds) + _seeds():
        q, info = solve_arm_ik(model, data, i, p_t, R_t, seed)
        if not (info["converged"] and info["within_limits"]):
            continue
        q = _wrap_pi(q, lo, hi)
        if any(np.max(np.abs(q - s)) < 1e-3 for s in sols):
            continue
        sols.append(q)
    cands = []
    for q in sols:
        data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        J = pinch_jacobian(model, data, i)
        e_p, e_r = _pose_error(model, data, pinch_site_id(model, i), p_t, R_t)
        con = _arm_contacts(unwanted_contacts(model, data, forward=False, _tags=tags), i)
        clr = arm_clearance(model, data, i, _tags=tags)
        c = {
            "q": q,
            "pos_err": float(np.linalg.norm(e_p)),
            "rot_err": float(np.linalg.norm(e_r)),
            "sigma_min": sigma_min(J),
            "joint_margin": joint_margin(model, i, q),
            "elbow_up": _elbow_up(model, data, i),
            "wrist": float(abs(np.sin(q[4]))),
            "facing_err": _facing_error(model, i, q[0], p_t),
            "contacts": con,
            "clearance": clr,
            "probe": None,
        }
        lex = (int(not con), int(c["elbow_up"]), int(c["facing_err"] < np.pi / 2))
        smin, marg = c["sigma_min"], c["joint_margin"]
        if len(probe_targets) and all(lex):
            ok, p_smin, p_marg = _track(model, data, i, q, probe_targets)
            c["probe"] = {"ok": ok, "sigma_min": p_smin, "joint_margin": p_marg}
            smin, marg = (min(smin, p_smin), min(marg, p_marg)) if ok else (0.0, marg)
        smooth = (smin
                  * min(1.0, marg / 0.5)
                  * min(1.0, max(clr["min"], 0.0) / 0.05)
                  * min(1.0, max(clr["gripper_payload"], 0.0) / 0.005)
                  * min(1.0, c["wrist"] / 0.5))
        c["score"] = lex + (smooth,)
        cands.append(c)
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def default_probe_poses(p_rest, R_rest, lift: float = 0.10, radius: float = 0.10,
                        dz: float = 0.03) -> list[tuple[np.ndarray, np.ndarray]]:
    """Payload poses of the nominal demo motion used to choose the IK branch:
    vertical lift by ``lift`` then an XY circle of ``radius`` with z +- ``dz``."""
    p_rest = np.asarray(p_rest, dtype=float)
    ups = [(p_rest + np.array([0.0, 0.0, z]), np.array(R_rest, dtype=float))
           for z in np.linspace(0.0, lift, max(2, int(np.ceil(lift / 0.02)) + 1))]
    return ups + circle_poses(p_rest + np.array([0.0, 0.0, lift]), R_rest, radius, dz=dz, n=36,
                              step=0.02)


def initial_configuration(model, *, probe="default", return_details: bool = False):
    """qpos with the payload at its model-default pose on the stand, grippers open
    (all 2F-85 joints at 0, the Menagerie open pose) and each ``ur{i}_pinch``
    exactly at ``handle_{i}``.

    Each arm is searched over many seeds (arm 1's solution is also used as a
    seed for arms 2..4: the scene is 90-degree symmetric) and ranked by
    ``arm_candidates``. ``probe`` is the list of payload poses ``(p, R)`` along
    which the branch is judged; ``"default"`` = ``default_probe_poses`` (lift
    0.10 m, circle R 0.10 m, z +- 0.03 m); ``None`` judges the rest pose only.
    With ``return_details`` returns ``(qpos, details)``: per-arm chosen solution,
    metrics and ranked candidates, plus the final unwanted-contact list.
    Raises RuntimeError if an arm has no IK solution or if the assembled
    configuration has unwanted contacts.
    """
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    for i in range(1, N_ARMS + 1):
        qpos[model.jnt_qposadr[gripper_joint_ids(model, i)]] = 0.0
    # park all arms upright (far from the plate) while each arm is searched
    park = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    for i in range(1, N_ARMS + 1):
        qpos[arm_qpos_adr(model, i)] = park
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    targets = [desired_pinch_pose(model, data, i) for i in range(1, N_ARMS + 1)]
    if isinstance(probe, str) and probe == "default":
        probe = default_probe_poses(*payload_pose_from_qpos(model, qpos))
    probe_t = [pinch_targets_for_payload_pose(model, p, R) for p, R in (probe or [])]
    tags = _geom_class(model)
    details = {"arms": {}}
    q_ref = None
    for i in range(1, N_ARMS + 1):
        p_t, R_t = targets[i - 1]
        extra = [q_ref] if q_ref is not None else []
        cands = arm_candidates(model, data, i, p_t, R_t, extra_seeds=extra,
                               probe_targets=[t[i - 1] for t in probe_t], tags=tags)
        if not cands:
            raise RuntimeError(f"initial_configuration: no IK solution for arm {i}")
        best = cands[0]
        data.qpos[arm_qpos_adr(model, i)] = best["q"]
        if q_ref is None:
            q_ref = best["q"].copy()
        details["arms"][i] = {
            "q": best["q"].tolist(),
            "n_candidates": len(cands),
            "sigma_min": best["sigma_min"],
            "joint_margin": best["joint_margin"],
            "elbow_up": best["elbow_up"],
            "wrist_abs_sin_q5": best["wrist"],
            "facing_err": best["facing_err"],
            "clearance": best["clearance"],
            "probe": best["probe"],
            "pos_err": best["pos_err"],
            "rot_err": best["rot_err"],
            "contacts": best["contacts"],
            "candidates": [
                {"q": np.round(c["q"], 4).tolist(), "sigma_min": round(c["sigma_min"], 4),
                 "joint_margin": round(c["joint_margin"], 3), "elbow_up": c["elbow_up"],
                 "wrist": round(c["wrist"], 3), "facing_err": round(c["facing_err"], 3),
                 "n_contacts": len(c["contacts"]),
                 "clearance_min": round(c["clearance"]["min"], 4),
                 "gripper_payload_clearance": round(c["clearance"]["gripper_payload"], 4),
                 "probe": c["probe"], "score": [float(x) for x in c["score"]]}
                for c in cands
            ],
        }
    qpos = data.qpos.copy()
    details["unwanted_contacts"] = unwanted_contacts(model, data, _tags=tags)
    if details["unwanted_contacts"]:
        raise RuntimeError("initial_configuration: best IK solutions still have unwanted contacts: "
                           f"{details['unwanted_contacts']}")
    return (qpos, details) if return_details else qpos


# --------------------------------- Workspace check along payload trajectories
def circle_poses(p_center, R, radius: float, dz: float = 0.0, n: int = 72,
                 z_cycles: int = 2, step: float = 0.01) -> list[tuple[np.ndarray, np.ndarray]]:
    """Payload poses on an XY circle of ``radius`` around ``p_center`` (constant
    orientation R), z = z_c + dz*sin(z_cycles*phi), with straight segments
    (``step`` spacing) from the centre to the circle start (phi = 0) and back."""
    p_center = np.asarray(p_center, dtype=float)
    R = np.array(R, dtype=float)
    n_in = max(2, int(np.ceil(radius / step)))
    radial = [p_center + np.array([s * radius, 0.0, 0.0])
              for s in np.linspace(0.0, 1.0, n_in, endpoint=False)]
    ring = [p_center + np.array([radius * np.cos(a), radius * np.sin(a), dz * np.sin(z_cycles * a)])
            for a in np.linspace(0.0, 2 * np.pi, n + 1)]
    return [(p, R) for p in radial + ring + radial[::-1]]


def _branch(q) -> tuple[int, int]:
    """(sign of elbow, sign of sin(wrist_2)) -- flips of either change IK branch."""
    return int(np.sign(q[2])), int(np.sign(np.sin(q[4])))


def workspace_report(model, trajectory_poses, *, q_start=None, check_collisions: bool = True,
                     jump_tol: float = 0.25) -> dict:
    """Solve IK for all four arms along a list of payload poses ``(p, R)``.

    Warm-starts each pose from the previous solution (first pose from
    ``q_start`` or ``initial_configuration``). Grippers stay open (the
    gripper-payload relative geometry is fixed along the path, so only
    arm/environment contacts change). Reports min sigma_min, min joint margin,
    max joint step between successive poses, unwanted contacts, minimum
    clearances (see ``arm_clearance``) and branch flips (elbow or wrist-2 sign
    change, or any joint step > ``jump_tol``). Payload<->stand contact counts
    as a collision once the payload is off its rest pose;
    ``min_payload_stand_xy_motion`` is the payload-stand gap while displaced
    laterally. ``(k, i)`` locations are (pose index, arm).
    """
    data = mujoco.MjData(model)
    qpos = initial_configuration(model) if q_start is None else np.array(q_start, dtype=float)
    data.qpos[:] = qpos
    tags = _geom_class(model)
    adr = [arm_qpos_adr(model, i) for i in range(1, N_ARMS + 1)]
    q_prev = [data.qpos[a].copy() for a in adr]
    br0 = [_branch(q) for q in q_prev]
    rep = {
        "n_poses": len(trajectory_poses), "all_converged": True,
        "min_sigma_min": np.inf, "min_sigma_min_at": None,
        "min_joint_margin": np.inf, "min_joint_margin_at": None,
        "max_joint_step": 0.0, "max_joint_step_at": None,
        "max_pos_err": 0.0, "max_rot_err": 0.0,
        "min_clearance": {}, "min_clearance_at": {},
        "min_payload_stand_xy_motion": np.inf, "min_payload_stand_xy_motion_at": None,
        "collisions": [], "branch_flips": [], "failures": [],
        "q_range": None,
    }
    qs = []
    p_rest = payload_pose_from_qpos(model, qpos)[0]
    pay_g = [g for g in sorted(tags["payload"]) if _collides(model, g)]
    stand_g = [g for g in sorted(tags["stand"]) if _collides(model, g)]
    fromto = np.zeros(6)
    for k, (p, R) in enumerate(trajectory_poses):
        set_payload_pose(model, data, p, R)
        mujoco.mj_kinematics(model, data)
        targets = pinch_targets_for_payload_pose(model, p, R)
        row = []
        for i in range(1, N_ARMS + 1):
            q, info = solve_arm_ik(model, data, i, *targets[i - 1], q_prev[i - 1])
            if not info["converged"]:
                rep["all_converged"] = False
                rep["failures"].append({"pose": k, "arm": i, "pos_err": info["pos_err"],
                                        "rot_err": info["rot_err"]})
            rep["max_pos_err"] = max(rep["max_pos_err"], info["pos_err"])
            rep["max_rot_err"] = max(rep["max_rot_err"], info["rot_err"])
            if info["sigma_min"] < rep["min_sigma_min"]:
                rep["min_sigma_min"], rep["min_sigma_min_at"] = info["sigma_min"], (k, i)
            if info["joint_margin"] < rep["min_joint_margin"]:
                rep["min_joint_margin"], rep["min_joint_margin_at"] = info["joint_margin"], (k, i)
            step = float(np.max(np.abs(q - q_prev[i - 1])))
            if step > rep["max_joint_step"]:
                rep["max_joint_step"], rep["max_joint_step_at"] = step, (k, i)
            if _branch(q) != br0[i - 1] or step > jump_tol:
                rep["branch_flips"].append({"pose": k, "arm": i, "step": step,
                                            "branch": _branch(q), "branch0": br0[i - 1]})
            q_prev[i - 1] = q
            row.append(q)
        qs.append(np.concatenate(row))
        if check_collisions:
            mujoco.mj_forward(model, data)
            moved = np.linalg.norm(np.asarray(p) - p_rest) > 1e-4  # off the stand: contact is unwanted
            con = unwanted_contacts(model, data, forward=False, allow_payload_stand=not moved,
                                    _tags=tags)
            if con:
                rep["collisions"].append({"pose": k, "contacts": con})
            if np.linalg.norm(np.asarray(p)[:2] - p_rest[:2]) > 1e-3:  # lateral motion over the stand
                for a in pay_g:
                    for b in stand_g:
                        dd = mujoco.mj_geomDistance(model, data, a, b, 0.5, fromto)
                        if dd < rep["min_payload_stand_xy_motion"]:
                            rep["min_payload_stand_xy_motion"] = float(dd)
                            rep["min_payload_stand_xy_motion_at"] = k
            for i in range(1, N_ARMS + 1):
                cl = arm_clearance(model, data, i, _tags=tags)
                for key in ("min", "floor", "stand", "payload", "gripper_payload", "other_arms"):
                    if cl[key] < rep["min_clearance"].get(key, np.inf):
                        rep["min_clearance"][key] = cl[key]
                        rep["min_clearance_at"][key] = (k, i, cl.get(key + "_pair"))
    qs = np.array(qs)
    rep["q_range"] = {"min": qs.min(axis=0).reshape(4, 6).tolist(),
                      "max": qs.max(axis=0).reshape(4, 6).tolist()} if len(qs) else None
    rep["n_collision_poses"] = len(rep["collisions"])
    rep["n_branch_flips"] = len(rep["branch_flips"])
    return rep
