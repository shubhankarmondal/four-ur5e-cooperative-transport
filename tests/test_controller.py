"""Tests for trajectory.py, allocation.py and controller.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from homtrans.allocation import DEFAULT_MOMENT_WEIGHT, allocate, grasp_matrix
from homtrans.controller import (
    ArmGains,
    ObjectGains,
    TORQUE_LIMITS,
    arm_torque,
    object_wrench,
    so3_exp,
    so3_log,
)
from homtrans.trajectory import PHASES, PayloadTrajectory

MODEL_XML = Path(__file__).resolve().parents[1] / "models" / "four_ur5e_transport.xml"
RNG = np.random.default_rng(0)
TRAJS = [
    PayloadTrajectory(),
    PayloadTrajectory(n_circles=2, k=3, radius=0.06, period=7.0, T_ramp=1.5, A_z=0.015),
]


def _sample_times(traj: PayloadTrajectory) -> np.ndarray:
    bounds = [b for _, t0, t1 in traj.phase_boundaries for b in (t0, t1)]
    grid = np.linspace(-0.5, traj.duration + 0.5, 1501)
    return np.unique(np.r_[grid, bounds])


# ------------------------------------------------------------------ trajectory
@pytest.mark.parametrize("traj", TRAJS)
def test_velocity_matches_central_difference_of_position(traj):
    h = 1e-5
    for t in _sample_times(traj):
        fd = (traj.sample(t + h).p - traj.sample(t - h).p) / (2 * h)
        assert np.max(np.abs(traj.sample(t).v - fd)) < 1e-6, t


@pytest.mark.parametrize("traj", TRAJS)
def test_acceleration_matches_central_differences(traj):
    bounds = np.array([b for _, t0, t1 in traj.phase_boundaries for b in (t0, t1)])
    for t in _sample_times(traj):
        s = traj.sample(t)
        h = 1e-6  # first difference of the analytic v, valid across boundaries (C^2)
        fd_v = (traj.sample(t + h).v - traj.sample(t - h).v) / (2 * h)
        assert np.max(np.abs(s.a - fd_v)) < 1e-6, t
        if np.min(np.abs(bounds - t)) > 1e-3:  # second difference of p, interior points
            h = 1e-4
            fd_p = (traj.sample(t + h).p - 2 * s.p + traj.sample(t - h).p) / h**2
            assert np.max(np.abs(s.a - fd_p)) < 1e-6, t


@pytest.mark.parametrize("traj", TRAJS)
def test_continuity_and_rest_at_every_phase_boundary(traj):
    eps = 1e-9
    names = [name for name, _, _ in traj.phase_boundaries]
    assert tuple(names) == PHASES
    for name, t0, _ in traj.phase_boundaries:
        lo, hi = traj.sample(t0 - eps), traj.sample(t0 + eps)
        for attr in ("p", "v", "a"):
            assert np.max(np.abs(getattr(lo, attr) - getattr(hi, attr))) < 1e-7, (name, attr)
        # every boundary is a rest point
        assert np.max(np.abs(hi.v)) < 1e-7 and np.max(np.abs(hi.a)) < 1e-6, name
        assert traj.sample(t0 + 1e-6).phase == name


@pytest.mark.parametrize("traj", TRAJS)
def test_circle_is_concentric_and_makes_full_turns_at_full_radius(traj):
    """Concentric circle about p_lift: spiral out, n full turns at radius, spiral in."""
    t_c0, t_end = [(t0, t1) for n, t0, t1 in traj.phase_boundaries if n == "circle"][0]
    s = traj.sample(t_end)
    assert s.phase == "settle"
    assert np.allclose(traj.sample(t_end - 1e-12).p, traj.p_lift, atol=1e-10)
    assert np.allclose(traj.sample(t_c0).p, traj.p_lift, atol=1e-12)
    assert np.allclose(traj.circle_center, traj.p_lift)
    # never farther than the radius from the lift point (horizontally)
    taus = np.linspace(0.0, traj.T_circle, 401)
    dist = [np.hypot(*(traj.sample(t_c0 + tau).p - traj.p_lift)[:2]) for tau in taus]
    assert max(dist) == pytest.approx(traj.radius, abs=1e-12)
    # at full radius for exactly n_circles * period, i.e. n full turns
    full = [tau for tau, d in zip(taus, dist) if abs(d - traj.radius) < 1e-12]
    swept = traj.circle_rate * (max(full) - min(full))
    assert swept == pytest.approx(2 * np.pi * traj.n_circles, rel=0.01)
    assert traj.circle_angle(traj.T_ramp + 0.5 * traj.T_plateau)[1] == pytest.approx(traj.circle_rate)


def test_trajectory_defaults_and_validation():
    traj = PayloadTrajectory()
    assert traj.T_plateau == pytest.approx(10.0) and traj.T_circle == pytest.approx(14.0)
    assert traj.duration == pytest.approx(1.0 + 2.0 + 1.0 + 14.0 + 1.5)
    assert np.allclose(traj.p_lift, [0.0, 0.0, 0.48])
    s = traj.sample(0.3)
    assert s.phase == "grip" and np.allclose(s.p, traj.p_rest) and np.allclose(s.R, np.eye(3))
    assert np.all(s.omega == 0) and np.all(s.alpha == 0)
    assert traj.sample(-1.0).phase == "grip" and traj.sample(1e3).phase == "settle"
    with pytest.raises(ValueError):
        PayloadTrajectory(radius=-0.1)
    with pytest.raises(ValueError):
        PayloadTrajectory(n_circles=0)


# ------------------------------------------------------------------ allocation
def _square_grasps(d=0.275, z=0.48):
    return np.array([[d, 0, z], [0, d, z], [-d, 0, z], [0, -d, z]]), np.array([0.0, 0.0, z])


def test_grasp_matrix_matches_wrench_sum():
    pts = RNG.normal(size=(4, 3))
    p = RNG.normal(size=3)
    w = RNG.normal(size=24).reshape(4, 6)
    W = np.zeros(6)
    for r, (fi, mi) in zip(pts, zip(w[:, :3], w[:, 3:])):
        W += np.r_[fi, mi + np.cross(r - p, fi)]
    assert np.allclose(grasp_matrix(p, pts) @ w.ravel(), W)


def test_allocation_exact_reconstruction_and_weighted_min_norm():
    pts, p = _square_grasps()
    pts = pts + 0.02 * RNG.normal(size=pts.shape)
    G = grasp_matrix(p, pts)
    for _ in range(10):
        W = RNG.normal(size=6)
        w, res = allocate(G, W)
        assert res < 1e-10 and np.allclose(G @ w, W, atol=1e-10)
        # optimality: w is Q-orthogonal to the null space of G
        _, _, Vt = np.linalg.svd(G)
        N = Vt[6:].T
        Q = np.tile(np.r_[np.ones(3), np.full(3, DEFAULT_MOMENT_WEIGHT)], 4)
        assert np.max(np.abs(N.T @ (Q * w))) < 1e-9
    w, res = allocate(G, W, damping=1e-2)
    assert res == pytest.approx(np.linalg.norm(G @ w - W)) and res > 0


def test_symmetric_weight_is_shared_equally_without_squeeze():
    pts, p = _square_grasps()
    m, g = 1.0, 9.81
    w, res = allocate(grasp_matrix(p, pts), np.r_[0, 0, m * g, 0, 0, 0])
    f = w.reshape(4, 6)
    assert res < 1e-12
    assert np.allclose(f[:, 2], m * g / 4, atol=1e-12)
    assert np.allclose(f[:, :2], 0.0, atol=1e-12)  # no internal horizontal squeeze
    assert np.allclose(f[:, 3:], 0.0, atol=1e-12)  # no gripper moments


def test_yaw_torque_is_realised_by_tangential_forces():
    pts, p = _square_grasps()
    tz = 1.0
    w, _ = allocate(grasp_matrix(p, pts), np.r_[0, 0, 0, 0, 0, tz])
    f = w.reshape(4, 6)
    from_forces = sum(np.cross(r - p, fi[:3])[2] for r, fi in zip(pts, f))
    from_moments = f[:, 5].sum()
    assert from_forces / tz > 0.99 and from_moments / tz < 0.01
    radial = [fi[:3] @ (r - p) / np.linalg.norm(r - p) for r, fi in zip(pts, f)]
    assert np.allclose(radial, 0.0, atol=1e-12)


# ------------------------------------------------------------------ controller
def test_so3_log_is_the_exact_inverse_of_exp():
    for _ in range(200):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        for th in (RNG.uniform(0, np.pi - 1e-2), 1e-9, 1e-5, 0.5, np.pi - 1e-4, np.pi - 1e-8):
            w = th * axis
            assert np.allclose(so3_log(so3_exp(w)), w, atol=1e-8), th
    R = so3_exp(np.pi * np.array([0.0, 0.6, 0.8]))
    assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-12)
    assert np.allclose(so3_log(np.eye(3)), 0.0)


def test_object_wrench_static_hold_and_pd_signs():
    traj = PayloadTrajectory()
    s = traj.sample(0.5)
    m, g = 1.3, 9.81
    I_body = np.diag([0.03, 0.02, 0.05])
    gains = ObjectGains()
    W = object_wrench(m, I_body, s.p, np.zeros(3), s.R, np.zeros(3), s, gains, g)
    assert np.allclose(W, [0, 0, m * g, 0, 0, 0], atol=1e-12)
    # position error along +x -> force along +x; small rotation error -> I Kr e_R
    W = object_wrench(m, I_body, s.p - [0.01, 0, 0], np.zeros(3), s.R, np.zeros(3), s, gains, g)
    assert W[0] == pytest.approx(m * gains.kp * 0.01)
    e = np.array([0.0, 0.0, 1e-3])
    R = so3_exp(-e) @ s.R  # R_d R^T = exp(e)
    W = object_wrench(m, I_body, s.p, np.zeros(3), R, np.zeros(3), s, gains, g)
    assert np.allclose(W[3:], R @ I_body @ R.T @ (gains.kr * e), atol=1e-12)
    # feedback disabled -> feed-forward only
    W = object_wrench(m, I_body, s.p - 0.1, np.ones(3), R, np.zeros(3), s, gains, g, feedback=False)
    assert np.allclose(W, [0, 0, m * g, 0, 0, 0], atol=1e-12)


def test_arm_torque_zero_error_is_bias_plus_JT_feedforward():
    J = 0.3 * RNG.normal(size=(6, 6))
    bias = RNG.normal(size=6)
    R = so3_exp(RNG.normal(size=3))
    p, v, w = RNG.normal(size=3), RNG.normal(size=3), RNG.normal(size=3)
    f_ff = RNG.normal(size=6)
    tau, sat = arm_torque(J, bias, p, R, v, w, p, R, v, w, f_ff, ArmGains())
    assert np.allclose(tau, bias + J.T @ f_ff, atol=1e-12) and not sat.any()
    # position error -> J^T K_p e ; saturation is clipped and flagged
    gains = ArmGains()
    e = np.array([0.001, -0.002, 0.0005])
    tau, _ = arm_torque(J, bias, p, R, v, w, p + e, R, v, w, np.zeros(6), gains)
    assert np.allclose(tau, bias + J.T @ np.r_[gains.kp * e, 0, 0, 0], atol=1e-10)
    tau, sat = arm_torque(np.eye(6), np.zeros(6), p, R, v, w, p, R, v, w, 1e3 * np.ones(6), gains)
    assert sat.all() and np.allclose(tau, TORQUE_LIMITS)


# ------------------------------------------------------------- MuJoCo smoke test
def test_cooperative_controller_smoke_on_scene():
    if not MODEL_XML.exists():
        pytest.skip(f"{MODEL_XML} not built yet (run scripts/build_scene.py)")
    import mujoco

    from homtrans.controller import CooperativeController, quat_to_mat

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    traj = PayloadTrajectory(p_rest=tuple(model.qpos0[0:3]))
    ctl = CooperativeController(model, traj)
    mujoco.mj_forward(model, data)
    ctrl, diag = ctl.compute(model, data, 0.0)
    assert ctrl.shape == (model.nu,) and np.all(np.isfinite(ctrl))
    assert np.all(ctrl[ctl.gripper_acts] == 0.0)
    assert np.allclose(diag["payload_pos_err"], 0.0) and diag["alloc_residual"] < 1e-9
    assert np.allclose(diag["f"][:, 2], model.body_mass[ctl.payload_body] * ctl.g / 4, atol=1e-9)
    # handle offsets reproduce MuJoCo's handle site poses at the rest pose
    for k in range(4):
        sid = model.site(f"handle_{k + 1}").id
        p_h = traj.p_rest + traj.R0 @ ctl.handle_pos[k]
        assert np.allclose(p_h, data.site_xpos[sid], atol=1e-12)
        assert np.allclose(traj.R0 @ ctl.handle_rot[k], data.site_xmat[sid].reshape(3, 3), atol=1e-12)
    # free-joint angular velocity is body-frame in qvel: check the world conversion
    data.qpos[3:7] = [0.9, 0.1, -0.3, 0.2]
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qvel[3:6] = [0.4, -0.7, 1.1]
    mujoco.mj_forward(model, data)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ctl.payload_body, vel, 0)
    _, R, _, omega = ctl.payload_state(data)
    assert np.allclose(R, quat_to_mat(data.qpos[3:7])) and np.allclose(omega, vel[:3], atol=1e-12)
    ctrl, diag = ctl.compute(model, data, 5.0)
    assert np.all(np.isfinite(ctrl)) and diag["phase"] == "circle"
    assert 0 < ctrl[ctl.gripper_acts[0]] <= 255.0
