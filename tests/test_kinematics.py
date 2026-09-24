"""Fast checks of homtrans.kinematics on models/four_ur5e_transport.xml."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from homtrans import kinematics as K

MODEL = Path(__file__).resolve().parents[1] / "models" / "four_ur5e_transport.xml"


@pytest.fixture(scope="module")
def model():
    if not MODEL.exists():
        pytest.skip(f"{MODEL} not built yet")
    return mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def q_init(model):
    return K.initial_configuration(model)


def _rot_angle(Ra, Rb):
    return float(np.linalg.norm(K.rotation_error(Ra, Rb)))


def test_initial_pinch_frames_match_handle_frames(model, q_init):
    data = mujoco.MjData(model)
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)
    for i in range(1, K.N_ARMS + 1):
        p_h, R_h = K.desired_pinch_pose(model, data, i)
        sid = K.pinch_site_id(model, i)
        assert np.linalg.norm(data.site_xpos[sid] - p_h) < 1e-5
        assert _rot_angle(R_h, data.site_xmat[sid].reshape(3, 3)) < 1e-5


def test_initial_configuration_payload_default_gripper_open(model, q_init):
    a = model.jnt_qposadr[model.joint(K.PAYLOAD_JOINT).id]
    np.testing.assert_allclose(q_init[a:a + 7], model.qpos0[a:a + 7], atol=1e-12)
    for i in range(1, K.N_ARMS + 1):
        g = model.jnt_qposadr[K.gripper_joint_ids(model, i)]
        assert np.all(q_init[g] == 0.0)
        lo, hi = K.arm_limits(model, i)
        q = q_init[K.arm_qpos_adr(model, i)]
        assert np.all(q > lo) and np.all(q < hi)


def test_initial_configuration_has_no_unwanted_contacts(model, q_init):
    data = mujoco.MjData(model)
    data.qpos[:] = q_init
    assert K.unwanted_contacts(model, data) == []


def test_initial_configuration_is_symmetric(model, q_init):
    q1 = q_init[K.arm_qpos_adr(model, 1)]
    for i in range(2, K.N_ARMS + 1):
        np.testing.assert_allclose(q_init[K.arm_qpos_adr(model, i)], q1, atol=1e-4)


def test_pinch_targets_match_handle_sites_for_moved_payload(model, q_init):
    data = mujoco.MjData(model)
    data.qpos[:] = q_init
    p = np.array([0.07, -0.04, 0.52])
    R = np.zeros(9)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0.3, -0.2, 1.0]) / np.linalg.norm([0.3, -0.2, 1.0]), 0.4)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    K.set_payload_pose(model, data, p, R)
    mujoco.mj_kinematics(model, data)
    targets = K.pinch_targets_for_payload_pose(model, p, R)
    for i, (pt, Rt) in enumerate(targets, start=1):
        p_s, R_s = K.desired_pinch_pose(model, data, i)
        assert np.linalg.norm(pt - p_s) < 1e-12
        assert _rot_angle(Rt, R_s) < 1e-9
        p_c, R_c = K.desired_pinch_pose(model, data, i, p, R)
        assert np.linalg.norm(p_c - p_s) < 1e-12
        assert _rot_angle(R_c, R_s) < 1e-9


@pytest.mark.parametrize("i", [1, 2, 3, 4])
def test_ik_converges_and_touches_only_arm_i(model, q_init, i):
    data = mujoco.MjData(model)
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)
    p_pay, R_pay = K.payload_pose_from_qpos(model, q_init)
    # lifted and shifted payload pose, cold-ish seed (init perturbed by 0.3 rad)
    p_t, R_t = K.pinch_targets_for_payload_pose(model, p_pay + np.array([0.06, -0.05, 0.10]), R_pay)[i - 1]
    adr = K.arm_qpos_adr(model, i)
    seed = q_init[adr] + 0.3 * np.array([1, -1, 1, -1, 1, -1])
    before = data.qpos.copy()
    q, info = K.solve_arm_ik(model, data, i, p_t, R_t, seed)
    assert info["converged"], info
    assert info["pos_err"] < 1e-6 and info["rot_err"] < 1e-5
    assert info["within_limits"]
    mask = np.ones(model.nq, dtype=bool)
    mask[adr] = False
    np.testing.assert_array_equal(data.qpos[mask], before[mask])
    np.testing.assert_array_equal(data.qpos[adr], q)
    mujoco.mj_forward(model, data)
    sid = K.pinch_site_id(model, i)
    assert np.linalg.norm(data.site_xpos[sid] - p_t) < 1e-6
    assert _rot_angle(R_t, data.site_xmat[sid].reshape(3, 3)) < 1e-5


def test_unwanted_contacts_flags_robot_collisions(model, q_init):
    data = mujoco.MjData(model)
    data.qpos[:] = q_init
    adr = K.arm_qpos_adr(model, 1)
    # arm 1 folded down into the floor
    data.qpos[adr] = [np.pi, 0.9, 0.3, 0.0, 0.0, 0.0]
    bad = K.unwanted_contacts(model, data)
    assert any({c["body1"], c["body2"]} & {"world"} for c in bad)
    # arm 1 pushed 5 cm along its approach axis: the palm hits the handle (not a pad contact)
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)
    p, R = K.desired_pinch_pose(model, data, 1)
    _, info = K.solve_arm_ik(model, data, 1, p + 0.05 * R[:, 2], R, q_init[adr])
    assert info["converged"]
    bad = K.unwanted_contacts(model, data)
    assert any("payload" in (c["body1"], c["body2"]) for c in bad)
    assert all("pad" not in c["body1"] + c["body2"] for c in bad)
