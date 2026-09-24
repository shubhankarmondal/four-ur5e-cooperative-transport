"""Fast subset of scripts/validate.py (< 60 s): model integrity, no payload weld,
and one short grip + lift run showing lift-off and a four-gripper pad grasp.

The full adversarial validation is ``scripts/validate.py``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def V():
    spec = importlib.util.spec_from_file_location("validate", HERE / "scripts" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate"] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def model(V):
    if not V.MODEL_XML.exists():
        pytest.skip(f"{V.MODEL_XML} not built yet")
    return mujoco.MjModel.from_xml_path(str(V.MODEL_XML))


@pytest.fixture(scope="module")
def lift_run(V, model):
    """Grip (0-1 s) + lift (1-3 s) + 0.6 s of hold, with the full per-step log."""
    return V.simulate(V.Scenario("test_grip_lift", duration=3.6, full_log=True))


def test_model_integrity(V, model):
    bad = [(c.id, c.name, c.value) for c in V.check_model(model) if c.status != "PASS" and c.id != "1g"]
    assert not bad, bad


def test_model_file_is_fresh_build_of_scene(V, model):
    (c,) = [c for c in V.check_model(model) if c.id == "1g"]
    assert c.status == "PASS", c.value


def test_no_payload_or_world_weld(model):
    pay = model.body("payload").id
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert [model.joint(j).name for j in free] == ["payload_free"]
    assert model.jnt_bodyid[free[0]] == pay
    for e in range(model.neq):
        ty = int(model.eq_type[e])
        o1, o2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if ty in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
            assert int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_BODY)
            assert pay not in (o1, o2) and 0 not in (o1, o2), f"equality {e} touches payload/world"
            names = (model.body(o1).name, model.body(o2).name)
        elif ty == int(mujoco.mjtEq.mjEQ_JOINT):
            assert o2 >= 0
            assert model.jnt_bodyid[o1] != pay and model.jnt_bodyid[o2] != pay
            names = (model.joint(o1).name, model.joint(o2).name)
        else:
            pytest.fail(f"unexpected equality type {ty} (id {e})")
        m1, m2 = (re.match(r"(ur\d)_g_", n) for n in names)
        assert m1 and m2 and m1.group(1) == m2.group(1), names
    assert model.nmocap == 0
    assert not np.any(model.body_gravcomp)


def test_initial_state(V, model):
    c = V.check_initial_state(model, V.nominal_qpos0())
    assert c.status == "PASS", c.value


def test_grip_and_lift(V, lift_run):
    r = lift_run
    assert r["status"] == "completed", r["reason"]
    L, t = r["log"], r["log"]["t"]
    z_rest, h_lift = r["traj"]["p_rest"][2], V.demo_trajectory().h_lift
    mg = r["mass"] * 9.81
    # resting: the stand carries the full weight, no pad touches the payload yet
    rest = (t > 0.1) & (t < 0.4)  # after the first-contact settling transient
    assert np.allclose(L["stand_fn"][rest], mg, rtol=0.02)
    assert not L["pad_fn"][rest].any()
    # lifted and held: stand force exactly zero, both pads of all four grippers loaded
    held = t >= 3.1
    assert held.sum() > 100
    assert np.all(L["stand_fn"][held] == 0.0)
    assert np.all(L["pad_fn"][held] > 1.0), L["pad_fn"][held].min(axis=0)
    assert np.all(L["p"][held, 2] > z_rest + 0.9 * h_lift)
    # the four grippers share the weight through their pads alone
    assert abs(L["pad_fz"][held].sum(axis=1).mean() - mg) < 0.05 * mg
    assert not L["finger_pay_n"].any() and not L["link_pay_n"].any()
    assert not L["armarm_n"].any() and not L["robot_floor_n"].any() and not L["robot_stand_n"].any()
    # nothing but gravity and contact acts on the payload
    assert np.abs(L["ne_exact"]).max() < 1e-9 * mg
    assert not L["tamper"].any()
