"""One 2F-85, jaws closing vertically, pinching a 0.25 kg block: holding capacity.

Applies an external force ramp (20 N/s, 0 -> 50 N) to the block along the
approach, lateral and downward directions, at the Menagerie default grip force,
and reports the force at which the block has moved 2 mm.  The 1.0 kg plate
shared by four grippers loads each with about 2.5 N.

    pixi run python scripts/diagnostics/grip_capacity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from homtrans.scene import GRIPPER_XML, _absolute_mesh_paths  # noqa: E402


def build(force_range: float = 5.0, mass: float = 0.25):
    s = mujoco.MjSpec()
    s.option.timestep = 0.001
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10
    g = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    _absolute_mesh_paths(g, GRIPPER_XML)
    g.actuator("fingers_actuator").forcerange = [-force_range, force_range]
    frame = s.worldbody.add_frame(pos=[0, 0, 0.5])
    R = np.column_stack([[0, -1, 0], [0, 0, 1], [1, 0, 0]])  # closing axis vertical
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    frame.quat = q
    s.attach(g, prefix="g_", frame=frame)
    m0 = s.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    b = s.worldbody.add_body(name="block", pos=d0.site("g_pinch").xpos.copy())
    b.add_freejoint(name="block_free")
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.015], mass=mass,
               friction=[1.0, 0.02, 0.001], priority=2, solref=[0.004, 1],
               solimp=[0.95, 0.99, 0.001, 0.5, 2])
    return s.compile()


def main() -> int:
    m = build()
    pinch = m.site("g_pinch").id
    for label, direction in (("approach", None), ("lateral", 0), ("down", 2)):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        bid, act = m.body("block").id, m.actuator("g_fingers_actuator").id
        approach = d.site_xmat[pinch].reshape(3, 3)[:, 2]
        p_ref, moved_at = None, None
        for _ in range(int(4.0 / m.opt.timestep)):
            t = d.time
            d.ctrl[act] = min(255.0, 255.0 * t / 0.5)
            d.xfrc_applied[bid, :] = 0.0
            if t > 1.5:
                F = 20.0 * (t - 1.5)
                if direction is None:
                    d.xfrc_applied[bid, :3] = F * approach
                elif direction == 2:
                    d.xfrc_applied[bid, 2] = -F
                else:
                    d.xfrc_applied[bid, direction] = F
            mujoco.mj_step(m, d)
            if p_ref is None and t > 1.5:
                p_ref = d.body("block").xpos.copy()
            if p_ref is not None and moved_at is None and \
                    np.linalg.norm(d.body("block").xpos - p_ref) > 0.002:
                moved_at = 20.0 * (d.time - 1.5)
        print(f"{label:9s}: 2 mm displacement at F = {moved_at} N (None = held up to 50 N)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
