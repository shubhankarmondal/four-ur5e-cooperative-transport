"""Close the grippers on the resting plate and report each arm's settled pinch offset.

The grip phase is stretched to 3 s so the plate stays on the stand, and the
allocated feed-forward is off: only the arm impedance and the jaws act.  A small
offset of the actual pinch point from its target (``handle_{i}``, placed by
``SceneParams.grasp_site_radius``) means the closed grasp is in equilibrium with
the arm target, i.e. closing the jaws does not squeeze the plate.

    pixi run python scripts/diagnostics/calibrate_grasp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from homtrans.controller import CooperativeController  # noqa: E402
from homtrans.kinematics import initial_configuration  # noqa: E402
from homtrans.scene import MODEL_XML, N_ARMS  # noqa: E402
from homtrans.trajectory import PayloadTrajectory  # noqa: E402


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    traj = PayloadTrajectory(t_grip=3.0)
    ctrl = CooperativeController(model, traj, load_ramp=(2.0, 2.5), feedforward_enabled=False)
    data.qpos[:] = initial_configuration(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step1(model, data)
        u, diag = ctrl.compute(model, data, data.time)
        data.ctrl[:] = u
        mujoco.mj_step2(model, data)
    err = np.asarray(diag["pinch_pos_err"])  # desired - actual, world frame
    for i in range(1, N_ARMS + 1):
        R_h = data.site(f"handle_{i}").xmat.reshape(3, 3)
        off = R_h.T @ (-err[i - 1])  # actual - target in the handle frame (y up, z approach)
        print(f"arm {i}: actual-target pinch offset (x, y_up, z_approach) = "
              f"{np.round(1000 * off, 2)} mm")
    print("payload shift from rest (mm):",
          np.round(1000 * (data.body("payload").xpos - traj.p_rest), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
