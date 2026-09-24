"""Peak normal force of every payload contact with a robot body other than the pads.

Runs the full transport (sampled every 10 steps) and prints, per robot body that
touched the payload, the peak normal force, its time and trajectory phase.  In a
clean pad grasp nothing is printed but the final summary line.

    pixi run python scripts/diagnostics/finger_contacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from homtrans.controller import CooperativeController  # noqa: E402
from homtrans.kinematics import initial_configuration  # noqa: E402
from homtrans.scene import MODEL_XML  # noqa: E402
from homtrans.trajectory import PayloadTrajectory  # noqa: E402


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    traj = PayloadTrajectory(radius=0.15, A_z=0.04)
    ctrl = CooperativeController(model, traj, load_ramp=(0.7, 1.0))
    data.qpos[:] = initial_configuration(model)
    mujoco.mj_forward(model, data)
    payload = model.body("payload").id
    force = np.zeros(6)
    peak: dict[str, tuple[float, float, str]] = {}  # body -> (normal force, t, phase)
    for step in range(int(traj.duration / model.opt.timestep)):
        mujoco.mj_step1(model, data)
        u, _ = ctrl.compute(model, data, data.time)
        data.ctrl[:] = u
        mujoco.mj_step2(model, data)
        if step % 10:
            continue
        for k in range(data.ncon):
            con = data.contact[k]
            b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
            if payload not in (b1, b2):
                continue
            name = model.body(b2 if b1 == payload else b1).name
            if not name.startswith("ur") or name.endswith("_pad"):
                continue
            mujoco.mj_contactForce(model, data, k, force)
            if abs(force[0]) > peak.setdefault(name, (0.0, 0.0, ""))[0]:
                peak[name] = (abs(force[0]), data.time, traj.sample(data.time).phase)
    for name, (fn, t, phase) in sorted(peak.items()):
        print(f"{name:32s} peak normal {fn:7.2f} N at t={t:5.2f} ({phase})")
    print(f"{len(peak)} non-pad robot bodies touched the payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
