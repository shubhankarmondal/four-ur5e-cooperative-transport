"""Four UR5e arms carry a plate around a circle: interactive MuJoCo demonstration.

The cooperative controller runs in real time in the MuJoCo viewer. Without a
display, or with ``--headless``, the same closed loop runs headless and prints a
summary instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from homtrans.controller import CooperativeController  # noqa: E402
from homtrans.kinematics import initial_configuration  # noqa: E402
from homtrans.scene import MODEL_XML  # noqa: E402
from homtrans.simulation import RunConfig, payload_state, rotation_angle, run  # noqa: E402
from homtrans.trajectory import PayloadTrajectory  # noqa: E402

CARRIED = ("hold", "circle", "settle")


def build() -> tuple[mujoco.MjModel, PayloadTrajectory, CooperativeController]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    trajectory = PayloadTrajectory()
    controller = CooperativeController(model, trajectory, load_ramp=(0.7, 1.0))
    return model, trajectory, controller


def headless(model, trajectory, controller) -> int:
    summary = run(controller, trajectory, initial_configuration(model),
                  RunConfig(name="demo", record_frames=False), model=model)
    print(f"status: {summary['status']} {summary['reason']}".rstrip())
    print(f"simulated {summary['t_end']:.2f} s in {summary['wall_seconds']:.1f} s wall time")
    print(f"payload tracking error: max {1e3 * summary['max_pos_err_m']:.2f} mm, "
          f"rms {1e3 * summary['rms_pos_err_m']:.2f} mm; "
          f"orientation error max {summary['max_rot_err_rad']:.4f} rad")
    print(f"peak arm torque {100 * summary['max_abs_torque_fraction']:.0f} % of limit; "
          f"min pad force while carried {summary['min_pad_force_while_carried_N']:.1f} N; "
          f"unwanted contacts: {len(summary['unwanted_contacts_first_seen'])}")
    return 0 if summary["status"] == "completed" else 1


def interactive(model, trajectory, controller, exit_when_done: bool) -> int:
    import mujoco.viewer  # noqa: PLC0415

    data = mujoco.MjData(model)
    data.qpos[:] = initial_configuration(model)
    mujoco.mj_forward(model, data)
    max_err = 0.0
    reported = False
    before = set(threading.enumerate())
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -30.0
            wall0, sim0, next_sync = time.perf_counter(), data.time, 0.0
            while viewer.is_running():
                mujoco.mj_step1(model, data)
                ctrl, _ = controller.compute(model, data, data.time)
                data.ctrl[:] = ctrl
                mujoco.mj_step2(model, data)
                sample = trajectory.sample(data.time)
                if sample.phase in CARRIED and data.time <= trajectory.duration:
                    p, _, _, _ = payload_state(model, data)
                    max_err = max(max_err, float(np.linalg.norm(sample.p - p)))
                if data.time >= next_sync:
                    viewer.sync()
                    next_sync = data.time + 1.0 / 60.0
                    lag = (data.time - sim0) - (time.perf_counter() - wall0)
                    if lag > 0.0:
                        time.sleep(lag)  # real-time pacing
                if not reported and data.time >= trajectory.duration:
                    _, R, _, _ = payload_state(model, data)
                    print(f"trajectory complete at t = {data.time:.2f} s: max payload tracking "
                          f"error {1e3 * max_err:.2f} mm, final orientation error "
                          f"{rotation_angle(sample.R.T @ R):.4f} rad")
                    reported = True
                    if exit_when_done:
                        break
                    print("holding the final pose; close the viewer window to exit")
            viewer.close()
    finally:
        # Wait for the viewer's render thread to tear down its GL context before the
        # interpreter shuts down; exiting while it does so can crash.
        for thread in set(threading.enumerate()) - before:
            thread.join(timeout=5.0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--headless", action="store_true", help="run without the viewer")
    ap.add_argument("--exit-when-done", action="store_true",
                    help="close the viewer when the trajectory ends")
    args = ap.parse_args()
    model, trajectory, controller = build()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if args.headless or not has_display:
        if not args.headless:
            print("no display found: running headless")
        return headless(model, trajectory, controller)
    return interactive(model, trajectory, controller, args.exit_when_done)


if __name__ == "__main__":
    raise SystemExit(main())
