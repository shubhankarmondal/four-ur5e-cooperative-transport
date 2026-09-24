"""Simulate the demonstration headlessly and render it to MP4.

Writes ``outputs/videos/four_ur5e_cooperative_transport.mp4``: an oblique view and
a top view with the recorded payload path, side by side, in real time. Needs an
offscreen OpenGL context (``MUJOCO_GL=egl``, or ``osmesa`` for software rendering).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen; set MUJOCO_GL=osmesa for software rendering
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402

from homtrans.controller import CooperativeController  # noqa: E402
from homtrans.kinematics import initial_configuration  # noqa: E402
from homtrans.render import FrameRecorder, render_video  # noqa: E402
from homtrans.scene import MODEL_XML  # noqa: E402
from homtrans.simulation import RUNS, RunConfig, run  # noqa: E402
from homtrans.trajectory import PayloadTrajectory  # noqa: E402

OUT = ROOT / "outputs" / "videos" / "four_ur5e_cooperative_transport.mp4"
TITLE = [
    "Four UR5e cooperative transport",
    "Four Robotiq 2F-85 contact grasps carry a 1.0 kg plate",
    "XY circle R = 0.15 m with Z sinusoid 0.04 m  |  closed-loop MuJoCo, real time",
    "left: oblique view   right: top view with the recorded payload path",
]


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    trajectory = PayloadTrajectory()
    controller = CooperativeController(model, trajectory, load_ramp=(0.7, 1.0))
    recorder = FrameRecorder(model, fps=30, model_path=MODEL_XML)
    summary = run(controller, trajectory, initial_configuration(model),
                  RunConfig(name="render"), model=model, recorder=recorder)
    print(f"simulation {summary['status']}: max payload tracking error "
          f"{1e3 * summary['max_pos_err_m']:.2f} mm")
    if summary["status"] != "completed":
        return 1
    report = render_video(MODEL_XML, RUNS / "render" / "frames.npz", OUT,
                          camera=["overview", "top"], trail=True, width=1920, height=1080,
                          title_card=TITLE, caption="t = {t:5.2f} s")
    print(f"video: {report.get('video', OUT)} (verified: {report.get('verified')})")
    return 0 if report.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
