"""Run the four-UR5e cooperative transport and (optionally) render the video.

    pixi run python scripts/run_demo.py --name final
    MUJOCO_GL=egl pixi run python scripts/run_demo.py --name final --render

Writes ``outputs/runs/<name>/`` (log.npz, frames.npz, summary.json) and, with
``--render``, ``outputs/videos/<name>.mp4``.  Exit status 1 if the run diverges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering

import mujoco  # noqa: E402

from homtrans.controller import CooperativeController  # noqa: E402
from homtrans.kinematics import initial_configuration  # noqa: E402
from homtrans.render import FrameRecorder  # noqa: E402
from homtrans.scene import MODEL_XML  # noqa: E402
from homtrans.simulation import RUNS, RunConfig, run  # noqa: E402
from homtrans.trajectory import PayloadTrajectory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="demo")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--az", type=float, default=0.04)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lift", type=float, default=0.08)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--no-feedforward", action="store_true")
    ap.add_argument("--no-payload-feedback", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    traj = PayloadTrajectory(radius=args.radius, period=args.period, A_z=args.az, k=args.k,
                             h_lift=args.lift)
    ctrl = CooperativeController(model, traj, load_ramp=(0.7, 1.0),
                                 feedforward_enabled=not args.no_feedforward,
                                 payload_feedback_enabled=not args.no_payload_feedback)
    qpos0 = initial_configuration(model)
    recorder = FrameRecorder(model, fps=30, model_path=MODEL_XML)
    cfg = RunConfig(name=args.name, duration=args.duration,
                    notes={"radius": args.radius, "period": args.period, "A_z": args.az,
                           "k": args.k, "h_lift": args.lift,
                           "feedforward": not args.no_feedforward,
                           "payload_feedback": not args.no_payload_feedback})
    summary = run(ctrl, traj, qpos0, cfg, model=model, recorder=recorder)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=float))
    if args.render and summary["status"] == "completed":
        from homtrans.render import render_video  # noqa: PLC0415
        out = HERE / "outputs" / "videos" / f"{args.name}.mp4"
        render_video(MODEL_XML, RUNS / args.name / "frames.npz", out, camera="overview")
        print(f"video: {out}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
