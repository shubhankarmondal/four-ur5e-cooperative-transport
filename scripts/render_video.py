#!/usr/bin/env python
"""Render a recorded run (.npz from homtrans.render.FrameRecorder) to a verified MP4.

    MUJOCO_GL=egl pixi run python scripts/render_video.py \
        --record outputs/runs/final/frames.npz --out outputs/videos/final.mp4 \
        --camera overview --title "Four UR5e cooperative transport" \
        --subtitle "real time, MuJoCo 3.13" --caption "t = {t:5.2f} s"

    # a still of one camera (default configuration or a recorded frame)
    ... scripts/render_video.py --still outputs/stills/overview.png --camera overview \
        [--record outputs/runs/final/frames.npz --frame 120]

Camera: a model camera name, "preset:<overview|top|closeup>" (tuned free cameras in
homtrans.render.CAMERA_PRESETS; a bare preset name is used only if the model lacks that
camera), or a free camera "az,el,dist,x,y,z" (degrees, metres).
Side by side: "--camera overview,top" (tiles split --width equally; the names overview/top
become their 8:9-tile variants, homtrans.render.TILE_CAMERAS); use ";" between tiles when
a tile is a free camera.  --trail draws the recorded payload path in the top tile.

    ... scripts/render_video.py --record outputs/runs/final/frames.npz \
        --out outputs/videos/final_split.mp4 --camera overview,top --trail --width 1920 --height 1080

Exit status 1 if verification fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from homtrans.render import render_still, render_video  # noqa: E402


def _one_camera(text: str):
    parts = text.split(",")
    if len(parts) == 6:
        az, el, dist, x, y, z = map(float, parts)
        return {"azimuth": az, "elevation": el, "distance": dist, "lookat": (x, y, z)}
    return int(text) if text.isdigit() else text


def parse_camera(text: str):
    """'overview' | 'az,el,dist,x,y,z' | tiles: 'overview,top' or 'overview;az,el,dist,x,y,z'."""
    if ";" in text:
        return [_one_camera(t) for t in text.split(";")]
    parts = text.split(",")
    if len(parts) == 6 and all(p.lstrip("-").replace(".", "", 1).isdigit() for p in parts):
        return _one_camera(text)
    return [_one_camera(t) for t in parts] if len(parts) > 1 else _one_camera(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "models" / "four_ur5e_transport.xml"))
    ap.add_argument("--record", help="recording .npz")
    ap.add_argument("--out", help="output .mp4")
    ap.add_argument("--still", help="write one PNG instead of a video")
    ap.add_argument("--frame", type=int, default=None, help="recorded frame for --still (negative: from the end)")
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=None, help="default: the recording's rate")
    ap.add_argument("--title", default=None, help="title card headline (omit for no card)")
    ap.add_argument("--subtitle", action="append", default=[], help="extra title-card line (repeatable)")
    ap.add_argument("--title-seconds", type=float, default=2.5)
    ap.add_argument("--caption", default=None, help='corner caption template, e.g. "t = {t:5.2f} s"')
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--trail", action="store_true", help="draw the recorded payload path in the top tile")
    args = ap.parse_args()
    camera = parse_camera(args.camera)

    if args.still:
        render_still(args.model, args.still, camera, args.width, args.height,
                     record=args.record, frame=args.frame, trail=args.trail)
        print(f"wrote {args.still}")
        return 0

    if not (args.record and args.out):
        ap.error("--record and --out are required for a video (or use --still)")
    title = [args.title, *args.subtitle] if args.title else None
    report = render_video(
        args.model, args.record, args.out, camera=camera, width=args.width, height=args.height,
        fps=args.fps, title_card=title, title_seconds=args.title_seconds, caption=args.caption, crf=args.crf,
        trail=args.trail,
    )
    print(json.dumps({k: report[k] for k in (
        "video", "verified", "problems", "sim_frames", "title_frames", "fps",
        "sim_duration_s", "video_duration_s", "max_frame_time_error_s", "probe", "decoded_pixel_mae",
    )}, indent=2, default=str))
    if "contact_sheet" in report:
        print(f"contact sheet: {report['contact_sheet']}")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
