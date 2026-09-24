"""Recording and headless MP4 rendering for the four-UR5e transport scene.

Two halves, joined by an ``.npz`` file:

* :class:`FrameRecorder` -- the simulation loop calls ``recorder.record(t, data)``
  after every ``mj_step``.  It keeps one state snapshot (qpos, qvel, act, ctrl)
  per video frame, at the simulation times ``t0 + k / fps``: frame ``k`` is the
  first step whose time reaches ``t0 + k / fps``.  The sample-time error is
  therefore below one timestep and never accumulates; the actual sample times
  are stored.  ``save(path)`` writes the npz, :func:`load_recording` reads it.
* :func:`render_video` -- replays the recorded states through ``mj_forward``,
  renders each with ``mujoco.Renderer`` (EGL, no display) and pipes raw RGB to
  ffmpeg (libx264, yuv420p).  One video frame per recorded frame at the
  recording's rate, so video duration equals simulated duration (no slow
  motion).  Optional plain title card before the simulation; at most a small
  caption strip in a corner during it.  Writes a 3x3 contact sheet PNG and a JSON
  provenance sidecar next to the MP4, then verifies the file by decoding it
  (frame count, duration, and a pixel comparison of a few decoded frames).

Rendering needs an offscreen OpenGL context (``MUJOCO_GL=egl``, or ``osmesa``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

try:  # text for the title card, caption strip and contact-sheet labels
    from PIL import Image, ImageDraw, ImageFont

    HAVE_PIL = True
except ImportError:  # pragma: no cover - pillow is a declared dependency
    HAVE_PIL = False

__all__ = [
    "CAMERA_PRESETS",
    "DEFAULT_FPS",
    "TILE_CAMERAS",
    "FrameRecorder",
    "Recording",
    "free_camera_to_fixed",
    "load_recording",
    "probe_video",
    "render_still",
    "render_video",
]

DEFAULT_FPS = 30.0

#: Free-camera fallbacks used when the model has no camera of that name.  Tuned on
#: the four-arm scene (bases 0.9 m from the centre, plate ~0.45 m high); the same
#: views are baked into the model as fixed cameras via :func:`free_camera_to_fixed`.
CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    # elevated oblique: all four arms, grippers, plate and stand
    "overview": {"lookat": (0.0, 0.0, 0.22), "distance": 2.25, "azimuth": 135.0, "elevation": -35.0},
    # straight down, rotated 45 deg so the arms lie on the 16:9 frame diagonals
    "top": {"lookat": (0.0, 0.0, 0.3), "distance": 2.45, "azimuth": 45.0, "elevation": -89.9},
    # side view of arm 1's grasp (handle_1 at +x): jaws closing across the handle
    "closeup": {"lookat": (0.30, 0.0, 0.41), "distance": 0.55, "azimuth": 100.0, "elevation": -12.0},
}

#: Variants substituted for plain names when ``camera`` is a LIST (side-by-side
#: tiles).  Tuned for 8:9 tiles (two tiles in 16:9: 960x1080 or 640x720) so that
#: every visible geom stays inside the tile over the default demonstration
#: (R = 0.15 m circle, 0.08 m lift, 0.04 m bob): its extents fill 92 % (overview)
#: and 94 % (top) of the tile width.  Pass a dict to use anything else.
TILE_CAMERAS: dict[str, dict[str, Any]] = {
    "overview": {"lookat": (0.0, 0.0, 0.3), "distance": 3.4, "azimuth": 130.0, "elevation": -42.0},
    "top": {"name": "top", "fovy": 56.0},
}
_TILE_ASPECT = 8.0 / 9.0

_TIME_EPS = 1e-9  # absorbs floating-point drift in data.time accumulated over steps


# --------------------------------------------------------------------------- recording


@dataclass
class Recording:
    """A loaded recording.  All arrays are materialised in memory (indexing an
    ``NpzFile`` member decompresses the whole member again on every access)."""

    t: np.ndarray  # (N,) actual simulation time of each frame [s]
    qpos: np.ndarray  # (N, nq)
    qvel: np.ndarray  # (N, nv)
    act: np.ndarray  # (N, na)
    ctrl: np.ndarray  # (N, nu)
    fps: float
    model_path: str
    model_sha256: str
    timestep: float
    overlays: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,), NaN = absent

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        """Duration covered by the frames at the nominal rate [s]."""
        return self.n_frames / self.fps


def _sha256(path: str | os.PathLike[str]) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


class FrameRecorder:
    """Samples simulation state at a fixed video rate in simulation time.

    Usage in the simulation loop::

        rec = FrameRecorder(model, fps=30, model_path="models/four_ur5e_transport.xml")
        rec.record(data.time, data)          # once before the first step (frame 0)
        for ...:
            mujoco.mj_step(model, data)
            rec.record(data.time, data, overlays={"z_err_mm": e})   # every step
        rec.save("outputs/runs/<name>.npz")

    ``record`` is cheap when no frame is due (one comparison).  If the step is
    longer than the frame period, the state is repeated so the frame clock never
    falls behind simulation time.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        fps: float = DEFAULT_FPS,
        model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = float(fps)
        self.nq, self.nv, self.na, self.nu = model.nq, model.nv, model.na, model.nu
        self.timestep = float(model.opt.timestep)
        self.model_path = str(Path(model_path).resolve()) if model_path else ""
        self.model_sha256 = _sha256(model_path) if model_path else ""
        self._t0: float | None = None
        self._k = 0  # index of the next frame due
        self.t: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self._overlays: dict[str, list[float]] = {}

    def __len__(self) -> int:
        return len(self.t)

    def record(self, t: float, data: mujoco.MjData, overlays: Mapping[str, float] | None = None) -> int:
        """Store a snapshot if a frame is due at time ``t``; return frames stored."""
        if self._t0 is None:
            self._t0 = float(t)
        n = 0
        while t >= self._t0 + self._k / self.fps - _TIME_EPS:
            self.t.append(float(t))
            self.qpos.append(data.qpos.copy())
            self.qvel.append(data.qvel.copy())
            self.act.append(data.act.copy())
            self.ctrl.append(data.ctrl.copy())
            index = len(self.t) - 1
            for name, series in self._overlays.items():  # pad known series
                series.append(math.nan)
            for name, value in (overlays or {}).items():
                series = self._overlays.setdefault(name, [math.nan] * (index + 1))
                series[index] = float(value)
            self._k += 1
            n += 1
        return n

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the recording to ``path`` (.npz, compressed) and return the path."""
        if not self.t:
            raise ValueError("nothing recorded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def stack(rows: list[np.ndarray], width: int) -> np.ndarray:
            return np.asarray(rows, dtype=np.float64).reshape(len(rows), width)

        arrays: dict[str, Any] = {
            "t": np.asarray(self.t, dtype=np.float64),
            "qpos": stack(self.qpos, self.nq),
            "qvel": stack(self.qvel, self.nv),
            "act": stack(self.act, self.na),
            "ctrl": stack(self.ctrl, self.nu),
            "fps": np.float64(self.fps),
            "timestep": np.float64(self.timestep),
            "model_path": np.str_(self.model_path),
            "model_sha256": np.str_(self.model_sha256),
            "overlay_names": np.asarray(sorted(self._overlays), dtype=np.str_),
        }
        for name, series in self._overlays.items():
            arrays[f"ov_{name}"] = np.asarray(series, dtype=np.float64)
        np.savez_compressed(path, **arrays)
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def load_recording(path: str | os.PathLike[str]) -> Recording:
    """Read a recording written by :meth:`FrameRecorder.save`."""
    with np.load(path, allow_pickle=False) as z:
        names = [str(n) for n in z["overlay_names"]] if "overlay_names" in z else []
        return Recording(
            t=np.array(z["t"]),
            qpos=np.array(z["qpos"]),
            qvel=np.array(z["qvel"]),
            act=np.array(z["act"]),
            ctrl=np.array(z["ctrl"]) if "ctrl" in z else np.zeros((len(z["t"]), 0)),
            fps=float(z["fps"]),
            model_path=str(z["model_path"]),
            model_sha256=str(z["model_sha256"]) if "model_sha256" in z else "",
            timestep=float(z["timestep"]) if "timestep" in z else math.nan,
            overlays={n: np.array(z[f"ov_{n}"]) for n in names},
        )


# --------------------------------------------------------------------------- cameras


def _free_camera(spec: Mapping[str, Any]) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = spec["lookat"]
    cam.distance = float(spec["distance"])
    cam.azimuth = float(spec["azimuth"])
    cam.elevation = float(spec["elevation"])
    return cam


def resolve_camera(model: mujoco.MjModel, camera: str | int | Mapping[str, Any]) -> Any:
    """Camera argument for ``Renderer.update_scene``.

    ``camera`` is a model camera name or id, a free-camera dict with keys
    lookat/distance/azimuth/elevation, ``"preset:<name>"`` for a
    :data:`CAMERA_PRESETS` entry, or a preset name the model lacks (then the
    preset free camera is used, with a warning).  A model camera wins over a
    preset of the same name.
    """
    if isinstance(camera, Mapping):
        return _free_camera(camera)
    if isinstance(camera, str) and camera.startswith("preset:"):
        return _free_camera(CAMERA_PRESETS[camera.removeprefix("preset:")])
    if isinstance(camera, (int, np.integer)):
        if not 0 <= camera < model.ncam:
            raise ValueError(f"camera id {camera} out of range (model has {model.ncam})")
        return int(camera)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0:
        return camera
    if camera in CAMERA_PRESETS:
        warnings.warn(f"model has no camera {camera!r}; using the free-camera preset", stacklevel=2)
        return _free_camera(CAMERA_PRESETS[camera])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    raise ValueError(f"unknown camera {camera!r}; model cameras: {names}; presets: {list(CAMERA_PRESETS)}")


def free_camera_to_fixed(
    lookat: Sequence[float], distance: float, azimuth: float, elevation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pose of a fixed MJCF camera that reproduces a free camera.

    Returns ``(pos, xyaxes)`` for ``<camera pos=".." xyaxes=".."/>`` (world frame,
    default fovy 45 deg = the free camera's ``visual/global/fovy`` default).
    """
    az, el = math.radians(azimuth), math.radians(elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(lookat, dtype=float) - distance * forward
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-9:  # looking straight down/up: pick azimuth's right
        right = np.array([math.sin(az), -math.cos(az), 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return pos, np.concatenate([right, up])


# --------------------------------------------------------------------------- text


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
)


def _font(size: int, bold: bool = False) -> Any:
    for candidate in (_BOLD_CANDIDATES if bold else ()) + _FONT_CANDIDATES:
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def _title_card(lines: Sequence[str], width: int, height: int) -> np.ndarray:
    """Plain full-frame card: first line large and bold, the rest smaller, centred."""
    img = Image.new("RGB", (width, height), (16, 18, 22))
    draw = ImageDraw.Draw(img)
    fonts = [_font(max(12, height // 14), bold=True)] + [_font(max(10, height // 30))] * (len(lines) - 1)
    colors = [(240, 240, 240)] + [(175, 180, 188)] * (len(lines) - 1)
    heights = [font.size * (1.9 if n == 0 else 1.5) for n, font in enumerate(fonts)]
    y = (height - sum(heights)) / 2
    for line, font, color, h in zip(lines, fonts, colors, heights):
        draw.text((width / 2, y + h / 2), line, font=font, fill=color, anchor="mm")
        y += h
    return np.asarray(img, dtype=np.uint8)


class _Caption:
    """Small translucent strip in the bottom-left corner."""

    def __init__(self, template: str, height: int) -> None:
        self.template = template
        self.font = _font(max(10, height // 42))
        self.pad = max(4, height // 120)

    def apply(self, frame: np.ndarray, fields: Mapping[str, Any]) -> np.ndarray:
        text = self.template.format(**fields)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=self.font)
        w, h = int(right - left) + 2 * self.pad, int(bottom - top) + 2 * self.pad
        H = frame.shape[0]
        x0, y0 = self.pad, H - h - self.pad
        region = frame[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
        box = Image.fromarray((0.45 * region).astype(np.uint8))  # darken 55 %
        ImageDraw.Draw(box).multiline_text(
            (self.pad - left, self.pad - top), text, font=self.font, fill=(235, 235, 235)
        )
        out = frame.copy()
        out[y0 : y0 + h, x0 : x0 + w] = np.asarray(box)
        return out


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(tile)
    draw = ImageDraw.Draw(img)
    font = _font(max(10, tile.shape[0] // 14))
    draw.rectangle(draw.textbbox((4, 2), text, font=font), fill=(0, 0, 0))
    draw.text((4, 2), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


# --------------------------------------------------------------------------- ffmpeg


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Decode the whole file and report what it holds (frames, container duration, fps)."""
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        check=False,
    ).stderr.decode("utf-8", "replace")
    frames = re.findall(r"frame=\s*(\d+)", out)
    dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", out)
    size = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out)
    return {
        "frames": int(frames[-1]) if frames else -1,
        "duration": (int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3])) if dur else math.nan,
        "fps": float(fps[1]) if fps else math.nan,
        "size": (int(size[1]), int(size[2])) if size else None,
    }


def _decode(path: Path, index: int, width: int, height: int) -> np.ndarray | None:
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-vf", f"select=eq(n\\,{index})", "-fps_mode", "passthrough", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=False,
    )
    n = width * height * 3
    if out.returncode != 0 or len(out.stdout) < n:
        return None
    return np.frombuffer(out.stdout[:n], dtype=np.uint8).reshape(height, width, 3)


class _Encoder:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int, preset: str) -> None:
        rate = str(Fraction(fps).limit_denominator(1001))
        self.path = path
        self.frames = 0
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", rate,
             "-i", "-", "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-r", rate, "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as err:
            self.proc.wait()
            raise RuntimeError(f"ffmpeg died after {self.frames} frames: {self.proc.stderr.read().decode()}") from err
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace").strip()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited {self.proc.returncode} after {self.frames} frames: {err}")


# --------------------------------------------------------------------------- rendering

#: Trail colour (bright cyan) and line width in pixels, drawn over a slightly wider
#: dark outline (1 mm lower) so it stays legible on the yellow plate after yuv420p.
TRAIL_RGBA = (0.0, 0.95, 1.0, 1.0)
TRAIL_WIDTH_PX = 3.0
TRAIL_OUTLINE_RGBA = (0.02, 0.05, 0.08, 1.0)
TRAIL_OUTLINE_PX = 5.0
_TRAIL_MIN_STEP = 1e-3  # m: consecutive trail points closer than this in XY are merged
_TRAIL_CLEARANCE = 3e-3  # m above the highest payload geom top


def _prepare_model(model_path: str | os.PathLike[str], width: int, height: int) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    model.vis.quality.offsamples = max(int(model.vis.quality.offsamples), 8)  # MSAA
    return model


def _set_state(model: mujoco.MjModel, data: mujoco.MjData, rec: Recording, k: int) -> None:
    data.qpos[:] = rec.qpos[k]
    data.qvel[:] = rec.qvel[k]
    data.act[:] = rec.act[k]
    if rec.ctrl.shape[1] == model.nu:
        data.ctrl[:] = rec.ctrl[k]
    data.time = rec.t[k]
    mujoco.mj_forward(model, data)


class _Trail:
    """Recorded payload path, drawn into the scene as connector line geoms.

    The points are the payload_free XY positions of every recorded frame up to the
    current one.  They are drawn at the CURRENT payload height plus the payload's
    top-geom offset: the whole circle lies inside the plate's footprint, so points
    drawn at their own (lower) heights would be hidden under the plate whenever
    the Z bob has it higher.  In the top view this changes the projected position
    of a past point by at most |x| * dz / depth (~3 px for 0.15 m, 0.08 m, 2.3 m);
    the head of the trail coincides with the plate centre exactly.
    """

    def __init__(self, model: mujoco.MjModel, rec: Recording) -> None:
        adr = model.jnt_qposadr[model.joint("payload_free").id]
        self.xyz = rec.qpos[:, adr : adr + 3].copy()
        body = model.body("payload").id
        tops = [
            model.geom_pos[g][2] + (model.geom_size[g][2] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                                    else model.geom_rbound[g])
            for g in range(model.ngeom) if model.geom_bodyid[g] == body
        ]
        self.lift = (max(tops) if tops else 0.0) + _TRAIL_CLEARANCE
        keep, last = [0], self.xyz[0, :2]
        for k in range(1, len(self.xyz)):
            if np.linalg.norm(self.xyz[k, :2] - last) >= _TRAIL_MIN_STEP:
                keep.append(k)
                last = self.xyz[k, :2]
        self.keep = np.asarray(keep)
        self._warned = False

    def points(self, k: int) -> np.ndarray:
        idx = self.keep[: np.searchsorted(self.keep, k, side="right")]
        if not len(idx) or idx[-1] != k:
            idx = np.append(idx, k)
        pts = self.xyz[idx].copy()
        pts[:, 2] = self.xyz[k, 2] + self.lift
        return pts

    def draw(self, scene: mujoco.MjvScene, k: int) -> None:
        pts = self.points(k)
        below = pts - np.array([0.0, 0.0, 1e-3])
        for line, rgba, width in ((below, TRAIL_OUTLINE_RGBA, TRAIL_OUTLINE_PX), (pts, TRAIL_RGBA, TRAIL_WIDTH_PX)):
            rgba = np.asarray(rgba, dtype=np.float32)
            for a, b in zip(line[:-1], line[1:]):
                if scene.ngeom >= scene.maxgeom:
                    if not self._warned:
                        warnings.warn("scene geom buffer full; trail truncated", stacklevel=2)
                        self._warned = True
                    return
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)
                scene.ngeom += 1


def _camera_label(spec: Any) -> str:
    if isinstance(spec, Mapping):
        return str(spec.get("name", "free"))
    return str(spec)


class _Compositor:
    """Renders one frame as ``len(cameras)`` horizontal tiles of equal width.

    A camera entry is anything :func:`resolve_camera` takes, or a dict with
    ``"name"`` (model camera / preset) and optional ``"fovy"`` (deg); a free-camera
    dict may also carry ``"fovy"``.  The trail is drawn in ``trail_tiles``.
    """

    SEPARATOR_RGB = (20, 22, 26)

    def __init__(self, model: mujoco.MjModel, cameras: Sequence[Any], width: int, height: int,
                 trail: _Trail | None = None, trail_tiles: Sequence[int] = ()) -> None:
        n = len(cameras)
        if n < 1 or width % n:
            raise ValueError(f"width {width} is not divisible into {n} tiles")
        self.model, self.width, self.height, self.tile_w = model, width, height, width // n
        self.tiles = []
        for spec in cameras:
            fovy = spec.get("fovy") if isinstance(spec, Mapping) else None
            target = spec["name"] if isinstance(spec, Mapping) and "name" in spec else spec
            cam = resolve_camera(model, target)
            if isinstance(cam, str):
                cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            self.tiles.append((cam, None if fovy is None else float(fovy)))
        self.trail, self.trail_tiles = trail, set(trail_tiles)
        self.renderer = mujoco.Renderer(model, height, self.tile_w)

    def render(self, data: mujoco.MjData, k: int = 0) -> np.ndarray:
        out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        for n, (cam, fovy) in enumerate(self.tiles):
            restore = None
            if fovy is not None:
                if isinstance(cam, int):
                    restore = ("cam", float(self.model.cam_fovy[cam]))
                    self.model.cam_fovy[cam] = fovy
                else:
                    restore = ("vis", float(self.model.vis.global_.fovy))
                    self.model.vis.global_.fovy = fovy
            self.renderer.update_scene(data, camera=cam)
            if self.trail is not None and n in self.trail_tiles:
                self.trail.draw(self.renderer.scene, k)
            out[:, n * self.tile_w : (n + 1) * self.tile_w] = self.renderer.render()
            if restore is not None:
                if restore[0] == "cam":
                    self.model.cam_fovy[cam] = restore[1]
                else:
                    self.model.vis.global_.fovy = restore[1]
        for n in range(1, len(self.tiles)):
            out[:, n * self.tile_w - 1 : n * self.tile_w + 1] = self.SEPARATOR_RGB
        return out

    def close(self) -> None:
        self.renderer.close()


def _as_camera_list(camera: Any, width: int, height: int) -> list[Any]:
    """Single camera -> [camera]; a list -> tiles, plain names replaced by TILE_CAMERAS."""
    if not isinstance(camera, (list, tuple)):
        return [camera]
    cameras = [TILE_CAMERAS.get(c, c) if isinstance(c, str) else c for c in camera]
    aspect = (width / len(cameras)) / height
    if any(isinstance(c, str) and c in TILE_CAMERAS for c in camera) and abs(aspect / _TILE_ASPECT - 1) > 0.1:
        warnings.warn(f"TILE_CAMERAS are tuned for 8:9 tiles; these tiles are {aspect:.2f}:1 and may clip", stacklevel=3)
    return cameras


def _default_trail_tiles(cameras: Sequence[Any], trail_tiles: Sequence[int] | None) -> list[int]:
    if trail_tiles is not None:
        return list(trail_tiles)
    tops = [n for n, c in enumerate(cameras) if _camera_label(c) in ("top", "preset:top")]
    if not tops:
        raise ValueError("trail=True needs a 'top' tile, or pass trail_tiles explicitly")
    return tops


def render_still(
    model_path: str | os.PathLike[str],
    out_png: str | os.PathLike[str],
    camera: Any = "overview",
    width: int = 1280,
    height: int = 720,
    qpos: np.ndarray | None = None,
    keyframe: int | str | None = None,
    record: str | os.PathLike[str] | None = None,
    frame: int | None = None,
    trail: bool = False,
    trail_tiles: Sequence[int] | None = None,
) -> np.ndarray:
    """Render one image: default config, a keyframe, given qpos, or recorded ``frame``
    of ``record`` (negative counts from the end).  ``camera`` may be a list (tiles);
    ``trail`` needs ``record``."""
    model = _prepare_model(model_path, width, height)
    data = mujoco.MjData(model)
    cameras = _as_camera_list(camera, width, height)
    rec = load_recording(record) if record is not None else None
    k = 0
    if rec is not None:
        k = (frame or 0) % rec.n_frames
        _set_state(model, data, rec, k)
    else:
        if keyframe is not None:
            key = keyframe if isinstance(keyframe, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            mujoco.mj_resetDataKeyframe(model, data, key)
        if qpos is not None:
            data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
    if trail and rec is None:
        raise ValueError("trail needs a recording")
    comp = _Compositor(model, cameras, width, height, _Trail(model, rec) if trail else None,
                       _default_trail_tiles(cameras, trail_tiles) if trail else ())
    try:
        pixels = comp.render(data, k)
    finally:
        comp.close()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if HAVE_PIL:
        Image.fromarray(pixels).save(out_png)
    else:  # pragma: no cover
        warnings.warn("PIL unavailable; still not written", stacklevel=2)
    return pixels


def _frame_schedule(rec: Recording, fps: float) -> np.ndarray:
    """Recorded-frame index for each video frame; identity when fps matches."""
    if abs(fps - rec.fps) < 1e-9:
        return np.arange(rec.n_frames)
    n_video = int(math.floor(rec.duration * fps + 1e-9))
    times = rec.t[0] + np.arange(n_video) / fps
    idx = np.clip(np.searchsorted(rec.t, times), 0, rec.n_frames - 1)
    prev = np.clip(idx - 1, 0, rec.n_frames - 1)
    take_prev = np.abs(rec.t[prev] - times) <= np.abs(rec.t[idx] - times)
    return np.where(take_prev, prev, idx)


def render_video(
    model_path: str | os.PathLike[str],
    record_npz: str | os.PathLike[str],
    out_mp4: str | os.PathLike[str],
    camera: Any = "overview",
    width: int = 1280,
    height: int = 720,
    fps: float | None = None,
    title_card: str | Sequence[str] | None = None,
    title_seconds: float = 2.5,
    caption: str | None = None,
    crf: int = 18,
    preset: str = "medium",
    contact_sheet: bool = True,
    trail: bool = False,
    trail_tiles: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Render a recording to MP4 in real time and verify the result.

    Args:
        camera: model camera name/id, free-camera dict, ``"preset:<name>"``, or a
            LIST of these for side-by-side tiles (``width`` is the total width and
            is split equally).  In a list, the plain names in :data:`TILE_CAMERAS`
            ("overview", "top") are replaced by their tile-tuned variants; any entry
            may be a dict ``{"name": ..., "fovy": ...}`` (fovy override, deg).
        fps: video rate; ``None`` uses the recording's rate.  A different rate is
            reached by nearest-time frame selection, never by changing playback
            speed, so the video always lasts as long as the simulation.
        title_card: text lines (or one string with newlines) for a plain full-frame
            card shown for ``title_seconds`` before the simulation.  Needs PIL.
        caption: small bottom-left strip, a ``str.format`` template with fields
            ``t`` (seconds since the first frame), ``t_abs`` and overlay names,
            e.g. ``"t = {t:5.2f} s"``.  Needs PIL.
        trail: draw the payload's recorded path (see :class:`_Trail`) in the
            ``trail_tiles`` (default: the tile(s) whose camera is ``top``).

    Returns:
        Report dict (also written as ``<out>.json``); ``report["verified"]`` is
        False and ``report["problems"]`` lists every mismatch found.

    Raises:
        RuntimeError: ffmpeg failed.  ValueError: recording does not fit the model.
    """
    if width % 2 or height % 2:
        raise ValueError(f"yuv420p needs even dimensions, got {width}x{height}")
    rec = load_recording(record_npz)
    cameras = _as_camera_list(camera, width, height)
    model = _prepare_model(model_path, width // len(cameras), height)
    for name, n_model, arr in (("qpos", model.nq, rec.qpos), ("qvel", model.nv, rec.qvel), ("act", model.na, rec.act)):
        if arr.shape[1] != n_model:
            raise ValueError(f"recording {name} has width {arr.shape[1]} but the model has {n_model}")
    problems: list[str] = []
    sha = _sha256(model_path)
    if rec.model_sha256 and sha != rec.model_sha256:
        msg = f"model file differs from the one recorded ({rec.model_path})"
        warnings.warn(msg, stacklevel=2)
        problems.append(msg)
    fps = rec.fps if fps is None else float(fps)
    schedule = _frame_schedule(rec, fps)
    tiles_with_trail = _default_trail_tiles(cameras, trail_tiles) if trail else []

    lines = [title_card] if isinstance(title_card, str) else list(title_card or [])
    lines = [piece for line in lines for piece in line.split("\n")]
    if (lines or caption) and not HAVE_PIL:
        warnings.warn("PIL is not available: title card and caption are skipped", stacklevel=2)
        lines, caption = [], None
    n_title = int(round(title_seconds * fps)) if lines else 0
    strip = _Caption(caption, height) if caption else None
    sheet_at = set(np.linspace(0, len(schedule) - 1, 9).round().astype(int).tolist())
    check_at = sorted({0, len(schedule) // 2, len(schedule) - 1})
    tiles: dict[int, np.ndarray] = {}
    sent: dict[int, np.ndarray] = {}

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    partial = out_mp4.with_name(out_mp4.stem + ".partial.mp4")
    data = mujoco.MjData(model)
    comp = _Compositor(model, cameras, width, height, _Trail(model, rec) if trail else None, tiles_with_trail)
    encoder = _Encoder(partial, width, height, fps, crf, preset)
    try:
        if n_title:
            card = _title_card(lines, width, height)
            for _ in range(n_title):
                encoder.write(card)
        for j, k in enumerate(schedule):
            _set_state(model, data, rec, k)
            frame = comp.render(data, k)
            if strip is not None:
                fields = {name: float(v[k]) for name, v in rec.overlays.items()}
                fields.update(t=float(rec.t[k] - rec.t[0]), t_abs=float(rec.t[k]))
                frame = strip.apply(frame, fields)
            encoder.write(frame)
            if j in sheet_at and HAVE_PIL:
                tiles[j] = np.asarray(Image.fromarray(frame).resize((width // 3, height // 3), Image.LANCZOS))
            if j in check_at:
                sent[n_title + j] = frame.copy()
    finally:
        encoder.close()
        comp.close()

    # ---- verification: decode what was written, compare with what was sent
    n_expected = n_title + len(schedule)
    probe = probe_video(partial)
    if probe["frames"] != n_expected:
        problems.append(f"decoded {probe['frames']} frames, sent {n_expected}")
    if not abs(probe["duration"] - n_expected / fps) <= 1.0 / fps + 0.02:
        problems.append(f"container duration {probe['duration']:.3f} s, expected {n_expected / fps:.3f} s")
    if probe["size"] != (width, height):
        problems.append(f"video size {probe['size']}, expected {(width, height)}")
    pixel_mae: dict[int, float] = {}
    for index, expected in sent.items():
        decoded = _decode(partial, index, width, height)
        if decoded is None:
            problems.append(f"frame {index} could not be decoded")
            continue
        pixel_mae[index] = float(np.mean(np.abs(decoded.astype(np.int16) - expected.astype(np.int16))))
        if pixel_mae[index] > 4.0:
            problems.append(f"frame {index}: decoded differs from rendered (MAE {pixel_mae[index]:.2f}/255)")
        if expected.std() < 2.0:
            problems.append(f"frame {index} is nearly uniform (std {expected.std():.2f}); blank render?")

    sim_duration = float(rec.t[schedule[-1]] - rec.t[schedule[0]]) + 1.0 / fps
    frame_times = rec.t[schedule] - rec.t[schedule[0]]
    report: dict[str, Any] = {
        "video": str(out_mp4),
        "verified": not problems,
        "problems": problems,
        "record": str(Path(record_npz).resolve()),
        "model": str(Path(model_path).resolve()),
        "model_sha256": sha,
        "camera": [c if isinstance(c, (str, int)) else dict(c) for c in cameras],
        "trail_tiles": tiles_with_trail,
        "width": width,
        "height": height,
        "fps": fps,
        "crf": crf,
        "title_frames": n_title,
        "sim_frames": len(schedule),
        "sim_duration_s": sim_duration,
        "video_duration_s": n_expected / fps,
        "max_frame_time_error_s": float(np.max(np.abs(frame_times - np.arange(len(schedule)) / fps))),
        "probe": probe,
        "decoded_pixel_mae": pixel_mae,
    }
    final = out_mp4 if not problems else out_mp4.with_name(out_mp4.stem + ".UNVERIFIED.mp4")
    os.replace(partial, final)
    report["video"] = str(final)
    if contact_sheet and HAVE_PIL and tiles:
        order = sorted(tiles)
        labelled = [_label(tiles[j], f"t = {frame_times[j]:.2f} s") for j in order]
        th, tw = labelled[0].shape[:2]
        grid = np.zeros((3 * th, 3 * tw, 3), dtype=np.uint8)
        for n, tile in enumerate(labelled[:9]):
            grid[(n // 3) * th : (n // 3 + 1) * th, (n % 3) * tw : (n % 3 + 1) * tw] = tile
        sheet = final.with_name(final.stem + "_contact.png")
        Image.fromarray(grid).save(sheet)
        report["contact_sheet"] = str(sheet)
    final.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    return report
