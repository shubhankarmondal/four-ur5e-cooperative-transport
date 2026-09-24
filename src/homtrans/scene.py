"""Four UR5e + Robotiq 2F-85 arms around one plate: the MuJoCo model builder.

The arm and gripper are the MuJoCo Menagerie models in ``assets/``; the only
changes made to them are

* the UR5e position servos are replaced by torque actuators (ctrl = N·m,
  ctrlrange = forcerange = ±(150,150,150,28,28,28)), because the cooperative
  controller computes joint torques itself;
* a small viscous joint damping is added to the arm joints (numerical hygiene
  for the implicit integrator; the controller supplies the real damping);
* the gripper keeps its Menagerie tendon position actuator unchanged
  (ctrl 0 = open, 255 = closed).

Names and frames (SI units, radians, world z up, gravity (0, 0, -9.81)):

* Every element of arm i (i = 1..4, UR5e and its 2F-85) carries the prefix
  ``ur{i}_``: joints ``ur{i}_shoulder_pan_joint`` .. ``ur{i}_wrist_3_joint``,
  torque actuators ``ur{i}_shoulder_pan`` .. ``ur{i}_wrist_3``, gripper actuator
  ``ur{i}_fingers_actuator``, base body ``ur{i}_base`` (on the floor, facing the
  plate centre).
* ``ur{i}_pinch``: the 2F-85 pinch point between the pads; +z is the approach
  direction (out of the gripper), +-y the finger closing axis.
* ``payload``: free body (joint ``payload_free``, the only free joint), frame at
  the plate centre and aligned with the world at rest, site ``payload_center``
  at its origin.  Nothing welds it to the world or to a robot: it is carried by
  the gripper pads alone.
* ``handle_{i}``: payload sites whose frame IS the desired ``ur{i}_pinch`` frame
  when grasped (see :func:`handle_frame`).
* ``stand``: static geom the plate rests on before lift-off.  Cameras
  ``overview``, ``top`` and ``closeup``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
UR5E_XML = ASSETS / "ur5e" / "ur5e.xml"
GRIPPER_XML = ASSETS / "robotiq_2f85" / "2f85.xml"
MODEL_XML = ROOT / "models" / "four_ur5e_transport.xml"

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
N_ARMS = 4


@dataclass(frozen=True)
class SceneParams:
    """Geometry and physics of the demonstration (all SI)."""

    base_distance: float = 0.90  # arm base centre to plate centre, horizontal
    plate_half: float = 0.25  # plate half side (square plate)
    plate_thickness: float = 0.012
    plate_mass: float = 0.80
    handle_length: float = 0.066  # radial protrusion (pads on; 1.2 mm clear of the spring links)
    handle_width: float = 0.050
    handle_thickness: float = 0.030  # the jaws close across this
    handle_mass: float = 0.05
    rest_height: float = 0.40  # plate centre height when resting on the stand
    stand_half: float = 0.05  # narrow post: never under a gripper during the circle
    arm_joint_damping: tuple[float, ...] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5)
    timestep: float = 0.001
    handle_friction: float = 1.0
    gripper_force_range: float = 5.0  # Menagerie default tendon force range [N]
    # Radius of the handle_{i} sites (= desired pinch point).  The open pads span
    # [-4.5, +33] mm radially outward of the pinch point, so any radius in
    # [plate_half + 0.0045, plate_half + handle_length - 0.033] keeps both pads fully
    # on the handle.  Chosen so the closed grasp settles at the arm target (offset
    # < 0.3 mm, scripts/diagnostics/calibrate_grasp.py): closing does not squeeze.
    grasp_site_radius: float = 0.285
    extra: dict = field(default_factory=dict)

    @property
    def handle_centre_radius(self) -> float:
        """Distance from plate centre to each handle box centre."""
        return self.plate_half + 0.5 * self.handle_length


def outward(i: int) -> np.ndarray:
    """Unit vector from the plate centre toward arm ``i`` (1..4) in the world."""
    angle = 0.5 * np.pi * (i - 1)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def handle_frame(i: int) -> np.ndarray:
    """Rotation of site ``handle_{i}`` in the payload frame.

    +z points from the handle toward the plate centre, +y is up (the jaws close
    vertically across the handle thickness), +x completes a right-handed frame.
    """
    z = -outward(i)
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def _absolute_mesh_paths(spec: mujoco.MjSpec, xml: Path) -> None:
    meshdir = xml.parent / (spec.meshdir or "")
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((meshdir / mesh.file).resolve())
    spec.meshdir = ""


def _arm_with_gripper(params: SceneParams) -> mujoco.MjSpec:
    """One UR5e with its 2F-85, torque-actuated, names unprefixed."""
    arm = mujoco.MjSpec.from_file(str(UR5E_XML))
    _absolute_mesh_paths(arm, UR5E_XML)
    for light in list(arm.lights):
        arm.delete(light)

    for name, limit in zip(ARM_ACTUATORS, TORQUE_LIMITS, strict=True):
        act = arm.actuator(name)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[:] = 0.0
        act.gainprm[0] = 1.0
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.biasprm[:] = 0.0
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-limit, limit]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-limit, limit]
    for name, damping in zip(ARM_JOINTS, params.arm_joint_damping, strict=True):
        arm.joint(name).damping = [damping, 0.0, 0.0]  # [linear, 0, 0]

    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    _absolute_mesh_paths(grip, GRIPPER_XML)
    grip.actuator("fingers_actuator").forcerange = [
        -params.gripper_force_range,
        params.gripper_force_range,
    ]
    arm.option.cone = grip.option.cone  # same options: no attach-conflict warning
    arm.option.impratio = grip.option.impratio
    arm.attach(grip, prefix="g_", site=arm.site("attachment_site"))
    # without the gripper prefix: ur{i}_pinch, ur{i}_fingers_actuator
    arm.site("g_pinch").name = "pinch"
    arm.actuator("g_fingers_actuator").name = "fingers_actuator"
    return arm


def _add_payload(spec: mujoco.MjSpec, params: SceneParams) -> None:
    body = spec.worldbody.add_body(name="payload", pos=[0.0, 0.0, params.rest_height])
    body.add_freejoint(name="payload_free")
    a, t = params.plate_half, params.plate_thickness
    body.add_geom(
        name="plate",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[a, a, 0.5 * t],
        mass=params.plate_mass,
        rgba=[0.85, 0.55, 0.20, 1.0],
        friction=[0.8, 0.02, 0.001],
    )
    body.add_site(name="payload_center", size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 0.5])
    L, w, h = params.handle_length, params.handle_width, params.handle_thickness
    for i in range(1, N_ARMS + 1):
        u = outward(i)
        centre = u * params.handle_centre_radius
        yaw = np.arctan2(u[1], u[0])
        R_h = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        )
        body.add_geom(
            name=f"handle_{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * L, 0.5 * w, 0.5 * h],
            pos=centre,
            quat=_quat(R_h),
            mass=params.handle_mass,
            rgba=[0.25, 0.25, 0.28, 1.0],
            friction=[params.handle_friction, 0.02, 0.001],
            priority=2,
            solref=[0.004, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )
        body.add_site(
            name=f"handle_{i}",
            pos=u * params.grasp_site_radius,
            quat=_quat(handle_frame(i)),
            size=[0.006, 0.006, 0.006],
            rgba=[0.1, 0.9, 0.1, 0.6],
        )


def build_spec(params: SceneParams | None = None) -> mujoco.MjSpec:
    params = params or SceneParams()
    spec = mujoco.MjSpec()
    spec.modelname = "four_ur5e_transport"
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = params.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    ground = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    world = spec.worldbody
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane"
    )
    world.add_light(pos=[0, 0, 3.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.5, 0.5, 0.5])
    world.add_light(pos=[1.5, -1.5, 2.5], dir=[-0.5, 0.5, -0.7], diffuse=[0.4, 0.4, 0.4])

    stand_top = params.rest_height - 0.5 * params.plate_thickness
    world.add_geom(
        name="stand",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[params.stand_half, params.stand_half, 0.5 * stand_top],
        pos=[0, 0, 0.5 * stand_top],
        rgba=[0.55, 0.55, 0.6, 1.0],
        friction=[0.8, 0.02, 0.001],
    )

    # render.CAMERA_PRESETS converted with render.free_camera_to_fixed
    world.add_camera(name="overview", pos=[1.3033, -1.3033, 1.5105],
                     xyaxes=[0.7071, 0.7071, 0.0, -0.4056, 0.4056, 0.8192])
    world.add_camera(name="top", pos=[0.0, 0.0, 2.75], fovy=51.0,  # 51: no clipping at 16:9
                     xyaxes=[0.7071, -0.7071, 0.0, 0.7071, 0.7071, 0.0])
    world.add_camera(name="closeup", pos=[0.3934, -0.5298, 0.5244],
                     xyaxes=[0.9848, 0.1736, 0.0, -0.0361, 0.2048, 0.9781])

    _add_payload(spec, params)

    for i in range(1, N_ARMS + 1):
        u = outward(i)
        yaw = np.arctan2(-u[1], -u[0])  # base faces the plate centre
        frame = world.add_frame(
            pos=u * params.base_distance,
            quat=[np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
        )
        spec.attach(_arm_with_gripper(params), prefix=f"ur{i}_", frame=frame)
    return spec


def build_model(params: SceneParams | None = None) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_spec(params)
    return spec.compile(), spec


def portable_xml(spec: mujoco.MjSpec, path: Path = MODEL_XML) -> str:
    """MJCF text of ``spec`` with mesh files relative to ``path``'s directory."""
    root = str(ROOT.resolve())
    xml = spec.to_xml().replace(f'file="{root}/', f'file="{os.path.relpath(root, path.parent)}/')
    if root in xml:
        raise RuntimeError("absolute repository path left in the exported MJCF")
    return xml


def write_model_xml(params: SceneParams | None = None, path: Path = MODEL_XML) -> Path:
    """Compile, export to MJCF and check the export reloads to the same model."""
    model, spec = build_model(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(portable_xml(spec, path))
    reloaded = mujoco.MjModel.from_xml_path(str(path))
    for attr in ("nq", "nv", "nu", "nbody", "ngeom", "neq", "ntendon", "nsite"):
        if getattr(model, attr) != getattr(reloaded, attr):
            raise RuntimeError(f"MJCF export changed {attr}: {getattr(model, attr)} -> "
                               f"{getattr(reloaded, attr)}")
    if not np.allclose(model.body_mass, reloaded.body_mass, atol=1e-9):
        raise RuntimeError("MJCF export changed body masses")
    if not np.allclose(model.actuator_gainprm, reloaded.actuator_gainprm):
        raise RuntimeError("MJCF export changed actuator gains")
    return path
