"""Build models/four_ur5e_transport.xml and print a short model summary.

    pixi run python scripts/build_scene.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402

from homtrans.scene import MODEL_XML, write_model_xml  # noqa: E402


def main() -> int:
    path = write_model_xml()
    model = mujoco.MjModel.from_xml_path(str(path))
    payload = model.body("payload").id
    print(f"wrote {path}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
          f"ngeom={model.ngeom} neq={model.neq} timestep={model.opt.timestep}")
    print(f"payload mass={model.body_subtreemass[payload]:.4f} kg")
    print("actuators:", [model.actuator(k).name for k in range(model.nu)])
    free = [model.joint(j).name for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    print("free joints:", free)
    assert MODEL_XML == path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
