"""Scenes for validating and benchmarking the MJX solver derivative backends.

Analytical scenes are inline MJCF. The robot scenes (Go1, G1) load the
mujoco_playground "feet only" MJX models; their location is taken from
``MJX_PLAYGROUND_PATH`` (the ``mujoco_playground`` package directory).
"""

import dataclasses
import os
import pathlib
from typing import Callable, Optional

import mujoco
import numpy as np

_PENDULUM = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" limited="true"
             range="-170 170" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" gear="1"/></actuator>
</mujoco>
"""

_CARTPOLE = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true"
             range="-2 2" damping="0.1"/>
      <geom type="box" size="0.2 0.1 0.05" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="5"/></actuator>
</mujoco>
"""

_SPHERE_PLANE = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" friction="0.8 0.005 0.0001"/>
    <body pos="0 0 0.0995">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Box resting on a plane: four corner contacts. ``qvel`` / ``qfrc_applied`` in
# the scene definitions select sliding versus sticking friction.
_BOX_PLANE = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" friction="0.5 0.005 0.0001"/>
    <body pos="0 0 0.0995">
      <freejoint/>
      <geom type="box" size="0.2 0.15 0.1" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""

# Small quadruped: free torso, four 2-DoF legs with sphere feet on the ground.
_QUADRUPED = """
<mujoco>
  <option timestep="0.004"/>
  <default>
    <joint damping="0.5" armature="0.01" limited="true" range="-90 90"/>
    <geom friction="0.8 0.005 0.0001" contype="0" conaffinity="0"/>
    <motor ctrlrange="-10 10" ctrllimited="true"/>
  </default>
  <worldbody>
    <geom type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
    <body name="torso" pos="0 0 0.395">
      <freejoint/>
      <geom type="box" size="0.25 0.12 0.05" mass="5"/>
      {legs}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>
"""


def _quadruped_xml() -> str:
  legs, motors = [], []
  for name, x, y in (('fl', 0.2, 0.15), ('fr', 0.2, -0.15),
                     ('hl', -0.2, 0.15), ('hr', -0.2, -0.15)):
    legs.append(f"""
      <body name="{name}_thigh" pos="{x} {y} 0">
        <joint name="{name}_hip" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.5"/>
        <body name="{name}_shin" pos="0 0 -0.15">
          <joint name="{name}_knee" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.02" mass="0.3"/>
          <geom name="{name}_foot" type="sphere" pos="0 0 -0.15" size="0.0305"
                mass="0.05" contype="1" conaffinity="1"/>
        </body>
      </body>""")
    motors.append(f'<motor joint="{name}_hip" gear="1"/>')
    motors.append(f'<motor joint="{name}_knee" gear="1"/>')
  return _QUADRUPED.format(legs=''.join(legs), motors=''.join(motors))


@dataclasses.dataclass(frozen=True)
class Scene:
  """A model plus an initial state."""

  name: str
  model: mujoco.MjModel
  qpos: np.ndarray
  qvel: np.ndarray
  ctrl: np.ndarray
  qfrc_applied: np.ndarray
  data: mujoco.MjData
  description: str = ''


def _configure(
    m: mujoco.MjModel,
    *,
    solver: Optional[int],
    cone: Optional[int],
    iterations: Optional[int],
    ls_iterations: Optional[int],
    tolerance: Optional[float],
    warmstart: bool,
) -> mujoco.MjModel:
  if solver is not None:
    m.opt.solver = solver
  if cone is not None:
    m.opt.cone = cone
  if iterations is not None:
    m.opt.iterations = iterations
  if ls_iterations is not None:
    m.opt.ls_iterations = ls_iterations
  if tolerance is not None:
    m.opt.tolerance = tolerance
  # MJX does not support islanding; keep MuJoCo C comparable.
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
  if not warmstart:
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
  return m


def _settle(m: mujoco.MjModel, d: mujoco.MjData, ctrl_fn, nstep: int):
  """Steps MuJoCo C so contacts are established and warmstart is populated."""
  for _ in range(nstep):
    if ctrl_fn is not None:
      d.ctrl[:] = ctrl_fn(m, d)
    mujoco.mj_step(m, d)


def _playground_path() -> pathlib.Path:
  path = os.environ.get('MJX_PLAYGROUND_PATH')
  if path is None:
    raise RuntimeError(
        'set MJX_PLAYGROUND_PATH to the mujoco_playground package directory '
        'to load the Go1/G1 scenes'
    )
  return pathlib.Path(path)


def _load_assets(dirs) -> dict[str, bytes]:
  assets = {}
  for dir_, pattern in dirs:
    for f in dir_.glob(pattern):
      if f.is_file():
        assets[f.name] = f.read_bytes()
  return assets


def _load_robot(robot: str) -> mujoco.MjModel:
  root = _playground_path()
  menagerie = root / 'external_deps' / 'mujoco_menagerie'
  if robot == 'go1':
    xmls = root / '_src' / 'locomotion' / 'go1' / 'xmls'
    scene = xmls / 'scene_mjx_feetonly_flat_terrain.xml'
    assets = _load_assets([
        (xmls, '*.xml'), (xmls / 'assets', '*'),
        (menagerie / 'unitree_go1', '*.xml'),
        (menagerie / 'unitree_go1' / 'assets', '*'),
    ])
  elif robot == 'g1':
    xmls = root / '_src' / 'locomotion' / 'g1' / 'xmls'
    scene = xmls / 'scene_mjx_feetonly_flat_terrain.xml'
    assets = _load_assets([
        (xmls, '*.xml'), (xmls / 'assets', '*'),
        (menagerie / 'unitree_g1', '*.xml'),
        (menagerie / 'unitree_g1' / 'assets', '*'),
    ])
  else:
    raise ValueError(robot)
  return mujoco.MjModel.from_xml_string(scene.read_text(), assets)


def _pd_hold(kp: float, kd: float):
  """PD controller holding the keyframe joint angles (for settling robots)."""

  def fn(m, d):
    q0 = m.key_qpos[0][7:] if m.nkey else np.zeros(m.nu)
    q, v = d.qpos[7:7 + m.nu], d.qvel[6:6 + m.nu]
    if m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE:
      return q0  # position actuators: target is the control
    return kp * (q0 - q) - kd * v

  return fn


SCENE_NAMES = (
    'pendulum', 'cartpole', 'sphere_plane', 'box_stick', 'box_slide',
    'quadruped', 'go1', 'g1',
)
ANALYTIC_SCENES = SCENE_NAMES[:6]
CONTACT_SCENES = ('sphere_plane', 'box_stick', 'box_slide', 'quadruped',
                  'go1', 'g1')


def make_scene(
    name: str,
    *,
    solver: Optional[int] = None,
    cone: Optional[int] = None,
    iterations: Optional[int] = None,
    ls_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
    warmstart: bool = True,
) -> Scene:
  """Builds a scene; ``None`` options keep the model's own setting."""
  cfg = dict(solver=solver, cone=cone, iterations=iterations,
             ls_iterations=ls_iterations, tolerance=tolerance,
             warmstart=warmstart)
  ctrl_fn: Optional[Callable] = None
  qfrc = None
  nsettle = 0
  key = None
  if name == 'pendulum':
    m = mujoco.MjModel.from_xml_string(_PENDULUM)
    qpos, qvel, desc = [0.7], [1.0], 'hinge, inactive joint limit, no contact'
  elif name == 'cartpole':
    m = mujoco.MjModel.from_xml_string(_CARTPOLE)
    qpos, qvel = [0.3, 0.4], [0.2, -0.5]
    desc = 'slide+hinge, inactive slider limit, no contact'
  elif name == 'sphere_plane':
    m = mujoco.MjModel.from_xml_string(_SPHERE_PLANE)
    qpos, qvel, nsettle = None, None, 50
    desc = 'free sphere resting on a plane, one contact'
  elif name == 'box_stick':
    m = mujoco.MjModel.from_xml_string(_BOX_PLANE)
    qpos, qvel, nsettle = None, None, 50
    qfrc = np.array([2.0, 1.0, 0, 0, 0, 0.3])  # well inside the cone
    desc = 'box on plane, 4 contacts, sticking friction'
  elif name == 'box_slide':
    m = mujoco.MjModel.from_xml_string(_BOX_PLANE)
    qpos, qvel, nsettle = None, None, 50
    desc = 'box on plane, 4 contacts, sliding at 1.5 m/s'
  elif name == 'quadruped':
    m = mujoco.MjModel.from_xml_string(_quadruped_xml())
    qpos, qvel, nsettle = None, None, 100
    ctrl_fn = lambda m, d: -20 * d.qpos[7:] - 1.0 * d.qvel[6:]
    desc = 'free torso + 8 hinges, 4 persistent foot contacts'
  elif name in ('go1', 'g1'):
    m = _load_robot(name)
    qpos, qvel, nsettle, key = None, None, 100, 0
    ctrl_fn = _pd_hold(30.0, 1.0)
    desc = f'mujoco_playground {name} feet-only, flat terrain'
  else:
    raise ValueError(f'unknown scene {name!r}')

  m = _configure(m, **cfg)
  d = mujoco.MjData(m)
  if key is not None and m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, key)
  if qpos is not None:
    d.qpos[:] = qpos
  if qvel is not None:
    d.qvel[:] = qvel
  _settle(m, d, ctrl_fn, nsettle)
  if name == 'box_slide':
    d.qvel[:] = 0
    d.qvel[0] = 1.5
  if qfrc is not None:
    d.qfrc_applied[:] = qfrc
  ctrl = ctrl_fn(m, d) if ctrl_fn is not None else np.zeros(m.nu)
  if name == 'pendulum' or name == 'cartpole':
    ctrl = np.full(m.nu, 0.3)
  d.ctrl[:] = ctrl
  mujoco.mj_forward(m, d)  # consistent efc / warmstart for put_data
  return Scene(
      name=name, model=m, qpos=d.qpos.copy(), qvel=d.qvel.copy(),
      ctrl=np.asarray(ctrl, dtype=float).copy(),
      qfrc_applied=d.qfrc_applied.copy(), data=d, description=desc,
  )
