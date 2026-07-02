"""Tests for differentiable soft collision detection in MJX."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import smooth
import numpy as np


_PLANE_BOX_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PLANE_SPHERE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PLANE_CAPSULE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.2" euler="30 0 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PLANE_ELLIPSOID_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.15" euler="20 10 0">
      <freejoint/>
      <geom type="ellipsoid" size="0.1 0.08 0.06" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PLANE_CYLINDER_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.2" euler="15 0 0">
      <freejoint/>
      <geom type="cylinder" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PLANE_CYLINDER_PARALLEL_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.12" euler="0 90 0">
      <freejoint/>
      <geom type="cylinder" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_SPHERE_SPHERE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body pos="0.15 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_SPHERE_SPHERE_COINCIDENT_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_SPHERE_CAPSULE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body pos="0.12 0 0.5">
      <freejoint/>
      <geom type="capsule" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_CAPSULE_CAPSULE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5" euler="0 90 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
    <body pos="0 0 0.62" euler="90 0 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_CAPSULE_CAPSULE_PARALLEL_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5" euler="0 90 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
    <body pos="0 0.12 0.5" euler="0 90 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_TEST_CASES = (
    ('plane_box', _PLANE_BOX_XML, False),
    ('plane_sphere', _PLANE_SPHERE_XML, False),
    ('plane_capsule', _PLANE_CAPSULE_XML, False),
    ('plane_ellipsoid', _PLANE_ELLIPSOID_XML, False),
    ('plane_cylinder', _PLANE_CYLINDER_XML, False),
    ('plane_cylinder_parallel', _PLANE_CYLINDER_PARALLEL_XML, False),
    ('sphere_sphere', _SPHERE_SPHERE_XML, False),
    ('sphere_sphere_coincident', _SPHERE_SPHERE_COINCIDENT_XML, True),
    ('sphere_capsule', _SPHERE_CAPSULE_XML, False),
    ('capsule_capsule', _CAPSULE_CAPSULE_XML, False),
    ('capsule_capsule_parallel', _CAPSULE_CAPSULE_PARALLEL_XML, False),
)


class SoftCollisionTest(parameterized.TestCase):

  @parameterized.parameters(*_TEST_CASES)
  def test_soft_collision_dist_and_gradient(
      self, name, xml, expect_zero_grad
  ):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    for mode in ('hard', 'c2'):
      mx_soft = mx.replace(opt=mx.opt.replace(softjax_mode=mode))
      dx_soft = jax.jit(collision_driver.collision)(mx_soft, dx)
      dist = np.asarray(dx_soft._impl.contact.dist)
      self.assertFalse(
          np.any(np.isnan(dist)),
          f'{name} softjax_mode={mode} produced NaN distance',
      )

    mx_soft = mx.replace(opt=mx.opt.replace(softjax_mode='c2'))

    def soft_collision_loss(qpos):
      dx_mod = dx.replace(qpos=qpos)
      dx_mod = smooth.kinematics(mx_soft, dx_mod)
      dx_mod = smooth.com_pos(mx_soft, dx_mod)
      dx_mod = collision_driver.collision(mx_soft, dx_mod)
      return jp.sum(dx_mod._impl.contact.dist)

    grad = np.asarray(jax.jit(jax.grad(soft_collision_loss))(dx.qpos))
    self.assertFalse(np.any(np.isnan(grad)), f'{name} gradient has NaNs')

    if expect_zero_grad:
      np.testing.assert_allclose(grad, 0)
    else:
      self.assertFalse(np.allclose(grad, 0), f'{name} gradient is all zero')


if __name__ == '__main__':
  absltest.main()
