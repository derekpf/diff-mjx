# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for forward functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import forward
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import test_util
import numpy as np


# tolerance for difference between MuJoCo and MJX forward calculations - mostly
# due to float precision
_TOLERANCE = 1e-5


def _assert_eq(a, b, name, tol=_TOLERANCE):
  tol = tol * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr, tol=_TOLERANCE):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr, tol=tol)


class ForwardTest(absltest.TestCase):

  def test_piecewise_contact_and_straight_through(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option gravity="0 0 0" timestep="0.001"/>
          <worldbody>
            <geom type="plane" size="1 1 .01" margin="0.01"/>
            <body pos="0 0 0.111">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    d = mujoco.MjData(m)
    d.qvel[2] = -1.0
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    precontact_solimp = jp.array([0.9, 0.95, 0.002, 0.5, 2.0])

    def contact_force(precontact_solimp, qpos):
      opt = mx.opt.replace(precontact_solimp=precontact_solimp)
      return mjx.forward(
          mx.replace(opt=opt), dx.replace(qpos=qpos)
      ).qfrc_constraint[2]

    def contact_force_and_grad(precontact_solimp):
      fn = lambda qpos: contact_force(precontact_solimp, qpos)
      return fn(dx.qpos), jax.grad(fn)(dx.qpos)[2]

    # The sphere sits just outside the contact margin: no force, no gradient.
    force, grad = contact_force_and_grad(None)
    self.assertEqual(force, 0.0)
    self.assertEqual(grad, 0.0)

    # The precontact row is straight-through: the force stays nominal and
    # only the gradient anticipates the contact.
    force, grad = contact_force_and_grad(precontact_solimp)
    self.assertEqual(force, 0.0)
    self.assertLess(grad, 0.0)

    # The surrogate the gradient comes from does push on the sphere.
    soft = forward._forward(
        mx.replace(opt=mx.opt.replace(precontact_solimp=precontact_solimp)), dx, soft=True
    )
    self.assertGreater(soft.qfrc_constraint[2], 0.0)

  def test_precontact_rows_keep_friction(self):
    """A separated contact's precontact row is a full contact row."""
    for cone in ('pyramidal', 'elliptic'):
      m = mujoco.MjModel.from_xml_string(f"""
          <mujoco>
            <option gravity="0 0 0" cone="{cone}"/>
            <worldbody>
              <geom type="plane" size="5 5 .1" condim="3"/>
              <body pos="0 0 0.15">
                <freejoint/>
                <geom type="sphere" size=".1" mass="1" condim="3"/>
              </body>
            </worldbody>
          </mujoco>
          """)
      d = mujoco.MjData(m)
      mx = mjx.put_model(m)
      mx = mx.replace(opt=mx.opt.replace(
          precontact_solimp=jp.array([0.0, 0.02, 1.0, 1.0, 10.0])
      ))

      def surrogate_force(qvel):
        d.qvel[:] = qvel
        dx = mjx.put_data(m, d)
        return np.asarray(forward._forward(mx, dx, soft=True).qfrc_constraint)

      # Sliding: the precontact friction opposes the motion.
      sliding = surrogate_force([1, 0, 0, 0, 0, 0])
      self.assertLess(sliding[0], 0.0, cone)
      # Approaching without sliding: a purely normal force.
      approach = surrogate_force([0, 0, -1, 0, 0, 0])
      self.assertGreater(approach[2], 0.0, cone)
      np.testing.assert_array_equal(approach[[0, 1, 3, 4, 5]], 0.0, cone)
      # At rest: no force.
      np.testing.assert_array_equal(surrogate_force([0] * 6), 0.0, cone)

  def test_precontact_gate_follows_liftoff_softness(self):
    """Precontact rows share the liftoff smoothing of in-contact rows."""
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <geom type="plane" size="5 5 .1" condim="3"/>
            <body pos="0 0 0.15">
              <freejoint/>
              <geom type="sphere" size=".1" mass="1" condim="3"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, mujoco.MjData(m))

    def accel_and_grad(liftoff_softness, vz):
      model = mx.replace(opt=mx.opt.replace(
          precontact_solimp=jp.array([0.0, 0.02, 1.0, 1.0, 10.0]),
          liftoff_softness=liftoff_softness,
      ))
      fn = lambda qpos, qvel: mjx.forward(
          model, dx.replace(qpos=qpos, qvel=qvel)
      ).qacc[2]
      qvel = dx.qvel.at[2].set(vz)
      value, grads = jax.jit(jax.value_and_grad(fn, (0, 1)))(dx.qpos, qvel)
      return value, np.array([grads[0][2], grads[1][2]])

    for vz in (0.0, 0.05):
      # Hard gate: no gradient at rest or moving away.
      value, grad = accel_and_grad(0.0, vz)
      self.assertEqual(value, 0.0)
      np.testing.assert_array_equal(grad, 0.0)
      # Smoothed gate: nominal forward; closer or approaching faster pushes up.
      value, grad = accel_and_grad(0.2, vz)
      self.assertEqual(value, 0.0)
      self.assertTrue(np.all(grad < 0), grad)
    # Approaching, both give the gated gradient up to the band's tail.
    np.testing.assert_allclose(
        accel_and_grad(0.2, -0.5)[1], accel_and_grad(0.0, -0.5)[1], rtol=1e-3
    )

  def test_precontact_forward_matches_full_straight_through(self):
    """Sharing the pre-constraint stages keeps values and gradients unchanged.

    Runs in float64: in float32 the two programs round differently.
    """
    with jax.enable_x64(True):
      m = test_util.load_test_file('humanoid/humanoid.xml')
      d = mujoco.MjData(m)
      mujoco.mj_step(m, d, 50)
      mx = mjx.put_model(m)
      mx = mx.replace(opt=mx.opt.replace(
          precontact_solimp=jp.array([0.0, 0.8, 0.05, 1.0, 10.0])
      ))
      dx = mjx.put_data(m, d)

      def reference(qpos, qvel):
        data = dx.replace(qpos=qpos, qvel=qvel)
        return mjx_math.straight_through(
            forward._forward(mx, data, soft=False),
            forward._forward(mx, data, soft=True),
        )

      def optimized(qpos, qvel):
        return mjx.forward(mx, dx.replace(qpos=qpos, qvel=qvel))

      def loss(fn, qpos, qvel):
        out = fn(qpos, qvel)
        return jp.sum(out.qacc ** 2) + jp.sum(out.sensordata) + jp.sum(
            out.qfrc_constraint
        )

      for name in ('qacc', 'qfrc_constraint', 'sensordata'):
        np.testing.assert_allclose(
            getattr(jax.jit(optimized)(dx.qpos, dx.qvel), name),
            getattr(jax.jit(reference)(dx.qpos, dx.qvel), name),
            rtol=1e-10, atol=1e-10, err_msg=name,
        )
      grad = lambda fn: jax.jit(jax.grad(lambda *x: loss(fn, *x), (0, 1)))
      for expected, actual in zip(
          grad(reference)(dx.qpos, dx.qvel), grad(optimized)(dx.qpos, dx.qvel)
      ):
        self.assertGreater(np.max(np.abs(expected)), 0.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

  def test_forward(self):
    m = test_util.load_test_file('constraints.xml')
    d = mujoco.MjData(m)
    # apply some control and xfrc input
    d.ctrl = np.array([-18, 0.59, 0.47])
    d.xfrc_applied[0, 2] = 0.1  # torque
    d.xfrc_applied[1, 4] = 0.3  # linear force
    mujoco.mj_step(m, d, 20)  # get some dynamics going
    # scale down velocities to minimize Jdotv effect (not in MJX)
    # TODO(team): remove this change when mjx supports this feature
    d.qvel[:] *= 1e-2
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)

    # fwd_actuation
    dx = mjx.put_data(m, d).replace(
        act_dot=np.zeros_like(d.act_dot),
        qfrc_actuator=np.zeros_like(d.qfrc_actuator),
        actuator_force=np.zeros_like(d.actuator_force),
    )
    dx = jax.jit(mjx.fwd_actuation)(mx, dx)
    _assert_attr_eq(d, dx, 'act_dot')
    _assert_attr_eq(d, dx, 'qfrc_actuator')
    _assert_attr_eq(d, dx, 'actuator_force')

    # fwd_accleration (fwd_position and fwd_velocity already tested elsewhere)
    dx = jax.jit(mjx.fwd_acceleration)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'qfrc_smooth')
    _assert_attr_eq(d, dx, 'qacc_smooth')

    # euler
    dx = jax.jit(mjx.euler)(mx, mjx.put_data(m, d))
    mujoco.mj_Euler(m, d)
    _assert_attr_eq(d, dx, 'act')
    _assert_attr_eq(d, dx, 'qpos')
    _assert_attr_eq(d, dx, 'time')

    # implicitfast
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # TODO(team): remove this override when the mjx feature matches mujoco
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    dx = jax.jit(mjx.implicit)(mx, mjx.put_data(m, d))
    mujoco.mj_implicit(m, d)
    _assert_attr_eq(d, dx, 'qpos')

  def test_step(self):
    m = test_util.load_test_file('constraints.xml')
    d = mujoco.MjData(m)
    # apply some control and xfrc input
    d.ctrl = np.array([-18, 0.59, 0.47])
    d.xfrc_applied[0, 2] = 0.1  # torque
    d.xfrc_applied[1, 4] = 0.3  # linear force
    mujoco.mj_step(m, d, 20)  # get some dynamics going
    # scale down velocities to minimize Jdotv effect (not in MJX)
    # TODO(team): remove this change when mjx supports this feature
    d.qvel[:] *= 1e-2

    dx = jax.jit(mjx.step)(mjx.put_model(m), mjx.put_data(m, d))
    mujoco.mj_step(m, d)
    _assert_attr_eq(d, dx, 'act')
    _assert_attr_eq(d, dx, 'time')
    _assert_attr_eq(d, dx, 'qvel', tol=5e-4)
    _assert_attr_eq(d, dx, 'qpos')

  def test_rk4(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option integrator="RK4">
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <geom type="plane" size="1 1 .01" pos="0 0 -1"/>
            <body pos="0.15 0 0">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
              <body pos="0.1 0 0">
                <joint type="slide" axis="1 0 0" stiffness="200"/>
                <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)

    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.array([0.2, -0.1])
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)

    dx = jax.jit(mjx.rungekutta4)(mjx.put_model(m), mjx.put_data(m, d))
    mujoco.mj_RungeKutta(m, d, 4)

    _assert_attr_eq(d, dx, 'qvel')
    _assert_attr_eq(d, dx, 'qpos')
    _assert_attr_eq(d, dx, 'act')
    _assert_attr_eq(d, dx, 'time')
    _assert_attr_eq(d, dx, 'xpos')

  def test_eulerdamp(self):
    m = test_util.load_test_file('pendula.xml')
    self.assertTrue((m.dof_damping > 0).any())

    d = mujoco.MjData(m)
    d.qvel[:] = 1.0
    d.qacc[:] = 1.0
    mujoco.mj_forward(m, d)
    dx = jax.jit(mjx.euler)(mjx.put_model(m), mjx.put_data(m, d))
    mujoco.mj_Euler(m, d)

    _assert_attr_eq(d, dx, 'qpos')

    # also test sparse
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    d.qvel[:] = 1.0
    d.qacc[:] = 1.0
    mujoco.mj_forward(m, d)
    dx = jax.jit(mjx.euler)(mjx.put_model(m), mjx.put_data(m, d))
    mujoco.mj_Euler(m, d)

    _assert_attr_eq(d, dx, 'qpos')

  def test_disable_eulerdamp(self):
    m = test_util.load_test_file('pendula.xml')
    self.assertTrue((m.dof_damping > 0).any())
    m.opt.disableflags = m.opt.disableflags | mjx.DisableBit.EULERDAMP

    d = mujoco.MjData(m)
    d.qvel[:] = 1.0
    d.qacc[:] = 1.0
    dx = jax.jit(mjx.euler)(mjx.put_model(m), mjx.put_data(m, d))

    np.testing.assert_allclose(dx.qvel, 1 + m.opt.timestep)



class ActuatorTest(parameterized.TestCase):

  @parameterized.parameters(
      'actuator/arm21.xml',
      'actuator/arm26.xml',
      'actuator/general_dyntype.xml',
  )
  def test_actuator(self, fname):
    m = test_util.load_test_file(fname)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    d.ctrl = 1.5 * np.random.random(m.nu)
    d.act = 0.5 * np.random.random(m.na)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_fwdActuation(m, d)
    dx = jax.jit(mjx.fwd_actuation)(mx, dx)

    _assert_attr_eq(d, dx, 'act_dot')
    _assert_attr_eq(d, dx, 'qfrc_actuator')
    _assert_attr_eq(d, dx, 'actuator_force')

    mujoco.mj_Euler(m, d)
    dx = jax.jit(mjx.euler)(mx, dx)
    _assert_attr_eq(d, dx, 'act')

  def test_tendon_force_clamp(self):
    m = test_util.load_test_file('actuator/tendon_force_clamp.xml')
    d = mujoco.MjData(m)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    dx = dx.replace(ctrl=jp.array([1.0, 1.0, 1.0, -4.0, 1.0, -20.0, 5.0, -5.0]))
    dx = mjx.forward(mx, dx)

    _assert_eq(
        dx.actuator_force,
        jp.array([1.0, 1.0, 1.0, -4.0 / 3.0, 1.0 / 3.0, -10.0, 5.0, -5.0]),
        'actuator_force',
    )

    _assert_eq(
        dx.sensordata,
        jp.array([3.0, -1.0, -10.0, 0.0]),
        'sensordata',
    )


if __name__ == '__main__':
  absltest.main()
