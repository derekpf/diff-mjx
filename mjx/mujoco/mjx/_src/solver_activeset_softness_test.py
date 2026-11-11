# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for contact_activeset_softness: IFT-only smoothing of the contact Jaref mask."""

import os
from unittest import mock

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

jax.config.update('jax_enable_x64', True)

_MASS = 1.0
_MU = 0.5
_G = 9.81

_BOX_XML = f"""
<mujoco>
  <option timestep="0.002" iterations="100" tolerance="1e-12" cone="pyramidal"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="{_MASS}" friction="{_MU} .005 .0001" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _resting_data(m: mujoco.MjModel) -> mujoco.MjData:
  """Box settled on the plane (in contact, at rest)."""
  d = mujoco.MjData(m)
  for _ in range(500):
    mujoco.mj_step(m, d)
  d.qvel[:] = 0.0
  mujoco.mj_forward(m, d)
  return d


def _model(softness: float, pw_solimp=None) -> mjx.Model:
  m = mujoco.MjModel.from_xml_string(_BOX_XML)
  mx = mjx.put_model(m)
  opt = mx.opt.replace(contact_activeset_softness=softness)
  if pw_solimp is not None:
    opt = opt.replace(pw_solimp=jp.array(pw_solimp))
  return mx.replace(opt=opt)


def _with_ift(fn):
  def wrapped(*args, **kwargs):
    with mock.patch.dict(os.environ, {'MJX_GRAD_BACKEND': 'ift'}, clear=False):
      return fn(*args, **kwargs)
  return wrapped


@_with_ift
def _slope(mx: mjx.Model, dx: mjx.Data, dof: int, forces: np.ndarray) -> np.ndarray:
  """d qacc[dof] / d qfrc_applied[dof] over a sweep of applied forces."""

  def qacc_of(f):
    d = dx.replace(qfrc_applied=dx.qfrc_applied.at[dof].set(f))
    return mjx.forward(mx, d).qacc[dof]

  return np.asarray(jax.jit(jax.vmap(jax.grad(qacc_of)))(jp.asarray(forces)))


def _max_jump(slopes: np.ndarray) -> float:
  return float(np.max(np.abs(np.diff(slopes))))


class ContactActiveTauTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    self.d = _resting_data(m)
    self.dx = mjx.put_data(m, self.d)

  @_with_ift
  def test_softness_zero_matches_default(self):
    """softness=0 (explicit) gives bit-identical forward values and gradients."""
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    mx_default, mx_zero = mjx.put_model(m), _model(0.0)

    def loss(mx, qvel):
      return jp.sum(mjx.forward(mx, self.dx.replace(qvel=qvel)).qacc ** 2)

    for mx in (mx_default, mx_zero):
      self.assertEqual(mx.opt.contact_activeset_softness, 0.0)
    g_default = jax.grad(lambda v: loss(mx_default, v))(self.dx.qvel + 0.1)
    g_zero = jax.grad(lambda v: loss(mx_zero, v))(self.dx.qvel + 0.1)
    np.testing.assert_array_equal(g_default, g_zero)

  @_with_ift
  def test_forward_unchanged(self):
    """softness>0 leaves forward qacc and efc_force unchanged."""
    d = self.dx.replace(qfrc_applied=self.dx.qfrc_applied.at[0].set(3.0))
    out0 = jax.jit(mjx.forward)(_model(0.0), d)
    out1 = jax.jit(mjx.forward)(_model(1.0), d)
    np.testing.assert_array_equal(out0.qacc, out1.qacc)
    np.testing.assert_array_equal(out0._impl.efc_force, out1._impl.efc_force)

  def _sweep_test(self, dof, center, half_width, softness):
    forces = np.linspace(center - half_width, center + half_width, 401)
    hard = _slope(_model(0.0), self.dx, dof, forces)
    soft = _slope(_model(softness), self.dx, dof, forces)
    self.assertTrue(np.isfinite(hard).all() and np.isfinite(soft).all())
    return forces, hard, soft

  def test_stick_slip_is_smoothed(self):
    """Tangential force across the friction threshold: hard slope jumps, smoothed slope does not."""
    forces, hard, soft = self._sweep_test(0, _MU * _MASS * _G, 4.0, softness=0.1)
    self.assertGreater(_max_jump(hard), 0.1)  # the stick -> slip switch
    self.assertLess(_max_jump(soft), 0.25 * _max_jump(hard))

  def test_liftoff_is_smoothed(self):
    """Upward force across the weight: hard slope jumps, smoothed slope does not."""
    forces, hard, soft = self._sweep_test(2, _MASS * _G, 4.0, softness=0.1)
    self.assertGreater(_max_jump(hard), 0.1)  # the contact -> liftoff switch
    self.assertLess(_max_jump(soft), 0.25 * _max_jump(hard))

  def test_converges_to_nominal_away_from_switch(self):
    """Far from the friction threshold, the smoothed slope approaches the hard one as softness -> 0."""
    forces = np.array([0.5, 1.0, 9.0, 10.0])  # deep stick, deep slip (threshold ~4.9)
    hard = _slope(_model(0.0), self.dx, 0, forces)
    errs = [np.max(np.abs(_slope(_model(softness), self.dx, 0, forces) - hard)) for softness in (1.0, 0.1, 0.01)]
    self.assertLess(errs[2], errs[0])
    np.testing.assert_allclose(_slope(_model(1e-3), self.dx, 0, forces), hard, atol=1e-3, rtol=1e-3)

  @_with_ift
  def test_pw_solimp_rows_untouched(self):
    """A separated contact within pw_solimp width is only a pw_solimp row; softness must not change it."""
    pw = [0.0, 0.4, 1.0, 1.0, 10.0]
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    d = mujoco.MjData(m)
    d.qpos[2] = 0.1 + 0.05  # 5 cm above the plane: separated, within width
    mujoco.mj_forward(m, d)
    dx = mjx.put_data(m, d)
    self.assertTrue((np.asarray(dx._impl.contact.dist) > 0).all())

    def loss(mx, qpos):
      return jp.sum(mjx.forward(mx, dx.replace(qpos=qpos)).qacc ** 2)

    g0 = jax.grad(lambda q: loss(_model(0.0, pw), q))(dx.qpos)
    g1 = jax.grad(lambda q: loss(_model(1.0, pw), q))(dx.qpos)
    self.assertTrue(np.isfinite(g1).all())
    np.testing.assert_array_equal(g0, g1)

  def test_elliptic_raises(self):
    m = mujoco.MjModel.from_xml_string(_BOX_XML.replace('cone="pyramidal"', 'cone="elliptic"'))
    mx = mjx.put_model(m)
    mx = mx.replace(opt=mx.opt.replace(contact_activeset_softness=1.0))
    dx = mjx.put_data(m, _resting_data(m))
    with self.assertRaises(NotImplementedError):
      _with_ift(jax.grad(lambda v: jp.sum(mjx.forward(mx, dx.replace(qvel=v)).qacc)))(dx.qvel)

  def test_non_ift_backend_raises(self):
    with mock.patch.dict(os.environ, {'MJX_GRAD_BACKEND': 'unrolled'}):
      with self.assertRaisesRegex(ValueError, 'MJX_GRAD_BACKEND=ift'):
        mjx.forward(_model(0.1), self.dx)
      mjx.forward(_model(0.0), self.dx)  # softness=0 is fine on any backend

  def test_invalid_softness_raises(self):
    with self.assertRaisesRegex(ValueError, 'finite and >= 0'):
      _with_ift(mjx.forward)(_model(-0.1), self.dx)
    with self.assertRaisesRegex(TypeError, 'Python float'):
      _with_ift(mjx.forward)(_model(jp.asarray(0.1)), self.dx)


_HINGE_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" iterations="100" tolerance="1e-12" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="j" type="hinge" axis="0 0 1" {attrs}/>
      <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _hinge(attrs: str, qpos: float, **softnesses):
  m = mujoco.MjModel.from_xml_string(_HINGE_XML.format(attrs=attrs))
  d = mujoco.MjData(m)
  d.qpos[0] = qpos
  mujoco.mj_forward(m, d)
  mx = mjx.put_model(m)
  return mx.replace(opt=mx.opt.replace(**softnesses)), mjx.put_data(m, d)


class LimitAndFrictionlossTauTest(absltest.TestCase):

  def _sweep(self, attrs, qpos, torques, **softnesses):
    mx, dx = _hinge(attrs, qpos, **softnesses)
    return _slope(mx, dx, 0, torques)

  def test_frictionloss_stick_slip_is_smoothed(self):
    """Torque across the friction-loss level: hard slope jumps, smoothed slope does not."""
    attrs = 'frictionloss="1"'
    # Fine grid across the stick -> slip threshold (~1.1 N m for this hinge).
    torques = np.linspace(0.9, 1.4, 1001)
    hard = self._sweep(attrs, 0.0, torques)
    soft = self._sweep(attrs, 0.0, torques, frictionloss_activeset_softness=0.1)
    self.assertTrue(np.isfinite(soft).all())
    self.assertGreater(_max_jump(hard), 0.1)
    self.assertLess(_max_jump(soft), 0.25 * _max_jump(hard))
    # Far from the switch (deep stick and deep slip), the smoothed slope matches the hard one.
    far = np.array([0.0, 3.0])
    np.testing.assert_allclose(self._sweep(attrs, 0.0, far, frictionloss_activeset_softness=0.1),
                               self._sweep(attrs, 0.0, far), rtol=2e-2)

  def test_limit_release_is_smoothed(self):
    """Hinge already past its limit (row engaged), torque sweeping across release."""
    attrs = 'limited="true" range="-0.5 0.5"'
    # Fine grid across the release torque (~-0.24 N m for this hinge just past its limit).
    torques = np.linspace(-0.6, -0.1, 1001)
    hard = self._sweep(attrs, 0.501, torques)
    soft = self._sweep(attrs, 0.501, torques, limit_activeset_softness=0.1)
    self.assertTrue(np.isfinite(soft).all())
    self.assertGreater(_max_jump(hard), 0.1)
    self.assertLess(_max_jump(soft), 0.25 * _max_jump(hard))

  @_with_ift
  def test_forward_unchanged(self):
    for attrs, qpos, softnesses in (('frictionloss="1"', 0.0, dict(frictionloss_activeset_softness=0.1)),
                              ('limited="true" range="-0.5 0.5"', 0.501, dict(limit_activeset_softness=0.1))):
      mx0, dx = _hinge(attrs, qpos)
      mx1, _ = _hinge(attrs, qpos, **softnesses)
      d = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(0.7))
      out0, out1 = mjx.forward(mx0, d), mjx.forward(mx1, d)
      np.testing.assert_array_equal(out0.qacc, out1.qacc)
      np.testing.assert_array_equal(out0._impl.efc_force, out1._impl.efc_force)

  def test_contact_softness_leaves_limit_and_frictionloss_rows(self):
    """contact_activeset_softness alone must not change limit or friction-loss gradients."""
    for attrs, qpos in (('frictionloss="1"', 0.0), ('limited="true" range="-0.5 0.5"', 0.501)):
      torques = np.linspace(-2.0, 2.0, 41)
      np.testing.assert_array_equal(self._sweep(attrs, qpos, torques),
                                    self._sweep(attrs, qpos, torques, contact_activeset_softness=0.1))


if __name__ == '__main__':
  absltest.main()
