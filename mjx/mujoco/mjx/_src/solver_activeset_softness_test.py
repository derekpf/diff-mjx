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

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
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


def _slope(mx: mjx.Model, dx: mjx.Data, dof: int, forces: np.ndarray) -> np.ndarray:
  """d qacc[dof] / d qfrc_applied[dof] over a sweep of applied forces."""

  def qacc_of(f):
    d = dx.replace(qfrc_applied=dx.qfrc_applied.at[dof].set(f))
    return mjx.forward(mx, d).qacc[dof]

  return np.asarray(jax.jit(jax.vmap(jax.grad(qacc_of)))(jp.asarray(forces)))


def _max_jump(slopes: np.ndarray) -> float:
  return float(np.max(np.abs(np.diff(slopes))))


class ContactActivesetSoftnessTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    self.d = _resting_data(m)
    self.dx = mjx.put_data(m, self.d)

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
      jax.grad(lambda v: jp.sum(mjx.forward(mx, dx.replace(qvel=v)).qacc))(dx.qvel)

  def test_non_ift_backend_raises(self):
    unrolled = lambda mx: mx.replace(opt=mx.opt.replace(grad_backend='unrolled'))
    with self.assertRaisesRegex(ValueError, 'ift gradient backend'):
      mjx.forward(unrolled(_model(0.1)), self.dx)
    mjx.forward(unrolled(_model(0.0)), self.dx)  # softness=0 is fine on any backend

  def test_invalid_softness_raises(self):
    with self.assertRaisesRegex(ValueError, 'finite and >= 0'):
      mjx.forward(_model(-0.1), self.dx)
    with self.assertRaisesRegex(TypeError, 'Python float'):
      mjx.forward(_model(jp.asarray(0.1)), self.dx)


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


class LimitAndFrictionlossActivesetSoftnessTest(absltest.TestCase):

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


def _refined_model(softness: float, iterations, tolerance=None, xml: str = _BOX_XML,
                   ls_iterations=None, fallback_enable=None) -> mjx.Model:
  mx = mjx.put_model(mujoco.MjModel.from_xml_string(xml))
  opt = mx.opt.replace(contact_activeset_softness=softness,
                       refinement_iterations=iterations)
  if tolerance is not None:
    opt = opt.replace(refinement_tolerance=tolerance)
  if ls_iterations is not None:
    opt = opt.replace(refinement_ls_iterations=ls_iterations)
  if fallback_enable is not None:
    opt = opt.replace(refinement_fallback_enable=fallback_enable)
  return mx.replace(opt=opt)


# Stiff friction: plain Newton refinement converges slowly here.
_STIFF_BOX_XML = _BOX_XML.replace('cone="pyramidal"/>', 'cone="pyramidal" impratio="100"/>')


def _smoothed_root_qacc(softness: float, dx: mjx.Data, dof: int, force: float,
                        newton_iters: int = 30) -> float:
  """Independent reference: Newton on the smoothed residual from the hard solution."""
  mx = _model(softness)
  d = dx.replace(qfrc_applied=dx.qfrc_applied.at[dof].set(force))
  d = jax.jit(mjx.forward)(_model(0.0), d)  # hard solution and constraint rows
  root = jax.jit(lambda q: solver._root(mx, d, q))
  jac = jax.jit(jax.jacfwd(lambda q: solver._root(mx, d, q)))
  q = d.qacc
  for _ in range(newton_iters):
    q = q - jp.linalg.solve(jac(q), root(q))
  assert float(jp.linalg.norm(root(q))) < 1e-9
  return float(q[dof])


class RefinementTest(absltest.TestCase):
  """opt.refinement_iterations: gradient evaluated at the smoothed root."""

  _THRESHOLD = _MU * _MASS * _G  # stick/slip along dof 0

  def setUp(self):
    super().setUp()
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    self.dx = mjx.put_data(m, _resting_data(m))

  def test_no_softness_is_bit_identical_to_hard(self):
    """With every softness 0 refinement is skipped, so nothing changes."""
    forces = np.linspace(self._THRESHOLD - 2.0, self._THRESHOLD + 2.0, 41)
    np.testing.assert_array_equal(
        _slope(_model(0.0), self.dx, 0, forces),
        _slope(_refined_model(0.0, 4), self.dx, 0, forces),
    )

  def test_forward_unchanged(self):
    """Forward values stay the hard solution (straight-through)."""
    for force in (self._THRESHOLD - 0.5, self._THRESHOLD, self._THRESHOLD + 0.5):
      d = self.dx.replace(qfrc_applied=self.dx.qfrc_applied.at[0].set(force))
      hard = jax.jit(mjx.forward)(_model(0.0), d)
      refined = jax.jit(mjx.forward)(_refined_model(0.1, 2), d)
      np.testing.assert_array_equal(hard.qacc, refined.qacc)
      np.testing.assert_array_equal(hard._impl.efc_force, refined._impl.efc_force)
      np.testing.assert_array_equal(hard.qfrc_constraint, refined.qfrc_constraint)

  def test_converged_refinement_is_smoothed_root_derivative(self):
    """Converged refinement gives the exact derivative of the smoothed root."""
    softness, h = 0.1, 1e-5
    for dof, thr in ((0, self._THRESHOLD), (2, _MASS * _G)):
      forces = np.linspace(thr - 1.0, thr + 1.0, 9)
      grad = _slope(_refined_model(softness, 12), self.dx, dof, forces)
      fd = np.array([
          (_smoothed_root_qacc(softness, self.dx, dof, f + h)
           - _smoothed_root_qacc(softness, self.dx, dof, f - h)) / (2 * h)
          for f in forces
      ])
      np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-7)

  def test_refinement_changes_gradient_near_switch(self):
    """The surrogate (0 iterations) and refined gradients differ near stick/slip."""
    forces = np.linspace(self._THRESHOLD - 1.0, self._THRESHOLD + 1.0, 81)
    surrogate = _slope(_model(0.1), self.dx, 0, forces)
    refined = _slope(_refined_model(0.1, 12), self.dx, 0, forces)
    scale = np.max(np.abs(_slope(_model(0.0), self.dx, 0, forces)))
    self.assertGreater(np.max(np.abs(refined - surrogate)) / scale, 0.05)

  def test_default_stiffness_iteration_counts(self):
    """Default stiffness: k=2 is surrogate-or-converged and mostly converged;
    k=4 matches the converged gradient everywhere."""
    forces = np.linspace(self._THRESHOLD - 3.0, self._THRESHOLD + 3.0, 61)
    surrogate = _slope(_model(0.1), self.dx, 0, forces)
    two = _slope(_refined_model(0.1, 2, ls_iterations=8), self.dx, 0, forces)
    four = _slope(_refined_model(0.1, 4, ls_iterations=8), self.dx, 0, forces)
    converged = _slope(_refined_model(0.1, 12), self.dx, 0, forces)
    scale = np.max(np.abs(converged))
    is_surrogate = two == surrogate
    is_converged = np.abs(two - converged) <= 1e-4 * scale
    self.assertTrue(np.all(is_surrogate | is_converged))
    self.assertGreater(np.mean(is_converged), 0.5)
    np.testing.assert_allclose(four, converged, rtol=0, atol=1e-6 * scale)

  def test_linesearch_handles_overshoot_and_undershoot(self):
    """The smoothed line search minimizes along the Newton direction both ways."""
    cases = (
        ('overshoot', jp.tanh, 1.5),  # full Newton step jumps far past 0
        ('undershoot', lambda q: q**3 + q, 2.0),  # curvature drops: step too short
    )
    for name, root, x0 in cases:
      q = jp.array([x0])
      direction = -jp.linalg.solve(jax.jacfwd(root)(q), root(q))
      alpha = solver._smoothed_linesearch(root, q, direction, 8)
      if name == 'undershoot':
        self.assertGreater(float(alpha), 1.0)  # needs bracket expansion
      else:
        self.assertLess(float(alpha), 1.0)
      self.assertLess(float(jp.abs(root(q + alpha * direction)[0])), 1e-6, msg=name)

  def test_refinement_converges_where_plain_newton_diverges(self):
    """Newton on arctan diverges from |x| > ~1.39; the line search fixes it."""
    out, accepted = solver._refine_smoothed_root(jp.arctan, jp.array([3.0]), 4, 1e-6, ls_iterations=8)
    self.assertTrue(bool(accepted))
    self.assertLess(float(jp.abs(out[0])), 1e-8)

  def test_fallback_rejects_partial_convergence(self):
    """Not converged to tolerance: return the start (the surrogate), never in between."""
    # 2-D, so one exact line search along the Newton direction cannot land on
    # the root (in 1-D it would).
    start = jp.array([2.0, 0.5])
    cubic = lambda q: jp.array([q[0]**3 + q[0] + 0.5 * q[1], 4.0 * q[1]**3 + q[1] + 0.5 * q[0]])
    out, accepted = solver._refine_smoothed_root(cubic, start, 1, 1e-12)
    self.assertFalse(bool(accepted))
    np.testing.assert_array_equal(out, start)
    # Same start, enough iterations: accepted.
    out, accepted = solver._refine_smoothed_root(cubic, start, 8, 1e-12, ls_iterations=8)
    self.assertTrue(bool(accepted))
    # Non-finite iterates are rejected: finite at the start, NaN where steps land.
    nan_root = lambda q: jp.where(q < 5.0, q - 10.0, jp.nan)
    out, accepted = solver._refine_smoothed_root(nan_root, jp.array([3.0]), 1, 1e-6)
    self.assertFalse(bool(accepted))
    np.testing.assert_array_equal(out, jp.array([3.0]))

  def test_stiff_refined_gradient_is_surrogate_or_converged(self):
    """With stiff friction, k=2 is not converged everywhere; the fallback must
    give exactly the surrogate there, and a near-converged gradient elsewhere."""
    m = mujoco.MjModel.from_xml_string(_STIFF_BOX_XML)
    dx = mjx.put_data(m, _resting_data(m))
    forces = np.linspace(self._THRESHOLD - 2.0, self._THRESHOLD + 2.0, 41)
    surrogate = _slope(_refined_model(0.1, 0, xml=_STIFF_BOX_XML), dx, 0, forces)
    two = _slope(_refined_model(0.1, 2, xml=_STIFF_BOX_XML, ls_iterations=8), dx, 0, forces)
    converged = _slope(_refined_model(0.1, 12, xml=_STIFF_BOX_XML), dx, 0, forces)
    scale = np.max(np.abs(converged))
    is_surrogate = two == surrogate
    is_converged = np.abs(two - converged) <= 1e-4 * scale
    self.assertTrue(np.all(is_surrogate | is_converged))
    self.assertTrue(np.any(is_surrogate))  # the fallback actually triggered

  def test_stiff_converges_in_six_iters(self):
    """With the line search, stiff friction converges in ~6 iterations (was ~12)."""
    m = mujoco.MjModel.from_xml_string(_STIFF_BOX_XML)
    dx = mjx.put_data(m, _resting_data(m))
    forces = np.linspace(self._THRESHOLD - 2.0, self._THRESHOLD + 2.0, 41)
    six = _slope(_refined_model(0.1, 6, xml=_STIFF_BOX_XML, ls_iterations=8), dx, 0, forces)
    converged = _slope(_refined_model(0.1, 12, xml=_STIFF_BOX_XML), dx, 0, forces)
    np.testing.assert_allclose(six, converged, rtol=0, atol=1e-6 * np.max(np.abs(converged)))

  def test_ls_iterations_zero_is_plain_newton(self):
    """refinement_ls_iterations=0 takes full Newton steps, bit-identically."""
    root = lambda q: jp.array([q[0]**3 + q[0] + 0.5 * q[1], 4.0 * q[1]**3 + q[1] + 0.5 * q[0]])
    start = jp.array([2.0, 0.5])
    q = start
    for _ in range(3):
      q = q - jp.linalg.solve(jax.jacfwd(root)(q), root(q))
    out, _ = solver._refine_smoothed_root(root, start, 3, float('inf'), ls_iterations=0)
    np.testing.assert_array_equal(out, q)
    # With the line search the iterates differ (and converge faster here).
    out_ls, _ = solver._refine_smoothed_root(root, start, 3, float('inf'), ls_iterations=8)
    self.assertFalse(np.array_equal(out_ls, q))
    self.assertLess(float(jp.linalg.norm(root(out_ls))), float(jp.linalg.norm(root(q))))

  def test_ls_iterations_zero_still_surrogate_or_converged(self):
    """Without the line search, the fallback still guarantees no in-between gradient."""
    m = mujoco.MjModel.from_xml_string(_STIFF_BOX_XML)
    dx = mjx.put_data(m, _resting_data(m))
    forces = np.linspace(self._THRESHOLD - 2.0, self._THRESHOLD + 2.0, 21)
    surrogate = _slope(_refined_model(0.1, 0, xml=_STIFF_BOX_XML), dx, 0, forces)
    plain = _slope(_refined_model(0.1, 4, xml=_STIFF_BOX_XML, ls_iterations=0), dx, 0, forces)
    converged = _slope(_refined_model(0.1, 12, xml=_STIFF_BOX_XML), dx, 0, forces)
    scale = np.max(np.abs(converged))
    self.assertTrue(np.all((plain == surrogate) | (np.abs(plain - converged) <= 1e-4 * scale)))

  def test_invalid_ls_iterations_raise(self):
    for bad, err in ((-1, ValueError), (2.0, TypeError), (True, TypeError),
                     (jp.asarray(8), TypeError)):
      with self.assertRaises(err, msg=repr(bad)):
        mjx.forward(_refined_model(0.1, 2, ls_iterations=bad), self.dx)

  def test_fallback_disabled_uses_partial_point(self):
    """refinement_fallback_enable=False skips the tolerance check (not the NaN guard)."""
    start = jp.array([2.0, 0.5])
    cubic = lambda q: jp.array([q[0]**3 + q[0] + 0.5 * q[1], 4.0 * q[1]**3 + q[1] + 0.5 * q[0]])
    on, acc_on = solver._refine_smoothed_root(cubic, start, 1, 1e-12, fallback_enable=True)
    off, acc_off = solver._refine_smoothed_root(cubic, start, 1, 1e-12, fallback_enable=False)
    self.assertFalse(bool(acc_on))
    np.testing.assert_array_equal(on, start)
    self.assertTrue(bool(acc_off))
    self.assertFalse(np.array_equal(off, start))  # the partial point is used
    # Non-finite points still fall back with the toggle off.
    nan_root = lambda q: jp.where(q < 5.0, q - 10.0, jp.nan)
    out, accepted = solver._refine_smoothed_root(
        nan_root, jp.array([3.0]), 1, 1e-6, fallback_enable=False)
    self.assertFalse(bool(accepted))
    np.testing.assert_array_equal(out, jp.array([3.0]))

  def test_fallback_disabled_gives_in_between_gradients(self):
    """Stiff box, k=2 plain Newton: with the fallback off, some gradients are
    neither the surrogate nor converged (what the fallback prevents); forward
    values are still unchanged."""
    m = mujoco.MjModel.from_xml_string(_STIFF_BOX_XML)
    dx = mjx.put_data(m, _resting_data(m))
    forces = np.linspace(self._THRESHOLD - 2.0, self._THRESHOLD + 2.0, 21)
    surrogate = _slope(_refined_model(0.1, 0, xml=_STIFF_BOX_XML), dx, 0, forces)
    converged = _slope(_refined_model(0.1, 12, xml=_STIFF_BOX_XML), dx, 0, forces)
    off = _slope(_refined_model(0.1, 2, xml=_STIFF_BOX_XML, fallback_enable=False), dx, 0, forces)
    scale = np.max(np.abs(converged))
    in_between = (off != surrogate) & (np.abs(off - converged) > 1e-4 * scale)
    self.assertTrue(np.any(in_between))
    d = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(self._THRESHOLD))
    hard = jax.jit(mjx.forward)(_refined_model(0.0, 0, xml=_STIFF_BOX_XML), d)
    off_fwd = jax.jit(mjx.forward)(
        _refined_model(0.1, 2, xml=_STIFF_BOX_XML, fallback_enable=False), d)
    np.testing.assert_array_equal(hard.qacc, off_fwd.qacc)

  def test_invalid_fallback_enable_raise(self):
    for bad in (1, 0, 'yes', jp.asarray(True)):
      with self.assertRaises(TypeError, msg=repr(bad)):
        mjx.forward(_refined_model(0.1, 2, fallback_enable=bad), self.dx)

  def test_invalid_tol_raise(self):
    for bad, err in ((-1.0, ValueError), (float('nan'), ValueError),
                     (float('inf'), ValueError), (True, TypeError),
                     (jp.asarray(1e-6), TypeError)):
      with self.assertRaises(err, msg=repr(bad)):
        mjx.forward(_refined_model(0.1, 2, tolerance=bad), self.dx)

  def test_invalid_iters_raise(self):
    for bad, err in ((-1, ValueError), (1.5, TypeError), (True, TypeError),
                     (jp.asarray(2), TypeError)):
      with self.assertRaises(err, msg=repr(bad)):
        mjx.forward(_refined_model(0.1, bad), self.dx)


if __name__ == '__main__':
  absltest.main()
