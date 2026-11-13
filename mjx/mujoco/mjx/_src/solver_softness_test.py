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
"""Tests for stickslip_softness: IFT-only smoothing of the contact Jaref mask."""


from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
from mujoco.mjx._src import support
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


def _box_contact_scale() -> float:
  """efc_softness_scale (k * imp * width) of the resting box's contact rows."""
  m = mujoco.MjModel.from_xml_string(_BOX_XML)
  dx = mjx.forward(mjx.put_model(m), mjx.put_data(m, _resting_data(m)))
  return float(dx._impl.efc_softness_scale[0])


def _model(softness: float, pw_solimp=None) -> mjx.Model:
  m = mujoco.MjModel.from_xml_string(_BOX_XML)
  mx = mjx.put_model(m)
  opt = mx.opt.replace(stickslip_softness=softness, liftoff_softness=softness)
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


def _is_surrogate(slopes: np.ndarray, surrogate: np.ndarray, scale: float) -> np.ndarray:
  """Fallback taken: the surrogate up to float rounding.

  Not bitwise: the refined and surrogate programs compile differently, so
  the same evaluation point can round differently in the backward pass.
  """
  return np.abs(slopes - surrogate) <= 1e-12 * scale


class ContactSoftnessTest(absltest.TestCase):

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
      self.assertEqual(mx.opt.stickslip_softness, 0.0)
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
    dx = mjx.put_data(m, _resting_data(m))
    for knob in ('stickslip_softness', 'liftoff_softness'):
      mk = mx.replace(opt=mx.opt.replace(**{knob: 1.0}))
      with self.assertRaises(NotImplementedError):
        jax.grad(lambda v: jp.sum(mjx.forward(mk, dx.replace(qvel=v)).qacc))(dx.qvel)

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


class LimitAndFrictionlossSoftnessTest(absltest.TestCase):

  def _sweep(self, attrs, qpos, torques, **softnesses):
    mx, dx = _hinge(attrs, qpos, **softnesses)
    return _slope(mx, dx, 0, torques)

  def test_frictionloss_stick_slip_is_smoothed(self):
    """Torque across the friction-loss level: hard slope jumps, smoothed slope does not."""
    attrs = 'frictionloss="1"'
    # Fine grid across the stick -> slip threshold (~1.1 N m for this hinge).
    torques = np.linspace(0.9, 1.4, 1001)
    hard = self._sweep(attrs, 0.0, torques)
    soft = self._sweep(attrs, 0.0, torques, frictionloss_softness=0.1)
    self.assertTrue(np.isfinite(soft).all())
    self.assertGreater(_max_jump(hard), 0.1)
    self.assertLess(_max_jump(soft), 0.25 * _max_jump(hard))
    # Far from the switch (deep stick and deep slip), the smoothed slope matches the hard one.
    far = np.array([0.0, 3.0])
    np.testing.assert_allclose(self._sweep(attrs, 0.0, far, frictionloss_softness=0.1),
                               self._sweep(attrs, 0.0, far), rtol=2e-2)

  def test_limit_release_is_smoothed(self):
    """Hinge already past its limit (row engaged), torque sweeping across release."""
    attrs = 'limited="true" range="-0.5 0.5"'
    # Fine grid across the release torque (~-0.24 N m for this hinge just past its limit).
    torques = np.linspace(-0.6, -0.1, 1001)
    hard = self._sweep(attrs, 0.501, torques)
    soft = self._sweep(attrs, 0.501, torques, limit_softness=0.1)
    self.assertTrue(np.isfinite(soft).all())
    self.assertGreater(_max_jump(hard), 0.1)
    self.assertLess(_max_jump(soft), 0.25 * _max_jump(hard))
  def test_forward_unchanged(self):
    for attrs, qpos, softnesses in (('frictionloss="1"', 0.0, dict(frictionloss_softness=0.1)),
                              ('limited="true" range="-0.5 0.5"', 0.501, dict(limit_softness=0.1))):
      mx0, dx = _hinge(attrs, qpos)
      mx1, _ = _hinge(attrs, qpos, **softnesses)
      d = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(0.7))
      out0, out1 = mjx.forward(mx0, d), mjx.forward(mx1, d)
      np.testing.assert_array_equal(out0.qacc, out1.qacc)
      np.testing.assert_array_equal(out0._impl.efc_force, out1._impl.efc_force)

  def test_stickslip_softness_leaves_limit_and_frictionloss_rows(self):
    """stickslip_softness alone must not change limit or friction-loss gradients."""
    for attrs, qpos in (('frictionloss="1"', 0.0), ('limited="true" range="-0.5 0.5"', 0.501)):
      torques = np.linspace(-2.0, 2.0, 41)
      np.testing.assert_array_equal(self._sweep(attrs, qpos, torques),
                                    self._sweep(attrs, qpos, torques, stickslip_softness=0.1))


def _refined_model(softness: float, iterations, tolerance=None, xml: str = _BOX_XML,
                   ls_iterations=None, fallback_enable=None) -> mjx.Model:
  mx = mjx.put_model(mujoco.MjModel.from_xml_string(xml))
  opt = mx.opt.replace(stickslip_softness=softness, liftoff_softness=softness,
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


_CONDIM_XML = """
<mujoco>
  <option timestep="0.002" iterations="100" tolerance="1e-12" cone="pyramidal"/>
  <worldbody>
    <geom type="plane" size="5 5 .1" condim="1"/>
    <body pos="0 0 0.1"><freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1" condim="3"/></body>
    <body pos="1 0 0.05"><freejoint/>
      <geom type="sphere" size=".05" mass="1" condim="1"/></body>
    <body pos="2 0 0.05"><freejoint/>
      <geom type="sphere" size=".05" mass="1" condim="4"/></body>
    <body pos="3 0 0.05"><freejoint/>
      <geom type="sphere" size=".05" mass="1" condim="6"/></body>
  </worldbody>
</mujoco>
"""


def _reference_cost(y, efc_d, ell, slip, lift):
  """The per-contact cost written out independently (one contact, rows y)."""
  n = jp.mean(y)
  if lift == 0:  # the l0 -> 0 limit: tau = slip * max(-n, 0), hard once unloaded
    tau = slip * jp.maximum(-n, 0.0)
    safe = jp.where(tau > 0, tau, 1.0)
    soft = jp.sum(efc_d * safe**2 * solver._softplus_integral(-y / safe))
    return jp.where(tau > 0, soft, 0.5 * jp.sum(efc_d * jp.maximum(-y, 0.0) ** 2))
  l0 = lift * ell / (1.0 + np.log(2.0))
  tau = l0 * (1.0 + jp.logaddexp(0.0, slip * -n / l0))
  return jp.sum(efc_d * tau**2 * solver._softplus_integral(-y / tau))


def _resting_box():
  m = mujoco.MjModel.from_xml_string(_BOX_XML)
  mx = mjx.put_model(m)
  return mx, mjx.forward(mx, mjx.put_data(m, _resting_data(m)))


def _two_knob_model(slip, lift, **opt):
  mx = mjx.put_model(mujoco.MjModel.from_xml_string(_BOX_XML))
  return mx.replace(opt=mx.opt.replace(
      stickslip_softness=slip, liftoff_softness=lift, **opt))


class PerContactCostTest(absltest.TestCase):
  """Contact smoothing: minus the gradient of a convex per-contact cost, with
  stick/slip in friction-utilization units and liftoff in solimp widths."""

  def test_softplus_integral(self):
    z = jp.linspace(-40.0, 40.0, 161)
    t = np.linspace(-60.0, 40.0, 400001)
    sp = np.logaddexp(0, t)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(t))])
    np.testing.assert_allclose(solver._softplus_integral(z), np.interp(np.asarray(z), t, cum), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(jax.vmap(jax.grad(solver._softplus_integral))(z), jax.nn.softplus(z), atol=1e-12)
    np.testing.assert_allclose(
        jax.vmap(jax.grad(jax.grad(solver._softplus_integral)))(z), jax.nn.sigmoid(z), atol=1e-12)

  def test_convexity_conditions(self):
    """h(y, tau) = tau^2 F(-y/tau): jointly convex, nondecreasing in tau."""
    u = jp.linspace(-60.0, 60.0, 120001)
    f, f1, f2 = solver._softplus_integral(u), jax.nn.softplus(u), jax.nn.sigmoid(u)
    self.assertTrue(np.all(np.asarray(f2) >= 0))
    self.assertTrue(np.all(np.asarray(2 * f * f2 - f1**2) > 0))
    g = solver._tau_term(u)
    self.assertTrue(np.all(np.asarray(g) > 0))
    np.testing.assert_allclose(g, 2 * f - u * f1, atol=1e-9)
    np.testing.assert_allclose(jax.vmap(jax.grad(solver._tau_term))(u), f1 - u * f2, atol=1e-9)
    np.testing.assert_allclose(solver._tau_term(jp.float32(1e4)), np.pi**2 / 3, rtol=1e-6)
    # The proof's steps: Q = 2 F F'' - F'^2 has Q' = 2 F sigmoid' > 0;
    # G' = softplus - u sigmoid > 0 (peak log 2 at 0).
    q = lambda v: 2 * solver._softplus_integral(v) * jax.nn.sigmoid(v) - jax.nn.softplus(v) ** 2
    sig1 = jax.vmap(jax.grad(jax.nn.sigmoid))(u)
    np.testing.assert_allclose(jax.vmap(jax.grad(q))(u), 2 * f * sig1, atol=1e-9)
    # G' = softplus(u) - u sigmoid(u); for u > 0 the cancellation-free form is
    # softplus(-u) + u sigmoid(-u) (the naive one rounds to 0 at |u| ~ 40).
    g1 = jp.where(u > 0, jax.nn.softplus(-u) + u * jax.nn.sigmoid(-u), f1 - u * f2)
    np.testing.assert_allclose(g1, f1 - u * f2, atol=1e-12)
    self.assertTrue(np.all(np.asarray(g1) > 0))
    self.assertAlmostEqual(float(jp.max(g1)), np.log(2.0), places=6)

  def test_band(self):
    """tau = s_liftoff widths at liftoff, a floor when separating, and a
    fraction s_slip of the load when loaded."""
    ell, slip, lift = 2.5, 0.2, 0.05
    band = lambda n: float(solver._contact_band(jp.asarray(n), ell, slip, lift))
    self.assertAlmostEqual(band(0.0), lift * ell, places=12)
    floor = lift * ell / (1 + np.log(2.0))
    self.assertAlmostEqual(band(1e3), floor, places=9)
    for load in (1e2, 1e4):
      np.testing.assert_allclose(band(-load), slip * load + floor, rtol=1e-9)
    n = jp.linspace(-5.0, 5.0, 1001)
    curv = jax.vmap(jax.grad(jax.grad(lambda v: solver._contact_band(v, ell, slip, lift))))(n)
    self.assertTrue(np.all(np.asarray(curv) > 0))

  def test_force_is_minus_gradient_and_hessian_is_psd(self):
    mx, dx = _resting_box()
    (start, ncon, rows), = solver._contact_edge_groups(mx)
    self.assertEqual((ncon, rows), (4, 4))
    scale, efc_d = dx._impl.efc_softness_scale, dx._impl.efc_D
    rng = np.random.default_rng(0)
    for slip, lift in ((0.05, 0.05), (0.3, 0.01), (1.0, 1.0), (0.02, 2.0)):
      for trial in range(15):
        y = jp.asarray(rng.normal(size=dx._impl.nefc) * 10.0 ** rng.uniform(-2, 2))
        force = solver._contact_cost_force(mx, dx, y, scale, slip, lift)
        for c in range(ncon):
          sl = slice(start + c * rows, start + (c + 1) * rows)
          args = (efc_d[sl], scale[sl][0], slip, lift)
          grad = jax.grad(_reference_cost)(y[sl], *args)
          np.testing.assert_allclose(force[sl], -grad, rtol=1e-9,
                                     atol=1e-9 * (float(jp.max(jp.abs(grad))) + 1))
          hess = np.asarray(jax.hessian(_reference_cost)(y[sl], *args))
          np.testing.assert_allclose(hess, hess.T, atol=1e-9 * np.abs(hess).max())
          self.assertGreaterEqual(np.linalg.eigvalsh(hess).min(), -1e-9 * np.abs(hess).max(),
                                  msg=(slip, lift, trial, c))

  def test_hard_limit(self):
    mx, dx = _resting_box()
    y = jp.asarray(np.random.default_rng(1).normal(size=dx._impl.nefc))
    force = solver._contact_cost_force(mx, dx, y, dx._impl.efc_softness_scale, 1e-6, 1e-6)
    np.testing.assert_allclose(force, dx._impl.efc_D * jax.nn.relu(-y), rtol=1e-4, atol=1e-6)

  def _edge_masks(self, y, efc_d, ell, slip, lift):
    jac = jax.jacfwd(lambda yy: -jax.grad(_reference_cost)(yy, efc_d, ell, slip, lift))(y)
    return np.diag(np.asarray(jac)) / -np.asarray(efc_d)

  def test_stick_slip_band_is_a_fraction_of_utilization(self):
    """Loaded well past the apex, the edge masks depend only on utilization."""
    mx, dx = _resting_box()
    (start, _, rows), = solver._contact_edge_groups(mx)
    ell = float(dx._impl.efc_softness_scale[start])
    efc_d = dx._impl.efc_D[start : start + rows]
    pair = jp.array([1.0, -1.0, 0.0, 0.0])
    slip, lift = 0.2, 0.05
    for utilization in (0.0, 0.6, 0.95, 1.05):
      masks = [self._edge_masks(-load * (jp.ones(rows) - utilization * pair), efc_d, ell, slip, lift)
               for load in (1e3 * lift * ell / slip, 1e5 * lift * ell / slip)]
      np.testing.assert_allclose(masks[0], masks[1], rtol=1e-3, err_msg=str(utilization))
    # The unloading edge fades across the friction limit. At the limit its mask
    # is 1/2 from its own softplus plus the curvature that tau's dependence on
    # the load adds to every edge, the same at any load.
    load = 1e4 * lift * ell / slip
    fade = [float(self._edge_masks(-load * (jp.ones(rows) - u * pair), efc_d, ell, slip, lift)[0])
            for u in (0.6, 0.95, 1.0, 1.05, 1.4)]
    self.assertTrue(np.all(np.diff(fade) < 0), msg=fade)
    self.assertGreater(fade[0], 0.9)
    self.assertLess(fade[-1], 0.2)
    self.assertGreater(fade[2], 0.5)
    self.assertLess(fade[2], 0.65)

  def test_liftoff_band_is_in_solimp_widths(self):
    """At the apex the band scales with l = k*imp*width; mask 1/2 at liftoff."""
    mx, dx = _resting_box()
    (start, _, rows), = solver._contact_edge_groups(mx)
    efc_d = dx._impl.efc_D[start : start + rows]
    slip, lift = 0.2, 0.3

    def mask(n, ell):
      return float(self._edge_masks(n * jp.ones(rows), efc_d, ell, slip, lift)[0])

    for ell in (0.5, 2.5, 12.0):
      np.testing.assert_allclose(mask(-0.3 * ell, ell), mask(-0.3 * 2.5, 2.5), rtol=1e-6)
      np.testing.assert_allclose(mask(0.4 * ell, ell), mask(0.4 * 2.5, 2.5), rtol=1e-6)
    self.assertAlmostEqual(mask(-1e-7, 2.5), mask(1e-7, 2.5), places=5)
    self.assertAlmostEqual(mask(-1e-7, 2.5), 0.5, delta=0.05)
    # Separating by one liftoff band: still partly on (the floor), then off.
    self.assertGreater(mask(lift * 2.5, 2.5), 0.05)
    self.assertLess(mask(20 * lift * 2.5, 2.5), 1e-6)

  def _random_contacts(self, rng, ncon, rows, loaded):
    """Edge rows n +- t per contact with the sign of n set by ``loaded``."""
    n = rng.uniform(0.05, 5.0, size=(ncon, 1)) * np.where(loaded, -1.0, 1.0)[:, None]
    t = rng.normal(size=(ncon, rows // 2)) * np.abs(n) * rng.uniform(0.0, 2.0, size=(ncon, 1))
    return jp.asarray(np.concatenate([n + t, n - t], axis=1))

  def test_zero_liftoff_softness_keeps_liftoff_hard(self):
    """liftoff_softness = 0: tau = slip |n| when loaded, the hard force once
    the contact unloads; still minus the gradient of a convex cost."""
    ell, rows = 2.5, 4
    n = jp.linspace(-5.0, 5.0, 101)
    np.testing.assert_allclose(solver._contact_band(n, ell, 0.2, 0.0), 0.2 * jp.maximum(-n, 0.0), atol=1e-15)
    rng = np.random.default_rng(3)
    loaded = np.array([True, False, True, False, True, True])
    efc_d = jp.asarray(rng.uniform(0.5, 50.0, size=(6, rows)))
    ells = jp.asarray(rng.uniform(0.5, 5.0, size=6))
    for slip in (0.05, 0.2, 1.0):
      for _ in range(10):
        y = self._random_contacts(rng, 6, rows, loaded)
        force = solver._contact_group_force(y, efc_d, ells, slip, 0.0)
        # Unloaded contacts: exactly the hard force.
        np.testing.assert_array_equal(force[~loaded], (efc_d * jp.maximum(-y, 0.0))[~loaded])
        for c in range(6):
          grad = jax.grad(_reference_cost)(y[c], efc_d[c], ells[c], slip, 0.0)
          np.testing.assert_allclose(force[c], -grad, rtol=1e-9, atol=1e-9 * (float(jp.max(jp.abs(grad))) + 1))
          if loaded[c]:
            hess = np.asarray(jax.hessian(_reference_cost)(y[c], efc_d[c], ells[c], slip, 0.0))
            self.assertGreaterEqual(np.linalg.eigvalsh(hess).min(), -1e-9 * np.abs(hess).max())
        # The analytic tangent rule against forward-mode autodiff of the reference.
        dy, dd = jp.asarray(rng.normal(size=y.shape)), jp.asarray(rng.normal(size=y.shape))
        _, t1 = jax.jvp(lambda a, b: solver._contact_group_force(a, b, ells, slip, 0.0), (y, efc_d), (dy, dd))
        ref = jax.vmap(lambda a, b, e: -jax.grad(_reference_cost)(a, b, e, slip, 0.0))
        _, t0 = jax.jvp(lambda a, b: ref(a, b, ells), (y, efc_d), (dy, dd))
        np.testing.assert_allclose(t1, t0, rtol=1e-9, atol=1e-9 * float(jp.max(jp.abs(t0))))
    # Continuous at liftoff: with the tangential part fixed, n -> 0 from the
    # loaded side gives the hard force, within O(D |n|). (Scaling the whole
    # contact down would not: with tau = slip |n| the cost is homogeneous.)
    pair = jp.array([1.0, -1.0, 0.0, 0.0])
    for n0 in (-1e-4, -1e-8):
      y = (n0 + 0.7 * pair)[None]
      err = solver._contact_group_force(y, efc_d[:1], ells[:1], 0.2, 0.0) - efc_d[:1] * jp.maximum(-y, 0.0)
      self.assertLess(float(jp.max(jp.abs(err))), 2.0 * float(jp.max(efc_d[:1])) * abs(n0))
    # Through the solver: finite, and different from smoothing liftoff too.
    mx, dx = _resting_box()
    forces = np.linspace(-4.0, 4.0, 17)
    slope = _slope(_two_knob_model(0.2, 0.0), dx, 0, forces)
    self.assertTrue(np.all(np.isfinite(slope)))
    self.assertFalse(np.allclose(slope, _slope(_two_knob_model(0.2, 0.2), dx, 0, forces)))

  def test_float32_vanishing_band_is_finite(self):
    """A loaded contact with a band near 0 (liftoff_softness = 0) stays finite."""
    rows = 4
    efc_d = jp.full((3, rows), 10.0, jp.float32)
    ells = jp.ones(3, jp.float32)
    y = jp.asarray([[-1e-12, -1e-12, 1e3, -1e3], [-1e-30, 1e-30, 0.0, 0.0], [5.0, -5.0, -1e3, 1e3]], jp.float32)
    for x in (y, -y):
      f, t = jax.jvp(lambda a: solver._contact_group_force(a, efc_d, ells, 0.2, 0.0), (x,), (jp.ones_like(x),))
      self.assertTrue(np.all(np.isfinite(np.asarray(f))) and np.all(np.isfinite(np.asarray(t))))

  def test_zero_stickslip_softness_gives_a_constant_band(self):
    """stickslip_softness = 0: a band of liftoff_softness widths at any load,
    i.e. the per-row softplus with no coupling."""
    ell, lift = 2.5, 0.2
    n = jp.linspace(-1e3, 1e3, 101)
    np.testing.assert_allclose(solver._contact_band(n, ell, 0.0, lift), lift * ell, rtol=1e-12)
    rng = np.random.default_rng(4)
    efc_d = jp.asarray(rng.uniform(0.5, 50.0, size=(3, 4)))
    y = jp.asarray(rng.normal(size=(3, 4)) * 5.0)
    tau = lift * ell
    np.testing.assert_allclose(solver._contact_group_force(y, efc_d, jp.full(3, ell), 0.0, lift),
                               efc_d * tau * jax.nn.softplus(-y / tau), rtol=1e-12)
    # Liftoff smoothing alone switches the per-contact cost on.
    mx, dx = _resting_box()
    forces = np.linspace(-4.0, 4.0, 17)
    self.assertFalse(np.allclose(_slope(_two_knob_model(0.0, lift), dx, 0, forces),
                                 _slope(_two_knob_model(0.0, 0.0), dx, 0, forces)))

  def test_liftoff_softness_is_validated(self):
    mx, dx = _resting_box()
    with self.assertRaises(ValueError):
      mjx.forward(_two_knob_model(0.2, -1.0), dx)
    with self.assertRaises(TypeError):
      mjx.forward(_two_knob_model(0.2, jp.array(0.1)), dx)

  def test_normal_coordinate_is_mean_of_edges(self):
    """mean_e Jaref_e = Jaref_n: the edges' mean row is the normal Jacobian and
    their aref differ only by the damping of the tangential parts."""
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    d = _resting_data(m)
    d.qvel[:] = [0.3, -0.2, 0.05, 0.4, -0.1, 0.2]
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.forward(mx, mjx.put_data(m, d))
    (start, ncon, rows), = solver._contact_edge_groups(mx)
    j = np.asarray(dx._impl.efc_J)[start : start + ncon * rows].reshape(ncon, rows, -1)
    aref = np.asarray(dx._impl.efc_aref)[start : start + ncon * rows].reshape(ncon, rows)
    con = dx._impl.contact
    bodies = np.asarray(m.geom_bodyid)[np.asarray(con.geom)]
    b = 2.0 / (0.95 * 0.02)  # default solref (0.02, 1), dmax 0.95
    for c in range(ncon):
      jac1, _ = support.jac(mx, dx, con.pos[c], int(bodies[c, 0]))
      jac2, _ = support.jac(mx, dx, con.pos[c], int(bodies[c, 1]))
      j_n = np.asarray(con.frame[c][0] @ (jac2 - jac1).T)
      np.testing.assert_allclose(j[c].mean(0), j_n, atol=1e-12)
      np.testing.assert_allclose(aref[c] - aref[c].mean(), -b * (j[c] - j_n) @ d.qvel, rtol=1e-9, atol=1e-9)

  def test_float32_matches_float64(self):
    mx, dx = _resting_box()
    y = jp.asarray(np.random.default_rng(2).normal(size=dx._impl.nefc) * 3.0)
    scale = dx._impl.efc_softness_scale
    for slip, lift in ((0.1, 0.1), (0.3, 0.01)):
      f64 = solver._contact_cost_force(mx, dx, y, scale, slip, lift)
      dx32 = dx.tree_replace({'_impl.efc_D': dx._impl.efc_D.astype(jp.float32)})
      f32 = solver._contact_cost_force(mx, dx32, y.astype(jp.float32), scale.astype(jp.float32), slip, lift)
      self.assertEqual(f32.dtype, jp.float32)
      np.testing.assert_allclose(f32, f64, rtol=2e-5, atol=2e-5 * float(jp.max(jp.abs(f64))))

  def test_ift_gradient_is_derivative_of_smoothed_root(self):
    """End to end: with refinement converged, the IFT gradient equals the
    finite-difference derivative of the smoothed root (Newton reference)."""
    mx0, dx = _resting_box()
    slip, lift = 0.2, 0.05
    soft = _two_knob_model(slip, lift)
    refined = _two_knob_model(slip, lift, refinement_iterations=30, refinement_ls_iterations=8,
                              refinement_fallback_enable=False)
    hard = _two_knob_model(0.0, 0.0)

    def smoothed_root(dof, force):
      d = jax.jit(mjx.forward)(hard, dx.replace(qfrc_applied=dx.qfrc_applied.at[dof].set(force)))
      root = jax.jit(lambda q: solver._root(soft, d, q))
      jac = jax.jit(jax.jacfwd(lambda q: solver._root(soft, d, q)))
      q = d.qacc
      for _ in range(60):
        q = q - jp.linalg.solve(jac(q), root(q))
      self.assertLess(float(jp.linalg.norm(root(q))), 1e-8)
      return float(q[dof])

    h = 1e-5
    for dof, forces in ((0, (1.0, 2.45, 4.0)), (2, (5.0, 9.81, 12.0))):  # push (stick/slip), lift
      grad = _slope(refined, dx, dof, np.array(forces))
      fd = np.array([(smoothed_root(dof, f + h) - smoothed_root(dof, f - h)) / (2 * h) for f in forces])
      np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-7, err_msg=f'dof {dof}')

  def test_ift_gradient_condim_4_and_6(self):
    """End to end for torsional (condim 4) and rolling (condim 6) pyramids."""
    slip, lift = 0.2, 0.05
    for condim in (4, 6):
      xml = f"""
      <mujoco>
        <option timestep="0.002" iterations="100" tolerance="1e-12" cone="pyramidal"/>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 0.05"><freejoint/>
            <geom type="sphere" size=".05" mass="1" condim="{condim}" friction="0.5 0.05 0.05"/>
          </body>
        </worldbody>
      </mujoco>"""
      m = mujoco.MjModel.from_xml_string(xml)
      dx = mjx.put_data(m, _resting_data(m))
      base = mjx.put_model(m)
      (_, ncon, rows), = solver._contact_edge_groups(base)
      self.assertEqual(rows, 2 * (condim - 1))
      mk = lambda **o: base.replace(opt=base.opt.replace(**o))
      soft = mk(stickslip_softness=slip, liftoff_softness=lift)
      refined = mk(stickslip_softness=slip, liftoff_softness=lift,
                   refinement_iterations=30, refinement_ls_iterations=8,
                   refinement_fallback_enable=False)
      hard = mk()

      def smoothed_root(dof, force):
        d = jax.jit(mjx.forward)(hard, dx.replace(qfrc_applied=dx.qfrc_applied.at[dof].set(force)))
        root = jax.jit(lambda q: solver._root(soft, d, q))
        jac = jax.jit(jax.jacfwd(lambda q: solver._root(soft, d, q)))
        q = d.qacc
        for _ in range(60):
          q = q - jp.linalg.solve(jac(q), root(q))
        self.assertLess(float(jp.linalg.norm(root(q))), 1e-8)
        return float(q[dof])

      h = 1e-5
      # push x (slip), twist about z (torsion), lift z (liftoff)
      for dof, forces in ((0, (1.0, 2.45, 4.0)), (5, (0.01, 0.05)), (2, (5.0, 9.81, 12.0))):
        grad = _slope(refined, dx, dof, np.array(forces))
        fd = np.array([(smoothed_root(dof, f + h) - smoothed_root(dof, f - h)) / (2 * h) for f in forces])
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-7, err_msg=f'condim {condim} dof {dof}')

  def test_contact_groups_match_rows(self):
    m = mujoco.MjModel.from_xml_string(_CONDIM_XML)
    mx = mjx.put_model(m)
    dx = mjx.forward(mx, mjx.put_data(m, _resting_data(m)))
    groups = solver._contact_edge_groups(mx)
    self.assertEqual([g[2] for g in groups], [1, 4, 6, 10])
    efc_type, pos = np.asarray(dx._impl.efc_type), np.asarray(dx._impl.efc_pos)
    for start, ncon, rows in groups:
      want = (mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS if rows == 1
              else mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
      blk = slice(start, start + ncon * rows)
      self.assertTrue(np.all(efc_type[blk] == int(want)))
      p = pos[blk].reshape(ncon, rows)
      np.testing.assert_allclose(p, p[:, :1] * np.ones((1, rows)))  # rows of a contact share efc_pos
    self.assertEqual(groups[-1][0] + groups[-1][1] * groups[-1][2], efc_type.shape[0])


_COORD_HINGE = """
<mujoco>
  <option timestep="0.002" iterations="100" tolerance="1e-12" gravity="0 0 0"/>
  <worldbody><body>
    <joint type="hinge" axis="0 0 1" {attr}/>
    <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05" mass="{m}"/>
  </body></worldbody>
</mujoco>"""

_COORD_BOX = """
<mujoco>
  <option timestep="0.002" iterations="100" tolerance="1e-12" cone="pyramidal" gravity="0 0 -{g}"/>
  <worldbody>
    <geom type="plane" size="5 5 .1" friction="{mu} .005 .0001" solref="{tc} 1" solimp="0.9 0.95 {w}"/>
    <body pos="0 0 0.1"><freejoint/>
      <geom type="box" size=".1 .1 .1" mass="{m}" friction="{mu} .005 .0001" condim="3"
            solref="{tc} 1" solimp="0.9 0.95 {w}"/>
    </body>
  </worldbody>
</mujoco>"""


class NormalizedCoordinateTest(absltest.TestCase):
  """Each softness knob means the same in its own coordinate for every model.

  The coordinate scale comes from MuJoCo's C solver (efc_KBIP, efc_D), not
  from efc_softness_scale; the mask -df/dJaref / D of the solver's smoothed
  force must trace the same curve in that coordinate for every parameter set.
  """

  _Z = np.linspace(-4.0, 4.0, 33)
  _S = 0.2

  def _solver_force(self, m, d, liftoff=None, **soft):
    mx = mjx.put_model(m)
    if liftoff is not None:
      mx = mx.replace(opt=mx.opt.replace(liftoff_softness=liftoff))
    dx = mjx.forward(mx, mjx.put_data(m, d))
    ctx = solver.Context.create(mx, dx, grad=False)
    return mx, dx, lambda y: solver._update_constraint(mx, dx, ctx.replace(Jaref=y), **soft).efc_force

  def _row_mask(self, f, dx, i, jarefs):
    e = jp.zeros(dx._impl.nefc).at[i].set(1.0)
    base = jp.zeros(dx._impl.nefc)
    return np.array([-float(jax.jvp(f, (base.at[i].set(v),), (e,))[1][i]) for v in jarefs]) / float(dx._impl.efc_D[i])

  def test_frictionloss_in_stick_windows(self):
    curves = []
    for f_, mass, tc, w in ((1.0, 1.0, 0.02, 0.001), (10.0, 0.1, 0.05, 0.01), (0.1, 10.0, 0.02, 0.001)):
      attr = f'frictionloss="{f_}" solreffriction="{tc} 1" solimpfriction="0.9 0.95 {w}"'
      m = mujoco.MjModel.from_xml_string(_COORD_HINGE.format(attr=attr, m=mass))
      d = mujoco.MjData(m); mujoco.mj_forward(m, d)
      i = int(np.nonzero(d.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)[0][0])
      b = d.efc_frictionloss[i] / d.efc_D[i]
      _, dx, f = self._solver_force(m, d, frictionloss_softness=self._S)
      curves.append(self._row_mask(f, dx, i, -self._Z * b))
    np.testing.assert_allclose(curves[1:], [curves[0]] * 2, atol=1e-12)

  def test_limit_in_solimp_widths(self):
    curves = []
    for mass, tc, w, pen in ((1.0, 0.02, 0.001, 0.0005), (10.0, 0.05, 0.01, 0.003), (0.1, 0.02, 0.001, 0.0002)):
      attr = f'limited="true" range="-0.5 0.5" solreflimit="{tc} 1" solimplimit="0.9 0.95 {w}"'
      m = mujoco.MjModel.from_xml_string(_COORD_HINGE.format(attr=attr, m=mass))
      d = mujoco.MjData(m); d.qpos[0] = 0.5 + pen; mujoco.mj_forward(m, d)
      i = int(np.nonzero(d.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)[0][0])
      ell = d.efc_KBIP[i, 0] * d.efc_KBIP[i, 2] * w
      _, dx, f = self._solver_force(m, d, limit_softness=self._S)
      curves.append(self._row_mask(f, dx, i, -self._Z * self._S * ell))
    np.testing.assert_allclose(curves[1:], [curves[0]] * 2, atol=1e-12)
    self.assertAlmostEqual(curves[0][len(self._Z) // 2], 0.5, places=12)

  def test_contact_liftoff_and_stick_slip_coordinates(self):
    lift_curves, slip_curves = [], []
    u = np.linspace(0.0, 2.0, 21)
    for p in (dict(m=1.0, mu=0.5, g=9.81, tc=0.02, w=0.001), dict(m=10.0, mu=1.0, g=30.0, tc=0.05, w=0.005)):
      m = mujoco.MjModel.from_xml_string(_COORD_BOX.format(**p))
      d = _resting_data(m)
      rows = np.nonzero(d.efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)[0]
      ell = float(np.mean(d.efc_KBIP[rows, 0] * d.efc_KBIP[rows, 2])) * p['w']
      mx, dx, f = self._solver_force(m, d, liftoff=self._S, stickslip_softness=self._S)
      (start, ncon, nrow), = solver._contact_edge_groups(mx)
      c0 = jp.arange(start, start + nrow)
      dsum = float(jp.sum(dx._impl.efc_D[c0]))
      base = jp.zeros(dx._impl.nefc).at[jp.arange(start, start + ncon * nrow)].set(-1.0)
      uni = jp.zeros(dx._impl.nefc).at[c0].set(-1.0)
      lift_curves.append([float(jp.sum(jax.jvp(f, (base.at[c0].set(v),), (uni,))[1][c0])) / dsum
                          for v in -self._Z * self._S * ell])
      l0 = self._S * ell / (1 + np.log(2.0))
      pair = jp.zeros(nrow).at[0].set(1.0).at[1].set(-1.0)
      e0 = jp.zeros(dx._impl.nefc).at[start].set(1.0)
      for load in (1e3, 1e5):
        big = load * l0 / self._S
        slip_curves.append([-float(jax.jvp(f, (base.at[c0].set(-big * (jp.ones(nrow) - uu * pair)),), (e0,))[1][start])
                            / float(dx._impl.efc_D[start]) for uu in u])
    np.testing.assert_allclose(lift_curves[1], lift_curves[0], atol=1e-12)
    np.testing.assert_allclose(slip_curves[1:], [slip_curves[0]] * 3, atol=1e-3)
    self.assertTrue(np.all(np.diff(slip_curves[0]) < 0))


class SoftnessScaleTest(absltest.TestCase):
  """efc_softness_scale: each row's own Jaref unit for the softnesses."""

  def test_contact_scale_is_k_imp_width_shared_by_edges(self):
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    d = _resting_data(m)
    dx = mjx.forward(mjx.put_model(m), mjx.put_data(m, d))
    scale = np.asarray(dx._impl.efc_softness_scale)
    contact = np.asarray(dx._impl.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
    self.assertGreater(contact.sum(), 0)
    # MuJoCo's per-row K and I (efc_KBIP), times solimp width (default 0.001).
    k, imp = d.efc_KBIP[:, 0], d.efc_KBIP[:, 2]
    expected = k * imp * m.geom_solimp[0, 2]
    np.testing.assert_allclose(np.sort(scale[contact]), np.sort(expected), rtol=1e-6)
    # All pyramid edges of a contact share solref, solimp and depth: one scale
    # per contact keeps the cone's shape.
    for c in range(d.ncon):
      adr = int(dx._impl.contact.efc_address[c])
      np.testing.assert_allclose(scale[adr : adr + 4], scale[adr], rtol=1e-12)

  def test_frictionloss_scale_is_stick_half_window(self):
    _, dx = _hinge('frictionloss="1"', 0.0)
    dx = mjx.forward(mjx.put_model(mujoco.MjModel.from_xml_string(
        _HINGE_XML.format(attrs='frictionloss="1"'))), dx)
    fl = np.asarray(dx._impl.efc_frictionloss) > 0
    self.assertEqual(fl.sum(), 1)
    np.testing.assert_allclose(
        np.asarray(dx._impl.efc_softness_scale)[fl],
        (np.asarray(dx._impl.efc_frictionloss) / np.asarray(dx._impl.efc_D))[fl],
        rtol=1e-12,
    )


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
    """Default stiffness: k=2 is surrogate-or-converged; k=3 and k=4 match the
    converged gradient everywhere. The per-contact cost's band follows the
    contact's load, so its smoothed residual is more nonlinear than the
    per-row band's and needs one more iteration (per-row: 97% at k=2)."""
    forces = np.linspace(self._THRESHOLD - 3.0, self._THRESHOLD + 3.0, 61)
    # Iteration counts depend on the band width; keep it near Jaref 0.1.
    soft = 0.1 / _box_contact_scale()
    surrogate = _slope(_model(soft), self.dx, 0, forces)
    two = _slope(_refined_model(soft, 2, ls_iterations=8), self.dx, 0, forces)
    three = _slope(_refined_model(soft, 3, ls_iterations=8), self.dx, 0, forces)
    four = _slope(_refined_model(soft, 4, ls_iterations=8), self.dx, 0, forces)
    converged = _slope(_refined_model(soft, 12), self.dx, 0, forces)
    scale = np.max(np.abs(converged))
    is_surrogate = _is_surrogate(two, surrogate, scale)
    is_converged = np.abs(two - converged) <= 1e-4 * scale
    self.assertTrue(np.all(is_surrogate | is_converged))
    np.testing.assert_allclose(three, converged, rtol=0, atol=1e-4 * scale)
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

  def test_linesearch_keeps_converged_step(self):
    """Spare trials after convergence must not move the step: the converged
    Newton point sits on a bracket end, and a strict bracket test bisected
    away from it (8 trials ended 0.54 from the minimizer, 4 within 1e-3)."""
    derivs = lambda a: (a**3 + a - 11.0, 3 * a**2 + 1)  # minimizer a* ~ 2.074 (needs expansion)
    exact = 2.0743407586046700
    for trials in (8, 16, 32):
      alpha = solver._linesearch_1d(derivs, trials, jp.float64)
      self.assertLess(abs(float(alpha) - exact), 1e-12, msg=trials)

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
    is_surrogate = _is_surrogate(two, surrogate, scale)
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
    self.assertTrue(np.all(_is_surrogate(plain, surrogate, scale)
                           | (np.abs(plain - converged) <= 1e-4 * scale)))

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
    in_between = ~_is_surrogate(off, surrogate, scale) & (np.abs(off - converged) > 1e-4 * scale)
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


# Contact, limit and friction-loss rows in one model.
_MIXED_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" iterations="100" tolerance="1e-12" cone="{cone}"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1" friction="0.5 .005 .0001" condim="3"/>
    </body>
    <body pos="2 0 1">
      <joint type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5" frictionloss="1"/>
      <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05" mass="1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
_ALL_SOFTNESSES = dict(stickslip_softness=0.2, liftoff_softness=0.2,
                       limit_softness=0.1, frictionloss_softness=0.1)


def _mixed(cone: str = 'pyramidal', **opt):
  """Box resting on the plane and a hinge just past its limit, after forward."""
  m = mujoco.MjModel.from_xml_string(_MIXED_XML.format(cone=cone))
  d = _resting_data(m)
  d.qpos[7] = 0.501
  d.qvel[:] = 0.0
  mujoco.mj_forward(m, d)
  mx = mjx.put_model(m)
  dx = jax.jit(mjx.forward)(mx, mjx.put_data(m, d))
  return mx.replace(opt=mx.opt.replace(**opt)), dx


_EQ_TENDON_XML = """
<mujoco>
  <option timestep="0.002" iterations="100" tolerance="1e-12"/>
  <worldbody>
    <body pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05" mass="1"/>
      <body name="b" pos=".5 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05" mass="1"/>
      </body>
    </body>
    <body name="c" pos="1 0 1">
      <joint name="j3" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed limited="true" range="-0.3 0.3" frictionloss="0.5">
      <joint joint="j1" coef="1"/><joint joint="j2" coef="-1"/>
    </fixed>
  </tendon>
  <equality>
    <joint joint1="j3" polycoef="0.1 0 0 0 0"/>
    <connect body1="c" body2="b" anchor="0 0 0"/>
  </equality>
</mujoco>
"""


def _old_group_force(y, efc_d, ell, slip, lift):
  """The per-contact force as first implemented (autodiff reference)."""
  n = jp.mean(y, axis=1)
  tau = solver._contact_band(n, ell, slip, lift)[:, None]
  u = -y / tau
  l0 = lift * ell / (1.0 + np.log(2.0))
  coupling = slip * jax.nn.sigmoid(slip * -n / l0) / y.shape[1] * jp.sum(efc_d * tau * solver._tau_term(u), 1)
  return efc_d * tau * solver._softplus(u) + coupling[:, None]


class BackwardImplementationTest(absltest.TestCase):
  """The analytic contact tangent rule and the structured IFT tangent solve
  reproduce independent references (float64)."""

  def test_contact_tangent_rule_matches_autodiff(self):
    rng = np.random.default_rng(0)
    for slip, lift in ((0.2, 0.05), (0.05, 0.01), (1.0, 1.0)):
      for rows in (1, 4, 6, 10):
        for _ in range(10):
          y = jp.asarray(rng.normal(size=(3, rows)) * 10.0 ** rng.uniform(-2, 2, size=(3, 1)))
          dd = jp.asarray(rng.uniform(0.5, 50.0, size=(3, rows)))
          ell = jp.asarray(rng.uniform(0.5, 5.0, size=3))
          tangents = (jp.asarray(rng.normal(size=(3, rows))), jp.asarray(rng.normal(size=(3, rows))))
          f0, t0 = jax.jvp(lambda a, b: _old_group_force(a, b, ell, slip, lift), (y, dd), tangents)
          f1, t1 = jax.jvp(lambda a, b: solver._contact_group_force(a, b, ell, slip, lift), (y, dd), tangents)
          np.testing.assert_allclose(f1, f0, rtol=1e-12, atol=1e-12 * float(jp.max(jp.abs(f0))))
          np.testing.assert_allclose(t1, t0, rtol=1e-10, atol=1e-10 * float(jp.max(jp.abs(t0))))

  def test_contact_second_derivative(self):
    """The tangent rule is itself differentiable (second-order use)."""
    rng = np.random.default_rng(1)
    y = jp.asarray(rng.normal(size=(2, 4)) * 3.0)
    dd = jp.asarray(rng.uniform(0.5, 50.0, size=(2, 4)))
    ell = jp.asarray([2.5, 1.0])
    w = jp.asarray(rng.normal(size=(2, 4)))
    for fn in (_old_group_force, solver._contact_group_force):
      jax.hessian(lambda a: jp.sum(w * fn(a, dd, ell, 0.2, 0.05)))(y)  # compiles and runs
    h0 = jax.hessian(lambda a: jp.sum(w * _old_group_force(a, dd, ell, 0.2, 0.05)))(y)
    h1 = jax.hessian(lambda a: jp.sum(w * solver._contact_group_force(a, dd, ell, 0.2, 0.05)))(y)
    np.testing.assert_allclose(h1, h0, rtol=1e-8, atol=1e-8 * float(jp.max(jp.abs(h0))))

  def _models(self):
    """(label, model, data) for cones, condims, limits, friction loss,
    equalities and tendons, hard and smoothed."""
    out = []
    smooth = dict(stickslip_softness=0.2, liftoff_softness=0.05, limit_softness=0.1, frictionloss_softness=0.1)
    for cone in ('pyramidal', 'elliptic'):
      mx, dx = _mixed(cone)
      soft = dict(smooth) if cone == 'pyramidal' else dict(limit_softness=0.1, frictionloss_softness=0.1)
      out += [(f'mixed {cone} hard', mx, dx), (f'mixed {cone} smoothed', mx.replace(opt=mx.opt.replace(**soft)), dx)]
    m = mujoco.MjModel.from_xml_string(_CONDIM_XML)
    mx = mjx.put_model(m)
    dx = jax.jit(mjx.forward)(mx, mjx.put_data(m, _resting_data(m)))
    out += [('condim 1/3/4/6 hard', mx, dx),
            ('condim 1/3/4/6 smoothed', mx.replace(opt=mx.opt.replace(stickslip_softness=0.2, liftoff_softness=0.05)), dx)]
    m = mujoco.MjModel.from_xml_string(_EQ_TENDON_XML)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.35, 0.0, 0.1]
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = jax.jit(mjx.forward)(mx, mjx.put_data(m, d))
    out += [('equality + tendon hard', mx, dx),
            ('equality + tendon smoothed', mx.replace(opt=mx.opt.replace(limit_softness=0.1, frictionloss_softness=0.1)), dx)]
    return out

  def test_ift_matches_inverse_residual_jacobian(self):
    """d qacc / d qfrc_applied = (dr/dqacc)^-1 at the solution, forward and
    reverse mode, against jacfwd of the residual (no custom_root)."""
    for label, mx, dx in self._models():
      types = set(np.asarray(dx._impl.efc_type).tolist())
      u0 = jp.zeros(mx.nv)
      fwd = lambda u: mjx.forward(mx, dx.replace(qfrc_applied=u)).qacc
      d_sol = jax.jit(mjx.forward)(mx, dx)
      jac_r = jax.jacfwd(lambda q: solver._root(mx, d_sol, q))(d_sol.qacc)
      ref = np.linalg.inv(np.asarray(jac_r))
      for mode, jac in (('forward', jax.jacfwd(fwd)(u0)), ('reverse', jax.jacrev(fwd)(u0))):
        np.testing.assert_allclose(jac, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max(), err_msg=f'{label} {mode}')
      self.assertTrue(types, msg=label)

  def test_ift_with_refinement_matches_smoothed_root(self):
    """With refinement the derivative is that of the smoothed root."""
    for label, mx, dx in self._models():
      if 'smoothed' not in label:
        continue
      refined = mx.replace(opt=mx.opt.replace(refinement_iterations=30, refinement_ls_iterations=8,
                                              refinement_fallback_enable=False))
      hard = mx.replace(opt=mx.opt.replace(stickslip_softness=0.0, liftoff_softness=0.0,
                                           limit_softness=0.0, frictionloss_softness=0.0))
      d_hard = jax.jit(mjx.forward)(hard, dx)
      root = jax.jit(lambda q: solver._root(mx, d_hard, q))
      jac_fn = jax.jit(jax.jacfwd(lambda q: solver._root(mx, d_hard, q)))
      q = d_hard.qacc
      for _ in range(60):
        q = q - jp.linalg.solve(jac_fn(q), root(q))
      ref = np.linalg.inv(np.asarray(jac_fn(q)))
      fwd = lambda u: mjx.forward(refined, dx.replace(qfrc_applied=u)).qacc
      u0 = jp.zeros(mx.nv)
      for mode, jac in (('forward', jax.jacfwd(fwd)(u0)), ('reverse', jax.jacrev(fwd)(u0))):
        np.testing.assert_allclose(jac, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max(), err_msg=f'{label} {mode}')


class RefinementNewtonStepTest(absltest.TestCase):
  """The structured Newton step is exact for both cones; absent row types are pruned."""

  # Elliptic: contact smoothing is pyramidal-only, so limit and friction loss.
  _CASES = (('pyramidal', _ALL_SOFTNESSES),
            ('elliptic', dict(limit_softness=0.1, frictionloss_softness=0.1)))

  def test_newton_step_matches_autodiff(self):
    """Residual, Newton direction (so the Jacobian) and line-search derivatives
    match jacfwd/JVP of the smoothed residual, at points where rows are partly
    switched (fractional smoothed slopes, elliptic cone zones mixed)."""
    for cone, softnesses in self._CASES:
      mx, dx = _mixed(cone, **softnesses)
      root = lambda q, mx=mx, dx=dx: solver._root(mx, dx, q)
      point, step = solver._smoothed_newton(mx, dx, solver._softnesses(mx))
      for seed in range(4):
        q = dx.qacc + 2.0 * jax.random.normal(jax.random.PRNGKey(seed), dx.qacc.shape)
        state = point(q)
        r = state[0]
        direction, derivs = step(q, state)
        np.testing.assert_allclose(r, root(q), rtol=1e-12, atol=1e-12, err_msg=cone)
        expected = -jp.linalg.solve(jax.jacfwd(root)(q), root(q))
        np.testing.assert_allclose(direction, expected, rtol=1e-9, atol=1e-12, err_msg=cone)
        reference = solver._jvp_line_derivs(root, q, direction)
        for alpha in (0.3, 1.0, 2.5):
          got, want = derivs(alpha), reference(alpha)
          scale = abs(float(want[0])) + abs(float(want[1]))
          np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * scale, err_msg=cone)

  def test_newton_step_refinement_matches_generic(self):
    """Same refined point as the generic (jacfwd + LU + JVP line search) path."""
    for cone, softnesses in self._CASES:
      mx, dx = _mixed(cone, **softnesses)
      root = lambda q, mx=mx, dx=dx: solver._root(mx, dx, q)
      start = dx.qacc + 0.5
      point, step = solver._smoothed_newton(mx, dx, solver._softnesses(mx))
      for ls in (0, 4):
        fast, _ = solver._refine_smoothed_root(root, start, 3, 1.0, ls_iterations=ls,
                                               fallback_enable=False, newton=(point, step))
        generic, _ = solver._refine_smoothed_root(root, start, 3, 1.0, ls_iterations=ls,
                                                  fallback_enable=False)
        np.testing.assert_allclose(fast, generic, rtol=1e-10, atol=1e-12, err_msg=cone)

  def test_elliptic_refinement_converges_to_smoothed_root(self):
    """Elliptic cone (limit and friction-loss smoothing): refinement gives the
    derivative of the smoothed root, like the pyramidal path."""
    softnesses = dict(limit_softness=0.1, frictionloss_softness=0.1)
    mx, dx = _mixed('elliptic', refinement_iterations=12, **softnesses)
    soft_only = mx.replace(opt=mx.opt.replace(refinement_iterations=0))
    hard = mx.replace(opt=mx.opt.replace(
        refinement_iterations=0, limit_softness=0.0,
        frictionloss_softness=0.0))
    dof = 6  # the hinge
    def smoothed_root(torque):
      d = dx.replace(qfrc_applied=dx.qfrc_applied.at[dof].set(torque))
      d = mjx.forward(hard, d)
      root = lambda q: solver._root(soft_only, d, q)
      q, _ = solver._refine_smoothed_root(root, d.qacc, 30, 1.0, fallback_enable=False)
      return q[dof]

    torques = np.linspace(-1.0, 1.0, 5)
    grad = _slope(mx, dx, dof, torques)
    h = 1e-5
    fd = np.array([(smoothed_root(t + h) - smoothed_root(t - h)) / (2 * h) for t in torques])
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-7)

  def test_absent_row_types_are_pruned(self):
    """The box has contact rows only: limit and friction-loss softnesses are
    dropped, so they alone do not trigger a refinement (bit-identical to hard)."""
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    dx = mjx.forward(mjx.put_model(m), mjx.put_data(m, _resting_data(m)))
    mx = _refined_model(0.0, 4).replace(opt=_refined_model(0.0, 4).opt.replace(
        limit_softness=0.1, frictionloss_softness=0.1))
    self.assertEqual(solver._effective_softnesses(mx, dx), (0.0, 0.0, 0.0, 0.0))
    mx_all = mx.replace(opt=mx.opt.replace(stickslip_softness=0.2, liftoff_softness=0.05))
    self.assertEqual(solver._effective_softnesses(mx_all, dx), (0.2, 0.05, 0.0, 0.0))
    forces = np.linspace(-2.0, 2.0, 21)
    np.testing.assert_array_equal(_slope(_model(0.0), dx, 0, forces), _slope(mx, dx, 0, forces))


if __name__ == '__main__':
  absltest.main()
