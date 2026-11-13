"""Gradient checks for the solver backends (``opt.grad_backend``).

Converged, regime-stable problems only: here the implicit (ift) gradient,
the unrolled (scan) gradient and central finite differences of the converged
solve must agree. Runs in float64.
"""


from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
from mujoco.mjx._src.types import DisableBit
import numpy as np

_SPHERE = """
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

_BOX = """
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

_ARM = """
<mujoco>
  <option timestep="0.004"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
    <body pos="0 0 0.5">
      <joint name="j0" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"
            contype="0" conaffinity="0"/>
      <body pos="0.3 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"
               limited="true" range="-150 150"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" mass="0.5"
              contype="0" conaffinity="0"/>
        <geom type="sphere" pos="0 0 -0.47" size="0.035" mass="0.1"
              contype="1" conaffinity="1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="j0"/><motor joint="j1"/></actuator>
</mujoco>
"""


def _scene(xml, cone, settle, qvel0=None, ctrl=None):
  m = mujoco.MjModel.from_xml_string(xml)
  m.opt.cone = cone
  m.opt.iterations = 100
  m.opt.ls_iterations = 50
  m.opt.tolerance = 1e-14
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
  d = mujoco.MjData(m)
  if ctrl is not None:
    d.ctrl[:] = ctrl
  for _ in range(settle):
    mujoco.mj_step(m, d)
  if qvel0 is not None:
    d.qvel[:] = qvel0
  mujoco.mj_forward(m, d)
  return m, d


def _loss(mx, dx, kind, backend):
  """Loss through ``mjx.forward`` with ``opt.grad_backend = backend``."""
  mx = mx.replace(opt=mx.opt.replace(grad_backend=backend))
  rng = np.random.default_rng(0)
  wa = jp.asarray(rng.normal(size=mx.nv))
  wc = jp.asarray(rng.normal(size=mx.nv))

  def f(theta):
    if kind == 'qvel':
      d = dx.replace(qvel=dx.qvel + theta)
    elif kind == 'qfrc_applied':
      d = dx.replace(qfrc_applied=dx.qfrc_applied + theta)
    elif kind == 'ctrl':
      d = dx.replace(ctrl=dx.ctrl + theta)
    else:
      raise ValueError(kind)
    d = mjx.forward(mx, d)
    return jp.dot(wa, d.qacc) + 0.1 * jp.dot(wc, d.qfrc_constraint)

  return f


def _compile(fn, *args):
  return jax.jit(fn).lower(*args).compile()


def _fd(f, x0, eps):
  eye = jp.eye(x0.shape[0], dtype=x0.dtype)
  return jax.vmap(lambda v: (f(x0 + eps * v) - f(x0 - eps * v)) / (2 * eps))(
      eye
  )


def _rel(a, b):
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


class SolverGradTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._x64 = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)

  def tearDown(self):
    jax.config.update('jax_enable_x64', self._x64)
    super().tearDown()

  @parameterized.product(
      scene=('sphere', 'box_stick', 'box_slide', 'arm'),
      cone=(mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC),
      kind=('qvel', 'qfrc_applied'),
  )
  def test_ift_matches_unrolled_and_fd(self, scene, cone, kind):
    if scene == 'sphere':
      m, d = _scene(_SPHERE, cone, 50)
    elif scene == 'box_stick':
      m, d = _scene(_BOX, cone, 50)
      d.qfrc_applied[:] = [2.0, 1.0, 0, 0, 0, 0.3]
    elif scene == 'box_slide':
      m, d = _scene(_BOX, cone, 50, qvel0=[1.5, 0, 0, 0, 0, 0])
    else:
      m, d = _scene(_ARM, cone, 100, ctrl=[0.5, -0.2])
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)
    f = lambda backend: _loss(mx, dx, kind, backend)
    th = jp.zeros(mx.nv)

    g_ift = np.asarray(_compile(jax.grad(f('ift')), th)(th))
    fds = [np.asarray(_compile(lambda x, e=e: _fd(f('while'), x, e), th)(th))
           for e in (1e-4, 1e-5, 1e-6)]
    fd_err = min(_rel(g_ift, fd) for fd in fds)

    self.assertTrue(np.all(np.isfinite(g_ift)))
    self.assertLess(fd_err, 1e-5, f'ift vs fd: {fd_err}')

    # Unrolled reference. With warmstart the solve can start at the solution
    # (static contact), where every line search reports no improvement and
    # the masked update carries no derivative, so the unrolled gradient drops
    # the solver sensitivity. Start from qacc_smooth instead.
    mx_cold = mx.tree_replace({
        'opt.disableflags': mx.opt.disableflags | DisableBit.WARMSTART
    })
    f_cold = lambda backend: _loss(mx_cold, dx, kind, backend)
    g_scan = np.asarray(_compile(jax.grad(f_cold('scan')), th)(th))
    g_ift_cold = np.asarray(_compile(jax.grad(f_cold('ift')), th)(th))
    self.assertLess(_rel(g_ift_cold, g_ift), 1e-8)
    if cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
      # Elliptic cones: once the line search stops improving, the masked
      # update freezes the derivative of the last improving Newton step, whose
      # Hessian was taken in a different cone zone (box_slide: ~6% error for
      # every K). Pyramidal Hessians are constant within an active set.
      self.assertLess(_rel(g_scan, g_ift), 1e-6)

  def test_unrolled_drops_sensitivity_at_converged_warmstart(self):
    """Documents why ift is preferred for quasi-static contact."""
    m, d = _scene(_ARM, mujoco.mjtCone.mjCONE_PYRAMIDAL, 100, ctrl=[0.5, -0.2])
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)
    f = lambda backend: _loss(mx, dx, 'qvel', backend)
    th = jp.zeros(mx.nv)
    mx_while = mx.replace(opt=mx.opt.replace(grad_backend='while'))
    niter = jax.jit(mjx.forward)(mx_while, dx)._impl.solver_niter
    g_ift = np.asarray(_compile(jax.grad(f('ift')), th)(th))
    g_scan = np.asarray(_compile(jax.grad(f('scan')), th)(th))
    fd = np.asarray(_compile(lambda x: _fd(f('while'), x, 1e-5), th)(th))

    self.assertEqual(int(niter), 0)  # warmstart is already the solution
    self.assertLess(_rel(g_ift, fd), 1e-6)
    self.assertGreater(_rel(g_scan, fd), 1e-2)

  def test_ift_vmap(self):
    m, d = _scene(_BOX, mujoco.mjtCone.mjCONE_PYRAMIDAL, 50)
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)
    f = _loss(mx, dx, 'qvel', 'ift')
    th = jp.zeros((4, mx.nv)).at[:, 0].set(jp.linspace(0.0, 0.3, 4))
    g_b = np.asarray(_compile(jax.vmap(jax.grad(f)), th)(th))
    for i in range(4):
      g_i = np.asarray(_compile(jax.grad(f), th[i])(th[i]))
      np.testing.assert_allclose(g_b[i], g_i, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
  absltest.main()
