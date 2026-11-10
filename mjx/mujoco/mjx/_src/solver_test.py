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
"""Tests for constraint functions."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
from mujoco.mjx._src import test_util
from mujoco.mjx._src.types import ConeType
import numpy as np


# tolerance for difference between MuJoCo and MJX solver calculations,
# mostly due to float precision
_TOLERANCE = 5e-3


def _solve_with_backend(backend, mx, dx):
  env = {} if backend is None else {'MJX_GRAD_BACKEND': backend}
  with mock.patch.dict(os.environ, env, clear=False):
    if backend is None:
      os.environ.pop('MJX_GRAD_BACKEND', None)
    return jax.jit(lambda m, d: mjx.solve(m, d))(mx, dx)


def _assert_eq(a, b, name, tol=_TOLERANCE):
  tol = tol * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr, tol=_TOLERANCE):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr, tol=tol)


class SolverTest(parameterized.TestCase):

  def _load_constraints(self, solver_, cone, iterations=4):
    m = test_util.load_test_file('constraints.xml')
    m.opt.solver = solver_
    m.opt.cone = cone
    m.opt.iterations = iterations
    if iterations < 5:
      m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 2)
    mujoco.mj_forward(m, d)
    return m, d

  @parameterized.parameters(
      # these scene challenges the solver, with CG you need to crank up
      # the iterations, otherwise it diverges
      (mujoco.mjtSolver.mjSOL_CG, mujoco.mjtCone.mjCONE_PYRAMIDAL, 100),
      (mujoco.mjtSolver.mjSOL_CG, mujoco.mjtCone.mjCONE_ELLIPTIC, 100),
      # Newton converges much more quickly, lower iterations to demonstrate
      # mgrad is being calculated optimally
      (mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_PYRAMIDAL, 2),
      (mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_ELLIPTIC, 2),
  )
  def test_solver(self, solver_, cone, iterations):
    """Test newton, CG solver with pyramidal, elliptic cones."""
    m = test_util.load_test_file('constraints.xml')
    m.opt.solver = solver_
    m.opt.cone = cone
    m.opt.iterations = iterations

    # with islanding on, MuJoCo CG converges *much* faster,
    # too fast for the low-iteration comparison to be meaningful,
    # so we disable islanding at low iteration count
    if iterations < 5:
      m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND

    d = mujoco.MjData(m)

    def cost(qacc):
      jaref = np.zeros(d.nefc, dtype=float)
      cost = np.zeros(1)
      mujoco.mj_mulJacVec(m, d, jaref, qacc)
      mujoco.mj_constraintUpdate(m, d, jaref - d.efc_aref, cost, 0)
      return cost

    # sample a mix of active/inactive constraints at different timesteps
    for key in range(0, 3):
      mujoco.mj_resetDataKeyframe(m, d, key)
      mujoco.mj_step(m, d)  # step to generate warmstart

      # compare costs
      mj_cost = cost(d.qacc)
      ctx = solver.Context.create(mjx.put_model(m), mjx.put_data(m, d))
      mjx_cost = ctx.cost - ctx.gauss
      _assert_eq(mj_cost, mjx_cost, 'cost')

      mujoco.mj_forward(m, d)
      dx = jax.jit(mjx.solve)(mjx.put_model(m), mjx.put_data(m, d))

      # MJX finds very similar solutions with the newton solver
      if solver_ == mujoco.mjtSolver.mjSOL_NEWTON:
        nnz = dx._impl.efc_J.any(axis=1)
        _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force')
        _assert_attr_eq(d, dx, 'qfrc_constraint')
        _assert_attr_eq(d, dx, 'qacc')

      # both CG and Newton find costs that are nearly the same as MuJoCo, often
      # lower (due to slight differences in the MJX linsearch algorithm)
      mj_cost = cost(d.qacc)
      mjx_cost = cost(dx.qacc)
      self.assertLess(mjx_cost, mj_cost * 1.015)

  def test_no_warmstart(self):
    """Test no warmstart."""
    m = test_util.load_test_file('constraints.xml')
    d = mujoco.MjData(m)
    # significant constraint forces keyframe 2
    mujoco.mj_resetDataKeyframe(m, d, 2)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = jax.jit(mjx.solve)(mx, mjx.put_data(m, d))
    nnz = dx._impl.efc_J.any(axis=1)
    # even without warmstart, newton converges quickly
    _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force', tol=2e-4)

  def test_sparse(self):
    """Test solver works with sparse mass matrices."""
    m = test_util.load_test_file('constraints.xml')
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    # significant constraint forces keyframe 2
    mujoco.mj_resetDataKeyframe(m, d, 2)

    mujoco.mj_forward(m, d)

    dx = jax.jit(mjx.solve)(mjx.put_model(m), mjx.put_data(m, d))

    _assert_attr_eq(d, dx, 'qacc')
    _assert_attr_eq(d, dx, 'qfrc_constraint')
    nnz = dx._impl.efc_J.any(axis=1)
    _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force')

  def test_quad_frictionloss(self):
    """Test a case with quadratic frictionloss constraints."""
    m = test_util.load_test_file('quadratic_frictionloss.xml')
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    dx = jax.jit(mjx.solve)(mjx.put_model(m), mjx.put_data(m, d))

    _assert_attr_eq(d, dx, 'qacc')
    _assert_attr_eq(d, dx, 'qfrc_constraint')
    nnz = dx._impl.efc_J.any(axis=1)
    _assert_eq(d.efc_force, dx._impl.efc_force[nnz], 'efc_force')

  # TODO(taylorhowell): condim=1 with ConeType.ELLIPTIC
  @parameterized.product(condim=(3, 4, 6), cone=tuple(ConeType))
  def test_condim(self, condim, cone):
    """Test contact dimension."""
    m = mujoco.MjModel.from_xml_string(f"""
       <mujoco>
          <worldbody>
            <geom size="0 0 1e-5" type="plane" condim="1"/>
            <body pos="0 0 0.09">
              <freejoint/>
              <geom size="0.1" condim="{condim}"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    m.opt.cone = cone
    solver.solve(mjx.put_model(m), mjx.put_data(m, mujoco.MjData(m)))

  @parameterized.parameters('ift', 'while')
  def test_backend_matches_unset(self, backend):
    m, d = self._load_constraints(
        mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_PYRAMIDAL
    )
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)

    unset = _solve_with_backend(None, mx, dx)
    other = _solve_with_backend(backend, mx, dx)

    _assert_attr_eq(unset, other, 'qacc')
    _assert_attr_eq(unset, other, 'qfrc_constraint')
    _assert_eq(unset._impl.efc_force, other._impl.efc_force, 'efc_force')

  @parameterized.product(
      backend=('unrolled', 'while', 'scan', 'ift'),
      cone=(mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC),
  )
  def test_backend_matches_mujoco(self, backend, cone):
    """Every backend reproduces MuJoCo C at a converged Newton solve."""
    m, d = self._load_constraints(mujoco.mjtSolver.mjSOL_NEWTON, cone, 10)
    mx = mjx.put_model(m)
    # MJX does not support islanding; give C the same non-island problem.
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    mujoco.mj_forward(m, d)
    dx = mjx.put_data(m, d).replace(qacc=d.qacc_warmstart)

    out = _solve_with_backend(backend, mx, dx)
    ref = _solve_with_backend('while', mx, dx)

    _assert_attr_eq(out, d, 'qacc')
    if cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
      # For elliptic cones on this scene, one qfrc_constraint entry differs
      # from C by ~10% for every backend (pre-existing MJX behaviour).
      _assert_attr_eq(out, d, 'qfrc_constraint')
    _assert_attr_eq(out, ref, 'qfrc_constraint')
    _assert_eq(out._impl.efc_force, ref._impl.efc_force, 'efc_force')
    self.assertGreater(int(out._impl.solver_niter), 0)

  def test_invalid_backend(self):
    m, d = self._load_constraints(
        mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_PYRAMIDAL
    )

    with mock.patch.dict(os.environ, {'MJX_GRAD_BACKEND': 'not-a-backend'}):
      with self.assertRaisesRegex(ValueError, 'MJX_GRAD_BACKEND'):
        solver.solve(mjx.put_model(m), mjx.put_data(m, d))

  @parameterized.product(
      backend=('unrolled', 'scan', 'ift'),
      solver_=(
          mujoco.mjtSolver.mjSOL_NEWTON,
          mujoco.mjtSolver.mjSOL_CG,
      ),
      cone=(
          mujoco.mjtCone.mjCONE_PYRAMIDAL,
          mujoco.mjtCone.mjCONE_ELLIPTIC,
      ),
  )
  def test_backend_reverse_mode(self, backend, solver_, cone):
    m, d = self._load_constraints(solver_, cone)
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)
    weights = jp.linspace(0.1, 1.0, dx.qacc.shape[0])

    with mock.patch.dict(os.environ, {'MJX_GRAD_BACKEND': backend}):

      def loss(qfrc_smooth):
        dy = mjx.solve(mx, dx.replace(qfrc_smooth=qfrc_smooth))
        return jp.dot(weights, dy.qacc)

      grad = jax.jit(jax.grad(loss))(dx.qfrc_smooth)
      grad.block_until_ready()

    self.assertTrue(np.all(np.isfinite(grad)))

  def test_ift_no_warmstart(self):
    m, d = self._load_constraints(
        mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtCone.mjCONE_PYRAMIDAL
    )
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)

    out = _solve_with_backend('ift', mx, dx)

    self.assertTrue(np.all(np.isfinite(out.qacc)))
    self.assertTrue(np.all(np.isfinite(out.qfrc_constraint)))
    self.assertTrue(np.all(np.isfinite(out._impl.efc_force)))


if __name__ == '__main__':
  absltest.main()
