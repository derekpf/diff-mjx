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
"""Constraint solvers."""

import numbers

import jax
from jax import numpy as jp
import mujoco
import softjax as sj
from mujoco.mjx._src import math
from mujoco.mjx._src import smooth
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src.types import ConeType
from mujoco.mjx._src.types import ConstraintType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DataJAX
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ModelJAX
from mujoco.mjx._src.types import OptionJAX
from mujoco.mjx._src.types import SolverType
# pylint: enable=g-importing-member


class Context(PyTreeNode):
  """Data updated during each solver iteration.

  Attributes:
    qacc: acceleration (from Data)                    (nv,)
    qfrc_constraint: constraint force (from Data)     (nv,)
    Jaref: Jac*qacc - aref                            (nefc,)
    efc_force: constraint force in constraint space   (nefc,)
    Ma: M*qacc                                        (nv,)
    grad: gradient of master cost                     (nv,)
    Mgrad: M / grad                                   (nv,)
    search: linesearch vector                         (nv,)
    gauss: gauss Cost
    cost: constraint + Gauss cost
    prev_cost: cost from previous iter
    solver_niter: number of solver iterations
    active: active (quadratic) constraints            (nefc,)
    fri: friction of regularized cone                 (num(con.dim > 1), 6)
    dm: regularized constraint mass                   (num(con.dim > 1))
    u: friction cone (normal and tangents)            (num(con.dim > 1), 6)
    h: cone hessian                                   (num(con.dim > 1), 6, 6)
  """

  qacc: jax.Array
  qfrc_constraint: jax.Array
  Jaref: jax.Array  # pylint: disable=invalid-name
  efc_force: jax.Array
  Ma: jax.Array  # pylint: disable=invalid-name
  grad: jax.Array
  Mgrad: jax.Array  # pylint: disable=invalid-name
  search: jax.Array
  gauss: jax.Array
  cost: jax.Array
  prev_cost: jax.Array
  solver_niter: jax.Array
  active: jax.Array
  fri: jax.Array
  dm: jax.Array
  u: jax.Array
  h: jax.Array

  @classmethod
  def create(
      cls,
      m: Model,
      d: Data,
      grad: bool = True,
      contact_softness: float = 0.0,
      limit_softness: float = 0.0,
      frictionloss_softness: float = 0.0,
  ) -> 'Context':
    if not isinstance(d._impl, DataJAX) or not isinstance(
        m.opt._impl, OptionJAX
    ):
      raise ValueError(
          'Constraint context requires JAX backend implementation.'
      )

    jaref = d._impl.efc_J @ d.qacc - d._impl.efc_aref
    # TODO(robotics-team): determine nv at which sparse mul is faster
    ma = support.mul_m(m, d, d.qacc)
    nv_0 = jp.zeros(m.nv)
    fri = 0.0
    if m.opt.cone == ConeType.ELLIPTIC:
      friction = d._impl.contact.friction[d._impl.contact.dim > 1]
      dim = d._impl.contact.dim[d._impl.contact.dim > 1]
      mu = friction[:, 0] / jp.sqrt(m.opt.impratio)
      fri = jp.concatenate((mu[:, None], friction), axis=1)
      for condim in (3, 4, 6):
        fri = fri.at[dim == condim, condim:].set(0)

    ctx = Context(
        qacc=d.qacc,
        qfrc_constraint=d.qfrc_constraint,
        Jaref=jaref,
        efc_force=d._impl.efc_force,
        Ma=ma,
        grad=nv_0,
        Mgrad=nv_0,
        search=nv_0,
        gauss=0.0,
        cost=jp.inf,
        prev_cost=0.0,
        solver_niter=0,
        active=0.0,
        fri=fri,
        dm=0.0,
        u=0.0,
        h=0.0,
    )
    ctx = _update_constraint(
        m, d, ctx, contact_softness=contact_softness, limit_softness=limit_softness,
        frictionloss_softness=frictionloss_softness,
    )
    if grad:
      ctx = _update_gradient(m, d, ctx)
      ctx = ctx.replace(search=-ctx.Mgrad)  # start with preconditioned gradient

    return ctx


class _LSPoint(PyTreeNode):
  """Line search evaluation point.

  Attributes:
    alpha: step size that reduces f(x + alpha * p) given search direction p
    cost: line search cost
    deriv_0: first derivative of quadratic
    deriv_1: second derivative of quadratic
  """

  alpha: jax.Array
  cost: jax.Array
  deriv_0: jax.Array
  deriv_1: jax.Array

  @classmethod
  def create(
      cls,
      m: Model,
      d: Data,
      ctx: Context,
      alpha: jax.Array,
      jv: jax.Array,
      quad: jax.Array,
      quad_gauss: jax.Array,
      uu: jax.Array,
      v0: jax.Array,
      uv: jax.Array,
      vv: jax.Array,
  ) -> '_LSPoint':
    """Creates a linesearch point with first and second derivatives."""
    # roughly corresponds to CGEval in mujoco/src/engine/engine_solver.c
    if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
      raise ValueError('LSPoint requires JAX backend implementation.')

    cost, deriv_0, deriv_1 = 0.0, 0.0, 0.0
    quad_total = quad_gauss
    x = ctx.Jaref + alpha * jv
    active = (x < 0).at[: d._impl.ne + d._impl.nf].set(True)

    dof_fl, ten_fl = m._impl.dof_hasfrictionloss, m._impl.tendon_hasfrictionloss
    if (dof_fl.any() or ten_fl.any()) and not (
        m.opt.disableflags & DisableBit.FRICTIONLOSS
    ):
      f = d._impl.efc_frictionloss
      r = 1.0 / (d._impl.efc_D + (d._impl.efc_D == 0.0) * mujoco.mjMINVAL)
      rf, z = r * f, jp.zeros_like(f)
      linear_neg = (x <= -rf)[:, None]
      linear_pos = (x >= rf)[:, None]
      qf = linear_neg * jp.array([f * (-0.5 * rf - ctx.Jaref), -f * jv, z]).T
      qf += linear_pos * jp.array([f * (-0.5 * rf + ctx.Jaref), f * jv, z]).T
      quad = jp.where((linear_neg | linear_pos) & (f[:, None] > 0), qf, quad)

    if m.opt.cone == ConeType.ELLIPTIC:
      mu, u0 = ctx.fri[:, 0], ctx.u[:, 0]
      n = u0 + alpha * v0
      tsqr = uu + alpha * (2 * uv + alpha * vv)
      t = sj.sqrt(tsqr)  # tangential force

      bottom_zone = ((tsqr <= 0) & (n < 0)) | ((tsqr > 0) & ((mu * n + t) <= 0))
      middle_zone = (tsqr > 0) & (n < (mu * t)) & ((mu * n + t) > 0)

      # quadratic cost for equality, friction, limits, frictionless contacts
      dim1 = d._impl.contact.efc_address[d._impl.contact.dim == 1]
      nefl = d._impl.ne + d._impl.nf + d._impl.nl
      active = active.at[nefl:].set(False).at[dim1].set(active[dim1])
      quad_efld = jax.vmap(jp.multiply)(quad, active)
      quad_total += jp.sum(quad_efld, axis=0)
      # elliptic bottom zone: quadratic cost
      efc_elliptic = d._impl.contact.efc_address[d._impl.contact.dim > 1]
      quad_c = jax.vmap(jp.multiply)(quad[efc_elliptic], bottom_zone)
      quad_total += jp.sum(quad_c, axis=0)
      # elliptic middle zone
      t += (t == 0) * mujoco.mjMINVAL
      tsqr += (tsqr == 0) * mujoco.mjMINVAL
      n1 = v0
      t1 = (uv + alpha * vv) / t
      t2 = vv / t - (uv + alpha * vv) * t1 / tsqr
      dm = ctx.dm * middle_zone
      nmt = n - mu * t
      cost = 0.5 * jp.sum(dm * jp.square(nmt))
      deriv_0 = jp.sum(dm * nmt * (n1 - mu * t1))
      deriv_1 = jp.sum(dm * (jp.square(n1 - mu * t1) - nmt * mu * t2))
    elif m.opt.cone == ConeType.PYRAMIDAL:
      quad = jax.vmap(jp.multiply)(quad, active)  # only active
      quad_total += jp.sum(quad, axis=0)
    else:
      raise NotImplementedError(f'unsupported cone type: {m.opt.cone}')

    alpha_sq = alpha * alpha
    cost += alpha_sq * quad_total[2] + alpha * quad_total[1] + quad_total[0]
    deriv_0 += 2 * alpha * quad_total[2] + quad_total[1]
    deriv_1 += 2 * quad_total[2] + (quad_total[2] == 0) * mujoco.mjMINVAL

    return _LSPoint(alpha=alpha, cost=cost, deriv_0=deriv_0, deriv_1=deriv_1)


class _LSContext(PyTreeNode):
  """Data updated during each line search iteration.

  Attributes:
    lo: low point bounding the line search interval
    hi: high point bounding the line search interval
    swap: True if low or hi was swapped in the line search iteration
    ls_iter: number of linesearch iterations
  """

  lo: _LSPoint
  hi: _LSPoint
  swap: jax.Array
  ls_iter: jax.Array


def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""

  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]


_ACTIVESET_SOFTNESS_FIELDS = (
    'contact_activeset_softness',
    'limit_activeset_softness',
    'frictionloss_activeset_softness',
)


def _activeset_softnesses(m: Model) -> tuple[float, float, float]:
  """Returns the validated active-set smoothing widths as Python floats.

  The widths select code paths at trace time, so they must be concrete,
  finite, non-negative scalars (0 disables smoothing).
  """
  softnesses = []
  for name in _ACTIVESET_SOFTNESS_FIELDS:
    value = getattr(m.opt, name)
    if isinstance(value, jax.Array):
      # A jax.Array option field is a pytree leaf, so it is traced under jit.
      raise TypeError(
          f'opt.{name} must be a Python float, not a jax.Array; it is static '
          'and selects code paths at trace time.'
      )
    try:
      softness = float(value)
    except TypeError as e:
      raise TypeError(f'opt.{name} must be a scalar, got {value!r}.') from e
    if not (softness >= 0.0 and softness < float('inf')):
      raise ValueError(f'opt.{name} must be finite and >= 0, got {softness}.')
    softnesses.append(softness)
  return tuple(softnesses)


def _static_count_option(m: Model, name: str) -> int:
  """Returns the validated static int option ``opt.<name>`` (>= 0)."""
  value = getattr(m.opt, name)
  if isinstance(value, (bool, jax.Array)) or not isinstance(
      value, numbers.Integral
  ):
    raise TypeError(
        f'opt.{name} must be a Python int, got {value!r}; it is static and '
        'sets a loop length at trace time.'
    )
  if value < 0:
    raise ValueError(f'opt.{name} must be >= 0, got {value}.')
  return int(value)


def _refinement_iterations(m: Model) -> int:
  """Returns the validated ``opt.refinement_iterations`` (static int >= 0)."""
  return _static_count_option(m, 'refinement_iterations')


def _refinement_ls_iterations(m: Model) -> int:
  """Returns the validated ``opt.refinement_ls_iterations``.

  Mirrors ``opt.ls_iterations`` for the refinement's smoothed line search.
  0 disables the line search (plain Newton, full steps).
  """
  return _static_count_option(m, 'refinement_ls_iterations')


def _refinement_tolerance(m: Model) -> float:
  """Returns the validated ``opt.refinement_tolerance`` (static float >= 0)."""
  value = m.opt.refinement_tolerance
  if isinstance(value, (bool, jax.Array)) or not isinstance(value, numbers.Real):
    raise TypeError(
        f'opt.refinement_tolerance must be a Python float, got {value!r}; '
        'it is static.'
    )
  tolerance = float(value)
  if not (tolerance >= 0.0 and tolerance < float('inf')):
    raise ValueError(
        f'opt.refinement_tolerance must be finite and >= 0, got {tolerance}.'
    )
  return tolerance


def _refinement_fallback_enable(m: Model) -> bool:
  """Returns the validated ``opt.refinement_fallback_enable`` (static bool)."""
  value = m.opt.refinement_fallback_enable
  if not isinstance(value, bool):
    raise TypeError(
        'opt.refinement_fallback_enable must be a Python bool, got '
        f'{value!r}; it is static.'
    )
  return value




def _smoothed_linesearch(
    root, qacc: jax.Array, direction: jax.Array, iterations: int
) -> jax.Array:
  """Step size along ``direction`` that minimizes the smoothed cost.

  ``root`` is the gradient of a convex cost (the smoothed stationarity
  residual), so along the line ``phi'(a) = root(q + a p) . p`` is increasing
  and ``phi''(a) = p^T (d root/d q) p > 0``. Like ``_linesearch``, this is a
  bracketed Newton iteration on ``phi'``, but it evaluates the smoothed cost's
  derivatives directly (one JVP per iteration) instead of the hard cost's
  piecewise-quadratic coefficients.

  Starting from the full Newton step ``a = 1``, it expands (``a <- max(Newton,
  2a)``) until ``phi'`` changes sign, because on the smoothed residual Newton
  typically undershoots: rows that switch off along the step lower the
  curvature, so the minimizer lies beyond ``a = 1``. Once bracketed, it takes
  Newton steps, bisecting when a Newton step leaves the bracket. If no sign
  change is found, it returns the furthest point with ``phi' < 0``, which still
  decreases the cost.
  """
  dtype = qacc.dtype
  inf = jp.asarray(jp.inf, dtype)

  def derivs(alpha):
    r, jr = jax.jvp(root, (qacc + alpha * direction,), (direction,))
    return jp.dot(r, direction), jp.dot(jr, direction)

  def body(_, carry):
    lo, hi, alpha = carry
    d1, d2 = derivs(alpha)
    lo = jp.where(d1 < 0, alpha, lo)
    hi = jp.where(d1 > 0, alpha, hi)
    newton = alpha - d1 / jp.maximum(d2, jp.finfo(dtype).tiny)
    bracketed = jp.isfinite(hi)
    inside = (newton > lo) & (newton < hi)
    next_bracketed = jp.where(inside, newton, 0.5 * (lo + hi))
    next_expand = jp.maximum(newton, 2 * alpha)
    alpha = jp.where(d1 == 0, alpha, jp.where(bracketed, next_bracketed, next_expand))
    return lo, hi, alpha

  init = (jp.asarray(0.0, dtype), inf, jp.asarray(1.0, dtype))
  lo, hi, alpha = jax.lax.fori_loop(0, iterations, body, init)
  # Bracketed: the Newton/bisection iterate lies in [lo, hi]. Not bracketed:
  # every point tried had phi' < 0, so the furthest one (lo) is safe.
  return jp.where(jp.isfinite(hi), alpha, jp.maximum(lo, jp.asarray(1.0, dtype)))


def _refine_smoothed_root(
    root,
    qacc_hard: jax.Array,
    iterations: int,
    tolerance: float,
    floor: jax.Array | float = 0.0,
    ls_iterations: int = 0,
    fallback_enable: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Line-searched Newton from the hard solution toward the smoothed root.

  The result is only the gradient's evaluation point; forward values stay
  hard. Each Newton step uses ``ls_iterations`` smoothed line-search trials;
  with 0 the full Newton step is taken (plain Newton, no line search cost).
  Returns ``(qacc_eval, accepted)``.

  With ``fallback_enable`` (the default), the refined point is accepted only if it is
  finite and its smoothed residual satisfies
  ``|r| <= max(tolerance * |r(qacc_hard)|, floor)``, where ``floor`` is the
  precision floor of the residual. Otherwise ``qacc_hard`` is returned, which
  gives the surrogate gradient, because a partially converged point can give a
  gradient worse than the surrogate. With ``fallback_enable=False`` the tolerance
  check is skipped and the refined point is used whenever it is finite;
  non-finite points always fall back, since they would only give NaN
  gradients.
  """

  def body(qacc, _):
    direction = -jp.linalg.solve(jax.jacfwd(root)(qacc), root(qacc))
    alpha = _smoothed_linesearch(root, qacc, direction, ls_iterations)
    return qacc + alpha * direction, None

  qacc, _ = jax.lax.scan(body, qacc_hard, None, length=iterations)
  r_hard = jp.linalg.norm(root(qacc_hard))
  r_eval = jp.linalg.norm(root(qacc))
  accepted = jp.isfinite(r_eval) & jp.all(jp.isfinite(qacc))
  if fallback_enable:
    accepted &= r_eval <= jp.maximum(tolerance * r_hard, floor)
  return jp.where(accepted, qacc, qacc_hard), accepted


def _update_constraint(
    m: Model,
    d: Data,
    ctx: Context,
    contact_softness: float = 0.0,
    limit_softness: float = 0.0,
    frictionloss_softness: float = 0.0,
) -> Context:
  """Updates constraint force and resulting cost given last solver iteration.

  Corresponds to CGupdateConstraint in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest qacc and smooth terms
    ctx: current solver context
    contact_softness: if > 0, the force of nominal contact rows (in contact,
      ``efc_pos < efc_margin``) uses ``D * sj.relu(-Jaref, softness)`` (smooth,
      non-gated: ``softness * softplus(-Jaref / softness)``, slope ``sigmoid``) instead
      of ``D * max(0, -Jaref)``. Used only by the IFT residual (``_root``), so
      the forward solve is unchanged; pyramidal cone only. Non-gated on
      purpose: the gated ``x * sigmoid(x)`` has negative slopes, which can make
      the IFT residual Jacobian indefinite.
    limit_softness: if > 0, engaged joint/tendon limit rows (``efc_pos < efc_margin``)
      use ``D * sj.relu(-Jaref, softness)`` (IFT residual only).
    frictionloss_softness: if > 0, friction-loss rows use a smooth clip of
      ``-Jaref`` to ``[-r f, r f]`` instead of the hard clip (IFT residual only).

  Returns:
    context with new constraint force and costs
  """
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('_update_constraint requires JAX backend implementation.')

  # ne constraints are always active, nf are conditionally active, others are
  # non-negative constraints.
  active = (ctx.Jaref < 0).at[: d._impl.ne + d._impl.nf].set(True)

  floss_force, floss_cost = jp.zeros(d._impl.nefc), 0.0
  dof_fl, ten_fl = m._impl.dof_hasfrictionloss, m._impl.tendon_hasfrictionloss
  if (dof_fl.any() or ten_fl.any()) and not (
      m.opt.disableflags & DisableBit.FRICTIONLOSS
  ):
    f = d._impl.efc_frictionloss
    r = 1.0 / (d._impl.efc_D + (d._impl.efc_D == 0.0) * mujoco.mjMINVAL)
    linear_neg = (ctx.Jaref <= -r * f) * (f > 0)
    linear_pos = (ctx.Jaref >= r * f) * (f > 0)
    active = active & ~linear_neg & ~linear_pos
    floss_force = linear_neg * f + linear_pos * -f
    floss_cost = linear_neg * (-0.5 * r * f * f - f * ctx.Jaref)
    floss_cost += linear_pos * (-0.5 * r * f * f + f * ctx.Jaref)
    floss_cost = floss_cost.sum()

  if m.opt.cone == ConeType.PYRAMIDAL:
    efc_force = d._impl.efc_D * -ctx.Jaref * active + floss_force
    cost = 0.5 * jp.sum(d._impl.efc_D * ctx.Jaref * ctx.Jaref * active)
    dm, u, h = 0.0, 0.0, 0.0
    if contact_softness > 0:
      # Smooth the Jaref mask (liftoff, stick/slip) of nominal contact rows only:
      # pw_solimp rows (efc_pos >= efc_margin) and padding rows keep the hard mask.
      nominal = (d._impl.efc_type >= ConstraintType.CONTACT_FRICTIONLESS) & (
          d._impl.efc_pos < d._impl.efc_margin
      )
      smooth_force = d._impl.efc_D * sj.relu(
          -ctx.Jaref, softness=contact_softness, mode='smooth', gated=False
      )
      efc_force = jp.where(nominal, smooth_force, efc_force)
  elif m.opt.cone == ConeType.ELLIPTIC:
    if contact_softness > 0:
      raise NotImplementedError('contact_activeset_softness requires the pyramidal cone.')
    friction = d._impl.contact.friction[d._impl.contact.dim > 1]
    efc_address = d._impl.contact.efc_address[d._impl.contact.dim > 1]
    dim = d._impl.contact.dim[d._impl.contact.dim > 1]
    # to prevent out of range append zeros to ctx.Jaref
    slice_fn = jax.vmap(
        lambda x: jax.lax.dynamic_slice(
            jp.concatenate((ctx.Jaref, jp.zeros((3)))), (x,), (6,)
        )
    )
    u = slice_fn(efc_address) * ctx.fri
    mu, n, t = ctx.fri[:, 0], u[:, 0], jax.vmap(math.norm)(u[:, 1:])

    # bottom zone: quadratic
    bottom_zone = ((t <= 0) & (n < 0)) | ((t > 0) & ((mu * n + t) <= 0))
    adr_i, adr_j = [], []
    for i, (condim, addr) in enumerate(zip(dim, efc_address)):
      adr_i.extend(range(addr, addr + condim))
      adr_j.extend([i] * condim)
    active = active.at[jp.array(adr_i)].set(bottom_zone[jp.array(adr_j)])
    efc_force = d._impl.efc_D * -ctx.Jaref * active + floss_force
    cost = 0.5 * jp.sum(d._impl.efc_D * ctx.Jaref * ctx.Jaref * active)

    # middle zone: cone
    middle_zone = (t > 0) & (n < (mu * t)) & ((mu * n + t) > 0)
    dm = d._impl.efc_D[efc_address] / jp.maximum(
        mu * mu * (1 + mu * mu), mujoco.mjMINVAL
    )
    nmt = n - mu * t
    cost += 0.5 * jp.sum(dm * nmt * nmt * middle_zone)
    # tangent and friction for middle zone:
    force = -dm * nmt * mu * middle_zone
    force_fri = -force / (t + ~middle_zone * mujoco.mjMINVAL)
    force_fri = force_fri[:, None] * u[:, 1:] * friction
    efc_force = efc_force.at[efc_address].add(force)
    efc_adr, adr_i, adr_j = [], [], []
    for i, (condim, addr) in enumerate(zip(dim, efc_address)):
      efc_adr.extend(range(addr + 1, addr + condim))
      adr_i.extend([i] * (condim - 1))
      adr_j.extend(range(condim - 1))
    efc_adr, adr_i, adr_j = jp.array(efc_adr), jp.array(adr_i), jp.array(adr_j)
    efc_force = efc_force.at[efc_adr].add(force_fri[(adr_i, adr_j)])

    # cone hessian
    h = 0.0
    if m.opt.solver == SolverType.NEWTON:
      t = jp.maximum(t, mujoco.mjMINVAL)
      # h = mu*N/T^3 * U*U'
      ttt = jp.maximum(t * t * t, mujoco.mjMINVAL)
      h = jax.vmap(lambda x, y: x * jp.outer(y, y.T))(mu * n / ttt, u)
      # add to diagonal: (mu^2 - mu*N/T) * I
      h += jax.vmap(lambda x: x * jp.eye(6, 6))(mu * mu - mu * n / t)
      # set first row: (1, -mu/T * U)
      h_0 = jax.vmap(lambda mu, t, u: jp.append(1, -mu / t * u[1:]))(mu, t, u)
      h = h.at[:, 0].set(h_0).at[:, :, 0].set(h_0)
      # pre and post multiply by diag(mu, friction), scale by Dm
      h *= jax.vmap(lambda d, f: d * jp.outer(f, f.T))(dm, ctx.fri)
      # only cone constraints
      h = jax.vmap(jp.multiply)(h, middle_zone)
  else:
    raise NotImplementedError(f'unsupported cone type: {m.opt.cone}')

  if limit_softness > 0:
    # Engaged limit rows only; disengaged rows keep their zero force.
    is_limit = (d._impl.efc_type == ConstraintType.LIMIT_JOINT) | (
        d._impl.efc_type == ConstraintType.LIMIT_TENDON
    )
    engaged = is_limit & (d._impl.efc_pos < d._impl.efc_margin)
    smooth_force = d._impl.efc_D * sj.relu(
        -ctx.Jaref, softness=limit_softness, mode='smooth', gated=False
    )
    efc_force = jp.where(engaged, smooth_force, efc_force)
  if frictionloss_softness > 0:
    # Smooth clip of x = -Jaref to [-b, b], b = r * f (the stick/slip threshold in Jaref units).
    is_fl = (
        (d._impl.efc_type == ConstraintType.FRICTION_DOF)
        | (d._impl.efc_type == ConstraintType.FRICTION_TENDON)
    ) & (d._impl.efc_frictionloss > 0)
    r = 1.0 / (d._impl.efc_D + (d._impl.efc_D == 0.0) * mujoco.mjMINVAL)
    b = r * d._impl.efc_frictionloss
    x = -ctx.Jaref
    soft_clip = (
        x
        - sj.relu(x - b, softness=frictionloss_softness, mode='smooth', gated=False)
        + sj.relu(-x - b, softness=frictionloss_softness, mode='smooth', gated=False)
    )
    efc_force = jp.where(is_fl, d._impl.efc_D * soft_clip, efc_force)

  qfrc_constraint = d._impl.efc_J.T @ efc_force
  gauss = 0.5 * jp.dot(ctx.Ma - d.qfrc_smooth, ctx.qacc - d.qacc_smooth)
  ctx = ctx.replace(
      qfrc_constraint=qfrc_constraint,
      gauss=gauss,
      cost=cost + gauss + floss_cost,
      prev_cost=ctx.cost,
      efc_force=efc_force,
      active=active,
      dm=dm,
      u=u,
      h=h,
  )

  return ctx


def _update_gradient(m: Model, d: Data, ctx: Context) -> Context:
  """Updates grad and M / grad given latest solver iteration.

  Corresponds to CGupdateGradient in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest smooth terms
    ctx: current solver context

  Returns:
    context with new grad and M / grad
  Raises:
    NotImplementedError: for unsupported solver type
  """
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('_update_gradient requires JAX backend implementation.')

  grad = ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint

  if m.opt.solver == SolverType.CG:
    mgrad = smooth.solve_m(m, d, grad)
  elif m.opt.solver == SolverType.NEWTON:
    if m.opt.cone == ConeType.ELLIPTIC:
      cm = jp.diag(d._impl.efc_D * ctx.active)
      efc_address = d._impl.contact.efc_address[d._impl.contact.dim > 1]
      dim = d._impl.contact.dim[d._impl.contact.dim > 1]
      # set efc of cone H along diagonal
      for i, (condim, addr) in enumerate(zip(dim, efc_address)):
        h_cone = ctx.h[i, :condim, :condim]
        cm = cm.at[addr : addr + condim, addr : addr + condim].add(h_cone)
      h = d._impl.efc_J.T @ cm @ d._impl.efc_J
    else:
      h = (d._impl.efc_J.T * d._impl.efc_D * ctx.active) @ d._impl.efc_J
    h = support.full_m(m, d) + h
    # Symmetrize to reduce the chance of numerical issues in cholesky.
    h_sym = (h + h.T) * 0.5
    h_ = jax.scipy.linalg.cho_factor(h_sym)
    mgrad = jax.scipy.linalg.cho_solve(h_, grad)
  else:
    raise NotImplementedError(f'unsupported solver type: {m.opt.solver}')

  ctx = ctx.replace(grad=grad, Mgrad=mgrad)

  return ctx


def _rescale(m: Model, value: jax.Array) -> jax.Array:
  return value / (m.stat.meaninertia * max(1, m.nv))


def _solver_done(m: Model, ctx: Context) -> jax.Array:
  improvement = _rescale(m, ctx.prev_cost - ctx.cost)
  gradient = _rescale(m, math.norm(ctx.grad))

  done = ctx.solver_niter >= m.opt.iterations
  done |= improvement < m.opt.tolerance
  done |= gradient < m.opt.tolerance

  return done


def _solver_step(m: Model, d: Data, ctx: Context) -> Context:
  ctx = _linesearch(m, d, ctx)
  prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad  # pylint: disable=invalid-name
  ctx = _update_constraint(m, d, ctx)
  ctx = _update_gradient(m, d, ctx)

  if m.opt.solver == SolverType.NEWTON:
    search = -ctx.Mgrad
  else:
    # polak-ribiere:
    beta = jp.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
    beta = beta / jp.maximum(mujoco.mjMINVAL, jp.dot(prev_grad, prev_Mgrad))
    beta = jp.maximum(0, beta)
    search = -ctx.Mgrad + beta * ctx.search
  ctx = ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

  return ctx


def _warmstart_qacc(m: Model, d: Data) -> jax.Array:
  qacc = d.qacc_smooth
  if not m.opt.disableflags & DisableBit.WARMSTART:
    warm = Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
    smth = Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
    qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)

  return qacc


def _finish_solve(d: Data, ctx: Context) -> Data:
  solver_niter = jp.asarray(ctx.solver_niter).astype(d._impl.solver_niter.dtype)
  return d.tree_replace({
      'qfrc_constraint': ctx.qfrc_constraint,
      'qacc': ctx.qacc,
      '_impl.efc_force': ctx.efc_force,
      '_impl.solver_niter': solver_niter,
  })


def _linesearch(m: Model, d: Data, ctx: Context) -> Context:
  """Performs a zoom linesearch to find optimal search step size.

  Args:
    m: model defining search options and other needed terms
    d: data with inertia matrix and other needed terms
    ctx: current solver context

  Returns:
    updated context with new qacc, Ma, Jaref
  """
  if (
      not isinstance(m._impl, ModelJAX)
      or not isinstance(d._impl, DataJAX)
      or not isinstance(m.opt._impl, OptionJAX)
  ):
    raise ValueError('_lineasearch requires JAX backend implementation.')

  smag = math.norm(ctx.search) * m.stat.meaninertia * max(1, m.nv)
  gtol = m.opt.tolerance * m.opt.ls_tolerance * smag

  # compute Mv, Jv
  mv = support.mul_m(m, d, ctx.search)
  jv = d._impl.efc_J @ ctx.search

  # prepare quadratics
  quad_gauss = jp.stack((
      ctx.gauss,
      jp.dot(ctx.search, ctx.Ma) - jp.dot(ctx.search, d.qfrc_smooth),
      0.5 * jp.dot(ctx.search, mv),
  ))
  quad = jp.stack((0.5 * ctx.Jaref * ctx.Jaref, jv * ctx.Jaref, 0.5 * jv * jv))
  quad = (quad * d._impl.efc_D).T
  uu, v0, uv, vv = 0.0, 0.0, 0.0, 0.0
  if m.opt.cone == ConeType.ELLIPTIC:
    mask = d._impl.contact.dim > 1
    # complete vector quadratic (for bottom zone)
    efc_con, efc_fri = [], []
    for condim, addr in zip(
        d._impl.contact.dim[mask], d._impl.contact.efc_address[mask]
    ):
      efc_con.extend([addr] * (condim - 1))
      efc_fri.extend(range(addr + 1, addr + condim))
    quad = quad.at[jp.array(efc_con)].add(quad[jp.array(efc_fri)])

    # rescale to make primal cone circular
    # to prevent out of range append zeros to jv
    jv_fn = jax.vmap(
        lambda x: jax.lax.dynamic_slice(
            jp.concatenate((jv, jp.zeros(3))), (x,), (6,)
        )
    )
    efc_elliptic = d._impl.contact.efc_address[mask]
    v = jv_fn(efc_elliptic) * ctx.fri
    uu = jp.sum(ctx.u[:, 1:] * ctx.u[:, 1:], axis=1)
    v0 = v[:, 0]
    uv = jp.sum(ctx.u[:, 1:] * v[:, 1:], axis=1)
    vv = jp.sum(v[:, 1:] * v[:, 1:], axis=1)

  point_fn = lambda a: _LSPoint.create(
      m, d, ctx, a, jv, quad, quad_gauss, uu, v0, uv, vv
  )

  def cond(ctx: _LSContext) -> jax.Array:
    done = ctx.ls_iter >= m.opt.ls_iterations
    done |= ~ctx.swap  # if we did not adjust the interval
    done |= (ctx.lo.deriv_0 < 0) & (ctx.lo.deriv_0 > -gtol)
    done |= (ctx.hi.deriv_0 > 0) & (ctx.hi.deriv_0 < gtol)

    return ~done

  def body(ctx: _LSContext) -> _LSContext:
    # always compute new bracket boundaries and a midpoint
    lo, hi = ctx.lo, ctx.hi
    lo_next = point_fn(lo.alpha - lo.deriv_0 / lo.deriv_1)
    hi_next = point_fn(hi.alpha - hi.deriv_0 / hi.deriv_1)
    mid = point_fn(0.5 * (lo.alpha + hi.alpha))

    # swap lo/hi if the derivative points to a narrower bracket width
    in_bracket = lambda x, y: ((x < y) & (y < 0) | (x > y) & (y > 0))
    swap_lo_next = in_bracket(lo.deriv_0, lo_next.deriv_0)
    lo = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_lo_next, y, x), lo, lo_next
    )
    swap_lo_mid = in_bracket(lo.deriv_0, mid.deriv_0)
    lo = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_lo_mid, y, x), lo, mid
    )
    swap_lo_hi_next = in_bracket(lo.deriv_0, hi_next.deriv_0)
    lo = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_lo_hi_next, y, x), lo, hi_next
    )
    swap_hi_next = in_bracket(hi.deriv_0, hi_next.deriv_0)
    hi = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_hi_next, y, x), hi, hi_next
    )
    swap_hi_mid = in_bracket(hi.deriv_0, mid.deriv_0)
    hi = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_hi_mid, y, x), hi, mid
    )
    swap_hi_lo_next = in_bracket(hi.deriv_0, lo_next.deriv_0)
    hi = jax.tree_util.tree_map(
        lambda x, y: jp.where(swap_hi_lo_next, y, x), hi, lo_next
    )
    swap = swap_lo_next | swap_lo_mid | swap_lo_hi_next
    swap = swap | swap_hi_next | swap_hi_mid | swap_hi_lo_next
    ctx = ctx.replace(lo=lo, hi=hi, swap=swap, ls_iter=ctx.ls_iter + 1)

    return ctx

  # initialize interval
  p0 = point_fn(jp.array(0.0))
  lo = point_fn(p0.alpha - p0.deriv_0 / p0.deriv_1)
  lesser_fn = lambda x, y: jp.where(lo.deriv_0 < p0.deriv_0, x, y)
  hi = jax.tree_util.tree_map(lesser_fn, p0, lo)
  lo = jax.tree_util.tree_map(lesser_fn, lo, p0)
  ls_ctx = _LSContext(lo=lo, hi=hi, swap=jp.array(True), ls_iter=0)
  if m.opt.scan_loop:
    ls_ctx = _while_loop_scan(cond, body, ls_ctx, m.opt.ls_iterations)
  else:
    ls_ctx = jax.lax.while_loop(cond, body, ls_ctx)

  # move to new solution if improved
  lo, hi = ls_ctx.lo, ls_ctx.hi
  improved = (lo.cost < p0.cost) | (hi.cost < p0.cost)
  alpha = jp.where(lo.cost < hi.cost, lo.alpha, hi.alpha)
  qacc = ctx.qacc + improved * ctx.search * alpha
  ma = ctx.Ma + improved * mv * alpha
  jaref = ctx.Jaref + improved * jv * alpha

  ctx = ctx.replace(qacc=qacc, Ma=ma, Jaref=jaref)

  return ctx


def _check_jax_impl(m: Model):
  if not isinstance(m.opt._impl, OptionJAX):
    raise ValueError('solve requires JAX backend implementation.')


def _solve_unrolled(m: Model, d: Data) -> Data:
  """diff-mjx solver: masked scan if ``opt.scan_loop``, else while loop."""
  _check_jax_impl(m)

  def cond(ctx: Context) -> jax.Array:
    return ~_solver_done(m, ctx)

  def body(ctx: Context) -> Context:
    return _solver_step(m, d, ctx)

  d = d.replace(qacc=_warmstart_qacc(m, d))

  ctx = Context.create(m, d)
  if m.opt.iterations == 1:
    ctx = body(ctx)
  elif m.opt.scan_loop:
    ctx = _while_loop_scan(cond, body, ctx, max_iter=m.opt.iterations)
  else:
    ctx = jax.lax.while_loop(cond, body, ctx)

  return _finish_solve(d, ctx)


def _solve_while(m: Model, d: Data) -> Data:
  """Finds forces that satisfy constraints using a while loop."""
  _check_jax_impl(m)

  def cond(ctx: Context) -> jax.Array:
    return ~_solver_done(m, ctx)

  def body(ctx: Context) -> Context:
    return _solver_step(m, d, ctx)

  d = d.replace(qacc=_warmstart_qacc(m, d))

  ctx = Context.create(m, d)
  if m.opt.iterations == 1:
    ctx = body(ctx)
  else:
    ctx = jax.lax.while_loop(cond, body, ctx)

  return _finish_solve(d, ctx)


def _solve_scan(m: Model, d: Data) -> Data:
  """Finds forces using an unrolled scan with reverse-mode support."""
  _check_jax_impl(m)

  d = d.replace(qacc=_warmstart_qacc(m, d))
  ctx = Context.create(m, d)

  def body(ctx: Context, _) -> tuple[Context, None]:
    return _solver_step(m, d, ctx), None

  ctx, _ = jax.lax.scan(body, ctx, None, length=m.opt.iterations)

  return _finish_solve(d, ctx)


def _root(m: Model, d: Data, qacc: jax.Array) -> jax.Array:
  """Stationarity residual ``M qacc - qfrc_smooth - qfrc_constraint``."""
  contact_softness, limit_softness, frictionloss_softness = _activeset_softnesses(m)
  ctx = Context.create(
      m,
      d.replace(qacc=qacc),
      grad=False,
      contact_softness=contact_softness,
      limit_softness=limit_softness,
      frictionloss_softness=frictionloss_softness,
  )
  return ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint


def _root_iterative_solve(
    m: Model, d: Data, qacc: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Runs the solver to convergence; returns (qacc, realised iterations).

  This solve is never differentiated, so it uses the masked scan rather than
  ``lax.while_loop``, which is slow on GPU; results and iteration counts are
  identical to the while loop.
  """
  d = d.replace(qacc=qacc)
  ctx = Context.create(m, d)

  def cond(ctx: Context) -> jax.Array:
    return ~_solver_done(m, ctx)

  def body(ctx: Context) -> Context:
    return _solver_step(m, d, ctx)

  if m.opt.iterations == 1:
    ctx = body(ctx)
  else:
    ctx = _while_loop_scan(cond, body, ctx, max_iter=m.opt.iterations)

  # Float, not int: custom_root's JVP gives integer aux outputs an integer
  # tangent, which JAX rejects (integer tangents must be float0).
  return ctx.qacc, jp.asarray(ctx.solver_niter, dtype=qacc.dtype)


def _root_tangent_solve(g, y):
  """Solves ``g(x) = y`` for the linear map ``g`` of the residual Jacobian.

  ``custom_root`` reverse-differentiates by transposing this function in
  ``y``, so it must be exactly linear in ``y`` (rules out iterative solves).
  """
  # g is linear, so applying it to the identity columns gives its matrix.
  eye = jp.eye(y.shape[0], dtype=y.dtype)
  a = jax.vmap(g, out_axes=1)(eye)
  return jp.linalg.solve(a, y)


def _solve_ift(m: Model, d: Data) -> Data:
  """Finds forces using a custom-root implicit gradient.

  With active-set smoothing on and ``opt.refinement_iterations > 0``, the
  gradient is evaluated at the smoothed residual's root (reached by Newton
  iterations from the hard solution) instead of at the hard solution, while
  the forward value stays the hard solution (straight-through).
  """
  _check_jax_impl(m)

  qacc_guess = _warmstart_qacc(m, d)
  root = lambda qacc: _root(m, d, qacc)
  iterations = _refinement_iterations(m)
  # With every softness 0 the smoothed root is the hard solution: skip.
  refine = iterations > 0 and any(s > 0 for s in _activeset_softnesses(m))

  if not refine:
    solve_hard = lambda _, qacc: _root_iterative_solve(m, d, qacc)
    # The iteration count is auxiliary output (not differentiated) so callers
    # can see how many iterations the forward solve actually needed.
    qacc, solver_niter = jax.lax.custom_root(
        root, qacc_guess, solve_hard, _root_tangent_solve, has_aux=True
    )
  else:

    tolerance = _refinement_tolerance(m)
    ls_iterations = _refinement_ls_iterations(m)
    fallback_enable = _refinement_fallback_enable(m)

    def solve_refined(_, qacc):
      qacc_hard, niter = _root_iterative_solve(m, d, qacc)
      # Residual precision floor: cancellation in Ma - qfrc_smooth - qfrc_con.
      ctx_hard = Context.create(m, d.replace(qacc=qacc_hard), grad=False)
      scale = (
          jp.linalg.norm(ctx_hard.Ma)
          + jp.linalg.norm(d.qfrc_smooth)
          + jp.linalg.norm(ctx_hard.qfrc_constraint)
      )
      floor = 64 * jp.finfo(qacc_hard.dtype).eps * scale
      qacc_eval, _ = _refine_smoothed_root(
          root, qacc_hard, iterations, tolerance, floor, ls_iterations,
          fallback_enable,
      )
      # The hard solution rides along as aux, so it is not solved twice.
      return qacc_eval, (niter, qacc_hard)

    qacc_eval, (solver_niter, qacc_hard) = jax.lax.custom_root(
        root, qacc_guess, solve_refined, _root_tangent_solve, has_aux=True
    )
    # Value of the hard solution, derivative of the smoothed root.
    qacc = jax.lax.stop_gradient(qacc_hard) + (
        qacc_eval - jax.lax.stop_gradient(qacc_eval)
    )
  d = d.replace(qacc=qacc)
  ctx = Context.create(m, d, grad=False).replace(solver_niter=solver_niter)

  return _finish_solve(d, ctx)



_BACKENDS = {
    'unrolled': _solve_unrolled,
    'while': _solve_while,
    'scan': _solve_scan,
    'ift': _solve_ift,
}


def grad_backend(m: Model) -> str:
  """Returns ``m.opt.grad_backend``, validated (case-insensitive).

  The option is static: a string from ``_BACKENDS`` (default ``'ift'``).
  """
  value = m.opt.grad_backend
  if not isinstance(value, str):
    raise TypeError(
        f'opt.grad_backend must be a str, got {value!r}; it is static and '
        'selects the solver at trace time.'
    )
  backend = value.lower()
  if backend not in _BACKENDS:
    raise ValueError(
        f'unsupported opt.grad_backend={value!r}; expected one of '
        f'{", ".join(_BACKENDS)}'
    )
  return backend


def solve(m: Model, d: Data) -> Data:
  """Finds constraint forces with the gradient backend from ``grad_backend``.

  The backend is ``opt.grad_backend`` (default ``ift``). It is chosen at
  trace time, so already-compiled functions keep the backend they were traced
  with:

  - ``unrolled``: the diff-mjx solver, differentiated through its
    iterations. With ``opt.scan_loop`` (the default) the outer loop is a
    masked ``lax.scan`` that freezes after convergence (reverse-mode
    compatible); otherwise ``lax.while_loop``.
  - ``while``: upstream ``lax.while_loop`` solver. Only reverse-mode compatible
    when ``m.opt.iterations == 1``.
  - ``scan``: fixed-depth unroll of exactly ``opt.iterations`` steps with no
    convergence check; differentiates through every step.
  - ``ift`` (default): implicit-function-theorem gradient via
    ``lax.custom_root``. The forward value equals the ``while`` solve
    (computed with a masked scan); the backward pass solves the adjoint of
    the stationarity residual with a dense LU factorization. The gradient is that of the converged fixed
    point, so it differs from the truncated solve when the forward solve
    stops before convergence (e.g. ``opt.iterations == 1``).

  For every backend the line search loop is selected by ``opt.scan_loop``.

  The active-set smoothing options (``opt.contact_activeset_softness``,
  ``opt.limit_activeset_softness``, ``opt.frictionloss_activeset_softness``) only act in the
  ``ift`` backward pass; enabling one with another backend raises instead of
  being silently ignored. ``opt.refinement_iterations`` (Newton
  iterations to the smoothed root, where the gradient is then evaluated) only
  acts when some softness is > 0.
  """
  backend = grad_backend(m)
  softnesses = _activeset_softnesses(m)
  _refinement_iterations(m)  # validate on every backend
  _refinement_tolerance(m)
  _refinement_ls_iterations(m)
  _refinement_fallback_enable(m)
  if backend != 'ift' and any(softness > 0 for softness in softnesses):
    enabled = [n for n, t in zip(_ACTIVESET_SOFTNESS_FIELDS, softnesses) if t > 0]
    raise ValueError(
        f'{", ".join(enabled)} > 0 requires the ift gradient backend '
        f'(opt.grad_backend; got {backend!r}); active-set '
        'smoothing only acts in the IFT backward pass.'
    )
  if softnesses[0] > 0 and m.opt.cone != ConeType.PYRAMIDAL:
    raise NotImplementedError('contact_activeset_softness requires the pyramidal cone.')
  return _BACKENDS[backend](m, d)
