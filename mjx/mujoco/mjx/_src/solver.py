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

import functools
import numbers

import jax
from jax import numpy as jp
from jax.custom_derivatives import SymbolicZero
import mujoco
import numpy as np
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
      stickslip_softness: float = 0.0,
      liftoff_softness: float = 0.0,
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
        m, d, ctx, stickslip_softness=stickslip_softness,
        liftoff_softness=liftoff_softness, limit_softness=limit_softness,
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


_SOFTNESS_FIELDS = (
    'stickslip_softness',
    'liftoff_softness',
    'limit_softness',
    'frictionloss_softness',
)


def _softnesses(m: Model) -> tuple[float, ...]:
  """Returns the validated softnesses, in ``_SOFTNESS_FIELDS`` order.

  The widths select code paths at trace time, so they must be concrete,
  finite, non-negative scalars (0 disables that smoothing).
  """
  return tuple(_static_softness(m, name) for name in _SOFTNESS_FIELDS)


def _static_softness(m: Model, name: str) -> float:
  """Validates one static softness option and returns it as a Python float."""
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
  return softness


def _softness_kwargs(softnesses: tuple[float, ...]) -> dict[str, float]:
  """Softnesses (``_SOFTNESS_FIELDS`` order) as keyword arguments."""
  return dict(zip(_SOFTNESS_FIELDS, softnesses))


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


def _effective_softnesses(m: Model, d: Data) -> tuple[float, ...]:
  """Active-set softnesses with those of absent row types set to 0.

  Row counts are static, so a softness whose rows do not exist in this model
  (no contacts, limits or friction loss) is dropped at trace time and cannot
  trigger a refinement that would leave the hard solution unchanged.
  """
  stickslip, liftoff, limit, frictionloss = _softnesses(m)
  impl = d._impl
  ncontact_rows = impl.nefc - impl.ne - impl.nf - impl.nl
  return (
      stickslip if ncontact_rows else 0.0,
      liftoff if ncontact_rows else 0.0,
      limit if impl.nl else 0.0,
      frictionloss if impl.nf else 0.0,
  )


def _refinement_fallback_enable(m: Model) -> bool:
  """Returns the validated ``opt.refinement_fallback_enable`` (static bool)."""
  value = m.opt.refinement_fallback_enable
  if not isinstance(value, bool):
    raise TypeError(
        'opt.refinement_fallback_enable must be a Python bool, got '
        f'{value!r}; it is static.'
    )
  return value




def _linesearch_1d(derivs, iterations: int, dtype) -> jax.Array:
  """Step size minimizing a convex 1-D cost given ``derivs(a) = (phi', phi'')``.

  ``phi'`` is increasing (the cost is the smoothed convex objective along a
  Newton direction). Like ``_linesearch``, this is a bracketed Newton
  iteration on ``phi'``. Starting from the full Newton step ``a = 1``, it
  expands (``a <- max(Newton, 2a)``) until ``phi'`` changes sign, because on
  the smoothed residual Newton typically undershoots: rows that switch off
  along the step lower the curvature, so the minimizer lies beyond ``a = 1``.
  Once bracketed, it takes Newton steps, bisecting when a Newton step leaves
  the bracket. If no sign change is found, it returns the furthest point with
  ``phi' < 0``, which still decreases the cost. With 0 iterations it returns 1.
  """
  inf = jp.asarray(jp.inf, dtype)

  def body(_, carry):
    lo, hi, alpha = carry
    d1, d2 = derivs(alpha)
    lo = jp.where(d1 < 0, alpha, lo)
    hi = jp.where(d1 > 0, alpha, hi)
    newton = alpha - d1 / jp.maximum(d2, jp.finfo(dtype).tiny)
    bracketed = jp.isfinite(hi)
    # Inclusive: once converged, Newton returns the trial point itself, which
    # sits on a bracket end; a strict test would bisect away from it.
    inside = (newton >= lo) & (newton <= hi)
    next_bracketed = jp.where(inside, newton, 0.5 * (lo + hi))
    next_expand = jp.maximum(newton, 2 * alpha)
    alpha = jp.where(d1 == 0, alpha, jp.where(bracketed, next_bracketed, next_expand))
    return lo, hi, alpha

  init = (jp.asarray(0.0, dtype), inf, jp.asarray(1.0, dtype))
  lo, hi, alpha = jax.lax.fori_loop(0, iterations, body, init)
  # Bracketed: the Newton/bisection iterate lies in [lo, hi]. Not bracketed:
  # every point tried had phi' < 0, so the furthest one (lo) is safe.
  return jp.where(jp.isfinite(hi), alpha, jp.maximum(lo, jp.asarray(1.0, dtype)))


def _jvp_line_derivs(root, qacc: jax.Array, direction: jax.Array):
  """``derivs(a) = (phi'(a), phi''(a))`` of ``phi'(a) = root(q + a p) . p``.

  One JVP of ``root`` per call: works for any residual (generic path).
  """

  def derivs(alpha):
    r, jr = jax.jvp(root, (qacc + alpha * direction,), (direction,))
    return jp.dot(r, direction), jp.dot(jr, direction)

  return derivs


def _smoothed_linesearch(
    root, qacc: jax.Array, direction: jax.Array, iterations: int
) -> jax.Array:
  """``_linesearch_1d`` along ``direction`` for a generic residual ``root``."""
  return _linesearch_1d(
      _jvp_line_derivs(root, qacc, direction), iterations, qacc.dtype
  )


def _jacfwd_newton(root):
  """Generic Newton system: dense ``jacfwd`` Jacobian, LU solve, JVP line search.

  Works for any residual; the default of ``_refine_smoothed_root``. Returns
  ``(point, step)`` as described there.
  """

  def point(qacc):
    return root(qacc), None

  def step(qacc, state):
    r, _ = state
    # The Jacobian is built here, not in point: point also runs once after
    # the last step (only for r(qacc_eval)), so a Jacobian there would be
    # wasted (~nv residual evaluations). The cost of this choice is that
    # jacfwd re-evaluates root(qacc), one extra evaluation per step.
    direction = -jp.linalg.solve(jax.jacfwd(root)(qacc), r)
    return direction, _jvp_line_derivs(root, qacc, direction)

  return point, step


def _smoothed_newton(
    m: Model, d: Data, softnesses: tuple[float, ...]
):
  """Newton system of the smoothed residual, exact for either cone.

  ``r(q) = M q - qfrc_smooth - J^T f(Jaref)`` with ``Jaref = J q - aref``,
  and only the constraint force ``f`` is nonlinear. The residual Jacobian
  ``M - J^T (df/dJaref) J`` therefore needs only ``(df/dJaref) J``: one JVP of
  ``f`` per column of ``J``, through the force alone (``jacfwd`` of the
  residual would also push every tangent through ``M``, ``J`` and ``J^T``).
  ``df/dJaref`` is block diagonal per contact (the per-contact cost couples a
  pyramidal contact's edge rows; the elliptic cone couples its rows too), and it is symmetric negative semidefinite (``f`` is minus
  the gradient of the convex constraint cost), so the Jacobian is symmetric
  positive definite and is solved with Cholesky. Along the direction ``p``,
  ``J p`` and ``M p`` are computed once, so each line-search trial evaluates
  ``f`` and one JVP of ``f`` along ``J p`` (no ``nv``-sized products).

  Returns ``(point, step)`` as described in ``_refine_smoothed_root``, with
  the smoothed ``Context`` as the state.
  """
  softness = _softness_kwargs(softnesses)
  efc_j = d._impl.efc_J
  m_full = support.full_m(m, d)

  def state_of(ctx):
    return ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint, ctx

  def point(qacc):
    return state_of(
        Context.create(m, d.replace(qacc=qacc), grad=False, **softness)
    )

  def step(qacc, state):
    del qacc  # the context holds everything at qacc
    r, ctx = state

    def force(jaref):
      return _update_constraint(m, d, ctx.replace(Jaref=jaref), **softness).efc_force

    def force_jvp(jaref, tangent):
      return jax.jvp(force, (jaref,), (tangent,))

    df_j = jax.vmap(
        lambda t: force_jvp(ctx.Jaref, t)[1], in_axes=1, out_axes=1
    )(efc_j)
    h = m_full - efc_j.T @ df_j
    factor = jax.scipy.linalg.cho_factor(0.5 * (h + h.T))
    direction = -jax.scipy.linalg.cho_solve(factor, r)

    jv = efc_j @ direction
    c0 = jp.dot(direction, ctx.Ma - d.qfrc_smooth)
    c1 = jp.dot(direction, support.mul_m(m, d, direction))

    def derivs(alpha):
      f, df_jv = force_jvp(ctx.Jaref + alpha * jv, jv)
      return c0 + alpha * c1 - jp.dot(jv, f), c1 - jp.dot(jv, df_jv)

    return direction, derivs

  return point, step


def _refine_smoothed_root(
    root,
    qacc_hard: jax.Array,
    iterations: int,
    tolerance: float,
    floor: jax.Array | float = 0.0,
    ls_iterations: int = 0,
    fallback_enable: bool = True,
    newton=None,
) -> tuple[jax.Array, jax.Array]:
  """Line-searched Newton from the hard solution toward the smoothed root.

  The result is only the gradient's evaluation point; forward values stay
  hard. Each Newton step uses ``ls_iterations`` line-search trials; with 0
  the full Newton step is taken (plain Newton, no line search cost). Returns
  ``(qacc_eval, accepted)``.

  ``newton = (point, step)`` defines the Newton system (default:
  ``_jacfwd_newton(root)``). ``point(q) -> (r(q), aux)`` evaluates the
  residual and whatever the step reuses at ``q``; ``step(q, state) -> (p,
  derivs)`` gives the Newton direction and ``derivs(a) = (phi'(a),
  phi''(a))`` along it. The state is carried through the iterations, so
  each point is evaluated once, and the last one gives ``r(qacc_eval)``.

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
  if iterations == 0:
    return qacc_hard, jp.asarray(True)
  point, step = newton if newton is not None else _jacfwd_newton(root)
  state = point(qacc_hard)
  r_hard = jp.linalg.norm(state[0])

  def body(carry, _):
    qacc, state = carry
    direction, derivs = step(qacc, state)
    alpha = _linesearch_1d(derivs, ls_iterations, qacc.dtype)
    qacc = qacc + alpha * direction
    return (qacc, point(qacc)), None

  (qacc, state), _ = jax.lax.scan(
      body, (qacc_hard, state), None, length=iterations
  )
  r_eval = jp.linalg.norm(state[0])
  accepted = jp.isfinite(r_eval) & jp.all(jp.isfinite(qacc))
  if fallback_enable:
    accepted &= r_eval <= jp.maximum(tolerance * r_hard, floor)
  return jp.where(accepted, qacc, qacc_hard), accepted


def _relu_per_row(x: jax.Array, tau: jax.Array) -> jax.Array:
  """``sj.relu(x, tau)`` (smooth, non-gated) with a per-row ``tau``.

  softjax computes ``softness * f(x / softness)``, so this equals
  ``sj.relu(x, softness=tau)``; it passes a scalar softness because
  ``sj.relu`` rejects an array softness outside of tracing.
  """
  return tau * sj.relu(x / tau, softness=1.0, mode='smooth', gated=False)


def _softplus(x: jax.Array) -> jax.Array:
  """``softplus(x)`` as softjax's smooth, non-gated ``relu`` at unit softness.

  The contact cost's closed forms (``_softplus_integral``, ``_tau_term``) are
  derived for this kernel, so it is fixed to ``mode='smooth'``.
  """
  return sj.relu(x, softness=1.0, mode='smooth', gated=False)


def _li2_terms(dtype) -> int:
  """Series terms for ``Li2(w)``, ``0 <= w <= 1/2``: the truncation error is
  below ``w**(K+1) / ((K+1)**2 (1 - w)) <= 2**-K / (K+1)**2``, under the
  dtype's rounding for ``K = 40`` (float64) and ``K = 20`` (float32 and
  narrower, ``2e-9`` against ``F(0) = pi**2/12``)."""
  return 40 if jp.finfo(dtype).bits > 32 else 20


def _li2_half(w: jax.Array) -> jax.Array:
  """Dilogarithm ``Li2(w) = sum_k w**k / k**2`` for ``0 <= w <= 1/2``, by
  Horner's rule (one multiply-add per term, no powers), with
  ``_li2_terms(w.dtype)`` terms."""
  k_max = _li2_terms(w.dtype)
  acc = jp.full_like(w, 1.0 / (k_max * k_max))
  for k in range(k_max - 1, 0, -1):
    acc = 1.0 / (k * k) + w * acc
  return w * acc


@jax.custom_jvp
def _softplus_integral(z: jax.Array) -> jax.Array:
  """``F(z) = integral of softplus from -inf to z = -Li2(-e^z)``.

  ``F' = softplus`` and ``F'' = sigmoid``. For ``z <= 0`` Landen's identity
  gives ``F(z) = Li2(sigmoid(z)) + softplus(z)**2 / 2`` (series argument at
  most 1/2); for ``z > 0`` the inversion ``F(z) = pi**2/6 + z**2/2 - F(-z)``
  reduces to that case.
  """
  # where, not minimum/maximum: those split the gradient at 0.
  neg = jp.where(z <= 0, z, 0.0)
  pos = jp.where(z > 0, z, 0.0)
  f_neg = _li2_half(jax.nn.sigmoid(neg)) + 0.5 * _softplus(neg) ** 2
  f_mpos = _li2_half(jax.nn.sigmoid(-pos)) + 0.5 * _softplus(-pos) ** 2
  return jp.where(z <= 0, f_neg, jp.pi**2 / 6 + 0.5 * pos * pos - f_mpos)


@_softplus_integral.defjvp
def _softplus_integral_jvp(primals, tangents):
  (z,), (dz,) = primals, tangents
  return _softplus_integral(z), _softplus(z) * dz


def _tau_term(u: jax.Array) -> jax.Array:
  """``G(u) = 2 F(u) - u softplus(u) >= 0``, the ``tau``-derivative factor.

  For ``u > 0`` both terms grow like ``u**2`` and cancel to about
  ``pi**2/3``; the inversion of ``F`` gives the cancellation-free
  ``G(u) = pi**2/3 - 2 F(-u) - u softplus(-u)``.
  """
  neg = jp.where(u <= 0, u, 0.0)
  pos = jp.where(u > 0, u, 0.0)
  g_neg = 2.0 * _softplus_integral(neg) - neg * _softplus(neg)
  g_pos = (
      jp.pi**2 / 3
      - 2.0 * _softplus_integral(-pos)
      - pos * _softplus(-pos)
  )
  return jp.where(u <= 0, g_neg, g_pos)


def _contact_edge_groups(m: Model) -> list[tuple[int, int, int]]:
  """Static ``(first_row, num_contacts, rows_per_contact)`` contact blocks.

  ``make_constraint`` lays contact rows out after the equality, friction-loss
  and limit rows: condim-1 (frictionless) contacts with one row each, then
  condim 3, 4 and 6 contacts with ``2 * (condim - 1)`` pyramid edge rows each
  (pyramidal cone) or ``condim`` rows each, normal first (elliptic cone),
  every contact's rows consecutive (``make_efc_type``).
  """
  # Imported here: constraint imports collision_driver, which imports more of
  # the package than this module needs at load time.
  from mujoco.mjx._src import collision_driver  # pylint: disable=g-import-not-at-top
  from mujoco.mjx._src import constraint  # pylint: disable=g-import-not-at-top

  if m.opt.disableflags & DisableBit.CONTACT:
    return []
  efc_type = constraint.make_efc_type(m)
  ne, nf, nl, _ = constraint.counts(efc_type)
  dim = collision_driver.make_condim(m)
  start, groups = int(ne + nf + nl), []
  for condim in (1, 3, 4, 6):
    n = int((dim == condim).sum())
    if condim == 1:
      rows = 1
    elif m.opt.cone == ConeType.ELLIPTIC:
      rows = condim
    else:
      rows = 2 * (condim - 1)
    if n:
      groups.append((start, n, rows))
    start += n * rows
  if start != efc_type.shape[0]:
    raise ValueError('contact rows do not match make_efc_type.')
  return groups


def _contact_band_terms(
    n: jax.Array, scale: jax.Array, slip_softness: float, liftoff_softness: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Per-contact band ``tau(n)`` and its first two derivatives in ``n``.

  With ``liftoff_softness > 0``, ``tau(n) = l0 * (1 + softplus(slip * (-n) / l0))``
  and ``l0 = liftoff_softness * scale / (1 + log 2)``, ``scale = k*imp*width``:

  - at the liftoff switch (``n = 0``) ``tau = liftoff_softness * scale``,
    ``liftoff_softness`` solimp widths;
  - separating (``n >> 0``) it settles to the floor ``l0``;
  - loaded (``-n >> l0 / slip``) ``tau ~ slip_softness * |n|``, a fraction
    ``slip_softness`` of the contact's load: friction utilization.

  With ``liftoff_softness = 0`` it is the ``l0 -> 0`` limit
  ``tau = slip_softness * max(-n, 0)``: the same utilization band, and 0 (the
  hard contact) from liftoff on. With ``slip_softness = 0`` it is the constant
  ``liftoff_softness * scale``. Non-negative and convex in ``n`` (the second
  derivative of the ``l0 -> 0`` kink is left out).
  """
  if liftoff_softness > 0:
    l0 = liftoff_softness * scale / (1.0 + jp.log(2.0))
    z = slip_softness * -n / l0
    sz = jax.nn.sigmoid(z)
    tau = l0 * (1.0 + _softplus(z))
    return tau, -slip_softness * sz, slip_softness**2 / l0 * sz * (1.0 - sz)
  loaded = n < 0
  tau = slip_softness * jp.where(loaded, -n, 0.0)
  return tau, -slip_softness * loaded.astype(n.dtype), jp.zeros_like(n)


def _contact_band(
    n: jax.Array, scale: jax.Array, slip_softness: float, liftoff_softness: float
) -> jax.Array:
  """Per-contact band ``tau(n)`` (see ``_contact_band_terms``)."""
  return _contact_band_terms(n, scale, slip_softness, liftoff_softness)[0]


def _contact_cost_force(
    m: Model,
    d: Data,
    jaref: jax.Array,
    scale: jax.Array,
    slip_softness: float,
    liftoff_softness: float,
) -> jax.Array:
  """Per-contact smoothed force: minus the gradient of a convex contact cost.

  For a contact with rows ``y_e = Jaref_e`` (pyramid edges ``n +- t``, or one
  frictionless row) and normal coordinate ``n = mean_e y_e = Jaref_n``:

    ``C(y) = sum_e D_e * h(y_e, tau(n))``,  ``h(y, tau) = tau**2 F(-y / tau)``

  with ``F' = softplus`` and the band ``tau(n)`` of ``_contact_band`` (``l``
  from ``efc_softness_scale``, held constant). With ``tau`` fixed each term is
  the per-row force ``D_e * tau * softplus(-y_e / tau)``, and ``tau -> 0``
  gives the hard ``1/2 D relu(-y)**2``. Stick/slip is smoothed over a fraction
  ``slip_softness`` of friction utilization and liftoff over
  ``liftoff_softness`` solimp widths; the two merge for contacts lighter than
  about ``|n| ~ l0 / slip_softness`` (the slip lines meet at the cone's apex).
  Either may be 0: ``liftoff_softness = 0`` keeps liftoff hard (``tau = 0``
  once the contact unloads, where the force is the hard ``D max(0, -y)``),
  ``slip_softness = 0`` gives a constant band.

  Convexity (``u = -y / tau``, ``F'' = sigmoid``):

  - ``h`` is jointly convex for ``tau > 0``: its Hessian in ``(y, tau)`` is
    ``[[F'', u F'' - F'], [u F'' - F', 2F - 2u F' + u**2 F'']]`` with
    determinant ``Q = 2 F F'' - F'**2``; ``Q' = 2 F sigmoid' > 0`` and
    ``Q(-inf) = 0``, so ``Q > 0``.
  - ``h`` is increasing in ``tau``: ``dh/dtau = tau G``, ``G = 2F - u F'``;
    ``G' = softplus(u) - u sigmoid(u) > 0`` (it rises from 0 for ``u < 0``,
    peaks at ``log 2`` and falls to ``0+`` for ``u > 0``) and ``G(-inf) = 0``,
    so ``G > 0``.
  - ``tau`` is non-negative and convex in ``y`` (softplus, or ``max(0, .)``,
    of an affine function); ``h`` extends continuously to ``tau = 0`` as the
    hard cost, which keeps both properties.

  A jointly convex function nondecreasing in ``tau``, composed with a convex
  ``tau``, is convex; so ``C`` is convex, ``f = -grad C`` and the residual
  Jacobian ``M + J^T (hess C) J`` is symmetric positive definite.
  """
  groups = _contact_edge_groups(m)
  if not groups:
    return jp.zeros_like(jaref)
  efc_d = d._impl.efc_D
  elliptic = m.opt.cone == ConeType.ELLIPTIC
  # Contact blocks are contiguous and end the row layout (_contact_edge_groups).
  parts = [jp.zeros(groups[0][0], jaref.dtype)]
  for start, ncon, rows in groups:
    sl = slice(start, start + ncon * rows)
    y = jaref[sl].reshape(ncon, rows)
    dd = efc_d[sl].reshape(ncon, rows)
    ell = scale[sl].reshape(ncon, rows)[:, 0]  # the normal row's (edges share it)
    if elliptic and rows > 1:
      # Elliptic condim-``rows`` contacts, in make_constraint's order.
      con = np.nonzero(np.asarray(d._impl.contact.dim) == rows)[0]
      if con.size != ncon:
        raise ValueError('elliptic contacts do not match their constraint rows.')
      friction = d._impl.contact.friction[con]
      mu = friction[:, 0] / jp.sqrt(m.opt.impratio)
      force = _elliptic_group_force(
          y, dd, ell, friction[:, : rows - 1], mu, slip_softness, liftoff_softness
      )
    else:
      force = _contact_group_force(y, dd, ell, slip_softness, liftoff_softness)
    parts.append(force.reshape(-1))
  return jp.concatenate(parts)


# Below this band a contact uses the hard force (the smoothed one differs by
# O(D * tau)); it keeps -y / tau finite.
_MIN_BAND = mujoco.mjMINVAL


def _contact_group_terms(y, efc_d, ell, slip_softness, liftoff_softness):
  """Force of one block of contacts (``y``: ``(ncon, rows)``) and the terms
  its tangent rule reuses. ``F`` is evaluated once, at ``-|u|``; ``F(u)`` and
  ``G(u) = 2 F(u) - u softplus(u)`` follow by inversion, cancellation-free.
  Contacts with a zero band (``liftoff_softness = 0``, unloaded) get the hard
  force."""
  rows = y.shape[1]
  n = jp.mean(y, axis=1)
  tau, dtau, d2tau = _contact_band_terms(n, ell, slip_softness, liftoff_softness)
  on = tau > _MIN_BAND
  tau = jp.where(on, tau, 1.0)  # finite u; the off contacts are replaced below
  tau_c = tau[:, None]
  u = -y / tau_c
  pos = u > 0
  sp, sg = _softplus(u), jax.nn.sigmoid(u)
  sp_m, sg_m = _softplus(-u), jax.nn.sigmoid(-u)
  f_m = _softplus_integral(-jp.abs(u))
  g = jp.where(pos, jp.pi**2 / 3 - 2.0 * f_m - u * sp_m, 2.0 * f_m - u * sp)
  force = efc_d * tau_c * sp + (
      -dtau / rows * jp.sum(efc_d * tau_c * g, axis=1)
  )[:, None]
  force = jp.where(on[:, None], force, efc_d * jp.where(y < 0, -y, 0.0))
  return force, (u, pos, sp, sg, sp_m, sg_m, f_m, g, tau, dtau, d2tau, on)


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4))
def _contact_group_force(y, efc_d, ell, slip_softness, liftoff_softness):
  """Per-contact force of one block, with an analytic tangent rule."""
  return _contact_group_terms(y, efc_d, ell, slip_softness, liftoff_softness)[0]


@functools.partial(_contact_group_force.defjvp, symbolic_zeros=True)
def _contact_group_force_jvp(slip_softness, liftoff_softness, primals, tangents):
  """``df = -(hess C) dy + f(y; dD)``.

  Per contact, ``hess C`` is the per-row mask ``diag(D sigmoid(u))`` plus a
  symmetric update along the normal mode ``1`` (``n = mean y``):
  ``(tau'/N) (a 1^T + 1 a^T) + c 1 1^T`` with ``a_e = D_e d2h/dy dtau =
  -D_e G'(u_e)``, ``c = (tau'/N)^2 sum D d2h/dtau2 + tau''/N^2 sum D tau G``;
  a contact with a zero band has the hard mask ``diag(D [y < 0])``.
  The force is linear in ``D``, so its ``D`` tangent is the force at ``dD``,
  formed from the terms already computed; zero tangents are skipped.
  ``ell`` is held constant (``stop_gradient`` upstream).
  """
  y, efc_d, ell = primals
  dy, d_efc_d, _ = tangents
  force, (u, pos, sp, sg, sp_m, sg_m, f_m, g, tau, dtau, d2tau, on) = (
      _contact_group_terms(y, efc_d, ell, slip_softness, liftoff_softness)
  )
  rows = y.shape[1]
  tau_c = tau[:, None]
  d_force = jp.zeros_like(force)
  if not _is_zero(d_efc_d):
    # Linear in D: the force at dD, from the terms already computed.
    d_force = jp.where(
        on[:, None],
        d_efc_d * tau_c * sp
        + (-dtau / rows * jp.sum(d_efc_d * tau_c * g, axis=1))[:, None],
        d_efc_d * jp.where(y < 0, -y, 0.0),
    )
  if _is_zero(dy):
    return force, d_force
  g1 = jp.where(pos, sp_m + u * sg_m, sp - u * sg)             # G'(u)
  h_tt = jp.where(                                             # d2h / dtau2
      pos,  # u * (u * s): u * u overflows for a vanishing band
      jp.pi**2 / 3 - 2.0 * f_m - 2.0 * u * sp_m - u * (u * sg_m),
      2.0 * f_m - 2.0 * u * sp + u * (u * sg),
  )
  a = -efc_d * g1
  one_dy = jp.sum(dy, axis=1)
  a_dy = jp.sum(a * dy, axis=1)
  c = (dtau / rows) ** 2 * jp.sum(efc_d * h_tt, axis=1) + d2tau / rows**2 * jp.sum(
      efc_d * tau[:, None] * g, axis=1
  )
  hess_dy = (
      efc_d * sg * dy
      + (dtau / rows)[:, None] * (a * one_dy[:, None] + a_dy[:, None])
      + (c * one_dy)[:, None]
  )
  hess_dy = jp.where(on[:, None], hess_dy, efc_d * (y < 0) * dy)
  return force, -hess_dy + d_force


def _kink_terms(u: jax.Array) -> dict[str, jax.Array]:
  """Per-kink terms of ``h(k, tau) = tau**2 F(-k / tau)`` at ``u = -k / tau``,
  with ``F`` evaluated once (at ``-|u|``) and cancellation-free forms:
  ``sp = softplus(u)``, ``sg = sigmoid(u)`` (and both at ``-u``),
  ``g = G(u) = 2F - u sp`` (so ``dh/dtau = tau G``), ``g1 = G'(u) = sp - u sg``
  (``d2h/dy dtau = -G'``) and ``htt = d2h/dtau2 = 2F - 2u sp + u**2 sg``."""
  pos = u > 0
  sp, sg = _softplus(u), jax.nn.sigmoid(u)
  sp_m, sg_m = _softplus(-u), jax.nn.sigmoid(-u)
  f_m = _softplus_integral(-jp.abs(u))
  return dict(
      sp=sp,
      sg=sg,
      sp_m=sp_m,
      sg_m=sg_m,
      g=jp.where(pos, jp.pi**2 / 3 - 2.0 * f_m - u * sp_m, 2.0 * f_m - u * sp),
      g1=jp.where(pos, sp_m + u * sg_m, sp - u * sg),
      htt=jp.where(  # u * (u * s): u * u overflows for a vanishing band
          pos,
          jp.pi**2 / 3 - 2.0 * f_m - 2.0 * u * sp_m - u * (u * sg_m),
          2.0 * f_m - 2.0 * u * sp + u * (u * sg),
      ),
  )


def _divided_difference(ga, gb, a, b, d1_mid, d3_mid):
  """``(g(b) - g(a)) / (b - a)`` from ``ga = g(a)``, ``gb = g(b)``; for
  ``|b - a| < 1e-2`` the midpoint expansion ``g'(m) + g3(m) (b - a)**2 / 24``
  (``d1_mid``, ``d3_mid``: the first and third derivatives at the midpoint),
  which avoids dividing by a vanishing width."""
  delta = b - a
  small = jp.abs(delta) < 1e-2
  far = (gb - ga) / jp.where(small, 1.0, delta)
  return jp.where(small, d1_mid + d3_mid * delta * delta / 24.0, far)


def _sigmoid_derivs(s: jax.Array):
  """First three derivatives of sigmoid where its value is ``s``."""
  s1 = s * (1.0 - s)
  return s1, s1 * (1.0 - 2.0 * s), s1 * (1.0 - 6.0 * s1)


def _elliptic_group_terms(y, efc_d, ell, friction, mu, slip_softness, liftoff_softness):
  """Force of one block of elliptic contacts and the terms its tangent rule
  reuses (see ``_elliptic_group_force``). The force is ``D_m`` times a unit
  force (``unit``), so its ``D`` tangent is a scaling."""
  mu = jp.maximum(mu, mujoco.mjMINVAL)
  mu2 = mu * mu
  n = mu * y[:, 0]
  t = friction * y[:, 1:]
  t2 = jp.sum(t * t, axis=1)
  moving = t2 > 0
  rho = jp.where(moving, jp.sqrt(jp.where(moving, t2, 1.0)), 0.0)
  rho_safe = jp.where(moving, rho, 1.0)
  tau, dtau, d2tau = _contact_band_terms(n, mu * ell, slip_softness, liftoff_softness)
  on = tau > _MIN_BAND
  tau = jp.where(on, tau, 1.0)  # finite u; the off contacts are replaced below
  dm_scale = 1.0 / jp.maximum(mu2 * (1.0 + mu2), mujoco.mjMINVAL)
  dm = efc_d[:, 0] * dm_scale
  u1 = (mu * rho - n) / tau
  u2 = -(n + rho / mu) / tau
  k1, k2 = _kink_terms(u1), _kink_terms(u2)
  # S = mean slope of softplus over [u1, u2], cancellation-free: the values at
  # -u when the midpoint is above 0 (softplus(x) = x + softplus(-x)).
  m = 0.5 * (u1 + u2)
  flip = m > 0
  s_mid = jax.nn.sigmoid(-jp.abs(m))  # sigmoid at the (flipped) midpoint, <= 1/2
  s1, s2, s3 = _sigmoid_derivs(s_mid)
  a_, b_ = jp.where(flip, -u1, u1), jp.where(flip, -u2, u2)
  slope = _divided_difference(
      jp.where(flip, k1['sp_m'], k1['sp']), jp.where(flip, k2['sp_m'], k2['sp']),
      a_, b_, s_mid, s2,
  )
  slope = jp.where(flip, 1.0 - slope, slope)
  f_n = tau * (k1['sp'] + mu2 * k2['sp']) - dtau * tau * (k1['g'] + mu2 * k2['g'])
  w = (1.0 + mu2) * slope  # Phi_rho / (rho D_m)
  # Hard force (MuJoCo's zones) for contacts with a zero band.
  a1 = jp.maximum(rho * mu - n, 0.0)
  a2 = jp.maximum(-n - rho / mu, 0.0)
  f_n = jp.where(on, f_n, a1 + mu2 * a2)
  w = jp.where(on, w, mu * (a1 - a2) / rho_safe)
  unit = jp.concatenate(((mu * f_n)[:, None], friction * (-w[:, None] * t)), axis=1)
  force = dm[:, None] * unit
  aux = dict(mu=mu, mu2=mu2, n=n, t=t, rho=rho, rho_safe=rho_safe, tau=tau,
             dtau=dtau, d2tau=d2tau, on=on, dm=dm, dm_scale=dm_scale, u1=u1,
             u2=u2, k1=k1, k2=k2, w=dm * w, a1=a1, a2=a2, m=m, flip=flip,
             a_=a_, b_=b_, s1=s1, s2=s2, s3=s3, unit=unit)
  return force, aux


@functools.partial(jax.custom_jvp, nondiff_argnums=(5, 6))
def _elliptic_group_force(y, efc_d, ell, friction, mu, slip_softness, liftoff_softness):
  """Per-contact smoothed force of one block of elliptic contacts.

  ``y``: ``(ncon, condim)`` rows (normal first), ``friction``: the
  ``condim - 1`` friction coefficients, ``mu = friction[0] / sqrt(impratio)``.
  In MuJoCo's scaled coordinates ``n = mu y_0``, ``t_j = f_j y_j``,
  ``rho = |t|``, the hard cost is ``D_m / 2`` times the squared distance to
  the cone, which splits into two kinks:

    ``psi = D_m / 2 [relu(-k1)**2 + mu**2 relu(-k2)**2]``,
    ``k1 = n - mu rho`` (liftoff), ``k2 = n + rho / mu`` (stick/slip),

  ``D_m = D_0 / (mu**2 (1 + mu**2))``. (MuJoCo's friction rows have
  ``D_j = D_0 f_j**2 / mu**2``, which makes the sticking zone isotropic in
  ``(n, t)``.) The smoothed cost replaces each ``relu(-k)**2 / 2`` with
  ``h(k, tau) = tau**2 F(-k / tau)``, both with the contact's band
  ``tau(n)`` (``_contact_band_terms``, scale ``mu * k*imp*width`` since
  ``n = mu y_0``):

    ``C = Phi(n, rho) = D_m [h(k1, tau) + mu**2 h(k2, tau)]``.

  The liftoff kink sits at ``n >= 0`` and the stick/slip kink at ``n < 0``,
  so liftoff is smoothed over ``liftoff_softness`` solimp widths and
  stick/slip over a fraction ``stickslip_softness`` of utilization
  ``rho / (mu |n|)``. With ``u1 = -k1 / tau``, ``u2 = -k2 / tau``:

    ``f_n = D_m tau [sp(u1) + mu**2 sp(u2)] - tau' D_m tau [G(u1) + mu**2 G(u2)]``
    ``f_t = -w t``,  ``w = Phi_rho / rho = D_m (1 + mu**2) S``,
    ``S = (sp(u2) - sp(u1)) / (u2 - u1)``,

  and the rows' force is ``(mu f_n, f_j (f_t)_j)``.

  Convexity: ``Phi(n, rho)`` is jointly convex (``h`` of linear arguments
  and a convex band) and, with one band for both kinks, nondecreasing in
  ``rho >= 0`` (``Phi_rho = D_m mu [h_y(k2) - h_y(k1)] >= 0`` since
  ``k2 >= k1``), so ``C = Phi(n, |t|)`` is convex; ``Phi_rho = 0`` at
  ``rho = 0``, so it is differentiable at ``t = 0``. Contacts with a zero
  band (``liftoff_softness = 0``, unloaded) get the hard force.
  """
  return _elliptic_group_terms(
      y, efc_d, ell, friction, mu, slip_softness, liftoff_softness
  )[0]


def _is_zero(tangent) -> bool:
  return isinstance(tangent, SymbolicZero)


@functools.partial(_elliptic_group_force.defjvp, symbolic_zeros=True)
def _elliptic_group_force_jvp(slip_softness, liftoff_softness, primals, tangents):
  """``df = -(hess C) dy + f(y; dD) + (friction and mu tangents)``.

  In ``(n, t)`` the Hessian of ``C = Phi(n, |t|)`` is

    ``H_nn = Phi_nn``,  ``H_nt = w_n t``,  ``H_tt = w I + (Phi_rhorho - w) that that^T``

  with ``that = t / rho`` and, writing ``c = (1, mu**2)`` for the two kinks,
  ``Phi_nn = D_m sum_i c_i [sg_i - 2 tau' G'_i + tau'**2 htt_i + tau'' tau G_i]``,
  ``Phi_rhorho = D_m [mu**2 sg_1 + sg_2]`` and ``w_n = -D_m (1 + mu**2) / tau``
  times the divided difference of ``sigmoid - tau' G'`` over ``[u1, u2]``;
  none divides by ``rho``, so it holds at ``t = 0``. The force is linear in
  ``D`` (its ``D`` tangent is the force at ``dD``); tangents in the friction
  coefficients and ``mu`` use autodiff, and only when present. ``ell`` is
  held constant (``stop_gradient`` upstream).
  """
  y, efc_d, ell, friction, mu = primals
  dy, d_efc_d, _, d_friction, d_mu = tangents
  force, a = _elliptic_group_terms(
      y, efc_d, ell, friction, mu, slip_softness, liftoff_softness
  )
  d_force = jp.zeros_like(force)
  if not _is_zero(dy):
    mu_, mu2, t, dm, tau, dtau = a['mu'], a['mu2'], a['t'], a['dm'], a['tau'], a['dtau']
    k1, k2, u1, u2, on = a['k1'], a['k2'], a['u1'], a['u2'], a['on']
    that = t / a['rho_safe'][:, None]
    # Smoothed Hessian terms.
    phi_nn = dm * sum(
        c * (k['sg'] - 2.0 * dtau * k['g1'] + dtau * dtau * k['htt'] + a['d2tau'] * tau * k['g'])
        for c, k in ((1.0, k1), (mu2, k2))
    )
    phi_rr = dm * (mu2 * k1['sg'] + k2['sg'])
    # Divided differences over [u1, u2]: sigmoid's is symmetric under u -> -u
    # (so it uses the flipped values, <= 1/2); G' uses the midpoint m.
    flip, m, s1, s2, s3 = a['flip'], a['m'], a['s1'], a['s2'], a['s3']
    dd_sg = _divided_difference(
        jp.where(flip, k1['sg_m'], k1['sg']), jp.where(flip, k2['sg_m'], k2['sg']),
        a['a_'], a['b_'], s1, s3,
    )
    s2_m = jp.where(flip, -s2, s2)  # sigmoid'' is odd; s1, s3 are even
    # (G')' = -u sigmoid', (G')''' = -2 sigmoid'' - u sigmoid''' (at m)
    dd_g1 = _divided_difference(k1['g1'], k2['g1'], u1, u2, -m * s1, -2.0 * s2_m - m * s3)
    w_n = -dm * (1.0 + mu2) / tau * (dd_sg - dtau * dd_g1)
    # Hard Hessian (MuJoCo's zones) for contacts with a zero band.
    i1 = (a['a1'] > 0).astype(t.dtype)
    i2 = (a['a2'] > 0).astype(t.dtype)
    phi_nn = jp.where(on, phi_nn, dm * (i1 + mu2 * i2))
    phi_rr = jp.where(on, phi_rr, dm * (mu2 * i1 + i2))
    w_n = jp.where(on, w_n, dm * mu_ * (i2 - i1) / a['rho_safe'])
    w = a['w']
    dn = mu_ * dy[:, 0]
    dt = friction * dy[:, 1:]
    that_dt = jp.sum(that * dt, axis=1)
    t_dt = a['rho'] * that_dt
    df_n = -(phi_nn * dn + w_n * t_dt)
    df_t = -(
        (w_n * dn)[:, None] * t
        + w[:, None] * dt
        + ((phi_rr - w) * that_dt)[:, None] * that
    )
    d_force = d_force + jp.concatenate(((mu_ * df_n)[:, None], friction * df_t), axis=1)
  if not _is_zero(d_efc_d):  # D_m * unit, D_m = D_0 / (mu**2 (1 + mu**2))
    d_force = d_force + (d_efc_d[:, 0] * a['dm_scale'])[:, None] * a['unit']
  if not (_is_zero(d_friction) and _is_zero(d_mu)):
    zero = lambda x, dx: jp.zeros_like(x) if _is_zero(dx) else dx
    _, d_par = jax.jvp(
        lambda f_, m_: _elliptic_group_terms(
            y, efc_d, ell, f_, m_, slip_softness, liftoff_softness
        )[0],
        (friction, mu),
        (zero(friction, d_friction), zero(mu, d_mu)),
    )
    d_force = d_force + d_par
  return force, d_force


def _nominal_contact_rows(m: Model, d: Data) -> jax.Array:
  """Rows of in-contact (``efc_pos < efc_margin``) contacts, decided per
  contact by its first row and broadcast to all of its rows."""
  owner = np.arange(d._impl.efc_type.shape[0])
  for start, ncon, rows in _contact_edge_groups(m):
    owner[start : start + ncon * rows] = start + rows * (
        np.arange(ncon * rows) // rows
    )
  row = (d._impl.efc_type >= ConstraintType.CONTACT_FRICTIONLESS) & (
      d._impl.efc_pos < d._impl.efc_margin
  )
  return row[owner]


def _update_constraint(
    m: Model,
    d: Data,
    ctx: Context,
    stickslip_softness: float = 0.0,
    liftoff_softness: float = 0.0,
    limit_softness: float = 0.0,
    frictionloss_softness: float = 0.0,
) -> Context:
  """Updates constraint force and resulting cost given last solver iteration.

  Corresponds to CGupdateConstraint in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest qacc and smooth terms
    ctx: current solver context
    stickslip_softness: if this or ``liftoff_softness`` is > 0, the force of
      nominal contacts (in contact, ``efc_pos < efc_margin``) is minus the
      gradient of a smoothed convex per-contact cost (``_contact_cost_force``)
      instead of the hard force: stick/slip is smoothed over a fraction
      ``stickslip_softness`` of friction utilization at any load (pyramid
      edges: ``_contact_cost_force``; elliptic cone: ``_elliptic_group_force``).
      Used only by the IFT residual (``_root``), so the forward solve is
      unchanged. Built on the non-gated softplus: the
      gated ``x * sigmoid(x)`` has negative slopes, which can make the IFT
      residual Jacobian indefinite.
    liftoff_softness: liftoff band of that cost, in solimp widths (0 keeps
      liftoff hard).
    limit_softness: if > 0, engaged joint/tendon limit rows (``efc_pos < efc_margin``)
      use ``D * _relu_per_row(-Jaref, tau)`` (IFT residual only).
    frictionloss_softness: if > 0, friction-loss rows use a smooth clip of
      ``-Jaref`` to ``[-r f, r f]`` instead of the hard clip (IFT residual only).

    Each softness is dimensionless, in the row's own Jaref scale
    ``efc_softness_scale`` (see ``constraint.make_constraint``): limit and
    friction-loss rows use ``tau = softness * efc_softness_scale``; contacts
    use it as the liftoff length of their cost. The scale is held constant
    (``stop_gradient``) so it adds no gradient of its own.

  Returns:
    context with new constraint force and costs
  """
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('_update_constraint requires JAX backend implementation.')

  # ne constraints are always active, nf are conditionally active, others are
  # non-negative constraints.
  active = (ctx.Jaref < 0).at[: d._impl.ne + d._impl.nf].set(True)
  # Per-row Jaref scale of the constraint cost smoothing bands (constant).
  scale = jax.lax.stop_gradient(d._impl.efc_softness_scale)

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
  elif m.opt.cone == ConeType.ELLIPTIC:
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

  if stickslip_softness > 0 or liftoff_softness > 0:
    # Smooth nominal contacts only: pw_solimp contacts (efc_pos >= efc_margin)
    # and padding rows keep the hard force. Whole contacts are selected by
    # their first (normal) row, so the per-contact force stays a gradient
    # (elliptic friction rows have efc_pos 0).
    nominal = _nominal_contact_rows(m, d)
    smooth_force = _contact_cost_force(
        m, d, ctx.Jaref, scale, stickslip_softness, liftoff_softness
    )
    efc_force = jp.where(nominal, smooth_force, efc_force)

  if limit_softness > 0:
    # Engaged limit rows only; disengaged rows keep their zero force.
    is_limit = (d._impl.efc_type == ConstraintType.LIMIT_JOINT) | (
        d._impl.efc_type == ConstraintType.LIMIT_TENDON
    )
    engaged = is_limit & (d._impl.efc_pos < d._impl.efc_margin)
    smooth_force = d._impl.efc_D * _relu_per_row(-ctx.Jaref, limit_softness * scale)
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
    tau = frictionloss_softness * scale
    soft_clip = (
        x
        - _relu_per_row(x - b, tau)
        + _relu_per_row(-x - b, tau)
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
  ctx = Context.create(
      m, d.replace(qacc=qacc), grad=False, **_softness_kwargs(_softnesses(m))
  )
  return ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint


def _root_iterative_ctx(m: Model, d: Data, qacc: jax.Array) -> Context:
  """Runs the solver to convergence from ``qacc``; returns the final context.

  This solve is never differentiated, so it uses the masked scan rather than
  ``lax.while_loop``, which is slow on GPU; results and iteration counts are
  identical to the while loop. The context's ``Ma`` and ``qfrc_constraint``
  are those of the returned ``qacc``.
  """
  d = d.replace(qacc=qacc)
  ctx = Context.create(m, d)

  def cond(ctx: Context) -> jax.Array:
    return ~_solver_done(m, ctx)

  def body(ctx: Context) -> Context:
    return _solver_step(m, d, ctx)

  if m.opt.iterations == 1:
    return body(ctx)
  return _while_loop_scan(cond, body, ctx, max_iter=m.opt.iterations)


def _solver_niter(ctx: Context, dtype) -> jax.Array:
  # Float, not int: custom_root's JVP gives integer aux outputs an integer
  # tangent, which JAX rejects (integer tangents must be float0).
  return jp.asarray(ctx.solver_niter, dtype=dtype)


def _ift_factor(m: Model, d: Data, qacc: jax.Array, softnesses) -> tuple:
  """Cholesky factor of the residual Jacobian ``dr/dqacc`` at ``qacc``.

  ``r(q) = M q - qfrc_smooth - J^T f(J q - aref)`` and only the constraint
  force is nonlinear, so ``dr/dq = M - J^T (df/dJaref) J``: ``(df/dJaref) J``
  takes one JVP of the force per column of ``J``, without pushing tangents
  through ``M``, ``J`` and ``J^T``. It is symmetric positive definite (the
  force is minus the gradient of the convex constraint cost), so Cholesky.
  """
  softness = _softness_kwargs(softnesses)
  # The context only supplies Jaref (and the elliptic cone's scaled friction):
  # build it without smoothing, which the force below applies itself.
  ctx = Context.create(m, d.replace(qacc=qacc), grad=False)
  efc_j = d._impl.efc_J

  def force(jaref):
    return _update_constraint(m, d, ctx.replace(Jaref=jaref), **softness).efc_force

  df_j = jax.vmap(
      lambda t: jax.jvp(force, (ctx.Jaref,), (t,))[1], in_axes=1, out_axes=1
  )(efc_j)
  h = support.full_m(m, d) - efc_j.T @ df_j
  return jax.scipy.linalg.cho_factor(0.5 * (h + h.T))


def _residual_floor(d: Data, ctx: Context) -> jax.Array:
  """Precision floor of ``|Ma - qfrc_smooth - qfrc_constraint|`` at ``ctx``.

  The residual cannot be resolved below the rounding error of the
  cancellation between its terms.
  """
  scale = (
      jp.linalg.norm(ctx.Ma)
      + jp.linalg.norm(d.qfrc_smooth)
      + jp.linalg.norm(ctx.qfrc_constraint)
  )
  return 64 * jp.finfo(ctx.qacc.dtype).eps * scale


def _solve_ift(m: Model, d: Data) -> Data:
  """Finds forces using a custom-root implicit gradient.

  The hard solve (and, with constraint cost smoothing on and
  ``opt.refinement_iterations > 0``, Newton refinement toward the smoothed
  residual's root) runs first, without gradients; ``custom_root`` then only
  supplies the implicit derivative at that point, solved with the Cholesky
  factor of the residual Jacobian built from force JVPs (``_ift_factor``).
  With refinement the forward value stays the hard solution and the
  derivative is that of the smoothed root (straight-through).
  """
  _check_jax_impl(m)

  iterations = _refinement_iterations(m)
  softnesses = _effective_softnesses(m, d)
  # With every (effective) softness 0 the smoothed root is the hard solution.
  refine = iterations > 0 and any(s > 0 for s in softnesses)

  d_const = jax.tree_util.tree_map(jax.lax.stop_gradient, d)
  ctx_hard = _root_iterative_ctx(m, d_const, _warmstart_qacc(m, d_const))
  qacc_hard = ctx_hard.qacc
  # The iteration count is reported, not differentiated.
  solver_niter = _solver_niter(ctx_hard, qacc_hard.dtype)
  qacc_eval = qacc_hard
  if refine:
    # The smoothed state at q* is evaluated fresh, not built from the forward
    # solve's incrementally updated Jaref and Ma: that saved no measurable
    # time and made k = 1 refinement less accurate.
    qacc_eval, _ = _refine_smoothed_root(
        lambda qacc: _root(m, d_const, qacc), qacc_hard, iterations,
        _refinement_tolerance(m), _residual_floor(d_const, ctx_hard),
        _refinement_ls_iterations(m), _refinement_fallback_enable(m),
        _smoothed_newton(m, d_const, softnesses),
    )
  factor = _ift_factor(m, d_const, qacc_eval, softnesses)
  # custom_root differentiates the residual in d at qacc_eval; the tangent
  # system is solved with the factor (linear in y, so reverse mode transposes
  # it; the matrix is symmetric).
  qacc = jax.lax.custom_root(
      lambda qacc: _root(m, d, qacc), qacc_eval, lambda _, x: x,
      lambda _, y: jax.scipy.linalg.cho_solve(factor, y),
  )
  if refine:
    # Value of the hard solution, derivative of the smoothed root.
    qacc = jax.lax.stop_gradient(qacc_hard) + (qacc - jax.lax.stop_gradient(qacc))
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
    the stationarity residual with a Cholesky factorization of its Jacobian
    (``_ift_factor``). The gradient is that of the converged fixed
    point, so it differs from the truncated solve when the forward solve
    stops before convergence (e.g. ``opt.iterations == 1``).

  For every backend the line search loop is selected by ``opt.scan_loop``.

  The constraint cost smoothing options (``opt.stickslip_softness``,
  ``opt.liftoff_softness``, ``opt.limit_softness``,
  ``opt.frictionloss_softness``) only act in the
  ``ift`` backward pass; enabling one with another backend raises instead of
  being silently ignored. ``opt.refinement_iterations`` (Newton
  iterations to the smoothed root, where the gradient is then evaluated) only
  acts when some softness is > 0.
  """
  backend = grad_backend(m)
  softnesses = _softnesses(m)
  _refinement_iterations(m)  # validate on every backend
  _refinement_tolerance(m)
  _refinement_ls_iterations(m)
  _refinement_fallback_enable(m)
  if backend != 'ift' and any(softness > 0 for softness in softnesses):
    enabled = [n for n, t in zip(_SOFTNESS_FIELDS, softnesses) if t > 0]
    raise ValueError(
        f'{", ".join(enabled)} > 0 requires the ift gradient backend '
        f'(opt.grad_backend; got {backend!r}); active-set '
        'smoothing only acts in the IFT backward pass.'
    )
  return _BACKENDS[backend](m, d)
