"""Gradient checks for the MJX solver derivative backends.

Backends are selected with ``MJX_GRAD_BACKEND`` at trace time, so every
function built here is traced inside :func:`backend`.
"""

import contextlib
import os
from typing import Callable

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import solver
from mujoco.mjx._src.types import JointType
import numpy as np

PARAM_KINDS = ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'friction', 'damping')
FD_EPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def compile_with(name: str, fn, *args):
  """AOT-compiles ``fn(*args)`` with backend ``name`` (tracing is lazy, so a
  plain ``jax.jit`` would only read ``MJX_GRAD_BACKEND`` at first call)."""
  with backend(name):
    return jax.jit(fn).lower(*args).compile()


@contextlib.contextmanager
def backend(name: str):
  """Selects the solver backend for functions traced inside the block."""
  old = os.environ.get('MJX_GRAD_BACKEND')
  os.environ['MJX_GRAD_BACKEND'] = name
  try:
    yield
  finally:
    if old is None:
      os.environ.pop('MJX_GRAD_BACKEND', None)
    else:
      os.environ['MJX_GRAD_BACKEND'] = old


def param_dim(m: mujoco.MjModel, kind: str) -> int:
  return {
      'qpos': m.nv, 'qvel': m.nv, 'ctrl': m.nu, 'qfrc_applied': m.nv,
      'friction': m.ngeom, 'damping': m.nv,
  }[kind]


def apply_param(mx: mjx.Model, dx: mjx.Data, kind: str, theta: jax.Array):
  """Returns (model, data) with ``theta`` applied as a perturbation.

  ``qpos`` perturbations are tangent-space displacements retracted onto the
  joint manifold, so free/ball joint quaternions stay normalized.
  """
  if kind == 'qpos':
    return mx, dx.replace(qpos=retract_qpos(mx, dx.qpos, theta))
  if kind == 'qvel':
    return mx, dx.replace(qvel=dx.qvel + theta)
  if kind == 'ctrl':
    return mx, dx.replace(ctrl=dx.ctrl + theta)
  if kind == 'qfrc_applied':
    return mx, dx.replace(qfrc_applied=dx.qfrc_applied + theta)
  if kind == 'friction':
    fr = mx.geom_friction.at[:, 0].multiply(1.0 + theta)
    return mx.replace(geom_friction=fr), dx
  if kind == 'damping':
    return mx.replace(dof_damping=mx.dof_damping * (1.0 + theta)), dx
  raise ValueError(kind)


def _retract_quat(q, w):
  # q * normalize([1, w/2]): a smooth retraction whose derivative at w = 0
  # matches the exponential map (math.quat_integrate divides by |w| and has a
  # degenerate derivative at w = 0).
  dq = jp.concatenate([jp.ones(1, w.dtype), 0.5 * w])
  return math.quat_mul(q, dq / jp.linalg.norm(dq))


def retract_qpos(mx: mjx.Model, qpos: jax.Array, dq: jax.Array) -> jax.Array:
  """qpos (+) dq for a tangent vector ``dq`` of size nv."""
  qs, qi, vi = [], 0, 0
  for jnt_typ in mx.jnt_type:
    if jnt_typ == JointType.FREE:
      qs.append(qpos[qi:qi + 3] + dq[vi:vi + 3])
      qs.append(_retract_quat(qpos[qi + 3:qi + 7], dq[vi + 3:vi + 6]))
      qi, vi = qi + 7, vi + 6
    elif jnt_typ == JointType.BALL:
      qs.append(_retract_quat(qpos[qi:qi + 4], dq[vi:vi + 3]))
      qi, vi = qi + 4, vi + 3
    else:
      qs.append(qpos[qi:qi + 1] + dq[vi:vi + 1])
      qi, vi = qi + 1, vi + 1
  return jp.concatenate(qs)


def loss_weights(nv: int, seed: int = 0):
  rng = np.random.default_rng(seed)
  return jp.asarray(rng.normal(size=nv)), jp.asarray(rng.normal(size=nv))


def make_step_loss(mx, dx, kind, seed=0) -> Callable[[jax.Array], jax.Array]:
  """Scalar loss of one forward() call: w_a . qacc + w_c . qfrc_constraint."""
  wa, wc = loss_weights(mx.nv, seed)
  # qfrc_constraint is O(mg); scale so both terms contribute comparably.
  def loss(theta):
    m, d = apply_param(mx, dx, kind, theta)
    d = mjx.forward(m, d)
    return jp.dot(wa, d.qacc) + 1e-1 * jp.dot(wc, d.qfrc_constraint)

  return loss


def central_fd(f, x0, eps, directions=None):
  """Central differences of scalar ``f`` along unit basis or ``directions``."""
  if directions is None:
    directions = jp.eye(x0.shape[0], dtype=x0.dtype)

  def one(v):
    return (f(x0 + eps * v) - f(x0 - eps * v)) / (2 * eps)

  return jax.vmap(one)(directions)


def compare(g, g_ref):
  g, g_ref = np.asarray(g, dtype=float), np.asarray(g_ref, dtype=float)
  nref = np.linalg.norm(g_ref)
  err = np.linalg.norm(g - g_ref)
  cos = float(np.dot(g, g_ref) / max(np.linalg.norm(g) * nref, 1e-300))
  return {
      'abs_err': float(err),
      'rel_err': float(err / max(nref, 1e-12)),
      'cos': cos,
      'ref_norm': float(nref),
      'finite': bool(np.all(np.isfinite(g))),
  }


def compare_per_component(g, fds: dict, rtol=1e-4) -> dict:
  """Per-component agreement with the best of several FD step sizes.

  Piecewise-smooth maps (e.g. truncated solves whose warmstart choice flips)
  make every global FD step straddle a jump for some component. A component
  "agrees" if FD at some step size matches it within ``rtol``; the fraction of
  agreeing components separates AD errors from FD-visible discontinuities.
  """
  g = np.asarray(g, dtype=float)
  stack = np.stack([np.asarray(v, dtype=float) for v in fds.values()])
  scale = np.maximum(np.abs(g), 1e-8 * max(np.abs(g).max(), 1e-300))
  rel = np.abs(stack - g[None]) / scale[None]
  best = rel.min(axis=0)
  return {'frac_agree': float(np.mean(best < rtol)),
          'max_best_rel': float(best.max())}


def solution_diagnostics(mx: mjx.Model, dx: mjx.Data) -> dict:
  """Fixed-point diagnostics of a solved ``dx`` (after mjx.forward)."""
  out = {'nefc': int(dx._impl.nefc), 'solver_niter': int(dx._impl.solver_niter)}
  if dx._impl.nefc == 0:
    return out
  qacc = dx.qacc
  root = lambda a: solver._root(mx, dx, a)
  res = root(qacc)
  jac = jax.jacfwd(root)(qacc)
  ctx = solver.Context.create(mx, dx.replace(qacc=qacc), grad=False)
  jaref = np.asarray(ctx.Jaref)
  ne_nf = int(dx._impl.ne + dx._impl.nf)
  ineq = jaref[ne_nf:]
  sv = np.linalg.svd(np.asarray(jac), compute_uv=False)
  jac_np = np.asarray(jac)
  out.update({
      'residual_norm': float(jp.linalg.norm(res)),
      'residual_rescaled': float(
          jp.linalg.norm(res) / (mx.stat.meaninertia * max(1, mx.nv))
      ),
      'n_active': int(np.sum(np.asarray(ctx.active))),
      # distance of inactive/active inequality rows from the switch Jaref=0,
      # relative to the largest |Jaref|; small values flag near-kinks.
      'switch_margin_rel': float(
          np.min(np.abs(ineq)) / max(np.max(np.abs(ineq)), 1e-300)
      ) if ineq.size else None,
      'jac_cond': float(sv[0] / max(sv[-1], 1e-300)),
      'jac_asym_rel': float(
          np.linalg.norm(jac_np - jac_np.T) / max(np.linalg.norm(jac_np), 1e-300)
      ),
  })
  return out


def grad_fn(mx, dx, kind, name, seed=0):
  loss = make_step_loss(mx, dx, kind, seed)
  theta0 = jp.zeros(param_dim_mx(mx, kind))
  return compile_with(name, jax.grad(loss), theta0)


def check_step(mx, dx, kind, backends=('ift', 'scan', 'unrolled'), seed=0,
               fd_backend='while', fd_eps=FD_EPS) -> dict:
  """Single forward() gradient: each backend vs central FD and vs each other."""
  n = int(param_dim_mx(mx, kind))
  theta0 = jp.zeros(n)
  f = make_step_loss(mx, dx, kind, seed)
  fd = {}
  for eps in fd_eps:
    fd_e = compile_with(fd_backend, lambda x, e=eps: central_fd(f, x, e), theta0)
    fd[eps] = np.asarray(fd_e(theta0))
  # FD self-consistency: how much the FD estimate moves between adjacent eps.
  fd_drift = {
      f'{a:g}->{b:g}': compare(fd[b], fd[a])['rel_err']
      for a, b in zip(fd_eps[:-1], fd_eps[1:])
  }
  out = {'kind': kind, 'dim': n, 'fd_drift': fd_drift, 'backends': {}}
  grads = {}
  for name in backends:
    g = np.asarray(grad_fn(mx, dx, kind, name, seed)(theta0))
    grads[name] = g
    per_eps = {f'{e:g}': compare(g, fd[e]) for e in fd_eps}
    best = min(per_eps.items(), key=lambda kv: kv[1]['rel_err'])
    out['backends'][name] = {
        'grad_norm': float(np.linalg.norm(g)),
        'finite': bool(np.all(np.isfinite(g))),
        'fd_best_eps': best[0],
        'fd_best': best[1],
        'fd_per_eps': per_eps,
        'fd_per_component': compare_per_component(g, fd),
    }
  names = list(grads)
  out['pairwise'] = {
      f'{a}_vs_{b}': compare(grads[a], grads[b])
      for i, a in enumerate(names) for b in names[i + 1:]
  }
  return out


def param_dim_mx(mx: mjx.Model, kind: str) -> int:
  return {
      'qpos': mx.nv, 'qvel': mx.nv, 'ctrl': mx.nu, 'qfrc_applied': mx.nv,
      'friction': mx.ngeom, 'damping': mx.nv,
  }[kind]


def make_rollout_loss(mx, dx, horizon: int, seed: int = 0, substeps: int = 1):
  """Loss over a trajectory driven by an open-loop control sequence.

  Controls (or generalized forces, for unactuated scenes) are the decision
  variables, shape (horizon, nu). Each control step runs ``substeps``
  physics steps. The loss tracks a target pose and penalizes velocity.
  """
  actuated = mx.nu > 0
  nu = mx.nu if actuated else mx.nv
  rng = np.random.default_rng(seed)
  wq = jp.asarray(rng.uniform(0.5, 1.5, size=mx.nq))
  target = dx.qpos + jp.asarray(rng.normal(scale=0.05, size=mx.nq))

  def loss(u_seq):
    def ctrl_step(d, u):
      if actuated:
        d = d.replace(ctrl=dx.ctrl + u)
      else:
        d = d.replace(qfrc_applied=dx.qfrc_applied + u)

      def phys(d, _):
        return mjx.step(mx, d), None

      d, _ = jax.lax.scan(phys, d, None, length=substeps)
      c = jp.sum(wq * jp.square(d.qpos - target)) + 1e-2 * jp.sum(
          jp.square(d.qvel))
      return d, c

    _, costs = jax.lax.scan(ctrl_step, dx, u_seq)
    return jp.sum(costs)

  return loss, (horizon, nu)


def check_rollout(mx, dx, horizon, backends=('ift', 'scan', 'unrolled'),
                  ndir=8, seed=0, fd_backend='while', fd_eps=FD_EPS,
                  substeps=1) -> dict:
  """Trajectory gradient vs directional central FD along random directions."""
  # put_data and step outputs differ in some integer dtypes under x64; one
  # forward() (which leaves qpos/qvel unchanged) gives a scan-stable carry.
  dx = compile_with(fd_backend, mjx.forward, mx, dx)(mx, dx)
  f, loss_shape = make_rollout_loss(mx, dx, horizon, seed, substeps)
  rng = np.random.default_rng(seed + 1)
  dirs = rng.normal(size=(ndir,) + loss_shape)
  dirs /= np.linalg.norm(dirs.reshape(ndir, -1), axis=1)[:, None, None]
  dirs = jp.asarray(dirs)
  u0 = jp.zeros(loss_shape)
  fd = {}
  for eps in fd_eps:
    fd_e = compile_with(fd_backend, lambda x, e=eps: jax.vmap(
        lambda v: (f(x + e * v) - f(x - e * v)) / (2 * e))(dirs), u0)
    fd[eps] = np.asarray(fd_e(u0))
  out = {'horizon': horizon, 'substeps': substeps, 'ndir': ndir,
         'loss': float(compile_with(fd_backend, f, u0)(u0)), 'backends': {},
         'fd_drift': {
             f'{a:g}->{b:g}': compare(fd[b], fd[a])['rel_err']
             for a, b in zip(fd_eps[:-1], fd_eps[1:])}}
  grads = {}
  for name in backends:
    g = np.asarray(compile_with(name, jax.grad(f), u0)(u0))
    grads[name] = g
    dd = np.asarray(dirs).reshape(ndir, -1) @ g.ravel()
    per_eps = {f'{e:g}': compare(dd, fd[e]) for e in fd_eps}
    best = min(per_eps.items(), key=lambda kv: kv[1]['rel_err'])
    out['backends'][name] = {
        'grad_norm': float(np.linalg.norm(g)),
        'finite': bool(np.all(np.isfinite(g))),
        'fd_best_eps': best[0], 'fd_best': best[1], 'fd_per_eps': per_eps,
    }
  names = list(grads)
  out['pairwise'] = {
      f'{a}_vs_{b}': compare(grads[a].ravel(), grads[b].ravel())
      for i, a in enumerate(names) for b in names[i + 1:]
  }
  return out
