"""Microbenchmark of the IFT tangent-solve components, batched over states.

Times, at the solved acceleration of each batch element:
  matrix construction  jax.jacobian(g) (reference) vs vmap(g)(I) ("linmap")
                       vs explicit residual Jacobian jax.jacfwd(F)
  factor + solve       QR + LU-on-R (reference), QR + triangular, LU,
                       Cholesky, Cholesky with LU fallback
  residual VJP         d F / d theta^T lambda (the remaining backward work)

  python profile_tangent.py --scene go1 --batch 512 --out results/tangent.json
"""

import argparse
import json
import pathlib
import sys
import time

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gradcheck as gc  # pylint: disable=g-import-not-at-top
import scenes  # pylint: disable=g-import-not-at-top


def timeit(fn, *args, reps=30, warmup=3):
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append(time.perf_counter() - t0)
  ts = np.asarray(ts) * 1e3
  return {'median': float(np.median(ts)), 'q25': float(np.percentile(ts, 25)),
          'q75': float(np.percentile(ts, 75))}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--scene', default='go1')
  ap.add_argument('--batch', type=int, default=512)
  ap.add_argument('--iterations', type=int, default=10)
  ap.add_argument('--reps', type=int, default=30)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  sc = scenes.make_scene(args.scene, iterations=args.iterations,
                         ls_iterations=5, tolerance=1e-8)
  mx = mjx.put_model(sc.model)
  dx = mjx.put_data(sc.model, sc.data)
  rng = np.random.default_rng(0)
  dqv = jp.asarray(rng.normal(scale=0.05, size=(args.batch, mx.nv)))

  def solved(dq):
    return mjx.forward(mx, dx.replace(qvel=dx.qvel + dq))

  with gc.backend('while'):
    db = jax.jit(jax.vmap(solved))(dqv)
  jax.block_until_ready(db)
  ybar = jp.asarray(rng.normal(size=(args.batch, mx.nv)), db.qacc.dtype)

  def residual(d, a):
    return solver._root(mx, d, a)

  # Linear map of the transposed residual Jacobian, as custom_root passes it.
  def gT(d):
    _, vjp = jax.vjp(lambda a: residual(d, a), d.qacc)
    return lambda v: vjp(v)[0]

  def build_jacrev(d, y):
    return jax.jacobian(gT(d))(y)

  def build_linmap(d, y):
    eye = jp.eye(y.shape[0], dtype=y.dtype)
    return jax.vmap(gT(d), out_axes=1)(eye)

  def build_jacfwd(d, y):
    del y
    return jax.jacfwd(lambda a: residual(d, a))(d.qacc).T

  res = {'scene': args.scene, 'batch': args.batch, 'nv': int(mx.nv),
         'nefc': int(dx._impl.nefc), 'device': str(jax.devices()[0])}
  vm = lambda f: jax.jit(jax.vmap(f))
  for name, fn in (('build_jacrev', build_jacrev),
                   ('build_linmap', build_linmap),
                   ('build_jacfwd', build_jacfwd)):
    res[name] = timeit(vm(fn), db, ybar, reps=args.reps)

  a = vm(build_linmap)(db, ybar)
  ref_solve = jax.vmap(lambda a, y: (lambda q, r: jp.linalg.solve(r, q.T @ y))(
      *jp.linalg.qr(a, mode='reduced')))
  solves = {
      'qr_lu_reference': ref_solve,
      'qr_triangular': jax.vmap(lambda a, y: (
          lambda q, r: jax.scipy.linalg.solve_triangular(r, q.T @ y))(
              *jp.linalg.qr(a, mode='reduced'))),
      'lu': jax.vmap(jp.linalg.solve),
      'cholesky': jax.vmap(lambda a, y: jax.scipy.linalg.cho_solve(
          jax.scipy.linalg.cho_factor(0.5 * (a + a.T)), y)),
  }
  x_ref = np.asarray(jax.jit(solves['lu'])(a, ybar))
  # float64 solve of the same (float32-rounded) matrices, as accuracy reference
  x64 = np.linalg.solve(np.asarray(a, np.float64),
                        np.asarray(ybar, np.float64)[..., None])[..., 0]
  for name, fn in solves.items():
    f = jax.jit(fn)
    res['solve_' + name] = timeit(f, a, ybar, reps=args.reps)
    x = np.asarray(f(a, ybar))
    res['solve_' + name]['rel_err_vs_lu'] = float(
        np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
    per = np.linalg.norm(x - x64, axis=1) / np.linalg.norm(x64, axis=1)
    res['solve_' + name]['rel_err_vs_f64_median'] = float(np.median(per))
    res['solve_' + name]['rel_err_vs_f64_max'] = float(per.max())

  an = np.asarray(a, dtype=np.float64)
  sym = np.linalg.norm(an - np.transpose(an, (0, 2, 1)), axis=(1, 2)) / (
      np.linalg.norm(an, axis=(1, 2)))
  eig = np.linalg.eigvalsh(0.5 * (an + np.transpose(an, (0, 2, 1))))
  res['matrix'] = {
      'asym_rel_max': float(sym.max()),
      'min_eig_min': float(eig.min()),
      'cond_median': float(np.median(eig.max(1) / np.abs(eig).min(1))),
      'cond_max': float(np.max(eig.max(1) / np.abs(eig).min(1))),
      'frac_not_pd': float(np.mean(eig.min(1) <= 0)),
  }

  # Compare with MJX's Newton Hessian M + J^T D_active J at the same point.
  def newton_h(d):
    ctx = solver.Context.create(mx, d, grad=False)
    if mx.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
      h = (d._impl.efc_J.T * d._impl.efc_D * ctx.active) @ d._impl.efc_J
      from mujoco.mjx._src import support  # pylint: disable=g-import-not-at-top
      return support.full_m(mx, d) + h
    return jp.full((mx.nv, mx.nv), jp.nan)

  hn = np.asarray(jax.jit(jax.vmap(newton_h))(db), dtype=np.float64)
  res['matrix']['newton_hessian_rel_diff_max'] = float(np.max(
      np.linalg.norm(hn - an, axis=(1, 2)) / np.linalg.norm(an, axis=(1, 2))))

  # Remaining backward work: VJP of the residual w.r.t. the data (theta).
  lam = jax.jit(solves['lu'])(a, ybar)

  def theta_vjp(d, l):
    _, vjp = jax.vjp(lambda dd: residual(dd, d.qacc), d)
    return vjp(l)[0].qvel

  res['residual_theta_vjp'] = timeit(vm(theta_vjp), db, lam, reps=args.reps)
  res['residual_eval'] = timeit(vm(lambda d: residual(d, d.qacc)), db,
                                reps=args.reps)

  for k, v in res.items():
    if isinstance(v, dict) and 'median' in v:
      extra = (f" err_f64 median={v['rel_err_vs_f64_median']:.1e} "
               f"max={v['rel_err_vs_f64_max']:.1e}"
               if 'rel_err_vs_f64_max' in v else '')
      print(f'{k:28s} {v["median"]:8.3f} ms  [{v["q25"]:.3f}, '
            f'{v["q75"]:.3f}]{extra}')
  print('matrix', res['matrix'])
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
  main()
