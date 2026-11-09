"""Forward and gradient validation of the MJX solver derivative backends.

Runs in float64 (CPU recommended) and writes one JSON file per study:

  python validate.py --out results/ --study matrix
  python validate.py --out results/ --study iterations
  python validate.py --out results/ --study rollout

Studies
  matrix      converged single-step gradients for every scene x cone x solver
              x warmstart x parameter kind; IFT / scan / default vs central FD.
  iterations  IFT vs unrolled gradients when the solve is truncated to K
              iterations, against FD of both the truncated and the converged
              solve. This is the regime of opt.iterations = 1..3 used in RL.
  rollout     gradients of a multi-step trajectory loss w.r.t. the control
              sequence, vs directional central FD.
"""

import argparse
import itertools
import json
import os
import pathlib
import platform
import sys
import time
import traceback

import jax

jax.config.update('jax_enable_x64', True)

from jax import numpy as jp  # pylint: disable=g-import-not-at-top
import mujoco
from mujoco import mjx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gradcheck as gc  # pylint: disable=g-import-not-at-top
import scenes  # pylint: disable=g-import-not-at-top

NEWTON, CG = mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtSolver.mjSOL_CG
PYR, ELL = mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC
_SOLVER = {NEWTON: 'newton', CG: 'cg'}
_CONE = {PYR: 'pyramidal', ELL: 'elliptic'}


def env_info() -> dict:
  return {
      'jax': jax.__version__, 'mujoco': mujoco.__version__,
      'platform': platform.platform(), 'python': platform.python_version(),
      'devices': [str(d) for d in jax.devices()],
      'x64': bool(jax.config.jax_enable_x64),
  }


def _put(scene):
  return mjx.put_model(scene.model), mjx.put_data(scene.model, scene.data)


def forward_equivalence(mx, dx) -> dict:
  """Forward outputs of every backend relative to the while solver."""
  outs = {}
  for name in ('while', 'default', 'scan', 'ift'):
    outs[name] = gc.compile_with(name, mjx.forward, mx, dx)(mx, dx)
  ref = outs['while']
  res = {'while_diag': gc.solution_diagnostics(mx, ref)}
  for name, o in outs.items():
    scale = max(float(jp.abs(ref.qacc).max()), 1e-12)
    fscale = max(float(jp.abs(ref.qfrc_constraint).max()), 1e-12)
    res[name] = {
        'qacc_rel': float(jp.abs(o.qacc - ref.qacc).max() / scale),
        'qfrc_constraint_rel': float(
            jp.abs(o.qfrc_constraint - ref.qfrc_constraint).max() / fscale),
        'efc_force_abs': float(jp.abs(o._impl.efc_force - ref._impl.efc_force
                                      ).max()) if ref._impl.nefc else 0.0,
        'solver_niter': int(o._impl.solver_niter),
    }
  return res


def study_matrix(args) -> list:
  rows = []
  scene_names = args.scenes or scenes.ANALYTIC_SCENES + ('go1', 'g1')
  for name, solver_, cone, warm in itertools.product(
      scene_names, (NEWTON, CG), (PYR, ELL), (True, False)):
    if not warm and name not in ('box_slide', 'quadruped'):
      continue  # warmstart off is checked on a subset
    if name in ('go1', 'g1') and solver_ == CG:
      continue  # CG on robots needs very many iterations; covered by Newton
    if name in ('pendulum', 'cartpole') and cone == ELL:
      # Pre-existing MJX bug (also in the unmodified solver): elliptic cones
      # with constraints but no contacts index with an empty float array.
      rows.append({'scene': name, 'solver': _SOLVER[solver_],
                   'cone': _CONE[cone], 'warmstart': warm,
                   'skipped': 'pre-existing MJX elliptic/no-contact bug'})
      continue
    iters = 100 if solver_ == NEWTON else 400
    t0 = time.time()
    row = {'scene': name, 'solver': _SOLVER[solver_], 'cone': _CONE[cone],
           'warmstart': warm, 'iterations': iters, 'tolerance': 1e-14}
    try:
      sc = scenes.make_scene(name, solver=solver_, cone=cone, iterations=iters,
                             ls_iterations=50, tolerance=1e-14, warmstart=warm)
      mx, dx = _put(sc)
      row['forward'] = forward_equivalence(mx, dx)
      kinds = [k for k in gc.PARAM_KINDS if gc.param_dim(sc.model, k) > 0]
      if name in ('pendulum', 'cartpole'):
        kinds = [k for k in kinds if k != 'friction']
      row['grads'] = {}
      for kind in kinds:
        row['grads'][kind] = gc.check_step(mx, dx, kind)
    except Exception as e:  # pylint: disable=broad-except
      row['error'] = repr(e)
      row['traceback'] = traceback.format_exc()
    row['seconds'] = time.time() - t0
    rows.append(row)
    print(_summ_matrix(row), flush=True)
  return rows


def _summ_matrix(row) -> str:
  head = (f"{row['scene']:12s} {row['solver']:6s} {row['cone']:9s} "
          f"warm={int(row['warmstart'])}")
  if 'error' in row:
    return head + ' ERROR ' + row['error'][:200]
  fw = row['forward']
  parts = [f"niter={fw['while']['solver_niter']} "
           f"res={fw['while_diag'].get('residual_rescaled', 0):.1e} "
           f"fwd_ift={fw['ift']['qacc_rel']:.1e} "
           f"fwd_scan={fw['scan']['qacc_rel']:.1e}"]
  for kind, g in row['grads'].items():
    e = {b: v['fd_best']['rel_err'] for b, v in g['backends'].items()}
    parts.append(f"{kind}:ift={e['ift']:.1e},scan={e['scan']:.1e},"
                 f"i/s={g['pairwise']['ift_vs_scan']['rel_err']:.1e}")
  return head + ' ' + ' '.join(parts)


def study_iterations(args) -> list:
  rows = []
  scene_names = args.scenes or ('box_slide', 'box_stick', 'quadruped', 'go1',
                                'g1')
  for name in scene_names:
    for k in (1, 2, 3, 5, 10, 20, 50):
      t0 = time.time()
      row = {'scene': name, 'iterations': k}
      try:
        # tolerance 0 disables early termination so while == scan forward.
        sc = scenes.make_scene(name, solver=NEWTON, cone=PYR, iterations=k,
                               ls_iterations=50, tolerance=0.0)
        sc_conv = scenes.make_scene(name, solver=NEWTON, cone=PYR,
                                    iterations=100, ls_iterations=50,
                                    tolerance=1e-14)
        mx, dx = _put(sc)
        mxc, _ = _put(sc_conv)
        dxo = gc.compile_with('scan', mjx.forward, mx, dx)(mx, dx)
        row['diag_truncated'] = gc.solution_diagnostics(mx, dxo)
        row['grads'] = {}
        for kind in ('qvel', 'ctrl' if sc.model.nu else 'qfrc_applied'):
          r = {}
          f_k = gc.make_step_loss(mx, dx, kind)
          f_c = gc.make_step_loss(mxc, dx, kind)
          th = jp.zeros(gc.param_dim_mx(mx, kind))
          fd_k = {e: np.asarray(gc.compile_with(
              'scan', lambda x, e=e: gc.central_fd(f_k, x, e), th)(th))
                  for e in gc.FD_EPS}
          fd_c = {e: np.asarray(gc.compile_with(
              'while', lambda x, e=e: gc.central_fd(f_c, x, e), th)(th))
                  for e in gc.FD_EPS}
          for b in ('ift', 'scan', 'default'):
            g = np.asarray(gc.grad_fn(mx, dx, kind, b)(th))
            r[b] = {
                'vs_fd_truncated_per_component': gc.compare_per_component(
                    g, fd_k),
                'vs_fd_truncated': min((gc.compare(g, v) for v in fd_k.values()),
                                       key=lambda c: c['rel_err']),
                'vs_fd_converged': min((gc.compare(g, v) for v in fd_c.values()),
                                       key=lambda c: c['rel_err']),
            }
          r['fd_truncated_vs_converged'] = min(
              (gc.compare(fd_k[e], fd_c[e]) for e in gc.FD_EPS),
              key=lambda c: c['rel_err'])
          row['grads'][kind] = r
      except Exception as e:  # pylint: disable=broad-except
        row['error'] = repr(e)
        row['traceback'] = traceback.format_exc()
      row['seconds'] = time.time() - t0
      rows.append(row)
      print(_summ_iter(row), flush=True)
  return rows


def _summ_iter(row) -> str:
  head = f"{row['scene']:12s} K={row['iterations']:3d}"
  if 'error' in row:
    return head + ' ERROR ' + row['error'][:200]
  dg = row['diag_truncated']
  parts = [f"res={dg.get('residual_rescaled', 0):.1e}"]
  for kind, r in row['grads'].items():
    parts.append(
        f"{kind}: ift->fdK={r['ift']['vs_fd_truncated']['rel_err']:.1e} "
        f"ift->fdC={r['ift']['vs_fd_converged']['rel_err']:.1e} "
        f"scan->fdK={r['scan']['vs_fd_truncated']['rel_err']:.1e} "
        f"(comp-agree={r['scan']['vs_fd_truncated_per_component']['frac_agree']:.2f}) "
        f"scan->fdC={r['scan']['vs_fd_converged']['rel_err']:.1e} "
        f"fdK->fdC={r['fd_truncated_vs_converged']['rel_err']:.1e}")
  return head + ' ' + ' | '.join(parts)


def study_rollout(args) -> list:
  rows = []
  scene_names = args.scenes or ('cartpole', 'box_slide', 'quadruped', 'go1')
  for name in scene_names:
    for horizon in (1, 8, 16):
      for iters, tol in ((100, 1e-14), (3, 0.0)):
        t0 = time.time()
        row = {'scene': name, 'horizon': horizon, 'iterations': iters}
        try:
          sc = scenes.make_scene(name, solver=NEWTON, cone=PYR,
                                 iterations=iters, ls_iterations=50,
                                 tolerance=tol)
          mx, dx = _put(sc)
          fd_backend = 'while' if tol > 0 else 'scan'
          row.update(gc.check_rollout(mx, dx, horizon, fd_backend=fd_backend))
        except Exception as e:  # pylint: disable=broad-except
          row['error'] = repr(e)
          row['traceback'] = traceback.format_exc()
        row['seconds'] = time.time() - t0
        rows.append(row)
        print(_summ_roll(row), flush=True)
  return rows


def _summ_roll(row) -> str:
  head = f"{row['scene']:12s} H={row['horizon']:3d} K={row['iterations']:3d}"
  if 'error' in row:
    return head + ' ERROR ' + row['error'][:200]
  e = {b: v['fd_best']['rel_err'] for b, v in row['backends'].items()}
  return (head + f" ift={e['ift']:.1e} scan={e['scan']:.1e} "
          f"default={e['default']:.1e} "
          f"i/s={row['pairwise']['ift_vs_scan']['rel_err']:.1e} "
          f"fd_drift={max(row['fd_drift'].values()):.1e}")


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('--out', default=str(pathlib.Path(__file__).parent / 'results'))
  p.add_argument('--study', choices=('matrix', 'iterations', 'rollout'),
                 required=True)
  p.add_argument('--scenes', nargs='*', default=None)
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  fn = {'matrix': study_matrix, 'iterations': study_iterations,
        'rollout': study_rollout}[args.study]
  t0 = time.time()
  rows = fn(args)
  out = pathlib.Path(args.out) / f'validation_{args.study}.json'
  out.write_text(json.dumps({'env': env_info(), 'rows': rows,
                             'seconds': time.time() - t0}, indent=1))
  print('wrote', out)


if __name__ == '__main__':
  main()
