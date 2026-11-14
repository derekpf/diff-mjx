"""Benchmark harness for the MJX solver derivative backends.

Each configuration runs in a fresh subprocess so compile caches and device
memory statistics are isolated. Typical use:

  python bench.py --sweep smoke  --out results/smoke.jsonl
  python bench.py --sweep iters  --out results/iters.jsonl
  python bench.py --sweep batch  --out results/batch.jsonl
  python bench.py --sweep horizon --out results/horizon.jsonl
  python bench.py --sweep tol    --out results/tol.jsonl

Measured per configuration (median and IQR over --reps repeats, after
compilation and --warmup untimed calls, synchronized with block_until_ready):

  t_f        forward-only loss (value) latency
  t_fb       fused value_and_grad latency
  t_b_eff    t_fb - t_f                       (includes any recomputation)
  t_fwd_res  forward that also saves VJP residuals (jax.vjp)
  t_b_dir    VJP application alone, timed directly on saved residuals
  R          t_b_eff / t_f ;   R_dir = t_b_dir / t_f
  compile_*  wall-clock lower+compile seconds
  mem_*      XLA buffer-assignment estimate (memory_analysis) and runtime
             peak_bytes_in_use for the process

Workloads
  step     one mjx.forward per batch element (isolated solve inside forward)
  rollout  BPTT through `horizon` control steps x `substeps` mjx.step calls;
           gradient w.r.t. the open-loop control sequence.
"""

import argparse
import dataclasses
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).parent


@dataclasses.dataclass
class Config:
  scene: str = 'go1'
  workload: str = 'rollout'  # step | rollout
  backend: str = 'ift'  # while | unrolled | scan | ift
  iterations: int = 3
  ls_iterations: int = 5
  tolerance: float = 1e-8
  batch: int = 64
  horizon: int = 8
  substeps: int = 1
  precision: str = 'f32'  # f32 | f64
  checkpoint: str = 'none'  # none | step | substep
  cone: str = 'pyramidal'
  solver: str = 'newton'
  grad: bool = True  # False -> forward-only measurement (e.g. while)
  reps: int = 20
  warmup: int = 3
  seed: int = 0


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------


def _worker(cfg: Config) -> dict:
  import jax  # pylint: disable=g-import-not-at-top

  if cfg.precision == 'f64':
    jax.config.update('jax_enable_x64', True)
  from jax import numpy as jp  # pylint: disable=g-import-not-at-top
  import mujoco  # pylint: disable=g-import-not-at-top
  from mujoco import mjx  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  sys.path.insert(0, str(_HERE))
  import scenes  # pylint: disable=g-import-not-at-top

  backend = _check_backend(cfg.backend)

  solver_ = {'newton': mujoco.mjtSolver.mjSOL_NEWTON,
             'cg': mujoco.mjtSolver.mjSOL_CG}[cfg.solver]
  cone = {'pyramidal': mujoco.mjtCone.mjCONE_PYRAMIDAL,
          'elliptic': mujoco.mjtCone.mjCONE_ELLIPTIC}[cfg.cone]
  sc = scenes.make_scene(cfg.scene, solver=solver_, cone=cone,
                         iterations=cfg.iterations,
                         ls_iterations=cfg.ls_iterations,
                         tolerance=cfg.tolerance)
  mx = mjx.put_model(sc.model)
  mx = mx.replace(opt=mx.opt.replace(grad_backend=backend))
  dx0 = mjx.put_data(sc.model, sc.data)

  # Batch of initial states: small velocity perturbations around the scene.
  rng = np.random.default_rng(cfg.seed)
  dqvel = jp.asarray(rng.normal(scale=0.05, size=(cfg.batch, mx.nv)))
  actuated = mx.nu > 0
  nu = mx.nu if actuated else mx.nv
  wq = jp.asarray(rng.uniform(0.5, 1.5, size=mx.nq))
  target = dx0.qpos + jp.asarray(rng.normal(scale=0.05, size=mx.nq))
  wa = jp.asarray(rng.normal(size=mx.nv))

  def apply_u(d, u):
    if actuated:
      return d.replace(ctrl=dx0.ctrl + u)
    return d.replace(qfrc_applied=dx0.qfrc_applied + u)

  maybe_ckpt = lambda f, lvl: jax.checkpoint(f) if cfg.checkpoint == lvl else f

  if cfg.workload == 'step':
    u_shape = (cfg.batch, nu)

    def loss_one(u, dqv):
      d = apply_u(dx0.replace(qvel=dx0.qvel + dqv), u)
      d = mjx.forward(mx, d)
      niter = d._impl.solver_niter.astype(float)
      return jp.dot(wa, d.qacc) + 1e-1 * jp.dot(wa, d.qfrc_constraint), niter
  elif cfg.workload == 'rollout':
    u_shape = (cfg.batch, cfg.horizon, nu)

    def loss_one(u_seq, dqv):
      def phys(d, _):
        d = mjx.step(mx, d)
        return d, d._impl.solver_niter.astype(float)

      phys = maybe_ckpt(phys, 'substep')

      def ctrl_step(d, u):
        d = apply_u(d, u)
        d, niter = jax.lax.scan(phys, d, None, length=cfg.substeps)
        c = jp.sum(wq * jp.square(d.qpos - target)) + 1e-2 * jp.sum(
            jp.square(d.qvel))
        return d, (c, jp.mean(niter))

      ctrl_step = maybe_ckpt(ctrl_step, 'step')
      d = dx0.replace(qvel=dx0.qvel + dqv)
      _, (costs, niter) = jax.lax.scan(ctrl_step, d, u_seq)
      return jp.sum(costs), jp.mean(niter)
  else:
    raise ValueError(cfg.workload)

  def loss_batch(u):
    v, niter = jax.vmap(loss_one)(u, dqvel)
    return jp.sum(v), (v, niter)

  dtype = jp.float64 if cfg.precision == 'f64' else jp.float32
  u0 = jp.zeros(u_shape, dtype=dtype)

  def mem(compiled):
    try:
      ma = compiled.memory_analysis()
      return {'temp': int(ma.temp_size_in_bytes),
              'arg': int(ma.argument_size_in_bytes),
              'out': int(ma.output_size_in_bytes),
              'alias': int(ma.alias_size_in_bytes)}
    except Exception:  # pylint: disable=broad-except
      return None

  res = {'config': dataclasses.asdict(cfg), 'device': str(jax.devices()[0]),
         'mjx_grad_backend': backend,  # key kept for existing result files
         'nv': int(mx.nv), 'nefc': int(dx0._impl.nefc), 'nu': int(nu),
         'ncon': int(dx0._impl.ncon)}

  def timed_compile(fn, *args):
    t0 = time.perf_counter()
    c = jax.jit(fn).lower(*args).compile()
    return c, time.perf_counter() - t0

  f_val, res['compile_f'] = timed_compile(loss_batch, u0)
  res['mem_f'] = mem(f_val)
  if cfg.grad:
    f_vg, res['compile_fb'] = timed_compile(
        jax.value_and_grad(loss_batch, has_aux=True), u0)
    res['mem_fb'] = mem(f_vg)

    # forward that stores residuals; returns the VJP closure as a pytree.
    def fwd_with_vjp(u):
      out, vjp = jax.vjp(lambda x: loss_batch(x)[0], u)
      return out, vjp

    f_fr, res['compile_fwd_res'] = timed_compile(fwd_with_vjp, u0)
    res['mem_fwd_res'] = mem(f_fr)
    _, vjp0 = f_fr(u0)
    one = jp.ones((), dtype=dtype)
    f_bw, res['compile_b'] = timed_compile(lambda vjp, ct: vjp(ct), vjp0, one)
    res['mem_b'] = mem(f_bw)
    del vjp0

  block = lambda x: jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)

  def bench(fn, *args):
    for _ in range(cfg.warmup):
      block(fn(*args))
    ts = []
    for _ in range(cfg.reps):
      t0 = time.perf_counter()
      block(fn(*args))
      ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts) * 1e3
    return {'median': float(np.median(ts)),
            'q25': float(np.percentile(ts, 25)),
            'q75': float(np.percentile(ts, 75)),
            'min': float(ts.min()), 'n': int(ts.size)}

  # Interleave forward / grad timing in rounds to share contention evenly.
  rounds = 2
  samples = {'t_f': [], 't_fb': [], 't_fwd_res': [], 't_b_dir': []}
  reps = cfg.reps
  cfg_reps = max(1, reps // rounds)
  for _ in range(rounds):
    cfg.reps = cfg_reps
    samples['t_f'].append(bench(f_val, u0))
    if cfg.grad:
      samples['t_fb'].append(bench(f_vg, u0))
      samples['t_fwd_res'].append(bench(f_fr, u0))
      _, vjp = f_fr(u0)
      samples['t_b_dir'].append(bench(f_bw, vjp, one))
      del vjp
  cfg.reps = reps

  def merge(lst):
    if not lst:
      return None
    med = float(np.median([s['median'] for s in lst]))
    return {'median': med, 'q25': float(min(s['q25'] for s in lst)),
            'q75': float(max(s['q75'] for s in lst)),
            'min': float(min(s['min'] for s in lst)),
            'n': int(sum(s['n'] for s in lst))}

  for k, v in samples.items():
    res[k] = merge(v)

  # Outputs: gradient of element 0 and solver iteration statistics.
  v, (vals, niter) = f_val(u0)
  res['loss'] = float(v)
  res['niter_mean'] = float(jp.mean(niter))
  res['niter_max'] = float(jp.max(niter))
  res['finite_loss'] = bool(np.all(np.isfinite(np.asarray(vals))))
  if cfg.grad:
    (_, _), g = f_vg(u0)
    g = np.asarray(g)
    res['grad_finite'] = bool(np.all(np.isfinite(g)))
    res['grad_norm'] = float(np.linalg.norm(g))
    res['grad0'] = np.asarray(g[0], dtype=float).ravel().tolist()
    tf, tfb = res['t_f']['median'], res['t_fb']['median']
    res['t_b_eff'] = tfb - tf
    res['R'] = (tfb - tf) / tf
    res['R_dir'] = res['t_b_dir']['median'] / tf
    steps = cfg.batch * (cfg.horizon * cfg.substeps
                         if cfg.workload == 'rollout' else 1)
    res['throughput_fb'] = steps / (tfb * 1e-3)
  steps = cfg.batch * (cfg.horizon * cfg.substeps
                       if cfg.workload == 'rollout' else 1)
  res['throughput_f'] = steps / (res['t_f']['median'] * 1e-3)
  try:
    res['mem_runtime_peak'] = int(
        jax.devices()[0].memory_stats()['peak_bytes_in_use'])
  except Exception:  # pylint: disable=broad-except
    res['mem_runtime_peak'] = None
  return res


BACKENDS = ('while', 'unrolled', 'scan', 'ift')


def _check_backend(name: str) -> str:
  if name not in BACKENDS:
    raise ValueError(f'unknown backend {name!r}; known: {", ".join(BACKENDS)}')
  return name


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def _gpu_util() -> dict:
  try:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10).stdout.strip()
    util, used = out.split(',')
    return {'gpu_util_pct': int(util), 'gpu_mem_used_mib': int(used)}
  except Exception:  # pylint: disable=broad-except
    return {}


def run_config(cfg: Config, timeout: int = 1800) -> dict:
  env = dict(os.environ)
  env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  before = _gpu_util()
  t0 = time.time()
  try:
    p = subprocess.run(
        [sys.executable, '-I', str(_HERE / 'bench.py'), '--worker',
         json.dumps(dataclasses.asdict(cfg))],
        capture_output=True, text=True, timeout=timeout, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
    if p.returncode != 0 or not line:
      res = {'config': dataclasses.asdict(cfg), 'error': p.stderr[-3000:]}
    else:
      res = json.loads(line[-1][len('RESULT '):])
  except subprocess.TimeoutExpired:
    res = {'config': dataclasses.asdict(cfg), 'error': 'timeout'}
  res['wall_s'] = time.time() - t0
  res['gpu_before'] = before
  res['gpu_after'] = _gpu_util()
  return res


def _fmt(res: dict) -> str:
  c = res['config']
  head = (f"{c['scene']:8s} {c['workload']:7s} {c['backend']:14s} "
          f"K={c['iterations']:<3d} B={c['batch']:<5d} H={c['horizon']:<3d} "
          f"S={c['substeps']} {c['precision']} ck={c['checkpoint']}")
  if 'error' in res:
    err = res['error'].strip().splitlines()
    return head + ' ERROR ' + (err[-1][:200] if err else '')
  s = head + f" f={res['t_f']['median']:.2f}ms"
  if 'R' in res:
    s += (f" fb={res['t_fb']['median']:.2f}ms b_dir="
          f"{res['t_b_dir']['median']:.2f}ms R={res['R']:.2f} "
          f"R_dir={res['R_dir']:.2f} memfb="
          f"{(res['mem_fb'] or {}).get('temp', 0) / 2**20:.1f}MiB")
  s += f" niter={res['niter_mean']:.2f} util={res['gpu_before'].get('gpu_util_pct')}"
  return s


MAIN = ('unrolled', 'scan', 'ift')


def sweep(name: str, backends=None) -> list:
  """Returns a list of Configs for a named sweep."""
  cfgs = []
  add = lambda **kw: cfgs.append(Config(**kw))
  backends = list(backends or MAIN)

  def per_backend(bks=None, **kw):
    add(backend='while', grad=False, **kw)
    for b in (bks or backends):
      add(backend=b, **kw)

  if name == 'smoke':
    per_backend(scene='box_slide', workload='rollout', iterations=3, batch=8,
                horizon=4, reps=6)
  elif name == 'iters':
    for sc, k in itertools.product(('go1', 'g1'), (1, 3, 5, 10, 20, 50)):
      per_backend(scene=sc, workload='rollout', iterations=k, batch=512,
                  horizon=8, reps=10)
  elif name == 'step':
    for sc, k in itertools.product(('box_slide', 'go1', 'g1'), (1, 3, 10, 50)):
      per_backend(scene=sc, workload='step', iterations=k, batch=512)
  elif name == 'batch':
    for b in (1, 64, 512, 2048, 4096):
      per_backend(scene='go1', workload='rollout', iterations=3, batch=b,
                  horizon=8, reps=10)
  elif name == 'horizon':
    for h in (1, 8, 16, 32):
      per_backend(scene='go1', workload='rollout', iterations=3, batch=256,
                  horizon=h, reps=10)
      for b in ('unrolled', 'scan'):
        add(scene='go1', workload='rollout', iterations=3, batch=256,
            horizon=h, backend=b, checkpoint='step', reps=10)
      add(scene='go1', workload='rollout', iterations=3, batch=256,
          horizon=h, backend='ift', checkpoint='step', reps=10)
  elif name == 'tol':
    for tol in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
      per_backend(scene='go1', workload='rollout', iterations=20, batch=256,
                  horizon=8, tolerance=tol, reps=10)
  elif name == 'scenes':
    for sc in ('pendulum', 'cartpole', 'sphere_plane', 'box_stick',
               'box_slide', 'quadruped', 'go1', 'g1'):
      per_backend(scene=sc, workload='rollout', iterations=5, batch=256,
                  horizon=8, reps=10)
  elif name == 'precision':
    for k in (3, 10):
      per_backend(scene='go1', workload='rollout', iterations=k, batch=256,
                  horizon=8, precision='f64', reps=10)
  else:
    raise ValueError(name)
  return cfgs


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument('--worker', default=None, help=argparse.SUPPRESS)
  ap.add_argument('--sweep', default=None)
  ap.add_argument('--backends', nargs='*', default=None)
  ap.add_argument('--out', default=None)
  ap.add_argument('--config', default=None,
                  help='single Config as JSON (keys of the Config dataclass)')
  args = ap.parse_args()

  if args.worker is not None:
    cfg = Config(**json.loads(args.worker))
    res = _worker(cfg)
    print('RESULT ' + json.dumps(res), flush=True)
    return

  if args.config:
    cfgs = [Config(**json.loads(args.config))]
  else:
    cfgs = sweep(args.sweep, args.backends)
  out = open(args.out, 'a') if args.out else None
  for cfg in cfgs:
    res = run_config(cfg)
    print(_fmt(res), flush=True)
    if out:
      out.write(json.dumps(res) + '\n')
      out.flush()


if __name__ == '__main__':
  main()
