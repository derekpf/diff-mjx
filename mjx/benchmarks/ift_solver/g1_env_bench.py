"""BPTT through the diff-playground G1 joystick env, per solver backend.

Mirrors forlax's scripts/train_g1_shac.py environment settings:
G1JoystickFlatTerrain, reward_softness=0.05, DiffEnvWrapper options
(softjax_mode='smooth', pw_solimp=[0, .8, 1, 1, 10], contact_st_enable=True),
num_envs=256, horizon_len=16. The gradient is taken w.r.t. an open-loop
action sequence (B, H, nu) of the summed reward, so per-env gradients are
separable and non-finite values can be attributed to environments.

Start states: reset, then `--warmup` steps of random actions (no gradient),
so the batch covers stance, swing, stumbles and falls.

  python g1_env_bench.py --out results/g1_env_runs.jsonl
"""

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).parent

BACKENDS = ('unrolled', 'scan', 'ift')


@dataclasses.dataclass
class Config:
  backend: str = 'unrolled'
  batch: int = 256
  horizon: int = 16
  warmup: int = 20
  warmup_scale: float = 0.3
  seed: int = 0
  iterations: int = 0  # 0: keep the model's own (G1: 3)
  action_scale: float = 0.3
  reps: int = 10
  ckpt: bool = False


def _worker(cfg: Config) -> dict:
  if cfg.backend not in BACKENDS:
    raise ValueError(f'unknown backend {cfg.backend!r}')
  backend = cfg.backend
  os.environ['MJX_GRAD_BACKEND'] = backend
  import jax  # pylint: disable=g-import-not-at-top
  from jax import numpy as jp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from mujoco_playground import registry  # pylint: disable=g-import-not-at-top

  env = registry.load('G1JoystickFlatTerrain')
  env.reward_softness = 0.05
  env.reward_st_enable = False
  mx = env.mjx_model
  opt = mx.opt.replace(
      softjax_mode='smooth', pw_solimp=jp.array([0.0, 0.8, 1.0, 1.0, 10.0]),
      contact_st_enable=True, sensor_softness=0.0, sensor_st_enable=False)
  if cfg.iterations:
    opt = opt.replace(iterations=cfg.iterations)
  env._mjx_model = mx.replace(opt=opt)  # pylint: disable=protected-access
  nu = env.action_size

  key = jax.random.PRNGKey(cfg.seed)
  k_reset, k_warm, k_act = jax.random.split(key, 3)
  reset = jax.jit(jax.vmap(env.reset))
  step = jax.vmap(env.step)
  state = reset(jax.random.split(k_reset, cfg.batch))

  @jax.jit
  def warm(state, keys):
    def body(s, k):
      a = cfg.warmup_scale * jax.random.uniform(
          k, (cfg.batch, nu), minval=-1, maxval=1)
      return step(s, a), None
    return jax.lax.scan(body, state, keys)[0]

  if cfg.warmup:
    state = warm(state, jax.random.split(k_warm, cfg.warmup))
  jax.block_until_ready(state)
  acts0 = cfg.action_scale * jax.random.normal(
      k_act, (cfg.batch, cfg.horizon, nu))

  # Same rematerialization as forlax's collector (remat_enable=True).
  step_fn = jax.remat(step, prevent_cse=False) if cfg.ckpt else step

  def rollout(acts):
    def body(s, a):
      s = step_fn(s, jp.tanh(a))
      return s, (s.reward, s.done)
    _, (r, done) = jax.lax.scan(body, state, jp.swapaxes(acts, 0, 1))
    return r, done  # (H, B)

  def loss(acts):
    r, done = rollout(acts)
    per_env = -jp.sum(r, axis=0)
    return jp.sum(per_env), (per_env, jp.any(done, axis=0))

  def timed_compile(fn, *args):
    t0 = time.perf_counter()
    c = jax.jit(fn).lower(*args).compile()
    return c, time.perf_counter() - t0

  def mem(c):
    try:
      return int(c.memory_analysis().temp_size_in_bytes)
    except Exception:  # pylint: disable=broad-except
      return None

  def bench(fn, *args):
    for _ in range(3):
      jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(cfg.reps):
      t0 = time.perf_counter()
      jax.block_until_ready(fn(*args))
      ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)

  # Measure one program at a time and free it before compiling the next, so
  # residuals of one measurement do not count against the next.
  f, c_f = timed_compile(loss, acts0)
  mem_f = mem(f)
  t_f = bench(f, acts0)
  del f
  vg, c_fb = timed_compile(jax.value_and_grad(loss, has_aux=True), acts0)
  mem_fb = mem(vg)
  t_fb = bench(vg, acts0)
  (_, (per_env, done)), g = vg(acts0)
  g, per_env, done = np.asarray(g), np.asarray(per_env), np.asarray(done)
  del vg

  t_b, c_fr, c_b, b_err = None, None, None, None
  try:
    fr, c_fr = timed_compile(lambda a: jax.vjp(lambda x: loss(x)[0], a), acts0)
    _, vjp0 = fr(acts0)
    del fr
    one = jp.ones(())
    bw, c_b = timed_compile(lambda v, ct: v(ct), vjp0, one)
    t_b = bench(bw, vjp0, one)
    del bw, vjp0
  except Exception as e:  # pylint: disable=broad-except
    b_err = str(e).strip().splitlines()[0][:300]

  gn = np.linalg.norm(g.reshape(cfg.batch, -1), axis=1)
  nonfinite_env = ~np.all(np.isfinite(g.reshape(cfg.batch, -1)), axis=1)
  fin = gn[np.isfinite(gn)]
  med = float(np.median(fin)) if fin.size else float('nan')
  flat = g.reshape(cfg.batch, -1)
  alive = ~done
  def frac(mask, sel):
    return float(np.mean(mask[sel])) if np.any(sel) else None
  explode = gn > 1e3 * med if fin.size else np.zeros_like(gn, bool)
  res = {
      'alive_env_frac': float(np.mean(alive)),
      'grad_nonfinite_alive_frac': frac(nonfinite_env, alive),
      'grad_nonfinite_done_frac': frac(nonfinite_env, done),
      'grad_exploding_alive_frac': frac(explode, alive),
      'grad_exploding_done_frac': frac(explode, done),
      'grad_norm_median_alive': float(np.median(gn[alive & np.isfinite(gn)]))
                                if np.any(alive & np.isfinite(gn)) else None,
      'grad_norm_max_alive': float(np.max(gn[alive & np.isfinite(gn)]))
                             if np.any(alive & np.isfinite(gn)) else None,
      'config': dataclasses.asdict(cfg), 'device': str(jax.devices()[0]),
      'mjx_grad_backend': backend,
      'iterations': int(env.mjx_model.opt.iterations), 'nu': int(nu),
      't_f': t_f, 't_fb': t_fb, 't_b_dir': t_b, 't_b_dir_error': b_err,
      'R_dir': None if t_b is None else t_b / t_f,
      'R_eff': (t_fb - t_f) / t_f,
      'compile_f': c_f, 'compile_fb': c_fb, 'compile_fwd_res': c_fr,
      'compile_b': c_b, 'mem_f': mem_f, 'mem_fb': mem_fb,
      'loss_finite_frac': float(np.mean(np.isfinite(per_env))),
      'grad_nonfinite_env_frac': float(np.mean(nonfinite_env)),
      'grad_nonfinite_entry_frac': float(np.mean(~np.isfinite(g))),
      'grad_norm_median': med,
      'grad_norm_p99': float(np.percentile(fin, 99)) if fin.size else None,
      'grad_norm_max': float(fin.max()) if fin.size else None,
      'grad_exploding_env_frac': float(np.mean(gn > 1e3 * med))
                                 if fin.size else None,
      'done_frac': float(np.mean(np.asarray(done))),
  }
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--worker', default=None)
  ap.add_argument('--out',
                  default=str(_HERE / 'results' / 'g1_env_runs.jsonl'))
  ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
  ap.add_argument('--backends', nargs='*', default=[
      'unrolled+remat', 'scan+remat', 'ift', 'ift+remat', 'unrolled'],
                  help='backends; suffix +remat for per-step remat')
  ap.add_argument('--ckpt', action='store_true')
  args = ap.parse_args()
  if args.worker:
    print('RESULT ' + json.dumps(_worker(Config(**json.loads(args.worker)))),
          flush=True)
    return
  env = dict(os.environ)
  env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  for seed in args.seeds:
    for spec in args.backends:
      b, _, ck = spec.partition('+')
      cfg = Config(backend=b, seed=seed, ckpt=args.ckpt or ck == 'remat')
      t0 = time.time()
      p = subprocess.run([sys.executable, '-I', __file__, '--worker',
                          json.dumps(dataclasses.asdict(cfg))],
                         capture_output=True, text=True, env=env,
                         timeout=3600)
      line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
      if line:
        r = json.loads(line[-1][7:])
      else:
        r = {'config': dataclasses.asdict(cfg), 'error': p.stderr[-3000:]}
      r['wall_s'] = time.time() - t0
      with open(args.out, 'a') as fo:
        fo.write(json.dumps(r) + '\n')
      b = spec
      if 'error' in r:
        print(f'seed={seed} {b:9s} ERROR {r["error"].strip().splitlines()[-1][:200]}',
              flush=True)
      else:
        tb = 'OOM' if r['t_b_dir'] is None else f"{r['t_b_dir']:.1f}ms"
        print(f"seed={seed} {b:9s} f={r['t_f']:.1f}ms fb={r['t_fb']:.1f}ms "
              f"b={tb} mem={r['mem_fb'] / 2**20:.0f}MiB "
              f"alive={r['alive_env_frac']:.2f} "
              f"nonfinite(alive/done)={r['grad_nonfinite_alive_frac']}/"
              f"{r['grad_nonfinite_done_frac']} "
              f"explode(alive/done)={r['grad_exploding_alive_frac']}/"
              f"{r['grad_exploding_done_frac']} "
              f"|g|med={r['grad_norm_median']:.2e} max={r['grad_norm_max']:.2e} "
              f"done={r['done_frac']:.2f} compile_fb={r['compile_fb']:.0f}s",
              flush=True)


if __name__ == '__main__':
  main()
