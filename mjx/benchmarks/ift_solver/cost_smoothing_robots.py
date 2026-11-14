"""Constraint cost smoothing on robot rollouts: the numbers cited in the README.

Sections (default: all, in this order):

  jacobian       smoothed residual Jacobian at the hard solution: smallest
                 eigenvalue of its symmetric part relative to M's, asymmetry
                 (float32, rollout states).
  refinement     refinement from the hard solution (k = 2, 4; 4 line-search
                 trials; tolerance 1e-6 with the residual precision floor):
                 accepted fraction (float32, rollout states).
  reproducible   float64: forward solves that stopped at opt.iterations
                 unconverged, and whether forward- and reverse-mode programs
                 agree on q* and on the gradient (hard and smoothed).
  float32        float32 IFT Jacobian d qacc / d ctrl against a float64
                 reference at the same q* (hard and smoothed).
  timing         BPTT through H control steps x B envs: hard, smoothed, and
                 smoothed with refinement (k = 4).

Softnesses: stick/slip 0.2, liftoff 0.05, limit and friction loss 0.2.

Run with forlax's venv (needs mujoco_playground):
  python cost_smoothing_robots.py Go1JoystickFlatTerrain [section ...]
"""

import functools
import os
import subprocess
import sys
import tempfile
import time

if '--x64' in sys.argv:
  import jax  # pylint: disable=g-import-not-at-top

  jax.config.update('jax_enable_x64', True)

import jax  # pylint: disable=g-import-not-at-top,reimported
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import forward as mjx_forward
from mujoco.mjx._src import solver
from mujoco.mjx._src import support
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
import numpy as np

from forlax.env import DiffEnvWrapper

print = functools.partial(print, flush=True)  # pylint: disable=redefined-builtin

SECTIONS = ('jacobian', 'refinement', 'reproducible', 'float32', 'timing')
SMOOTH = dict(stickslip_softness=0.2, liftoff_softness=0.05,
              limit_softness=0.2, frictionloss_softness=0.2)
N_STATES = 64  # random-action states for the per-state checks
ROLLOUT_B, ROLLOUT_T, SAMPLE_EVERY = 32, 600, 60  # rollout states (substeps)
TIMING_B, TIMING_H, TIMING_REPS = 32, 16, 15


def _setup(name):
  env = registry.load(name)
  DiffEnvWrapper(env, softjax_mode=None, pw_solimp=None)
  hard = env.mjx_model
  return env, hard, hard.replace(opt=hard.opt.replace(**SMOOTH))


def _random_action_states(env, n):
  """States after 10 random-action control steps from n resets."""
  st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), n))
  step = jax.jit(jax.vmap(env.step))
  for t in range(10):
    st = step(st, 0.3 * jax.random.normal(jax.random.PRNGKey(t), (n, env.action_size)))
  return st.data


def _rollout_states(env, hard):
  """Hard solutions sampled along random-action rollouts."""
  st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), ROLLOUT_B))
  d = st.data
  step = jax.jit(jax.vmap(lambda dd, c: mjx.step(hard, dd.replace(ctrl=c))))
  fwd = jax.jit(jax.vmap(lambda dd: mjx.forward(hard, dd)))
  dpos, ascale = env._default_pose, env._config.action_scale  # pylint: disable=protected-access
  key, out = jax.random.PRNGKey(1), []
  for t in range(ROLLOUT_T):
    if t % env.n_substeps == 0:
      key, k = jax.random.split(key)
      ctrl = dpos + ascale * jp.clip(0.3 * jax.random.normal(k, (ROLLOUT_B, env.action_size)), -1, 1)
    d = step(d, ctrl)
    if t % SAMPLE_EVERY == SAMPLE_EVERY - 1:
      out.append(fwd(d.replace(ctrl=ctrl)))
  return out


def _q(a):
  a = np.asarray(a)
  return f'median {np.median(a):.2g} / 90% {np.percentile(a, 90):.2g} / max {a.max():.2g}'


def _rel(a, b):
  a, b = np.asarray(a).reshape(len(a), -1), np.asarray(b).reshape(len(b), -1)
  return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)


def section_jacobian_and_refinement(name, sections):
  env, hard, soft = _setup(name)
  states = _rollout_states(env, hard)

  def jacobian(dd):
    jac = jax.jacfwd(lambda q: solver._root(soft, dd, q))(dd.qacc)  # pylint: disable=protected-access
    sym = 0.5 * (jac + jac.T)
    lam_m = jp.linalg.eigvalsh(support.full_m(hard, dd))[0]
    return jp.linalg.eigvalsh(sym)[0] / lam_m, jp.max(jp.abs(jac - jac.T)) / jp.max(jp.abs(jac))

  def accepted(dd, iters):
    root = lambda q: solver._root(soft, dd, q)  # pylint: disable=protected-access
    newton = solver._smoothed_newton(soft, dd, solver._effective_softnesses(soft, dd))  # pylint: disable=protected-access
    ma = support.mul_m(hard, dd, dd.qacc)
    qfrc_c = dd._impl.efc_J.T @ dd._impl.efc_force  # pylint: disable=protected-access
    floor = 64 * jp.finfo(dd.qacc.dtype).eps * (
        jp.linalg.norm(ma) + jp.linalg.norm(dd.qfrc_smooth) + jp.linalg.norm(qfrc_c))
    _, ok = solver._refine_smoothed_root(root, dd.qacc, iters, 1e-6, floor, 4, True, newton)  # pylint: disable=protected-access
    return ok

  n = sum(int(s.qacc.shape[0]) for s in states)
  if 'jacobian' in sections:
    stats = [jax.jit(jax.vmap(jacobian))(s) for s in states]
    eig = np.concatenate([np.asarray(e) for e, _ in stats])
    asym = np.concatenate([np.asarray(a) for _, a in stats])
    print(f'  jacobian ({n} rollout states): lambda_min(sym) / lambda_min(M) min {eig.min():.3g}, '
          f'median {np.median(eig):.3g}; max relative asymmetry {asym.max():.2g}')
  if 'refinement' in sections:
    for iters in (2, 4):
      ok = np.concatenate([np.asarray(jax.jit(jax.vmap(lambda dd: accepted(dd, iters)))(s)) for s in states])
      print(f'  refinement k={iters} ({n} rollout states): accepted {ok.mean():.1%}')


def section_reproducible(name):
  """float64; forward- and reverse-mode programs of the same function."""
  env, hard, soft = _setup(name)
  d = _random_action_states(env, N_STATES)

  @jax.jit
  @jax.vmap
  def residual(dd):
    x = mjx.forward(hard, dd)
    r = solver._root(hard, x, x.qacc)  # pylint: disable=protected-access
    scale = jp.linalg.norm(x.qfrc_smooth) + jp.linalg.norm(support.mul_m(hard, x, x.qacc))
    return jp.linalg.norm(r) / jp.maximum(scale, 1e-30)

  unconverged = np.asarray(residual(d)) > 1e-10
  print(f'  forward solves unconverged at opt.iterations={hard.opt.iterations}: '
        f'{int(unconverged.sum())} of {N_STATES}')
  for label, model in (('hard', hard), ('smoothed', soft)):
    f = lambda dd, c, model=model: mjx.forward(model, dd.replace(ctrl=c)).qacc

    def fwd(dd, c, f=f):
      q, lin = jax.linearize(lambda cc: f(dd, cc), c)
      return q, jax.vmap(lin, out_axes=1)(jp.eye(c.shape[0], dtype=c.dtype))

    def rev(dd, c, f=f):
      q, pull = jax.vjp(lambda cc: f(dd, cc), c)
      return q, pull(2 * q)[0]

    qf, j = jax.jit(jax.vmap(fwd))(d, d.ctrl)
    qr, v = jax.jit(jax.vmap(rev))(d, d.ctrl)
    q_diff = _rel(qf, qr)
    g_diff = _rel(v, np.einsum('bv,bvu->bu', 2 * np.asarray(qr), np.asarray(j)))
    same = q_diff < 1e-12
    line = (f'  {label}: q* identical in both programs in {int(same.sum())} states '
            f'(gradients agree to {g_diff[same].max():.1e})')
    if (~same).any():
      line += (f'; differs in {int((~same).sum())} (all unconverged: '
               f'{bool(np.all(unconverged[~same]))}; q* up to {q_diff[~same].max():.1e} '
               f'apart, gradients up to {g_diff[~same].max():.1e})')
    print(line, flush=True)


def section_float32(name, stage=None, path=None):
  """Stage 'a' (float32) saves states, q* and Jacobians; stage 'b' (float64)
  computes the reference -(dr/dq)^-1 dr/dctrl at the same q*."""
  if stage is None:
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'float32.npz')
      for st, extra in (('a', []), ('b', ['--x64'])):
        subprocess.run([sys.executable, __file__, name, '--float32-stage', st, path] + extra, check=True)
    return
  env, hard, soft = _setup(name)
  models = {'hard': hard, 'smoothed': soft}
  if stage == 'a':
    d = _random_action_states(env, N_STATES)
    save = {f'leaf{i}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(d))}
    for label, model in models.items():
      def fwd(dd, c, model=model):
        q, lin = jax.linearize(lambda cc: mjx.forward(model, dd.replace(ctrl=cc)).qacc, c)
        return q, jax.vmap(lin, out_axes=1)(jp.eye(c.shape[0], dtype=c.dtype))
      q, j = jax.jit(jax.vmap(fwd))(d, d.ctrl)
      save[f'{label}_q'], save[f'{label}_j'] = np.asarray(q), np.asarray(j)
    np.savez(path, **save)
    return
  z = np.load(path)
  tmpl = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), N_STATES)).data
  leaves, tdef = jax.tree.flatten(tmpl)
  d = jax.tree.unflatten(tdef, [jp.asarray(z[f'leaf{i}']).astype(l.dtype) for i, l in enumerate(leaves)])

  def pre(model, dd, c):
    dd = mjx_forward.fwd_position(model, dd.replace(ctrl=c))
    dd = mjx_forward.fwd_velocity(model, dd)
    dd = mjx_forward.fwd_actuation(model, dd)
    return mjx_forward.fwd_acceleration(model, dd)

  for label, model in models.items():
    @jax.jit
    @jax.vmap
    def reference(dd, q, model=model):
      d1 = pre(model, dd, dd.ctrl)
      h = jax.jacfwd(lambda qq: solver._root(model, d1, qq))(q)  # pylint: disable=protected-access
      b = jax.jacfwd(lambda cc: solver._root(model, pre(model, dd, cc), q))(dd.ctrl)  # pylint: disable=protected-access
      return -jp.linalg.solve(h, b), jp.linalg.cond(h)

    ref, cond = reference(d, jp.asarray(z[f'{label}_q']).astype(jp.float64))
    err = _rel(z[f'{label}_j'].astype(np.float64), ref)
    print(f'  {label}: float32 error vs float64 at the same q*: {_q(err)}; '
          f'cond(dr/dq) median {np.median(cond):.1e}')


def section_timing(name):
  env, hard, soft = _setup(name)
  dpos, ascale = env._default_pose, env._config.action_scale  # pylint: disable=protected-access
  d0 = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), TIMING_B)).data
  acts = 0.3 * jax.random.normal(jax.random.PRNGKey(2), (TIMING_H, TIMING_B, env.action_size))
  refined = soft.replace(opt=soft.opt.replace(refinement_iterations=4, refinement_ls_iterations=4))
  for label, model in (('hard', hard), ('smoothed', soft), ('smoothed, refinement k=4', refined)):
    @jax.jit
    def grad(a, dd, model=model):
      def rollout(a):
        def body(x, aa):
          x = jax.vmap(lambda y, c: mjx_env.step(model, y, c, env.n_substeps))(x, dpos + ascale * aa)
          return x, x.qvel
        return jp.sum(jax.lax.scan(body, dd, a)[1] ** 2)
      return jax.grad(rollout)(a)

    grad(acts, d0).block_until_ready()
    ts = []
    for _ in range(TIMING_REPS):
      t0 = time.perf_counter()
      grad(acts, d0).block_until_ready()
      ts.append(time.perf_counter() - t0)
    ts = 1e3 * np.asarray(ts)
    print(f'  BPTT {TIMING_H} steps x {TIMING_B} envs, {label:26s} median {np.median(ts):7.1f} ms '
          f'(IQR {np.percentile(ts, 75) - np.percentile(ts, 25):.1f} ms)')


def main():
  args = [a for a in sys.argv[1:] if a != '--x64']
  name = args[0] if args else 'Go1JoystickFlatTerrain'
  if '--float32-stage' in args:
    i = args.index('--float32-stage')
    section_float32(name, args[i + 1], args[i + 2])
    return
  sections = [a for a in args[1:] if a in SECTIONS] or list(SECTIONS)
  print(f'== {name} ({jax.devices()[0].device_kind})', flush=True)
  if 'jacobian' in sections or 'refinement' in sections:
    section_jacobian_and_refinement(name, sections)
  if 'reproducible' in sections:
    subprocess.run([sys.executable, __file__, name, '--reproducible-x64', '--x64'], check=True)
  if 'float32' in sections:
    section_float32(name)
  if 'timing' in sections:
    section_timing(name)


if __name__ == '__main__':
  if '--reproducible-x64' in sys.argv:
    section_reproducible([a for a in sys.argv[1:] if not a.startswith('--')][0])
  else:
    main()
