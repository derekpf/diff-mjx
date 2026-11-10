"""Summarizes benchmark JSONL results into markdown tables and plots.

  python analyze.py --results results

Results recorded before the IFT options were fixed label variants
(ift_ref = faithful port, ift_ABDm = the now-fixed implementation); they are
still read.

Writes results/summary.md and results/plots/*.png. ``--opt`` names the
variant reported as "Optimized IFT".
"""

import argparse
import collections
import json
import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np

# Fixed slot per backend so color follows the entity across every chart
# (validated: scripts in the dataviz skill, light surface).
STYLE = {
    'scan': ('#2a78d6', 'o', 'Scan (unrolled)'),
    'default': ('#eb6834', 's', 'Default (masked scan)'),
    'ift_ref': ('#1baf7a', '^', 'Reference IFT'),
    'opt': ('#6250d6', 'D', 'Optimized IFT'),
}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load(results: pathlib.Path):
  rows = []
  for f in sorted(results.glob('*.jsonl')):
    if f.name.startswith('explore'):
      continue
    for line in f.read_text().splitlines():
      r = json.loads(line)
      r['sweep'] = f.stem
      rows.append(r)
  return rows


def key(c, drop=()):
  fields = ('scene', 'workload', 'iterations', 'ls_iterations', 'tolerance',
            'batch', 'horizon', 'substeps', 'precision', 'cone', 'solver')
  return tuple(c[f] for f in fields if f not in drop)


def rel(a, b):
  a, b = np.asarray(a), np.asarray(b)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def groups(rows):
  """config key (without backend) -> {backend[/ck]: row}."""
  g = collections.defaultdict(dict)
  for r in rows:
    if 'error' in r:
      continue
    c = r['config']
    name = c['backend'] + ('' if c['checkpoint'] == 'none' else
                           f"+ck_{c['checkpoint']}")
    g[(r['sweep'],) + key(c)][name] = r
  return g


def mib(r, field='mem_fb'):
  m = r.get(field) or {}
  return m.get('temp', 0) / 2**20


def fmt_row(label, backend, r, base):
  tf = r['t_f']['median']
  if 't_fb' not in r:
    return f'| {label} | {backend} | {tf:.2f} | – | – | {tf:.2f} | {mib(r, "mem_f"):.0f} | – |'
  tb = r['t_b_dir']['median']
  tfb = r['t_fb']['median']
  agree = ''
  if base is not None and 'grad0' in base and 'grad0' in r:
    agree = f'{rel(r["grad0"], base["grad0"]):.1e}'
  return (f'| {label} | {backend} | {tf:.2f} | {tb:.2f} | {tb / tf:.2f} | '
          f'{tfb:.2f} | {mib(r):.0f} | {agree} |')


def main_table(g, opt):
  lines = [
      '| Workload | Backend | Forward ms | Backward ms (direct) | Bwd/Fwd | '
      'Total ms (fused f+b) | XLA temp MiB | grad rel. diff vs Ref IFT |',
      '|---|---|---:|---:|---:|---:|---:|---:|',
  ]
  picks = [
      ('step', 'box_slide', 3, 'Contact scene (box, 1 step, B=512, K=3)'),
      ('step', 'box_slide', 10, 'Contact scene (box, 1 step, B=512, K=10)'),
      ('iters', 'go1', 3, 'Go1 rollout (H=8, B=512, K=3)'),
      ('iters', 'go1', 10, 'Go1 rollout (H=8, B=512, K=10)'),
      ('iters', 'g1', 3, 'G1 rollout (H=8, B=512, K=3)'),
      ('iters', 'g1', 10, 'G1 rollout (H=8, B=512, K=10)'),
  ]
  for sweep, scene, k, label in picks:
    cand = [v for kk, v in g.items()
            if kk[0] == sweep and kk[1] == scene and kk[3] == k]
    if not cand:
      continue
    v = cand[0]
    base = v.get('ift_ref')
    for b, name in (('while', 'While (forward only)'), ('scan', 'Scan'),
                    ('default', 'Default (masked scan)'),
                    ('ift_ref', 'Reference IFT'), (opt, 'Optimized IFT')):
      if b in v:
        lines.append(fmt_row(label, name, v[b], base))
  return '\n'.join(lines)


def variants_table(g):
  lines = ['| Workload | Variant | Forward ms | Backward ms (direct) | '
           'Total ms | Bwd/Fwd | Δ total vs ift_ref |',
           '|---|---|---:|---:|---:|---:|---:|']
  for kk, v in sorted(g.items()):
    if kk[0] != 'variants' or 'ift_ref' not in v:
      continue
    ref = v['ift_ref']['t_fb']['median']
    for b in ('ift_ref', 'ift_A', 'ift_AB', 'ift_AB_chol', 'ift_ABD',
              'ift_ABDm'):
      if b not in v:
        continue
      r = v[b]
      tf, tb, tfb = (r['t_f']['median'], r['t_b_dir']['median'],
                     r['t_fb']['median'])
      lines.append(f'| {kk[1]} K={kk[3]} | {b} | {tf:.2f} | {tb:.2f} | '
                   f'{tfb:.2f} | {tb / tf:.2f} | {100 * (tfb / ref - 1):+.0f}% |')
  return '\n'.join(lines)


def _style_ax(ax, xlabel, ylabel, title):
  ax.set_facecolor(SURF)
  ax.set_xlabel(xlabel, color=INK2)
  ax.set_ylabel(ylabel, color=INK2)
  ax.set_title(title, color=INK, loc='left', fontsize=11)
  ax.grid(True, color=GRID, linewidth=0.8)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)
  ax.tick_params(colors=INK2)


def line_plot(path, series, xlabel, ylabel, title, logx=False, logy=False):
  """series: {style_key: (xs, ys)}."""
  fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
  fig.patch.set_facecolor(SURF)
  for k, (xs, ys) in series.items():
    if not xs:
      continue
    color, marker, label = STYLE[k]
    ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2,
            label=label, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0),
                textcoords='offset points', va='center', fontsize=8,
                color=INK2)
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  _style_ax(ax, xlabel, ylabel, title)
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)


def collect(g, sweep, scene, xfield, metric, opt, extra=None):
  # key: (sweep, scene, workload, iterations, ls_iterations, tolerance, batch,
  #       horizon, substeps, precision, cone, solver)
  idx = {'iterations': 3, 'tolerance': 5, 'batch': 6, 'horizon': 7}[xfield]
  out = {k: ([], []) for k in STYLE}
  for kk in sorted(g, key=lambda t: t[idx]):
    if kk[0] != sweep or kk[1] != scene:
      continue
    if extra and not extra(kk):
      continue
    v = g[kk]
    for sk, b in (('scan', 'scan'), ('default', 'default'),
                  ('ift_ref', 'ift_ref'), ('opt', opt)):
      if b in v and 't_fb' in v[b]:
        out[sk][0].append(kk[idx])
        out[sk][1].append(metric(v[b], v))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results',
                  default=str(pathlib.Path(__file__).parent / 'results'))
  ap.add_argument('--opt', default='ift')
  args = ap.parse_args()
  res = pathlib.Path(args.results)
  plots = res / 'plots'
  plots.mkdir(exist_ok=True)
  rows = load(res)
  g = groups(rows)
  opt = args.opt

  ratio = lambda r, v: r['t_b_dir']['median'] / r['t_f']['median']
  # Backward relative to the fastest forward of the group: separates a cheap
  # backward from a slow forward.
  def ratio_common(r, v):
    tf = min(x['t_f']['median'] for x in v.values())
    return r['t_b_dir']['median'] / tf
  total = lambda r, v: r['t_fb']['median']
  mem = lambda r, v: mib(r)

  for scene in ('go1', 'g1'):
    line_plot(plots / f'ratio_vs_iters_{scene}.png',
              collect(g, 'iters', scene, 'iterations', ratio, opt),
              'solver iterations K', 'backward / forward (own forward)',
              f'{scene}: backward/forward ratio vs K (H=8, B=512)', logx=True)
    line_plot(plots / f'ratio_common_vs_iters_{scene}.png',
              collect(g, 'iters', scene, 'iterations', ratio_common, opt),
              'solver iterations K', 'backward / fastest forward',
              f'{scene}: backward vs common forward baseline', logx=True)
    line_plot(plots / f'total_vs_iters_{scene}.png',
              collect(g, 'iters', scene, 'iterations', total, opt),
              'solver iterations K', 'forward + backward (ms)',
              f'{scene}: total BPTT latency vs K (H=8, B=512)', logx=True,
              logy=True)
    line_plot(plots / f'memory_vs_iters_{scene}.png',
              collect(g, 'iters', scene, 'iterations', mem, opt),
              'solver iterations K', 'XLA temp buffer (MiB)',
              f'{scene}: peak gradient memory vs K (H=8, B=512)', logx=True,
              logy=True)
  line_plot(plots / 'ratio_vs_batch_go1.png',
            collect(g, 'batch', 'go1', 'batch', ratio, opt),
            'batch size', 'backward / forward (own forward)',
            'go1: backward/forward ratio vs batch (K=3, H=8)', logx=True)
  line_plot(plots / 'total_vs_batch_go1.png',
            collect(g, 'batch', 'go1', 'batch', total, opt),
            'batch size', 'forward + backward (ms)',
            'go1: total latency vs batch (K=3, H=8)', logx=True, logy=True)
  line_plot(plots / 'total_vs_horizon_go1.png',
            collect(g, 'horizon', 'go1', 'horizon', total, opt),
            'control steps H', 'forward + backward (ms)',
            'go1: total BPTT runtime vs horizon (K=3, B=256)')
  line_plot(plots / 'memory_vs_horizon_go1.png',
            collect(g, 'horizon', 'go1', 'horizon', mem, opt),
            'control steps H', 'XLA temp buffer (MiB)',
            'go1: gradient memory vs horizon (K=3, B=256)', logy=True)

  # Runtime vs gradient accuracy: tolerance sweep, error against the
  # tightest-tolerance optimized-IFT gradient.
  tol_ref = None
  for kk, v in g.items():
    if kk[0] == 'tol' and kk[5] == 0.0 and opt in v:
      tol_ref = v[opt]['grad0']
  if tol_ref is not None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor(SURF)
    for sk, b in (('scan', 'scan'), ('default', 'default'),
                  ('ift_ref', 'ift_ref'), ('opt', opt)):
      xs, ys, ts = [], [], []
      for kk in sorted(g, key=lambda t: -t[5]):
        if kk[0] != 'tol' or b not in g[kk]:
          continue
        r = g[kk][b]
        xs.append(r['t_fb']['median'])
        ys.append(max(rel(r['grad0'], tol_ref), 1e-9))
        ts.append(kk[5])
      if not xs:
        continue
      color, marker, label = STYLE[sk]
      ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=6,
              label=label, markeredgecolor=SURF, markeredgewidth=1.5)
      for x, y, t in zip(xs, ys, ts):
        ax.annotate(f'tol={t:g}', (x, y), xytext=(4, 4),
                    textcoords='offset points', fontsize=6, color=INK2)
    ax.set_yscale('log')
    _style_ax(ax, 'forward + backward (ms)',
              'grad rel. diff vs optimized IFT @ tol=0',
              'go1: runtime vs gradient accuracy (K=20, H=8, B=256)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(plots / 'runtime_vs_accuracy_go1.png')
    plt.close(fig)

  # Markdown summary.
  md = ['# Benchmark summary', '',
        f'Optimized IFT = `{opt}`. Times are medians (ms). "Backward ms" is '
        'the VJP timed directly on saved residuals; "Total" is the fused '
        '`value_and_grad`. Memory is the XLA buffer-assignment temp size of '
        'the fused gradient program.', '', '## Main table', '',
        main_table(g, opt), '', '## IFT variants (Optimizations A, B, D)', '',
        variants_table(g), '']
  errs = [r for r in rows if 'error' in r]
  md += ['## Failed configurations', '']
  for r in errs:
    c = r['config']
    md.append(f"- {r['sweep']}: {c['scene']} {c['backend']} K={c['iterations']}"
              f" B={c['batch']} H={c['horizon']} ck={c['checkpoint']}: "
              f"{r['error'].strip().splitlines()[-1][:160] if r['error'] else ''}")
  md += ['', '## All configurations', '',
         '| sweep | scene | wl | K | B | H | tol | prec | backend | f ms | '
         'b_dir ms | fb ms | R_eff | R_dir | MiB | compile fb s | niter | '
         'GPU util % |',
         '|' + '---|' * 18]
  for r in rows:
    if 'error' in r:
      continue
    c = r['config']
    name = c['backend'] + ('' if c['checkpoint'] == 'none' else
                           f"+ck_{c['checkpoint']}")
    fb = (r.get('t_fb') or {}).get('median', float('nan'))
    bd = (r.get('t_b_dir') or {}).get('median', float('nan'))
    md.append(
        f"| {r['sweep']} | {c['scene']} | {c['workload']} | {c['iterations']} "
        f"| {c['batch']} | {c['horizon']} | {c['tolerance']:g} | "
        f"{c['precision']} | {name} | {r['t_f']['median']:.2f} | {bd:.2f} | "
        f"{fb:.2f} | {r.get('R', float('nan')):.2f} | "
        f"{r.get('R_dir', float('nan')):.2f} | {mib(r):.0f} | "
        f"{r.get('compile_fb', float('nan')):.1f} | {r['niter_mean']:.2f} | "
        f"{r['gpu_before'].get('gpu_util_pct', '')} |")
  (res / 'summary.md').write_text('\n'.join(md) + '\n')
  print('wrote', res / 'summary.md', 'and', plots)


if __name__ == '__main__':
  main()
