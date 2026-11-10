# Implicit vs unrolled differentiation of the MJX constraint solver

Benchmarks and validation for the `MJX_GRAD_BACKEND` solver backends
(`default`, `while`, `scan`, `ift`). The backends are documented in the
repository README ("Constraint-solver derivative backends").

## Results

`results/` holds the measurements cited in the repository README. All GPU
numbers were taken in float32 on an RTX 5080 shared with another job; the
validation ran in float64 on CPU.

| file | contents | produced by |
|---|---|---|
| `g1_env.json` | G1 joystick env BPTT: `default`/`ift`, with and without remat, plus tangent-solve headroom runs | `g1_env_bench.py` (headroom runs: same harness with the solver patched at runtime) |
| `variants.jsonl` | IFT implementation variants on G1/Go1 rollouts (forward-loop and tangent-solve speedups) | `bench.py` (the `variants` sweep, since removed) |
| `tangent_go1.json`, `tangent_g1.json` | tangent-solve components: matrix build, QR/LU/Cholesky cost and float32 accuracy | `profile_tangent.py` |
| `validation_matrix.json`, `validation_rollout.json` | float64 gradients vs central finite differences, per scene/cone/solver, single step and rollouts | `validate.py` |

These files were recorded while the IFT implementation was still
configurable, so some use older variant names: `ift_ref` is the faithful
port of calumarnott/mujoco (commit e453381c), and `ift_ABDm` is the current
`ift`. The float64 validation ran on the faithful port; the current `ift`
matches it to 1e-14 relative or better, and `solver_grad_test.py` repeats the
finite-difference checks on the current code.

## Selecting a backend

```bash
export MJX_GRAD_BACKEND=ift        # default | while | scan | ift
```

The variable is read when a function is **traced**; already-compiled
functions keep their backend. Trace inside the setting (e.g. compile with
`jax.jit(f).lower(...).compile()` while the variable is set).

| backend | forward loop | reverse mode | notes |
|---|---|---|---|
| unset / `default` | masked scan (`opt.scan_loop`) or while | unrolled | unchanged diff-mjx behaviour |
| `while` | `lax.while_loop` | only `iterations == 1` | upstream MJX |
| `scan` | fixed `opt.iterations` steps | unrolled | reference fork's `scan` |
| `ift` | masked scan (not differentiated) | implicit (`lax.custom_root`) | gradient of the converged fixed point |

## Running

Environment used for the reported numbers: an isolated venv with
`jax[cuda12]==0.10.2`, `mujoco==3.10.0`, `softjax`, `matplotlib`, and this
fork installed editable (`pip install -e mjx`). The Go1/G1 scenes need
`MJX_PLAYGROUND_PATH` pointing at the `mujoco_playground` package directory.

```bash
cd mjx/benchmarks/ift_solver
export MJX_PLAYGROUND_PATH=/path/to/mujoco_playground

# correctness (float64, CPU)
JAX_PLATFORMS=cpu python validate.py --study matrix
JAX_PLATFORMS=cpu python validate.py --study iterations
JAX_PLATFORMS=cpu python validate.py --study rollout
JAX_PLATFORMS=cpu python -m pytest ../../mujoco/mjx/_src/solver_grad_test.py

# performance (GPU); each configuration runs in its own subprocess
./run_all.sh                       # all sweeps (~200 configs, ~3-4 h on an RTX 5080)
python bench.py --sweep iters --out results/iters.jsonl   # one sweep
python bench.py --config '{"scene": "go1", "backend": "ift", "iterations": 3}'
python profile_tangent.py --scene g1 --batch 512 --out results/tangent_g1.json

# tables and plots
python analyze.py                  # -> results/summary.md, results/plots/
python g1_env_bench.py             # G1 joystick env BPTT, per backend
                                   # (needs mujoco_playground installed)
```

## Files

| file | purpose |
|---|---|
| `scenes.py` | analytic MJCF scenes, toy quadruped, Go1/G1 loaders |
| `gradcheck.py` | AD vs central-FD checks, tangent-space qpos perturbations, fixed-point diagnostics |
| `validate.py` | correctness studies -> `results/validation_*.json` |
| `bench.py` | timing/memory harness and sweeps -> `results/<sweep>.jsonl` |
| `profile_tangent.py` | tangent-solve component microbenchmark |
| `analyze.py` | summary tables and plots |
| `g1_env_bench.py` | BPTT through the diff-playground G1 joystick env (forlax SHAC config) |
