# SoftJAX surrogate flag guidance

Softness and ST settings belong to the user. Reward and event-history calls
honor `reward_st_enable`; sensor and physics ST settings remain independent.
Calibrate other optional flags against the backward quantity consumed by each
call site, checking direction and monotonicity across user settings. With ST
enabled, the forward must match the nominal operation. Composite gradients can
still change with ST because downstream operations see different input values.

- **Keep softness tied to physical units with `standardize=False`.** SoftJAX's
  default standardization normalizes and squashes values before computing soft
  selection weights, so `softness` no longer directly tracks a distance, height,
  or force scale. Leave standardization enabled when scale-normalized selection
  is intentional, and tune softness in that normalized domain.

- **Choose the selection-gradient rule to match the max contract.** For
  SoftJAX's selection-based `max`, `gated_grad=True` differentiates through the
  soft index and gives the exact derivative of the relaxed weighted-value
  output. `False` stops that path, leaving soft-index weights as the gradient.
  Those weights form a convex combination of hard-max subgradients. Choose
  `False` when that monotone max-like backward is the intended surrogate; choose
  `True` when the desired gradient is the exact derivative of the relaxed
  weighted-value output. Keep the choice fixed while comparing ST modes: ST
  controls the forward value, not the selection-gradient rule.

- **Check the complete backward path.** Probe ties, thresholds, and realistic
  input gaps. Measure both the operator Jacobian and the contact or sensor
  Jacobian that consumes it. A nonzero local gradient alone is insufficient.
  Verify nominal forward parity separately when ST is enabled.

For example, Go1's swing-peak maximum uses height in meters, so
`standardize=False` keeps softness on the native scale. At a height 1 cm below
the peak, `standardize=False` alone gave gradient `0.401`; also disabling
`gated_grad` gave `0.450`, so that point alone does not justify the latter. The
max contract gives a separate reason: over 201 ST-mode candidate gaps from
`-0.10` to `+0.10` m at softness `0.05`, `gated_grad=True` produced gradient
components outside `[0, 1]` at 74 points (range `[-0.0908, 1.0908]`), while
`False` stayed in `[0.1192, 0.8808]`; both sums were one. Since increasing either
candidate must not reduce a maximum, Go1 uses `False` for its swing-peak
backward. The gradient choice has the same monotonicity in both ST modes. This
local check does not determine the best gradient width or establish that the
resulting touchdown reward gradient is useful.

Convex support extrema use `standardize=False` so their smoothing bands remain
in the support coordinates (relative to box scale in SAT). They use
`gated_grad=False` because increasing a maximum candidate, or decreasing a
minimum candidate, should not move the selected support in the opposite
direction. Contact sensor `top_k` is different: both `mindist` and `maxforce`
retain rank gradients for payload fields that change when the winning contact
changes. The reported distance for `mindist` and force vector for `maxforce`
instead use direct selection weights. This avoids negative candidate
derivatives in their score-aligned quantities while allowing other fields to
respond to rank changes. The choice is independent of ST. Both use contact
scores with `standardize=False`; their gradient widths must be assessed in
score units.

For the plane support maximum, two candidates separated by `1e-6` at
`softness=1e-6` in smooth mode gave the lower candidate gradient `0` with
standardization and `0.269` without it. In C2 mode, that same gap is already
outside the active smoothing region and its gradient remains `0`; the mode's
compact support must be considered when choosing a physical smoothing width.
With `gated_grad=False`, sampled smooth and C2 support gradients stayed in
`[0, 1]` and summed to one. These are intentional surrogate gradients, so they
need not equal finite differences of a relaxed weighted-value forward.
For a two-contact `mindist` selector with one distance and one unrelated
payload field, the distance candidate gradients were `[0.881, 0.119]` while
the other field retained score sensitivity `[21.0, -21.0]` per distance unit.
With rank gradients stopped for all fields, that other field had zero score
sensitivity. For a one-dimensional `maxforce` example with forces 0.5 and
0.7 and the actual squared-force score, direct force gradients were
`[0.083, 0.917]`; differentiating the rank gave `[-0.069, 1.130]`. A second
payload field retained rank sensitivity `[-15.25, 21.35]` per force unit.
These are local selector checks, not end-to-end sensor or policy-gradient
validation.

Other optional flags were checked by their backward contract. The default
`gated=False` for ReLU and clip yields derivatives in `[0, 1]` near a boundary;
gated ReLU at `x=-2*softness` has derivative `-0.091`. Product-based `any` and
`all` retain unit sensitivity to a decisive input at an all-off or all-on
boundary, whereas the geometric-mean option divides it by the number of
inputs. These defaults stay unchanged.

## diff-mjx changes

- **Per-operation ST wiring.** Propagates `opt.contact_st_enable` (default `False`) through softened collision and contact-physics operations, preserving nominal hard forward values while using surrogate gradients.

- **Closer nominal MJX behavior.** Corrects collision feature selection, tie-breaking, normalization, contact masks, duplicate contacts, and sign conventions.

- **Plane–cylinder degenerate-vector gate.** Uses softness `1e-8`.

- **Capsule–capsule gate.** Uses softness `4e-8`.

- **Box–box SAT softness.** Uses `7e-5` for SAT and `2e-5` for other operations.

- **Plane–convex selection is hard.** Plane–box and plane–mesh contacts always select their contact vertices with the nominal hard rule, in every `softjax_mode`, so distance and position gradients come only from the selected vertices' geometry. The softened selection (at softness `1e-6`) produced distance gradients up to ~1e5 for separated vertices near selection boundaries, which flat contact faces sit on. With `pw_solimp` these contacts enter the dynamics, and BPTT through G1 and Berkeley Humanoid diverged. Over 2M PatSappo steps, hard selection learned faster on both.

- **Soft validity gradients retained.** Removes forced gradient stops around box–box candidate validity, manifold masks, and contact counts.

- **Numerical safeguards.** Adds finite-gradient protection for C2 clipping, coincident contacts, cylinder SDF derivatives, tendon wrapping, and muscle smoothing while retaining nominal forwards.

- **Dynamics consistency.** Shares straight-through handling between forward and inverse piecewise-solimp calculations; limits contact-specific clamping to contact constraints.

- **Independent sensor options.** Adds MJX options `sensor_softness=0.0` and `sensor_st_enable=False`, loaded during model conversion. They are independent of reward and physics-contact settings.

- **Sensor forward/backward behavior.** Zero softness preserves nominal behavior. Positive softness enables relaxed forward/backward evaluation; enabling sensor ST preserves nominal forward values with surrogate gradients. Relaxations use `softjax_mode`, falling back to `"smooth"`.

- **Differentiable contact sensors.** Softens contact eligibility, counts, requested-slot selection, and payload masks, including force, torque, distance, position, normal, and tangent. ST wraps the complete output so inactive masks retain backward paths.

- **Differentiable touch sensors.** Adds `ray_geom_hit(...)` for softened primitive intersection validity and candidate aggregation, propagating gradients through contact eligibility and force accumulation. Existing ray/rangefinder intersection behavior is unchanged.

- **Differentiable sensor cutoffs.** Extends `_apply_cutoff(...)` with softness, ST, and mode arguments; clipping now optionally uses softjax.

- **Shared sensor drop-ins.** Extends the shared softjax exports with the ST wrapper, selection, comparison, and safe-square-root operations required by these paths.

- **Realised solver iterations.** Every solver backend writes the iteration count it actually ran to `Data._impl.solver_niter`.

- **Active-set smoothing.** Adds `opt.contact_activeset_softness`, `opt.limit_activeset_softness`, and `opt.frictionloss_activeset_softness`, which smooth linear (per-row) active-set switches in the `ift` backward pass only. See [Active-set smoothing](#active-set-smoothing-ift-backward).

## Constraint-solver derivative backends

The JAX constraint solver can be differentiated in several ways, selected by the `MJX_GRAD_BACKEND` environment variable. The forward physics (MJX's convex Gauss-principle solve, Newton or CG, pyramidal or elliptic cones) is the same for every backend; only the gradient differs. Ported from [calumarnott/mujoco](https://github.com/calumarnott/mujoco) (`3.9.0-derivatives`, Layeghi et al., [arXiv:2607.24959](https://arxiv.org/abs/2607.24959)) and adapted to this fork.

| `MJX_GRAD_BACKEND` | forward loop | gradient |
|---|---|---|
| unset / `ift` | masked scan, never differentiated | implicit function theorem on the stationarity residual |
| `unrolled` | masked scan (`opt.scan_loop=True`, freezes after convergence) or `while_loop` | unrolled through the iterations; the diff-mjx behavior before `ift` became the default |
| `while` | `lax.while_loop` | reverse mode only for `opt.iterations == 1` (upstream MJX) |
| `scan` | exactly `opt.iterations` steps, no convergence check | unrolled through every step |

`ift` is the default. Set `MJX_GRAD_BACKEND=unrolled` to restore the previous unrolled gradient. Results recorded before this change call the unrolled backend `default`.

The variable is read when a function is **traced**, so set it before `jax.jit` traces (or compiles) the function; already-compiled functions keep their backend.

**`ift` implementation.** At the solution, `F(a) = M a - qfrc_smooth - qfrc_constraint(a) = 0`. `lax.custom_root` runs the existing solver (`_root_iterative_solve`) to obtain `a*` without recording it for differentiation, then for the backward pass solves the adjoint system `(dF/da)^T λ = dL/da` and returns `-λ^T dF/dθ`, where `θ` is everything `F` depends on (state, controls, model parameters, constraint Jacobians and impedances, including any softjax/ST surrogates computed before the solve).

- **Residual (`_root`).** Evaluates `F` directly from the constraint context. The reference built the full solver gradient update, including an unused Newton search-direction solve; XLA already eliminated that, so this change is for clarity only.
- **Forward loop.** A masked scan instead of `lax.while_loop`: identical solution and iteration counts, but faster on GPU, where the while loop syncs to the host every iteration. Measured on batched 8-step rollouts (512 envs, shared RTX 5080, single runs), forward time dropped about 1.5x on G1 (19.5 to 12.8 ms at 3 iterations) and 1.6 to 1.9x on Go1 (3 to 10 iterations). Together with the LU tangent solve, total forward+backward time fell about 1.3x on G1 and 1.45x (3 iterations) to 2.3x (10 iterations) on Go1, relative to the reference port.
- **Tangent solve (`_root_tangent_solve`).** Forms the `nv x nv` residual Jacobian by applying the linear map `custom_root` provides to the identity columns, then solves it with a dense LU factorization. The reference used `jax.jacobian` plus a QR factorization followed by a general solve on `R`; LU is as accurate in float32 and about 2.7x faster at `nv = 35`. `custom_root` transposes this solve in its right-hand side, so it must be exactly linear in it: iterative (CG-type) adjoint solves are not possible here and would need a `custom_vjp`.
- **Outputs.** `qfrc_constraint` and `efc_force` are recomputed from `a*` with ordinary autodiff, so their gradients include both the implicit path through `a*` and the explicit dependence on `θ`. The warmstart only seeds the forward solve and receives no gradient.

**What the gradient means.** `ift` differentiates the *converged* fixed point. When the forward solve stops short (small `opt.iterations`, loose `opt.tolerance`), its value is the truncated iterate but its gradient is that of the exact solution. Unrolled backends instead differentiate the iterations they ran. Measured in float64 against central finite differences (`mjx/benchmarks/ift_solver`):

- **Converged solves.** `ift` matched finite differences (relative error 1e-9 to 1e-16) for every tested scene, cone, solver, and warmstart setting.
- **Unrolled gradients drop or distort the solver sensitivity when:**
  - the warmstart is already the solution (static or standing contact): the line search reports no improvement, and the masked update carries no derivative;
  - the cone is elliptic: the last improving Newton step used the Hessian of a different cone zone (about 6% error, independent of the iteration count);
  - the solver is CG: errors of 100% or more.
- **Non-finite gradients from other sources are not affected.** In the G1 joystick environment, non-finite gradients occurred at similar rates under `unrolled` and `ift`, so they originate outside the constraint-solver differentiation.

**Cost.** The `ift` backward does one `nv x nv` matrix build, one LU solve, and one residual VJP per solve, independent of `opt.iterations`, and it stores no per-iteration residuals. Unrolled backward time and memory grow with the iteration count.

Measured on BPTT through the diff-playground `G1JoystickFlatTerrain` environment, with forlax's G1 SHAC settings (256 envs, 16-step horizon, `pw_solimp` with `contact_st_enable`, 3 solver iterations, float32, one seed, RTX 5080):

| backend | forward | backward | forward + backward | memory |
|---|---:|---:|---:|---:|
| `unrolled`, no remat | – | – | out of memory (needs ~19 GiB) | – |
| `unrolled` + per-step remat | 355 ms | 1583 ms | 1767 ms | 1.33 GB |
| `ift` + per-step remat | 355 ms | 1071 ms | 1252 ms (1.41x faster) | 0.38 GB |
| `ift`, no remat | 356 ms | 359 ms | 1027 ms (1.72x faster) | 5.76 GB |

Remat is `jax.remat(env.step, prevent_cse=False)`, as in forlax's `remat_enable`. The tangent solve (matrix build plus LU) is now about 3% of the `ift` backward; the rest is differentiating the environment itself. In this environment, about 45% of envs produced non-finite gradients over 16 steps under both `unrolled` and `ift`.

Benchmarks, validation scripts, raw results, and reproduction commands are in [`mjx/benchmarks/ift_solver`](mjx/benchmarks/ift_solver/README.md).

**Tests.** `mjx/mujoco/mjx/_src/solver_test.py` (backend selection and forward equivalence against MuJoCo C) and `mjx/mujoco/mjx/_src/solver_grad_test.py` (gradients against unrolled differentiation and finite differences).

## Active-set smoothing (IFT backward)

**What it smooths.** At the solution, each inequality row's force is a ReLU of its residual `Jaref = J qacc - aref`: `f_i = D_i * max(0, -Jaref_i)`. The `ift` backward differentiates the stationarity residual at the converged `qacc`, so each row adds `m_i * D_i * J_i^T J_i` to the residual Jacobian, where `m_i = [Jaref_i < 0]` is the row's active-set mask. The force is continuous where a row switches on or off, but the gradient jumps by that row's rank-1 term. These switches are liftoff, stick/slip (pyramidal edges switching), limit release, and friction-loss stick/slip.

**Method.** In the residual used only by the `ift` backward (`solver._root`), the ReLU of the selected rows becomes

```
f_i = D_i * sj.relu(-Jaref_i, softness, mode='smooth', gated=False)
    = D_i * softness * softplus(-Jaref_i / softness)
```

Its slope is `D_i * sigmoid(-Jaref_i / softness)`. Near a switch, the gradient blends the Jacobians on either side of it instead of jumping between them. The slope stays in `(0, D_i)`, so the residual Jacobian stays positive definite. The gated form, `x * sigmoid(x)`, has negative slopes and can make it indefinite, so it is not used. The forward solve, line search, and reported outputs (`qacc`, `efc_force`, `qfrc_constraint`) keep the hard mask. Forward values are therefore unchanged. The forward solve finds the root of the hard residual, not the smoothed one, so the result is a surrogate gradient, not the derivative of any model.

| option | rows | switch |
|---|---|---|
| `contact_activeset_softness` | nominal contact rows: frictionless and pyramidal-edge rows in contact (`efc_pos < efc_margin`) | liftoff, stick/slip |
| `limit_activeset_softness` | engaged joint and tendon limit rows (`efc_pos < efc_margin`) | limit release |
| `frictionloss_activeset_softness` | DOF and tendon friction-loss rows | friction-loss stick/slip: the hard clip of `x = -Jaref` to `[-b, b]`, `b = r * frictionloss`, becomes `x - sj.relu(x - b) + sj.relu(-x - b)` |

**Not supported.**

- **Cone smoothing.** Only linear (per-row) switches are smoothed. Pyramidal contacts are linear rows, so `cone="pyramidal"` is covered. Elliptic cones switch zones (stick, slip, separate) through a coupled per-contact projection onto a second-order cone. That needs a different formulation, with care where the tangential magnitude is zero. Setting `contact_activeset_softness > 0` with `cone="elliptic"` raises `NotImplementedError`. Every registered diff-playground environment uses the pyramidal cone.
- **Limit touchdown.** Limit engagement is not smoothed. A row only exists once a joint or tendon reaches its limit margin, and it switches on at full impedance with `aref = -b * v`, so an approaching limit still gets a force jump. `limit_activeset_softness` covers only the release of an engaged limit. pw_solimp anticipates contact touchdown only and does not apply to limits.
- **Contact touchdown is left to pw_solimp.** This is not handled by active-set smoothing. pw_solimp rows (`efc_pos >= efc_margin`) and zeroed padding rows keep the hard mask, so the two mechanisms do not overlap. Under `contact_st_enable`, gradients come from the pw_solimp branch. Smoothing applies to that branch's nominal rows, whose definition is identical to the nominal branch's.
- **Other backends.** Only `ift` reads these options. Setting any of them above 0 with `MJX_GRAD_BACKEND` set to `unrolled`, `scan`, or `while` raises an error instead of being ignored.

**Usage.**

- **Static values.** Each option is a static Python float, so changing it retraces the compiled step. A `jax.Array` raises `TypeError`, because it would be traced. Negative or non-finite values raise `ValueError`.
- **Zero is exactly hard.** At `0` (the default), the smoothing branch is not traced, so behavior is identical to the hard mask. A small positive softness is not the same: at `Jaref = 0` the slope is 0.5, not 0.
- **Units.** Softness is in `Jaref` units, the acceleration along the row.
- **forlax.** `DiffEnvWrapper` accepts the same keyword arguments and validates them at construction.

**Gradient refinement (`opt.refinement_iterations`).** By default (`0`), the smoothed slopes are evaluated at the hard solution `q*`, which is not a root of the smoothed residual. The result is a surrogate that is not the derivative of any model, and its error can compound over a rollout. With `refinement_iterations = k > 0`, the `ift` backward first takes `k` line-searched Newton iterations on the smoothed residual, starting from `q*`. It then evaluates the gradient at that point, the smoothed root, so it becomes the exact derivative of the smoothed model at its own solution. DiffContact calls the analogous point the backward evaluation point. Forward values stay `q*` through straight-through (`stop_gradient(q*) + (q_eval - stop_gradient(q_eval))`), so states, contacts, and outputs are unchanged. Refinement does not change the active set or the forward solve. It only moves the point where the gradient is evaluated.

- **When it runs.** It is skipped (not traced) when every softness is 0, because the smoothed root is then `q*`. It shares the residual of all three softness options.
- **Line search (`opt.refinement_ls_iterations`, default `0`, mirroring `opt.ls_iterations`).** At the default of `0`, refinement takes full Newton steps (plain Newton, with no line-search cost). With `refinement_ls_iterations = n > 0`, each Newton direction is followed by `n` trials of a line search on the smoothed cost (`_smoothed_linesearch`). It uses the same idea as the forward `_linesearch`, a bracketed Newton iteration on the directional derivative. It evaluates the smoothed cost's derivatives (one JVP per trial), whereas the forward search uses the hard cost's piecewise-quadratic coefficients. It has no tolerance counterpart to `ls_tolerance`, because it always runs all `n` trials. It is a separate function so the forward solve stays untouched. Plain Newton on the smoothed residual tends to undershoot rather than overshoot: rows that switch off along the step lower the curvature, so the minimizer lies beyond the full step. The search therefore expands (`alpha <- max(Newton, 2 alpha)`) until the derivative changes sign. On the stiff-friction box (`n = 8`), this cut the iterations to round-off from about 12 to about 6 for stick/slip, and from 12 to 2 for liftoff. Whether that saves wall-clock time, given the added JVPs, has not been measured, so the line search is off by default.
- **Fallback (`opt.refinement_fallback_enable`, default `True`; `opt.refinement_tolerance`, default `1e-6`).** A partially converged point can give a gradient worse than the surrogate. It is neither the hard state nor the smoothed model's solution, and partial convergence still reduces the residual, so a reduce-only check would not catch it. The refined point is used only if it is finite and `|r(q_eval)| <= max(tolerance * |r(q*)|, floor)`, where `floor = 64 * eps * (|M qacc| + |qfrc_smooth| + |qfrc_constraint|)` is the residual's precision floor. Otherwise `q*` is used, which gives exactly the surrogate. The gradient is therefore always either the surrogate or near-converged, never in between. The decision is per solve, so per environment under `vmap`. Setting `refinement_fallback_enable = False` skips the tolerance check, so the refined point is used even when partially converged. This is for experiments, and it can give in-between gradients. Non-finite points still fall back, because they would only give NaN gradients. The toggle follows the fork's `*_enable` naming for boolean options (`contact_st_enable`, `sensor_st_enable`). MuJoCo's own boolean options are bits in `opt.disableflags` and `opt.enableflags`, which cannot be extended without changing MuJoCo C.
- **Validation.** `refinement_iterations` and `refinement_ls_iterations` must be static Python `int >= 0`, `refinement_tolerance` a static finite Python float `>= 0`, and `refinement_fallback_enable` a Python `bool`. Wrong types (including bools for the counts and `jax.Array` for any option) raise `TypeError`, and out-of-range values raise `ValueError`.
- **Cost.** Each iteration builds the `nv x nv` smoothed-residual Jacobian (`nv` forward-mode residual evaluations), does one dense solve, and runs `refinement_ls_iterations` line-search JVPs.
  - **6-DOF box** (100-step rollout, CPU, float64, before the line search): 7.7 ms hard, 8.95 ms surrogate, and 9.5, 9.9, and 10.6 ms for refinement at `k = 2`, `4`, and `8`.
  - **Go1** (`Go1JoystickFlatTerrain`, `nv = 18`, 5 solver iterations, 256 envs, 16-step horizon, RTX 5080, float32, before the line search): 147 ms hard, 149 ms surrogate, and about +5.6 ms per refinement iteration (+9% at `k = 2`, +31% at `k = 8`).
- **Choosing `k`.** Iterations to round-off on the box, for softness 0.1 to 0.4, without the line search (`refinement_ls_iterations = 0`) and with it (`= 8`):
  - **Default contact (`impratio = 1`):** stick/slip 4 without and 3-4 with; liftoff about 6 without and 1-2 with.
  - **Stiff friction (`impratio = 100`):** stick/slip 8-12 without and about 6 with; liftoff about 12 without and 2 with.
  - **When `k` is too small,** the fallback returns the surrogate wherever the solve has not converged.
- **Float32 (Go1, GPU).** Over 2048 random-action states, the smoothed residual at `q*` was typically only about 2 times the precision floor at softness 0.1. In most states the hard solution is already the smoothed root to float32 precision, so refinement barely moves the evaluation point. Refinement was accepted in 99.5% of states at `k = 2` and 100% at `k >= 4` (98% and 100% at softness 0.4). The rejected states were the partially converged ones, which received the surrogate.
- **Effect.** Against the converged consistent gradient, the surrogate differed by 13-42% near stick/slip and liftoff in a single step. Over a 100-step rollout at softness 1, the surrogate's error compounded to about 343 times, while the refined gradient stayed bounded. Neither one matches the hard dynamics. Refinement makes the gradient self-consistent, not more physical.

**Related work.** These are the closest methods and the gaps that motivated this one. None of them is this exact method: a backward-only softplus on the per-row Jaref mask of MuJoCo's convex soft-constraint solve, restricted to in-contact rows so that it composes with pw_solimp.

- **DiffContact** ([arXiv:2610.35852](https://arxiv.org/abs/2610.35852)). The closest in spirit. The forward rollout is hard contact. The backward pass relaxes the complementarity conditions (separation, clamping, sliding) and differentiates the relaxed point, so the sensitivity stays continuous through liftoff and stick/slip. It targets an NCP solver, not MuJoCo's convex solve.
- **Dojo** (Howell et al., [arXiv:2203.00806](https://arxiv.org/abs/2203.00806)). It computes smooth gradients by applying the implicit function theorem to a relaxed interior-point solution (central-path parameter > 0) of a hard-contact solver.
- **Analytic smoothing** (Pang et al., [arXiv:2206.10787](https://arxiv.org/abs/2206.10787)). It smooths a convex quasi-dynamic contact model with a log-barrier relaxation, which gives force-at-a-distance. This changes the model itself, not only the gradient.
- **Contacts From Distance** (Paulus et al., ICLR 2026, [arXiv:2506.14186](https://arxiv.org/abs/2506.14186)). A backward-only, straight-through mechanism in MJX that gives contact gradients before touch. It is the counterpart for touchdown that pw_solimp implements here.
- **Implicit contact differentiation in MJX** (Layeghi et al., [arXiv:2607.24959](https://arxiv.org/abs/2607.24959)). The source of the `ift` backend. Its derivative is local to a fixed contact set and friction mode, and its authors note it need not stay valid across contact or friction-mode changes. This method addresses that gap for linear rows.
- **Simple** (Le Lidec et al., 2024, [paper](https://simple-robotics.github.io/publications/simple-gradients/static/paper/simple2024gradients.pdf)). It observes that active-set implicit gradients do not capture contact-mode boundaries, and suggests combining them with smoothing.

**Tests.** `mjx/mujoco/mjx/_src/solver_activeset_softness_test.py` checks the following: softness 0 is bit-identical to the default, forward values are unchanged, stick/slip, liftoff, limit release, and friction-loss transitions are smoothed, the gradient converges to the hard one away from a switch, pw_solimp rows are untouched, and the elliptic, backend, and invalid-value guards raise. Its `RefinementTest` class checks refinement on the following points:

- It is bit-identical to hard when every softness is 0, and it leaves forward values unchanged.
- Converged refinement equals the finite-difference derivative of an independently computed smoothed root, and it differs from the surrogate near a switch.
- The line search handles both overshoot and undershoot (bracket expansion), and it converges where plain Newton diverges.
- The fallback rejects partial and non-finite convergence.
- On default and stiff-friction boxes, `k = 2` gives exactly the surrogate or a near-converged gradient at every point. `k = 4` (default) and `k = 6` (stiff) match the converged gradient.
- Invalid iteration counts and tolerances raise.

<h1>
  <a href="#"><img alt="MuJoCo" src="banner.png" width="100%"/></a>
</h1>

<p>
  <a href="https://github.com/google-deepmind/mujoco/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco/build.yml?branch=main">
  </a>
  <a href="https://mujoco.readthedocs.io/" alt="Documentation">
    <img src="https://readthedocs.org/projects/mujoco/badge/?version=latest">
  </a>
  <a href="https://github.com/google-deepmind/mujoco/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco">
  </a>
</p>

**MuJoCo** stands for **Mu**lti-**Jo**int dynamics with **Co**ntact. It is a
general purpose physics engine that aims to facilitate research and development
in robotics, biomechanics, graphics and animation, machine learning, and other
areas which demand fast and accurate simulation of articulated structures
interacting with their environment.

This repository is maintained by [Google DeepMind](https://www.deepmind.com/).

MuJoCo has a C API and is intended for researchers and developers. The runtime
simulation module is tuned to maximize performance and operates on low-level
data structures that are preallocated by the built-in XML compiler. The library
includes interactive visualization with a native GUI, rendered in OpenGL. MuJoCo
further exposes a large number of utility functions for computing
physics-related quantities.

We also provide [Python bindings] and a plug-in for the [Unity] game engine.

## Documentation

MuJoCo's documentation can be found at [mujoco.readthedocs.io]. Upcoming
features due for the next release can be found in the [changelog] in the
"latest" branch.

## Getting Started

There are two easy ways to get started with MuJoCo:

1. **Run `simulate` on your machine.**
[This video](https://www.youtube.com/watch?v=P83tKA1iz2Y) shows a screen capture
of `simulate`, MuJoCo's native interactive viewer. Follow the steps described in
the [Getting Started] section of the documentation to get `simulate` running on
your machine.

2. **Explore our online IPython notebooks.**
If you are a Python user, you might want to start with our tutorial notebooks
running on Google Colab:

 - The **introductory** tutorial teaches MuJoCo basics:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)
 - The **Model Editing** tutorial shows how to create and edit models procedurally:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb)
 - The **rollout** tutorial shows how to use the multithreaded `rollout` module:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb)
 - The **LQR** tutorial synthesizes a linear-quadratic controller, balancing a
   humanoid on one leg:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb)
 - The **least-squares** tutorial explains how to use the Python-based nonlinear
   least-squares solver:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb)
 - The **MJX** tutorial provides usage examples of
   [MuJoCo XLA](https://mujoco.readthedocs.io/en/stable/mjx.html), a branch of MuJoCo written in JAX:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb)
 - The **differentiable physics** tutorial trains locomotion policies with
   analytical gradients automatically derived from MuJoCo's physics step:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb)

## Installation

### Prebuilt binaries

Versioned releases are available as precompiled binaries from the GitHub
[releases page], built for Linux (x86-64 and AArch64), Windows (x86-64 only),
and macOS (universal). This is the recommended way to use the software.

### Building from source

Users who wish to build MuJoCo from source should consult the [build from
source] section of the documentation. However, note that the commit at
the tip of the `main` branch may be unstable.

### Python (>= 3.10)

The native Python bindings, which come pre-packaged with a copy of MuJoCo, can
be installed from [PyPI] via:

```bash
pip install mujoco
```

Note that Pre-built Linux wheels target `manylinux2014`, see
[here](https://github.com/pypa/manylinux) for compatible distributions. For more
information such as building the bindings from source, see the [Python bindings]
section of the documentation.

## Versioning

We aim to release MuJoCo in the first week of each month. Our versioning
standards changed to modified Semantic Versioning in 3.5.0,
see [versioning](VERSIONING.md) for details.

## Contributing

We welcome community engagement: questions, requests for help, bug reports and
feature requests. To read more about bug reports, feature requests and more
ambitious contributions, please see our [contributors guide](CONTRIBUTING.md)
and [style guide](STYLEGUIDE.md).

## Asking Questions

Questions and requests for help are welcome as a GitHub
["Asking for Help" Discussion](https://github.com/google-deepmind/mujoco/discussions/categories/asking-for-help)
and should focus on a specific problem or question.

## Bug reports and feature requests

GitHub [Issues](https://github.com/google-deepmind/mujoco/issues) are reserved
for bug reports, feature requests and other development-related subjects.

## Related software
MuJoCo is the backbone for numerous environment packages. Below we list several
bindings and converters.

### Bindings

These packages give users of various languages access to MuJoCo functionality:

#### First-party bindings:

- [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
  - [dm_control](https://github.com/google-deepmind/dm_control), Google
    DeepMind's related environment stack, includes
    [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md),
    a module for procedural manipulation of MuJoCo models.
- [JavaScript bindings and WebAssembly support](/wasm/README.md) (inspired [stillonearth](https://github.com/stillonearth) and [zalo](https://github.com/zalo)'s community projects; [mjswan](https://github.com/ttktjmt/mjswan) extends these with real-time policy control, interactive force
application, and more).
- [C# bindings and Unity plug-in](https://mujoco.readthedocs.io/en/stable/unity.html)

#### Third-party bindings:

- **MATLAB Simulink**: [Simulink Blockset for MuJoCo Simulator](https://github.com/mathworks-robotics/mujoco-simulink-blockset)
  by [Manoj Velmurugan](https://github.com/vmanoj1996).
- **Swift**: [swift-mujoco](https://github.com/liuliu/swift-mujoco)
- **Java**: [mujoco-java](https://github.com/CommonWealthRobotics/mujoco-java)
- **Julia**: [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl)
- **Rust**: [MuJoCo-rs](https://github.com/davidhozic/mujoco-rs)

### Converters

- **OpenSim**: [MyoConverter](https://github.com/MyoHub/myoconverter) converts
  OpenSim models to MJCF.
- **SDFormat**: [gz-mujoco](https://github.com/gazebosim/gz-mujoco/) is a
  two-way SDFormat <-> MJCF conversion tool.
- **OBJ**: [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)
  a script for converting composite OBJ files into a loadable MJCF model.
- **onshape**: [Onshape to Robot](https://github.com/rhoban/onshape-to-robot)
  Converts [onshape](https://www.onshape.com/en/) CAD assemblies to MJCF.

## Citation

If you use MuJoCo for published research, please cite:

```
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

## License and Disclaimer

Copyright 2021 DeepMind Technologies Limited.

Box collision code ([`engine_collision_box.c`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c))
is Copyright 2016 Svetoslav Kolev.

ReStructuredText documents, images, and videos in the `doc` directory are made
available under the terms of the Creative Commons Attribution 4.0 (CC BY 4.0)
license. You may obtain a copy of the License at
https://creativecommons.org/licenses/by/4.0/legalcode.

Source code is licensed under the Apache License, Version 2.0. You may obtain a
copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.

[build from source]: https://mujoco.readthedocs.io/en/latest/programming#building-from-source
[Getting Started]: https://mujoco.readthedocs.io/en/latest/programming#getting-started
[Unity]: https://unity.com/
[releases page]: https://github.com/google-deepmind/mujoco/releases
[mujoco.readthedocs.io]: https://mujoco.readthedocs.io
[changelog]: https://mujoco.readthedocs.io/en/latest/changelog.html
[Python bindings]: https://mujoco.readthedocs.io/en/stable/python.html#python-bindings
[PyPI]: https://pypi.org/project/mujoco/
