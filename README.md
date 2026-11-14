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

- **Per-operation ST wiring.** Propagates `opt.contact_st_enable` (default `False`) through softened contact-physics operations, preserving nominal hard forward values while using surrogate gradients. Collision detection itself is always hard (below).

- **Closer nominal MJX behavior.** Corrects collision feature selection, tie-breaking, normalization, contact masks, duplicate contacts, and sign conventions.

- **Collision detection is hard for every pair.** Both `collider` decorators (`collision_primitive.py`, `collision_convex.py`) call every collision function with `soft=False`, in every `softjax_mode` and with or without `contact_st_enable`, so contact selection always follows the nominal hard rule. Softened selection (sign, threshold and layout switches, deepest-vertex and manifold choice) has surrogate derivatives that scale with the inverse softness at selection boundaries, and resting contacts sit on those boundaries: plane–convex selection at softness `1e-6` gave distance gradients up to ~1e5 for separated vertices of flat faces (BPTT through G1 and Berkeley Humanoid diverged), and the plane–cylinder layout switch at softness `1e-3` gave contact-position derivatives ~100x the rigid-body scale for a cylinder lying on its side. The per-pair soft code paths remain in the source but are not reached.

- **Numerical safeguards.** Adds finite-gradient protection for C2 clipping, coincident contacts, cylinder SDF derivatives, tendon wrapping, and muscle smoothing while retaining nominal forwards.

- **Dynamics consistency.** Shares straight-through handling between forward and inverse piecewise-solimp calculations; limits contact-specific clamping to contact constraints.

- **Independent sensor options.** Adds MJX options `sensor_softness=0.0` and `sensor_st_enable=False`, loaded during model conversion. They are independent of reward and physics-contact settings.

- **Sensor forward/backward behavior.** Zero softness preserves nominal behavior. Positive softness enables relaxed forward/backward evaluation; enabling sensor ST preserves nominal forward values with surrogate gradients. Relaxations use `softjax_mode`, falling back to `"smooth"`.

- **Differentiable contact sensors.** Softens contact eligibility, counts, requested-slot selection, and payload masks, including force, torque, distance, position, normal, and tangent. ST wraps the complete output so inactive masks retain backward paths.

- **Differentiable touch sensors.** Adds `ray_geom_hit(...)` for softened primitive intersection validity and candidate aggregation, propagating gradients through contact eligibility and force accumulation. Existing ray/rangefinder intersection behavior is unchanged.

- **Differentiable sensor cutoffs.** Extends `_apply_cutoff(...)` with softness, ST, and mode arguments; clipping now optionally uses softjax.

- **Shared sensor drop-ins.** Extends the shared softjax exports with the ST wrapper, selection, comparison, and safe-square-root operations required by these paths.

- **Realised solver iterations.** Every solver backend writes the iteration count it actually ran to `Data._impl.solver_niter`.

- **Constraint cost smoothing.** Adds `opt.stickslip_softness`, `opt.liftoff_softness`, `opt.limit_softness`, and `opt.frictionloss_softness`, which smooth active-set switches in the `ift` backward pass only: each constraint's term of the solver's cost is smoothed, and its force is minus the smoothed cost's gradient. See [Constraint cost smoothing](#constraint-cost-smoothing-ift-backward).

## Constraint-solver derivative backends

The JAX constraint solver can be differentiated in several ways, selected per model by the static option `opt.grad_backend`. The forward physics (MJX's convex Gauss-principle solve, Newton or CG, pyramidal or elliptic cones) is the same for every backend; only the gradient differs. Ported from [calumarnott/mujoco](https://github.com/calumarnott/mujoco) (`3.9.0-derivatives`, Layeghi et al., [arXiv:2607.24959](https://arxiv.org/abs/2607.24959)) and adapted to this fork.

| backend | forward loop | gradient |
|---|---|---|
| `ift` (default) | masked scan, never differentiated | implicit function theorem on the stationarity residual |
| `unrolled` | masked scan (`opt.scan_loop=True`, freezes after convergence) or `while_loop` | unrolled through the iterations |
| `while` | `lax.while_loop` | reverse mode only for `opt.iterations == 1` (upstream MJX) |
| `scan` | exactly `opt.iterations` steps, no convergence check | unrolled through every step |

**Selecting the backend.** Set `opt.grad_backend` on the model, for example `mx.replace(opt=mx.opt.replace(grad_backend='unrolled'))` or forlax's `DiffEnvWrapper(env, grad_backend='unrolled')`. It defaults to `ift`. The choice belongs to the model, so models in one process can use different backends. Names are case-insensitive. An unknown name raises `ValueError`, and a non-string value (including `None` or a `jax.Array`, which would be traced) raises `TypeError`. The choice is made at trace time, so already-compiled functions keep the backend they were traced with.

**`ift` implementation.** At the solution, `F(a) = M a - qfrc_smooth - qfrc_constraint(a) = 0`. The existing solver runs first, on stop-gradient inputs, to obtain `a*` (and the refined point, if refinement is on); `lax.custom_root` then only supplies the implicit derivative there: for the backward pass it solves the adjoint system `(dF/da)^T λ = dL/da` and returns `-λ^T dF/dθ`, where `θ` is everything `F` depends on (state, controls, model parameters, constraint Jacobians and impedances, including any softjax/ST surrogates computed before the solve).

- **Residual (`_root`).** Evaluates `F` directly from the constraint context.
- **Forward loop.** A masked scan instead of `lax.while_loop`: identical solution and iteration counts, but faster on GPU, where the while loop syncs to the host every iteration. Measured on batched 8-step rollouts (512 envs, shared RTX 5080, single runs), forward time dropped about 1.5x on G1 (19.5 to 12.8 ms at 3 iterations) and 1.6 to 1.9x on Go1 (3 to 10 iterations).
- **Tangent solve (`_ift_factor`).** Builds the `nv x nv` residual Jacobian from its structure, `dF/da = M - J^T (df/dJaref) J`, with one JVP of the constraint force per column of `J`, and factorizes it by Cholesky, since it is symmetric positive definite (`df/dJaref` is minus the Hessian of the convex constraint cost; the matrix is symmetrized as `(H + H^T) / 2` against rounding). The hard solve (and refinement) run before `custom_root`, on stop-gradient inputs, so `custom_root` only supplies the derivative. In float32, the gradient's median error against a float64 reference at the same point is 2.2e-3 on G1 and 3.1e-4 on Go1 (residual Jacobian condition numbers about 1.5e5 and 7e3; `mjx/benchmarks/ift_solver/cost_smoothing_robots.py`). `custom_root` transposes this solve in its right-hand side, so it must be exactly linear in it: iterative (CG-type) adjoint solves are not possible here and would need a `custom_vjp`.
- **Outputs.** `qfrc_constraint` and `efc_force` are recomputed from `a*` with ordinary autodiff, so their gradients include both the implicit path through `a*` and the explicit dependence on `θ`. The warmstart only seeds the forward solve and receives no gradient.

**What the gradient means.** `ift` differentiates the *converged* fixed point. When the forward solve stops short (small `opt.iterations`, loose `opt.tolerance`), its value is the truncated iterate but its gradient is that of the exact solution. Unrolled backends instead differentiate the iterations they ran. Measured in float64 against central finite differences (`mjx/benchmarks/ift_solver`):

- **Converged solves.** `ift` matched finite differences (relative error 1e-9 to 1e-16) for every tested scene, cone, solver, and warmstart setting.
- **Unconverged solves are not reproducible across programs.** Where the forward solve stops at `opt.iterations` before converging, separately compiled programs, such as the forward- and reverse-mode gradients of one function, round differently and can stop at different forward solutions. The derivative is evaluated at a point that is not a root, so their gradients differ there. On Go1 at its default 5 iterations, 5 of 64 random-action states stopped unconverged, and in 2 of them the two programs stopped up to 5e-4 apart (smoothed gradients up to 1.3e-3 apart); in every other state, and in all of G1's (10 iterations, all converged), they agree to 2e-13 or better in float64 (`mjx/benchmarks/ift_solver/cost_smoothing_robots.py`). Raise `opt.iterations` if this matters.
- **Unrolled gradients drop or distort the solver sensitivity when:**
  - the warmstart is already the solution (static or standing contact): the line search reports no improvement, and the masked update carries no derivative;
  - the cone is elliptic: the last improving Newton step used the Hessian of a different cone zone (about 6% error, independent of the iteration count);
  - the solver is CG: errors of 100% or more.
- **Non-finite gradients from other sources are not affected.** In the G1 joystick environment, non-finite gradients occurred at similar rates under `unrolled` and `ift`, so they originate outside the constraint-solver differentiation.

**Cost.** The `ift` backward does one `nv x nv` matrix build, one Cholesky solve, and one residual VJP per solve, independent of `opt.iterations`, and it stores no per-iteration residuals. Unrolled backward time and memory grow with the iteration count.

Measured on BPTT through the diff-playground `G1JoystickFlatTerrain` environment, with forlax's G1 SHAC settings (256 envs, 16-step horizon, `pw_solimp` with `contact_st_enable`, 3 solver iterations, float32, one seed, RTX 5080):

| backend | forward | backward | forward + backward | memory |
|---|---:|---:|---:|---:|
| `unrolled`, no remat | – | – | out of memory (needs ~19 GiB) | – |
| `unrolled` + per-step remat | 355 ms | 1583 ms | 1767 ms | 1.33 GB |
| `ift` + per-step remat | 355 ms | 1071 ms | 1252 ms (1.41x faster) | 0.38 GB |
| `ift`, no remat | 356 ms | 359 ms | 1027 ms (1.72x faster) | 5.76 GB |

Remat is `jax.remat(env.step, prevent_cse=False)`, as in forlax's `remat_enable`. In this environment, about 45% of envs produced non-finite gradients over 16 steps under both `unrolled` and `ift`.

Benchmarks, validation scripts, raw results, and reproduction commands are in [`mjx/benchmarks/ift_solver`](mjx/benchmarks/ift_solver/README.md).

**Tests.** `mjx/mujoco/mjx/_src/solver_test.py` (backend selection and forward equivalence against MuJoCo C) and `mjx/mujoco/mjx/_src/solver_grad_test.py` (gradients against unrolled differentiation and finite differences).

## Constraint cost smoothing (IFT backward)

**What it smooths.** At the solution, each inequality row's force is a ReLU of its residual `Jaref = J qacc - aref`: `f_i = D_i * max(0, -Jaref_i)`. The `ift` backward differentiates the stationarity residual at the converged `qacc`, so each row adds `m_i * D_i * J_i^T J_i` to the residual Jacobian, where `m_i = [Jaref_i < 0]` is the row's active-set mask. The force is continuous where a row switches on or off, but the gradient jumps by that row's rank-1 term. These switches are liftoff, stick/slip (pyramidal edges switching, or an elliptic contact changing zone), limit release, and friction-loss stick/slip.

**Where it acts.** Only in the residual that the `ift` backward differentiates (`solver._root`) and in refinement's Newton iterations on that residual. The forward solve, its line search, and the reported outputs (`qacc`, `efc_force`, `qfrc_constraint`) keep the hard mask, so forward values are unchanged. Without refinement the gradient is evaluated at the hard solution, which is not a root of the smoothed residual, so it is a surrogate gradient, not the derivative of any model (see [Constraint gradient refinement](#constraint-gradient-refinement)).

| option | rows | switch | unit |
|---|---|---|---|
| `stickslip_softness` | nominal contacts: frictionless, pyramidal and elliptic contacts in contact (`efc_pos < efc_margin`) | stick/slip | fraction of the contact's friction utilization |
| `liftoff_softness` | the same contacts | liftoff | solimp widths |
| `limit_softness` | engaged joint and tendon limit rows (`efc_pos < efc_margin`) | limit release | solimp widths |
| `frictionloss_softness` | DOF and tendon friction-loss rows | friction-loss stick/slip | stick half-windows `r * frictionloss` |

Every softness defaults to 0, which leaves its switch exactly hard, and each acts independently.

**Normalization.** Each softness is dimensionless: it multiplies the row's own `Jaref` scale `efc_softness_scale`, computed with the row in `make_constraint` from that constraint's parameters, so one value means the same thing in every model:

| rows | `efc_softness_scale` | meaning |
|---|---|---|
| contact, limit | `k * imp * width` (solref stiffness `k`, impedance `imp`, solimp `width`) | the reference acceleration of a penetration one solimp width deep |
| friction loss | `r * frictionloss` | the stick half-window `b` |

All pyramid edges of one contact share solref, solimp and depth, so they share one scale. The scale is held constant (`stop_gradient`): it sets the band width and adds no gradient of its own. `put_data` fills it with ones as a placeholder; the next `make_constraint` recomputes it. Contacts also use the contact's own load for stick/slip (below). `NormalizedCoordinateTest` checks that the smoothed curves coincide in these coordinates across different solref, solimp, mass and friction-loss settings.

### Cost formulation

MuJoCo's constraint solve is the convex problem

$$
\min_{\dot v}\; \tfrac12 \lVert \dot v - \dot v_{\mathrm{smooth}} \rVert_M^2 + \sum_c C_c(y), \qquad y = J \dot v - a_{\mathrm{ref}},
$$

over the accelerations $\dot v$ (`qacc`), with $y$ the constraint rows' `Jaref`; the constraint force is minus the gradient of the constraint cost $C$. For a limit or contact row the hard cost is $\tfrac12 D_i \max(0, -y_i)^2$, whose kink at $y_i = 0$ is the active-set switch. Constraint cost smoothing replaces each constraint's cost with a smooth convex one, measured in that constraint's own coordinates, and takes the force (and its derivatives) from the smoothed cost. It uses the antiderivative of softplus,

$$
F(z) = \int_{-\infty}^{z} \mathrm{softplus}(t)\, dt = -\mathrm{Li}_2(-e^{z}), \qquad F' = \mathrm{softplus}, \quad F'' = \sigma,
$$

and the smoothed quadratic $h(y, \tau) = \tau^2 F(-y/\tau)$, which tends to the hard $\tfrac12 \max(0, -y)^2$ as $\tau \to 0$. `softplus` is softjax's `sj.relu(., softness=1, mode='smooth', gated=False)`.

**Limits and friction loss.** A limit or friction-loss row is a one-row constraint with a constant band $\tau_i = s\,\ell_i$: its softness $s$ times its `efc_softness_scale` $\ell_i$. A limit row's cost $D_i\, h(y_i, \tau_i)$ gives

$$
f_i = D_i\, \tau_i\, \mathrm{softplus}(-y_i / \tau_i),
$$

with slope $D_i\, \sigma(-y_i/\tau_i) \in (0, D_i)$. A friction-loss row's hard force clips $x = -y$ to $[-b, b]$, $b = r f$ with $f$ the row's `frictionloss` (the gradient of a Huber cost); smoothed, it is

$$
f_i = D_i \left( x - \tau_i\, \mathrm{softplus}\!\left(\tfrac{x - b}{\tau_i}\right) + \tau_i\, \mathrm{softplus}\!\left(\tfrac{-x - b}{\tau_i}\right) \right),
$$

with slope in $(0, D_i)$ as well.

**Contacts.** A contact's rows (its $N = 2(\mathrm{condim} - 1)$ pyramid edges $y_e = n \pm \mu t$, or one frictionless row) are one constraint, smoothed with one band that follows the contact's normal coordinate $n = \tfrac1N \sum_e y_e$ (the normal row's `Jaref`):

$$
C(y) = \sum_e D_e\, \tau(n)^2\, F\!\left(-\frac{y_e}{\tau(n)}\right), \qquad
\tau(n) = \ell_0 \left( 1 + \mathrm{softplus}\!\left( \frac{s_{\mathrm{slip}}\,(-n)}{\ell_0} \right) \right), \qquad
\ell_0 = \frac{s_{\mathrm{lift}}\, k\, \mathrm{imp}\, w}{1 + \ln 2},
$$

with $s_{\mathrm{slip}}$ = `stickslip_softness`, $s_{\mathrm{lift}}$ = `liftoff_softness`, and $k\,\mathrm{imp}\,w$ the contact's `efc_softness_scale` (solref stiffness, impedance, solimp width). The band is in the contact's own coordinates:

- at liftoff ($n = 0$), $\tau = s_{\mathrm{lift}}$ solimp widths;
- separating ($n \gg 0$), it settles to the floor $\ell_0$;
- loaded, $\tau \approx s_{\mathrm{slip}} \lvert n \rvert + \ell_0$, so stick/slip is smoothed over a fraction $s_{\mathrm{slip}}$ of friction utilization (an edge's share of the contact's load) at any load;
- for contacts lighter than about $\lvert n \rvert \sim \ell_0 / s_{\mathrm{slip}}$ the two merge: the slip lines meet at the cone's apex.

With $u_e = -y_e/\tau$ and $G(u) = 2F(u) - u\, \mathrm{softplus}(u)$ (so $\partial h / \partial \tau = \tau G$), the force is

$$
f_e = -\frac{\partial C}{\partial y_e}
= \underbrace{D_e\, \tau\, \mathrm{softplus}(u_e)}_{\text{the edge's own softplus}}
\;+\; \underbrace{\frac{-\tau'(n)}{N} \sum_k D_k\, \tau\, G(u_k)}_{\text{coupling through the band}},
\qquad \tau'(n) = -s_{\mathrm{slip}}\, \sigma\!\left( \frac{s_{\mathrm{slip}}\,(-n)}{\ell_0} \right).
$$

The coupling term acts equally on every edge, as a normal force, and vanishes for a constant band ($s_{\mathrm{slip}} = 0$). It makes a loaded contact about $1 + \pi^2 s_{\mathrm{slip}}^2 / 3$ stiffer in the backward pass (about $1.12\,D$ at 0.2), and puts its edge masks at about 0.68 rather than 0.5 at liftoff. The force's Jacobian $\nabla^2 C$ is the diagonal mask $\mathrm{diag}(D_e\, \sigma(u_e))$ plus a rank-2 update along the normal mode, so it is block diagonal per contact.

Either knob can be 0 while the other is on. With $s_{\mathrm{lift}} = 0$ the band is the $\ell_0 \to 0$ limit $\tau = s_{\mathrm{slip}} \max(0, -n)$: stick/slip is smoothed as before, and once the contact unloads $\tau = 0$ gives exactly the hard force. With $s_{\mathrm{slip}} = 0$ the band is the constant $s_{\mathrm{lift}}\, k\, \mathrm{imp}\, w$ on every edge. That smooths liftoff and the stick/slip of contacts light enough for their friction limit to lie inside the band; a heavily loaded contact's stick/slip is then effectively, not exactly, hard.

**Convexity.** $h$ is jointly convex for $\tau > 0$ (its Hessian determinant is $2FF'' - F'^2 > 0$) and nondecreasing in $\tau$ ($G > 0$), and $\tau(n)$ is non-negative and convex, so $C$ is convex (full proof in the `_contact_cost_force` docstring). The residual Jacobian $M + J^\top \nabla^2 C\, J$ is therefore symmetric positive definite, and the IFT and refinement factor it by Cholesky. On Go1 and G1 rollout states its smallest eigenvalue is at least 0.94 times that of $M$ (`mjx/benchmarks/ift_solver/cost_smoothing_robots.py`).

**Elliptic cones.** MuJoCo's elliptic contact has no edges: its rows are the normal and the friction components, in scaled coordinates $n = \mu y_0$ and $t_j = f_j y_j$ ($f_j$ the friction coefficients, $\mu = f_1/\sqrt{\text{impratio}}$). Its cost is $\tfrac{D_m}{2}$ times the squared distance to the cone $n \ge \mu \lVert t \rVert$, with three zones: no force (liftoff), force on the cone (sliding), force inside it (sticking). That cost splits into two kinks:

$$
\psi(n, t) = \frac{D_m}{2} \Big[ \max(0, -k_1)^2 + \mu^2 \max(0, -k_2)^2 \Big],
\qquad k_1 = n - \mu \lVert t \rVert \;\text{(liftoff)}, \quad k_2 = n + \lVert t \rVert / \mu \;\text{(stick/slip)},
$$

with $D_m = D_0 / (\mu^2(1+\mu^2))$ and $D_0$ the normal row's `efc_D`. (MuJoCo gives the friction rows $D_j = D_0 f_j^2/\mu^2$, which makes the sticking zone isotropic; the two kinks then reproduce its three zones exactly.) Each kink gets the same smoothed quadratic, with the contact's band $\tau(n)$ evaluated once for both:

$$
C(n, t) = D_m \Big[ h\big(k_1, \tau(n)\big) + \mu^2\, h\big(k_2, \tau(n)\big) \Big],
$$

with $\tau(n)$ as above and $\ell_0 = \mu\, s_{\mathrm{lift}}\, k\,\mathrm{imp}\,w / (1+\ln 2)$, since $n = \mu y_0$. The liftoff kink lies at $n \ge 0$ and the stick/slip kink at $n < 0$, so liftoff is smoothed over $s_{\mathrm{lift}}$ solimp widths ($k_1/\tau = y_0 / (s_{\mathrm{lift}}\, k\,\mathrm{imp}\,w)$ without sliding), and stick/slip over a fraction $s_{\mathrm{slip}}$ of utilization $U = \lVert t\rVert / (\mu \lvert n \rvert)$ ($k_2/\tau \approx -(1-U)/s_{\mathrm{slip}}$ under load), as for pyramidal contacts. With $u_i = -k_i/\tau$ the force is

$$
f_n = D_m\, \tau \big[ \mathrm{softplus}(u_1) + \mu^2\, \mathrm{softplus}(u_2) \big] - \tau'(n)\, D_m\, \tau \big[ G(u_1) + \mu^2 G(u_2) \big],
\qquad
f_t = -D_m (1 + \mu^2)\, \frac{\mathrm{softplus}(u_2) - \mathrm{softplus}(u_1)}{u_2 - u_1}\, t,
$$

and on the rows $(\mu f_n,\ f_j (f_t)_j)$. $C$ is convex: $\varphi(n, \rho) = D_m[h(n - \mu\rho, \tau(n)) + \mu^2 h(n + \rho/\mu, \tau(n))]$ is jointly convex in $(n, \rho)$, and because both kinks share one band it is nondecreasing in $\rho \ge 0$ ($\partial_\rho \varphi = D_m \mu [h_y(k_2) - h_y(k_1)] \ge 0$ with $k_2 \ge k_1$), so $C = \varphi(n, \lVert t\rVert)$ is convex. $\partial_\rho \varphi = 0$ at $\rho = 0$, so $C$ is also differentiable at zero tangential acceleration, the common sticking state. Both knobs work as for pyramidal contacts, including either at 0. Frictionless (condim-1) contacts are smoothed as on the pyramidal cone; note that MJX's `make_data` rejects condim-1 contacts on the elliptic cone (G1's feet, for example), so such models only run with data from `put_data`.

**In the solver.** Only the backward pass changes: the residual that the IFT differentiates (`_root`), and refinement's Newton iterations on it, use the smoothed forces; the forward solve and its outputs stay hard. Both contact forces have an analytic tangent rule (`jax.custom_jvp`), with $F$ computed in closed form once per edge or kink: a dilogarithm series evaluated by Horner's rule, 40 terms in float64 and 20 in float32 (below each precision's rounding). For elliptic contacts the Hessian in $(n, t)$ is $H_{nn} = \varphi_{nn}$, $H_{nt} = w_n\, t$, $H_{tt} = w\, I + (\varphi_{\rho\rho} - w)\, \hat t \hat t^\top$ with $w = \partial_\rho\varphi / \rho$; $w$ and $w_n$ are divided differences over $[u_1, u_2]$, so nothing divides by $\lVert t\rVert$. Tangents in the friction coefficients use autodiff, only when present. In-contact contacts are selected by their normal row, since elliptic friction rows have `efc_pos` 0.

**Cost.** BPTT through 16 control steps x 32 envs (float32, RTX 5080, stick/slip 0.2, liftoff 0.05, limit and friction loss 0.2, median of 15; `mjx/benchmarks/ift_solver/cost_smoothing_robots.py`): Go1 118.6 ms hard, 121.8 ms smoothed, 159.7 ms smoothed with refinement (`k = 4`, 4 line-search trials); G1 523.6, 527.9 and 643.3 ms.

**Not supported.**

- **Limit touchdown.** Limit engagement is not smoothed. A row only exists once a joint or tendon reaches its limit margin, and it switches on at full impedance with `aref = -b * v`, so an approaching limit still gets a force jump. `limit_softness` covers only the release of an engaged limit. pw_solimp anticipates contact touchdown only and does not apply to limits.
- **Contact touchdown is left to pw_solimp.** pw_solimp rows (`efc_pos >= efc_margin`) and zeroed padding rows keep the hard mask, so the two mechanisms do not overlap. Under `contact_st_enable`, gradients come from the pw_solimp branch. Smoothing applies to that branch's nominal rows, whose definition is identical to the nominal branch's.
- **Other backends.** Only `ift` reads these options. Setting any of them above 0 with the `unrolled`, `scan`, or `while` backend raises an error instead of being ignored.

**Usage.**

- **Static values.** Each option is a static Python float, so changing it retraces the compiled step. A `jax.Array` raises `TypeError`, because it would be traced. Negative or non-finite values raise `ValueError`.
- **Zero is exactly hard.** At `0` (the default), the smoothing branch is not traced, so behavior is bit-identical to the hard mask. A small positive softness is not the same: at a switch the slope is about 0.5 (0.68 for a contact at liftoff), not 0 or 1.
- **forlax.** `DiffEnvWrapper` accepts the same keyword arguments and validates them at construction.

### Constraint gradient refinement

**`opt.refinement_iterations`.** By default (`0`), the smoothed slopes are evaluated at the hard solution `q*`, which is not a root of the smoothed residual. The result is a surrogate that is not the derivative of any model, and its error can compound over a rollout. With `refinement_iterations = k > 0`, the `ift` backward first takes `k` line-searched Newton iterations on the smoothed residual, starting from `q*`. It then evaluates the gradient at that point, the smoothed root, so it becomes the exact derivative of the smoothed model at its own solution. DiffContact calls the analogous point the backward evaluation point. Forward values stay `q*` through straight-through (`stop_gradient(q*) + (q_eval - stop_gradient(q_eval))`), so states, contacts, and outputs are unchanged. Refinement does not change the active set or the forward solve. It only moves the point where the gradient is evaluated.

- **When it runs.** It is skipped (not traced) when every softness is 0, because the smoothed root is then `q*`. It uses the residual of all four softness options.
- **Line search (`opt.refinement_ls_iterations`, default `0`, mirroring `opt.ls_iterations`).** At the default of `0`, refinement takes full Newton steps (plain Newton, with no line-search cost). With `refinement_ls_iterations = n > 0`, each Newton direction is followed by `n` trials of a line search on the smoothed cost (`_smoothed_linesearch`). It uses the same idea as the forward `_linesearch`, a bracketed Newton iteration on the directional derivative. It evaluates the smoothed cost's derivatives exactly, whereas the forward search uses the hard cost's piecewise-quadratic coefficients. `J p` and `M p` are computed once per direction, so each trial evaluates only the smoothed force and one JVP of it along `J p`. It has no tolerance counterpart to `ls_tolerance`, because it always runs all `n` trials. It is a separate function so the forward solve stays untouched. Plain Newton on the smoothed residual tends to undershoot rather than overshoot: rows that switch off along the step lower the curvature, so the minimizer lies beyond the full step. The search therefore expands (`alpha <- max(Newton, 2 alpha)`) until the derivative changes sign.
- **Fallback (`opt.refinement_fallback_enable`, default `True`; `opt.refinement_tolerance`, default `1e-6`).** A partially converged point can give a gradient worse than the surrogate. It is neither the hard state nor the smoothed model's solution, and partial convergence still reduces the residual, so a reduce-only check would not catch it. The refined point is used only if it is finite and `|r(q_eval)| <= max(tolerance * |r(q*)|, floor)`, where `floor = 64 * eps * (|M qacc| + |qfrc_smooth| + |qfrc_constraint|)` is the residual's precision floor. Otherwise `q*` is used, which gives the surrogate (the same evaluation point; equal up to float rounding, since the two compiled programs can order the backward arithmetic differently). The gradient is therefore always either the surrogate or near-converged, never in between. The decision is per solve, so per environment under `vmap`. Setting `refinement_fallback_enable = False` skips the tolerance check, so the refined point is used even when partially converged. This is for experiments, and it can give in-between gradients. Non-finite points still fall back, because they would only give NaN gradients. The toggle follows the fork's `*_enable` naming for boolean options (`contact_st_enable`, `sensor_st_enable`). MuJoCo's own boolean options are bits in `opt.disableflags` and `opt.enableflags`, which cannot be extended without changing MuJoCo C.
- **Validation.** `refinement_iterations` and `refinement_ls_iterations` must be static Python `int >= 0`, `refinement_tolerance` a static finite Python float `>= 0`, and `refinement_fallback_enable` a Python `bool`. Wrong types (including bools for the counts and `jax.Array` for any option) raise `TypeError`, and out-of-range values raise `ValueError`.
- **Implementation.** Only the constraint force is nonlinear: `r(q) = M q - qfrc_smooth - J^T f(Jaref)` with `Jaref = J q - aref`. Each Newton step (`_smoothed_newton`) therefore builds the exact residual Jacobian as `M - J^T (df/dJaref) J`, getting `(df/dJaref) J` from one JVP of the force per column of `J`. This pushes the `nv` tangents through the force alone. It holds for either cone: `df/dJaref` is block diagonal per contact (the per-contact cost couples a pyramidal contact's edges; the elliptic cone couples its rows), and diagonal for limit and friction-loss rows. With the hard 0/1 slopes it is the forward Newton Hessian. It is symmetric positive definite, so it is solved with Cholesky. The precision floor uses the forward solve's final context. Each iteration's smoothed context is carried to the next step, so every point is evaluated once and the last one gives `r(q_eval)`. The smoothed residual at `q*` is evaluated fresh. (Not from the forward solve's final `Jaref` and `Ma`, which carry its line search's incremental rounding.) A softness whose row type the model lacks (no contacts, limits, or friction loss) is dropped at trace time (`_effective_softnesses`), so on its own it does not trigger a refinement.
- **Cost.** Each iteration does one smoothed constraint evaluation, `nv` force JVPs, one `nv x nv` matrix build and Cholesky solve, and `refinement_ls_iterations` line-search trials that each evaluate only the force and one force JVP. Measured cost: see the constraint cost smoothing timing above (`k = 4` adds about 31% to Go1's BPTT and 22% to G1's).
- **Choosing `k`.** The per-contact band follows the contact's load, so the smoothed residual is nonlinear in the normal direction. On the box sweep of `RefinementTest.test_default_stiffness_iteration_counts` (61 pushes within 3 N of the switch, 8 line-search trials), the gradient converged in 18% of points at `k = 2` and in all of them at `k >= 3`; the stiff-friction box needs `k = 6`. On Go1 and G1 rollouts (stick/slip 0.2, liftoff 0.05; `mjx/benchmarks/ift_solver/cost_smoothing_robots.py`), refinement was accepted in 71% and 67% of solves at `k = 2`, and in 99.7% and 100% at `k = 4`. Use `k = 4` or more. When `k` is too small, the fallback returns the surrogate wherever the solve has not converged.
- **Meaning.** Neither the surrogate nor the refined gradient is the hard dynamics' derivative. Refinement makes the gradient self-consistent (the derivative of the smoothed model at its own solution), not more physical.

**Related work.** These are the closest methods and the gaps that motivated this one. None of them is this exact method: a backward-only smoothing of MuJoCo's convex soft-constraint cost, per contact in the contact's own coordinates, restricted to in-contact rows so that it composes with pw_solimp.

- **DiffContact** ([arXiv:2610.35852](https://arxiv.org/abs/2610.35852)). The closest in spirit. The forward rollout is hard contact. The backward pass relaxes the complementarity conditions (separation, clamping, sliding) and differentiates the relaxed point, so the sensitivity stays continuous through liftoff and stick/slip. It targets an NCP solver, not MuJoCo's convex solve.
- **Dojo** (Howell et al., [arXiv:2203.00806](https://arxiv.org/abs/2203.00806)). It computes smooth gradients by applying the implicit function theorem to a relaxed interior-point solution (central-path parameter > 0) of a hard-contact solver.
- **Analytic smoothing** (Pang et al., [arXiv:2206.10787](https://arxiv.org/abs/2206.10787)). It smooths a convex quasi-dynamic contact model with a log-barrier relaxation, which gives force-at-a-distance. This changes the model itself, not only the gradient.
- **Contacts From Distance** (Paulus et al., ICLR 2026, [arXiv:2506.14186](https://arxiv.org/abs/2506.14186)). A backward-only, straight-through mechanism in MJX that gives contact gradients before touch. It is the counterpart for touchdown that pw_solimp implements here.
- **Implicit contact differentiation in MJX** (Layeghi et al., [arXiv:2607.24959](https://arxiv.org/abs/2607.24959)). The source of the `ift` backend. Its derivative is local to a fixed contact set and friction mode, and its authors note it need not stay valid across contact or friction-mode changes. This method addresses that gap for contacts (pyramidal and elliptic), limits and friction loss.
- **Simple** (Le Lidec et al., 2024, [paper](https://simple-robotics.github.io/publications/simple-gradients/static/paper/simple2024gradients.pdf)). It observes that active-set implicit gradients do not capture contact-mode boundaries, and suggests combining them with smoothing.

**Tests.** `mjx/mujoco/mjx/_src/solver_softness_test.py`:

- `ContactSoftnessTest` (and `EllipticContactSoftnessTest`, the same on the elliptic cone), `LimitAndFrictionlossSoftnessTest`: softness 0 is bit-identical to the default, forward values are unchanged, stick/slip, liftoff, limit release, and friction-loss transitions are smoothed, the gradient converges to the hard one away from a switch, pw_solimp rows are untouched, one row type's softness leaves the others alone, and the backend and invalid-value guards raise.
- `PerContactCostTest`: the softplus integral, its dilogarithm series (Horner's rule against a long direct sum; float32's 20 terms against float64) and the convexity proof's identities; the band; the force is minus the gradient of an independently written cost with a positive semidefinite Hessian; the hard limit; stick/slip in utilization units at any load; liftoff in solimp widths; each contact knob at 0 (exactly hard liftoff, or a constant band); `mean_e Jaref_e = Jaref_n`; float32; the IFT gradient against the finite-difference derivative of the smoothed root (condim 1, 3, 4 and 6); the static row blocks.
- `EllipticContactCostTest`: the two-kink form reproduces MJX's hard elliptic force (condim 3, 4, 6; impratio); the force is minus the gradient of an independently written cost with a positive semidefinite Hessian, for each knob at 0 too; differentiability at zero tangential acceleration (float64 and float32); liftoff in solimp widths and stick/slip in utilization, coinciding across mass, gravity, friction, impratio, solref, solimp and condim (scales from MuJoCo C's `efc_KBIP`); exactly hard liftoff at `liftoff_softness = 0`; the analytic tangent rule (tangents in the rows, `D`, the friction coefficients and $\mu$) and its second derivative against autodiff of the reference cost; float32 against float64; the IFT gradient against the finite-difference derivative of the smoothed root (condim 3, 4, 6); row layout and in-contact selection.
- `NormalizedCoordinateTest`: friction loss, limits, liftoff and stick/slip give the same curves in their own coordinates across parameter sets.
- `SoftnessScaleTest`: a contact's scale is MuJoCo's `K * I * width`, shared by all of its pyramid edges, and a friction-loss row's scale is its stick half-window.
- `BackwardImplementationTest`: the analytic contact tangent rule (also with only the rows or only `D` perturbed) and its second derivative against autodiff; the IFT against `inv(jacfwd(_root))` in forward and reverse mode (pyramidal and elliptic, condim 1-6, equality, tendon, limit and friction-loss rows; hard and smoothed, including elliptic contact smoothing); refinement against a Newton-converged smoothed root.
- `RefinementTest`:
  - bit-identical to hard when every softness is 0, and forward values unchanged;
  - converged refinement equals the finite-difference derivative of an independently computed smoothed root, and differs from the surrogate near a switch;
  - the line search handles both overshoot and undershoot (bracket expansion), and converges where plain Newton diverges;
  - the fallback rejects partial and non-finite convergence;
  - on the default box, `k = 2` gives the surrogate or a converged gradient at every point and `k = 3` and `4` match the converged gradient; on the stiff-friction box, `k = 2` falls back where unconverged and `k = 6` converges;
  - invalid iteration counts and tolerances raise.
- `RefinementNewtonStepTest`: the Newton step (residual, direction, and line-search derivatives) against `jacfwd` and JVPs where rows are partly switched, for both cones; it reaches the same refined point as the generic path; elliptic refinement gives the finite-difference derivative of the smoothed root; softnesses of absent row types are dropped.

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
