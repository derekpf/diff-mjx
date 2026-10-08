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

- **Per-operation ST wiring.** Propagates `opt.contact_st_enable` (default `False`) through softened collision and contact-physics operations, preserving nominal hard forward values while using surrogate gradients. Forlax's `DiffContactWrapper` exposes the same `contact_st_enable` argument, independently of `sensor_st_enable`.

- **Closer nominal MJX behavior.** Corrects collision feature selection, tie-breaking, normalization, contact masks, duplicate contacts, and sign conventions.

- **Plane–cylinder degenerate-vector gate.** Uses softness `1e-8`.

- **Capsule–capsule gate.** Uses softness `4e-8`.

- **Box–box SAT softness.** Uses `7e-5` for SAT and `2e-5` for other operations.

- **Soft validity gradients retained.** Removes forced gradient stops around box–box candidate validity, manifold masks, and contact counts.

- **Numerical safeguards.** Adds finite-gradient protection for C2 clipping, coincident contacts, cylinder SDF derivatives, tendon wrapping, and muscle smoothing while retaining nominal forwards.

- **Dynamics consistency.** Shares straight-through handling between forward and inverse piecewise-solimp calculations; limits contact-specific clamping to contact constraints.

- **Independent sensor options.** Adds MJX options `sensor_softness=0.0` and `sensor_st_enable=False`, loaded during model conversion and exposed through Forlax's `DiffContactWrapper`. They are independent of reward and physics-contact settings.

- **Sensor forward/backward behavior.** Zero softness preserves nominal behavior. Positive softness enables relaxed forward/backward evaluation; enabling sensor ST preserves nominal forward values with surrogate gradients. Relaxations use `softjax_mode`, falling back to `"smooth"`.

- **Differentiable contact sensors.** Softens contact eligibility, counts, requested-slot selection, and payload masks, including force, torque, distance, position, normal, and tangent. ST wraps the complete output so inactive masks retain backward paths.

- **Differentiable touch sensors.** Adds `ray_geom_hit(...)` for softened primitive intersection validity and candidate aggregation, propagating gradients through contact eligibility and force accumulation. Existing ray/rangefinder intersection behavior is unchanged.

- **Differentiable sensor cutoffs.** Extends `_apply_cutoff(...)` with softness, ST, and mode arguments; clipping now optionally uses softjax.

- **Shared sensor drop-ins.** Extends the shared softjax exports with the ST wrapper, selection, comparison, and safe-square-root operations required by these paths.

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
