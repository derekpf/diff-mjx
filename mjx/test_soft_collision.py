"""Test script for soft collision detection in MJX."""

import sys

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import smooth
import numpy as np

passed = 0
failed = 0


def run_test(name, xml, expect_zero_grad=False, skip_grad=False):
    """Run hard, soft-hard, soft-c2, and gradient tests for a given model."""
    global passed, failed
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    # --- Hard collision ---
    mx_hard = mx.replace(opt=mx.opt.replace(softjax_mode=None))
    collision_hard = jax.jit(collision_driver.collision)(mx_hard, dx)
    dist_hard = np.array(collision_hard.contact.dist)
    print(f"  Hard dist: {dist_hard}")

    # --- Soft collision, softjax_mode="hard" ---
    mx_soft_hard = mx.replace(opt=mx.opt.replace(softjax_mode="hard"))
    collision_soft_hard = jax.jit(collision_driver.collision)(mx_soft_hard, dx)
    dist_soft_hard = np.array(collision_soft_hard.contact.dist)
    print(f"  Soft-hard dist: {dist_soft_hard}")

    has_nan = np.any(np.isnan(dist_soft_hard))
    if has_nan:
        print(f"  FAILED [{name}] soft-hard: NaN in dist")
        failed += 1
    else:
        print(f"  PASSED [{name}] soft-hard: no NaN")
        passed += 1

    # --- Soft collision, softjax_mode="c2" ---
    mx_soft = mx.replace(opt=mx.opt.replace(softjax_mode="c2"))
    collision_soft = jax.jit(collision_driver.collision)(mx_soft, dx)
    dist_soft = np.array(collision_soft.contact.dist)
    print(f"  Soft-c2 dist: {dist_soft}")

    has_nan = np.any(np.isnan(dist_soft))
    if has_nan:
        print(f"  FAILED [{name}] soft-c2: NaN in dist")
        failed += 1
    else:
        print(f"  PASSED [{name}] soft-c2: no NaN")
        passed += 1

    # --- Gradient test ---
    if skip_grad:
        print(f"  SKIPPED [{name}] gradient: known upstream NaN (pre-existing)")
        return

    def soft_collision_loss(qpos):
        dx_mod = dx.replace(qpos=qpos)
        dx_mod = smooth.kinematics(mx_soft, dx_mod)
        dx_mod = smooth.com_pos(mx_soft, dx_mod)
        dx_mod = collision_driver.collision(mx_soft, dx_mod)
        return jp.sum(dx_mod.contact.dist)

    grad_fn = jax.jit(jax.grad(soft_collision_loss))
    grads = grad_fn(dx.qpos)
    grads_np = np.array(grads)
    print(f"  Grad wrt qpos: {grads_np}")

    grad_nan = np.any(np.isnan(grads_np))
    grad_zero = np.allclose(grads_np, 0)
    if grad_nan:
        print(f"  FAILED [{name}] gradient: NaN")
        failed += 1
    elif grad_zero and not expect_zero_grad:
        print(f"  FAILED [{name}] gradient: all zero")
        failed += 1
    elif grad_zero and expect_zero_grad:
        print(f"  PASSED [{name}] gradient: zero as expected (degenerate config)")
        passed += 1
    else:
        print(f"  PASSED [{name}] gradient: flows correctly")
        passed += 1


# ============================================================
# Test models — one per collision pair we've softened
# ============================================================

# 1. plane_box (plane_convex) — original test
PLANE_BOX_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 2. plane_sphere
PLANE_SPHERE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 3. plane_capsule
PLANE_CAPSULE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.2" euler="30 0 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 4. plane_ellipsoid
PLANE_ELLIPSOID_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.15" euler="20 10 0">
      <freejoint/>
      <geom type="ellipsoid" size="0.1 0.08 0.06" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 5. plane_cylinder
PLANE_CYLINDER_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.2" euler="15 0 0">
      <freejoint/>
      <geom type="cylinder" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 6. sphere_sphere
SPHERE_SPHERE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body pos="0.15 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 7. sphere_capsule
SPHERE_CAPSULE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body pos="0.12 0 0.5">
      <freejoint/>
      <geom type="capsule" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 8. capsule_capsule
CAPSULE_CAPSULE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5" euler="0 90 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
    <body pos="0 0 0.62" euler="90 0 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 9. plane_cylinder (axis nearly parallel to plane — exercises the prjaxis~0 branch)
PLANE_CYLINDER_PARALLEL_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.12" euler="0 90 0">
      <freejoint/>
      <geom type="cylinder" size="0.05 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 10. sphere_sphere (coincident — exercises the dist==0 fallback)
SPHERE_SPHERE_COINCIDENT_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 11. capsule_capsule (parallel — exercises segment-to-segment edge case)
#     Capsules separated enough that MuJoCo C finds <=1 contact,
#     but close enough that MJX exercises the parallel segment path.
CAPSULE_CAPSULE_PARALLEL_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.5" euler="0 90 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
    <body pos="0 0.12 0.5" euler="0 90 0">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# ============================================================
# Run all tests
# ============================================================

run_test("plane_box (plane_convex)", PLANE_BOX_XML)
run_test("plane_sphere", PLANE_SPHERE_XML)
run_test("plane_capsule", PLANE_CAPSULE_XML)
run_test("plane_ellipsoid", PLANE_ELLIPSOID_XML)
run_test("plane_cylinder", PLANE_CYLINDER_XML)
run_test("plane_cylinder (parallel)", PLANE_CYLINDER_PARALLEL_XML)
run_test("sphere_sphere", SPHERE_SPHERE_XML)
run_test("sphere_sphere (coincident)", SPHERE_SPHERE_COINCIDENT_XML, expect_zero_grad=True)
run_test("sphere_capsule", SPHERE_CAPSULE_XML)
run_test("capsule_capsule", CAPSULE_CAPSULE_XML)
run_test("capsule_capsule (parallel)", CAPSULE_CAPSULE_PARALLEL_XML)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  SUMMARY: {passed} passed, {failed} failed")
print(f"{'='*60}")
if failed > 0:
    sys.exit(1)
