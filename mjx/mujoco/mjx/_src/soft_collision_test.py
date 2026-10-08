"""Tests for differentiable soft collision detection in MJX."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_convex
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import collision_types
from mujoco.mjx._src import mesh
from mujoco.mjx._src import math
from mujoco.mjx._src import smooth
import numpy as np


_PLANE_BOX_XML = """
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

_PLANE_SPHERE_XML = """
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

_PLANE_CAPSULE_XML = """
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

_PLANE_ELLIPSOID_XML = """
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

_PLANE_CYLINDER_XML = """
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

_PLANE_CYLINDER_PARALLEL_XML = """
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

_SPHERE_SPHERE_XML = """
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

_SPHERE_SPHERE_COINCIDENT_XML = """
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

_SPHERE_CAPSULE_XML = """
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

_CAPSULE_CAPSULE_XML = """
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

_CAPSULE_CAPSULE_PARALLEL_XML = """
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

_BOX_BOX_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="box" size="0.2 0.15 0.1"/>
    <body pos="0.3 0.02 0.01">
      <freejoint/>
      <geom type="box" size="0.2 0.15 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_TEST_CASES = (
    ('plane_box', _PLANE_BOX_XML, False),
    ('plane_sphere', _PLANE_SPHERE_XML, False),
    ('plane_capsule', _PLANE_CAPSULE_XML, False),
    ('plane_ellipsoid', _PLANE_ELLIPSOID_XML, False),
    ('plane_cylinder', _PLANE_CYLINDER_XML, False),
    ('plane_cylinder_parallel', _PLANE_CYLINDER_PARALLEL_XML, False),
    ('sphere_sphere', _SPHERE_SPHERE_XML, False),
    ('sphere_sphere_coincident', _SPHERE_SPHERE_COINCIDENT_XML, True),
    ('sphere_capsule', _SPHERE_CAPSULE_XML, False),
    ('capsule_capsule', _CAPSULE_CAPSULE_XML, False),
    ('capsule_capsule_parallel', _CAPSULE_CAPSULE_PARALLEL_XML, False),
    ('box_box', _BOX_BOX_XML, False),
)


def _box(pos, size=(0.2, 0.15, 0.1), mat=None) -> collision_types.ConvexInfo:
  """Builds one unbatched box using the collider's established mesh path."""
  if mat is None:
    mat = jp.eye(3)
  info = collision_types.GeomInfo(
      jp.asarray(pos)[None], jp.asarray(mat)[None], jp.asarray(size)[None]
  )
  box = mesh.box(info)
  return box.replace(
      pos=box.pos[0],
      mat=box.mat[0],
      size=box.size[0],
      vert=box.vert[0],
      face=box.face[0],
  )


def _sort_rows(x):
  x = np.asarray(x)
  keys = np.round(x, decimals=5)
  return x[np.lexsort(keys.T[::-1])]


def _contact_set_error(x, y):
  x, y = np.asarray(x), np.asarray(y)
  return min(
      np.linalg.norm(x[list(order)] - y)
      for order in itertools.permutations(range(x.shape[0]))
  )


class SoftCollisionTest(parameterized.TestCase):

  @parameterized.product(case=_TEST_CASES, mode=('smooth', 'c2'))
  def test_model_st_contact_forward_values(self, case, mode):
    name, xml, _ = case
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)

    def contact(mode, st_enable):
      model = mx.replace(opt=mx.opt.replace(
          softjax_mode=mode, contact_st_enable=st_enable, pw_solimp=None
      ))
      result = jax.jit(collision_driver.collision)(model, dx)._impl.contact
      return result.dist, result.pos, result.frame

    hard = contact('hard', False)
    straight_through = contact(mode, True)
    for actual, expected in zip(straight_through, hard):
      np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5,
                                 err_msg=name)
    soft = contact(mode, False)
    for value in soft:
      self.assertTrue(np.all(np.isfinite(np.asarray(value))), name)
    # These fixtures lie inside a smoothing band.  Other fixtures can have
    # identical hard and soft values, especially with compact C2 smoothing.
    if mode == 'smooth' and name in (
        'plane_box', 'plane_capsule', 'plane_cylinder_parallel',
        'sphere_sphere', 'sphere_sphere_coincident', 'sphere_capsule', 'box_box'
    ):
      self.assertTrue(any(not np.allclose(a, b, atol=1e-5, rtol=1e-5)
                          for a, b in zip(soft, straight_through)), name)

    model = mx.replace(opt=mx.opt.replace(
        softjax_mode=mode, contact_st_enable=True, pw_solimp=None
    ))
    def contact_vector(qpos):
      data = smooth.kinematics(model, dx.replace(qpos=qpos))
      result = collision_driver.collision(model, data)._impl.contact
      return jp.concatenate([x.reshape(-1) for x in (
          result.dist, result.pos, result.frame
      )])

    tangent = jp.full_like(dx.qpos, 0.01)
    derivative = jax.jit(lambda qpos: jax.jvp(
        contact_vector, (qpos,), (tangent,)
    )[1])(dx.qpos)
    self.assertTrue(np.all(np.isfinite(np.asarray(derivative))), name)

  @parameterized.parameters('smooth', 'c2')
  def test_frame_st_preserves_surrogate_gradients(self, mode):
    direction = jp.array([0.8, 0.49, 0.3])
    def loss(a, selected_mode, st_enable):
      return jp.sum(math.make_frame_soft(a, selected_mode,
                                         st_enable=st_enable))
    st_grad = jax.jit(jax.grad(lambda a: loss(a, mode, True)))(direction)
    hard_grad = jax.grad(lambda a: loss(a, 'hard', False))(direction)
    self.assertTrue(np.all(np.isfinite(np.asarray(st_grad))))
    self.assertFalse(np.allclose(st_grad, hard_grad))

  def test_box_box_sat_axes(self):
    box_a, box_b = _box((0.0, 0.0, 0.0)), _box((0.3, 0.0, 0.0))
    axes, degenerate = collision_convex._box_box_axes_soft(
        box_a.face_normal,
        box_b.face_normal,
        jp.eye(3),
        jp.eye(3),
    )

    self.assertEqual(axes.shape, (15, 3))
    self.assertEqual(int(jp.sum(degenerate)), 3)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(axes[~degenerate]), axis=1), 1.0
    )

  @parameterized.parameters(
      ((0.0, 0.0, -0.1), 0.5),
      ((0.75, 0.0, -0.1), 0.5),
      ((2.0, 0.0, -0.1), 0.5),
  )
  def test_box_box_soft_clipping_matches_hard(self, offset, size):
    clipping_poly = jp.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ])
    subject_poly = clipping_poly * size + jp.asarray(offset)
    normal = jp.array([0.0, 0.0, 1.0])
    hard_pts, hard_mask = collision_convex._clip(
        clipping_poly, subject_poly, normal, normal
    )
    soft_pts, soft_mask = collision_convex._clip_soft(
        clipping_poly,
        subject_poly,
        normal,
        normal,
        'c2',
        1e-6,
        jp.array(1.0),
    )
    np.testing.assert_allclose(soft_pts, hard_pts, atol=1e-5)
    np.testing.assert_allclose(soft_mask, hard_mask, atol=1e-5)

  def test_box_box_soft_manifold_points_match_hard(self):
    poly = jp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.3, 1.5, 0.0],
        [-0.2, 0.4, 0.0],
    ])
    mask = jp.ones(poly.shape[0], dtype=bool)
    normal = jp.array([0.0, 0.0, 1.0])
    hard = collision_convex._manifold_points(poly, mask, normal)
    soft = collision_convex._manifold_points_soft(
        poly, mask, normal, 'c2', 1e-6
    )
    np.testing.assert_array_equal(jp.argmax(soft, axis=1), hard)

  @parameterized.parameters('smooth', 'c2')
  def test_box_box_converges_to_hard(self, mode):
    angle = 0.23
    mat = jp.array([
        [jp.cos(angle), -jp.sin(angle), 0.0],
        [jp.sin(angle), jp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    box_a = _box((0.0, 0.0, 0.0))
    box_b = _box((0.31, 0.04, 0.03), mat=mat)
    hard = collision_convex._box_box(box_a, box_b)
    loose = collision_convex._box_box_soft(
        box_a, box_b, mode, softness=1e-2, sat_softness=1e-2)
    tight = collision_convex._box_box_soft(
        box_a, box_b, mode, softness=1e-6, sat_softness=1e-6)

    loose_error = np.linalg.norm(np.sort(loose[0]) - np.sort(hard[0]))
    tight_error = np.linalg.norm(np.sort(tight[0]) - np.sort(hard[0]))
    self.assertLess(tight_error, loose_error)
    self.assertLess(
        np.linalg.norm(np.asarray(tight[2]) - np.asarray(hard[2])),
        np.linalg.norm(np.asarray(loose[2]) - np.asarray(hard[2])),
    )
    self.assertLess(
        _contact_set_error(tight[1], hard[1]),
        _contact_set_error(loose[1], hard[1]),
    )
    np.testing.assert_allclose(np.sort(tight[0]), np.sort(hard[0]), atol=1e-4)
    np.testing.assert_allclose(tight[2], hard[2], atol=1e-4)
    np.testing.assert_allclose(
        _sort_rows(tight[1]), _sort_rows(hard[1]), atol=1e-4
    )

  def test_box_box_operational_softness_calibration(self):
    angle = 0.23
    mat = jp.array([
        [jp.cos(angle), -jp.sin(angle), 0.0],
        [jp.sin(angle), jp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    box_a = _box((0.0, 0.0, 0.0))
    box_b = _box((0.31, 0.04, 0.03), mat=mat)
    hard = collision_convex._box_box(box_a, box_b)
    tuned = collision_convex._box_box_soft(box_a, box_b, 'smooth')
    explicit = collision_convex._box_box_soft(
        box_a,
        box_b,
        'smooth',
        softness=collision_convex._BOX_BOX_SOFTNESS,
        sat_softness=collision_convex._BOX_BOX_SAT_SOFTNESS,
    )
    biased = collision_convex._box_box_soft(
        box_a, box_b, 'smooth', softness=1e-3, sat_softness=1e-3
    )

    self.assertEqual(collision_convex._BOX_BOX_SOFTNESS, 2e-5)
    self.assertEqual(collision_convex._BOX_BOX_SAT_SOFTNESS, 7e-5)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, tuned, explicit
    )
    np.testing.assert_allclose(tuned[0], hard[0], atol=1e-4)
    np.testing.assert_allclose(tuned[2], hard[2], atol=1e-4)
    np.testing.assert_allclose(
        _sort_rows(tuned[1]), _sort_rows(hard[1]), atol=1e-4
    )
    self.assertGreater(
        _contact_set_error(biased[1], hard[1]),
        _contact_set_error(tuned[1], hard[1]) + 1e-3,
    )

    # At a nearby SAT feature transition, the operational value supplies a
    # useful gradient where the 1e-6 hard-limit value has already saturated.
    def normal_y(offset, softness):
      moving = _box((0.31, offset, 0.03), mat=mat)
      return collision_convex._box_box_soft(
          box_a, moving, 'smooth', sat_softness=softness
      )[2][0, 1]

    hard_limit_grad = jax.grad(
        lambda offset: normal_y(offset, 1e-6)
    )(jp.array(0.0352))
    tuned_grad = jax.grad(
        lambda offset: normal_y(
            offset, collision_convex._BOX_BOX_SAT_SOFTNESS
        )
    )(jp.array(0.0352))
    self.assertLess(abs(float(hard_limit_grad)), 1e-5)
    self.assertGreater(abs(float(tuned_grad)), 1.0)

  def test_box_box_tie_priority_and_edge_padding(self):
    idx = collision_convex._soft_select(jp.zeros(4), jp.ones(4), 'c2', 1e-6)
    np.testing.assert_array_equal(idx, jp.array([1.0, 0.0, 0.0, 0.0]))
    tie_grad = jax.grad(
        lambda score: jp.dot(
            collision_convex._soft_select(
                score, jp.ones(4), 'c2', 1e-6
            ),
            jp.arange(4.0),
        )
    )(jp.zeros(4))
    self.assertTrue(np.all(np.isfinite(np.asarray(tie_grad))))

    x, y, z = jp.deg2rad(jp.array([20.0, 30.0, 10.0]))
    rot_x = jp.array([
        [1.0, 0.0, 0.0],
        [0.0, jp.cos(x), -jp.sin(x)],
        [0.0, jp.sin(x), jp.cos(x)],
    ])
    rot_y = jp.array([
        [jp.cos(y), 0.0, jp.sin(y)],
        [0.0, 1.0, 0.0],
        [-jp.sin(y), 0.0, jp.cos(y)],
    ])
    rot_z = jp.array([
        [jp.cos(z), -jp.sin(z), 0.0],
        [jp.sin(z), jp.cos(z), 0.0],
        [0.0, 0.0, 1.0],
    ])
    box_a = _box((0.0, 0.0, 0.0))
    box_b = _box((0.25, 0.2, 0.14), mat=rot_z @ rot_y @ rot_x)
    dist, _, _ = collision_convex._box_box_soft(box_a, box_b, 'c2')
    self.assertLess(float(dist[0]), 0.0)
    self.assertTrue(np.all(np.asarray(dist[1:]) > 0.0))

    separated = _box((1.0, 0.0, 0.0))
    separated_dist, _, _ = collision_convex._box_box_soft(
        box_a, separated, 'c2'
    )
    self.assertTrue(np.all(np.asarray(separated_dist) > 0.0))

  @parameterized.parameters('smooth', 'c2')
  def test_box_box_jit_vmap_and_gradients(self, mode):
    box_a = _box((0.0, 0.0, 0.0))
    box_b = _box((0.3, 0.02, 0.01))

    def collide(pos):
      return collision_convex._box_box_soft(box_a.replace(pos=pos), box_b, mode)

    eager = collide(box_a.pos)
    compiled = jax.jit(collide)(box_a.pos)
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y), eager, compiled
    )
    batched = jax.vmap(collide)(
        jp.stack([box_a.pos, box_a.pos + jp.array([0.01, 0.0, 0.0])])
    )
    self.assertEqual(batched[0].shape, (2, 4))

    grad = jax.grad(lambda pos: jp.sum(collide(pos)[0]))(box_a.pos)
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))
    self.assertFalse(np.allclose(grad, 0.0))
    eps = 1e-4
    direction = jp.array([1.0, 0.0, 0.0])
    finite_difference = (
        jp.sum(collide(box_a.pos + eps * direction)[0])
        - jp.sum(collide(box_a.pos - eps * direction)[0])
    ) / (2.0 * eps)
    np.testing.assert_allclose(
        jp.dot(grad, direction), finite_difference, atol=2e-2, rtol=2e-2
    )

  def test_box_box_hard_mode_parity(self):
    m = mujoco.MjModel.from_xml_string(_BOX_BOX_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)

    hard_none = collision_driver.collision(mx, dx)._impl.contact
    mx_hard = mx.replace(opt=mx.opt.replace(softjax_mode='hard'))
    hard_mode = collision_driver.collision(mx_hard, dx)._impl.contact
    np.testing.assert_array_equal(hard_mode.dist, hard_none.dist)
    np.testing.assert_array_equal(hard_mode.pos, hard_none.pos)
    np.testing.assert_array_equal(hard_mode.frame, hard_none.frame)

    for mode in ('smooth', 'c2'):
      mx_soft = mx.replace(opt=mx.opt.replace(softjax_mode=mode))
      contact = collision_driver.collision(mx_soft, dx)._impl.contact
      self.assertTrue(np.all(np.isfinite(np.asarray(contact.frame))))
      np.testing.assert_allclose(
          np.linalg.norm(np.asarray(contact.frame[:, 0]), axis=1),
          1.0,
          atol=1e-5,
      )

  def test_box_box_symmetry_and_rigid_transform(self):
    box_a = _box((0.0, 0.0, 0.0))
    box_b = _box((0.3, 0.02, 0.01))
    dist, pos, normal = collision_convex._box_box_soft(box_a, box_b, 'c2')
    rev_dist, rev_pos, rev_normal = collision_convex._box_box_soft(
        box_b, box_a, 'c2'
    )
    np.testing.assert_allclose(np.sort(rev_dist), np.sort(dist), atol=1e-5)
    np.testing.assert_allclose(_sort_rows(rev_pos), _sort_rows(pos), atol=1e-5)
    np.testing.assert_allclose(rev_normal, -normal, atol=1e-5)

    angle = 0.4
    rotation = jp.array([
        [jp.cos(angle), -jp.sin(angle), 0.0],
        [jp.sin(angle), jp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = jp.array([0.7, -0.2, 0.5])

    def transform(box):
      return box.replace(
          pos=translation + rotation @ box.pos,
          mat=rotation @ box.mat,
      )

    transformed = collision_convex._box_box_soft(
        transform(box_a), transform(box_b), 'c2'
    )
    np.testing.assert_allclose(transformed[0], dist, atol=1e-5)
    np.testing.assert_allclose(
        transformed[1], translation + pos @ rotation.T, atol=1e-5
    )
    np.testing.assert_allclose(transformed[2], normal @ rotation.T, atol=1e-5)

  def test_box_box_jvp_vjp_duality(self):
    box_a = _box((0.0, 0.0, 0.0))
    box_b = _box((0.3, 0.02, 0.01))

    def contact_vector(pos):
      contact = collision_convex._box_box_soft(
          box_a.replace(pos=pos), box_b, 'c2'
      )
      return jp.concatenate([x.reshape(-1) for x in contact])

    tangent = jp.array([0.2, -0.1, 0.3])
    output, jvp = jax.jvp(contact_vector, (box_a.pos,), (tangent,))
    cotangent = jp.linspace(-0.2, 0.3, output.size)
    _, pullback = jax.vjp(contact_vector, box_a.pos)
    vjp = pullback(cotangent)[0]
    np.testing.assert_allclose(
        jp.dot(cotangent, jvp), jp.dot(tangent, vjp), atol=1e-5
    )

  def test_box_box_short_step_rollout(self):
    m = mujoco.MjModel.from_xml_string(_BOX_BOX_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    mx = mx.replace(opt=mx.opt.replace(softjax_mode='c2'))
    dx = mjx.put_data(m, d)

    def rollout(qpos):
      data = dx.replace(qpos=qpos)
      for _ in range(2):
        data = mjx.step(mx, data)
      return data.qpos

    terminal_loss, grad = jax.value_and_grad(
        lambda qpos: jp.sum(rollout(qpos))
    )(dx.qpos)
    self.assertTrue(np.isfinite(np.asarray(terminal_loss)))
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))

  @parameterized.parameters(*_TEST_CASES)
  def test_soft_collision_dist_and_gradient(
      self, name, xml, expect_zero_grad
  ):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    for mode in ('hard', 'smooth', 'c2'):
      mx_soft = mx.replace(opt=mx.opt.replace(softjax_mode=mode))
      dx_soft = jax.jit(collision_driver.collision)(mx_soft, dx)
      dist = np.asarray(dx_soft._impl.contact.dist)
      self.assertFalse(
          np.any(np.isnan(dist)),
          f'{name} softjax_mode={mode} produced NaN distance',
      )

    mx_soft = mx.replace(opt=mx.opt.replace(softjax_mode='c2'))

    def soft_collision_loss(qpos):
      dx_mod = dx.replace(qpos=qpos)
      dx_mod = smooth.kinematics(mx_soft, dx_mod)
      dx_mod = smooth.com_pos(mx_soft, dx_mod)
      dx_mod = collision_driver.collision(mx_soft, dx_mod)
      return jp.sum(dx_mod._impl.contact.dist)

    grad = np.asarray(jax.jit(jax.grad(soft_collision_loss))(dx.qpos))
    self.assertFalse(np.any(np.isnan(grad)), f'{name} gradient has NaNs')

    if expect_zero_grad:
      np.testing.assert_allclose(grad, 0)
    else:
      self.assertFalse(np.allclose(grad, 0), f'{name} gradient is all zero')


if __name__ == '__main__':
  absltest.main()
