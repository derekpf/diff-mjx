"""SoftJAX contact operations with optional straight-through evaluation."""

import functools

from jax import numpy as jp
import softjax as _softjax


def _with_st_enable(fn, st_fn):
  """Selects the native hard-forward, soft-gradient implementation."""

  @functools.wraps(fn)
  def select(*args, st_enable: bool = False, **kwargs):
    return (st_fn if st_enable else fn)(*args, **kwargs)

  return select


def _clip(x, a, b, softness=0.1, mode='smooth', gated=False):
  """Native clip with finite C2 polynomial evaluation at small temperatures."""
  if mode != 'c2':
    return _softjax.clip(x, a, b, softness=softness, mode=mode, gated=gated)

  def softrelu(value):
    # C2 is constant/linear outside [-5s, 5s]. Its unused sixth-degree
    # polynomial can overflow in autodiff if evaluated on value/s directly.
    radius = 5.0 * softness
    # Unit-temperature evaluation also avoids large inverse powers of s in
    # the compiled polynomial VJP, even when its primal input is bounded.
    bounded = jp.clip(value / softness, -5.0, 5.0)
    curved = softness * _softjax.softrelu(
        bounded, softness=1.0, mode=mode, gated=gated)
    return jp.where(value < -radius, 0.0,
                    jp.where(value > radius, value, curved))

  return a + softrelu(x - a) - softrelu(x - b)


abs = _with_st_enable(_softjax.abs, _softjax.abs_st)
argmax = _with_st_enable(_softjax.argmax, _softjax.argmax_st)
argmin = _with_st_enable(_softjax.argmin, _softjax.argmin_st)
clip = _with_st_enable(_clip, _softjax.st(_clip))
greater = _with_st_enable(_softjax.greater, _softjax.greater_st)
greater_equal = _with_st_enable(_softjax.greater_equal,
                                _softjax.greater_equal_st)
less = _with_st_enable(_softjax.less, _softjax.less_st)
less_equal = _with_st_enable(_softjax.less_equal, _softjax.less_equal_st)
max = _with_st_enable(_softjax.max, _softjax.max_st)
min = _with_st_enable(_softjax.min, _softjax.min_st)
relu = _with_st_enable(_softjax.relu, _softjax.relu_st)
sign = _with_st_enable(_softjax.sign, _softjax.sign_st)

# These operations have no mode argument.  They propagate hard forward values
# and surrogate gradients from SoftBool and SoftIndex inputs unchanged.
any = _softjax.any
div = _softjax.div
dynamic_index_in_dim = _softjax.dynamic_index_in_dim
logical_and = _softjax.logical_and
logical_not = _softjax.logical_not
norm = _softjax.norm
sqrt = _softjax.sqrt
where = _softjax.where
st = _softjax.st
top_k = _softjax.top_k
take = _softjax.take
