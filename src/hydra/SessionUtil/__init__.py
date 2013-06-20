# -----------------------------------------------------------------------------
# Utility routines for saving and restoring extensions state with
# SimpleSession.
#

# -----------------------------------------------------------------------------
# The objecttree routines support writing trees of instances of classes.
# They write human-readable text unlike the standard python pickle module.
#
from .objecttree import instance_tree_to_basic_tree
from .objecttree import write_basic_tree
from .objecttree import basic_tree_to_instance_tree

# -----------------------------------------------------------------------------
# Classes for restoring the state of basic Chimera model types.
#
from .stateclasses import Model_State, Xform_State

# -----------------------------------------------------------------------------
#
def set_window_position(widget, geometry, minimum_pixels_onscreen = 100):

  # Remove window size from geometry spec
  s = [i for i in (geometry.find('+'), geometry.find('-')) if i > 0]
  if s:
    geometry = geometry[min(s):]
  if geometry and minimum_pixels_onscreen > 0:
    # Don't place window if it is close to being off-screen.
    sxy = geometry.replace('+', ' ').replace('-', ' ').strip().split()
    x, y = [int(s) for s in sxy]
    if (x + minimum_pixels_onscreen > widget.winfo_screenwidth() or
        y + minimum_pixels_onscreen > widget.winfo_screenheight()):
      return
  widget.autoposition = False   # tell baseDialog.py not to reposition
  widget.wm_geometry(geometry)
  
# -----------------------------------------------------------------------------
#
from numpy import int32, float32, intc, single as floatc
  
# -----------------------------------------------------------------------------
#
def array_to_string(a, dtype):

  if a is None:
    return None
  typename, shape, s = encoded_array(a, dtype)
  return s

# -----------------------------------------------------------------------------
#
def string_to_array(s, dtype, size2 = 0):

  if s is None:
    return None
  from base64 import b64decode
  from zlib import decompress
  from numpy import fromstring
  a = fromstring(decompress(b64decode(s)), dtype)
  if size2:
    n = len(a)/size2
    a = a.reshape((n,size2))
  return a

# -----------------------------------------------------------------------------
#
def encoded_array(a, dtype):

  if a is None:
    return None

  from base64 import b64encode
  from zlib import compress
  from numpy import ndarray, array
  ea = a if isinstance(a, ndarray) and a.dtype == dtype else array(a,dtype)
  sa = ea.tostring()
  s = b64encode(compress(sa))
  return (ea.dtype.name, tuple(ea.shape), s)

# -----------------------------------------------------------------------------
#
def decoded_array(ea):

  if ea is None:
    return None

  typename, shape, s = ea
  from numpy import dtype
  atype = dtype(typename)
  a = string_to_array(s, atype)
  a = a.reshape(shape)
  return a
