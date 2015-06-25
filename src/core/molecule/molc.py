from numpy import object as string, uintp as cptr, object as pyobject

from numpy import empty, ndarray
from ctypes import byref

# -----------------------------------------------------------------------------
# Create a property that calls a C library function using ctypes.
#
def c_property(func_name, value_type, value_count = 1, read_only = False, astype = None):

    if isinstance(value_count,str):
        return c_varlen_property(func_name, value_type, value_count, read_only, astype)
    elif value_count > 1:
        return c_array_property(func_name, value_type, value_count, read_only, astype)

    vtype = numpy_type_to_ctype[value_type]
    v = vtype()
    v_ref = byref(v)

    cget = c_array_function(func_name, value_type)
    if astype is None:
        def get_prop(self):
            cget(self._c_pointer_ref, 1, v_ref)
            return v.value
    else:
        def get_prop(self):
            cget(self._c_pointer_ref, 1, v_ref)
            return astype(v.value)

    if read_only:
        set_prop = None
    else:
        cset = c_array_function('set_'+func_name, value_type)
        def set_prop(self, value):
            v.value = value
            cset(self._c_pointer_ref, 1, v_ref)

    return property(get_prop, set_prop)
 
# -----------------------------------------------------------------------------
# Create a property which has a fixed length array value obtained by
# calling a C library function using ctypes.
#
def c_array_property(func_name, value_type, value_count, read_only = False, astype = None):

    v = empty((value_count,), value_type)       # Numpy array return value
    v_ref = pointer(v)

    cget = c_array_function(func_name, value_type)
    if astype is None:
        def get_prop(self):
            cget(self._c_pointer_ref, 1, v_ref)
            return v.copy()
    else:
        def get_prop(self):
            cget(self._c_pointer_ref, 1, v_ref)
            return astype(v)

    if read_only:
        set_prop = None
    else:
        cset = c_array_function('set_'+func_name, value_type)
        vtype = numpy_type_to_ctype[value_type]
        vs = (vtype*value_count)()      # ctypes array is faster than numpy
        def set_prop(self, value):
            vs[:] = value
            cset(self._c_pointer_ref, 1, vs)

    return property(get_prop, set_prop)
 
# -----------------------------------------------------------------------------
# Create a property which has a variable length array value obtained by
# calling a C library function using ctypes.
#
def c_varlen_property(func_name, value_type, value_count, read_only = False, astype = None):

    cget = c_array_function(func_name, value_type)
    def get_prop(self):
        vcount = getattr(self, value_count)
        v = empty((vcount,), value_type)       # Numpy array return value
        cget(self._c_pointer_ref, 1, pointer(v))
        return v if astype is None else astype(v)

    if read_only:
        set_prop = None
    else:
        cset = c_array_function('set_'+func_name, value_type)
        def set_prop(self, value):
            vtype = numpy_type_to_ctype[value_type]
            vcount = getattr(self, value_count)
            vs = (vtype*vcount)()      # ctypes array is faster than numpy
            vs[:] = value
            cset(self._c_pointer_ref, 1, vs)

    return property(get_prop, set_prop)

# -----------------------------------------------------------------------------
# Create a property that calls a C library function using ctypes taking an
# array of pointer to objects.
#
def cvec_property(func_name, value_type, value_count = 1, read_only = False, astype = None, per_object = True):

    cget = c_array_function(func_name, value_type)
    def get_prop(self):
        # Get an attribute for multiple objects.
        n = len(self._pointer_array)
        vc = getattr(self,value_count) if isinstance(value_count,str) else value_count
        shape = ((n,) if vc == 1 or not per_object else (n,vc)) if per_object else (vc.sum(),)
        values = empty(shape, value_type)
        cget(self._c_pointers, n, pointer(values))
        return values if astype is None else astype(values)

    if read_only:
        set_prop = None
    else:
        cset = c_array_function('set_'+func_name, value_type)
        def set_prop(self, values):
            n = len(self._pointer_array)
            vdim = 1 if value_count == 1 else 2
            if isinstance(values,ndarray) and values.ndim == vdim:
                # Values are already specified as a numpy array.
                if len(values) != n:
                    raise ValueError('Values array length %d does not match objects array length %d'
                                     % (len(values), n))
                v = pointer(values)
            else:
                # Allocate numpy array of values to pass to C.
                va = empty((n,value_count), value_type)
                va[:] = values
                v = pointer(va)
            cset(self._c_pointers, n, v)

    return property(get_prop, set_prop)

# -----------------------------------------------------------------------------
# Set the object C pointer used as the first argument of C get/set methods
# for that object.
#
def set_c_pointer(self, pointer):
    self._c_pointer = cp = ctypes.c_void_p(int(pointer))
    self._c_pointer_ref = byref(cp)

# -----------------------------------------------------------------------------
# Set the object C pointer used as the first argument of C get/set methods
# for that object.
#
def set_cvec_pointer(self, pointers):
    self._c_pointers = pointer(pointers)
    self._pointer_array = pointers

# -----------------------------------------------------------------------------
# Look up a C function and set its argument types if they have not been set.
#
def c_array_function(name, dtype):
    import ctypes
    args = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(numpy_type_to_ctype[dtype]))
    return c_function(name, args = args)

# -----------------------------------------------------------------------------
#
_molc_lib = None
def c_function(func_name, lib_name = 'libmolc', args = None, ret = None):
    global _molc_lib
    if _molc_lib is None:
        from os import path
        libpath = path.join(path.dirname(__file__), lib_name)
        from numpy import ctypeslib
        _molc_lib = ctypeslib.load_library(libpath, '.')
    f = getattr(_molc_lib, func_name)
    if not args is None and f.argtypes is None:
        f.argtypes = args
        f.restype = ret
    return f

# -----------------------------------------------------------------------------
# Map numpy array value types (numpy.dtype) to ctypes value types.
#
import numpy, ctypes
numpy_type_to_ctype = {
    numpy.float64: ctypes.c_double,
    numpy.float32: ctypes.c_float,
    numpy.int32: ctypes.c_int,
    numpy.uint8: ctypes.c_uint8,
    numpy.uintp: ctypes.c_void_p,
    numpy.bool: ctypes.c_uint8,
    numpy.bool_: ctypes.c_uint8,
    numpy.object_: ctypes.py_object,
    numpy.object: ctypes.py_object,
}

# -----------------------------------------------------------------------------
# Create ctypes pointer to numpy array data.
#
def pointer(a):
    cty = numpy_type_to_ctype[a.dtype.type]
    p = a.ctypes.data_as(ctypes.POINTER(cty))
    return p
