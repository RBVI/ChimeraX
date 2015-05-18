from numpy import object as string, uintp as cptr, object as pyobject

from numpy import empty, ndarray
from ctypes import byref

# -----------------------------------------------------------------------------
# Call a C function that takes an array of object pointers and returns an
# array of values.  This routine allocates the array of returned values.
#
def get_value(func_name, c_pointers, value_type, value_count = 1, per_object = True):

    cfunc = c_array_function(func_name, value_type)

    if isinstance(c_pointers, cptr):
        # Getting attribute for one object.
        _c_void_p_temp.value = int(c_pointers)
        if value_count == 1:
            # Single return value.
            v = ctype_temp[value_type]
            cfunc(byref(_c_void_p_temp), 1, byref(v))
            return v.value
        else:
            value = empty((value_count,), value_type)
            cfunc(byref(_c_void_p_temp), 1, pointer(value))
            return value
    else:
        # Getting an attribute for multiple objects.
        n = len(c_pointers)
        shape = (n,) if value_count == 1 or not per_object else (n,value_count)
        values = empty(shape, value_type)
        cfunc(pointer(c_pointers), n, pointer(values))
        return values

# -----------------------------------------------------------------------------
# Call a C function that sets an attribute for objects.  Takes an array of
# object pointers and array of values.
#
def set_value(func_name, c_pointers, values, value_type, value_count = 1):

    cfunc = c_array_function(func_name, value_type)

    if isinstance(c_pointers, cptr):
        # Optimize setting attribute for one object.
        _c_void_p_temp.value = int(c_pointers)
        if value_count == 1:
            vp = ctype_temp[value_type]
            vp.value = values
        elif isinstance(values,ndarray):
            vp = pointer(values)
        else:
            vtype = (value_type,value_count)
            vp = ctype_temp.get(vtype, None)
            if vp is None:
                ctype_temp[vtype] = vp = (numpy_type_to_ctype[value_type]*value_count)()
            vp[:] = values
        cfunc(byref(_c_void_p_temp), 1, vp)
        return

    # Setting attribute for multiple objects
    cp = pointer(c_pointers)
    n = len(c_pointers)

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

    cfunc(cp, n, v)

# -----------------------------------------------------------------------------
# Look up a C function and set its argument types if they have not been set.
#
def c_array_function(name, dtype):
    f = molc_function(name)
    if f.argtypes is None:
        f.restype = None
        import ctypes
        f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(numpy_type_to_ctype[dtype])]
    return f

# -----------------------------------------------------------------------------
#
_molc_lib = None
def molc_function(func_name, lib_name = 'libmolc.dylib'):
    global _molc_lib
    if _molc_lib is None:
        from os import path
        libpath = path.join(path.dirname(__file__), lib_name)
        from numpy import ctypeslib
        _molc_lib = ctypeslib.load_library(libpath, '.')
    f = getattr(_molc_lib, func_name)
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

ctype_temp = dict((nt,ct()) for nt,ct in numpy_type_to_ctype.items())
_c_void_p_temp = ctypes.c_void_p()

# -----------------------------------------------------------------------------
# Create ctypes pointer to numpy array data.
#
def pointer(a):
    cty = numpy_type_to_ctype[a.dtype.type]
    p = a.ctypes.data_as(ctypes.POINTER(cty))
    return p
