from numpy import object as string, uintp as cptr, object as pyobject

# -----------------------------------------------------------------------------
# Call a C function that takes an array of object pointers and returns an
# array of values.  This routine allocates the array of returned values.
#
def get_value(func_name, c_pointers, value_type, value_count = None, per_object = True):
    from numpy import ndarray, empty
    single = isinstance(c_pointers, cptr)
    if single:
        cp = empty((1,), cptr)
        cp[0] = c_pointers
        c_pointers = cp
    cfunc = c_array_function(func_name, value_type)
    n = len(c_pointers)
    values = empty((n,), value_type) if value_count is None or not per_object else empty((n,value_count), value_type)
    cfunc(pointer(c_pointers), n, pointer(values))
    return values[0] if single else values

# -----------------------------------------------------------------------------
# Call a C function that sets an attribute for objects.  Takes an array of
# object pointers and array of values.
#
def set_value(func_name, c_pointers, values, value_type, value_count = None):
    from numpy import ndarray, empty
    single = isinstance(c_pointers, cptr)
    if single:
        cp = empty((1,), cptr)
        cp[0] = c_pointers
        c_pointers = cp
        v = empty((1,value_count), value_type)
        v[:] = values
        values = v

    n = len(c_pointers)
    if not isinstance(values, ndarray) or (values.ndim == 1 and not value_count is None):
        # Expand single value to an array.
        shape = (n,) if value_count is None else (n,value_count)
        v = numpy.empty(shape, value_type)
        v[:] = values
        values = v

    if len(values) != n:
        raise ValueError('Values array length %d does not match objects array length %d'
                         % (len(values), n))
    cfunc = c_array_function(func_name, value_type)
    cfunc(pointer(c_pointers), n, pointer(values))

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
    numpy.object_: ctypes.c_void_p,
    numpy.object: ctypes.c_void_p,
}

# -----------------------------------------------------------------------------
# Create ctypes pointer to numpy array data.
#
def pointer(a):
    from numpy import float32, int32, uint8
    t = a.dtype.type
    cty = numpy_type_to_ctype.get(t, None)
    if cty is None:
        raise ValueError('Converting numpy array of type %s to ctypes pointer not supported' % str(t))
    p = a.ctypes.data_as(ctypes.POINTER(cty))
    return p
