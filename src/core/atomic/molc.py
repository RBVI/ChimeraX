# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from numpy import object as string, uintp as cptr, object as pyobject

from numpy import empty, ndarray
import ctypes

# -----------------------------------------------------------------------------
# Create a property that calls a C library function using ctypes.
#
def c_property(func_name, value_type, value_count = 1, read_only = False, astype = None,
        doc = None):

    if isinstance(value_count,str):
        return c_varlen_property(func_name, value_type, value_count, read_only, astype, doc)
    elif value_count > 1:
        return c_array_property(func_name, value_type, value_count, read_only, astype, doc)

    vtype = numpy_type_to_ctype[value_type]
    v = vtype()
    v_ref = ctypes.byref(v)

    cget = c_array_function(func_name, ret=value_type)
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
        cset = c_array_function('set_'+func_name, ret=value_type)
        def set_prop(self, value):
            v.value = value
            cset(self._c_pointer_ref, 1, v_ref)

    return property(get_prop, set_prop, doc = doc)
 
# -----------------------------------------------------------------------------
# Create a property which has a fixed length array value obtained by
# calling a C library function using ctypes.
#
def c_array_property(func_name, value_type, value_count, read_only = False, astype = None,
        doc = None):

    v = empty((value_count,), value_type)       # Numpy array return value
    v_ref = pointer(v)

    cget = c_array_function(func_name, ret=value_type)
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
        cset = c_array_function('set_'+func_name, ret=value_type)
        vtype = numpy_type_to_ctype[value_type]
        vs = (vtype*value_count)()      # ctypes array is faster than numpy
        def set_prop(self, value):
            vs[:] = value
            cset(self._c_pointer_ref, 1, vs)

    return property(get_prop, set_prop, doc = doc)
 
# -----------------------------------------------------------------------------
# Create a property which has a variable length array value obtained by
# calling a C library function using ctypes.
#
def c_varlen_property(func_name, value_type, value_count, read_only = False, astype = None,
        doc = None):

    cget = c_array_function(func_name, ret=value_type)
    def get_prop(self):
        vcount = getattr(self, value_count)
        v = empty((vcount,), value_type)       # Numpy array return value
        cget(self._c_pointer_ref, 1, pointer(v))
        return v if astype is None else astype(v)

    if read_only:
        set_prop = None
    else:
        cset = c_array_function('set_'+func_name, ret=value_type)
        def set_prop(self, value):
            vtype = numpy_type_to_ctype[value_type]
            vcount = getattr(self, value_count)
            vs = (vtype*vcount)()      # ctypes array is faster than numpy
            vs[:] = value
            cset(self._c_pointer_ref, 1, vs)

    return property(get_prop, set_prop, doc = doc)

# -----------------------------------------------------------------------------
# Create a property that calls a C library function using ctypes taking an
# array of pointer to objects.
#
def cvec_property(func_name, value_type, value_count = 1, read_only = False, astype = None, per_object = True, doc = None):

    cget = c_array_function(func_name, ret=value_type)
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
        cset = c_array_function('set_'+func_name, ret=value_type)
        def set_prop(self, values):
            n = len(self._pointer_array)
            vdim = 1 if value_count == 1 else 2
            if isinstance(values,ndarray) and values.ndim == vdim:
                # Values are already specified as a numpy array.
                if len(values) != n:
                    raise ValueError('Values array length %d does not match objects array length %d'
                                     % (len(values), n))
                v = pointer(values)
            elif isinstance(values, bytearray):
                tmp_type = ctypes.c_char * n * value_count
                v = tmp_type.from_buffer(values)
                v = ctypes.cast(v, ctypes.POINTER(ctypes.c_char))
            else:
                # Allocate numpy array of values to pass to C.
                va = empty((n,value_count), value_type)
                va[:] = values
                v = pointer(va)
            cset(self._c_pointers, n, v)

    return property(get_prop, set_prop, doc = doc)

# -----------------------------------------------------------------------------
# Set the object C pointer used as the first argument of C get/set methods
# for that object.
#
def set_c_pointer(self, pointer):
    self._c_pointer = cp = ctypes.c_void_p(int(pointer))
    self._c_pointer_ref = ctypes.byref(cp)

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
# array functions take an array of objects, the number of objects, optional
# arguments given as numpy types (if per_object, then arrays of those types),
# optional return value given as a numpy type that will be an array.
def c_array_function(name, args=(), ret=None, per_object=True):
    if per_object:
        c_args = tuple(ctypes.POINTER(numpy_type_to_ctype[d]) for d in args)
    else:
        c_args = tuple(numpy_type_to_ctype[d] for d in args)
    if ret is None:
        ret_arg = ()
    else:
        ret_arg = (ctypes.POINTER(numpy_type_to_ctype[ret]),)
    args = (ctypes.c_void_p, ctypes.c_size_t) + c_args + ret_arg
    return c_function(name, args=args)

# -----------------------------------------------------------------------------
#
_molc_lib = None
def c_function(func_name, lib_name = 'libmolc', args = None, ret = None):
    global _molc_lib
    if _molc_lib is None:
        import os
        import sys
        if sys.platform.startswith('darwin'):
            exts = ['.dylib', '.so']
        elif sys.platform.startswith('linux'):
            exts = ['.so']
        else: # Windows
            exts = ['.pyd', '.dll']
        for ext in exts:
            if os.path.isabs(lib_name):
                lib_path = lib_name
            else:
                lib_path = os.path.join(os.path.dirname(__file__), lib_name)
            lib_path += ext
            if not os.path.exists(lib_path):
                continue
            _molc_lib = ctypes.PyDLL(lib_path)
            break
        else:
            raise RuntimeError("Unable to find '%s' shared library" % lib_name)
    f = getattr(_molc_lib, func_name)
    if args is not None and f.argtypes is None:
        f.argtypes = args
    if ret is not None:
        f.restype = ret
    return f

# -----------------------------------------------------------------------------
# Map numpy array value types (numpy.dtype) to ctypes value types.
#
import numpy
numpy_type_to_ctype = {
    numpy.float64: ctypes.c_double,
    numpy.float32: ctypes.c_float,
    numpy.int32: ctypes.c_int,
    numpy.uint8: ctypes.c_uint8,
    numpy.uintp: ctypes.c_void_p,
    numpy.byte: ctypes.c_char,
    numpy.bool: ctypes.c_bool,
    numpy.bool_: ctypes.c_bool,
    numpy.object_: ctypes.py_object,
    numpy.object: ctypes.py_object,
}

# -----------------------------------------------------------------------------
# Map ctype value types to numpy array value types (numpy.dtype)
#
ctype_type_to_numpy = {
    ctypes.c_double: numpy.float64,
    ctypes.c_float: numpy.float32,
    ctypes.c_int: numpy.int32,
    ctypes.c_uint8: numpy.uint8,
    ctypes.c_char: numpy.byte,
    ctypes.c_void_p: numpy.uintp,
    ctypes.c_uint8: numpy.uint8,
    ctypes.c_bool: numpy.bool_,
    ctypes.py_object: numpy.object_,
}

if ctypes.sizeof(ctypes.c_long) == 4:
    numpy_type_to_ctype[numpy.int64] = ctypes.c_longlong
elif ctypes.sizeof(ctypes.c_long) == 8:
    numpy_type_to_ctype[numpy.int64] = ctypes.c_long
else:
    raise RuntimeError("Only support 4- and 8-byte longs")
if ctypes.sizeof(ctypes.c_size_t) == 4:
    ctype_type_to_numpy[ctypes.c_size_t] = numpy.int32
elif ctypes.sizeof(ctypes.c_size_t) == 8:
    ctype_type_to_numpy[ctypes.c_size_t] = numpy.int64
else:
    raise RuntimeError("Only support 4- and 8-byte size_t")


# -----------------------------------------------------------------------------
# Create ctypes pointer to numpy array data.
#
def pointer(a):
    cty = numpy_type_to_ctype[a.dtype.type]
    p = a.ctypes.data_as(ctypes.POINTER(cty))
    return p
