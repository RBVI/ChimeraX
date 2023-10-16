# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from numpy import uintp as cptr
string = object
pyobject = object

from numpy import empty, ndarray
import ctypes

class CFunctions:
    '''Access C functions from a shared library and create Python properties using these functions.'''

    def __init__(self, library_path):
        self._c_lib = ctypes_open(library_path)
        
    # -----------------------------------------------------------------------------
    #
    def c_function(self, func_name, args = None, ret = None):
        '''Look up a C function and set its argument types if they have not been set.'''
        f = getattr(self._c_lib, func_name)
        if args is not None and f.argtypes is None:
            f.argtypes = args
        if ret is not None:
            f.restype = ret
        return f

    # -----------------------------------------------------------------------------
    #
    def c_array_function(self, name, args=(), ret=None, per_object=True):
        '''
        Look up a C function and set its argument types if they have not been set.

        Array functions take an array of objects, the number of objects, optional
        arguments given as numpy types (if per_object, then arrays of those types),
        optional return value given as a numpy type that will be an array.
        '''
        if per_object:
            c_args = tuple(ctypes.POINTER(numpy_type_to_ctype[d]) for d in args)
        else:
            c_args = tuple(numpy_type_to_ctype[d] for d in args)
        if ret is None:
            ret_arg = ()
        else:
            ret_arg = (ctypes.POINTER(numpy_type_to_ctype[ret]),)
        args = (ctypes.c_void_p, ctypes.c_size_t) + c_args + ret_arg
        return self.c_function(name, args=args)
    
    # -----------------------------------------------------------------------------
    #
    def c_property(self, func_name, value_type, value_count = 1,
                   read_only = False, astype = None, doc = None):
        '''Create a Python property that calls a C library function using ctypes.'''

        if isinstance(value_count,str):
            return self.c_varlen_property(func_name, value_type, value_count, read_only, astype, doc)
        elif value_count > 1:
            return self.c_array_property(func_name, value_type, value_count, read_only, astype, doc)

        vtype = numpy_type_to_ctype[value_type]
        cget = self.c_array_function(func_name, ret=value_type)
        if vtype == ctypes.py_object:
            # Using ctypes py_object for value v make the value get an extra reference count.
            # The ctypes code looks like it is not intended to have C++ code set the pointer PyObject * value.
            # Use a one element numpy array to get the reference counting right.
            v = empty((1,), value_type)       # Numpy array return value
            v_ref = pointer(v)
            if astype is None:
                def get_prop(self):
                    cget(self._c_pointer_ref, 1, v_ref)
                    vv = v[0]
                    v[0] = None		# Avoid holding an extra reference to value
                    return vv
            else:
                def get_prop(self):
                    cget(self._c_pointer_ref, 1, v_ref)
                    vv = v[0]
                    v[0] = None		# Avoid holding an extra reference to value
                    return astype(vv)
        else:
            v = vtype()
            v_ref = ctypes.byref(v)
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
            cset = self.c_array_function('set_'+func_name, ret=value_type)
            if vtype == ctypes.py_object:
                def set_prop(self, value):
                    v[0] = value
                    cset(self._c_pointer_ref, 1, v_ref)
                    v[0] = None		# Avoid holding an extra reference to value
            else:
                def set_prop(self, value):
                    v.value = value
                    cset(self._c_pointer_ref, 1, v_ref)

        return property(get_prop, set_prop, doc = doc)
 
    # -----------------------------------------------------------------------------
    #
    def c_array_property(self, func_name, value_type, value_count,
                         read_only = False, astype = None, doc = None):
        '''
        Create a property which has a fixed length array value obtained by
        calling a C library function using ctypes.
        '''

        v = empty((value_count,), value_type)       # Numpy array return value
        v_ref = pointer(v)

        cget = self.c_array_function(func_name, ret=value_type)
        if astype is None:
            def get_prop(self):
                cget(self._c_pointer_ref, 1, v_ref)
                return v.copy()
        else:
            def get_prop(self):
                cget(self._c_pointer_ref, 1, v_ref)
                return astype(v.copy())

        if read_only:
            set_prop = None
        else:
            cset = self.c_array_function('set_'+func_name, ret=value_type)
            vtype = numpy_type_to_ctype[value_type]
            vs = (vtype*value_count)()      # ctypes array is faster than numpy
            def set_prop(self, value):
                vs[:] = value
                cset(self._c_pointer_ref, 1, vs)

        return property(get_prop, set_prop, doc = doc)
 
    # -----------------------------------------------------------------------------
    #
    def c_varlen_property(self, func_name, value_type, value_count,
                          read_only = False, astype = None, doc = None):
        '''
        Create a property which has a variable length array value obtained by
        calling a C library function using ctypes.
        '''

        cget = self.c_array_function(func_name, ret=value_type)
        def get_prop(self):
            vcount = getattr(self, value_count)
            v = empty((vcount,), value_type)       # Numpy array return value
            cget(self._c_pointer_ref, 1, pointer(v))
            return v if astype is None else astype(v)

        if read_only:
            set_prop = None
        else:
            cset = self.c_array_function('set_'+func_name, ret=value_type)
            def set_prop(self, value):
                vtype = numpy_type_to_ctype[value_type]
                vcount = getattr(self, value_count)
                vs = (vtype*vcount)()      # ctypes array is faster than numpy
                vs[:] = value
                cset(self._c_pointer_ref, 1, vs)

        return property(get_prop, set_prop, doc = doc)

    # -----------------------------------------------------------------------------
    #
    def cvec_property(self, func_name, value_type, value_count = 1,
                      read_only = False, astype = None, per_object = True, doc = None):
        '''
        Create a property that calls a C library function using ctypes taking an
        array of pointer to objects.
        '''

        cget = self.c_array_function(func_name, ret=value_type)
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
            cset = self.c_array_function('set_'+func_name, ret=value_type)
            def set_prop(self, values):
                n = len(self._pointer_array)
                vdim = 1 if value_count == 1 else 2
                if isinstance(values,ndarray) and values.ndim == vdim and values.flags.c_contiguous:
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
                    vshape = (n,) if value_count == 1 else (n,value_count)
                    va = empty(vshape, value_type)
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
#
def ctypes_open(lib_name):
    import os
    import sys
    if sys.platform.startswith('darwin'):
        exts = ['.dylib', '.so']
    elif sys.platform.startswith('linux'):
        exts = ['.so']
    elif sys.platform.startswith('win'):
        exts = ['.pyd', '.dll']
    for ext in exts:
        if os.path.isabs(lib_name):
            lib_path = lib_name
        else:
            lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        lib_path += ext
        if not os.path.exists(lib_path):
            continue
        lib = ctypes.PyDLL(lib_path)
        break
    else:
        raise RuntimeError("Unable to find '%s' shared library" % lib_name)
    return lib

# -----------------------------------------------------------------------------
# Map numpy array value types (numpy.dtype) to ctypes value types.
#
import numpy
numpy_type_to_ctype = {
    numpy.float64: ctypes.c_double,
    numpy.float32: ctypes.c_float,
    numpy.int32: ctypes.c_int,
    numpy.uint32: ctypes.c_uint,
    numpy.uint8: ctypes.c_uint8,
    numpy.uintp: ctypes.c_void_p,
    numpy.byte: ctypes.c_char,
    bool: ctypes.c_bool,
    numpy.bool_: ctypes.c_bool,
    numpy.object_: ctypes.py_object,
    object: ctypes.py_object,
}

# -----------------------------------------------------------------------------
# Map ctype value types to numpy array value types (numpy.dtype)
#
ctype_type_to_numpy = {
    ctypes.c_double: numpy.float64,
    ctypes.c_float: numpy.float32,
    ctypes.c_int: numpy.int32,
    ctypes.c_uint: numpy.uint32,
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

size_t = ctype_type_to_numpy[ctypes.c_size_t]   # numpy dtype for size_t

# -----------------------------------------------------------------------------
# Create ctypes pointer to numpy array data.
#
def pointer(a):
    cty = numpy_type_to_ctype[a.dtype.type]
    p = a.ctypes.data_as(ctypes.POINTER(cty))
    return p
