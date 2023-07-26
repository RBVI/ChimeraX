# coding: utf-8
# vim: set expandtab shiftwidth=4 softtabstop=4:
# cython: language_level=3, embedsignature=True
# distutils: language=c++

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

from msgpack import Packer, ExtType, Unpacker, unpackb
import numpy
from collections import OrderedDict, deque
from datetime import datetime, timezone, timedelta
from PIL import Image
from .session import _UniqueName
from .state import FinalizedState
import tinyarray
from tinyarray import ndarray_int, ndarray_float, ndarray_complex

from cpython.tuple cimport (
    PyTuple_New, PyTuple_SetItem
)
from cpython.ref cimport (
    Py_INCREF
)

from libcpp.string cimport string

# default maximum lenth is 1024*1024 in Python msgpack package
MSGPACK_MAX_LEN = 2 ** 32 - 1

#
# Encoding and decoding msgpack extension types using in serialization

PRIMITIVE_TYPES = frozenset((
    # supported types natively by msgpack
    bool, int, float, bytes, bytearray, str, dict, list, memoryview, type(None),
    # additionally supported types
    complex, tuple, range, _UniqueName,
    numpy.ndarray, numpy.number, numpy.bool_, numpy.bool8,
    set, frozenset, deque, OrderedDict,
    datetime, timedelta, timezone,
    Image.Image, FinalizedState,
    ndarray_int, ndarray_float, ndarray_complex,
))


cdef string _encode_unique_name(object un):
    # Return byte representation for serialization
    class_name = un.uid[0]
    cdef unsigned long long ordinal = un.uid[1]
    cdef string result
    cdef size_t num_bytes
    cdef size_t len_bn
    cdef size_t len_cn
    cdef char buf[8]
    if ordinal == 0:
        buf[0] = 0
        num_bytes = 1
    else:
        num_bytes = 0
        while ordinal > 0:
            buf[num_bytes] = ordinal & 0xff
            ordinal >>= 8
            num_bytes += 1
    if isinstance(class_name, str):
        cn = bytes(class_name, 'utf-8')
        len_cn = len(cn)
        result.reserve(3 + len_cn + num_bytes)
        result.push_back(0)   # tag
        result.push_back(len_cn)
        result.push_back(num_bytes)
        result.append(cn)
        result.append(buf, num_bytes)
    else:
        bn = bytes(class_name[0], 'utf-8')
        len_bn = len(bn)
        cn = bytes(class_name[1], 'utf-8')
        len_cn = len(cn)
        result.reserve(4 + len_bn + len_cn + num_bytes);
        result.push_back(1)  # tag
        result.push_back(len_bn)
        result.push_back(len_cn)
        result.push_back(num_bytes)
        result.append(bn)
        result.append(cn)
        result.append(buf, num_bytes)
    return result


cdef object _decode_unique_name(bytes buf):
    # restore _UniqueName from serialized representation
    cdef unsigned char len_bn, len_ch, num_bytes
    cdef size_t ordinal = 0
    cdef size_t offset
    cdef bytes bundle_name
    cdef bytes class_name
    cdef size_t i
    cdef int shift = 0
    if buf[0] == 0:
        len_cn, num_bytes = buf[1:3]
        offset = 3 + len_cn
        class_name = buf[3:offset]
        for i in range(num_bytes):
            ordinal |= (buf[offset + i] << shift)
            shift += 8
        uid = (class_name.decode(), ordinal)
        return _UniqueName(uid)
    else:
        # assert buf[0] == 1
        len_bn, len_cn, num_bytes = buf[1:4]
        offset2 = 4 + len_bn
        bundle_name = buf[4:offset2]
        offset = offset2 + len_cn
        class_name = buf[offset2:offset]
        for i in range(num_bytes):
            ordinal |= (buf[offset + i] << shift)
            shift += 8
        uid = ((bundle_name.decode(), class_name.decode()), ordinal)
        return _UniqueName(uid)


cdef dict _encode_ndarray(object o):
    # inspired by msgpack-numpy package
    if o.dtype.kind == 'V':
        # structured array
        kind = b'V'
        dtype = o.dtype.descr
    else:
        kind = b''
        dtype = o.dtype.str
    if 'O' in dtype:
        raise TypeError("Serializing numpy arrays of objects is not supported")
    return {
        b'kind': kind,
        b'dtype': dtype,
        b'shape': list(o.shape),
        b'data': o.tobytes()
    }


cdef object _decode_ndarray(dict data):
    kind = data[b'kind']
    dtype = data[b'dtype']
    if kind == b'V':
        # dtype = [tuple(str(t) for t in d) for d in dtype]
        tmp = []
        for d in dtype:
            tmp.append(tuple(str(t) for t in d))
        dtype = tmp
    return numpy.frombuffer(data[b'data'], numpy.dtype(dtype)).reshape(data[b'shape'])


cdef bytes _encode_image(object img):
    import io
    stream = io.BytesIO()
    img.save(stream, format='PNG')
    data = stream.getvalue()
    stream.close()
    return data


cdef object _decode_image(bytes data):
    import io
    stream = io.BytesIO(data)
    img = Image.open(stream)
    img.load()
    return img


cdef dict _encode_numpy_number(object o):
    return {
        b'dtype': o.dtype.str,
        b'data': o.tobytes()
    }


cdef object _decode_numpy_number(dict data):
    return numpy.frombuffer(data[b'data'], numpy.dtype(data[b'dtype']))[0]


cdef _decode_datetime(str data):
    from dateutil.parser import parse
    return parse(data)


cdef bytes _pack_as_array(object obj):
    # save as msgpack array without converting to list first
    packer = Packer(**_packer_args, autoreset=False)
    n = len(obj)
    packer.pack_array_header(n)
    for o in obj:
        packer.pack(o)
    return packer.bytes()


cdef object _encode_ext(object obj):
    # Return msgpack extension type, limited to 0-127
    # In simple session test: # of tuples > # of UniqueNames > # of numpy arrays > the rest
    if isinstance(obj, tuple):
        # restored as a tuple
        return ExtType(12, _pack_as_array(obj))
    if isinstance(obj, _UniqueName):
        return ExtType(0, _encode_unique_name(obj))
    if isinstance(obj, numpy.ndarray):
        # handle numpy array subclasses
        packer = Packer(**_packer_args)
        return ExtType(1, packer.pack(_encode_ndarray(obj)))
    if isinstance(obj, complex):
        # restored as a tuple
        packer = Packer(**_packer_args)
        return ExtType(2, packer.pack([obj.real, obj.imag]))
    if isinstance(obj, set):
        return ExtType(3, _pack_as_array(obj))
    if isinstance(obj, frozenset):
        return ExtType(4, _pack_as_array(obj))
    if isinstance(obj, OrderedDict):
        return ExtType(5, _pack_as_array(obj.items()))
    if isinstance(obj, deque):
        return ExtType(6, _pack_as_array(obj))
    if isinstance(obj, datetime):
        packer = Packer(**_packer_args)
        return ExtType(7, packer.pack(obj.isoformat()))
    if isinstance(obj, timedelta):
        # restored as a tuple
        return ExtType(8, _pack_as_array((obj.days, obj.seconds, obj.microseconds)))
    if isinstance(obj, Image.Image):
        return ExtType(9, _encode_image(obj))
    if isinstance(obj, (numpy.number, numpy.bool_, numpy.bool8)):
        # handle numpy scalar subclasses
        packer = Packer(**_packer_args)
        return ExtType(10, packer.pack(_encode_numpy_number(obj)))
    if isinstance(obj, FinalizedState):
        packer = Packer(**_packer_args)
        return ExtType(11, packer.pack(obj.data))
    if isinstance(obj, timezone):
        # restored as a tuple
        return ExtType(13, _pack_as_array(obj.__getinitargs__()))
    if isinstance(obj, (ndarray_int, ndarray_float, ndarray_complex)):
        return ExtType(14, _pack_as_array(obj.__reduce__()[1]))
    if isinstance(obj, range):
        return ExtType(15, _pack_as_array(obj.__reduce__()[1]))

    raise RuntimeError("Can't convert object of type: %s" % type(obj))


cdef inline object _decode_bytes(bytes buf):
    return unpackb(buf, **_unpacker_args)


cdef object _decode_bytes_as_tuple(bytes buf):
    unpacker = Unpacker(None, max_buffer_size=len(buf), **_unpacker_args)
    unpacker.feed(buf)
    cdef size_t n = unpacker.read_array_header()
    unpack = unpacker.unpack
    obj = PyTuple_New(n)
    cdef size_t i
    for i in range(n):
        u = unpack()
        Py_INCREF(u)  # cython doesn't know about PyTuple_SetItem stealing reference
        PyTuple_SetItem(obj, i, u)
    return obj


cdef object _decode_tinyarray(object decoded_buffer):
    shape, format, data = decoded_buffer
    return tinyarray._reconstruct(shape, format, data)


cdef object _decode_ext(int n, bytes buf):
    # cython translates if/elif chain to a C switch statement!
    if n == 0:
        return _decode_unique_name(buf)
    elif n == 1:
        return _decode_ndarray(_decode_bytes(buf))
    elif n == 2:
        return complex(*_decode_bytes_as_tuple(buf))
    elif n == 3:
        return set(_decode_bytes(buf))
    elif n == 4:
        return frozenset(_decode_bytes(buf))
    elif n == 5:
        return OrderedDict(_decode_bytes(buf))
    elif n == 6:
        return deque(_decode_bytes(buf))
    elif n == 7:
        return _decode_datetime(_decode_bytes(buf))
    elif n == 8:
        return timedelta(*_decode_bytes_as_tuple(buf))
    elif n == 9:
        return _decode_image(buf)
    elif n == 10:
        return _decode_numpy_number(_decode_bytes(buf))
    elif n == 11:
        return FinalizedState(_decode_bytes(buf))
    elif n == 12:
        return _decode_bytes_as_tuple(buf)
    elif n == 13:
        return timezone(*_decode_bytes_as_tuple(buf))
    elif n == 14:
        return _decode_tinyarray(_decode_bytes(buf))
    elif n == 15:
        return range(*_decode_bytes_as_tuple(buf))
    else:
        raise RuntimeError("Unknown extension type: %d" % n)


# When packing or unpacking nested data structures, always use the
# same keyword options
_packer_args = {
    'default': _encode_ext,
    'use_bin_type': True,
    'use_single_float': False,
    'strict_types': True
}

_unpacker_args = {
    'ext_hook': _decode_ext,
    'raw': False,
    'use_list': True,
    'strict_map_key': False,
}


def msgpack_serialize_stream(stream):
    packer = Packer(**_packer_args)
    return stream, packer


def msgpack_deserialize_stream(stream):
    unpacker = Unpacker(stream, max_buffer_size=2**32 - 1, **_unpacker_args)
    return unpacker
