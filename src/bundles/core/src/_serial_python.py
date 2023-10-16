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

#######
# NOT USED: original Python implementation
#######

#
# Encoding and decoding msgpack extension types using in serialization

from msgpack import Packer, ExtType, Unpacker, unpackb
import numpy
import struct
from collections import OrderedDict, deque
from datetime import datetime, timezone, timedelta
from PIL import Image
from .session import _UniqueName
from .state import FinalizedState
import tinyarray
from tinyarray import ndarray_int, ndarray_float, ndarray_complex


def _encode_unique_name(un):
    # Return byte representation for serialization
    class_name, ordinal = un.uid
    try:
        bin_ord = struct.pack("<Q", ordinal)
        if ordinal < 2 ** 8:
            num_bytes = 1
        elif ordinal < 2 ** 16:
            num_bytes = 2
        elif ordinal < 2 ** 24:
            num_bytes = 3
        elif ordinal < 2 ** 32:
            num_bytes = 4
        elif ordinal < 2 ** 40:
            num_bytes = 5
        elif ordinal < 2 ** 48:
            num_bytes = 6
        elif ordinal < 2 ** 56:
            num_bytes = 7
        else:
            num_bytes = 8
        if isinstance(class_name, str):
            cn = bytes(class_name, 'utf-8')
            len_cn = len(cn)
            return struct.pack(
                "<BBB%ds%ds" % (len_cn, num_bytes),
                0, len_cn, num_bytes, cn, bin_ord)
        else:
            bn = bytes(class_name[0], 'utf-8')
            len_bn = len(bn)
            cn = bytes(class_name[1], 'utf-8')
            len_cn = len(cn)
            return struct.pack(
                "<BBBB%ds%ds%ds" % (len_bn, len_cn, num_bytes),
                1, len_bn, len_cn, num_bytes, bn, cn, bin_ord)
    except struct.error:
        # TODO: either string length > 255 or ordinal > 2^64-1
        raise RuntimeError("Unable to encode unique id")


def _decode_unique_name(buf):
    # restore _UniqueName from serialized representation
    import struct
    if buf[0] == 0:
        len_cn, num_bytes = struct.unpack("<BB", buf[1:3])
        class_name, ordinal = struct.unpack(
            "<%ds%ds" % (len_cn, num_bytes), buf[3:])
        class_name = class_name.decode()
    else:
        # assert buf[0] == 1
        len_bn, len_cn, num_bytes = struct.unpack("<BBB", buf[1:4])
        bundle_name, class_name, ordinal = struct.unpack(
            "<%ds%ds%ds" % (len_bn, len_cn, num_bytes), buf[4:])
        class_name = (bundle_name.decode(), class_name.decode())
    ordinal += (8 - num_bytes) * b'\0'
    ordinal, = struct.unpack("<Q", ordinal)
    uid = (class_name, ordinal)
    return _UniqueName(uid)


def _encode_ndarray(o):
    # inspired by msgpack-numpy package
    if o.dtype.kind == 'V':
        # structured array
        kind = b'V'
        dtype = o.dtype.descr
    else:
        kind = b''
        dtype = o.dtype.str
    if 'O' in dtype:
        raise TypeError("Can not serialize numpy arrays of objects")
    return {
        b'kind': kind,
        b'dtype': dtype,
        b'shape': list(o.shape),
        b'data': o.tobytes()
    }


def _decode_ndarray(data):
    kind = data[b'kind']
    dtype = data[b'dtype']
    if kind == b'V':
        dtype = [tuple(str(t) for t in d) for d in dtype]
    return numpy.fromstring(data[b'data'], numpy.dtype(dtype)).reshape(data[b'shape'])


def _encode_image(img):
    import io
    stream = io.BytesIO()
    img.save(stream, format='PNG')
    data = stream.getvalue()
    stream.close()
    return data


def _decode_image(data):
    import io
    stream = io.BytesIO(data)
    img = Image.open(stream)
    img.load()
    return img


def _encode_numpy_number(o):
    return {
        b'dtype': o.dtype.str,
        b'data': o.tobytes()
    }


def _decode_numpy_number(data):
    return numpy.fromstring(data[b'data'], numpy.dtype(data[b'dtype']))[0]


def _decode_datetime(data):
    from dateutil.parser import parse
    return parse(data)


def _encode_ext(obj):
    # Return msgpack extension type, limited to 0-127
    # In simple session test: # of tuples > # of UniqueNames > # of numpy arrays > the rest
    packer = Packer(**_packer_args)
    if isinstance(obj, tuple):
        # TODO: save as msgpack array without converting to list first
        # restored as a tuple
        return ExtType(12, packer.pack(list(obj)))
    if isinstance(obj, _UniqueName):
        return ExtType(0, _encode_unique_name(obj))
    if isinstance(obj, numpy.ndarray):
        # handle numpy array subclasses
        return ExtType(1, packer.pack(_encode_ndarray(obj)))
    if isinstance(obj, complex):
        # restored as a tuple
        return ExtType(2, packer.pack([obj.real, obj.imag]))
    if isinstance(obj, set):
        # TODO: save as msgpack array without converting to list first
        return ExtType(3, packer.pack(list(obj)))
    if isinstance(obj, frozenset):
        # TODO: save as msgpack array without converting to list first
        return ExtType(4, packer.pack(list(obj)))
    if isinstance(obj, OrderedDict):
        # TODO: save as msgpack array without converting to list first
        return ExtType(5, packer.pack(list(obj.items())))
    if isinstance(obj, deque):
        # TODO: save as msgpack array without converting to list first
        return ExtType(6, packer.pack(list(obj)))
    if isinstance(obj, datetime):
        return ExtType(7, packer.pack(obj.isoformat()))
    if isinstance(obj, timedelta):
        # restored as a tuple
        return ExtType(8, packer.pack([obj.days, obj.seconds, obj.microseconds]))
    if isinstance(obj, Image.Image):
        return ExtType(9, _encode_image(obj))
    if isinstance(obj, (numpy.number, numpy.bool_, numpy.bool8)):
        # handle numpy scalar subclasses
        return ExtType(10, packer.pack(_encode_numpy_number(obj)))
    if isinstance(obj, FinalizedState):
        return ExtType(11, packer.pack(obj.data))
    if isinstance(obj, timezone):
        # TODO: save as msgpack array without converting to list first
        # restored as a tuple
        return ExtType(13, packer.pack(list(obj.__getinitargs__())))
    if isinstance(obj, (ndarray_int, ndarray_float, ndarray_complex)):
        return ExtType(14, packer.pack(list(obj.__reduce__()[1])))

    raise RuntimeError("Can't convert object of type: %s" % type(obj))


_decode_handlers = (
    # order must match _encode_ext ExtType's type code
    _decode_unique_name,                                            # 0
    lambda buf: _decode_ndarray(_decode_bytes(buf)),                # 1
    lambda buf: complex(*_decode_bytes_as_tuple(buf)),              # 2
    lambda buf: set(_decode_bytes(buf)),                            # 3
    lambda buf: frozenset(_decode_bytes(buf)),                      # 4
    lambda buf: OrderedDict(_decode_bytes(buf)),                    # 5
    lambda buf: deque(_decode_bytes(buf)),                          # 6
    lambda buf: _decode_datetime(_decode_bytes(buf)),               # 7
    lambda buf: timedelta(*_decode_bytes_as_tuple(buf)),            # 8
    lambda buf: _decode_image(buf),                                 # 9
    lambda buf: _decode_numpy_number(_decode_bytes(buf)),           # 10
    lambda buf: FinalizedState(_decode_bytes(buf)),                 # 11
    lambda buf: _decode_bytes_as_tuple(buf),                        # 12
    lambda buf: timezone(*_decode_bytes_as_tuple(buf)),             # 13
    lambda buf: _decode_tinyarray(*_decode_bytes_as_tuple(buf)),    # 14
)
assert len(_decode_handlers) == 15


def _decode_bytes(buf):
    return unpackb(buf, **_unpacker_args)


def _decode_bytes_as_tuple(buf):
    unpacker = Unpacker(None, **_unpacker_args)
    unpacker.feed(buf)
    n = unpacker.read_array_header()

    def extract(unpacker=unpacker, n=n):
        for i in range(n):
            yield unpacker.unpack()
    return tuple(extract())

def _decode_tinyarray(shape, format, data):
    return tinyarray._reconstruct(shape, format, data)

def _decode_ext(n, buf):
    # assert 0 <= n < len(_decode_handlers)
    return _decode_handlers[n](buf)


_packer_args = {
    'default': _encode_ext,
    'use_bin_type': True,
    'use_single_float': False,
    'strict_types': True
}

_unpacker_args = {
    'ext_hook': _decode_ext,
    'raw': False
}


def msgpack_serialize_stream(stream):
    packer = Packer(**_packer_args)
    return stream, packer


def msgpack_deserialize_stream(stream):
    unpacker = Unpacker(stream, **_unpacker_args)
    return unpacker
