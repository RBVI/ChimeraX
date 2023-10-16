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

"""
serialize: Support serialization of "simple" types
==================================================

Provide object serialization and deserialization for simple Python objects.
In this case, simple means numbers (int, float, numpy arrays), strings,
bytes, booleans, and non-recursive tuples, lists, sets, and dictionaries.
Recursive data-structures are not checked for and thus can cause an infinite
loop.  Arbitrary Python objects are not allowed.

Version 3 of the protocol supports instances of the following types:

    :py:class:`bool`; :py:class:`int`; :py:class:`float`; :py:class:`complex`;
    numpy :py:class:`~numpy.ndarray`;
    :py:class:`str`; :py:class:`bytes`; :py:class:`bytearray`;
    type(:py:data:`None`);
    :py:class:`set`; :py:class:`frozenset`;
    :py:class:`dict`;
    :py:mod:`collections`' :py:class:`~collections.OrderedDict`,
    and :py:class:`~collections.deque`;
    :py:mod:`datetime`'s :py:class:`~datetime.datetime`,
    :py:class:`~datetime.timezone`; and :py:class:`~datetime.timedelta`;
    :pillow:`PIL.Image.Image`;
    tinyarray, :py:class:`~tinyarray.ndarray_int`, :py:class:`~tinyarray.ndarray_float`, :py:class:`~tinyarray.ndarray_complex`

"""

from . import _serialize

# from ._serial_python import msgpack_serialize_stream, msgpack_deserialize_stream
from ._serialize import msgpack_serialize_stream, msgpack_deserialize_stream, PRIMITIVE_TYPES
import pickle  # to recognize old session files


class _RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        # Forbid everything not builtin
        fullname = '%s.%s' % (module, name)
        raise pickle.UnpicklingError("global '%s' is forbidden" % fullname)


def pickle_deserialize(stream):
    """Recover object from a binary stream"""
    unpickler = _RestrictedUnpickler(stream)
    return unpickler.load()


def msgpack_serialize(stream, obj):
    # _count_object_types(obj)  # DEBUG
    stream, packer = stream
    stream.write(packer.pack(obj))


def msgpack_deserialize(stream):
    try:
        return next(stream)
    except StopIteration:
        return None


# Debuging code for finding out object types used

import msgpack
# imports for supported "primitive" types and collections
# ** must be keep in sync with state.py **
import numpy
from collections import OrderedDict, deque
from datetime import datetime, timezone, timedelta
from PIL import Image
from .session import _UniqueName
from .state import FinalizedState

_object_counts = {
    type(None): 0,
    bool: 0,
    int: 0,
    bytes: 0,
    bytearray: 0,
    str: 0,
    memoryview: 0,
    float: 0,
    list: 0,
    dict: 0,

    # extension types
    _UniqueName: 0,
    numpy.ndarray: 0,
    complex: 0,
    set: 0,
    frozenset: 0,
    OrderedDict: 0,
    deque: 0,
    datetime: 0,
    timedelta: 0,
    Image.Image: 0,
    numpy.number: 0,
    FinalizedState: 0,
    tuple: 0,
    timezone: 0,
}

_extention_types = [
    _UniqueName,
    numpy.ndarray,
    complex,
    set,
    frozenset,
    OrderedDict,
    deque,
    datetime,
    timedelta,
    Image.Image,
    numpy.number,
    FinalizedState,
    tuple,
    timezone,
]


def _reset_object_counts():
    for t in _object_counts:
        _object_counts[t] = 0


def _count_object_types(obj):
    # handle numpy subclasses
    if isinstance(obj, numpy.ndarray):
        _object_counts[numpy.ndarray] += 1
        return
    if isinstance(obj, (numpy.number, numpy.bool_, numpy.bool8)):
        _object_counts[numpy.number] += 1
        return
    t = type(obj)
    _object_counts[t] += 1
    if t == FinalizedState:
        _count_object_types(obj.data)
        return
    if t in (list, tuple, set, frozenset, deque):
        for o in obj:
            _count_object_types(o)
        return
    if t in (dict, OrderedDict):
        for k, v in obj.items():
            _count_object_types(k)
            _count_object_types(v)
        return


def _print_object_counts():
    types = list(_object_counts)
    # types = list(_extention_types)
    types.sort(key=lambda t: t.__name__)
    for t in types:
        print('%s: %s' % (t.__name__, _object_counts[t]))


if __name__ == '__main__':
    import io

    def serialize(buf, obj):
        packer = msgpack_serialize_stream(buf)
        msgpack_serialize(packer, obj)

    def deserialize(buf):
        unpacker = msgpack_deserialize_stream(buf)
        return msgpack_deserialize(unpacker)

    def test(obj, msg, expect_pass=True, compare=None):
        passed = 'pass' if expect_pass else 'fail'
        failed = 'fail' if expect_pass else 'pass'
        with io.BytesIO() as buf:
            try:
                serialize(buf, obj)
            except Exception as e:
                if failed == "fail":
                    print('%s (serialize): %s: %s' % (failed, msg, e))
                else:
                    print('%s (serialize): %s' % (failed, msg))
                return
            buf.seek(0)
            try:
                result = deserialize(buf)
            except Exception as e:
                if failed == "fail":
                    print('%s (deserialize): %s: %s' % (failed, msg, e))
                else:
                    print('%s (deserialize): %s' % (failed, msg))
                return
            try:
                if compare is not None:
                    assert compare(result, obj)
                else:
                    assert result == obj
            except AssertionError:
                print('%s: %s: not idempotent' % (failed, msg))
                print('  original:', obj)
                print('  result:', result)
            else:
                print('%s: %s' % (passed, msg))

    # test: basic type support
    test(3, 'an int')
    test(42.0, 'a float')
    test('chimera', 'a string')
    test(complex(3, 4), 'a complex')
    test(False, 'False')
    test(True, 'True')
    test(None, 'None')
    test(b'xyzzy', 'some bytes')
    test(((0, 1), (2, 0)), 'nested tuples')
    test([[0, 1], [2, 0]], 'nested lists')
    test({'a': {0: 1}, 'b': {2: 0}}, 'nested dicts')
    test({1, 2, frozenset([3, 4])}, 'frozenset nested in a set')
    test(bool, 'can not serialize bool', expect_pass=False)
    test(float, 'can not serialize float', expect_pass=False)
    test(int, 'can not serialize int', expect_pass=False)
    test(set, 'can not serialize set', expect_pass=False)

    # test: objects
    class C:
        pass
    test_obj = C()
    test_obj.test = 12
    test(C, 'can not serialize class definition', expect_pass=False)
    test(test_obj, 'can not serialize objects', expect_pass=False)

    # test: functions
    test(serialize, 'can not serialize function objects', expect_pass=False)
    test(abs, 'can not serialize builtin functions', expect_pass=False)

    # test: numpy arrays
    test_obj = numpy.zeros((2, 2), dtype=numpy.float32)
    test(test_obj, 'numerical numpy array', compare=numpy.array_equal)
    test_obj = numpy.empty((2, 2), dtype=numpy.float32)
    test(test_obj, 'empty numerical numpy array', compare=numpy.array_equal)

    class C:
        pass
    test_obj = numpy.empty((2, 2), dtype=object)
    test_obj[:, :] = C()
    test(test_obj, 'can not serialize numpy array of objects',
         expect_pass=False, compare=numpy.array_equal)
    test_obj = numpy.float32(3.14159)
    test(test_obj, 'numpy float32 number')

    import sys
    if sys.platform.startswith('win'):
        with open("nul:") as f:
            test(f, 'can not serialize file object', expect_pass=False)
    else:
        with open("/bin/ls") as f:
            test(f, 'can not serialize file object', expect_pass=False)

    # d = date(2000, 1, 1)
    # test(d, 'date')
    # t = time()
    # test(t, 'time')
    t = timedelta()
    test(t, 'timedelta')
    d = datetime.now()
    test(d, 'datetime')
    d = datetime.now().astimezone()
    test(d, 'datetime&timezone')
    d = datetime.now(timezone.utc)
    test(d, 'datetime&utc timezone')
    test(timezone.utc, 'utc timezone')

    import enum

    class Color(enum.Enum):
        red = 1
    c = Color.red
    test(c, 'can not serialize Enum subclass', expect_pass=False)

    class Color(enum.IntEnum):
        red = 1
    c = Color.red
    test(c, 'IntEnum subclass instance', expect_pass=False)

    d = OrderedDict([(1, 2), (3, 4), (5, 6), (7, 8)])
    test(d, 'ordered dict')

    def image_compare(a, b):
        return a.tobytes() == b.tobytes()

    test(Image.new("RGB", (32, 32), "white"), 'RGB image', compare=image_compare)
    img = Image.new('RGBA', (127, 253), color=(255, 10, 140, 127))
    test(img, 'RBGA image', compare=image_compare)

    test(_UniqueName((('bundle', 'class'), 10010)), 'UniqueName in bundle')
    test(_UniqueName(('class_name', 65537)), 'UniqueName in core')
