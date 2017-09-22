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

"""
serialize: Support serialization of "simple" types
==================================================

Provide object serialization and deserialization for simple Python objects.
In this case, simple means numbers (int, float, numpy arrays), strings,
bytes, booleans, and non-recursive tuples, lists, sets, and dictionaries.
Recursive data-structures are not checked for and thus can cause an infinite
loop.  Arbitrary Python objects are not allowed.

Internally use pickle, with safeguards on what is written (so the author of
the code is more likely to find the bug before a user of the software does),
and on what is read.  The reading is more restrictive because the C-version
of the pickler will pickle objects, like arbtrary functions.  The
deserializer catches those mistakes, but later when the session is opened.

Version 1 of the protocol supports instances of the following types:

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
    and :pillow:`PIL.Image.Image`.

"""
import msgpack
import pickle
import types  # for pickle support
# imports for supported "primitive" types and collections
# ** must be keep in sync with state.py **
import numpy
from collections import OrderedDict, deque
from datetime import datetime, timezone, timedelta
from PIL import Image
from .session import _UniqueName
from .state import FinalizedState

# from ._serial_python import msgpack_serialize_stream, msgpack_deserialize_stream
from ._serialize import msgpack_serialize_stream, msgpack_deserialize_stream

# TODO: remove pickle and msgpack v2 after corresponding sessions
# are no longer supported.
_PICKLE_PROTOCOL = 4


class _RestrictTable(dict):

    def __init__(self, *args, **kwds):
        dict.__init__(self, *args, **kwds)
        import copyreg
        if complex in copyreg.dispatch_table:
            self[complex] = copyreg.dispatch_table[complex]
        try:
            import numpy
            self[numpy.ndarray] = numpy.ndarray.__reduce__
        except ImportError:
            pass

    def get(self, cls, default=None):
        if isinstance(cls, types.BuiltinFunctionType):
            # need to allow for unpickling numpy arrays and other types
            return default
        if cls not in self:
            raise TypeError("Can not serialize class: %s" % cls.__name__)
        return dict.__getitem__(self, cls)


def pickle_serialize(stream, obj):
    """Put object in to a binary stream"""
    pickler = pickle.Pickler(stream, protocol=_PICKLE_PROTOCOL)
    pickler.fast = True     # no recursive lists/dicts/sets
    pickler.dispatch_table = _RestrictTable()
    pickler.dump(obj)


class _RestrictedUnpickler(pickle.Unpickler):

    supported = {
        'builtins': {'complex'},
        'collections': {'deque', 'Counter', 'OrderedDict'},
        'datetime': {'timedelta', 'timezone', 'datetime'},
        'numpy': {'ndarray', 'dtype'},
        'numpy.core.multiarray': {'_reconstruct', 'scalar'},
        'PIL.Image': {'Image'},
    }
    supported[_UniqueName.__module__] = {_UniqueName.__name__}
    supported[FinalizedState.__module__] = {FinalizedState.__name__}

    def find_class(self, module, name):
        if module in self.supported and name in self.supported[module]:
            return getattr(__import__(module, fromlist=(name,)), name)
        # Forbid everything else.
        fullname = '%s.%s' % (module, name)
        raise pickle.UnpicklingError("global '%s' is forbidden" % fullname)


def pickle_deserialize(stream):
    """Recover object from a binary stream"""
    unpickler = _RestrictedUnpickler(stream)
    return unpickler.load()


def _decode_image(data):
    import io
    stream = io.BytesIO(data)
    img = Image.open(stream)
    img.load()
    return img


def _decode_ndarray(data):
    kind = data[b'kind']
    dtype = data[b'dtype']
    if kind == b'V':
        dtype = [tuple(str(t) for t in d) for d in dtype]
    return numpy.fromstring(data[b'data'], numpy.dtype(dtype)).reshape(data[b'shape'])


def _decode_numpy_number(data):
    return numpy.fromstring(data[b'data'], numpy.dtype(data[b'dtype']))[0]


def _decode_datetime(data):
    from dateutil.parser import parse
    return parse(data)


_decode_handlers_v2 = [
    # order must match encode's __type__ values
    lambda args: _UniqueName(args[0][1]),
    lambda args: _decode_ndarray(dict(args)),
    lambda args: complex(*args[0][1]),
    lambda args: set(args[0][1]),
    lambda args: frozenset(args[0][1]),
    OrderedDict,
    lambda args: deque(args[0][1]),
    lambda args: _decode_datetime(args[0][1]),
    lambda args: timedelta(*args[0][1]),
    lambda args: _decode_image(args[0][1]),
    lambda args: _decode_numpy_number(dict(args)),
    lambda args: FinalizedState(args[0][1]),
    lambda args: timezone(*args[0][1]),
]


def _decode_pairs_v2(pairs):
    try:
        len(pairs)
    except TypeError:
        pairs = tuple(pairs)
    if not pairs:
        return dict()
    if pairs[0][0] != '__type__':
        return OrderedDict(pairs)
    cvt = _decode_handlers_v2[pairs[0][1]]
    return cvt(pairs[1:])


def msgpack_deserialize_stream_v2(stream):
    unpacker = msgpack.Unpacker(
        stream, object_pairs_hook=_decode_pairs_v2, encoding='utf-8',
        use_list=False)
    return unpacker


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
        # packer = msgpack_serialize_stream_v2(buf)
        packer = msgpack_serialize_stream(buf)
        msgpack_serialize(packer, obj)

    def deserialize(buf):
        # unpacker = msgpack_deserialize_stream_v2(buf)
        unpacker = msgpack_deserialize_stream(buf)
        return msgpack_deserialize(unpacker)

    # serialize = pickle_serialize
    # deserialize = pickle_deserialize

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
