# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
serialize: support serialization of "simple" types
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
    :py:class:`~collections.deque`, and :py:class:`~collections.Counter`;
    :py:mod:`datetime`'s :py:class:`~datetime.date`,
    :py:class:`~datetime.time`, :py:class:`~datetime.timedelta`,
    :py:class:`~datetime.datetime`, and :py:class:`~datetime.timezone`;
    and :pillow:`PIL.Image.Image`.

"""
import pickle
import types

#: VERSION number changes if supported data types change
VERSION = 1

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


def serialize(stream, obj):
    """Put object in to a binary stream"""
    pickler = pickle.Pickler(stream, protocol=_PICKLE_PROTOCOL)
    pickler.fast = True     # no recursive lists/dicts/sets
    pickler.dispatch_table = _RestrictTable()
    pickler.dump(obj)


class _RestrictedUnpickler(pickle.Unpickler):

    supported = {
        'builtins': {'complex'},
        'collections': {'deque', 'Counter', 'OrderedDict'},
        'datetime': {'date', 'time', 'timedelta', 'datetime', 'timezone'},
        'numpy': {'ndarray', 'dtype'},
        'numpy.core.multiarray': {'_reconstruct', 'scalar'},
        'PIL.Image': {'Image'},
    }
    from .geometry import Place, Places
    supported[Place.__module__] = {Place.__name__, Places.__name__}

    def find_class(self, module, name):
        if module in self.supported and name in self.supported[module]:
            return getattr(__import__(module, fromlist=(name,)), name)
        # Forbid everything else.
        fullname = '%s.%s' % (module, name)
        raise pickle.UnpicklingError("global '%s' is forbidden" % fullname)


def deserialize(stream):
    """Recover object from a binary stream"""
    unpickler = _RestrictedUnpickler(stream)
    return unpickler.load()

if __name__ == '__main__':
    import io
    import numpy

    def test(obj, msg, expect_pass=True, idempotent=True):
        passed = 'pass' if expect_pass else 'fail'
        failed = 'fail' if expect_pass else 'pass'
        with io.BytesIO() as buf:
            try:
                serialize(buf, obj)
                buf.seek(0)
                result = deserialize(buf)
                assert(numpy.array_equal(result, obj))
            except AssertionError:
                if idempotent:
                    print('%s: %s: not idempotent' % (failed, msg))
                else:
                    print('%s: %s' % (passed, msg))
            except TypeError as e:
                if failed == "fail":
                    print('%s (early): %s: %s' % (failed, msg, e))
                else:
                    print('%s (early): %s' % (failed, msg))
            except pickle.UnpicklingError as e:
                if failed == "fail":
                    print('%s: %s: %s' % (failed, msg, e))
                else:
                    print('%s: %s' % (failed, msg))
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
    test(test_obj, 'numerical numpy array')
    test_obj = numpy.empty((2, 2), dtype=numpy.float32)
    test(test_obj, 'empty numerical numpy array')

    class C:
        pass
    test_obj = numpy.empty((2, 2), dtype=object)
    test_obj[:, :] = C()
    test(test_obj, 'can not serialize numpy array of objects',
         expect_pass=False)

    with open("/bin/ls") as f:
        test(f, 'can not serialize file object', expect_pass=False)

    import datetime
    d = datetime.date(2000, 1, 1)
    test(d, 'date')
    t = datetime.time()
    test(t, 'time')
    t = datetime.timedelta()
    test(t, 'timedelta')
    d = datetime.datetime.now(datetime.timezone.utc)
    test(d, 'datetime&timezone')

    import enum

    class Color(enum.Enum):
        red = 1
    c = Color.red
    test(c, 'can not serialize Enum subclass', expect_pass=False)

    class Color(enum.IntEnum):
        red = 1
    c = Color.red
    test(c, 'can not serialize IntEnum subclass', expect_pass=False)

    import collections
    d = collections.OrderedDict([(1, 2), (3, 4)])
    test(d, 'ordered dict')

    from PIL import Image
    test(Image.new("RGB", (32, 32), "white"), 'PIL image', idempotent=False)
