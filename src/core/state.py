# vim: set expandtab shiftwidth=4 softtabstop=4:
import abc

CORE_STATE_VERSION = 1  #: version of core session state data


class RestoreError(RuntimeError):
    """Raised when session file has a problem being restored"""
    pass


class State(metaclass=abc.ABCMeta):
    """Session state API for classes that support saving session state

    Session state consists only of "simple" types, i.e.,
    those that are supported by the :py:mod:`.serialize` module
    and instances from known classes in tools.

    TODO: confirm:
    Since scenes are snapshots of the current session state,
    the State API is reused for scenes.
    """
    #: state flag
    SCENE = 0x1
    #: state flag
    SESSION = 0x2
    ALL = SCENE | SESSION

    #: Which "tool" this state is from (None is core)
    bundle_info = None

    def take_snapshot(self, session, flags):
        """Return snapshot of current state, [version, data], of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped.
        The default implementation is for non-core classes and
        returns a copy of the instance dictionary (a deep copy of
        lists/dicts/etc., but shallow copy of named objects).
        Named objects are later converted to unique names. 
        """
        version = self.bundle_info.state_version
        data = self.vars().copy()
        if 'bundle_info' in data:
            del data['bundle_info']
        return version, data

    @classmethod
    def restore_snapshot_new(cls, session, bundle_info, version, data):
        """Restore data snapshot into instance.

        Named instances in data will have been replaced with actual instance.
        """
        return cls.__new__(cls)

    def restore_snapshot_init(self, session, bundle_info, version, data):
        obj.__dict__ = data
        if obj.bundle_info is None:
            obj.bundle_info = bundle_info

    @abc.abstractmethod
    def reset_state(self, session):
        """Reset state to data-less state"""
        pass

    # possible animation API
    # include here to emphasize that state aware code
    # needs to support animation
    # def restore_frame(self, frame, timeline, transition):
    #    # frame would be the frame number
    #    # timeline would be sequence of (start frame, scene)
    #    # transition would be method to get to given frame that might need
    #    #   look at several scenes
    #    pass


# Would like to use set's, but isinstance requires a tuple
_final_primitives = ()
_container_primitives = ()


def _init_primitives():
    # These primitives should be exactly the same ones that can be serialized
    global _final_primitives, _container_primitives
    import collections
    import numpy
    import datetime
    from PIL import Image
    from . import geometry
    _final_primitives = (
        type(None), type(Ellipsis),
        bool, bytes, bytearray,
        complex, float,
        int, range, str,
        collections.Counter,
        datetime.date, datetime.time, datetime.timedelta, datetime.datetime,
        datetime.timezone,
        geometry.Place, geometry.Places,
        Image.Image,
    )
    _container_primitives = (
        dict, frozenset, list, set, tuple,
        collections.deque, collections.OrderedDict,
        numpy.ndarray,
    )


def copy_state(data, convert=None):
    """Return a deep copy of primitives, but keep instances from tools.

    Parameters
    ----------
    data : any
        The data to copy
    convert : function
        Optional function to convert objects of tool classes.

    Objects that would be named in a session file are not copied.
    Only known data structures are copied.
    """

    if not _final_primitives:
        _init_primitives()

    if convert is None:
        def convert(x):
            return x

    from collections import Mapping  # deque, Sequence, Set
    from numpy import ndarray

    def _copy(data):
        global _final_primitives, _container_primitives
        nonlocal convert, Mapping, ndarray
        if hasattr(data, 'bundle_info'):
            return convert(data)
        if isinstance(data, _final_primitives):
            return data
        if not isinstance(data, _container_primitives):
            raise ValueError("unable to copy %s.%s objects" % (
                data.__class__.__module__, data.__class__.__name__))
        if isinstance(data, Mapping):
            items = [(_copy(k), _copy(v)) for k, v in data.items()]
        elif isinstance(data, ndarray):
            if data.dtype != object:
                return data.copy()
            items = [_copy(o) for o in data]
        else:
            # must be isinstance(data, (deque, Sequence, Set)):
            items = [_copy(o) for o in data]
        return data.__class__(items)
    return _copy(data)


def dereference_state(data, convert, convert_cls):
    """Inverse of copy_state"""

    if not _final_primitives:
        _init_primitives()

    from collections import Mapping  # deque, Sequence, Set
    from numpy import ndarray

    def _copy(data):
        global _final_primitives, _container_primitives
        nonlocal convert, convert_cls, Mapping, ndarray
        if isinstance(data, convert_cls):
            return convert(data)
        if isinstance(data, _final_primitives):
            return data
        if not isinstance(data, _container_primitives):
            raise ValueError("unable to copy %s objects" % data.__class__.__name__)
        if isinstance(data, Mapping):
            items = [(_copy(k), _copy(v)) for k, v in data.items()]
        elif isinstance(data, ndarray):
            if data.dtype != object:
                return data.copy()
            items = [_copy(o) for o in data]
        else:
            # must be isinstance(data, (deque, Sequence, Set)):
            items = [_copy(o) for o in data]
        return data.__class__(items)
    return _copy(data)
