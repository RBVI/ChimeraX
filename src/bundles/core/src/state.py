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

import abc


class RestoreError(RuntimeError):
    """Raised when session file has a problem being restored"""
    pass


class State:
    """Session state API for classes that support saving session state

    Session state consists only of "simple" types, i.e.,
    those that are supported by the :py:mod:`.serialize` module
    and instances from known classes in bundles.

    TODO: confirm:
    Since scenes are snapshots of the current session state,
    the State API is reused for scenes.
    """
    #: state flag
    SCENE = 0x1
    #: state flag
    SESSION = 0x2
    #: state flag
    INCLUDE_MAPS = 0x4
    ALL = SCENE | SESSION | INCLUDE_MAPS

    def take_snapshot(self, session, flags):
        """Return snapshot of current state of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped.
        The default implementation is for non-core classes and
        returns a copy of the instance dictionary (a deep copy of
        lists/dicts/etc., but shallow copy of named objects).
        Named objects are later converted to unique names.
        """
        return vars(self).copy()

    @classmethod
    def restore_snapshot(cls, session, data):
        """Create object using snapshot data."""
        obj = cls()
        obj.__dict__ = data
        return obj

    # possible animation API
    # include here to emphasize that state aware code
    # needs to support animation
    # def restore_frame(self, frame, timeline, transition):
    #    # frame would be the frame number
    #    # timeline would be sequence of (start frame, scene)
    #    # transition would be method to get to given frame that might need
    #    #   look at several scenes
    #    pass


class StateManager(State, metaclass=abc.ABCMeta):

    def init_state_manager(self, session, base_tag=None):
        ''' Should only be called for StateManagers that are not assigned to session attributes
            since those will be added to the session StateManager list as a side effect of __setattr__.
            The 'base_tag' is used as the tag when registering the state manager with the session, so
            for debugging purposes it is useful to provide a string that indicates the state manager's
            function, though not mandatory.  If multiple managers of the same type will be created (and
            therefore base_tag may not be unique) an integer will be appended to the text to make it
            unique.
        '''
        if base_tag is None:
            base_tag = "anonymous"
        try:
            session.get_state_manager(base_tag)
        except KeyError:
            tag = base_tag
        else:
            i = 1
            while True:
                tag = "%s %d" % (base_tag, i)
                try:
                    session.get_state_manager(tag)
                except KeyError:
                    break
                i += 1

        self.__tag = tag
        self.__session = session
        session.add_state_manager(tag, self)

    def destroy(self):
        ''' Like init_state_manager, should only be called on StateManagers that are not also session
            attributes, when the manager should be disposed of.
        '''
        self.__session.remove_state_manager(self.__tag)

    @abc.abstractmethod
    def reset_state(self, session):
        """Reset state to data-less state"""
        pass

    def include_state(self):
        """Return if state manager's state should be included in session"""
        return True


class FinalizedState:
    """Used for efficiency if state data is known to be nothing but Python simple primitives"""
    def __init__(self, data):
        self.data = data


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
    import tinyarray

    def numpy_numbers():
        for n in dir(numpy):
            try:
                t = getattr(numpy, n)
                if issubclass(t, numpy.number):
                    yield t
            except Exception:
                pass
        yield numpy.bool_
        yield numpy.bool8

    _final_primitives = (
        type(None),
        # type(Ellipsis), -- primitive in Python, no equivalent in msgpack
        bool, bytes, bytearray,
        complex, float,
        int, range, str,
        collections.Counter,
        datetime.datetime, datetime.timedelta,
        datetime.timezone,
        Image.Image,
        FinalizedState,
        # tinyarrays are immutable
        tinyarray.ndarray_complex,
        tinyarray.ndarray_float,
        tinyarray.ndarray_int,
    )
    _final_primitives += tuple(numpy_numbers())
    _container_primitives = (
        dict, frozenset, list, set, tuple,
        collections.deque, collections.OrderedDict,
        numpy.ndarray,
    )


def copy_state(data, convert=None):
    """Return a deep copy of primitives, but keep instances from bundles.

    Parameters
    ----------
    data : any
        The data to copy
    convert : function
        Optional function to convert objects of bundle classes.

    Objects that would be named in a session file are not copied.
    Only known data structures are copied.
    """

    if not _final_primitives:
        _init_primitives()

    if convert is None:
        def convert(x):
            return x

    from collections.abc import Mapping  # deque, Sequence, Set
    import numpy

    def _copy(data):
        global _final_primitives, _container_primitives
        nonlocal convert, Mapping, numpy
        if isinstance(data, _final_primitives):
            return data
        if isinstance(data, _container_primitives):
            if isinstance(data, Mapping):
                items = [(_copy(k), _copy(v)) for k, v in data.items()]
            elif isinstance(data, numpy.ndarray):
                if data.dtype != object:
                    return data.copy()
                a = numpy.array([_copy(o) for o in data.flat], dtype=object)
                a.shape = data.shape
                return a
            else:
                # must be isinstance(data, (deque, Sequence, Set)):
                items = [_copy(o) for o in data]
            return data.__class__(items)
        else:
            # Use state methods to convert object to primitives
            return convert(data)

    return _copy(data)


def dereference_state(data, convert, convert_cls):
    """Inverse of copy_state"""

    if not _final_primitives:
        _init_primitives()

    from collections.abc import Mapping  # deque, Sequence, Set
    import numpy

    def _copy(data):
        global _final_primitives, _container_primitives
        nonlocal convert, convert_cls, Mapping, numpy
        if isinstance(data, convert_cls):
            return convert(data)
        if isinstance(data, FinalizedState):
            return data.data
        if isinstance(data, _final_primitives):
            return data
        if not isinstance(data, _container_primitives):
            raise ValueError("unable to copy %s objects" % data.__class__.__name__)
        if isinstance(data, Mapping):
            items = [(_copy(k), _copy(v)) for k, v in data.items()]
        elif isinstance(data, numpy.ndarray):
            if data.dtype != object:
                return data.copy()
            a = numpy.array([_copy(o) for o in data.flat], dtype=object)
            a.shape = data.shape
            return a
        else:
            # must be isinstance(data, (deque, Sequence, Set)):
            items = [_copy(o) for o in data]
        return data.__class__(items)
    return _copy(data)
