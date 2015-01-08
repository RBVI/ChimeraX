# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
session: application session support
====================================

A session provides access to most of the application's state.
At a minimum, it does not include the operating system state,
like the current directory, nor the environment,
nor any Python interpreter state
-- e.g., the exception hook, module globals, etc.

Code should be designed to support mutiple sessions per process
since it is easier to start with that assumption rather than add it later.
Possible uses of multiple sessions include:
one session per tabbed graphics window,
or for comparing two sessions.

Session data, ie., data that is archived, uses the :py:class:`State` API.
"""

import abc
import weakref
from . import cli
from . import serialize

_builtin_open = open
SUFFIX = ".chimera2"


class State(metaclass=abc.ABCMeta):
    """Session state API for classes that support saving session state

    Session state consists only of "simple" types, i.e.,
    those that are supported by the :py:mod:`.serialize` module.

    Since scenes are snapshots of the current session state,
    the State API is reused for scenes.

    References to objects should use the session's unique identifier
    for the object.  The object's class needs to be registered with
    :py:func:`register_unique_class`.
    """
    #: state type
    SCENE = 0x1
    #: state type
    SESSION = 0x2
    ALL = SCENE | SESSION

    #: state restoration phase
    PHASE1 = 'create objects'
    #: state restoration phase
    PHASE2 = 'resolve object references'

    #: common exception for needing a newer version of the application
    NEED_NEWER = RuntimeError(
        "Need newer version of application to restore session")

    @abc.abstractmethod
    def take_snapshot(self, session, flags):
        """Return snapshot of current state, [version, data], of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped."""
        version = 0
        data = {}
        return [version, data]

    @abc.abstractmethod
    def restore_snapshot(self, phase, session, version, data):
        """Restore data snapshot into instance.

        Restoration is done in two phases: PHASE1 and PHASE2.  The
        first phase should restore all of the data.  The
        second phase should restore references to other objects (data is None).
        The session instance is used to convert unique ids into instances.
        """
        if version != 0 or len(data) > 0:
            raise RuntimeError("Unexpected version or data")

    @abc.abstractmethod
    def reset_state(self):
        """Reset state to data-less state"""
        pass

    # possible animation API
    # include here to emphasize that state aware code
    # needs to support animation
    # def restore_frame(self, phase, frame, timeline, transition):
    #    # frame would be the frame number
    #    # timeline would be sequence of (start frame, scene)
    #    # transition would be method to get to given frame that might need
    #    #   look at several scenes
    #    pass


class ParentState(State):
    """Mixin for classes that manage other state instances.

    This class makes the assumptions that the state instances
    are kept as values in a dictionary, and that all of the instances
    are known.
    """

    VERSION = 0
    _child_attr_name = None  # replace in subclass

    def take_snapshot(self, session, flags):
        """Return snapshot of current state"""
        child_dict = getattr(self, self._child_attr_name)
        data = {}
        for name, child in child_dict.items():
            data[name] = child.take_snapshot(session, flags)
        return [self.VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        """Restore data snapshot into instance"""
        # TODO: handle previous versions
        if version != self.VERSION or not data:
            raise RuntimeError("Unexpected version or data")
        child_dict = getattr(self, self._child_attr_name)
        for name, [child_version, child_data] in data.items():
            if name not in child_dict:
                # TODO: warn about missing child
                #       or create missing child to fill in
                continue
            child = child_dict[name]
            child.restore_snapshot(phase, session, child_version, child_data)

    def reset_state(self):
        """Reset state to data-less state"""
        child_dict = getattr(self, self._child_attr_name)
        for child in child_dict.values():
            child.reset_state()


class Scenes(State):
    """Manage scenes within a session"""
    VERSION = 1

    def __init__(self, session):
        self._session = weakref.ref(session)
        self._scenes = {}

    def save(self, name, metadata=None):
        """Save snapshot of current state in a scene.

        The state consists of the registered attributes that support
        the State API.  The metadata should be a dictionary of
        metadata about the scene.  For example, a thumbnail image.
        metadata contents must be serializable like State data.
        """
        session = self._session()  # resolve back reference
        scene = []
        for tag in session._state_managers:
            manager = session._state_managers[tag]
            snapshot = manager.take_snapshot(scene, State.SCENE)
            if snapshot is None:
                continue
            version, data = snapshot
            scene.append([tag, version, data])
        self._scenes[name] = [metadata, scene]

    def restore(self, name):
        """Restore session to the state represented in the named scene."""
        if name not in self._scenes:
            raise ValueError("Unknown scene")
        scene = self._scenes[name]
        session = self._session()  # resolve back reference
        managers = []
        for tag, version, data in scene:
            if tag not in session._state_managers:
                continue
            manager = session._state_managers[tag]
            manager.restore_snapshot(State.PHASE1, session, version, data)
            managers.append((manager, version))
        for manager, version in managers:
            manager.restore_snapshot(State.PHASE2, session, version, None)

    def delete(self, name):
        """Delete named scene"""
        if name not in self._scenes:
            raise ValueError("Unknown scene")
        del self._scenes[name]

    def metadata(self, name):
        """Return scene's metadata"""
        return self._scenes[name][0]

    def names(self):
        """Return list of scene names"""
        return list(self._scenes.keys())

    def take_snapshot(self, session, flags):
        # documentation from base class
        if (flags & State.SESSION) == 0:
            # don't save scene in scenes
            return None
        return [self.VERSION, self._scenes]

    def restore_snapshot(self, phase, session, version, data):
        # documentation from base class
        if phase != State.PHASE1:
            return
        if version > self.VERSION:
            raise State.NEED_NEWER
        self._scenes = data

    def reset_state(self):
        # documentation from base class
        self._scenes.clear()


_name_to_class = {}
_class_to_name = {}


def register_unique_class(cls, name=None):
    """Register unique class for session files

    The name should be understandable to a user.
    """
    if name is None:
        name = '%s.%s' % (cls.__module__, cls.__name__)
    _name_to_class[name] = cls
    _class_to_name[cls] = name


def unique_class_name(cls):
    """Return registered unique name for class"""
    return _class_to_name[cls]


def unique_class_from_name(name):
    """Return registered class with name"""
    return _name_to_class[name]


class Session:
    """Session management

    The metadata attribute should be a dictionary with information about
    the session, e.g., a thumbnail, a description, the author, etc.

    To preemptively detect problems where different tools want to use the same
    session attribute, session attributes may only be assigned to once,
    and may not be deleted.
    Attributes that support the State API are included
    Consequently, each attribute is an instance that supports the State API.

    Each session attribute, that should be archived,
    must implement the State API, and is then automatically archived.
    """

    def __init__(self):
        # manage
        self._obj_ids = weakref.WeakValueDictionary()
        self._cls_ordinals = {}
        from collections import OrderedDict
        self._state_managers = OrderedDict()   # objects that support State API
        self.metadata = {}          #: session metadata

    def reset(self):
        """Reset session to data-less state"""
        self.metadata.clear()
        self._cls_ordinals.clear()
        for tag in self._state_managers:
            manager = self._state_managers[tag]
            manager.reset_state()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # preemptive debugging for third party packages
            raise AttributeError("attribute already set")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # preemptive debugging for third party packages
        raise AttributeError("can not remove attributes")

    def add_state_manager(self, tag, manager):
        if not isinstance(manager, State):
            raise ValueError('manager needs to implement State API')
        if tag in self._state_managers:
            raise ValueError('already have manager for %s' % tag)
        self._state_managers[tag] = manager

    def replace_state_manager(self, tag, manager):
        """Explictly replace state manager with alternate implementation"""
        if not isinstance(manager, State):
            raise ValueError('manager needs to implement State API')
        if tag not in self._state_managers:
            raise ValueError('missing manager for %s' % tag)
        self._state_managers[tag] = manager

    def replace_attribute(self, name, value):
        """Explictly replace attribute with alternate implementation"""
        object.__setattr__(self, name, value)

    def unique_id(self, obj):
        """Return a unique identifier for an object in session"""

        if hasattr(obj, "_cache_uid"):
            ident = obj._cache_uid
            if ident in self._obj_ids:
                return ident
        key = unique_class_name(obj.__class__)
        ordinal = self._cls_ordinals.get(key, 0)
        ordinal += 1
        self._cls_ordinals[key] = ordinal
        ident = "%s_%d" % (key, ordinal)
        self._obj_ids[ident] = obj
        obj._cache_uid = ident
        return ident

    def unique_obj(self, ident):
        """Return the object that corresponds to the unique identifier"""
        ref = self._obj_ids.get(ident, None)
        if ref is None:
            return None
        return ref()

    def restore_unique_id(self, obj, uid):
        """Restore unique identifier for an object"""
        obj._cache_uid = uid
        self._obj_ids[uid] = obj

    def save(self, stream):
        """Serialize session to stream."""
        serialize.serialize(stream, serialize.VERSION)
        serialize.serialize(stream, self.metadata)
        for tag in self._state_managers:
            manager = self._state_managers[tag]
            snapshot = manager.take_snapshot(self, State.SESSION)
            if snapshot is None:
                continue
            version, data = snapshot
            serialize.serialize(stream, [tag, version, data])
        serialize.serialize(stream, [None, 0, None])
        serialize.serialize(stream, self._cls_ordinals)

    def restore(self, stream, version=None):
        """Deserialize session from stream."""
        self.reset()
        skip_over_metadata = version is not None
        if not skip_over_metadata:
            version = serialize.deserialize(stream)
        if version > serialize.VERSION:
            raise State.NEED_NEWER
        if not skip_over_metadata:
            self.metadata.update(self.read_metadata(stream, skip_version=True))
        # TODO: how much typechecking?
        assert(type(self._cls_ordinals) is dict)
        managers = []
        while True:
            tag, version, data = serialize.deserialize(stream)
            if tag is None:
                break
            if tag not in self._state_managers:
                continue
            manager = self._state_managers[tag]
            manager.restore_snapshot(State.PHASE1, self, version, data)
            managers.append((manager, version))
        self._cls_ordinals = serialize.deserialize(stream)
        for manager, version in managers:
            manager.restore_snapshot(State.PHASE2, self, version, None)

    def read_metadata(self, stream, skip_version=False):
        """Deserialize session metadata from stream."""
        if not skip_version:
            version = serialize.deserialize(stream)
        metadata = serialize.deserialize(stream)
        return version, metadata


@cli.register('save', cli.CmdDesc(required=[('filename', cli.StringArg)]))
def save(session, filename):
    """command line version of saving a session"""
    if not filename.endswith(SUFFIX):
        filename += SUFFIX
    with _builtin_open(filename, 'wb') as output:
        session.save(output)


@cli.register('dump', cli.CmdDesc(required=[('filename', cli.StringArg)],
                                  optional=[('output', cli.StringArg)]))
def dump(session, filename, output=None):
    """dump contents of session for debugging"""
    if not filename.endswith(SUFFIX):
        filename += SUFFIX
    try:
        input = _builtin_open(filename, 'rb')
    except IOError as e:
        session.logger.error(str(e))
        return
    if output is not None:
        output = _builtin_open(output, 'w')
    from pprint import pprint
    with input:
        print("session version:", file=output)
        version = serialize.deserialize(input)
        pprint(version, stream=output)
        print("session metadata:", file=output)
        metadata = serialize.deserialize(input)
        pprint(metadata, stream=output)
        while True:
            tag, version, data = serialize.deserialize(input)
            if tag is None:
                break
            print(tag, 'version:', version, file=output)
            pprint(data, stream=output)
        print("unique id counts:", file=output)
        cls_ordinals = serialize.deserialize(input)
        pprint(cls_ordinals, stream=output)


def open(session, stream, *args, **kw):
    if hasattr(stream, 'read'):
        input = stream
    else:
        # it's really a filename
        input = _builtin_open(stream, 'rb')
    # TODO: active trigger to allow user to stop overwritting
    # current session
    session.restore(input)
    return [], "opened chimera session"


def _initialize():
    from . import io
    io.register_format(
        "Chimera session", io.SESSION, SUFFIX,
        prefixes="session",
        mime="application/x-chimera2-session",
        reference="http://www.rbvi.ucsf.edu/chimera/",
        open_func=open, save_func=save)
_initialize()


def common_startup(sess):
    """Initialize session with common data managers"""
    assert(hasattr(sess, 'app_name'))
    assert(hasattr(sess, 'debug'))
    from . import logger
    sess.logger = logger.Logger()
    from . import triggerset
    sess.triggers = triggerset.TriggerSet()
    sess.scenes = Scenes(sess)
    sess.add_state_manager('scenes', sess.scenes)
    from . import models
    sess.models = models.Models(sess)
    sess.add_state_manager('models', sess.models)
    from .graphics.view import View
    sess.main_view = View(sess.models.drawing, (256, 256), None, sess.logger)
    from . import commands
    commands.register(sess)
    from . import stl
    stl.register()
