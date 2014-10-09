"""
session: application session support
====================================

A session instance provides access to most of the application's state.
self._scenes.clear()
and the environment, etc.;
and Python state -- e.g., the exception hook, module globals, etc.

To preemptively detect problems where different tools want to use the same
session attribute, session attributes may only be assigned to once,
and may not be deleted.
Consequently, each attribute is an instance that supports a Chimera API.

Data that is archived, either for sessions or scenes, uses the State API.
Session attributes, that should be archived, need to be registered.
"""

import abc
import weakref

class State(metaclass=abc.ABCMeta):
    """Session state API for classes that support saving session and/or scene state

    Session state consists only of "simple" types, i.e.,
    lists, dictionaries, strings, integers,
    floating point numbers, booleans, and None.
    Tuples are treated like lists.
    enum.Enum subclasses are converted to integers.
    TODO: things that would break JSON compatibility: sets, datetime, bytes

    References to objects should use the session's unique identifier
    for the object.
    """
    # state types
    SCENE = 0x1
    SESSION = 0x2
    ALL = SCENE | SESSION

    # state restoration phases
    PHASE1 = 'create objects'
    PHASE2 = 'resolve object references'

    NEED_NEWER = RuntimeError("Need newer version of application to restore session")

    @abc.abstractmethod
    def take_snapshot(self, session, flags=ALL):
        """Return snapshot of current state, [version, data], of instance.
        
        Return None if should be skipped."""
        version = 0
        data = {}
        return [version, data]

    @abc.abstractmethod
    def restore_snapshot(self, phase, session, version, data):
        """Restore data snapshot into instance.
        
        Restoration is done in two phases: PHASE1 and PHASE2.  The
        first phase should restore all of the data.  The
        second phase should restore references to other objects (version
        and data are None).
        The session instance is used to convert unique ids into instances.
        """
        if len(data) != 2 or data[0] != 0 or data[1]:
            raise RuntimeError("Unexpected version or data")

    #abc.abstractmethod
    def reset(self):
        """Reset state to data-less state"""
        pass

class Scenes(State):
    """Manage scenes within a session"""
    VERSION = 1

    def __init__(self, session):
        self._session = weakref.proxy(session)
        self._scenes = {}

    def save(self, name, metadata=None):
        """Save snapshot of current state in a scene.

        The state consists of the registered attributes that support
        the State API.  The metadata should be a dictionary of
        metadata about the scene.  For example, a thumbnail image.
	metadata contents must be serializable like State data.
        """
        session = self._session # resolve back reference
        scene = []
        for attr_name in session._state_attrs:
            attr = getattr(session, attr_name)
            snapshot = attr.take_snapshot(scene, State.SCENE)
            if snapshot is None:
                continue
            version, data = snapshot
            scene.append([attr_name, version, data])
        self._scenes[name] = [metadata, scene]

    def restore(self, name):
        """Restore session to the state represented in the named scene."""
        if name not in self._scenes:
            raise ValueError("Unknown scene")
        scene = self._scenes[name]
        session = self._session # resolve back reference
        attrs = []
        for attr_name, version, data in scene:
            if attr_name is not in session._session_attrs:
                continue
            attr = getattr(session, attr_name)
            attr.restore_snapshot(State.PHASE1, session, version, data)
            attrs.append(attr)
        for attr in attrs:
            attr.restore_snapshot(State.PHASE2, session, None, None)

    def delete(self, name):
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
        if (flags & State.SESSION) == 0:
            # don't save scene in scenes
            return None
        return [self.VERSION, self._scenes]

    def restore_snapshot(self, phase, session, version, data):
        if phase != State.PHASE1:
            return
        if version > self.VERSION:
            raise State.NEED_NEWER
        self._scenes = data

    def reset(self):
        self._scenes.clear()

class Session:
    """Session management

    The metadata attribute should be a dictionary with information about
    the session, eg., a thumbnail, a description, the author, etc.
    """
    VERSION = 1

    def __init__(self):
        # manage
        self._obj_ids = weakref.WeakValueDictionary()
        self._cls_ordinals = {}
        self._state_attrs = []   # which attributes support State API
        self.metadata = {}

    def reset(self):
        """Reset session to data-less state"""
        self.metadata.clear()
        self._cls_ordinals.clear()
        for attr_name in self._state_attrs:
            attr = getattr(self, attr_name)
            attr.reset()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # preemptive debugging for third party packages
            raise AttributeError("attribute already set")
        if isinstance(value, State):
            self._state_attrs.append(name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # preemptive debugging for third party packages
        raise AttributeError("can not remove attributes")

    def replace_attribute(self, name, value):
        if isinstance(getattr(self, name), State) != isinstance(value, State):
            raise ValueError("Use of State API changed")
        object.__setattr__(self, name, value)

    def unique_id(self, obj):
        """Return a unique identifier for an object in session"""

        if hasattr(obj, "_cache_uid"):
            ident = obj._cache_uid
            if ident in self._obj_ids:
                return ident
        cls = obj.__class__
        key = '%s.%s' % (cls.__module__, cls.__name__)
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
        serialize(stream, self.VERSION)
        serialize(stream, self.metadata)
        serialize(stream, self._cls_ordinals)
        for attr_name in self._state_attrs:
            attr = getattr(self, attr_name)
            snapshot = attr.take_snapshot(self, State.SESSION)
            if snapshot is None:
                continue
            version, data = snapshot
            serialize(stream, [attr_name, version, data])
        serialize(stream, [None, None, None])

    def restore(self, stream, version=None):
        """Deserialize session from stream."""
	self.reset()
        skip_over_metadata = version is not None
        if not skip_over_metadata:
            version = deserialize(stream)
        if version > self.VERSION:
            raise State.NEED_NEWER
        if not skip_over_metadata:
            self.metadata.update(self.read_metadata(stream, skip_version=True))
        self._cls_ordinals = deserialize(stream)   # TODO: typecheck?
        attrs = []
        while True:
            attr_name, version, data = deserialize(stream)
            if attr_name is None:
                break
            if attr_name is not in self._session_attrs:
                continue
            attr = getattr(self, attr_name)
            attr.restore_snapshot(State.PHASE1, session, version, data)
            attrs.append(attr)
        for attr in attrs:
            attr.restore_snapshot(State.PHASE2, session, None, None)

    def read_metadata(self, stream, skip_version=False):
        """Deserialize session metadata from stream."""
        if not skip_version:
            version = deserialize(stream)
        metadata = deserialize(stream)
        return version, metadata

    def reset(self):
        """Reset session state."""
        for attr_name in self._state_attrs:
            attr = getattr(self, attr_name)
            attr.reset()

def common_startup(sess):
    """Initialize session with common data managers"""
    sess.scenes = Scenes(sess)
    from . import models
    sess.models = models.Models(sess)
