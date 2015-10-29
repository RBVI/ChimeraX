# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
session: Application session support
====================================

A session provides access to most of the application's state.
At a minimum, it does not include the operating system state,
like the current directory, nor the environment,
nor any Python interpreter state
-- e.g., the exception hook, module globals, etc.

Code should be designed to support multiple sessions per process
since it is easier to start with that assumption rather than add it later.
Possible uses of multiple sessions include:
one session per tabbed graphics window,
or for comparing two sessions.

Session data, ie., data that is archived, uses the :py:class:`State` API.
"""

import weakref
from . import serialize
from .state import State, RestoreError

_builtin_open = open
SESSION_SUFFIX = ".c2ses"

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
            snapshot = manager.take_snapshot(scene, self.SAVE_PHASE, State.SCENE)
            if snapshot is None:
                continue
            version, data = snapshot
            scene.append([tag, version, data])
        for tag in session._state_managers:
            manager = session._state_managers[tag]
            manager.take_snapshot(scene, self.CLEANUP_PHASE, State.SCENE)
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
            manager.restore_snapshot(self.CREATE_PHASE, session, version, data)
            managers.append((manager, version))
        for manager, version in managers:
            manager.restore_snapshot(self.RESOLVE_PHASE, session, version, None)

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

    def take_snapshot(self, phase, session, flags):
        # documentation from base class
        if phase == self.CLEANUP_PHASE:
            return
        if phase != self.SAVE_PHASE:
            return
        if (flags & State.SESSION) == 0:
            # don't save scene in scenes
            return
        return [self.VERSION, self._scenes]

    def restore_snapshot(self, phase, session, version, data):
        # documentation from base class
        if version > self.VERSION:
            raise State.NeedNewerError
        if phase == self.CREATE_PHASE:
            self._scenes = data

    def reset_state(self):
        # documentation from base class
        self._scenes.clear()


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
        self._obj_uids = weakref.WeakValueDictionary()
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

    def unique_id(self, obj, tool_info=None):
        """Return a unique identifier for an object in session

        Consequently, the identifier is composed of simple data types.

        Parameters
        ----------
        obj : any object
        tool_info : optional :py:class:`~chimera.core.toolshed.ToolInfo` instance
            Explicitly denote which tool object comes from.
        """

        cls = obj.__class__
        if hasattr(obj, 'tool_info'):
            tool_info = obj.tool_info
        elif hasattr(cls, 'tool_info'):
            tool_info = cls.tool_info
        if tool_info is not None:
            class_name = (tool_info.name, tool_info.version, cls.__name__)
            if 1:  # DEBUG
                # double check that class will be able to be restored
                t = self.toolshed.find_tool(tool_info.name,
                                            version=tool_info.version)
                if cls != t.get_class(cls.__name__):
                    raise RuntimeError(
                        'unable to restore objects of %s class in %s tool' %
                        (class_name, tool_info.name))
        else:
            if not cls.__module__.startswith('chimera.core.'):
                raise RuntimeError('No tool information for %s.%s' % (
                    cls.__module__, cls.__name__))
            class_name = cls.__name__
            # double check that class will be able to be restored
            from chimera.core import get_class
            if cls != get_class(class_name):
                raise RuntimeError('unable to restore objects of %s class' % class_name)
        if hasattr(obj, "_cache_uid"):
            ordinal = obj._cache_uid
            uid = (class_name, ordinal)
            if uid in self._obj_uids:
                return uid
        ordinal = self._cls_ordinals.get(class_name, 0)
        ordinal += 1
        uid = (class_name, ordinal)
        self._cls_ordinals[class_name] = ordinal
        self._obj_uids[uid] = obj
        obj._cache_uid = ordinal
        return uid

    def unique_obj(self, uid):
        """Return the object that corresponds to the unique identifier"""
        ref = self._obj_uids.get(uid, None)
        return ref

    def restore_unique_id(self, obj, uid):
        """Restore unique identifier for an object"""
        class_name, ordinal = uid
        obj._cache_uid = ordinal
        self._obj_uids[uid] = obj
        current_ordinal = self._cls_ordinals.get(class_name, 0)
        if ordinal > current_ordinal:
            self._cls_ordinals[class_name] = ordinal

    def class_name_of_unique_id(self, uid):
        """Extract class name associated with unique id for messages"""
        class_name = uid[0]
        if isinstance(class_name, str):
            return class_name
        return "Tool %s %s's %s" % class_name

    def class_of_unique_id(self, uid, base_class):
        """Return class associated with unique id
        
        Parameters
        ----------
        uid : unique identifer for class
        base_class : the expected base class of the class
        tool_name : internal name of tool that provides class
            If not given, then it must be in the chimera core.
        tool_version : the tool's version
            If not given, then it must be in the chimera core.

        Raises
        ------
        KeyError
        """
        class_name, ordinal = uid
        if isinstance(class_name, str):
            from chimera.core import get_class
            cls = get_class(class_name)
        else:
            tool_name, tool_version, class_name = class_name
            t = self.toolshed.find_tool(tool_name, version=tool_version)
            if t is None:
                # TODO: load tool from toolshed
                session.logger.error("Missing '%s' (internal name) tool" % tool_name)
                return
            cls = t.get_class(class_name)

        try:
            assert(issubclass(cls, base_class))
        except Exception as e:
            raise KeyError(str(e))
        return cls

    def save(self, stream):
        """Serialize session to stream."""
        serialize.serialize(stream, serialize.VERSION)
        serialize.serialize(stream, self.metadata)
        # guarantee that tools are serialized first, so on restoration,
        # all of the related code will be loaded before the rest of the
        # session is restored
        managers = list(self._state_managers)
        if 'tools' in self._state_managers:
            managers.remove('tools')
            managers.insert(0, 'tools')
        for tag in list(managers):
            manager = self._state_managers[tag]
            snapshot = manager.take_snapshot(self, State.SAVE_PHASE, State.SESSION)
            if snapshot is None:
                managers.remove(tag)
                continue
            version, data = snapshot
            serialize.serialize(stream, [tag, version, data])
        for tag in managers:
            manager = self._state_managers[tag]
            snapshot = manager.take_snapshot(self, State.CLEANUP_PHASE, State.SESSION)
        serialize.serialize(stream, [None, 0, None])

    def restore(self, stream, version=None):
        """Deserialize session from stream."""
        self.reset()
        skip_over_metadata = version is not None
        if not skip_over_metadata:
            version = serialize.deserialize(stream)
        if version > serialize.VERSION:
            raise State.NeedNewerError
        if not skip_over_metadata:
            self.metadata.update(self.read_metadata(stream, skip_version=True))
        # TODO: how much typechecking?
        assert(type(self._cls_ordinals) is dict)
        managers = []
        try:
            tag = ''
            while True:
                tag, version, data = serialize.deserialize(stream)
                if tag is None:
                    break
                if tag not in self._state_managers:
                    continue
                manager = self._state_managers[tag]
                manager.restore_snapshot(State.CREATE_PHASE, self, version, data)
                managers.append((tag, manager, version, data))
        except RestoreError as e:
            e.args = ("While restoring phase1 %s: %s" % (tag, e.args[0]),)
            raise
        for tag, manager, version, data in managers:
            try:
                manager.restore_snapshot(State.RESOLVE_PHASE, self, version, data)
            except RestoreError as e:
                e.args = ("While restoring phase2 %s: %s" % (tag, e.args[0]),)
                raise

    def read_metadata(self, stream, skip_version=False):
        """Deserialize session metadata from stream."""
        if not skip_version:
            version = serialize.deserialize(stream)
        metadata = serialize.deserialize(stream)
        if skip_version:
            return metadata
        return version, metadata


def save(session, filename, **kw):
    """command line version of saving a session"""
    my_open = None
    if hasattr(filename, 'write'):
        # called via export, it's really a stream
        output = filename
    else:
        from os.path import expanduser
        filename = expanduser(filename)         # Tilde expansion
        if not filename.endswith(SESSION_SUFFIX):
            filename += SESSION_SUFFIX
        from .safesave import SaveBinaryFile, SaveFile
        my_open = SaveBinaryFile
        try:
            # default to saving compressed files
            import gzip
            filename += ".gz"

            def my_open(filename):
                return SaveFile(
                    filename,
                    open=lambda filename: gzip.GzipFile(filename, 'wb'))
        except ImportError:
            pass
        try:
            output = my_open(filename)
        except IOError as e:
            from .errors import UserError
            raise UserError(e)

    try:
        session.save(output)
    except:
        if my_open is not None:
            output.close("exceptional")
        raise
    finally:
        if my_open is not None:
            output.close()


from .commands import CmdDesc, StringArg, register
@register('sdump', CmdDesc(required=[('filename', StringArg)],
                           optional=[('output', StringArg)],
                           synopsis="create human-readable session"))
def dump(session, filename, output=None):
    """dump contents of session for debugging"""
    if not filename.endswith(SESSION_SUFFIX):
        filename += SESSION_SUFFIX
    input = None
    try:
        input = _builtin_open(filename, 'rb')
    except IOError:
        filename2 = filename + '.gz'
        try:
            import gzip
            input = gzip.GzipFile(filename2, 'rb')
        except ImportError:
            import os
            if os.exists(filename2):
                session.logger.error("Unable to open compressed files: %s"
                                     % filename2)
                return
        except IOError:
            pass
        if input is None:
            session.logger.error(
                "Unable to find compressed nor uncompressed file: %s"
                % filename)
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
        "Chimera session", io.SESSION, SESSION_SUFFIX,
        prefixes="ses",
        mime="application/x-chimera2-session",
        reference="http://www.rbvi.ucsf.edu/chimera/",
        open_func=open, export_func=save)
_initialize()


class Selection:

    def __init__(self, all_models):
        self._all_models = all_models

    def all_models(self):
        return self._all_models.list()

    def models(self):
        return [m for m in self.all_models() if m.any_part_selected()]

    def items(self, itype):
        si = []
        for m in self.models():
            s = m.selected_items(itype)
            si.extend(s)
        return si

    def empty(self):
        for m in self.all_models():
            if m.any_part_selected():
                return False
        return True

    def clear(self):
        for m in self.models():
            m.clear_selection()

    def clear_hierarchy(self):
        for m in self.models():
            m.clear_selection_promotion_history()

    def promote(self):
        for m in self.models():
            m.promote_selection()

    def demote(self):
        for m in self.models():
            m.demote_selection()


def common_startup(sess):
    """Initialize session with common data managers"""
    assert(hasattr(sess, 'app_name'))
    assert(hasattr(sess, 'debug'))
    from . import triggerset
    sess.triggers = triggerset.TriggerSet()
    from .core_triggers import register_core_triggers
    register_core_triggers(sess.triggers)
    sess.scenes = Scenes(sess)
    sess.add_state_manager('scenes', sess.scenes)
    from . import models
    sess.models = models.Models(sess)
    sess.add_state_manager('models', sess.models)
    sess.selection = Selection(sess.models)
    from . import colors
    sess.user_colors = colors.UserColors()
    sess.add_state_manager('user_colors', sess.user_colors)
    sess.user_colormaps = colors.UserColormaps()
    sess.add_state_manager('user_colormaps', sess.user_colormaps)
    from .graphics.view import View
    sess.main_view = View(sess.models.drawing, window_size = (256, 256),
                          trigger_set = sess.triggers)
    try:
        from .core_settings import settings
        sess.main_view.background_color = settings.bg_color.rgba
    except ImportError:
        pass
    from .graphics.gsession import ViewState
    sess.add_state_manager('view', ViewState(sess.main_view))
    from .updateloop import UpdateLoop
    sess.update_loop = UpdateLoop()
    from .atomic import PseudobondManager, ChangeTracker, LevelOfDetail
    sess.change_tracker = ChangeTracker()
    sess.pb_manager = PseudobondManager(sess.change_tracker)
    sess.atomic_level_of_detail = LevelOfDetail()

    from . import commands
    commands.register_core_commands(sess)
    commands.register_core_selectors(sess)

    _register_core_file_formats()

def _register_core_file_formats():
    from . import stl
    stl.register()
    from .atomic import pdb
    pdb.register()
    from .atomic import mmcif
    mmcif.register()
    from . import scripting
    scripting.register()
    from . import map
    map.register_map_file_readers()
    map.register_eds_fetch()
    map.register_emdb_fetch()
    from .atomic import readpbonds
    readpbonds.register()
