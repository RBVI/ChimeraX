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

from .state import RestoreError, State, copy_state, dereference_state
from .commands import CmdDesc, StringArg, register, commas

_builtin_open = open
#: session file suffix
SESSION_SUFFIX = ".cxs"

# triggers:

#: version of session file that is written
CORE_SESSION_VERSION = 1


class _UniqueName:
    # uids are (class_name, ordinal)
    # The class_name is either the name of a core class
    # or a tuple (tool name, class name)

    _cls_ordinals = {}  # {class: ordinal}
    _uid_to_obj = {}    # {uid: obj}
    _obj_to_uid = {}    # {id(obj): uid}
    _tool_infos = set()

    __slots__ = ['uid']

    @classmethod
    def reset(cls):
        cls._cls_ordinals.clear()
        cls._uid_to_obj.clear()
        cls._obj_to_uid.clear()

    @classmethod
    def lookup(cls, unique_name):
        return cls._uid_to_obj.get(unique_name.uid, None)

    @classmethod
    def add(cls, uid, obj):
        # allows for non-_UniqueName uids
        cls._uid_to_obj[uid] = obj
        cls._obj_to_uid[id(obj)] = uid

    def obj(self):
        return self._uid_obj[self.uid]

    def __init__(self, uid):
        self.uid = uid

    @classmethod
    def from_obj(cls, obj):
        """Return a unique identifier for an object in session
        Consequently, the identifier is composed of simple data types.

        Parameters
        ----------
        obj : any object
        tool_info : optional :py:class:`~chimera.core.toolshed.ToolInfo` instance
            Explicitly denote which tool object comes from.
        """

        uid = cls._obj_to_uid.get(id(obj), None)
        if uid is not None:
            return cls(uid)
        obj_cls = obj.__class__
        tool_info = None
        if hasattr(obj, 'tool_info'):
            tool_info = obj.tool_info
        elif hasattr(obj_cls, 'tool_info'):
            tool_info = obj_cls.tool_info
        if tool_info is None:
            # no tool info, must be in core
            if not obj_cls.__module__.startswith('chimera.core.'):
                raise RuntimeError('No tool information for %s.%s' % (
                    obj_cls.__module__, obj_cls.__name__))
            class_name = obj_cls.__name__
            # double check that class will be able to be restored
            from chimera.core import get_class
            if obj_cls != get_class(class_name):
                raise RuntimeError('unable to restore objects of %s class' % class_name)
        else:
            class_name = (tool_info.name, obj_cls.__name__)
            # double check that class will be able to be restored
            if obj_cls != tool_info.get_class(obj_cls.__name__):
                raise RuntimeError(
                    'unable to restore objects of %s class in %s tool' %
                    (class_name, tool_info.name))

        ordinal = cls._cls_ordinals.get(class_name, 0)
        if ordinal == 0 and tool_info is not None:
            cls._tool_infos.add(tool_info)
        ordinal += 1
        uid = (class_name, ordinal)
        cls._cls_ordinals[class_name] = ordinal
        cls._uid_to_obj[uid] = obj
        cls._obj_to_uid[id(obj)] = uid
        return cls(uid)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.uid)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def class_name_of(self):
        """Extract class name associated with unique id for messages"""
        class_name = self.uid[0]
        if isinstance(class_name, str):
            return "Core's %s" % class_name
        return "Tool %s's %s" % class_name

    def tool_info_and_class_of(self, session):
        """Return class associated with unique id

        The restore process makes sure that the right tools are present,
        so if there is an error, return None.
        """
        class_name, ordinal = self.uid
        if isinstance(class_name, str):
            from chimera.core import get_class
            tool_info = None
            cls = get_class(class_name)
        else:
            tool_name, class_name = class_name
            tool_info = session.toolshed.find_tool(tool_name)
            if tool_info is None:
                return None, None
            cls = tool_info.get_class(class_name)
        return tool_info, cls

#    @classmethod
#    def restore_unique_id(self, obj, uid):
#        """Restore unique identifier for an object"""
#        class_name, ordinal = uid
#        obj._cache_uid = ordinal
#        self._uid_to_obj[uid] = obj
#        current_ordinal = self._cls_ordinals.get(class_name, 0)
#        if ordinal > current_ordinal:
#            self._cls_ordinals[class_name] = ordinal


class _SaveManager:
    """Manage session saving"""

    def __init__(self, session, state_flags):
        self.session = session
        self.state_flags = state_flags
        self.graph = {}         # dependency graph
        self.unprocessed = []   # item: LIFO [obj]
        self.processed = {}     # item: {_UniqueName(obj)/attr_name: data}
        self._found_objs = []
        _UniqueName.reset()

    def discovery(self, containers):
        for key, value in containers.items():
            if not isinstance(value, State) and len(value) == 0:
                continue
            try:
                if not isinstance(value, State):
                    self.processed[key] = self.process(value)
                    self.graph[key] = self._found_objs
                else:
                    self.unprocessed.append(value)
                    uid = _UniqueName.from_obj(value)
                    self.processed[key] = uid
                    self.graph[key] = [uid]
            except ValueError as e:
                raise ValueError("error processing: %r" % key)
        while self.unprocessed:
            obj = self.unprocessed.pop()
            key = _UniqueName.from_obj(obj)
            try:
                self.processed[key] = self.process(obj)
            except ValueError as e:
                raise ValueError("error processing key: %s: %s" % (key, e))
            self.graph[key] = self._found_objs

    def _add_obj(self, obj):
        uid = _UniqueName.from_obj(obj)
        self._found_objs.append(uid)
        if uid not in self.processed:
            self.unprocessed.append(obj)
        return uid

    def process(self, obj):
        self._found_objs = []
        if isinstance(obj, State):
            data = obj.take_snapshot(self.session, self.state_flags)
        else:
            data = obj
        return copy_state(data, convert=self._add_obj)

    def walk(self):
        # generator that walks processed items in correct order
        from .order_dag import order_dag
        odg = order_dag(self.graph)
        while 1:
            try:
                key = next(odg)
                value = self.processed[key]
                yield key, value
            except StopIteration:
                return

    def tool_infos(self):
        tool_infos = {}
        for ti in _UniqueName._tool_infos:
            tool_infos[ti.name] = (ti.version, ti.session_write_version)
        return tool_infos


class _RestoreManager:

    #: common exception for needing a newer version of the application
    NeedNewerError = RestoreError(
        "Need newer version of tool to restore session")

    def __init__(self):
        _UniqueName.reset()
        self.tool_infos = {}

    def check_tools(self, session, tool_infos):
        missing_tools = []
        out_of_date_tools = []
        for tool_name, (tool_version, tool_state_version) in tool_infos.items():
            t = session.toolshed.find_tool(tool_name)
            if t is None:
                missing_tools.append(tool_name)
                continue
            # check if installed tools can restore data
            if tool_state_version not in t.session_versions:
                out_of_date_tools.append(t)
        if missing_tools or out_of_date_tools:
            msg = "Unable to restore session"
            if missing_tools:
                msg += "; missing tools: " + commas(missing_tools, ' and')
            if out_of_date_tools:
                msg += "; out of date tools: " + commas(out_of_date_tools, ' and')
            raise RestoreError(msg)
        self.tool_infos = tool_infos

    def resolve_references(self, data):
        # resolve references in data
        return dereference_state(data, _UniqueName.lookup, _UniqueName)

    def add_reference(self, name, obj):
        _UniqueName.add(name.uid, obj)


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

    Attributes
    ----------
    app_bin_dir : string
        Application executable binaries directory
    app_data_dir : string
        Application data directory
    app_lib_dir : string
        Application shared code library directory
    logger : An instance of :py:class:`~chimera.core.logger.Logger`
        Use to log information, warning, errors.
    metadata : dict
        Information kept at beginning of session file, eg., a thumbnail
    models : Instance of :py:class:`~chimera.core.models.Models`.
    triggers : An instance of :py:class:`~chimera.core.triggerset.TriggerSet`
        Starts with session triggers.
    """

    def __init__(self, app_name, *, debug=False, minimal=False):
        self.app_name = app_name
        self.debug = debug
        from . import logger
        self.logger = logger.Logger(self)
        from . import triggerset
        self.triggers = triggerset.TriggerSet()
        self._state_containers = {}  # stuff to save in sessions
        self.metadata = {}           #: session metadata
        if minimal:
            return

        import chimera
        self.app_data_dir = chimera.app_data_dir
        self.app_bin_dir = chimera.app_bin_dir
        self.app_lib_dir = chimera.app_lib_dir

        # initialize state managers for various properties
        from . import models
        self.add_state_manager('models', models.Models(self))
        from .graphics.view import View
        self.main_view = View(self.models.drawing, window_size=(256, 256),
                              trigger_set=self.triggers)
        from .graphics.gsession import ViewState
        self.add_state_manager('view', ViewState(self, 'main_view'))

        from . import colors
        self.add_state_manager('user_colors', colors.UserColors())
        self.add_state_manager('user_colormaps', colors.UserColormaps())
        # tasks and tools are initialized later
        # TODO: scenes need more work
        # from .scenes import Scenes
        # sess.add_state_manager('scenes', Scenes(sess))

    @property
    def models(self):
        return self._state_containers['models']

    @property
    def scenes(self):
        return self._state_containers['scenes']

    @property
    def tools(self):
        return self._state_containers['tools']

    @property
    def tasks(self):
        return self._state_containers['tasks']

    @property
    def user_colors(self):
        return self._state_containers['user_colors']

    @property
    def user_colormaps(self):
        return self._state_containers['user_colormaps']

    def reset(self):
        """Reset session to data-less state"""
        self.metadata.clear()
        for tag in self._state_containers:
            container = self._state_containers[tag]
            if isinstance(container, State):
                container.reset_state(self)
            else:
                container.clear()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # preemptive debugging for third party packages
            raise AttributeError("attribute already set")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # preemptive debugging for third party packages
        raise AttributeError("can not remove attributes")

    def add_state_manager(self, tag, container):
        # if tag in self._state_containers:
        #     return
        if not isinstance(container, State) and not hasattr(container, 'clear'):
            raise ValueError('container must follow State API')
        self._state_containers[tag] = container

    def get_state_manager(self, tag):
        return self._state_containers[tag]

    def replace_attribute(self, attribute_name, value):
        """Explictly replace attribute with alternate implementation"""
        object.__setattr__(self, attribute_name, value)

    def save(self, stream):
        """Serialize session to stream."""
        from . import serialize
        self.triggers.activate_trigger("begin save session", self)
        serialize.serialize(stream, CORE_SESSION_VERSION)
        serialize.serialize(stream, self.metadata)
        # guarantee that tools are serialized first, so on restoration,
        # all of the related code will be loaded before the rest of the
        # session is restored
        mgr = _SaveManager(self, State.SESSION)
        mgr.discovery(self._state_containers)
        serialize.serialize(stream, mgr.tool_infos())
        # TODO: collect OrderDAGError exceptions from walk and analyze
        for name, data in mgr.walk():
            serialize.serialize(stream, name)
            serialize.serialize(stream, data)
        serialize.serialize(stream, None)
        self.triggers.activate_trigger("end save session", self)

    def restore(self, stream, version=None):
        """Deserialize session from stream."""
        from . import serialize
        skip_over_metadata = version is not None
        if not skip_over_metadata:
            version = serialize.deserialize(stream)
        if version > CORE_SESSION_VERSION:
            raise State.NeedNewerError()
        if not skip_over_metadata:
            metadata = self.read_metadata(stream, skip_version=True)
        else:
            metadata = None
        mgr = _RestoreManager()
        tool_infos = serialize.deserialize(stream)
        mgr.check_tools(self, tool_infos)

        self.reset()
        self.triggers.activate_trigger("begin restore session", self)
        if metadata is not None:
            self.metadata.update(metadata)
        while True:
            name = serialize.deserialize(stream)
            if name is None:
                break
            data = serialize.deserialize(stream)
            data = mgr.resolve_references(data)
            if isinstance(name, str):
                self.add_state_manager(name, data)
            else:
                # _UniqueName: find class
                tool_info, cls = name.tool_info_and_class_of(self)
                if cls is None:
                    continue
                cls_version, cls_data = data
                obj = cls.restore_snapshot_new(self, tool_info, cls_version, cls_data)
                obj.restore_snapshot_init(self, tool_info, cls_version, cls_data)
                mgr.add_reference(name, obj)
        self.triggers.activate_trigger("end restore session", self)

    def read_metadata(self, stream, skip_version=False):
        """Deserialize session metadata from stream."""
        from . import serialize
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

    # TODO: put thumbnail in session metadata
    try:
        session.save(output)
    except:
        if my_open is not None:
            output.close("exceptional")
        raise
    finally:
        if my_open is not None:
            output.close()


def dump(session, session_file, output=None):
    """dump contents of session for debugging"""
    from . import serialize
    if not session_file.endswith(SESSION_SUFFIX):
        session_file += SESSION_SUFFIX
    input = None
    from .io import open_filename
    from .errors import UserError
    try:
        input = open_filename(session_file, 'rb')
    except UserError:
        session_file2 = session_file + '.gz'
        try:
            input = open_filename(session_file2, 'rb')
        except UserError:
            pass
        if input is None:
            session.logger.error(
                "Unable to find compressed nor uncompressed file: %s"
                % session_file)
            return
    if output is not None:
        output = open_filename(output, 'w')
    from pprint import pprint
    with input:
        print("session version:", file=output)
        version = serialize.deserialize(input)
        pprint(version, stream=output)
        print("session metadata:", file=output)
        metadata = serialize.deserialize(input)
        pprint(metadata, stream=output)
        print("tool info:", file=output)
        tool_infos = serialize.deserialize(input)
        pprint(tool_infos, stream=output)
        while True:
            name = serialize.deserialize(input)
            if name is None:
                break
            data = serialize.deserialize(input)
            print('name/uid:', name, file=output)
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
        mime="application/x-chimerax-session",
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
    """Initialize session with common data containers"""
    from .core_triggers import register_core_triggers
    register_core_triggers(sess.triggers)
    sess.selection = Selection(sess.models)
    try:
        from .core_settings import settings
        sess.main_view.background_color = settings.bg_color.rgba
    except ImportError:
        pass
    from .updateloop import UpdateLoop
    sess.update_loop = UpdateLoop()
    from .atomic import PseudobondManager, ChangeTracker, LevelOfDetail
    sess.change_tracker = ChangeTracker()
    sess.pb_manager = PseudobondManager(sess)
    sess.atomic_level_of_detail = LevelOfDetail()

    from . import commands
    commands.register_core_commands(sess)
    commands.register_core_selectors(sess)

    register(
        'sdump',
        CmdDesc(required=[('session_file', StringArg)],
                optional=[('output', StringArg)],
                synopsis="create human-readable session"),
        dump
    )

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
    from .surface import collada
    collada.register()
