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
from .commands import CmdDesc, OpenFileNameArg, SaveFileNameArg, register, commas

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
    _bundle_infos = set()

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
        bundle_info : optional :py:class:`~chimerax.core.toolshed.BundleInfo` instance
            Explicitly denote which tool object comes from.
        """

        uid = cls._obj_to_uid.get(id(obj), None)
        if uid is not None:
            return cls(uid)
        obj_cls = obj.__class__
        bundle_info = None
        if hasattr(obj, 'bundle_info'):
            bundle_info = obj.bundle_info
        elif hasattr(obj_cls, 'bundle_info'):
            bundle_info = obj_cls.bundle_info
        if bundle_info is None:
            # no tool info, must be in core
            if not obj_cls.__module__.startswith('chimerax.core.'):
                raise RuntimeError('No tool information for %s.%s' % (
                    obj_cls.__module__, obj_cls.__name__))
            class_name = obj_cls.__name__
            # double check that class will be able to be restored
            from chimerax.core import get_class
            if obj_cls != get_class(class_name):
                raise RuntimeError('Will not be able to restore objects of %s class' % class_name)
        else:
            class_name = (bundle_info.name, obj_cls.__name__)
            # double check that class will be able to be restored
            if obj_cls != bundle_info.get_class(obj_cls.__name__):
                raise RuntimeError(
                    'unable to restore objects of %s class in %s tool' %
                    (class_name, bundle_info.name))

        ordinal = cls._cls_ordinals.get(class_name, 0)
        if ordinal == 0 and bundle_info is not None:
            cls._bundle_infos.add(bundle_info)
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

    def class_of(self, session):
        """Return class associated with unique id

        The restore process makes sure that the right bundles are present,
        so if there is an error, return None.
        """
        class_name, ordinal = self.uid
        if isinstance(class_name, str):
            from chimerax.core import get_class
            cls = get_class(class_name)
        else:
            tool_name, class_name = class_name
            bundle_info = session.toolshed.find_bundle(tool_name)
            if bundle_info is None:
                cls = None
            else:
                cls = bundle_info.get_class(class_name)
        return cls

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
            sm = self.session.snapshot_methods(value)
            if sm is None and len(value) == 0:
                continue
            try:
                if sm is None:
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
        sm = self.session.snapshot_methods(obj)
        if sm:
            # TODO: if data is None or exception, log failure
            data = sm.take_snapshot(obj, self.session, self.state_flags)
            if data is None:
                raise RuntimeError('take_snapshot() for "%s" instance returned None' % obj.__class__.__name__)
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

    def bundle_infos(self):
        bundle_infos = {}
        for bi in _UniqueName._bundle_infos:
            bundle_infos[bi.name] = (bi.version, bi.session_write_version)
        return bundle_infos


class _RestoreManager:

    #: common exception for needing a newer version of the application
    NeedNewerError = RestoreError(
        "Need newer version of tool to restore session")

    def __init__(self):
        _UniqueName.reset()
        self.bundle_infos = {}

    def check_bundles(self, session, bundle_infos):
        missing_bundles = []
        out_of_date_bundles = []
        for tool_name, (tool_version, tool_state_version) in bundle_infos.items():
            t = session.toolshed.find_bundle(tool_name)
            if t is None:
                missing_bundles.append(tool_name)
                continue
            # check if installed bundles can restore data
            if tool_state_version not in t.session_versions:
                out_of_date_bundles.append(t)
        if missing_bundles or out_of_date_bundles:
            msg = "Unable to restore session"
            if missing_bundles:
                msg += "; missing bundles: " + commas(missing_bundles, ' and')
            if out_of_date_bundles:
                msg += "; out of date bundles: " + commas(out_of_date_bundles, ' and')
            raise RestoreError(msg)
        self.bundle_infos = bundle_infos

    def resolve_references(self, data):
        # resolve references in data
        return dereference_state(data, _UniqueName.lookup, _UniqueName)

    def add_reference(self, name, obj):
        _UniqueName.add(name.uid, obj)


class Session:
    """Session management

    The metadata attribute should be a dictionary with information about
    the session, e.g., a thumbnail, a description, the author, etc.

    To preemptively detect problems where different bundles want to use the same
    session attribute, session attributes may only be assigned to once,
    and may not be deleted.
    Attributes that support the State API are included
    Consequently, each attribute is an instance that supports the State API.

    Each session attribute, that should be archived,
    must implement the State API, and is then automatically archived.

    Attributes
    ----------
    logger : An instance of :py:class:`~chimerax.core.logger.Logger`
        Use to log information, warning, errors.
    metadata : dict
        Information kept at beginning of session file, eg., a thumbnail
    models : Instance of :py:class:`~chimerax.core.models.Models`.
    triggers : An instance of :py:class:`~chimerax.core.triggerset.TriggerSet`
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
        self.in_script = InScriptFlag()
        if minimal:
            return

        # initialize state managers for various properties
        from . import models
        self.add_state_manager('models', models.Models(self))
        from .graphics.view import View
        self.main_view = View(self.models.drawing, window_size=(256, 256),
                              trigger_set=self.triggers)
        self.add_state_manager('view', self.main_view)

        from . import colors
        self.add_state_manager('user_colors', colors.UserColors())
        self.add_state_manager('user_colormaps', colors.UserColormaps())
        # tasks and bundles are initialized later
        # TODO: scenes need more work
        # from .scenes import Scenes
        # sess.add_state_manager('scenes', Scenes(sess))

    @property
    def models(self):
        return self._state_containers['models']

    @property
    def pb_manager(self):
        return self._state_containers['pb_manager']

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
            sm = self.snapshot_methods(container)
            if sm:
                sm.reset_state(container, self)
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
        sm = self.snapshot_methods(container)
        if sm is None and not hasattr(container, 'clear'):
            raise ValueError('container "%s" of type "%s" does not have snapshot methods and does not have clear method' % (tag, str(type(container))))
        self._state_containers[tag] = container

    def get_state_manager(self, tag):
        return self._state_containers[tag]

    def delete_state_manager(self, tag):
        del self._state_containers[tag]

    def replace_attribute(self, attribute_name, value):
        """Explictly replace attribute with alternate implementation"""
        object.__setattr__(self, attribute_name, value)

    def delete_attribute(self, attribute_name):
        """Explictly delete attribute"""
        object.__delattr__(self, attribute_name)

    def snapshot_methods(self, object, instance = True):
        """Return an object having take_snapshot() and restore_snapshot() methods for the given object.
        Can return if no save/restore methods are available, for instance for primitive types.
        """
        cls = object.__class__ if instance else object
        if issubclass(cls, State):
            return cls
        elif not hasattr(self, '_snapshot_methods'):
            from .graphics import View, MonoCamera, Lighting, Material, ClipPlane, Drawing, gsession as g
            from .geometry import Place, Places, psession as p
            self._snapshot_methods = {
                View : g.ViewState,
                MonoCamera : g.CameraState,
                Lighting : g.LightingState,
                Material : g.MaterialState,
                ClipPlane : g.ClipPlaneState,
                Drawing : g.DrawingState,
                Place : p.PlaceState,
                Places : p.PlacesState,
            }

        methods = self._snapshot_methods.get(cls, None)
        return methods
    
    def save(self, stream):
        """Serialize session to stream."""
        from . import serialize
        self.triggers.activate_trigger("begin save session", self)
        serialize.serialize(stream, CORE_SESSION_VERSION)
        serialize.serialize(stream, self.metadata)
        # guarantee that bundles are serialized first, so on restoration,
        # all of the related code will be loaded before the rest of the
        # session is restored
        mgr = _SaveManager(self, State.SESSION)
        mgr.discovery(self._state_containers)
        serialize.serialize(stream, mgr.bundle_infos())
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
        bundle_infos = serialize.deserialize(stream)
        mgr.check_bundles(self, bundle_infos)

        self.triggers.activate_trigger("begin restore session", self)
        try:
            self.reset()
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
                    cls = name.class_of(self)
                    if cls is None:
                        continue
                    sm = self.snapshot_methods(cls, instance = False)
                    if sm is None:
                        print('no snapshot methods for class', cls.__name__)
                    obj = sm.restore_snapshot(self, data)
                    mgr.add_reference(name, obj)
        except:
            import traceback
            self.logger.error("Unable to restore session, resetting.\n\n%s"
                              % traceback.format_exc())
            self.reset()
        finally:
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

class InScriptFlag:
    def __init__(self):
        self._level = 0
    def __enter__(self):
        self._level += 1
    def __exit__(self, *_):
        self._level -= 1
    def __bool__(self):
        return self._level > 0

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

    # Associate thumbnail image with session file for display by operating system file browser.
    from . import utils
    if isinstance(filename, str) and utils.can_set_file_icon():
        width = height = 512
        image = session.main_view.image(width, height)
        utils.set_file_icon(filename, image)

    # Remember session in file history
    if isinstance(filename, str):
        from .filehistory import remember_file
        remember_file(session, filename, 'ses', 'all models', file_saved = True)

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
        bundle_infos = serialize.deserialize(input)
        pprint(bundle_infos, stream=output)
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
    return [], "opened ChimeraX session"


def _initialize():
    from . import io
    io.register_format(
        "ChimeraX session", io.SESSION, SESSION_SUFFIX, ("ses",),
        mime="application/x-chimerax-session",
        reference="http://www.rbvi.ucsf.edu/chimerax/",
        open_func=open, export_func=save)
_initialize()


def common_startup(sess):
    """Initialize session with common data containers"""

    from .core_triggers import register_core_triggers
    register_core_triggers(sess.triggers)

    from .selection import Selection
    sess.selection = Selection(sess.models)

    try:
        from .core_settings import settings
        sess.main_view.background_color = settings.bg_color.rgba
    except ImportError:
        pass

    from .updateloop import UpdateLoop
    sess.update_loop = UpdateLoop()

    from .atomic import ChangeTracker, LevelOfDetail
    sess.change_tracker = ChangeTracker()
    # change_tracker needs to exist before global pseudobond manager
    # can be created
    from .atomic import PseudobondManager
    sess.add_state_manager('pb_manager', PseudobondManager(sess))
    sess.atomic_level_of_detail = LevelOfDetail()

    from . import commands
    commands.register_core_commands(sess)
    commands.register_core_selectors(sess)

    register(
        'sdump',
        CmdDesc(required=[('session_file', OpenFileNameArg)],
                optional=[('output', SaveFileNameArg)],
                synopsis="create human-readable session"),
        dump
    )

    _register_core_file_formats()
    _register_core_database_fetch(sess)

def _register_core_file_formats():
    from . import stl
    stl.register()
    from .atomic import pdb
    pdb.register_pdb_format()
    from .atomic import mmcif
    mmcif.register_mmcif_format()
    from . import scripting
    scripting.register()
    from . import map
    map.register_map_file_readers()
    from .atomic import readpbonds
    readpbonds.register_pbonds_format()
    from .surface import collada
    collada.register_collada_format()

def _register_core_database_fetch(session):
    s = session
    from .atomic import pdb
    pdb.register_pdb_fetch(s)
    from .atomic import mmcif
    mmcif.register_mmcif_fetch(s)
    from . import map
    map.register_eds_fetch(s)
    map.register_emdb_fetch(s)
