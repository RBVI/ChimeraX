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

Session data, ie., data that is archived, uses the :py:class:`State` and
:py:class:`StateManager` API.
"""

from .state import RestoreError, State, StateManager, copy_state, dereference_state
from .commands import CmdDesc, OpenFileNameArg, SaveFileNameArg, register, commas, plural_form
from .errors import UserError

_builtin_open = open
#: session file suffix
SESSION_SUFFIX = ".cxs"

# List of type objects that are in bundle "builtins"
BUILTIN_TYPES = frozenset((bool, bytearray, bytes, complex, dict, frozenset, int, float, list, range, set, slice, str, tuple))


class _UniqueName:
    # uids are (class_name, ordinal).
    # The class_name is either the name of a core class, the word builtin
    # following by a supported builtin type's name,
    # or a tuple (bundle name, class name).
    # If the ordinal is zero, then the uid refers to class object.

    _cls_info = {}  # {class: [bundle_info, class_name, ordinal]}
    _uid_to_obj = {}    # {uid: obj}
    _obj_to_uid = {}    # {id(obj): uid}
    _bundle_infos = set()

    __slots__ = ['uid']

    @classmethod
    def reset(cls):
        cls._cls_info.clear()
        cls._uid_to_obj.clear()
        cls._obj_to_uid.clear()
        for b in BUILTIN_TYPES:
            cls._uid_to_obj[('builtin %s' % b.__name__, 0)] = b

    @classmethod
    def lookup(cls, unique_name):
        uid = unique_name.uid
        return cls._uid_to_obj.get(uid, None)

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
    def from_obj(cls, session, obj):
        """Return a unique identifier for an object in session
        Consequently, the identifier is composed of simple data types.

        Parameters
        ----------
        session : instance of :py:class:`Session`
        obj : any object
        """

        uid = cls._obj_to_uid.get(id(obj), None)
        if uid is not None:
            return cls(uid)
        if isinstance(obj, type):
            bundle_info = session.toolshed.find_bundle_for_class(obj)
            if bundle_info is not None:
                # double check that class will be able to be restored
                if obj != bundle_info.get_class(obj.__name__, session.logger):
                    raise RuntimeError(
                        'Unable to restore %s class object from %s bundle' %
                        (obj.__name__, bundle_info.name))
                class_name = (bundle_info.name, obj.__name__)
                if obj not in cls._cls_info:
                    cls._cls_info[obj] = bundle_info, class_name, 0
            elif obj.__module__ == 'builtins' and obj in BUILTIN_TYPES:
                class_name = 'builtin %s' % obj.__name__
            else:
                raise RuntimeError(
                    'Unable to restore %s.%s type instance' %
                    (obj.__module__, obj.__name__))
            uid = (class_name, 0)
            cls._uid_to_obj[uid] = obj
            cls._obj_to_uid[id(obj)] = uid
            return cls(uid)

        obj_cls = obj.__class__
        bundle_info, class_name, ordinal = cls._cls_info.get(obj_cls, (None, None, 0))
        known_class = bundle_info is not None
        if not known_class:
            bundle_info = session.toolshed.find_bundle_for_class(obj_cls)
            if bundle_info is None:
                raise RuntimeError('No bundle information for %s.%s' % (
                    obj_cls.__module__, obj_cls.__name__))

        if class_name is None:
            from . import BUNDLE_NAME
            if bundle_info.name == BUNDLE_NAME:
                class_name = obj_cls.__name__
            else:
                class_name = (bundle_info.name, obj_cls.__name__)
            # double check that class will be able to be restored
            if obj_cls != bundle_info.get_class(obj_cls.__name__, session.logger):
                raise RuntimeError(
                    'Unable to restore objects of %s class in %s bundle'
                    ' because the class name is not listed in the name to class table'
                    ' for session restore' %
                    (obj_cls.__name__, bundle_info.name))

        if not known_class and bundle_info != 'builtin':
            cls._bundle_infos.add(bundle_info)
        ordinal += 1
        uid = (class_name, ordinal)
        cls._cls_info[obj_cls] = bundle_info, class_name, ordinal
        cls._uid_to_obj[uid] = obj
        cls._obj_to_uid[id(obj)] = uid
        return cls(uid)

    def __repr__(self):
        return '<%r, %r>' % self.uid

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
            if ' ' in class_name:
                return class_name  # 'builtin int', etc.
            return "Core's %s" % class_name
        return "Bundle %s's %s" % class_name

    def class_of(self, session):
        """Return class associated with unique id

        The restore process makes sure that the right bundles are present,
        so if there is an error, return None.
        """
        class_name, ordinal = self.uid
        if isinstance(class_name, str):
            from . import bundle_api
            cls = bundle_api.get_class(class_name)
        else:
            bundle_name, class_name = class_name
            bundle_info = session.toolshed.find_bundle(bundle_name, session.logger)
            if bundle_info is None:
                cls = None
            else:
                cls = bundle_info.get_class(class_name, session.logger)
        return cls


def _obj_chain(parents, obj):
    # string representation of chain of objects
    # first element of chain is string with name of state manager
    chain = []
    for o in parents + (obj,):
        name = None
        if hasattr(o, 'name'):
            try:
                if callable(o.name):
                    name = o.name()
                else:
                    name = o.name
            except Exception:
                pass
        if name is None:
            chain.append(repr(o))
        else:
            chain.append(repr(o) + " %r" % o.name)
    return " -> ".join(chain)


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

    def cleanup(self):
        # remove references
        self.graph.clear()
        self.unprocessed.clear()
        self.processed.clear()
        self._found_objs.clear()
        _UniqueName.reset()

    def discovery(self, containers):
        for key, value in containers.items():
            sm = self.session.snapshot_methods(value)
            if sm is None and len(value) == 0:
                continue
            try:
                if sm is None:
                    self.processed[key] = self.process(value, (key,))
                    self.graph[key] = self._found_objs
                else:
                    self.unprocessed.append((value, (key,)))
                    uid = _UniqueName.from_obj(self.session, value)
                    self.processed[key] = uid
                    self.graph[key] = [uid]
            except ValueError:
                raise ValueError("error processing state container: %r" % key)
        while self.unprocessed:
            obj, parents = self.unprocessed.pop()
            key = _UniqueName.from_obj(self.session, obj)
            if key not in self.processed:
                try:
                    self.processed[key] = self.process(obj, parents)
                except ValueError as e:
                    raise ValueError("error processing: %s: %s" % (_obj_chain(parents, obj), e))
                self.graph[key] = self._found_objs

    def _add_obj(self, obj, parents=()):
        uid = _UniqueName.from_obj(self.session, obj)
        self._found_objs.append(uid)
        if uid not in self.processed:
            self.unprocessed.append((obj, parents))
        return uid

    def process(self, obj, parents):
        self._found_objs = []
        if isinstance(obj, type):
            return None
        data = None
        session = self.session
        sm = session.snapshot_methods(obj)
        if sm:
            try:
                data = sm.take_snapshot(obj, session, self.state_flags)
            except Exception as e:
                msg = 'Error while saving session data for %s: %s' % (_obj_chain(parents, obj), e)
                raise RuntimeError(msg)
        elif isinstance(obj, type):
            return None
        if data is None:
            session.logger.warning('Unable to save %s".  Session might not restore properly.'
                                   % _obj_chain(parents, obj))

        def convert(obj, parents=parents + (obj,), add_obj=self._add_obj):
            return add_obj(obj, parents)
        return copy_state(data, convert=convert)

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

    def __init__(self):
        _UniqueName.reset()
        self.bundle_infos = {}

    def cleanup(self):
        # remove references
        _UniqueName.reset()
        self.bundle_infos.clear()

    def check_bundles(self, session, bundle_infos):
        missing_bundles = []
        out_of_date_bundles = []
        for bundle_name, (bundle_version, bundle_state_version) in bundle_infos.items():
            # put the below kludge in to allow sessions saved before the seq_view
            # bundle name change to restore; remove on or after 1.0 release
            bi = session.toolshed.find_bundle(bundle_name, session.logger)
            if bi is None:
                missing_bundles.append(bundle_name)
                continue
            # check if installed bundles can restore data
            if bundle_state_version not in bi.session_versions:
                out_of_date_bundles.append(bi)
        if missing_bundles or out_of_date_bundles:
            msg = "Unable to restore all of session"
            if missing_bundles:
                msg += "; missing %s: %s" % (
                    plural_form(missing_bundles, 'bundle'),
                    commas(missing_bundles, 'and'))
            if out_of_date_bundles:
                msg += "; out of date %s: %s" % (
                    plural_form(out_of_date_bundles, 'bundle'),
                    commas(out_of_date_bundles, 'and'))
            raise UserError(msg)
        self.bundle_infos = bundle_infos

    def resolve_references(self, data):
        # resolve references in data
        return dereference_state(data, _UniqueName.lookup, _UniqueName)

    def add_reference(self, name, obj):
        _UniqueName.add(name.uid, obj)


class UserAliases(StateManager):

    ALIAS_STATE_VERSION = 1

    def reset_state(self, session):
        """Reset state to data-less state"""
        # keep all aliases
        pass

    def take_snapshot(self, session, flags):
        # only save user aliases
        from .commands.cli import list_aliases, expand_alias
        aliases = {}
        for name in list_aliases():
            aliases[name] = expand_alias(name)
        data = {
            'aliases': aliases,
            'version': self.ALIAS_STATE_VERSION,
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        from .commands.cli import create_alias
        aliases = data['aliases']
        for name, text in aliases.items():
            create_alias(name, text, user=True)
        obj = cls()
        return obj


class Session:
    """Supported API. Session management

    The metadata attribute should be a dictionary with information about
    the session, e.g., a thumbnail, a description, the author, etc.
    See :py:func:`standard_metadata`.

    Attributes that support the :py:class:`StateManager` API are automatically added as state managers
    (e.g. the session's add_state_manager method is called with the 'tag' argument
    the same as the attribute name).
    Conversely, deleting the attribute will call remove_state_manager.
    If a session attribute is not desired, the add/remove_state_manager methods can
    be called directly.

    Attributes
    ----------
    logger : An instance of :py:class:`~chimerax.core.logger.Logger`
        Used to log information, warning, errors.
    metadata : dict
        Information kept at beginning of session file, eg., a thumbnail
    models : Instance of :py:class:`~chimerax.core.models.Models`.
    triggers : An instance of :py:class:`~chimerax.core.triggerset.TriggerSet`
        Starts with session triggers.
    main_view : An instance of :py:class:`~chimerax.core.graphics.View`
        Default view.
    """

    def __init__(self, app_name, *, debug=False, silent=False, minimal=False):
        self.app_name = app_name
        self.debug = debug
        self.silent = silent
        from . import logger
        self.logger = logger.Logger(self)
        from . import triggerset
        self.triggers = triggerset.TriggerSet()
        self._state_containers = {}  # stuff to save in sessions
        self.metadata = {}           #: session metadata
        self.in_script = InScriptFlag()
        self.session_file_path = None  # Last saved or opened session file.
        if minimal:
            return

        # initialize state managers for various properties
        from . import models
        self.models = models.Models(self)
        from .graphics.view import View
        self.main_view = View(self.models.scene_root_model, window_size=(256, 256),
                              trigger_set=self.triggers)
        self.user_aliases = UserAliases()

        from . import colors
        self.user_colors = colors.UserColors()
        self.user_colormaps = colors.UserColormaps()
        # tasks and bundles are initialized later
        # TODO: scenes need more work
        # from .scenes import Scenes
        # sess.add_state_manager('scenes', Scenes(sess))

        self.save_options = {}          # Options used when saving session files.
        self.restore_options = {}       # Options used when restoring session files.

    def _get_view(self):
        return self._state_containers['main_view']

    def _set_view(self, view):
        self._state_containers['main_view'] = view
    view = property(_get_view, _set_view)

    # TODO:
    # @property
    # def scenes(self):
    #     return self._state_containers['scenes']

    def reset(self):
        """Reset session to data-less state"""
        self.metadata.clear()
        self.session_file_path = None
        for tag in self._state_containers:
            container = self._state_containers[tag]
            sm = self.snapshot_methods(container, base_type=StateManager)
            if sm:
                sm.reset_state(container, self)
            else:
                container.clear()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if not name.startswith('_') and self.snapshot_methods(value, base_type=StateManager) is not None:
            self.add_state_manager(name, value)

    def __delattr__(self, name):
        if name in self._state_containers:
            self.remove_state_manager(name)
        object.__delattr__(self, name)

    def add_state_manager(self, tag, container):
        sm = self.snapshot_methods(container)
        if sm is None and not hasattr(container, 'clear'):
            raise ValueError('container "%s" of type "%s" does not have snapshot methods and does not have clear method' % (tag, str(type(container))))
        self._state_containers[tag] = container

    def get_state_manager(self, tag):
        return self._state_containers[tag]

    def remove_state_manager(self, tag):
        del self._state_containers[tag]

    def snapshot_methods(self, obj, instance=True, base_type=State):
        """Return an object having take_snapshot(), restore_snapshot(),
        and reset_state() methods for the given object.
        Can return None if no save/restore methods are available,
        for instance for primitive types.
        """
        cls = obj.__class__ if instance else obj
        from .serialize import PRIMITIVE_TYPES
        if cls in PRIMITIVE_TYPES:
            return None
        if issubclass(cls, base_type):
            return cls
        elif not hasattr(self, '_snapshot_methods'):
            from .graphics import View, MonoCamera, OrthographicCamera, Lighting, Material
            from .graphics import SceneClipPlane, CameraClipPlane, ClipPlane, Drawing
            from .graphics import gsession as g
            from .geometry import Place, Places, psession as p
            self._snapshot_methods = {
                View: g.ViewState,
                MonoCamera: g.CameraState,
                OrthographicCamera: g.CameraState,
                Lighting: g.LightingState,
                Material: g.MaterialState,
                ClipPlane: g.ClipPlaneState,
                SceneClipPlane: g.SceneClipPlaneState,
                CameraClipPlane: g.CameraClipPlaneState,
                Drawing: g.DrawingState,
                Place: p.PlaceState,
                Places: p.PlacesState,
            }

        methods = self._snapshot_methods.get(cls, None)
        return methods

    def save(self, stream, version, include_maps=False):
        """Serialize session to binary stream."""
        from . import serialize
        flags = State.SESSION
        if include_maps:
            flags |= State.INCLUDE_MAPS
        mgr = _SaveManager(self, flags)
        self.triggers.activate_trigger("begin save session", self)
        try:
            if version == 1:
                raise UserError("Version 1 session files are no longer supported")
            elif version == 2:
                raise UserError("Version 2 session files are no longer supported")
            else:
                if version != 3:
                    raise UserError("Only version 3 session files are supported")
                stream.write(b'# ChimeraX Session version 3\n')
                stream = serialize.msgpack_serialize_stream(stream)
                fserialize = serialize.msgpack_serialize
            metadata = standard_metadata(self.metadata)
            # TODO: put thumbnail in metadata
            # stash attribute info into metadata...
            attr_info = {}
            for tag, container in self._state_containers.items():
                attr_info[tag] = getattr(self, tag, None) == container
            metadata['attr_info'] = attr_info
            fserialize(stream, metadata)
            # guarantee that bundles are serialized first, so on restoration,
            # all of the related code will be loaded before the rest of the
            # session is restored
            mgr.discovery(self._state_containers)
            fserialize(stream, mgr.bundle_infos())
            # TODO: collect OrderDAGError exceptions from walk and analyze
            for name, data in mgr.walk():
                fserialize(stream, name)
                fserialize(stream, data)
            fserialize(stream, None)
        finally:
            mgr.cleanup()
            self.triggers.activate_trigger("end save session", self)

    def restore(self, stream, path=None, resize_window=None, restore_camera=True,
                metadata_only=False):
        """Deserialize session from binary stream."""
        from . import serialize
        if hasattr(stream, 'peek'):
            use_pickle = stream.peek(1)[0] != ord(b'#')
        elif hasattr(stream, 'buffer'):
            use_pickle = stream.buffer.peek(1)[0] != ord(b'#')
        elif stream.seekable():
            use_pickle = stream.read(1)[0] != ord(b'#')
            stream.seek(0)
        else:
            raise RuntimeError('Could not peek at first byte of session file.')
        if use_pickle:
            version = serialize.pickle_deserialize(stream)
            if version != 1:
                raise UserError('Not a ChimeraX session file')
            raise UserError("Session file format version 1 detected.  Convert using UCSF ChimeraX 0.8")
        else:
            line = stream.readline(256)   # limit line length to avoid DOS
            tokens = line.split()
            if line[-1] != ord(b'\n') or len(tokens) < 5 or tokens[0:4] != [b'#', b'ChimeraX', b'Session', b'version']:
                raise RuntimeError('Not a ChimeraX session file')
            version = int(tokens[4])
            if version == 2:
                raise UserError("Session file format version 2 detected.  DO NOT USE.  Recreate session from scratch, and then save.")
            elif version == 3:
                stream = serialize.msgpack_deserialize_stream(stream)
            else:
                raise UserError(
                    "Need newer version of ChimeraX to restore session")
            fdeserialize = serialize.msgpack_deserialize
        metadata = fdeserialize(stream)
        if metadata is None:
            raise UserError("Corrupt session file (missing metadata)")
        metadata['session_version'] = version
        if metadata_only:
            self.metadata.update(metadata)
            return

        mgr = _RestoreManager()
        bundle_infos = fdeserialize(stream)
        try:
            mgr.check_bundles(self, bundle_infos)
        except RestoreError as e:
            self.logger.warning(str(e))

        if resize_window is not None:
            self.restore_options['resize window'] = resize_window
        self.restore_options['restore camera'] = restore_camera

        self.triggers.activate_trigger("begin restore session", self)
        is_gui = hasattr(self, 'ui') and self.ui.is_gui
        from .tools import ToolInstance
        try:
            self.reset()
            self.session_file_path = path
            self.metadata.update(metadata)
            attr_info = self.metadata.pop('attr_info', {})
            while True:
                name = fdeserialize(stream)
                if name is None:
                    break
                data = fdeserialize(stream)
                data = mgr.resolve_references(data)
                if isinstance(name, str):
                    if attr_info.get(name, False):
                        setattr(self, name, data)
                    else:
                        self.add_state_manager(name, data)
                else:
                    # _UniqueName: find class
                    cls = name.class_of(self)
                    if cls is None:
                        continue
                    if name.uid[1] == 0:
                        obj = cls
                    else:
                        if not is_gui and issubclass(cls, ToolInstance):
                            mgr.add_reference(name, None)
                            continue
                        sm = self.snapshot_methods(cls, instance=False)
                        if sm is None:
                            obj = None
                            self.logger.warning('Unable to restore "%s" object' % cls.__name__)
                        else:
                            obj = sm.restore_snapshot(self, data)
                    mgr.add_reference(name, obj)
        except Exception:
            import traceback
            self.logger.bug("Unable to restore session, resetting.\n\n%s"
                            % traceback.format_exc())
            self.reset()
        finally:
            self.triggers.activate_trigger("end restore session", self)
            self.restore_options.clear()
            mgr.cleanup()


class InScriptFlag:

    def __init__(self):
        self._level = 0

    def __enter__(self):
        self._level += 1

    def __exit__(self, *_):
        self._level -= 1

    def __bool__(self):
        return self._level > 0


def standard_metadata(previous_metadata={}):
    """Fill in standard metadata for created files

    Parameters
    ----------
    previous_metadata : dict
        Optional dictionary of previous metadata.

    The standard metadata consists of:

    generator :
        Application that created file in HTML User Agent format
        (app name version (os))
    created :
        Date first created
    modified :
        Date last modified after being created
    creator :
        User name(s)
    dateCopyrighted :
        Copyright(s)

    creator and dateCopyrighted can be lists if there
    is previous metadata with different values.

    Dates are in ISO 8601 UTC time.  Also see
    <http://www.w3.org/TR/NOTE-datetime>.

    Metadata names are inspired by the HTML5 metadata,
    https://www.w3.org/TR/html5/document-metadata.html.
    """
    from .fetch import html_user_agent
    from chimerax import app_dirs
    from html import unescape
    import os
    import datetime
    from . import buildinfo

    metadata = {}
    if previous_metadata:
        metadata.update(previous_metadata)
    generator = unescape(html_user_agent(app_dirs))
    generator += ", http://www.rbvi.ucsf.edu/chimerax/"
    metadata['generator'] = generator
    now = datetime.datetime.utcnow()
    iso_date = now.isoformat() + 'Z'
    if 'created' in previous_metadata:
        metadata['modified'] = iso_date
    else:
        metadata['created'] = iso_date
    year = now.year
    # TODO: get user and copy right from settings
    # TODO: better way to get full user name
    user = os.environ.get('USERNAME', None)
    if user is None:
        user = os.environ.get('LOGNAME', None)
    if user is None:
        user = 'Unknown user'
    tmp = metadata.setdefault('creator', [])
    if not isinstance(tmp, list):
        tmp = [tmp]
    if user not in tmp:
        tmp = tmp + [user]
    if len(tmp) == 1:
        tmp = tmp[0]
    metadata['creator'] = tmp
    cpyrght = '\N{COPYRIGHT SIGN} %d %s' % (year, user)
    tmp = metadata.setdefault('dateCopyrighted', [])
    if not isinstance(tmp, list):
        tmp = [tmp]
    if cpyrght not in tmp:
        tmp = tmp + [cpyrght]
    if len(tmp) == 1:
        tmp = tmp[0]
    metadata['dateCopyrighted'] = tmp
    # build information
    # version is in 'generator'
    metadata['%s-commit' % app_dirs.appname] = buildinfo.commit
    metadata['%s-date' % app_dirs.appname] = buildinfo.date
    metadata['%s-branch' % app_dirs.appname] = buildinfo.branch
    return metadata


def save(session, path, version=3, uncompressed=False, include_maps=False):
    """command line version of saving a session"""
    my_open = None
    if hasattr(path, 'write'):
        # called via export, it's really a stream
        output = path
    else:
        from os.path import expanduser
        path = expanduser(path)         # Tilde expansion
        if not path.endswith(SESSION_SUFFIX):
            path += SESSION_SUFFIX

        if uncompressed:
            from .safesave import SaveBinaryFile
            my_open = SaveBinaryFile
        else:
            # Save compressed files
            from .safesave import SaveFile

            def my_open(path):
                import gzip
                f = SaveFile(path, open=lambda path: gzip.GzipFile(path, 'wb'))
                return f
        try:
            output = my_open(path)
        except IOError as e:
            raise UserError(e)

    session.session_file_path = path
    try:
        session.save(output, version=version, include_maps=include_maps)
    except Exception:
        if my_open is not None:
            output.close("exceptional")
        session.logger.report_exception()
        raise
    finally:
        if my_open is not None:
            output.close()

    # Associate thumbnail image with session file for display by operating system file browser.
    from . import utils
    if isinstance(path, str) and utils.can_set_file_icon():
        width = height = 512
        try:
            image = session.main_view.image(width, height)
        except RuntimeError:
            pass
        else:
            utils.set_file_icon(path, image)

    # Remember session in file history
    if isinstance(path, str):
        from .filehistory import remember_file
        remember_file(session, path, 'ses', 'all models', file_saved=True)


def sdump(session, session_file, output=None):
    """dump contents of session for debugging"""
    from . import serialize
    if not session_file.endswith(SESSION_SUFFIX):
        session_file += SESSION_SUFFIX
    if is_gzip_file(session_file):
        import gzip
        stream = gzip.open(session_file, 'rb')
    else:
        stream = _builtin_open(session_file, 'rb')
    if output is not None:
        # output = open_filename(output, 'w')
        output = _builtin_open(output, 'w')
    from pprint import pprint
    with stream:
        if hasattr(stream, 'peek'):
            use_pickle = stream.peek(1)[0] != ord(b'#')
        else:
            use_pickle = stream.buffer.peek(1)[0] != ord(b'#')
        if use_pickle:
            raise UserError("Use UCSF ChimeraX 0.8 for Session file format version 1.")
        else:
            tokens = stream.readline().split()
            version = int(tokens[4])
            if version == 2:
                raise UserError("Use UCSF ChimeraX 0.8 for Session file format version 2.")
            else:
                stream = serialize.msgpack_deserialize_stream(stream)
            fdeserialize = serialize.msgpack_deserialize
        print("==== session version:", file=output)
        pprint(version, stream=output)
        print("==== session metadata:", file=output)
        metadata = fdeserialize(stream)
        pprint(metadata, stream=output)
        print("==== bundle info:", file=output)
        bundle_infos = fdeserialize(stream)
        pprint(bundle_infos, stream=output)
        while True:
            name = fdeserialize(stream)
            if name is None:
                break
            data = fdeserialize(stream)
            print('==== name/uid:', name, file=output)
            pprint(data, stream=output)


def open(session, path, resize_window=None):
    if hasattr(path, 'read'):
        # Given a stream instead of a file name.
        fname = path.name
        path.close()
    else:
        fname = path

    if is_gzip_file(fname):
        import gzip
        stream = gzip.open(fname, 'rb')
    else:
        stream = _builtin_open(fname, 'rb')
    # TODO: active trigger to allow user to stop overwritting
    # current session
    session.session_file_path = path
    session.restore(stream, path=path, resize_window=resize_window)
    return [], "opened ChimeraX session"


def is_gzip_file(filename):
    f = _builtin_open(filename, 'rb')
    magic = f.read(2) + b'00'
    f.close()
    return (magic[0] == 0x1f and magic[1] == 0x8b)


def save_x3d(session, path, transparent_background=False):
    # Settle on using Interchange profile as that is the intent of
    # X3D exporting.
    from . import x3d
    x3d_scene = x3d.X3DScene()
    metadata = standard_metadata()

    # record needed X3D components
    x3d_scene.need(x3d.Components.EnvironmentalEffects, 1)  # Background
    # x3d_scene.need(x3d.Components.EnvironmentalEffects, 2)  # Fog
    camera = session.main_view.camera
    if camera.name == "orthographic":
        x3d_scene.need(x3d.Components.Navigation, 3)  # OrthoViewpoint
    else:
        x3d_scene.need(x3d.Components.Navigation, 1)  # Viewpoint, NavigationInfo
    # x3d_scene.need(x3d.Components.Rendering, 5)  # ClipPlane
    # x3d_scene.need(x3d.Components.Lighting, 1)  # DirectionalLight
    for m in session.models.list():
        m.x3d_needs(x3d_scene)

    with _builtin_open(path, 'w', encoding='utf-8') as stream:
        x3d_scene.write_header(
            stream, 0, metadata, profile_name='Interchange',
            # TODO? Skip units since it confuses X3D viewers and requires version 3.3
            units={'length': ('ångström', 1e-10)},
            # not using any of Chimera's extensions yet
            # namespaces={"chimera": "http://www.cgl.ucsf.edu/chimera/"}
        )
        cofr = session.main_view.center_of_rotation
        r, a = camera.position.rotation_axis_and_angle()
        t = camera.position.translation()
        if camera.name == "orthographic":
            hw = camera.field_width / 2
            f = (-hw, -hw, hw, hw)
            print('  <OrthoViewpoint centerOfRotation="%g %g %g" fieldOfView="%g %g %g %g" orientation="%g %g %g %g" position="%g %g %g"/>'
                  % (cofr[0], cofr[1], cofr[2], f[0], f[1], f[2], f[3], r[0], r[1], r[2], a, t[0], t[1], t[2]), file=stream)
        else:
            from math import tan, atan, radians
            h, w = session.main_view.window_size
            horiz_fov = radians(camera.field_of_view)
            vert_fov = 2 * atan(tan(horiz_fov / 2) * h / w)
            fov = min(horiz_fov, vert_fov)
            print('  <Viewpoint centerOfRotation="%g %g %g" fieldOfView="%g" orientation="%g %g %g %g" position="%g %g %g"/>'
                  % (cofr[0], cofr[1], cofr[2], fov, r[0], r[1], r[2], a, t[0], t[1], t[2]), file=stream)
        print('  <NavigationInfo type=\'"EXAMINE" "ANY"\' headlight=\'true\'/>', file=stream)
        c = session.main_view.background_color
        if transparent_background:
            t = 1
        else:
            t = 0
        print("  <Background skyColor='%g %g %g' transparency='%g'/>" % (c[0], c[1], c[2], t), file=stream)
        # TODO: write out lighting?
        from .geometry import Place
        p = Place()
        for m in session.models.list():
            m.write_x3d(stream, x3d_scene, 2, p)
        x3d_scene.write_footer(stream, 0)


def register_session_format(session):
    from .commands import CmdDesc, register, SaveFileNameArg, IntArg, BoolArg
    from .commands.cli import add_keyword_arguments
    from .commands.toolshed import register_command
    register_command(session.logger)
    from .commands import devel as devel_cmd, open as open_cmd, save as save_cmd
    devel_cmd.register_command(session.logger)
    open_cmd.register_command(session.logger)
    from . import io, toolshed
    io.register_format(
        "ChimeraX session", toolshed.SESSION, SESSION_SUFFIX, ("session",),
        mime="application/x-chimerax-session",
        reference="help:user/commands/save.html",
        open_func=open, export_func=save)
    add_keyword_arguments('open', {'resize_window': BoolArg})

    save_cmd.register_command(session.logger)
    desc = CmdDesc(
        required=[('filename', SaveFileNameArg)],
        keyword=[('version', IntArg), ('uncompressed', BoolArg)],
        hidden=['version', 'uncompressed'],
        synopsis='save session'
    )
    add_keyword_arguments('save', {'include_maps': BoolArg})

    def save_session(session, filename, **kw):
        kw['format'] = 'session'
        from .commands.save import save
        save(session, filename, **kw)
    register('save session', desc, save_session, logger=session.logger)
    add_keyword_arguments('save session', {'include_maps': BoolArg})

    import sys
    if sys.platform.startswith('linux'):
        from .commands.linux import register_command
        register_command(session.logger)


def register_x3d_format():
    from . import io, toolshed
    io.register_format(
        "X3D", toolshed.GENERIC3D, ".x3d", "x3d",
        mime="model/x3d+xml",
        reference="http://www.web3d.org/standards",
        export_func=save_x3d)


def common_startup(sess):
    """Initialize session with common data containers"""

    from .core_triggers import register_core_triggers
    register_core_triggers(sess.triggers)

    from .triggerset import set_exception_reporter
    set_exception_reporter(lambda preface, logger=sess.logger:
                           logger.report_exception(preface=preface))

    from .selection import Selection
    sess.selection = Selection(sess)

    try:
        from .core_settings import settings
        sess.main_view.background_color = settings.background_color.rgba
    except ImportError:
        pass

    from .updateloop import UpdateLoop
    sess.update_loop = UpdateLoop(sess)

    register(
        'debug sdump',
        CmdDesc(required=[('session_file', OpenFileNameArg)],
                optional=[('output', SaveFileNameArg)],
                synopsis="create human-readable session"),
        sdump,
        logger=sess.logger
    )
    register(
        'debug exception',
        CmdDesc(synopsis="generate exception to test exception handling"),
        _gen_exception,
        logger=sess.logger
    )

    _register_core_file_formats(sess)
    _register_core_database_fetch()


def _gen_exception(session):
    raise RuntimeError("Generated exception for testing purposes")


def register_session_save_options_gui(save_dialog):
    '''
    Session save gui options are registered in the ui module instead of when the
    format is registered because the ui does not exist when the format is registered.
    '''
    from chimerax.ui import SaveOptionsGUI

    class SessionSaveOptionsGUI(SaveOptionsGUI):
        @property
        def format_name(self):
            return "ChimeraX session"

        def wildcard(self):
            from chimerax.ui.open_save import export_file_filter
            from chimerax.core import toolshed
            return export_file_filter(toolshed.SESSION)

        def make_ui(self, parent):
            from PyQt5.QtWidgets import QFrame, QVBoxLayout, QCheckBox

            container = QFrame(parent)

            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            container.setLayout(layout)

            self._include_maps = im = QCheckBox('Include maps', container)
            layout.addWidget(im)

            return container

        def save(self, session, filename):
            import os.path
            ext = os.path.splitext(filename)[1]
            from chimerax.core import io
            fmt = io.format_from_name("ChimeraX session")
            exts = fmt.extensions
            if exts and ext not in exts:
                filename += exts[0]
            from chimerax.core.commands import run, quote_if_necessary
            cmd = "save session %s" % quote_if_necessary(filename)
            if self._include_maps.isChecked():
                cmd += ' includeMaps true'
            run(session, cmd)

    save_dialog.register(SessionSaveOptionsGUI())


def _register_core_file_formats(session):
    register_session_format(session)
    from . import scripting
    scripting.register()
    from . import image
    image.register_image_save(session)
    register_x3d_format()


def _register_core_database_fetch():
    from . import fetch
    fetch.register_web_fetch()
