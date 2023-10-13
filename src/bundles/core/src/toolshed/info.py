# vim: set expandtab ts=4 sw=4:

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

# data_dir:1: WARNING: duplicate object description of chimerax.core.toolshed.BundleInfo.data_dir, other instance in core/toolshed, use :noindex: for one of them
# executable_dir:1: WARNING: duplicate object description of chimerax.core.toolshed.BundleInfo.executable_dir, other instance in core/toolshed, use :noindex: for one of them
# include_dir:1: WARNING: duplicate object description of chimerax.core.toolshed.BundleInfo.include_dir, other instance in core/toolshed, use :noindex: for one of them
# library_dir:1: WARNING: duplicate object description of chimerax.core.toolshed.BundleInfo.library_dir, other instance in core/toolshed, use :noindex: for one of them
from . import ToolshedError, BundleAPI
from . import _debug


class BundleInfo:
    """Supported API. Metadata about a bundle, whether installed or available.

    A :py:class:`BundleInfo` instance stores the properties about a bundle and
    can create a tool instance.

    Attributes:
        commands: list of :py:class:`CommandInfo`
            List of commands registered for this bundle.
        tools: list of :py:class:`ToolInfo`
            List of tools registered for this bundle.
        installed: boolean
            True if this bundle is installed locally; False otherwise.
        session_versions: range
            Given as the minimum and maximum session versions
            that this bundle can read.
        session_write_version: integer
            The session version that bundle data is written in.
            Defaults to maximum of 'session_versions'.
        custom_init: boolean
            Whether bundle has custom initialization code
    """

    def __init__(self, name, installed,
                 version=None,
                 api_package_name=None,
                 categories=(),
                 synopsis=None,
                 description="Unknown",
                 session_versions=range(1, 1 + 1),
                 custom_init=False,
                 data_dir=None,
                 include_dir=None,
                 library_dir=None,
                 executable_dir=None,
                 managers=None,
                 providers=None,
                 inits=None,
                 packages=None, supersedes=None):
        """Initialize instance.

        Parameters:
            name: str
                Name of Python distribution that provided this bundle.
            installed: boolean
                Whether this bundle is locally installed.
            categories: list of str
                List of categories in which this bundle belong.
            version: str
                Version of Python distribution that provided this bundle.
            api_package_name: str
                Name of package with bundle's API.  Package name must be a dotted Python name or blank if no ChimeraX deliverables in bundle.
            packages: list of tuples
                List of the Python packages implementing by this bundle.  The packages are given as a tuple e.g., ('chimerax', 'core') for chimerax.core.
            session_versions: range
                Range of session versions that this bundle can read.
            custom_init: boolean
                Whether bundle has custom initialization code.
            data_dir:
                Path (relative to bundle root) of directory of data files.
            include_dir:
                Path (relative to bundle root) of directory of compilation include files.
            library_dir:
                Path (relative to bundle root) of directory of link libraries.
            executable_dir:
                Path (relative to bundle root) of directory of executables.
            managers:
                Dictionary of manager names to init keywords
            providers:
                Dictionary of provider names to manager names + init keywords
            inits:
                Dictionary of initialization dependencies

        """

        # Public attributes
        self.installed = installed
        self.session_versions = session_versions
        self.session_write_version = session_versions.stop - 1
        self.custom_init = custom_init
        self.categories = categories
        if packages is None:
            packages = []
        self.packages = packages
        self.tools = []
        self.commands = []
        self.formats = []
        self.selectors = []
        self.fetches = []
        self.description = description
        if supersedes is None:
            supersedes = []
        self.supersedes = supersedes
        self.package_name = api_package_name
        self.installed_data_dir = data_dir
        self.installed_include_dir = include_dir
        self.installed_library_dir = library_dir
        self.installed_executable_dir = executable_dir
        self.managers = managers if managers else {}
        self.providers = providers if providers else {}
        self.inits = inits if inits else {}

        # Private attributes
        self._name = name
        self._version = version
        self._synopsis = synopsis

    @property
    def supercedes(self):
        # deprecated in ChimeraX 1.2
        return self.supersedes

    @property
    def name(self):
        """Supported API.

        Returns: Internal name of the bundle.
        """
        return self._name

    @property
    def short_name(self):
        """Supported API.

           Returns:
               A short name for the bundle.  Typically the same as 'name' with 'ChimeraX-' omitted.
        """
        boilerplate = "ChimeraX-"
        if self._name.startswith(boilerplate):
            return self._name[len(boilerplate):]
        return self._name

    @property
    def version(self):
        """Supported API.

           Returns:
               Bundle version (which is actually the same as the distribution version,
               so all bundles from the same distribution share the same version).
        """
        return self._version

    @property
    def synopsis(self):
        """Supported API.

           Returns:
               Bundle synopsis; a short description of this bundle.
        """
        return self._synopsis or "no synopsis available"

    def __repr__(self):
        # TODO:
        s = self._name
        if self.installed:
            s += " (installed)"
        else:
            s += " (available)"
        s += " [version: %s]" % self._version
        s += " [api package: %s]" % self.package_name
        if self.categories:
            s += " [category: %s]" % ', '.join(self.categories)
        for t in self.tools:
            s += " [tool: %r]" % t
        for c in self.commands:
            s += " [command: %r]" % c
        for d in self.formats:
            s += " [data format: %r]" % d
        return s

    def cache_data(self):
        """Supported API. Return state data that can be used to recreate the instance.

        Returns
        -------
        2-tuple of (list, dict)
            List and dictionary suitable for passing to :py:class:`BundleInfo`.
        """
        args = (self._name, self.installed)
        # TODO:
        kw = {
            "categories": self.categories,
            "synopsis": self._synopsis,
            "session_versions": (self.session_versions.start,
                                 self.session_versions.stop),
            "custom_init": self.custom_init,
            "version": self._version,
            "api_package_name": self.package_name,
            "packages": self.packages,
            "description": self.description,
            "supersedes": self.supersedes,
            "data_dir": self.installed_data_dir,
            "include_dir": self.installed_include_dir,
            "library_dir": self.installed_library_dir,
            "executable_dir": self.installed_executable_dir,
        }
        more = {
            'tools': [ti.cache_data() for ti in self.tools],
            'commands': [ci.cache_data() for ci in self.commands],
            'formats': [fi.cache_data() for fi in self.formats],
            'selectors': [si.cache_data() for si in self.selectors],
            'fetches': self.fetches,
            "managers": self.managers,
            "providers": self.providers,
            "inits": self.inits,
        }
        return args, kw, more

    @classmethod
    def from_cache_data(cls, data):
        """Supported API. Class method for reconstructing instance from cache data.
        Returns
        -------
        instance of BundleInfo
        """
        args, kw, more = data
        if 'supercedes' in kw:
            # handle spelling mistake from ChimeraX 1.1 and earlier
            kw['supersedes'] = kw['supercedes']
            del kw['supercedes']
        kw['session_versions'] = range(*kw['session_versions'])
        kw['packages'] = [tuple(x) for x in kw['packages']]
        tools = [ToolInfo.from_cache_data(d) for d in more['tools']]
        commands = [CommandInfo.from_cache_data(d) for d in more['commands']]
        formats = [FormatInfo.from_cache_data(d) for d in more['formats']]
        selectors = [SelectorInfo.from_cache_data(d) for d in more['selectors']]
        bi = BundleInfo(*args, **kw)
        bi.tools = tools
        bi.commands = commands
        bi.formats = formats
        bi.selectors = selectors
        if 'data_dir' in more:
            bi.installed_data_dir = more['data_dir']
        if 'include_dir' in more:
            bi.installed_include_dir = more['include_dir']
        if 'library_dir' in more:
            bi.installed_library_dir = more['library_dir']
        if 'executable_dir' in more:
            bi.installed_executable_dir = more['executable_dir']
        if 'fetches' in more:
            bi.fetches = more['fetches']
        if 'managers' in more:
            bi.managers = more['managers']
        if 'providers' in more:
            bi.providers = more['providers']
        if 'inits' in more:
            bi.inits = more['inits']
        return bi

    def distribution(self):
        """Supported API. Return distribution information.

        Returns
        -------
        2-tuple of (str, str).
            Distribution name and version.
        """
        return self._name, self._version

    def register(self, logger):
        """Supported API. Register bundle commands, tools, data formats, selectors, etc.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Where to log error messages.
        """
        _debug("register bundle", self._name, self._version)
        self._register_commands(logger)
        self._register_selectors(logger)

    def deregister(self, logger):
        """Supported API. Deregister bundle commands, tools, data formats, selectors, etc.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Where to log error messages.
        """
        self._deregister_selectors(logger)
        # self._deregister_file_types(logger)
        self._deregister_commands(logger)

    def _register_commands(self, logger):
        from chimerax.core.commands import cli
        for ci in self.commands:
            def cb(s=self, ci=ci, logger=logger):
                s._register_cmd(ci, logger)
            _debug("delay_registration", ci.name)
            try:
                cli.delay_registration(ci.name, cb, logger=logger)
            except Exception as e:
                logger.warning("Unable to register command %s: %s" % (ci.name, str(e)))

    def _register_cmd(self, ci, logger):
        try:
            api = self._get_api(logger)
            api._api_caller.register_command(api, self, ci, logger)
        except Exception as e:
            raise ToolshedError(
                "register_command() failed for command %s in bundle %s:\n%s" % (ci.name, self.name, str(e)))

    def _deregister_commands(self, logger):
        from chimerax.core.commands import cli
        for ci in self.commands:
            _debug("deregister_command", ci.name)
            try:
                cli.deregister(ci.name)
            except RuntimeError:
                pass  # don't care if command was already missing

    def _register_selectors(self, logger):
        from ..commands import register_selector
        for si in self.selectors:
            def selector_cb(session, models, results, si=si):
                try:
                    api = self._get_api(logger)
                    api._api_caller.register_selector(api, self, si, logger)
                except Exception as e:
                    raise ToolshedError(
                        "register_selector() failed for selector %s in bundle %s:\n%s" % (si.name, self.name, str(e)))
                from ..commands import get_selector
                sel = get_selector(si.name)
                if callable(sel):
                    return sel(session, models, results)
                else:
                    return sel
            try:
                register_selector(si.name, selector_cb, logger,
                                  desc=si.synopsis, atomic=si.atomic)
            except Exception as e:
                logger.warning("Unable to register selector %r: %s" % (si.name, str(e)))

    def _deregister_selectors(self, logger):
        from ..commands import deregister_selector
        for si in self.selectors:
            deregister_selector(si.name, logger)

    def initialize(self, session):
        """Supported API. Initialize bundle by calling custom initialization code if needed."""
        if self.custom_init:
            try:
                api = self._get_api(session.logger)
                api._api_caller.initialize(api, session, self)
            except Exception as e:
                import traceback
                session.logger.warning(traceback.format_exc())
                raise ToolshedError(
                    "initialize() failed in bundle %s:\n%s" % (self.name, str(e)))

    def init_manager(self, session, name, **kw):
        """Supported API. Initialize bundle manager if needed."""
        try:
            api = self._get_api(session.logger)
            return api._api_caller.init_manager(api, session, self, name, **kw)
        except Exception as e:
            import traceback
            session.logger.warning(traceback.format_exc())
            raise ToolshedError(
                "init_manager() failed in bundle %s:\n%s" % (self.name, str(e)))

    def run_provider(self, session, name, mgr, **kw):
        """Supported API. Called by manager to invoke bundle provider."""
        api = self._get_api(session.logger)
        return api._api_caller.run_provider(api, session, name, mgr, **kw)

    def finish(self, session):
        """Supported API. Deinitialize bundle by calling custom finish code if needed.

        This method is only called when a bundle is explicitly unloaded.
        In particular, it is *not* called when ChimeraX exits normally."""
        if self.get_module(force_import=False):
            try:
                api = self._get_api(session.logger)
                api._api_caller.finish(api, session, self)
            except Exception as e:
                import sys
                import traceback
                traceback.print_exc(file=sys.stdout)
                raise ToolshedError(
                    "finish() failed in bundle %s:\n%s" % (self.name, str(e)))

    def include_dir(self):
        """Supported API.

           Returns: Path (relative to bundle root) of the bundle's directory of compilation include files.
        """
        return self._bundle_path(self.installed_include_dir)

    def library_dir(self):
        """Supported API.

           Returns: Path (relative to bundle root) of the bundle's directory of link libraries.
        """
        return self._bundle_path(self.installed_library_dir)

    def executable_dir(self):
        """Supported API.
           Returns: Path (relative to bundle root) of the bundle's directory of executables.
        """
        return self._bundle_path(self.installed_executable_dir)

    def data_dir(self):
        """Supported API.

           Returns:
               Path (relative to the bundle root) of the bundle's data directory.
        """
        return self._bundle_path(self.installed_data_dir)

    def _bundle_path(self, filename):
        # Find path for a filename in bundle without
        # actually importing the Python package
        if not filename:
            return None
        from importlib.util import find_spec
        try:
            s = find_spec(self.package_name)
            if s is None:
                return None
        except ModuleNotFoundError:
            return None
        import os.path
        if s.submodule_search_locations:
            for d in s.submodule_search_locations:
                p = os.path.join(d, filename)
                if os.path.exists(p):
                    return p
        return None

    def unload(self, logger):
        """Supported API. Unload bundle modules (as best as we can)."""
        import sys
        logger.info("unloading module %s" % self.package_name)
        name = self.package_name
        prefix = name + '.'
        remove_list = [k for k in sys.modules.keys()
                       if k == name or k.startswith(prefix)]
        for k in remove_list:
            del sys.modules[k]

    def imported(self):
        import sys
        # modules = ('.'.join(p) for p in self.packages)
        modules = (p[0] for p in self.packages if len(p) == 1)
        return any(m in sys.modules for m in modules)

    def get_class(self, class_name, logger):
        """Supported API. Return bundle's class with given name."""
        try:
            f = self._get_api(logger).get_class
        except AttributeError:
            raise ToolshedError("no get_class function found for bundle \"%s\""
                                % self.name)
        return f(class_name)

    def get_module(self, force_import=True):
        """Supported API. Return module that has bundle's code"""
        if not self.package_name:
            raise ToolshedError("Bundle %s has no module" % self.name)
        import sys
        try:
            return sys.modules[self.package_name]
        except KeyError:
            if not force_import:
                return None
        import importlib
        try:
            m = importlib.import_module(self.package_name)
        except Exception as e:
            raise ToolshedError("Error importing bundle %s's module: %s" % (self.name, str(e)))
        return m

    def update_library_path(self):
        # _debug("update_library_path", self.name, self.package_name)
        libdir = self.library_dir()
        if not libdir:
            # _debug("  update_library_path: no libdir")
            return
        import sys
        if sys.platform.startswith('win'):
            import os
            os.add_dll_directory(libdir)
            # _debug("  update_library_path: windows", paths)

    def _get_api(self, logger=None):
        """Return BundleAPI instance for this bundle."""
        m = self.get_module()
        try:
            bundle_api = getattr(m, 'bundle_api')
        except AttributeError:
            # XXX: Should we return a default BundleAPI instance?
            raise ToolshedError("missing bundle_api for bundle \"%s\"" % self.name)
        # _debug("_get_api", self.package_name, m, bundle_api)
        return bundle_api

    def get_path(self, subpath):
        p = self._bundle_path(subpath)
        if p is None:
            return None
        import os.path
        return p if os.path.exists(p) else None

    def start_tool(self, session, tool_name, *args, **kw):
        """Supported API. Create and return a tool instance.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance
            The session in which the tool will run.
        args : any
            Positional arguments to pass to tool instance initializer.
        kw : any
            Keyword arguments to pass to tool instance initializer.

        Returns
        -------
        :py:class:`~chimerax.core.tools.ToolInstance` instance
            The registered running tool instance.

        Raises
        ------
        ToolshedError
            If the tool is not installed or cannot be started.
        """
        if not self.installed:
            raise ToolshedError("bundle \"%s\" is not installed" % self.name)
        if not session.ui.is_gui:
            raise ToolshedError("tool \"%s\" is not supported without a GUI"
                                % tool_name)
        tool_info = None
        for tinfo in self.tools:
            if tool_name == tinfo.name:
                tool_info = tinfo
                break
        from ..errors import CancelOperation, NotABug
        try:
            api = self._get_api(session.logger)
            ti = api._api_caller.start_tool(api, session, self, tool_info)
            if ti is not None:
                ti.display(True)  # in case the instance is a singleton not currently shown
            return ti
        except (CancelOperation, NotABug):
            raise
        except Exception as e:
            raise ToolshedError(
                "start_tool() failed for tool %s in bundle %s:\n%s" % (tool_name, self.name, str(e)))

    def newer_than(self, bi):
        """Supported API. Return whether this :py:class:`BundleInfo` instance is newer than given one

        Parameters
        ----------
        bi : :py:class:`BundleInfo` instance
            The instance to compare against

        Returns
        -------
        Boolean
            True if this instance is newer; False if 'bi' is newer.
        """
        from pkg_resources import parse_version
        return parse_version(self.version) > parse_version(bi.version)

    def dependents(self, logger):
        """Supported API. Return set of bundles that directly depends on this one.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Where to log error messages.

        Returns
        -------
        set of :py:class:`~chimerax.core.toolshed.BundleInfo` instances
            Dependent bundles.
        """
        from . import get_toolshed
        from pkg_resources import working_set
        keep = set()
        for d in working_set:
            for req in d.requires():
                if req.name == self.name:
                    keep.add(d)
                    break
        ts = get_toolshed()
        deps = set()
        for d in keep:
            bi = ts.find_bundle(d.project_name, logger)
            if bi:
                deps.add(bi)
        return deps


class ToolInfo:
    """Metadata about a tool

    Attributes
    ----------
    name : str
       Tool name.
    categories : list of str
        Menu categories that tool should be in.
    synopsis : str
        One line description.
    """
    def __init__(self, name, categories, synopsis=None):
        self.name = name
        self.categories = categories
        if synopsis:
            self.synopsis = synopsis
        else:
            self.synopsis = "No synopsis given"

    def __repr__(self):
        s = self.name
        if self.categories:
            s += " [categories: %s]" % ', '.join(self.categories)
        if self.synopsis:
            s += " [synopsis: %s]" % self.synopsis
        return s

    def cache_data(self):
        return (self.name, self.categories, self.synopsis)

    @classmethod
    def from_cache_data(cls, data):
        return cls(*data)


class CommandInfo(ToolInfo):
    """Metadata about a command"""
    pass


class SelectorInfo(ToolInfo):
    """Metadata about a selector

    Attributes
    ----------
    name : str
       Tool name.
    synopsis : str
        One line description.
    atomic : boolean
        Whether selector applies to atoms and bonds.
    """
    def __init__(self, name, synopsis=None, atomic=True):
        self.name = name
        if synopsis:
            self.synopsis = synopsis
        else:
            self.synopsis = "No synopsis given"
        self.atomic = atomic

    def __repr__(self):
        s = self.name
        if self.synopsis:
            s += " [atomic: %s, synopsis: %s]" % (self.atomic, self.synopsis)
        return s

    def cache_data(self):
        return (self.name, self.synopsis, self.atomic)

    @classmethod
    def from_cache_data(cls, data):
        return cls(*data)


class FormatInfo:
    """Metadata about a data format

    Attributes
    ----------
    name : str
        Data format name
    categetory : str
        Data category -- see io.py for known categories
    open_extensions : list of str
        Recognized extensions for opening file
    save_extensions : list of str
        Recognized extensions for saving file
    mime_types : list of str
        List of media types, *e.g.*, chimera/x-pdb
    documentation_url : str
        URL to documentation about data format
    dangerous : bool
        True if data format is insecure, defaults to true if a script
    encoding : None or str
        text encoding if a text format
    icon : str
        filename in bundle of icon for data format
    """

    def __init__(self, name, category, *, nicknames=None, suffixes=None,
                 mime_types=None, url=None, synopsis=None,
                 dangerous=None, icon=None, encoding=None,
                 open_kwds=None, save_kwds=None,
                 has_open=False, has_save=False):
        self.name = name
        self.nicknames = nicknames
        self.category = category
        self.suffixes = suffixes
        self.mime_types = mime_types
        self.documentation_url = url
        self.dangerous = dangerous
        self.encoding = encoding
        self.icon = icon
        self.synopsis = synopsis
        self.has_open = has_open
        self.open_kwds = open_kwds
        self.has_save = has_save
        self.save_kwds = save_kwds

    def __repr__(self):
        s = self.name
        s += " [category: %s]" % self.category
        if self.suffixes:
            s += " [suffixes: %s]" % ', '.join(self.suffixes)
        if self.mime_types:
            s += " [mime_types: %s]" % ', '.join(self.mime_types)
        if self.documentation_url:
            s += " [url: %s]" % self.documentation_url
        if self.dangerous:
            s += " (dangerous)"
        if self.encoding:
            s += " [encoding: %s]" % self.encoding
        if self.synopsis:
            s += " [synopsis: %s]" % self.synopsis
        return s

    def cache_data(self):
        return {
            'name': self.name,
            'nicknames': self.nicknames,
            'category': self.category,
            'suffixes': self.suffixes,
            'mime_types': self.mime_types,
            'url': self.documentation_url,
            'dangerous': self.dangerous,
            'encoding': self.encoding,
            'icon': self.icon,
            'synopsis': self.synopsis,
            'has_open': self.has_open,
            'open_kwds': self.open_kwds,
            'has_save': self.has_save,
            'save_kwds': self.save_kwds,
        }

    @classmethod
    def from_cache_data(cls, data):
        return cls(**data)


#
# Class-independent utility functions
#


def _convert_keyword_types(kwds, bi, logger):
    from .. import commands
    arg_cache = {}
    result = {}
    for kw in kwds:
        desc, arg_name = kwds[kw]
        try:
            arg_type = arg_cache[arg_name]
        except KeyError:
            try:
                arg_type = getattr(commands, arg_name + "Arg")
            except AttributeError:
                arg_type = AnnotationProxy(bi, arg_name, logger)
                arg_cache[arg_name] = arg_type
        result[kw] = arg_type
    return result


class AnnotationProxy:

    def __init__(self, bi, arg_name, logger):
        self._bundle_info = bi
        self._arg_name = arg_name
        self._logger = logger
        self._proxy = None

    def __getattr__(self, attr):
        if self._proxy is None:
            bundle_api = self._bundle_info._get_api(self._logger)
            try:
                self._proxy = getattr(bundle_api, self._arg_name + "Arg")
            except AttributeError:
                print('unable to find %s argument type in bundle %s' %
                      (self._arg_name, self._bundle_info.name))
                raise
        return getattr(self._proxy, attr)
