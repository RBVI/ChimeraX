# vim: set expandtab ts=4 sw=4:

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


from . import ToolshedError, BundleAPI
from . import _debug


class BundleInfo:
    """Metadata about a bundle, whether installed or available.

    A :py:class:`BundleInfo` instance stores the properties about a bundle and
    can create a tool instance.

    Attributes
    ----------
    commands : list of :py:class:`CommandInfo`
        List of commands registered for this bundle.
    installed : boolean
        True if this bundle is installed locally; False otherwise.
    file_formats : list of :py:class:`DataInfo`
        List of data formats that this bundle knows about.
    session_versions : range
        Given as the minimum and maximum session versions
        that this bundle can read.
    session_write_version : integer
        The session version that bundle data is written in.
        Defaults to maximum of 'session_versions'.
    custom_init : boolean
        Whether bundle has custom initialization code
    name : readonly str
        The internal name of the bundle.
    synopsis : readonly str
        Short description of this bundle.
    version : readonly str
        Bundle version (which is actually the same as the distribution version,
        so all bundles from the same distribution share the same version).
    """

    def __init__(self, name, installed,
                 version=None,
                 api_package_name=None,
                 categories=(),
                 synopsis=None,
                 description="Unknown",
                 session_versions=range(1, 1 + 1),
                 custom_init=False,
                 packages=[], supercedes=[]):
        """Initialize instance.

        Parameters
        ----------
        name : str
            Name of Python distribution that provided this bundle.
        installed : boolean
            Whether this bundle is locally installed.
        categories : list of str
            List of categories in which this bundle belong.
        version : str
            Version of Python distribution that provided this bundle.
        api_package_name : str
            Name of package with bundle's API.  Package name must be a dotted Python name or blank if no ChimeraX deliverables in bundle.
        packages : list of tuples
            List of the Python packages implementing by this bundle.  The packages are given as a tuple e.g., ('chimerax', 'core') for chimerax.core.
        session_versions : range
            Range of session versions that this bundle can read.
        custom_init : boolean
            Whether bundle has custom initialization code
        """
        # Public attributes
        self.installed = installed
        self.session_versions = session_versions
        self.session_write_version = session_versions.stop - 1
        self.custom_init = custom_init
        self.categories = categories
        self.packages = packages
        self.tools = []
        self.commands = []
        self.formats = []
        self.selectors = []
        self.fetches = []
        self.description = description
        self.supercedes = supercedes

        # Private attributes
        self._name = name
        self._version = version
        self._api_package_name = api_package_name
        self._synopsis = synopsis

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def synopsis(self):
        return self._synopsis or "no synopsis available"

    def __repr__(self):
        # TODO:
        s = self._name
        if self.installed:
            s += " (installed)"
        else:
            s += " (available)"
        s += " [version: %s]" % self._version
        s += " [api package: %s]" % self._api_package_name
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
        """Return state data that can be used to recreate the instance.

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
            "api_package_name": self._api_package_name,
            "packages": self.packages,
            "description": self.description,
            "supercedes": self.supercedes,
        }
        more = {
            'tools': [ti.cache_data() for ti in self.tools],
            'commands': [ci.cache_data() for ci in self.commands],
            'formats': [fi.cache_data() for fi in self.formats],
            'selectors': [si.cache_data() for si in self.selectors],
            'fetches': self.fetches,
        }
        return args, kw, more

    @classmethod
    def from_cache_data(cls, data):
        args, kw, more = data
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
        if 'fetches' in more:
            bi.fetches = more['fetches']
        return bi

    def distribution(self):
        """Return distribution information.

        Returns
        -------
        2-tuple of (str, str).
            Distribution name and version.
        """
        return self._name, self._version

    def register(self, logger):
        """Register bundle commands, tools, data formats, selectors, etc.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Where to log error messages.
        """
        self._register_commands(logger)
        self._register_file_types(logger)
        self._register_selectors(logger)

    def deregister(self, logger):
        """Deregister bundle commands, tools, data formats, selectors, etc.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Where to log error messages.
        """
        self._deregister_selectors(logger)
        self._deregister_file_types(logger)
        self._deregister_commands(logger)

    def _register_commands(self, logger):
        """Register commands with cli."""
        from chimerax.core.commands import cli
        for ci in self.commands:
            def cb(s=self, n=ci.name, l=logger):
                s._register_cmd(n, l)
            _debug("delay_registration", ci.name)
            cli.delay_registration(ci.name, cb, logger=logger)

    def _register_cmd(self, command_name, logger):
        """Called when commands need to be really registered."""
        try:
            f = self._get_api(logger).register_command
        except AttributeError:
            raise ToolshedError(
                "no register_command function found for bundle \"%s\""
                % self.name)
        try:
            if f == BundleAPI.register_command:
                raise ToolshedError("bundle \"%s\"'s API forgot to override register_command()" % self.name)
            f(command_name, logger)
        except Exception as e:
            raise ToolshedError(
                "register_command() failed for command %s in bundle %s:\n%s" % (command_name, self.name, str(e)))

    def _deregister_commands(self, logger):
        """Deregister commands with cli."""
        from chimerax.core.commands import cli
        for ci in self.commands:
            _debug("deregister_command", ci.name)
            try:
                cli.deregister(ci.name)
            except RuntimeError:
                pass  # don't care if command was already missing

    def _register_file_types(self, logger):
        """Register file types."""
        from chimerax.core import io, fetch
        for fi in self.formats:
            _debug("register_file_type", fi.name)
            format = io.register_format(
                fi.name, fi.category, fi.suffixes, fi.nicknames,
                mime=fi.mime_types, reference=fi.documentation_url,
                dangerous=fi.dangerous, icon=fi.icon, encoding=fi.encoding,
                synopsis=fi.synopsis
            )
            if fi.has_open:
                def boot_open(bi_self=self, logger=logger):
                    try:
                        f = bi_self._get_api(logger).open_file
                    except AttributeError:
                        raise ToolshedError(
                            "no open_file function found for bundle \"%s\"" % bi_self.name)
                    if f == BundleAPI.open_file:
                        raise ToolshedError(
                            "bundle \"%s\"'s API forgot to override open_file()" % bi_self.name)
                    return f
                format._boot_open_func = boot_open

                if fi.open_kwds:
                    from ..commands import cli
                    try:
                        cli.add_keyword_arguments('open', _convert_keyword_types(
                            fi.open_kwds, self, logger))
                    except ValueError as e:
                        logger.warning(
                                "unable to register \"open\" keywords in bundle \"%s\": %s"
                                % (self.name, str(e)))
            if fi.has_save:
                def boot_save(bi_self=self, logger=logger):
                    try:
                        f = bi_self._get_api(logger).save_file
                    except AttributeError:
                        raise ToolshedError(
                            "no save_file function found for bundle \"%s\"" % bi_self.name)
                    if f == BundleAPI.save_file:
                        raise ToolshedError(
                            "bundle \"%s\"'s API forgot to override save_file()" % bi_self.name)
                    return f
                format._boot_export_func = boot_save

                if fi.save_kwds:
                    from ..commands import cli
                    try:
                        cli.add_keyword_arguments('save', _convert_keyword_types(
                            fi.save_kwds, self, logger))
                    except ValueError as e:
                        logger.warning(
                                "unable to register \"save\" keywords in bundle \"%s\": %s"
                                % (self.name, str(e)))
        for (database_name, format_name, prefixes, example_id, is_default) in self.fetches:
            if io.format_from_name(format_name) is None:
                print('warning: unknown format %r given for database %r' % (format_name, database_name))

            def fetch_cb(session, identifier, database_name=database_name, format_name=format_name, **kw):
                try:
                    f = self._get_api(logger).fetch_from_database
                except AttributeError:
                    raise ToolshedError(
                        "no fetch_from_database function found for bundle \"%s\""
                        % self.name)
                if f == BundleAPI.save_file:
                    raise ToolshedError("bundle \"%s\"'s API forgot to override fetch_from_database()" % self.name)
                # optimize by replacing fetch_from_database for (database, format)

                def fetch_shim(session, identifier, f=f, database_name=database_name, format_name=format_name, **kw):
                    return f(session, identifier, database_name=database_name, format_name=format_name, **kw)
                fetch.register_fetch(database_name, fetch_shim, format_name)
                return fetch_shim(session, identifier, **kw)
            fetch.register_fetch(
                database_name, fetch_cb, format_name, prefixes=prefixes,
                is_default_format=is_default, example_id=example_id)

    def _deregister_file_types(self, logger):
        """Deregister file types."""
        from chimerax.core import io, fetch
        # Deregister fetch first since it might use format info
        for (database_name, format_name, prefixes, example_id, is_default) in self.fetches:
            fetch.deregister_fetch(database_name, format_name, prefixes=prefixes)
        for fi in self.formats:
            io.deregister_format(fi.name)

    def _register_selectors(self, logger):
        from ..commands import register_selector
        for si in self.selectors:
            def selector_cb(session, models, results, _name=si.name):
                try:
                    reg = self._get_api(logger).register_selector
                except AttributeError:
                    raise ToolshedError(
                        "no register_selector function found for bundle \"%s\""
                        % self.name)
                if reg == BundleAPI.register_selector:
                    raise ToolshedError("bundle \"%s\"'s API forgot to override register_selector()" % self.name)
                reg(_name, session.logger)
                from ..commands import get_selector
                return get_selector(_name)(session, models, results)
            register_selector(si.name, selector_cb, logger)

    def _deregister_selectors(self, logger):
        from ..commands import deregister_selector
        for si in self.selectors:
            deregister_selector(si.name)

    def register_available_commands(self, logger):
        """Register available commands with cli."""
        from chimerax.core.commands import cli, CmdDesc
        for ci in self.commands:
            cd = CmdDesc(synopsis=ci.synopsis)
            def cb(session, s=self, n=ci.name, l=logger):
                s._available_cmd(n, l)
            cli.register_available(ci.name, cd, function=cb, logger=logger)

    def _available_cmd(self, name, logger):
        msg = ("\"%s\" is provided by the uninstalled bundle \"%s\""
               % (name, self.name))
        logger.status(msg, log=True)

    def initialize(self, session):
        """Initialize bundle by calling custom initialization code if needed."""
        if self.custom_init:
            try:
                f = self._get_api(session.logger).initialize
            except AttributeError:
                raise ToolshedError(
                    "no initialize function found for bundle \"%s\""
                    % self.name)
            if f == BundleAPI.initialize:
                session.logger.warning("bundle \"%s\"'s API forgot to override initialize()" % self.name)
                return
            try:
                f(session, self)
            except:
                import traceback, sys
                traceback.print_exc(file=sys.stdout)
                raise ToolshedError(
                    "initialization failed for bundle \"%s\"" % self.name)

    def finish(self, session):
        """Deinitialize bundle by calling custom finish code if needed."""
        if self.custom_init:
            try:
                f = self._get_api(session.logger).finish
            except AttributeError:
                raise ToolshedError("no finish function found for bundle \"%s\""
                                    % self.name)
            if f == BundleAPI.finish:
                session.logger.warning("bundle \"%s\"'s API forgot to override finish()" % self.name)
                return
            f(session, self)

    def unload(self, logger):
        """Unload bundle modules (as best as we can)."""
        import sys
        m = self.get_module()
        logger.info("unloading module %s" % m.__name__)
        name = m.__name__
        prefix = name + '.'
        remove_list = [k for k in sys.modules.keys()
                       if k == name or k.startswith(prefix)]
        for k in remove_list:
            del sys.modules[k]

    def get_class(self, class_name, logger):
        """Return bundle's class with given name."""
        try:
            f = self._get_api(logger).get_class
        except AttributeError:
            raise ToolshedError("no get_class function found for bundle \"%s\""
                                % self.name)
        return f(class_name)

    def get_module(self):
        """Return module that has bundle's code"""
        if not self._api_package_name:
            raise ToolshedError("Bundle %s has no module" % self.name)
        import importlib
        try:
            m = importlib.import_module(self._api_package_name)
        except Exception as e:
            raise ToolshedError("Error importing bundle %s's module: %s" % (self.name, str(e)))
        return m

    def _get_api(self, logger):
        """Return BundleAPI instance for this bundle."""
        m = self.get_module()
        try:
            bundle_api = getattr(m, 'bundle_api')
        except AttributeError:
            raise ToolshedError("missing bundle_api for bundle \"%s\"" % self.name)
        _debug("_get_api", self._api_package_name, m, bundle_api)
        return bundle_api

    def find_icon_path(self, icon_name):
        import os
        m = self.get_module()
        icon_dir = os.path.dirname(m.__file__)
        return os.path.join(icon_dir, icon_name)

    def start_tool(self, session, tool_name, *args, **kw):
        """Create and return a tool instance.

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
        ToolshedUninstalledError
            If the bundle is not installed.
        ToolshedError
            If the tool cannot be started.
        """
        if not self.installed:
            raise ToolshedError("bundle \"%s\" is not installed"
                                           % self.name)
        if not session.ui.is_gui:
            raise ToolshedError("tool \"%s\" is not supported without a GUI"
                                % tool_name)
        try:
            f = self._get_api(session.logger).start_tool
        except AttributeError:
            raise ToolshedError("no start_tool function found for bundle \"%s\""
                                % self.name)
        if f == BundleAPI.start_tool:
            raise ToolshedError("bundle \"%s\"'s API forgot to override start_tool()" % self.name)
        ti = f(session, tool_name, *args, **kw)
        if ti is not None:
            ti.display(True)  # in case the instance is a singleton not currently shown
        return ti

    def newer_than(self, bi):
        """Return whether this :py:class:`BundleInfo` instance is newer than given one

        Parameters
        ----------
        bi : :py:class:`BundleInfo` instance
            The instance to compare against

        Returns
        -------
        Boolean
            True if this instance is newer; False if 'bi' is newer.
        """
        from distlib.version import NormalizedVersion as Version
        return Version(self.version) > Version(bi.version)

    def dependents(self, logger):
        """Return set of bundles that directly depends on this one.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Where to log error messages.

        Returns
        -------
        set of :py:class:`~chimerax.core.toolshed.BundleInfo` instances
            Dependent bundles.
        """
        from . import Toolshed
        from distlib.database import DistributionPath
        keep = set()
        for d in DistributionPath().get_distributions():
            for req in d.run_requires:
                if req.split()[0] == self.name:
                    keep.add(d)
                    break
        ts = Toolshed.get_toolshed()
        deps = set()
        for d in keep:
            bi = ts.find_bundle(d.name, logger)
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
    """
    def __init__(self, name, synopsis=None):
        self.name = name
        if synopsis:
            self.synopsis = synopsis
        else:
            self.synopsis = "No synopsis given"

    def __repr__(self):
        s = self.name
        if self.synopsis:
            s += " [synopsis: %s]" % self.synopsis
        return s

    def cache_data(self):
        return (self.name, self.synopsis)

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
    bundle_api = None
    arg_cache = {}

    def get_arg(arg_name):
        nonlocal bundle_api
        a = arg_cache.get(arg_name, None)
        if a is not None:
            return a
        full_arg_name = arg_name + 'Arg'
        if hasattr(commands, full_arg_name):
            a = getattr(commands, full_arg_name)
        else:
            if bundle_api is None:
                bundle_api = bi._get_api(logger)
            if hasattr(bundle_api, full_arg_name):
                a = getattr(bundle_api, full_arg_name)
            else:
                print('unable to find %s argument type in bundle %s' % (arg_name, bi.name))
                return None
        arg_cache[arg_name] = a
        return a

    result = {}
    for kw in kwds:
        desc, arg_name = kwds[kw]
        arg_type = get_arg(arg_name)
        if arg_type is None:
            continue
        result[kw] = arg_type
    return result
