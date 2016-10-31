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

"""
The Toolshed provides an interface for finding installed
bundles as well as bundles available for
installation from a remote server.
The Toolshed can handle updating, installing and uninstalling
bundles while taking care of inter-bundle dependencies.

The Toolshed interface uses :py:mod:`distlib` heavily.
For example, `Distribution` instances from :py:mod:`distlib`
are tracked for both available and installed bundles;
the :py:class:`distlib.locators.Locator` class is used for finding
an installed :py:class:`distlib.database.Distribution`.

Each Python distribution, a ChimeraX Bundle,
(ChimeraX uses :py:class:`distlib.wheel.Wheel`)
may contain multiple tools, commands, data formats, and specifiers,
with metadata entries for each deliverable.

In addition to the normal Python package metadta,
The 'ChimeraX-*' classifier entries give additional information.
Depending on the values of 'ChimeraX-*' metadata fields,
modules need to override methods of the :py:class:`BundleAPI` class.
Each bundle needs a 'ChimeraX-Bundle' entry
that consists of the following fields separated by double colons (``::``).

1. ``ChimeraX-Bundle`` : str constant
    Field identifying entry as bundle metadata.
2. ``categories`` : str
    Comma-separated list of categories in which the bundle belongs.
3. ``session_versions`` : two comma-separated integers
    Minimum and maximum session version that the bundle can read.
4. ``custom_init`` : str
    Whether bundle has initialization code that must be called when
    ChimeraX starts.  Either 'true' or 'false'.  If 'true', the bundle
    must override the BundleAPI's 'initialize' and 'finish' functions.

Bundles that provide tools need:

1. ``ChimeraX-Tool`` : str constant
    Field identifying entry as tool metadata.
2. ``tool_name`` : str
    The globally unique name of the tool (also shown on title bar).
3. ``categories`` : str
    Comma-separated list of categories in which the tool belongs.
    Should be a subset of the bundle's categories.
4. ``synopsis`` : str
    A short description of the tool.  It is here for uninstalled tools,
    so that users can get more than just a name for deciding whether
    they want the tool or not.

Tools are created via the bundle's 'start_tool' function.
Bundles may provide more than one tool.

Bundles that provide commands need:

1. ``ChimeraX-Command`` : str constant
    Field identifying entry as command metadata.
2. ``command name`` : str
    The (sub)command name.  Subcommand names have spaces in them.
3. ``categories`` : str
    Comma-separated list of categories in which the command belongs.
    Should be a subset of the bundle's categories.
4. ``synopsis`` : str
    A short description of the command.  It is here for uninstalled commands,
    so that users can get more than just a name for deciding whether
    they want the command or not.

Commands are lazily registered,
so the argument specification isn't needed until the command is first used.
Bundles may provide more than one command.

Bundles that provide selectors need:

1. ``ChimeraX-Selector`` : str constant
    Field identifying entry as command metadata.
2. ``selector name`` : str
    The selector's name.
3. ``synopsis`` : str
    A short description of the selector.  It is here for uninstalled selectors,
    so that users can get more than just a name for deciding whether
    they want the selector or not.

Commands are lazily registered,
so the argument specification isn't needed until the command is first used.
Bundles may provide more than one command.

Bundles that provide data formats need:

1. ``ChimeraX-DataFormat`` : str constant
    Field identifying entry as data format metadata.
2. ``data_name`` : str
    The name of the data format.
3. ``alternate_names`` : str
    An optional comma-separated list of alternative names.
    Often a short name is provided.
4. ``category`` : str
    The toolshed category.
5. ``suffixes`` : str
    An optional comma-separated list of strings with leading periods,
    e.g., '.pdb'.
6. ``mime_types`` : str
    An optinal comma-separated list of strings, e.g., 'chemical/x-pdb'.
7. ``url`` : str
    A string that has a URL that points to the data format's docmentation.
8. ``dangerous`` : str
    An optional boolean and should be 'true' if the data
    format is insecure -- defaults to true if a script.
9. ``icon`` : str
    An optional string containing the filename of the icon --
    it defaults to the default icon for the category.
    The file should be ?TODO? -- metadata dir?  package dir?
10. ``synopsis
    A short description of the data format.  It is here
    because it needs to be part of the metadata available for
    uninstalled data format, so that users can get more than just a
    name for deciding whether they want the data format or not.

Bundles may provide more than one data format.
The data format metadata includes everything needed for the Mac OS X
application property list.

Data formats that can be fetched:

# ChimeraX-Fetch :: database_name :: format_name :: prefixes :: example_id :: is_default

Data formats that can be opened:

# ChimeraX-Open :: format_name :: tag :: is_default

Data formats that can be saved:

# ChimeraX-Save :: format_name :: tag :: is_default

Attributes
----------
TOOLSHED_BUNDLE_INFO_ADDED : str
    Name of trigger fired when new bundle metadata is registered.
    The trigger data is a :py:class:`BundleInfo` instance.
TOOLSHED_BUNDLE_INSTALLED : str
    Name of trigger fired when a new bundle is installed.
    The trigger data is a :py:class:`BundleInfo` instance.
TOOLSHED_BUNDLE_UNINSTALLED : str
    Name of trigger fired when an installed bundle is removed.
    The trigger data is a :py:class:`BundleInfo` instance.
TOOLSHED_BUNDLE_INFO_RELOADED : str
    Name of trigger fired when bundle metadata is reloaded.
    The trigger data is a :py:class:`BundleInfo` instance.

Notes
-----
The term 'installed' refers to bundles whose corresponding Python
module or package is installed on the local machine.  The term
'available' refers to bundles that are listed on a remote server
but have not yet been installed on the local machine.

"""
from ..orderedset import OrderedSet

# Toolshed trigger names
TOOLSHED_BUNDLE_INFO_ADDED = "bundle info added"
TOOLSHED_BUNDLE_INSTALLED = "bundle installed"
TOOLSHED_BUNDLE_UNINSTALLED = "bundle uninstalled"
TOOLSHED_BUNDLE_INFO_RELOADED = "bundle info reloaded"

# Known bundle catagories
DYNAMICS = "Molecular trajectory"
GENERIC3D = "Generic 3D objects"
SCRIPT = "Command script"
SEQUENCE = "Sequence alignment"
SESSION = "Session data"
STRUCTURE = "Molecular structure"
SURFACE = "Molecular surface"
VOLUME = "Volume data"
Categories = [
    DYNAMICS,
    GENERIC3D,
    SCRIPT,
    SEQUENCE,
    SESSION,
    STRUCTURE,
    SURFACE,
    VOLUME,
]


def _hack_distlib(f):
    from functools import wraps

    @wraps(f)
    def hacked_f(*args, **kw):
        # distlib and wheel packages disagree on the name for
        # the metadata file in wheels.  (wheel uses PEP345 while
        # distlib uses PEP427.)  distlib is backwards compatible,
        # so we hack the file name when we get distributions.
        from distlib import metadata, database, wheel
        save = metadata.METADATA_FILENAME
        metadata.METADATA_FILENAME = "metadata.json"
        database.METADATA_FILENAME = metadata.METADATA_FILENAME
        wheel.METADATA_FILENAME = metadata.METADATA_FILENAME
        _debug("changing METADATA_FILENAME", metadata.METADATA_FILENAME)
        try:
            v = f(*args, **kw)
        finally:
            # Restore hacked name
            metadata.METADATA_FILENAME = save
            database.METADATA_FILENAME = save
            wheel.METADATA_FILENAME = save
            _debug("changing back METADATA_FILENAME", metadata.METADATA_FILENAME)
        return v
    return hacked_f


def _debug(*args, **kw):
    return


def _get_installed_packages(dist):
    """Return set of tuples representing the packages in the distribution.

    For example, 'foo.bar' from foo/bar/__init__.py becomes ('foo', 'bar')
    """
    packages = []
    for path, hash, size in dist.list_installed_files():
        if not path.endswith('/__init__.py'):
            continue
        parts = path.split('/')
        packages.append(tuple(parts[:-1]))
    return packages


# Package constants


# Default URL of remote toolshed
_RemoteURL = "http://localhost:8080"
# _RemoteURL = "https://chi2ti-macosx-daily.rbvi.ucsf.edu"
# Default name for toolshed cache and data directories
_Toolshed = "toolshed"
# Defaults names for installed ChimeraX bundles
_ChimeraNamespace = "chimerax"
_ChimeraCore = "ChimeraX-Core"  # ChimeraX Core's bundle name


# Exceptions raised by Toolshed class


class ToolshedError(Exception):
    """Generic Toolshed error."""


class ToolshedUninstalledError(ToolshedError):
    """Uninstalled-bundle error.

    This exception derives from :py:class:`ToolshedError` and is usually
    raised when trying to uninstall a bundle that has not been installed."""


class ToolshedInstalledError(ToolshedError):
    """Bundle-already-installed error.

    This exception derives from :py:class:`ToolshedError` and is usually
    raised when trying to install a bundle that is already installed."""


class ToolshedUnavailableError(ToolshedError):
    """Bundle-not-found error.

    This exception derives from ToolshedError and is usually
    raised when no Python distribution can be found for a bundle."""


# Toolshed and BundleInfo are session-independent


class Toolshed:
    """Toolshed keeps track of the list of bundle metadata, aka :py:class:`BundleInfo`.

    Tool metadata may be for "installed" bundles, where their code
    is already downloaded from the remote server and installed
    locally, or "available" bundles, where their code is not locally
    installed.

    Attributes
    ----------
    triggers : :py:class:`~chimerax.core.triggerset.TriggerSet` instance
        Where to register handlers for toolshed triggers
    """

    def __init__(self, logger,
                 rebuild_cache=False, check_remote=False, remote_url=None):
        """Initialize Toolshed instance.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            A logging object where warning and error messages are sent.
        rebuild_cache : boolean
            True to ignore local cache of installed bundle information and
            rebuild it by scanning Python directories; False otherwise.
        check_remote : boolean
            True to check remote server for updated information;
            False to ignore remote server;
            None to use setting from user preferences.
        remote_url : str
            URL of the remote toolshed server.
            If set to None, a default URL is used.
        """
        # Initialize with defaults
        _debug("__init__", rebuild_cache, check_remote, remote_url)
        if remote_url is None:
            self.remote_url = _RemoteURL
        else:
            self.remote_url = remote_url
        self._repo_locator = None
        self._inst_locator = None
        self._installed_bundle_info = []
        self._available_bundle_info = []
        self._all_installed_distributions = None
        self._installed_packages = {}   # cache mapping packages to bundles

        # Compute base directories
        import os.path
        from chimerax import app_dirs
        self._cache_dir = os.path.join(app_dirs.user_cache_dir, _Toolshed)
        _debug("cache dir: %s" % self._cache_dir)
        self._data_dir = os.path.join(app_dirs.user_data_dir, _Toolshed)
        _debug("data dir: %s" % self._data_dir)

        # Add directories to sys.path
        import os.path
        self._site_dir = os.path.join(self._data_dir, "site-packages")
        _debug("site dir: %s" % self._site_dir)
        import os
        os.makedirs(self._site_dir, exist_ok=True)
        import site
        site.addsitedir(self._site_dir)

        # Create triggers
        from .. import triggerset
        self.triggers = triggerset.TriggerSet()
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INFO_ADDED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INSTALLED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_UNINSTALLED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INFO_RELOADED)

        # Reload the bundle info list
        _debug("loading bundles")
        self.reload(logger, check_remote=check_remote,
                    rebuild_cache=rebuild_cache)
        _debug("finished loading bundles")

    def check_remote(self, logger):
        """Check remote shed for updated bundle info.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Logging object where warning and error messages are sent.

        Returns
        -------
        list of :py:class:`BundleInfo` instances
            List of bundle metadata from remote server.
        """

        _debug("check_remote")
        if self._repo_locator is None:
            from .chimera_locator import ChimeraLocator
            self._repo_locator = ChimeraLocator(self.remote_url)
        distributions = self._repo_locator.get_distributions()
        bi_list = []
        for d in distributions:
            bi = self._make_bundle_info(d, False, logger)
            if bi is not None:
                bi_list.append(bi)
            _debug("added remote distribution:", d)
        return bi_list

    def reload(self, logger, *, session=None, rebuild_cache=False, check_remote=False):
        """Discard and reread bundle info.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            A logging object where warning and error messages are sent.
        rebuild_cache : boolean
            True to ignore local cache of installed bundle information and
            rebuild it by scanning Python directories; False otherwise.
        check_remote : boolean
            True to check remote server for updated information;
            False to ignore remote server;
            None to use setting from user preferences.
        """

        _debug("reload", rebuild_cache, check_remote)
        for bi in reversed(self._installed_bundle_info):
            for p in bi.packages:
                try:
                    del self._installed_packages[p]
                except KeyError:
                    pass
            if session is not None:
                bi.finish(session)
            bi.deregister()
        self._installed_bundle_info = []
        inst_bi_list = self._load_bundle_infos(logger, rebuild_cache=rebuild_cache)
        for bi in inst_bi_list:
            self.add_bundle_info(bi)
            bi.register()
            if session is not None:
                bi.initialize(session)
        if check_remote:
            self._available_bundle_info = []
            self._repo_locator = None
            remote_bi_list = self.check_remote(logger)
            for bi in remote_bi_list:
                self.add_bundle_info(bi)
                # XXX: do we want to register commands so that we can
                # ask user whether to install bundle when invoked?
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_INFO_RELOADED, bi)

    def bundle_info(self, installed=True, available=False):
        """Return list of bundle info.

        Parameters
        ----------
        installed : boolean
            True to include installed bundle metadata in return value;
            False otherwise
        available : boolean
            True to include available bundle metadata in return value;
            False otherwise

        Returns
        -------
        list of :py:class:`BundleInfo` instances
            Combined list of all selected types of bundle metadata.  """

        _debug("bundle_info", installed, available)
        if installed and available:
            return self._installed_bundle_info + self._available_bundle_info
        elif installed:
            return self._installed_bundle_info
        elif available:
            return self._available_bundle_info
        else:
            return []

    def add_bundle_info(self, bi):
        """Add metadata for a bundle.

        Parameters
        ----------
        bi : :py:class:`BundleInfo` instance
            Must be a constructed instance, *i.e.*, not an existing instance
            returned by :py:func:`bundle_info`.

        Notes
        -----
        A :py:const:`TOOLSHED_BUNDLE_INFO_ADDED` trigger is fired after the addition.
        """
        _debug("add_bundle_info", bi)
        if bi.installed:
            container = self._installed_bundle_info
            for p in bi.packages:
                if p in self._installed_packages:
                    bi2 = self._installed_packages[p]
                    print('warning: both %s and %s supply package %s' % (
                          bi.name, bi2.name, '.'.join(p)))
                self._installed_packages[p] = bi
        else:
            container = self._available_bundle_info
        container.append(bi)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_INFO_ADDED, bi)

    def install_bundle(self, bi, logger, *, system=False, session=None):
        """Install the bundle by retrieving it from the remote shed.

        Parameters
        ----------
        bi : :py:class:`BundleInfo` instance
            Should be from the available bundle list.
        system : boolean
            False to install bundle only for the current user (default);
            True to install for everyone.
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Logging object where warning and error messages are sent.

        Raises
        ------
        ToolshedInstalledError
            Raised if the bundle is already installed.

        Notes
        -----
        A :py:const:`TOOLSHED_BUNDLE_INSTALLED` trigger is fired after installation.
        """
        _debug("install_bundle", bi)
        if bi.installed:
            raise ToolshedInstalledError("bundle \"%s\" already installed"
                                         % bi.name)
        # Make sure that our install location is on chimerax module.__path__
        # so that newly installed modules may be found
        import importlib
        import os.path
        cx_dir = os.path.join(self._site_dir, _ChimeraNamespace)
        m = importlib.import_module(_ChimeraNamespace)
        if cx_dir not in m.__path__:
            m.__path__.append(cx_dir)
        # Install bundle and update cache
        self._install_bundle(bi, system, logger, session)
        self._write_cache(self._installed_bundle_info, logger)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_INSTALLED, bi)

    def uninstall_bundle(self, bi, logger, *, session=None):
        """Uninstall bundle by removing the corresponding Python distribution.

        Parameters
        ----------
        bi : :py:class:`BundleInfo` instance
            Should be from the installed bundle list.
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Logging object where warning and error messages are sent.

        Raises
        ------
        ToolshedInstalledError
            Raised if the bundle is not installed.

        Notes
        -----
        A :py:const:`TOOLSHED_BUNDLE_UNINSTALLED` trigger is fired after package removal.
        """
        _debug("uninstall_bundle", bi)
        self._uninstall_bundle(bi, logger, session)
        self._write_cache(self._installed_bundle_info, logger)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_UNINSTALLED, bi)

    def find_bundle(self, name, installed=True, version=None):
        """Return a :py:class:`BundleInfo` instance with the given name.

        Parameters
        ----------
        name : str
            Name (internal or display name) of the bundle of interest.
        installed : boolean
            True to check only for installed bundles; False otherwise.
        version : str
            None to find any version; specific string to check for
            one particular version.

        """
        _debug("find_bundle", name, installed, version)
        if installed:
            container = self._installed_bundle_info
        else:
            container = self._available_bundle_info
        from distlib.version import NormalizedVersion as Version
        best_bi = None
        best_version = None
        for bi in container:
            if bi.name != name:
                continue
            if version == bi.version:
                return bi
            if version is None:
                if best_bi is None:
                    best_bi = bi
                    best_version = Version(bi.version)
                else:
                    v = Version(bi.version)
                    if v > best_version:
                        best_bi = bi
                        best_version = v
        return best_bi

    def find_bundle_for_tool(self, name):
        """Find named tool and its bundle

        Return the bundle it is in and its true name.
        """
        folded_name = name.casefold()
        tools = []
        for bi in self._installed_bundle_info:
            for tool in bi.tools:
                tname = tool.name.casefold()
                if tname == folded_name:
                    return (bi, tool.name)
                if tname.startswith(folded_name):
                    tools.append((bi, tool.name))
        if len(tools) == 0:
            return None, name
        # TODO: longest match?
        return tools[0]

    def find_bundle_for_class(self, cls):
        """Find bundle that has given class"""

        package = tuple(cls.__module__.split('.'))
        while package:
            try:
                return self._installed_packages[package]
            except KeyError:
                pass
            package = package[0:-1]
        return None

    def bootstrap_bundles(self, session):
        """Do custom initialization for installed bundles

        After adding the :py:class:`Toolshed` singleton to a session,
        allow bundles need to install themselves into the session,
        (For symmetry, there should be a way to uninstall all bundles
        before a session is discarded, but we don't do that yet.)
        """
        _debug("initialize_bundles")
        failed = []
        for bi in self._installed_bundle_info:
            try:
                bi.initialize(session)
            except ToolshedError:
                failed.append(bi)
        for bi in failed:
            self._installed_bundle_info.remove(bi)
            # TODO: update _installed_packages

    #
    # End public API
    # All methods below are private
    #

    def _load_bundle_infos(self, logger, rebuild_cache=False):
        # Load bundle info.  If not rebuild_cache, try reading
        # it from a cache file.  If we cannot use the cache,
        # read the information from the data directory and
        # try to create the cache file.
        _debug("_load_bundle_infos", rebuild_cache)
        if not rebuild_cache:
            bi_list = self._read_cache()
            if bi_list is not None:
                return bi_list
        self._scan_installed(logger)
        bi_list = []
        for d in self._inst_tool_dists:
            bi = self._make_bundle_info(d, True, logger)
            if bi is not None:
                bi_list.append(bi)
        self._write_cache(bi_list, logger)
        return bi_list

    @_hack_distlib
    def _scan_installed(self, logger):
        # Scan installed packages for ChimeraX bundles

        # Initialize distlib paths and locators
        _debug("_scan_installed")
        if self._inst_locator is None:
            from distlib.database import DistributionPath
            self._inst_path = DistributionPath()
            _debug("_inst_path", self._inst_path)
            from distlib.locators import DistPathLocator
            self._inst_locator = DistPathLocator(self._inst_path)
            _debug("_inst_locator", self._inst_locator)

        # Keep only wheels

        all_distributions = []
        for d in self._inst_path.get_distributions():
            try:
                d.run_requires
                _debug("_scan_installed distribution", d)
            except:
                continue
            else:
                all_distributions.append(d)

        # Look for core package
        core = self._inst_locator.locate(_ChimeraCore)
        if core is None:
            self._inst_core = set()
            self._inst_tool_dists = OrderedSet()
            logger.warning("\"%s\" bundle not found" % _ChimeraCore)
            return

        # Partition packages into core and bundles
        from distlib.database import make_graph
        dg = make_graph(all_distributions)
        known_dists = set([core])
        self._inst_chimera_core = core
        self._inst_core = set([core])
        self._inst_tool_dists = OrderedSet()
        self._all_installed_distributions = {_ChimeraCore: core}
        for d, label in dg.adjacency_list[core]:
            known_dists.add(d)
            self._inst_core.add(d)
            self._all_installed_distributions[d.name] = d
        check_list = list(all_distributions)
        check_list.sort(key=lambda d: len(dg.adjacency_list[d]))
        count = 0
        # Use an upper bound to prevent circular dependencies from
        # breaking computing the dependency order.  In theory, this
        # bound might not be high enough, but it should be fine in
        # practice.
        upper_bound = 2 * len(check_list) * max(
            [len(v) for k, v in dg.adjacency_list.items()], default=1)
        while count < len(check_list):
            dist = check_list[count]
            count += 1
            if count < upper_bound:
                # not a circular circular reference
                if dist in known_dists:
                    continue
                _debug("checking", dist)
                unknown = any(d for d, label in dg.adjacency_list[dist]
                              if d not in known_dists)
                if unknown:
                    check_list.append(dist)
                    continue
            # all dependencies are known or there is a circular reference
            known_dists.add(dist)
            self._inst_tool_dists.add(dist)
            self._all_installed_distributions[dist.name] = dist

    def _bundle_cache(self, must_exist):
        """Return path to bundle cache file."""
        _debug("_bundle_cache", must_exist)
        if must_exist:
            import os
            os.makedirs(self._cache_dir, exist_ok=True)
        import os.path
        return os.path.join(self._cache_dir, "bundle_info.cache")

    def _read_cache(self):
        """Read installed bundle information from cache file.

        Returns boolean on whether cache file was read."""
        _debug("_read_cache")
        import filelock
        import json
        cache_file = self._bundle_cache(False)
        try:
            lock = filelock.FileLock(cache_file + '.lock')
            with lock.acquire():
                f = open(cache_file, "r", encoding='utf-8')
                try:
                    with f:
                        data = json.load(f)
                    return [BundleInfo.from_cache_data(x) for x in data]
                except:
                    return None
        except OSError:
            return None

    def _write_cache(self, bundle_info, logger):
        """Write current bundle information to cache file."""
        _debug("_write_cache", bundle_info)
        import filelock
        import json
        cache_file = self._bundle_cache(True)
        lock = filelock.FileLock(cache_file + '.lock')
        with lock.acquire():
            try:
                f = open(cache_file, 'w', encoding='utf-8')
            except IOError as e:
                logger.error("\"%s\": %s" % (cache_file, str(e)))
            else:
                with f:
                    json.dump([bi.cache_data() for bi in bundle_info], f,
                              ensure_ascii=False, check_circular=False)

    def _old_bundle_info(self, parts, kw, installed, logger, bi):
        name = kw['name']
        # Name of bundle
        bundle_name = parts[1]  # not used
        # Synopsis of bundle/tool/command/format
        synopsis = parts[9]
        # Name of module implementing bundle API
        kw["api_package_name"] = parts[2]
        # Display name of tool
        tool_name = parts[3]
        # Menu categories in which tool should appear
        categories = [v.strip() for v in parts[5].split(',')]
        # CLI command names (just the first word)
        commands = []
        if parts[4]:
            commands = [CommandInfo(v.strip(), categories, synopsis) for v in parts[4].split(',')]
        # File types that bundle can open
        file_types = parts[6]
        types = []
        if file_types:
            for t in file_types.split(','):
                spec = [v.strip() for v in t.split(':')]
                if len(spec) < 3:
                    logger.warning("Malformed ChimeraX-Bundle line in %s skipped." % name)
                    logger.warning("File type has fewer than three fields.")
                    return None
                format_name, format_category, suffix = spec
                format = FormatInfo(format_name, format_category, suffixes=[suffix])
                format.has_open = True
                types.append(format)
        # Session version numbers
        session_versions = parts[7]
        if session_versions:
            vs = [v.strip() for v in session_versions.split(',')]
            if len(vs) != 2:
                logger.warning("Malformed ChimeraX-Bundle line in %s skipped." % name)
                logger.warning("Expected 2 version numbers and got %d." % len(vs))
                return None
            try:
                lo = int(vs[0])
                hi = int(vs[1])
            except ValueError:
                logger.warning("Malformed ChimeraX-Bundle line in %s skipped." % name)
                logger.warning("Found non-integer version numbers.")
                return kw
            if lo > hi:
                logger.warning("Minimum version is greater than maximium.")
                hi = lo
            kw["session_versions"] = range(lo, hi + 1)
        # Does bundle have custom initialization code?
        custom_init = parts[8]
        if custom_init:
            kw["custom_init"] = (custom_init == "true")
        if not bi:
            bi = BundleInfo(installed=installed, **kw)
        if 'Hidden' not in categories:
            ti = ToolInfo(tool_name, categories, synopsis)
            bi.tools.append(ti)
        bi.formats.extend(types)
        bi.commands.extend(commands)
        return bi

    def _make_bundle_info(self, d, installed, logger):
        """Convert distribution into a list of :py:class:`BundleInfo` instances."""
        name = d.name
        version = d.version
        md = d.metadata.dictionary

        if 'classifiers' not in md:
            return None

        bi = None
        kw = {"name": name, "version": version}
        try:
            kw['synopsis'] = md["summary"]
        except KeyError:
            return None
        kw['packages'] = _get_installed_packages(d)
        for classifier in md["classifiers"]:
            parts = [v.strip() for v in classifier.split("::")]
            if parts[0] == "ChimeraX-Bundle":
                # 'ChimeraX-Bundle' :: categories :: session_versions :: module_name :: custom_init
                if len(parts) == 10:
                    bi = self._old_bundle_info(parts, kw, installed, logger, bi)
                    continue
                elif bi is not None:
                    logger.warning("Second ChimeraX-Bundle line ignored.")
                    break
                elif len(parts) != 5:
                    logger.warning("Malformed ChimeraX-Bundle line in %s skipped." % name)
                    logger.warning("Expected 5 fields and got %d." % len(parts))
                    continue
                # Categories in which bundle should appear
                categories = parts[1]
                kw["categories"] = [v.strip() for v in categories.split(',')]
                # Session version numbers
                session_versions = parts[2]
                if session_versions:
                    vs = [v.strip() for v in session_versions.split(',')]
                    if len(vs) != 2:
                        logger.warning("Malformed ChimeraX-Bundle line in %s skipped." % name)
                        logger.warning("Expected 2 version numbers and got %d." % len(vs))
                        return None
                    try:
                        lo = int(vs[0])
                        hi = int(vs[1])
                    except ValueError:
                        logger.warning("Malformed ChimeraX-Bundle line in %s skipped." % name)
                        logger.warning("Found non-integer version numbers.")
                        return kw
                    if lo > hi:
                        logger.warning("Minimum version is greater than maximium.")
                        hi = lo
                    kw["session_versions"] = range(lo, hi + 1)
                # Name of package implementing bundle API
                kw["api_package_name"] = parts[3]
                # Does bundle have custom initialization code?
                custom_init = parts[4]
                if custom_init:
                    kw["custom_init"] = (custom_init == "true")
                bi = BundleInfo(installed=installed, **kw)
            elif parts[0] == "ChimeraX-Tool":
                # 'ChimeraX-Tool' :: tool_name :: categories :: synopsis
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                if len(parts) != 4:
                    logger.warning("Malformed ChimeraX-Tool line in %s skipped." % name)
                    logger.warning("Expected 4 fields and got %d." % len(parts))
                    continue
                # Menu categories in which tool should appear
                name = parts[1]
                categories = parts[2]
                if not categories:
                    logger.warning("Missing tool categories")
                    continue
                categories = [v.strip() for v in categories.split(',')]
                synopsis = parts[3]
                ti = ToolInfo(name, categories, synopsis)
                bi.tools.append(ti)
            elif parts[0] == "ChimeraX-Command":
                # 'ChimeraX-Command' :: name :: categories :: synopsis
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                if len(parts) != 4:
                    logger.warning("Malformed ChimeraX-Command line in %s skipped." % name)
                    logger.warning("Expected 4 fields and got %d." % len(parts))
                    continue
                name = parts[1]
                categories = parts[2]
                if not categories:
                    logger.warning("Missing command categories")
                    continue
                categories = [v.strip() for v in categories.split(',')]
                synopsis = parts[3]
                ci = CommandInfo(name, categories, synopsis)
                bi.commands.append(ci)
            elif parts[0] == "ChimeraX-Selector":
                # 'ChimeraX-Selector' :: name :: synopsis
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                if len(parts) != 3:
                    logger.warning("Malformed ChimeraX-Selector line in %s skipped." % name)
                    logger.warning("Expected 3 fields and got %d." % len(parts))
                    continue
                name = parts[1]
                synopsis = parts[2]
                si = SelectorInfo(name, synopsis)
                bi.selectors.append(si)
            elif parts[0] == "ChimeraX-DataFormat":
                # ChimeraX-DataFormat :: format_name :: alternate_names :: category :: suffixes :: mime_types :: url :: dangerous :: icon :: synopsis
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                if len(parts) != 10:
                    logger.warning("Malformed ChimeraX-DataFormat line in %s skipped." % name)
                    logger.warning("Expected 3 fields and got %d." % len(parts))
                    continue
                name = parts[1]
                alternates = [v.strip() for v in parts[2].split(',')] if parts[2] else None
                category = parts[3]
                suffixes = [v.strip() for v in parts[4].split(',')] if parts[4] else None
                mime_types = [v.strip() for v in parts[5].split(',')] if parts[5] else None
                url = parts[6]
                dangerous = parts[7]
                icon = parts[8]
                synopsis = parts[9]
                fi = FormatInfo(name=name, alternates=alternates,
                                category=category, suffixes=suffixes,
                                mime_types=mime_types, url=url, icon=icon,
                                dangerous=dangerous, synopsis=synopsis)
                bi.formats.append(fi)
            elif parts[0] == "ChimeraX-Fetch":
                # ChimeraX-Fetch :: database_name :: format_name :: prefixes :: example_id :: is_default
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                database_name = parts[1]
                format_name = parts[2]
                prefixes = [v.strip() for v in parts[3].split(',')] if parts[3] else ()
                example_id = parts[4]
                is_default = (parts[5] == 'true')
                bi.fetches.append((database_name, format_name, prefixes, example_id, is_default))
            elif parts[0] == "ChimeraX-Open":
                # ChimeraX-Open :: format_name :: tag :: is_default
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                if len(parts) != 4:
                    logger.warning("Malformed ChimeraX-Open line in %s skipped." % name)
                    logger.warning("Expected 4 fields and got %d." % len(parts))
                    continue
                name = parts[1]
                tag = parts[2]
                is_default = parts[3]
                try:
                    fi = [fi for fi in bi.formats if fi.name == name][0]
                except KeyError:
                    logger.warning("Unknown format name: %r." % name)
                    continue
                fi.has_open = True
            elif parts[0] == "ChimeraX-Save":
                if bi is None:
                    logger.warning('ChimeraX-Bundle entry must be first')
                    return None
                name = parts[1]
                tag = parts[2]
                is_default = parts[3]
                try:
                    fi = [fi for fi in bi.formats if fi.name == name][0]
                except KeyError:
                    logger.warning("Unknown format name: %r." % name)
                    continue
                fi.has_save = True
        if bi is None:
            return None
        try:
            description = md['extensions']['python.details']['document_names']['description']
            import os
            dpath = os.path.join(d.path, description)
            description = open(dpath, encoding='utf-8').read()
            if description.startswith("UNKNOWN"):
                description = "Missing bundle description"
            bi.description = description
        except (KeyError, OSError):
            pass
        return bi

    # Following methods are used for installing and removing
    # distributions

    def _install_bundle(self, bundle_info, system, logger, session):
        # Install a bundle.  This entails:
        #  - finding all distributions that this one depends on
        #  - making sure things will be compatible if installed
        #  - installing all the distributions
        #  - updating any bundle installation status
        _debug("_install_bundle")
        want_update = []
        need_update = []
        self._install_dist_tool(bundle_info, want_update, logger)
        self._install_cascade(want_update, need_update, logger)
        incompatible = self._install_check_incompatible(need_update, logger)
        if incompatible:
            return
        self._install_wheels(need_update, system, logger)
        # update bundle installation status
        updated = set([d.name for d in need_update])
        keep = [bi for bi in self._installed_bundle_info
                if bi.name not in updated]
        self._installed_bundle_info = keep
        # TODO: update _installed_packages
        updated = set([(d.name, d.version) for d in need_update])
        if self._all_installed_distributions is not None:
            self._inst_path = None
            self._inst_locator = None
            self._all_installed_distributions = None
        import copy
        newly_installed = [copy.copy(bi) for bi in self._available_bundle_info
                           if bi.distribution() in updated]
        for bi in newly_installed:
            bi.installed = True
            self.add_bundle_info(bi)
            bi.register()
            if session is not None:
                bi.initialize(session)

    def _install_dist_core(self, want, logger):
        # Add ChimeraX core distribution to update list
        _debug("_install_dist_core")
        d = self._install_distribution(_ChimeraCore, None, logger)
        if d:
            want.append(d)

    def _install_dist_tool(self, bundle_info, want, logger):
        # Add the distribution that provides the
        # given bundle to update list
        _debug("_install_dist_tool", bundle_info)
        d = self._install_distribution(bundle_info.name,
                                       bundle_info.version, logger)
        if d:
            want.append(d)

    def _install_distribution(self, name, version, logger):
        # Return either a distribution that needs to be
        # installed/updated or None if it is already
        # installed and up-to-date
        _debug("_install_distribution", name)
        req = name
        if version:
            req += " (== %s)" % version
        repo_dist = self._repo_locator.locate(req)
        if repo_dist is None:
            raise ToolshedUnavailableError("cannot find new distribution "
                                           "named \"%s\"" % name)
        if self._inst_locator is None:
            self._scan_installed(logger)
        inst_dist = self._inst_locator.locate(name)
        if inst_dist is None:
            return repo_dist
        else:
            from distlib.version import NormalizedVersion as Version
            inst_version = Version(inst_dist.version)
            # Check if installed version is the same as requested version
            if version is not None:
                if inst_version != Version(version):
                    return repo_dist
            repo_version = Version(repo_dist.version)
            if inst_version < repo_version:
                return repo_dist
            elif inst_version > repo_version:
                logger.warning("installed \"%s\" is newer than latest: %s > %s"
                               % (name, inst_dist.version, repo_dist.version))
        return None

    def _install_cascade(self, want, need, logger):
        # Find all distributions that need to be installed
        # in order for distributions on the ``want`` list to work
        _debug("_install_cascade", want)
        seen = set()
        check = set(want)
        while check:
            d = check.pop()
            seen.add(d)
            need.append(d)
            for req in d.run_requires:
                nd = self._install_distribution(req, None, logger)
                if nd and nd not in seen:
                    check.add(nd)

    def _get_all_installed_distributions(self, logger):
        _debug("_get_all_installed_distributions")
        if self._all_installed_distributions is None:
            self._scan_installed(logger)
        return self._all_installed_distributions

    def _install_check_incompatible(self, need, logger):
        # Make sure everything is compatible (no missing or
        # conflicting distribution requirements)
        _debug("_install_check_incompatible", need)
        all = dict(self._get_all_installed_distributions(logger).items())
        all.update([(d.name, d) for d in need])
        _debug("all", all)
        from distlib.database import make_graph
        graph = make_graph(all.values())
        if graph.missing:
            _debug("graph.missing", graph.missing)
            from ..commands import commas
            for d, req_list in graph.missing.items():
                s = commas([repr(r) for r in req_list], " and ")
                logger.warning("\"%s\" needs %s" % (d.name, s))
            return True
        else:
            return False

    def _install_wheels(self, need, system, logger):
        # Find all packages that should be deleted
        _debug("_install_wheels", need, system)
        all = self._get_all_installed_distributions(logger)
        from distlib.database import make_graph
        import itertools
        graph = make_graph(itertools.chain(all.values(), need))
        l = need[:]    # what we started with
        ordered = []   # ordered by least dependency
        depend = {}    # dependency relationship cache
        while l:
            for d in l:
                for d2 in l:
                    if d2 is d:
                        continue
                    try:
                        dep = depend[(d, d2)]
                    except KeyError:
                        dep = self._depends_on(graph, d, d2)
                        depend[(d, d2)] = dep
                    if dep:
                        break
                else:
                    ordered.append(d)
                    l.remove(d)
                    break
            else:
                # This can only happen if there is
                # circular dependencies in which case
                # we just process the distributions in
                # given order since its no worse than
                # anything else
                ordered.extend(l)
                break
        remove_list = []
        check = set()
        for d in ordered:
            if d in remove_list:
                continue
            try:
                rd = all[d.name]
            except KeyError:
                pass
            else:
                remove_list.append(rd)
                al = graph.adjacency_list[rd]
                if al:
                    check.update([sd for sd, sl in al])
        # Repeatedly go through the list of distributions to
        # see whether they can be removed.  It must be iterative.
        # Suppose A and B need to be removed; C depends on A;
        # D depends on B and C; if we check D first, it will not
        # be removable since C is not marked for removal
        # yet; but a second pass will show that D is removable.
        # Iteration ends when no new packages are marked as removable.
        while check:
            any_deletion = False
            new_check = set()
            for d in check:
                for pd in graph.reverse_list[d]:
                    if pd not in remove_list:
                        new_check.add(d)
                        break
                else:
                    any_deletion = True
                    remove_list.append(d)
                    for sd, l in graph.adjacency_list[d]:
                        if (sd not in remove_list and sd not in check):
                            new_check.add(sd)
            if not any_deletion:
                break
            check = new_check

        # If a package is being updated, it should be
        # installed in the same location as before, so we
        # need to keep track.
        old_location = {}
        for d in remove_list:
            old_location[d.name] = self._remove_distribution(d, logger)

        # Now we (re)install the needed distributions
        import os.path
        wheel_cache = os.path.join(self._cache_dir, "wheels.cache")
        import os
        os.makedirs(wheel_cache, exist_ok=True)
        default_paths = self._install_make_paths(system)
        from distlib.scripts import ScriptMaker
        maker = ScriptMaker(None, None)
        try:
            from urllib.request import urlretrieve, URLError
        except ImportError:
            from urllib import urlretrieve, URLError
        from distlib.wheel import Wheel
        from distlib import DistlibException
        for d in need:
            try:
                old_site = old_location[d.name]
            except KeyError:
                paths = default_paths
            else:
                paths = self._install_make_paths(system, old_site)
            url = d.source_url
            filename = url.split('/')[-1]
            dloc = os.path.join(wheel_cache, filename)
            if not os.path.isfile(dloc):
                need_fetch = True
            else:
                t = d.metadata.dictionary["modified"]
                import calendar
                import time
                d_mtime = calendar.timegm(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
                c_mtime = os.path.getmtime(dloc)
                # print("distribution", time.ctime(d_mtime))
                # print("cache", time.ctime(c_mtime))
                need_fetch = (d_mtime > c_mtime)
            if need_fetch:
                # print("fetching wheel")
                try:
                    fn, headers = urlretrieve(url, dloc)
                except URLError as e:
                    logger.warning("cannot fetch %s: %s" % (url, str(e)))
                    continue
            else:
                # print("using cached wheel")
                pass
            w = Wheel(dloc)
            try:
                w.verify()
            except DistlibException as e:
                logger.warning("cannot verify %s: %s" % (d.name, str(e)))
                continue
            logger.info("installing %s (%s)" % (w.name, w.version))
            _debug("paths", paths)
            w.install(paths, maker)

    def _install_make_paths(self, system, sitepackages=None):
        # Create path associated with either only-this-user
        # or system distributions
        _debug("_install_make_paths", system)
        import site
        import sys
        import os.path
        if system:
            base = sys.prefix
        else:
            base = self._data_dir
        if sitepackages is None:
            if system:
                sitepackages = site.getsitepackages()[-1]
            else:
                sitepackages = self._site_dir
        paths = {
            "prefix": sys.prefix,
            "purelib": sitepackages,
            "platlib": sitepackages,
            "headers": os.path.join(base, "include"),
            "scripts": os.path.join(base, "bin"),
            "data": os.path.join(base, "lib"),
        }
        return paths

    def _depends_on(self, graph, da, db):
        # Returns whether distribution "da" depends on "db"
        # "graph" is a distlib.depgraph.DependencyGraph instance
        # Do depth-first search
        for depa, label in graph.adjacency_list[da]:
            if depa is db or self._depends_on(graph, depa, db):
                return True
        return False

    def _remove_distribution(self, d, logger):
        _debug("_remove_distribution", d)
        from distlib.database import InstalledDistribution
        if not isinstance(d, InstalledDistribution):
            raise ToolshedUninstalledError("trying to remove uninstalled "
                                           "distribution: %s (%s)"
                                           % (d.name, d.version))
        # HACK ALERT: since there is no API for uninstalling
        # a distribution (as of distlib 0.1.9), here's my hack:
        #   assume that d.list_installed_files() returns paths
        #     relative to undocumented dirname(d.path)
        #   remove all listed installed files while keeping track of
        #     directories from which we removed files
        #   try removing the directories, longest first (this will
        #     remove children directories before parents)
        import os.path
        basedir = os.path.dirname(d.path)
        dircache = set()
        try:
            for path, hash, size in d.list_installed_files():
                p = os.path.join(basedir, path)
                os.remove(p)
                dircache.add(os.path.dirname(p))
        except OSError as e:
            logger.warning("cannot remove distribution: %s" % str(e))
            return basedir
        try:
            # Do not try to remove the base directory (probably
            # "site-packages somewhere)
            dircache.remove(basedir)
        except KeyError:
            pass
        for d in reversed(sorted(dircache, key=len)):
            try:
                os.rmdir(d)
            except OSError as e:
                # If directory not empty, just ignore
                pass
        return basedir

    def _uninstall_bundle(self, bundle_info, logger, session):
        _debug("_uninstall", bundle_info)
        dv = bundle_info.distribution()
        name, version = dv
        all = self._get_all_installed_distributions(logger)
        d = all[name]
        if d.version != version:
            raise KeyError("distribution \"%s %s\" does not match bundle version "
                           "\"%s\"" % (name, version, d.version))
        keep = []
        for bi in reversed(self._installed_bundle_info):
            if bi.distribution() != dv:
                keep.append(bi)
            else:
                for p in bi.packages:
                    try:
                        del self._installed_packages[p]
                    except KeyError:
                        pass
                bi.deregister()
                if session is not None:
                    bi.finish(session)
        self._installed_bundle_info = list(reversed(keep))
        # TODO: update _installed_packages
        self._remove_distribution(d, logger)

    # End methods for installing and removing distributions


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

    def __init__(self, name, category, alternates=None, suffixes=None,
                 mime_types=None, url=None, synopsis=None,
                 dangerous=None, icon=None,
                 has_open=False, has_save=False):
        self.name = name
        self.alternatives = alternates
        self.category = category
        self.suffixes = suffixes
        self.mime_types = mime_types
        self.documentation_url = url
        self.dangerous = dangerous
        # self.encoding = encoding
        self.icon = icon
        self.synopsis = synopsis
        self.has_open = has_open
        self.has_save = has_save

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
        # if self.encoding:
        #     s += " [encoding: %s]" % self.encoding
        if self.synopsis:
            s += " [synopsis: %s]" % self.synopsis
        return s

    def cache_data(self):
        return {
            'name': self.name,
            'category': self.category,
            'suffixes': self.suffixes,
            'mime_types': self.mime_types,
            'url': self.documentation_url,
            'dangerous': self.dangerous,
            'icon': self.icon,
            'synopsis': self.synopsis,
            'alternatives': self.alternatives,
            'has_open': self.has_open,
            'has_save': self.has_save,
        }

    @classmethod
    def from_cache_data(cls, data):
        return cls(**data)


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
                 packages=[]):
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

    def register(self):
        self._register_commands()
        self._register_file_types()
        self._register_selectors()

    def deregister(self):
        self._deregister_selectors()
        self._deregister_file_types()
        self._deregister_commands()

    def _register_commands(self):
        """Register commands with cli."""
        from chimerax.core.commands import cli
        for ci in self.commands:
            def cb(s=self, n=ci.name):
                s._register_cmd(n)
            _debug("delay_registration", ci.name)
            cli.delay_registration(ci.name, cb)

    def _register_cmd(self, command_name):
        """Called when commands need to be really registered."""
        try:
            f = self._get_api().register_command
        except AttributeError:
            raise ToolshedError(
                "no register_command function found for bundle \"%s\""
                % self.name)
        if f == BundleAPI.register_command:
            raise ToolshedError("bundle \"%s\"'s API forgot to override register_command()" % self.name)
        f(command_name)

    def _deregister_commands(self):
        """Deregister commands with cli."""
        from chimerax.core.commands import cli
        for ci in self.commands:
            _debug("deregister_command", ci.name)
            try:
                cli.deregister(ci.name)
            except RuntimeError:
                pass  # don't care if command was already missing

    def _register_file_types(self):
        """Register file types."""
        from chimerax.core import io, fetch
        for fi in self.formats:
            _debug("register_file_type", fi.name)
            format = io.register_format(fi.name, fi.category, fi.suffixes)
            if fi.has_open:
                def open_cb(*args, format_name=fi.name, **kw):
                    try:
                        f = self._get_api().open_file
                    except AttributeError:
                        raise ToolshedError(
                            "no open_file function found for bundle \"%s\""
                            % self.name)
                    if f == BundleAPI.open_file:
                        raise ToolshedError("bundle \"%s\"'s API forgot to override open_file()" % self.name)

                    # optimize by replacing open_func for format
                    def open_shim(*args, f=f, format_name=format_name, **kw):
                        return f(*args, format_name=format_name, **kw)
                    format = io.format_from_name(format_name)
                    format.open_func = open_shim
                    return open_shim(*args, **kw)
                format.open_func = open_cb
            if fi.has_save:
                def save_cb(*args, format_name=fi.name, **kw):
                    try:
                        f = self._get_api().save_file
                    except AttributeError:
                        raise ToolshedError(
                            "no save_file function found for bundle \"%s\""
                            % self.name)
                    if f == BundleAPI.save_file:
                        raise ToolshedError("bundle \"%s\"'s API forgot to override save_file()" % self.name)

                    # optimize by replacing save_func for format
                    def save_shim(*args, f=f, format_name=format_name, **kw):
                        return f(*args, format_name=format_name, **kw)
                    format = io.format_from_name(format_name)
                    format.export_func = save_shim
                    return save_shim(*args, format_name=format_name, **kw)
                format.export_func = save_cb
        for (database_name, format_name, prefixes, example_id, is_default) in self.fetches:
            if io.format_from_name(format_name) is None:
                print('warning: unknown format %r given for database %r' % (format_name, database_name))
            def fetch_cb(session, identifier, database_name=database_name, format_name=format_name, **kw):
                try:
                    f = self._get_api().fetch_url
                except AttributeError:
                    raise ToolshedError(
                        "no fetch_url function found for bundle \"%s\""
                        % self.name)
                if f == BundleAPI.save_file:
                    raise ToolshedError("bundle \"%s\"'s API forgot to override fetch_url()" % self.name)
                # optimize by replacing fetch_url for (database, format)
                def fetch_shim(session, identifier, f=f, database_name=database_name, format_name=format_name, **kw):
                    return f(session, identifier, database_name=database_name, format_name=format_name, **kw)
                fetch.register_fetch(database_name, fetch_shim, format_name)
                return fetch_shim(session, identifier, **kw)
            fetch.register_fetch(
                database_name, fetch_cb, format_name, prefixes=prefixes,
                is_default_format=is_default, example_id=example_id)

    def _deregister_file_types(self):
        """Deregister file types."""
        # TODO: implement
        pass

    def _register_selectors(self):
        from ..commands import register_selector
        for si in self.selectors:
            def selector_cb(session, models, results, _name=si.name):
                try:
                    reg = self._get_api().register_selector
                except AttributeError:
                    raise ToolshedError(
                        "no register_selector function found for bundle \"%s\""
                        % self.name)
                if reg == BundleAPI.register_selector:
                    raise ToolshedError("bundle \"%s\"'s API forgot to override register_selector()" % self.name)
                reg(_name)
                from ..commands import get_selector
                return get_selector(_name)(session, models, results)
            register_selector(si.name, selector_cb)

    def _deregister_selectors(self):
        from ..commands import deregister_selector
        for si in self.selectors:
            deregister_selector(si.name)

    def initialize(self, session):
        """Initialize bundle by calling custom initialization code if needed."""
        if self.custom_init:
            try:
                f = self._get_api().initialize
            except AttributeError:
                raise ToolshedError(
                    "no initialize function found for bundle \"%s\""
                    % self.name)
            if f == BundleAPI.initialize:
                session.logger.warning("bundle \"%s\"'s API forgot to override initialize()" % self.name)
                return
            f(session, self)

    def finish(self, session):
        """Deinitialize bundle by calling custom finish code if needed."""
        if self.custom_init:
            try:
                f = self._get_api().finish
            except AttributeError:
                raise ToolshedError("no finish function found for bundle \"%s\""
                                    % self.name)
            if f == BundleAPI.finish:
                session.logger.warning("bundle \"%s\"'s API forgot to override finish()" % self.name)
                return
            f(session, self)

    def get_class(self, class_name):
        """Return bundle's class with given name."""
        try:
            f = self._get_api().get_class
        except AttributeError:
            raise ToolshedError("no get_class function found for bundle \"%s\""
                                % self.name)
        return f(class_name)

    def _get_api(self):
        """Return BundleAPI instance for this bundle."""
        if not self._api_package_name:
            raise ToolshedError("no API package specified for bundle \"%s\"" % self.name)
        import importlib
        try:
            m = importlib.import_module(self._api_package_name)
        except Exception as e:
            raise ToolshedError("Error importing bundle API \"%s\": %s" % (self.name, str(e)))
        try:
            bundle_api = getattr(m, 'bundle_api')
        except AttributeError:
            raise ToolshedError("missing bundle_api for bundle \"%s\"" % self.name)
        _debug("_get_api", self._api_package_name, m, bundle_api)
        return bundle_api

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
            raise ToolshedUninstalledError("bundle \"%s\" is not installed"
                                           % self.name)
        if not session.ui.is_gui:
            raise ToolshedError("tool \"%s\" is not supported without a GUI"
                                % tool_name)
        try:
            f = self._get_api().start_tool
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


class BundleAPI:
    """API for accessing bundles

    The metadata for the bundle indicates which of the methods need to be
    implemented.
    """

    @staticmethod
    def start_tool(session, tool_name):
        """Called to lazily create a tool instance.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        tool_name : str.

        Errors should be reported via exceptions.
        """
        raise NotImplementedError

    @staticmethod
    def register_command(command_name):
        """Called when delayed command line registration occurs.

        Parameters
        ----------
        command_name : :py:class:`str`

        ``command_name`` is a string of the command to be registered.
        This function is called when the command line interface is invoked
        with one of the registered command names.
        """
        raise NotImplementedError

    @staticmethod
    def register_selector(selector_name):
        """Called when delayed selector registration occurs.

        Parameters
        ----------
        selector_name : :py:class:`str`

        ``selector_name`` is the name of the selector to be registered.
        This function is called when the selector invoked with one of
        the registered names.
        """
        raise NotImplementedError

    @staticmethod
    def open_file(session, stream, name, **kw):
        """Called to open a file.

        Arguments and return values are as described for open functions in
        :py:mod:`chimerax.core.io`.
        The format name will be in the **format_name** keyword.
        """
        raise NotImplementedError

    @staticmethod
    def save_file(session, stream, name, **kw):
        """Called to save a file.

        Arguments and return values are as described for save functions in
        :py:mod:`chimerax.core.io`.
        The format name will be in the **format_name** keyword.
        """
        raise NotImplementedError

    @staticmethod
    def initialize(session, bundle_info):
        """Called to initialize a bundle in a session.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.

        Must be defined if the ``custom_init`` metadata field is set to 'true'.
        ``initialize`` is called when the bundle is first loaded.
        To make ChimeraX start quickly, custom initialization is discouraged.
        """
        raise NotImplementedError

    @staticmethod
    def finish(session, bundle_info):
        """Called to deinitialize a bundle in a session.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.

        Must be defined if the ``custom_init`` metadata field is set to 'true'.
        ``finish`` is called when the bundle is unloaded.
        """
        raise NotImplementedError

    @staticmethod
    def get_class(name):
        """Called to get named class from bundle.

        Parameters
        ----------
        name : str
            Name of class in bundle.

        Used when restoring sessions.  Instances whose class can't be found via
        'get_class' can not be saved in sessions.  And those classes must implement
        the :py:class:`~chimerax.core.state.State` API.
        """
        return None


# Toolshed is a singleton.  Multiple calls to init returns the same instance.
_toolshed = None


def init(*args, debug=False, **kw):
    """Initialize toolshed.

    The toolshed instance is a singleton across all sessions.
    The first call creates the instance and all subsequent
    calls return the same instance.  The toolshed debugging
    state is updated at each call.

    Parameters
    ----------
    debug : boolean
        If true, debugging messages are sent to standard output.
        Default value is false.
    other arguments : any
        All other arguments are passed to the `Toolshed` initializer.

    Returns
    -------
    :py:class:`Toolshed` instance
        The toolshed singleton.
    """
    global _debug
    if debug:
        def _debug(*args, **kw):
            import sys
            print("Toolshed:", *args, file=sys.stderr, **kw)
    else:
        def _debug(*args, **kw):
            return
    global _toolshed
    if _toolshed is None:
        _toolshed = Toolshed(*args, **kw)
    return _toolshed
