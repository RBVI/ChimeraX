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
The 'ChimeraX' classifier entries give additional information.
Depending on the values of 'ChimeraX' metadata fields,
modules need to override methods of the :py:class:`BundleAPI` class.
Each bundle needs a 'ChimeraX :: Bundle' entry
that consists of the following fields separated by double colons (``::``).

1. ``ChimeraX :: Bundle`` : str constant
    Field identifying entry as bundle metadata.
2. ``categories`` : str
    Comma-separated list of categories in which the bundle belongs.
3. ``session_versions`` : two comma-separated integers
    Minimum and maximum session version that the bundle can read.
4. ``supercedes`` : str
   Comma-separated list of superceded bundle names.
5. ``custom_init`` : str
    Whether bundle has initialization code that must be called when
    ChimeraX starts.  Either 'true' or 'false'.  If 'true', the bundle
    must override the BundleAPI's 'initialize' and 'finish' functions.

Bundles that provide tools need:

1. ``ChimeraX :: Tool`` : str constant
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

1. ``ChimeraX :: Command`` : str constant
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

1. ``ChimeraX :: Selector`` : str constant
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

1. ``ChimeraX :: DataFormat`` : str constant
    Field identifying entry as data format metadata.
2. ``data_name`` : str
    The name of the data format.
3. ``nicknames`` : str
    An optional comma-separated list of alternative names.
    Often a short name is provided.  If not provided,
    it defaults to the lowercase version of the data format name.
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
10. ``synopsis`` : str
    A short description of the data format.  It is here
    because it needs to be part of the metadata available for
    uninstalled data format, so that users can get more than just a
    name for deciding whether they want the data format or not.

Bundles may provide more than one data format.
The data format metadata includes everything needed for the Mac OS X
application property list.

Data formats that can be fetched:

# ChimeraX :: Fetch :: database_name :: format_name :: prefixes :: example_id :: is_default

Data formats that can be opened:

# ChimeraX :: Open :: format_name :: tag :: is_default

Data formats that can be saved:

# ChimeraX :: Save :: format_name :: tag :: is_default

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

_TIMESTAMP = 'install-timestamp'
_debug_toolshed = False


def _debug(*args, **kw):
    if _debug_toolshed:
        import sys
        print("Toolshed:", *args, file=sys.__stderr__, flush=True, **kw)


# Package constants


# Default URL of remote toolshed
_RemoteURL = "https://cxtoolshed.rbvi.ucsf.edu"
# Default name for toolshed cache and data directories
_ToolshedFolder = "toolshed"
# Defaults names for installed ChimeraX bundles
_ChimeraNamespace = "chimerax"


# Exceptions raised by Toolshed class


class ToolshedError(Exception):
    """Generic Toolshed error."""


class ToolshedInstalledError(ToolshedError):
    """Bundle-already-installed error.

    This exception derives from :py:class:`ToolshedError` and is usually
    raised when trying to install a bundle that is already installed
    or to uninstall a bundle that is not installed yet."""


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

    def __init__(self, logger, rebuild_cache=False, check_remote=False,
                 remote_url=None, check_available=True):
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
        self._installed_bundle_info = None
        self._available_bundle_info = None
        self._installed_packages = {}   # cache mapping packages to bundles

        # Compute base directories
        import os
        from chimerax import app_dirs
        self._cache_dir = os.path.join(app_dirs.user_cache_dir, _ToolshedFolder)
        _debug("cache dir: %s" % self._cache_dir)
        self._data_dir = os.path.join(app_dirs.user_data_dir, _ToolshedFolder)
        _debug("data dir: %s" % self._data_dir)

        # Add directories to sys.path
        import site
        self._site_dir = site.USER_SITE
        _debug("site dir: %s" % self._site_dir)
        import os
        os.makedirs(self._site_dir, exist_ok=True)
        site.addsitedir(self._site_dir)

        # Create triggers
        from .. import triggerset
        self.triggers = triggerset.TriggerSet()
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INFO_ADDED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INSTALLED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_UNINSTALLED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INFO_RELOADED)

        # Variables for updating list of available bundles
        from threading import RLock
        self._abc_lock = RLock()
        self._abc_updating = False

        # Reload the bundle info list
        _debug("loading bundles")
        self.reload(logger, check_remote=check_remote, rebuild_cache=rebuild_cache)
        if check_available and not check_remote:
            # Did not check for available bundles synchronously
            # so start a thread and do it asynchronously
            self.async_reload_available(logger)
        _debug("finished loading bundles")

    def reload(self, logger, *, session=None, reread_cache=True, rebuild_cache=False,
               check_remote=False, report=False):
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
        if reread_cache or rebuild_cache:
            from .installed import InstalledBundleCache
            save = self._installed_bundle_info
            self._installed_bundle_info = InstalledBundleCache()
            cache_file = self._bundle_cache(False, logger)
            self._installed_bundle_info.load(logger, cache_file=cache_file,
                                             rebuild_cache=rebuild_cache,
                                             write_cache=True)
            if report:
                if save is None:
                    logger.info("Initial installed bundles.")
                else:
                    from .installed import _report_difference
                    _report_difference(logger, save, self._installed_bundle_info)
            if save is not None:
                save.deregister_all(logger, session, self._installed_packages)
            self._installed_bundle_info.register_all(logger, session,
                                                     self._installed_packages)
        if check_remote:
            self.reload_available(logger)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_INFO_RELOADED, self)

    def async_reload_available(self, logger):
        with self._abc_lock:
            self._abc_updating = True
        from threading import Thread
        t = Thread(target=self.reload_available, args=(logger,),
                   name="Update list of available bundles")
        t.start()

    def reload_available(self, logger):
        from urllib.error import URLError
        from .available import AvailableBundleCache
        abc = AvailableBundleCache()
        try:
            abc.load(logger, self.remote_url)
        except URLError as e:
            logger.info("Updating list of available bundles failed: %s"
                        % str(e.reason))
            with self._abc_lock:
                self._abc_updating = False
        except Exception as e:
            logger.info("Updating list of available bundles failed: %s"
                        % str(e))
            with self._abc_lock:
                self._abc_updating = False
        else:
            with self._abc_lock:
                self._available_bundle_info = abc
                self._abc_updating = False
                from ..commands import cli
                cli.clear_available()

    def register_available_commands(self, logger):
        for bi in self._get_available_bundles(logger):
            bi.register_available_commands(logger)

    def set_install_timestamp(self, per_user=False):
        """Set last installed timestamp."""
        _debug("set_install_timestamp")
        self._installed_bundle_info.set_install_timestamp(per_user=per_user)

    def bundle_info(self, logger, installed=True, available=False):
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

        # _installed_bundle_info should always be defined
        # but _available_bundle_info may need to be initialized
        if available and self._available_bundle_info is None:
            self.reload(logger, reread_cache=False, check_remote=True)
        if installed and available:
            return self._installed_bundle_info + self._get_available_bundles(logger)
        elif installed:
            return self._installed_bundle_info
        elif available:
            return self._get_available_bundles(logger)
        else:
            return []

    def install_bundle(self, bundle, logger, *, per_user=True, reinstall=False, session=None):
        """Install the bundle by retrieving it from the remote shed.

        Parameters
        ----------
        bundle : string or :py:class:`BundleInfo` instance
            If string, path to wheel installer.
            If instance, should be from the available bundle list.
        per_user : boolean
            True to install bundle only for the current user (default);
            False to install for everyone.
        reinstall : boolean
            True to force reinstall package.
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
        _debug("install_bundle", bundle)
        # Make sure that our install location is on chimerax module.__path__
        # so that newly installed modules may be found
        import importlib, os.path, re
        cx_dir = os.path.join(self._site_dir, _ChimeraNamespace)
        m = importlib.import_module(_ChimeraNamespace)
        if cx_dir not in m.__path__:
            m.__path__.append(cx_dir)
        try:
            if bundle.installed:
                if not reinstall:
                    raise ToolshedInstalledError("bundle \"%s\" already installed" % bundle.name)
                if bundle in self._installed_bundle_info:
                    bundle.deregister(logger)
                    bundle.unload(logger)
                    self._installed_bundle_info.remove(bundle)
                    # The reload that will happen later will undo the effect
                    # of the unload by accessing the module again, so we
                    # explicitly remove the bundle right now
            bundle = bundle.name
        except AttributeError:
            # If "bundle" is not an instance, it must be a string.
            # Treat it like a path to a wheel and get a putative
            # bundle name.  If it is install, deregister and unload it.
            basename = os.path.split(bundle)[1]
            name = basename.split('-')[0]
            bi = self.find_bundle(name, logger, installed=True)
            if bi in self._installed_bundle_info:
                bi.deregister(logger)
                bi.unload(logger)
                self._installed_bundle_info.remove(bi)
        results = self._pip_install(bundle, per_user=per_user, reinstall=reinstall)
        installed = re.findall(r"^\s*Successfully installed.*$", results, re.M)
        if installed:
            logger.info('\n'.join(installed))
        self.set_install_timestamp(per_user)
        self.reload(logger, rebuild_cache=True, report=True)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_INSTALLED, bundle)

    def uninstall_bundle(self, bundle, logger, *, session=None):
        """Uninstall bundle by removing the corresponding Python distribution.

        Parameters
        ----------
        bundle : string or :py:class:`BundleInfo` instance
            If string, path to wheel installer.
            If instance, should be from the available bundle list.
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
        import re
        _debug("uninstall_bundle", bundle)
        try:
            if not bundle.installed:
                raise ToolshedInstalledError("bundle \"%s\" not installed" % bundle.name)
            bundle.deregister(logger)
            bundle.unload(logger)
            bundle = bundle.name
        except AttributeError:
            # If "bundle" is not an instance, just leave it alone
            pass
        results = self._pip_uninstall(bundle)
        uninstalled = re.findall(r"^\s*Successfully uninstalled.*$", results, re.M)
        if uninstalled:
            logger.info('\n'.join(uninstalled))
        self.reload(logger, rebuild_cache=True, report=True)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_UNINSTALLED, bundle)

    def find_bundle(self, name, logger, installed=True, version=None):
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
            container = self._get_available_bundles(logger)
        from distlib.version import NormalizedVersion as Version
        lc_name = name.lower().replace('_', '-')
        best_bi = None
        best_version = None
        for bi in container:
            if lc_name not in bi.name.lower():
                continue
            #if bi.name != name and name not in bi.supercedes:
            #    continue
            if version == bi.version:
                return bi
            if version is None:
                if best_bi is None:
                    best_bi = bi
                    best_version = Version(bi.version)
                elif best_bi.name != bi.name:
                    logger("%r matches multiple bundles" % name)
                    return None
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
        _debug(session.logger, "initialize_bundles")
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

    def _get_available_bundles(self, logger):
        with self._abc_lock:
            if self._available_bundle_info is None:
                from .available import AvailableBundleCache
                if self._abc_updating:
                    logger.warning("still retrieving bundle list from toolshed")
                else:
                    logger.warning("could not retrieve bundle list from toolshed")
                self._available_bundle_info = AvailableBundleCache()
            elif self._abc_updating:
                logger.warning("still updating bundle list from toolshed")
            return self._available_bundle_info

    def _bundle_cache(self, must_exist, logger):
        """Return path to bundle cache file."""
        _debug("_bundle_cache", must_exist)
        if must_exist:
            import os
            os.makedirs(self._cache_dir, exist_ok=True)
        import os
        return os.path.join(self._cache_dir, "bundle_info.cache")

    def _pip_install(self, bundle_name, per_user=True, reinstall=False):
        # Run "pip" with our standard arguments (index location, update
        # strategy, etc) plus the given arguments.  Return standard
        # output as string.  If there was an error, raise RuntimeError
        # with stderr as parameter.
        import sys
        command = ["install", "--upgrade",
                   "--extra-index-url", self.remote_url + "/pypi/",
                   "--upgrade-strategy", "only-if-needed",
                   # "--only-binary", ":all:"   # msgpack-python is not binary
                   ]
        if per_user:
            command.append("--user")
        if reinstall:
            # XXX: Not sure how this interacts with "only-if-needed"
            command.append("--force-reinstall")
        # bundle_name can be either a file path or a bundle name in repository
        command.append(bundle_name)
        results = self._run_pip(command)
        self._remove_scripts()
        return results

    def _pip_uninstall(self, bundle_name):
        # Run "pip" and return standard output as string.  If there
        # was an error, raise RuntimeError with stderr as parameter.
        import sys
        command = ["uninstall", "--yes", bundle_name]
        return self._run_pip(command)

    def _run_pip(self, command):
        import sys, subprocess
        _debug("_run_pip command:", command)
        cp = subprocess.run([sys.executable, "-m", "pip"] + command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        if cp.returncode != 0:
            output = cp.stdout.decode("utf-8")
            error = cp.stderr.decode("utf-8")
            _debug("_run_pip return code:", cp.returncode, file=sys.__stderr__)
            _debug("_run_pip output:", output, file=sys.__stderr__)
            _debug("_run_pip error:", error, file=sys.__stderr__)
            raise RuntimeError(output + error)
        result = cp.stdout.decode("utf-8")
        _debug("_run_pip result:", result)
        return result

    def _remove_scripts(self):
        # remove pip installed scripts since they have hardcoded paths to
        # python and thus don't work when ChimeraX is installed elsewhere
        from chimerax import app_bin_dir
        import sys, os
        if sys.platform.startswith('win'):
            # Windows
            script_dir = os.path.join(app_bin_dir, 'Scripts')
            for dirpath, dirnames, filenames in os.walk(script_dir, topdown=False):
                for f in filenames:
                    path = os.path.join(dirpath, f)
                    os.remove(path)
                os.rmdir(dirpath)
        else:
            # Linux, Mac OS X
            for filename in os.listdir(app_bin_dir):
                path = os.path.join(app_bin_dir, filename)
                if not os.path.isfile(path):
                    continue
                with open(path, 'br') as f:
                    line = f.readline()
                    if line[0:2] != b'#!' or b'/bin/python' not in line:
                        continue
                #print('removing (pip installed)', path)
                os.remove(path)


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
        raise NotImplementedError("BundleAPI.start_tool")

    @staticmethod
    def register_command(command_name, logger):
        """Called when delayed command line registration occurs.

        Parameters
        ----------
        command_name : :py:class:`str`
        logger : :py:class:`~chimerax.core.logger.Logger` instance.

        ``command_name`` is a string of the command to be registered.
        This function is called when the command line interface is invoked
        with one of the registered command names.
        """
        raise NotImplementedError("BundleAPI.register_command")

    @staticmethod
    def register_selector(selector_name, logger):
        """Called when delayed selector registration occurs.

        Parameters
        ----------
        selector_name : :py:class:`str`
        logger : :py:class:`~chimerax.core.logger.Logger` instance.

        ``selector_name`` is the name of the selector to be registered.
        This function is called when the selector invoked with one of
        the registered names.
        """
        raise NotImplementedError("BundleAPI.register_selector")

    @staticmethod
    def open_file(session, stream, name, **kw):
        """Called to open a file.

        Arguments and return values are as described for open functions in
        :py:mod:`chimerax.core.io`.
        The format name will be in the **format_name** keyword.
        """
        raise NotImplementedError("BundleAPI.open_file")

    @staticmethod
    def save_file(session, stream, name, **kw):
        """Called to save a file.

        Arguments and return values are as described for save functions in
        :py:mod:`chimerax.core.io`.
        The format name will be in the **format_name** keyword.
        """
        raise NotImplementedError("BundleAPI.save_file")

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
        raise NotImplementedError("BundleAPI.initialize")

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
        raise NotImplementedError("BundleAPI.finish")

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


def init(*args, debug=None, **kw):
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
    if debug is not None:
        global _debug_toolshed
        _debug_toolshed = debug
    global _toolshed
    if _toolshed is None:
        _toolshed = Toolshed(*args, **kw)
    return _toolshed
