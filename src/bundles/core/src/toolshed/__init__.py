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

"""
The Toolshed provides an interface for finding installed
bundles as well as bundles available for
installation from a remote server.
The Toolshed can handle updating, installing and uninstalling
bundles while taking care of inter-bundle dependencies.

The Toolshed interface uses :py:mod:`pkg_resources` heavily.

Each Python distribution, a ChimeraX Bundle,
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
4. ``supersedes`` : str
   Comma-separated list of superseded bundle names.
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
4. ``atomic`` : str
    An optional boolean specifying whether the selector applies to
    atoms and bonds.  Defaults to 'true' and should be set to
    'false' if selector should not appear in Basic Actions tool,
    e.g., showing/hiding selected items does nothing.

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
    A string that has a URL that points to the data format's documentation.
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

Bundles that have other data:

# ChimeraX :: DataDir :: dir_path
# ChimeraX :: IncludeDir :: dir_path
# ChimeraX :: LibraryDir :: dir_path

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
TOOLSHED_OUT_OF_DATE_BUNDLES : str
    Name of trigger fired when out-of-date bundles are detected.
    The trigger data is None.

Notes
-----
The term 'installed' refers to bundles whose corresponding Python
module or package is installed on the local machine.  The term
'available' refers to bundles that are listed on a remote server
but have not yet been installed on the local machine.

"""
import abc
from ..tasks import Task

# Toolshed trigger names
TOOLSHED_BUNDLE_INFO_ADDED = "bundle info added"
TOOLSHED_BUNDLE_INSTALLED = "bundle installed"
TOOLSHED_BUNDLE_UNINSTALLED = "bundle uninstalled"
TOOLSHED_BUNDLE_INFO_RELOADED = "bundle info reloaded"
TOOLSHED_OUT_OF_DATE_BUNDLES = "out of date bundles found"

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


def _debug(*args, file=None, flush=True, **kw):
    if _debug_toolshed:
        if file is None:
            import sys
            file = sys.__stderr__
        print("Toolshed:", *args, file=file, flush=flush, **kw)


# Package constants


# URL of remote toolshed
_DefaultRemoteURL = "https://cxtoolshed.rbvi.ucsf.edu"
# URL of experimental remote toolshed
_PreviewRemoteURL = "https://cxtoolshed-preview.rbvi.ucsf.edu"
# Default name for toolshed cache and data directories
_ToolshedFolder = "toolshed"
# Defaults names for installed ChimeraX bundles
_ChimeraXNamespace = "chimerax"


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


class ToolshedInitializationError(ToolshedError):
    """Initialization error.

    This exception derives from ToolshedError and is usually
    raised when doing manager, provider or custom initialization."""


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
                 remote_url=None, check_available=True, session=None):
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
            False to ignore remote server
        remote_url : str
            URL of the remote toolshed server.
            If set to None, a default URL is used.
        """
        # Initialize with defaults
        _debug("__init__", rebuild_cache, check_remote, remote_url)
        if remote_url is None:
            self.remote_url = _DefaultRemoteURL
        else:
            self.remote_url = remote_url
        self._safe_mode = None
        self._repo_locator = None
        self._installed_bundle_info = None
        self._available_bundle_info = None
        self._installed_packages = {}   # cache mapping packages to bundles
        # map from manager name to manager instance
        from weakref import WeakValueDictionary
        self._manager_instances = WeakValueDictionary()

        # Compute base directories
        import os
        from chimerax import app_dirs
        if os.path.exists(app_dirs.user_cache_dir):
            self._cache_dir = os.path.join(app_dirs.user_cache_dir, _ToolshedFolder)
        else:
            self._cache_dir = None
        _debug("cache dir: %s" % self._cache_dir)
        # TODO: unused so far
        # self._data_dir = os.path.join(app_dirs.user_data_dir, _ToolshedFolder)
        # _debug("data dir: %s" % self._data_dir)

        # Insert directories to sys.path to take precedence over
        # installed distribution.  addsitedir checks and does not
        # add the directory a second time.
        import site
        import os
        import sys
        self._site_dir = site.USER_SITE
        _debug("site dir: %s" % self._site_dir)
        os.makedirs(self._site_dir, exist_ok=True)
        sys.path.insert(0, self._site_dir)
        site.addsitedir(self._site_dir)

        # Create triggers
        from .. import triggerset
        self.triggers = triggerset.TriggerSet()
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INFO_ADDED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INSTALLED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_UNINSTALLED)
        self.triggers.add_trigger(TOOLSHED_BUNDLE_INFO_RELOADED)
        self.triggers.add_trigger(TOOLSHED_OUT_OF_DATE_BUNDLES)
        self.triggers.add_trigger("selector registered")
        self.triggers.add_trigger("selector deregistered")

        # Variables for updating list of available bundles
        from threading import RLock
        self._abc_lock = RLock()
        self._abc_updating = False

        # Reload the bundle info list
        _debug("loading bundles")
        try:
            self.init_available_from_cache(logger)
        except Exception:
            logger.report_exception("Error preloading available bundles")
        if (check_available and self._available_bundle_info is not None and
                                self._available_bundle_info.toolshed_url != self.remote_url):
            self._available_bundle_info = None
        self.reload(logger, check_remote=check_remote, rebuild_cache=rebuild_cache, _session=session)
        from datetime import datetime
        from ..core_settings import settings
        now = datetime.now()
        if check_available and not check_remote:
            # Did not check for available bundles synchronously
            # so start a thread and do it asynchronously if necessary
            from . import available
            if not available.has_cache_file(self._cache_dir):
                need_check = True
            else:
                need_check = need_to_check(
                    settings.toolshed_last_check, settings.toolshed_update_interval, now)
            if need_check:
                if session is None or not session.ui.is_gui:
                    self.async_reload_available(logger)
                else:
                    def delayed_available(trigger_name, data, toolshed=self, logger=logger):
                        toolshed.async_reload_available(logger)
                    session.ui.triggers.add_handler('ready', delayed_available)
                settings.toolshed_last_check = now.isoformat()
                _debug("Initiated toolshed check: %s" %
                       settings.toolshed_last_check)
        if check_available:
            need_check = need_to_check(
                settings.newer_last_check, settings.newer_update_interval, now)
            # need_check = True  # DEBUG
            if session and need_check:
                if not session.ui.is_gui or session.ui.main_window:
                    NewerVersionQuery(session)
                else:
                    session.ui.triggers.add_handler(
                        'ready', lambda *args, sesssion=session: NewerVersionQuery(session))
                settings.newer_last_check = now.isoformat()
        _debug("finished loading bundles")

    def reload(self, logger, *, session=None, reread_cache=True, rebuild_cache=False,
               check_remote=False, report=False, _session=None):
        """Supported API. Discard and reread bundle info.

        Parameters
        ----------
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            A logging object where warning and error messages are sent.
        rebuild_cache : boolean
            True to ignore local cache of installed bundle information and
            rebuild it by scanning Python directories; False otherwise.
        check_remote : boolean
            True to check remote server for updated information;
            False to ignore remote server
        """

        _debug("reload", rebuild_cache, check_remote)
        changes = {}
        if reread_cache or rebuild_cache:
            from .installed import InstalledBundleCache
            save = self._installed_bundle_info
            self._installed_bundle_info = InstalledBundleCache()
            cache_file = self._bundle_cache(False, logger)
            self._installed_bundle_info.load(logger, cache_file=cache_file,
                                             rebuild_cache=rebuild_cache,
                                             write_cache=cache_file is not None)
            if report:
                if save is None:
                    logger.info("Initial installed bundles.")
                else:
                    from .installed import _report_difference
                    changes = _report_difference(logger, save,
                                                 self._installed_bundle_info)
            if save is not None:
                save.deregister_all(logger, session, self._installed_packages)
            self._installed_bundle_info.register_all(logger, session,
                                                     self._installed_packages)
        if check_remote:
            self.reload_available(logger, _session=_session)
        self.triggers.activate_trigger(TOOLSHED_BUNDLE_INFO_RELOADED, self)
        return changes

    def async_reload_available(self, logger):
        with self._abc_lock:
            self._abc_updating = True
        from threading import Thread
        t = Thread(target=self.reload_available, args=(logger,),
                   name="Update list of available bundles")
        t.start()

    def reload_available(self, logger, _session=None):
        from urllib.error import URLError
        from .available import AvailableBundleCache
        abc = AvailableBundleCache(self._cache_dir)
        try:
            abc.load(logger, self.remote_url)
        except URLError as e:
            if _session is not None and _session.ui.is_gui:
                logger.info("Updating list of available bundles failed: %s"
                            % str(e.reason))
            with self._abc_lock:
                self._abc_updating = False
        except Exception as e:
            if _session is not None and _session.ui.is_gui:
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
        # check if there are newer version of installed bundles
        from packaging.version import Version
        has_out_of_date = False
        installed_name = None
        installed_version = None
        for available in abc:
            if available.name != installed_name:
                bi = self.find_bundle(available.name, logger)
                if bi is None:
                    installed_version = None
                    continue
                installed_name = available.name
                installed_version = Version(bi.version)
            elif installed_version is None:
                continue
            new_version = Version(available.version)
            if new_version > installed_version:
                has_out_of_date = True
                break
        if has_out_of_date:
            if _session is None:
                self.triggers.activate_trigger(TOOLSHED_OUT_OF_DATE_BUNDLES, self)
            else:
                # too early for trigger handler to be registered
                if _session.ui.is_gui:
                    def when_ready(trigger_name, data, toolshed=self):
                        toolshed.triggers.activate_trigger(TOOLSHED_OUT_OF_DATE_BUNDLES, toolshed)
                    _session.ui.triggers.add_handler('ready', when_ready)

    def init_available_from_cache(self, logger):
        from .available import AvailableBundleCache
        abc = AvailableBundleCache(self._cache_dir)
        try:
            abc.load_from_cache()
        except FileNotFoundError:
            logger.info("available bundle cache has not been initialized yet")
        else:
            self._available_bundle_info = abc

    def register_available_commands(self, logger):
        from sortedcontainers import SortedDict
        available = SortedDict()
        for bi in self._get_available_bundles(logger):
            # bi.register_available_commands(logger)
            for ci in bi.commands:
                a = available.get(ci.name, None)
                if a is None:
                    available[ci.name] = (set([(bi.name, bi.version)]), ci.synopsis)
                else:
                    bundles, synopsis = available[ci.name]
                    b = (bi.name, bi.version)
                    bundles.add(b)
                    # TODO: update synopsis if newer version of bundle
        from chimerax.core.commands import cli, CmdDesc, WholeRestOfLine
        for name in available:
            bundles, synopsis = available[name]
            cd = CmdDesc(
                optional=[('unknown_arguments', WholeRestOfLine)],
                synopsis=synopsis)

            def cb(session, s=self, name=name, bundles=bundles, logger=logger, unknown_arguments=None):
                s._available_cmd(name, bundles, logger)
            try:
                cli.register_available(name, cd, function=cb, logger=logger)
            except Exception as e:
                logger.warning("Unable to register available command %s: %s" % (ci.name, str(e)))

    def _available_cmd(self, name, bundles, logger):
        from chimerax.core.commands import commas, plural_form
        bundle_names, bundle_refs = self._bundle_names_and_refs(bundles)
        log_msg = "<b>%s</b> is provided by the uninstalled %s %s" % (
           name, plural_form(bundle_refs, "bundle"),
           commas(bundle_refs, 'and')
        )
        logger.info(log_msg, is_html=True)
        # TODO: if not self.session.ui.is_gui:
        #     return
        status_msg = '"%s" is provided by the uninstalled %s %s' % (
           name, plural_form(bundle_names, "bundle"),
           commas(['"%s"' % b for b in bundle_names], 'and')
        )
        logger.status(status_msg)

    def _bundle_names_and_refs(self, bundles):
        from packaging.version import Version
        bundle_names = set()
        bundle_refs = []
        version_info = {}
        for name, version in bundles:
            versions = version_info.setdefault(name, [])
            versions.append(Version(version))

        for name in version_info:
            bname = name
            all_versions = version_info[name]
            all_versions.sort()
            if bname.startswith('ChimeraX-'):
                bname = bname[len('ChimeraX-'):]
            if bname in bundle_names:
                continue
            bundle_names.add(bname)
            if len(all_versions) == 1:
                versions = f' version {all_versions[0]}'
            else:
                versions = f' versions {all_versions[0]} &ndash; {all_versions[-1]}'
            # TODO: what are the app store rules for toolshed names?
            toolshed_name = name.casefold().replace('-', '')
            ref = '<a href="https://cxtoolshed.rbvi.ucsf.edu/apps/%s">%s</a> %s' % (
                    toolshed_name, bname, versions
            )
            bundle_refs.append(ref)
        return bundle_names, bundle_refs

    def set_install_timestamp(self, per_user=False):
        _debug("set_install_timestamp")
        self._installed_bundle_info.set_install_timestamp(per_user=per_user)

    def bundle_url(self, bundle_name):
        app_name = bundle_name.casefold().replace('-', '').replace('_', '')
        return f"{self.remote_url}/apps/{app_name}"

    def bundle_link(self, bundle_name, if_available=True):
        from html import escape
        if bundle_name.startswith("ChimeraX-"):
            short_name = bundle_name[len("ChimeraX-"):]
        else:
            short_name = bundle_name
        if self._available_bundle_info is None or not self._available_bundle_info.find_by_name(bundle_name):
            # not available, so link would not work
            return escape(short_name)
        return f'<a href="{self.bundle_url(bundle_name)}">{escape(short_name)}</a>'

    def bundle_info(self, logger, installed=True, available=False):
        """Supported API. Return list of bundle info.

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

    def find_bundle(self, name, logger, installed=True, version=None):
        """Supported API. Return a :py:class:`BundleInfo` instance with the given name.

        Parameters
        ----------
        name : str
            Name (internal or display name) of the bundle of interest.
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Logging object where warning and error messages are sent.
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
        from pkg_resources import parse_version
        # put the below kludge in to allow sessions saved before some
        # bundles got renamed to restore
        name = {
            "ChimeraX-Atom-Search": "ChimeraX-AtomSearch",
            "ChimeraX-Bug-Reporter": "ChimeraX-BugReporter",
            "ChimeraX-Cage-Builder": "ChimeraX-CageBuilder",
            "ChimeraX-Connect-Structure": "ChimeraX-ConnectStructure",
            "ChimeraX-Dist-Monitor": "ChimeraX-DistMonitor",
            "ChimeraX-Dist-UI": "ChimeraX-DistUI",
            "ChimeraX-List-Info": "ChimeraX-ListInfo",
            "ChimeraX-MD-crds": "ChimeraX-MDcrds",
            "ChimeraX-Preset-Mgr": "ChimeraX-PresetMgr",
            "ChimeraX-Read-Pbonds": "ChimeraX-ReadPbonds",
            "ChimeraX-Rotamer-Lib-Mgr": "ChimeraX-RotamerLibMgr",
            "ChimeraX-Rotamer-Libs-Dunbrack": "ChimeraX-RotamerLibsDunbrack",
            "ChimeraX-Rotamer-Libs-Dynameomics": "ChimeraX-RotamerLibsDynameomics",
            "ChimeraX-Rotamer-Libs-Richardson": "ChimeraX-RotamerLibsRichardson",
            "ChimeraX-Scheme-Mgr": "ChimeraX-SchemeMgr",
            "ChimeraX-SEQ-VIEW": "ChimeraX-SeqView",
            "ChimeraX-Std-Commands": "ChimeraX-StdCommands",
        }.get(name, name)
        lc_name = name.casefold().replace('_', '-')
        lc_names = [lc_name]
        if not lc_name.startswith("chimerax-"):
            lc_names.append("chimerax-" + lc_name)
        best_bi = None
        best_version = None
        for bi in container:
            if bi.name.casefold() not in lc_names:
                for name in bi.supersedes:
                    if name.casefold() in lc_names:
                        break
                else:
                    continue
            if version == bi.version:
                return bi
            if version is None:
                if best_bi is None:
                    best_bi = bi
                    best_version = parse_version(bi.version)
                elif best_bi.name != bi.name:
                    logger.warning("%r matches multiple bundles %s, %s" % (name, best_bi.name, bi.name))
                    return None
                else:
                    v = parse_version(bi.version)
                    if v > best_version:
                        best_bi = bi
                        best_version = v
        return best_bi

    def find_bundle_for_tool(self, name, prefix_okay=False):
        """Supported API. Find named tool and its bundle

        Return the bundle it is in and its true name.

        Parameters
        ----------
        name : str
            Name or prefix of the tool of interest.
        prefix_okay : boolean
            Whether name only needs to be a prefix of a tool name
            or must be an exact match.
        """
        lc_name = name.casefold()
        tools = []
        for bi in self._installed_bundle_info:
            for tool in bi.tools:
                tname = tool.name.casefold()
                if tname == lc_name or (prefix_okay and tname.startswith(lc_name)):
                    tools.append((bi, tool.name))
        return tools

    def find_bundle_for_command(self, cmd):
        """Supported API. Find bundle registering given command

        `cmd` must be the full command name, not an abbreviation."""
        for bi in self._installed_bundle_info:
            for ci in bi.commands:
                if ci.name == cmd:
                    return bi
        return None

    def find_bundle_for_class(self, cls):
        """Supported API. Find bundle that has given class"""

        package = tuple(cls.__module__.split('.'))
        while package:
            try:
                return self._installed_packages[package]
            except KeyError:
                pass
            package = package[0:-1]
        return None

    def bootstrap_bundles(self, session, safe_mode):
        """Supported API. Do custom initialization for installed bundles

        After adding the :py:class:`Toolshed` singleton to a session,
        allow bundles need to install themselves into the session,
        (For symmetry, there should be a way to uninstall all bundles
        before a session is discarded, but we don't do that yet.)
        """
        _debug("initialize_bundles", safe_mode)
        self._safe_mode = safe_mode
        if safe_mode:
            return
        for bi in self._installed_bundle_info:
            bi.update_library_path()    # for bundles with dynamic libraries
        failed = self._init_managers(session)
        failed += self._init_custom(session)
        bad_packages = set()
        for bi in failed:
            session.logger.error("Bundle %r custom initialization failed" %
                                 bi.name)
            try:
                self._installed_bundle_info.remove(bi)
            except ValueError:
                pass
            bad_packages.update(bi.packages)
        for pkg in bad_packages:
            del self._installed_packages[pkg]

    def _init_custom(self, session):
        failed = []
        done = set()
        initializing = set()
        for bi in self._installed_bundle_info:
            self._init_bundle_custom(session, bi, done, initializing, failed)
        return failed

    def _init_bundle_custom(self, session, bi, done, initializing, failed):
        if not bi.custom_init or bi in done:
            return
        try:
            init_after = bi.inits["custom"]
        except KeyError:
            pass
        else:
            initializing.add(bi)
            for bundle_name in init_after:
                dbi = self.find_bundle(bundle_name, None)
                if dbi:
                    if dbi in initializing:
                        raise ToolshedInitializationError("circular dependency in bundle custom initialization")
                    self._init_bundle_custom(session, dbi, done, initializing, failed)
            initializing.remove(bi)
        try:
            _debug("custom initialization for bundle %r" % bi.name)
            bi.initialize(session)
        except ToolshedError:
            failed.append(bi)
        done.add(bi)

    def _init_managers(self, session):
        failed = []
        done = set()
        initializing = set()
        for bi in self._installed_bundle_info:
            self._init_bundle_manager(session, bi, done, initializing, failed)
        return failed

    def _init_bundle_manager(self, session, bi, done, initializing, failed):
        if not bi.managers or bi in done:
            return
        try:
            init_after = bi.inits["manager"]
        except KeyError:
            pass
        else:
            initializing.add(bi)
            for bundle_name in init_after:
                dbi = self.find_bundle(bundle_name, None)
                if dbi:
                    if dbi in initializing:
                        raise ToolshedInitializationError("circular dependency in bundle manager initialization")
                    self._init_bundle_manager(session, dbi, done, initializing, failed)
            initializing.remove(bi)
        try:
            for mgr, kw in bi.managers.items():
                if not session.ui.is_gui and kw.pop("guiOnly", "false") == "true":
                    _debug("skip non-GUI manager %s for bundle %r" % (mgr, bi.name))
                    continue
                if kw.pop("autostart", "true") == "false":
                    _debug("skip non-autostart manager %s for bundle %r" % (mgr, bi.name))
                    continue
                _debug("initialize manager %s for bundle %r" % (mgr, bi.name))
                bi.init_manager(session, mgr, **kw)
        except ToolshedError:
            failed.append(bi)
        done.add(bi)

    def _init_single_manager(self, mgr):
        if self._available_bundle_info:
            all_bundles = self._installed_bundle_info + self._available_bundle_info
        else:
            all_bundles = self._installed_bundle_info
        self._manager_instances[mgr.name] = mgr
        for pbi in all_bundles:
            for name, kw in pbi.providers.items():
                p_mgr, pvdr = name.split('/', 1)
                if p_mgr == mgr.name:
                    mgr.add_provider(pbi, pvdr, **kw)
        mgr.end_providers()

    def import_bundle(self, bundle_name, logger,
                      install="ask", session=None):
        """Supported API. Return the module for the bundle with the given name.

        Parameters
        ----------
        bundle_name : str
            Name (internal or display name) of the bundle of interest.
        logger : :py:class:`~chimerax.core.logger.Logger` instance
            Logging object where warning and error messages are sent.
        install: str
            Action to take if bundle is uninstalled but available.
            "ask" (default) means to ask user, if `session` is not `None`;
            "never" means not to install; and
            "always" means always install.
        session : :py:class:`chimerax.core.session.Session` instance.
            Session that is requesting the module.  Defaults to `None`.

        Raises
        ------
        ImportError
            Raised if a module for the bundle cannot be found.
        """
        from chimerax.toolshed_utils import _import_bundle
        _import_bundle(self, bundle_name, logger, install, session)

    def install_bundle(self, bundle, logger, *, per_user=True, reinstall=False, session=None, no_deps=False):
        """Supported API. Install the bundle by retrieving it from the remote shed.

        Parameters
        ----------
        bundle : string or :py:class:`BundleInfo` instance or sequence of them
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
        from chimerax.toolshed_utils import _install_bundle
        try:
            _install_bundle(self, bundle, logger, per_user=per_user, reinstall=reinstall, session=session, no_deps=no_deps)
        except ToolshedInstalledError as e:
            logger.error(str(e))

    def uninstall_bundle(self, bundle, logger, *, session=None, force_remove=False):
        """Supported API. Uninstall bundle by removing the corresponding Python distribution.

        Parameters
        ----------
        bundle : string or :py:class:`BundleInfo` instance or sequence of them
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
        from chimerax.toolshed_utils import _uninstall_bundle
        _uninstall_bundle(self, bundle, logger, session=session, force_remove=force_remove)

    #
    # End public API
    # All methods below are private
    #

    def _get_available_bundles(self, logger):
        with self._abc_lock:
            if self._available_bundle_info is None:
                if self._abc_updating:
                    logger.warning("still retrieving bundle list from toolshed")
                else:
                    pass  # Fix #1254 -- already warned during initialization
                    # logger.warning("could not retrieve bundle list from toolshed")
                from .available import AvailableBundleCache
                self._available_bundle_info = AvailableBundleCache(self._cache_dir)
                # TODO: trigger have available bundle information
            elif self._abc_updating:
                logger.warning("still updating bundle list from toolshed")
            return self._available_bundle_info

    def _bundle_cache(self, must_exist, logger):
        """Return path to bundle cache file.  None if not available."""
        _debug("_bundle_cache", must_exist)
        if self._cache_dir is None:
            return None
        if must_exist:
            import os
            os.makedirs(self._cache_dir, exist_ok=True)
        import os.path
        return os.path.join(self._cache_dir, "bundle_info.cache")


class ProviderManager(metaclass=abc.ABCMeta):
    """API for managers created by bundles"""

    def __init__(self, manager_name):
        self.name = manager_name
        ts = get_toolshed()
        ts._init_single_manager(self)

    @abc.abstractmethod
    def add_provider(self, bundle_info, provider_name, **kw):
        """Callback invoked to add provider to this manager.

        Parameters
        ----------
        session : :py:class:`chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.
        provider_name : str.
        """
        pass

    def end_providers(self):
        """Callback invoked after all providers have been added."""
        pass


class BundleAPI:
    """API for accessing bundles

    The metadata for the bundle indicates which of the methods need to be
    implemented.
    """

    api_version = 0

    @staticmethod
    def start_tool(*args):
        """Supported API. This method is called when the tool is invoked,
        typically from the application menu.
        Errors should be reported via exceptions.

        Parameters
        ----------
        session : :py:class:`chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.
        tool_info : :py:class:`ToolInfo` instance.

            Version 1 of the API passes in information for both
            the tool to be started and the bundle where it was defined.

        session : :py:class:`chimerax.core.session.Session` instance.
        tool_name : str.

            Version 0 of the API only passes in the name of
            the tool to be started.


        Returns
        -------
        :py:class:`~chimerax.core.tools.ToolInstance` instance
            The created tool.
        """
        raise NotImplementedError("BundleAPI.start_tool")

    @staticmethod
    def register_command(*args):
        """Supported API. When ChimeraX starts, it registers placeholders for
        commands from all bundles.  When a command from this
        bundle is actually used, ChimeraX calls this method to
        register the function that implements the command
        functionality, and then calls the command function.
        On subsequent uses of the command, ChimeraX will
        call the command function directly instead of calling
        this method. The API version for this method is defined
        by the :code:`api_version` class variable and defaults to 0.

        Parameters
        ----------
        bundle_info : :py:class:`BundleInfo` instance.
        command_info : :py:class:`CommandInfo` instance.
        logger : :py:class:`~chimerax.core.logger.Logger` instance.

            Version 1 of the API passes in information for both
            the command to be registered and the bundle where
            it was defined.

        command : str
        logger : :py:class:`~chimerax.core.logger.Logger` instance.

            Version 0 of the API only passes in the name of the
            command to be registered.
        """
        raise NotImplementedError("BundleAPI.register_command")

    @staticmethod
    def register_selector(*args):
        """Supported API. This method is called the first time when the selector is used.

        Parameters
        ----------
        bundle_info : :py:class:`BundleInfo` instance.
        selector_info : :py:class:`SelectorInfo` instance.
        logger : :py:class:`chimerax.core.logger.Logger` instance.

            Version 1 of the API passes in information about
            both the selector to be registered and the bundle
            where it is defined.

        selector_name : str
        logger : :py:class:`chimerax.core.logger.Logger` instance.

            Version 0 of the API only passes in the name of the
            selector to be registered.
        """
        raise NotImplementedError("BundleAPI.register_selector")

    @staticmethod
    def initialize(session, bundle_info):
        """Supported API. Called to initialize a bundle in a session.

        Must be defined if the ``custom_init`` metadata field is set to 'true'.
        ``initialize`` is called when the bundle is first loaded.
        To make ChimeraX start quickly, custom initialization is discouraged.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.
        """
        raise NotImplementedError("BundleAPI.initialize")

    @staticmethod
    def finish(session, bundle_info):
        """Supported API. Called to deinitialize a bundle in a session.

        Must be defined if the ``custom_init`` metadata field is set to 'true'.
        ``finish`` is called when the bundle is unloaded.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.
        """
        raise NotImplementedError("BundleAPI.finish")

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Supported API. Called to create a manager in a bundle at startup.

        Must be defined if there is a ``Manager`` tag in the bundle,
        unless that tag has an autostart="false" attribute, in which
        case the bundle is in charge of creating the manager as needed.
        ``init_manager`` is called when bundles are first loaded.
        It is the responsibility of ``init_manager`` to make the manager
        locatable, e.g., assign as an attribute of `session`.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        bundle_info : :py:class:`BundleInfo` instance.
        name : str.
            Name of manager to initialize.
        kw : keyword arguments.
            Keyword arguments listed in the bundle_info.xml.

        Returns
        -------
        :py:class:`ProviderManager` instance
            The created manager.
        """
        raise NotImplementedError("BundleAPI.init_manager")

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        """Supported API. Called to invoke a provider in a bundle.

        Must be defined if there is a ``Provider`` tag in the bundle.
        ``run_provider`` is called by the associated manager to perform
        the corresponding task.

        Parameters
        ----------
        session : :py:class:`~chimerax.core.session.Session` instance.
        name : str.
            Name of provider to initialize.
        mgr : str.
            Name of manager for this provider.
        kw : keyword arguments.
            Keyword arguments, if any, provided by the calling manager.
            Such keywords are specific to the manager and would be documented
            by the manager.
        """
        raise NotImplementedError("BundleAPI.run_provider")

    @staticmethod
    def get_class(name):
        """Supported API. Called to get named class from bundle.

        Used when restoring sessions.  Instances whose class can't be found via
        'get_class' can not be saved in sessions.  And those classes must implement
        the :py:class:`~chimerax.core.state.State` API.

        Parameters
        ----------
        name : str
            Name of class in bundle.
        """
        return None

    @staticmethod
    def include_dir(bundle_info):
        """Returns path to directory of C++ header files.

        Used to get directory path to C++ header files needed for
        compiling against libraries provided by the bundle.

        Parameters
        ----------
        bundle_info : :py:class:`BundleInfo` instance.

        Returns
        -------
        str or None

        """
        return None

    @staticmethod
    def library_dir(bundle_info):
        """Returns path to directory of compiled libraries.

        Used to get directory path to libraries (shared objects, DLLs)
        for linking against libraries provided by the bundle.

        Parameters
        ----------
        bundle_info : :py:class:`BundleInfo` instance.

        Returns
        -------
        str or None
        """
        return None

    @staticmethod
    def executable_dir(bundle_info):
        """Returns path to directory of compiled executables.

        Used to get directory path to executables at run-time.

        Parameters
        ----------
        bundle_info : :py:class:`BundleInfo` instance.

        Returns
        -------
        str or None
        """
        return None

    @staticmethod
    def data_dir(bundle_info):
        """Supported API. Returns path to directory of bundle-specific data.

        Used to get directory path to data included in the bundle.

        Parameters
        ----------
        bundle_info : :py:class:`BundleInfo` instance.

        Returns
        -------
        str or None
        """
        return None

    @property
    def _api_caller(self):
        try:
            return _CallBundleAPI[self.api_version]
        except KeyError:
            raise ToolshedError("bundle uses unsupport bundle API version %s" % self.api_version)


#
# _CallBundleAPI is used to call a bundle method with the
# correct arguments depending on the API version used by the
# bundle.  Note that get_class is not called via this mechanism.
# get_class() is more of a lookup than an invocation and the
# calling convention should not change.
#
class _CallBundleAPIv0:

    api_version = 0

    @classmethod
    def start_tool(cls, api, session, bi, ti):
        return cls._get_func(api, "start_tool")(session, ti.name)

    @classmethod
    def register_command(cls, api, bi, ci, logger):
        return cls._get_func(api, "register_command")(ci.name, logger)

    @classmethod
    def register_selector(cls, api, bi, si, logger):
        return cls._get_func(api, "register_selector")(si.name, logger)

    @classmethod
    def initialize(cls, api, session, bi):
        return cls._get_func(api, "initialize")(session, bi)

    @classmethod
    def init_manager(cls, api, session, bi, name, **kw):
        return cls._get_func(api, "init_manager")(session, bi, name, **kw)

    @classmethod
    def run_provider(cls, api, session, name, mgr, **kw):
        return cls._get_func(api, "run_provider")(session, name, mgr, **kw)

    @classmethod
    def finish(cls, api, session, bi):
        return cls._get_func(api, "finish")(session, bi)

    @classmethod
    def include_dir(cls, api, bi):
        return cls._get_func(api, "include_dir", default_okay=True)(bi)

    @classmethod
    def library_dir(cls, api, bi):
        return cls._get_func(api, "library_dir", default_okay=True)(bi)

    @classmethod
    def executable_dir(cls, api, bi):
        return cls._get_func(api, "executable_dir", default_okay=True)(bi)

    @classmethod
    def data_dir(cls, api, bi):
        return cls._get_func(api, "data_dir", default_okay=True)(bi)

    @classmethod
    def _get_func(cls, api, func_name, default_okay=False):
        try:
            f = getattr(api, func_name)
        except AttributeError:
            raise ToolshedError("bundle has no %s method" % func_name)
        if not default_okay and f is getattr(BundleAPI, func_name):
            raise ToolshedError("bundle forgot to override %s method" % func_name)
        return f


class _CallBundleAPIv1(_CallBundleAPIv0):

    api_version = 1

    @classmethod
    def start_tool(cls, api, session, bi, ti):
        return cls._get_func(api, "start_tool")(session, bi, ti)

    @classmethod
    def register_command(cls, api, bi, ci, logger):
        return cls._get_func(api, "register_command")(bi, ci, logger)

    @classmethod
    def register_selector(cls, api, bi, si, logger):
        return cls._get_func(api, "register_selector")(bi, si, logger)


_CallBundleAPI = {
    0: _CallBundleAPIv0,
    1: _CallBundleAPIv1,
}


# Import classes that developers might want to use
from .info import BundleInfo, CommandInfo, ToolInfo, SelectorInfo, FormatInfo


# Toolshed is a singleton.  Multiple calls to init returns the same instance.
_toolshed = None

_default_help_dirs = None


def init(*args, debug=None, **kw):
    """Supported API. Initialize toolshed.

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
    """
    if debug is not None:
        global _debug_toolshed
        _debug_toolshed = debug
    global _toolshed
    if _toolshed is None:
        _toolshed = Toolshed(*args, **kw)


def get_toolshed():
    """Supported API. Return current toolshed.

    Returns
    -------
    :py:class:`Toolshed` instance
        The toolshed singleton.

    The toolshed singleton will be None if py:func:`init` hasn't been called yet.
    """
    return _toolshed


def get_help_directories():
    global _default_help_dirs
    if _default_help_dirs is None:
        import chimerax
        if hasattr(chimerax, 'app_dirs'):
            from chimerax import app_data_dir, app_dirs
            import os
            _default_help_dirs = [
                os.path.join(app_dirs.user_cache_dir, 'docs'),  # for generated files
                os.path.join(app_data_dir, 'docs')              # for builtin files
            ]
        else:
            _default_help_dirs = []
    hd = _default_help_dirs[:]
    if _toolshed is not None:
        hd.extend(_toolshed._installed_bundle_info.help_directories)
    return hd


def default_toolshed_url():
    return _DefaultRemoteURL


def preview_toolshed_url():
    return _PreviewRemoteURL


def restart_action_info():
    import chimerax
    import os
    inst_dir = os.path.join(chimerax.app_dirs.user_cache_dir, "installers")
    restart_file = os.path.join(inst_dir, "on_restart")
    return inst_dir, restart_file


def _get_user():
    # robust version of getpass.getuser
    import os
    user = os.getenv("LOGNAME") or os.getenv("USER") or os.getenv("USERNAME")
    if user:
        return user
    import sys
    if sys.platform.startswith("win"):
        import win32api
        return win32api.GetUserName()
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except Exception:
        return f"uid-{os.getuid()}"


def chimerax_uuid():
    # Return anonymous unique string that represents
    # the current user for accessing ChimeraX toolshed
    import uuid
    node = uuid.getnode()   # Locality
    name = _get_user()
    dn = "CN=%s, L=%s" % (name, node)
    # and now make it anonymous
    # (uuid is based on the first 16 bytes of a 20 byte SHA1 hash)
    return uuid.uuid5(uuid.NAMESPACE_X500, dn)


def need_to_check(last_check, update_interval, now):
    if update_interval == "never":
        return False
    if not last_check:
        return True

    from datetime import datetime, timedelta
    last_check = datetime.strptime(last_check, "%Y-%m-%dT%H:%M:%S.%f")
    delta = now - last_check
    max_delta = timedelta(days=1)
    if update_interval == "week":
        max_delta = timedelta(days=7)
    elif update_interval == "day":
        max_delta = timedelta(days=1)
    elif update_interval == "month":
        max_delta = timedelta(days=30)
    return delta > max_delta


class NewerVersionQuery(Task):
    # asynchonously check for newer version of ChimeraX

    SERVICE_NAME = "chimerax/newer"
    # This is the default for Task, but just so there's
    # no ambiguity...
    SESSION_SAVE = False

    def __init__(self, session):
        super().__init__(session)
        import platform
        from .. import buildinfo
        from cxservices.api import default_api
        from . import chimerax_uuid
        self.api = default_api.DefaultApi()
        self.result = None
        system = platform.system()
        if system == "Darwin":
            system = "macosx"
            version = platform.mac_ver()[0]
        elif system == "Windows":
            system = "windows"
            version = platform.version()
        elif system == "Linux":
            import distro
            system = distro.id()
            like = distro.like()
            if like:
                system = f"{system} {like}"
            version = distro.version(best=True)
        params = {
            # use cxservices API names for keys
            "uuid": str(chimerax_uuid()),
            "os": system,
            "os_version": version,
            "chimera_x_version": buildinfo.version,
        }
        # params = {  # DEBUG
        #     # DEBUG DEBUG DEBUG
        #     "uuid": str(chimerax_uuid()),
        #     "os": "macosx",
        #     "os_version": "10.14",
        #     "chimera_x_version": "1.1",
        # }
        self.start(self.SERVICE_NAME, params, blocking=False)

    def run(self, service_name, params, blocking=False):
        self.result = self.api.check_for_updates(**params, async_req=not blocking)

    def on_finish(self):
        # If async_req is True, then need to call self.result.get()
        try:
            versions = self.result.get()
        except Exception:
            # Ignore problems getting results.  Might be a network error or
            # a server error.  It doesn't matter, just let ChimeraX run.
            return
        if not versions:
            return

        # don't bother user about releases they've choosen to ignore
        from ..core_settings import settings
        versions = [v for v in versions if v[0] not in settings.ignore_update]
        if not versions:
            return

        # notify user of newer versions
        from chimerax.core.commands import plural_form
        from .. import buildinfo
        # TODO: would like link to release notes and/or change log

        if not self.session.ui.is_gui:
            message = (
                "There is a newer version of UCSF ChimeraX available.  Downloads are available"
                "\nat the https://www.rbvi.ucsf.edu/chimerax/download.html."
            )
            self.session.logger.info(message)
            return

        from Qt.QtWidgets import QDialog

        class NewerDialog(QDialog):

            def __init__(self, parent):
                from Qt.QtWidgets import QDialogButtonBox, QGridLayout, QLabel, QStyle, QFrame, QCheckBox, QFrame
                from Qt.QtCore import Qt, QSize
                from Qt.QtGui import QPalette
                super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
                self.setWindowTitle("ChimeraX Update Available")
                self.setModal(False)
                self.setBackgroundRole(QPalette.Base)
                self.setAutoFillBackground(True)
                self.ignored = {}

                info = QLabel()
                icon = info.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
                info.setMinimumSize(64, 64)
                info.setPixmap(icon.pixmap(QSize(64, 64)))

                version = buildinfo.version
                # version = "1.1"  # DEBUG
                header = QLabel(
                    f"You are currently running UCSF ChimeraX {version}."
                    "<p>Click here to download a newer release for your system:"
                )
                footer = QLabel(
                    "You can get other releases from the "
                    "<a href='https://www.rbvi.ucsf.edu/chimerax/download.html'>"
                    "ChimeraX download page</a>."
                )
                footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
                footer.setOpenExternalLinks(True)

                updates = QFrame()
                # two columns: versioned link, remind
                layout = QGridLayout()
                updates.setLayout(layout)
                for row, (version, link) in enumerate(reversed(versions)):
                    html = f"&bull; <a href='{link}'>UCSF ChimeraX {version}</a>\n"
                    w = QLabel(html)
                    w.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
                    w.setOpenExternalLinks(True)
                    remind = QCheckBox("Remind me later")
                    remind.setToolTip("Show this release in future update checks")
                    remind.setChecked(True)
                    remind.stateChanged.connect(lambda state, version=version: self.remind_later(state, version))
                    layout.addWidget(w, row, 0)
                    layout.addWidget(remind, row, 1)

                hr = QFrame(self)
                hr.setFrameShape(QFrame.HLine)
                hr.setFixedHeight(1)
                hr.setForegroundRole(QPalette.Midlight)
                hr.setAutoFillBackground(True)

                bbox = QDialogButtonBox(self)
                bbox.setBackgroundRole(QPalette.Window)
                bbox.setAutoFillBackground(True)
                bbox.setStandardButtons(QDialogButtonBox.Close)
                bbox.accepted.connect(self.accept)
                bbox.rejected.connect(self.reject)

                layout = QGridLayout()
                self.setLayout(layout)
                layout.addWidget(info, 0, 0, 2, 1, Qt.AlignmentFlag.AlignTop)
                layout.addWidget(header, 0, 1)
                layout.addWidget(updates, 1, 1, Qt.AlignmentFlag.AlignLeft)
                layout.addWidget(footer, 2, 1)
                layout.addWidget(hr, 3, 0, 1, 2)
                layout.addWidget(bbox, 4, 0, 1, 2)

                # Need button box to be flush with edges of dialog, so move
                # layout's margins to widgets within the layout
                layout.setHorizontalSpacing(0)
                layout.setVerticalSpacing(0)
                margins = layout.getContentsMargins()  # left, top, right, bottom
                layout.setContentsMargins(0, 0, 0, 0)
                info.setContentsMargins(margins[0], 0, 0, 0)
                header.setContentsMargins(0, margins[1], margins[2], 0)
                updates.setContentsMargins(margins[0], 0, margins[2], 0)
                footer.setContentsMargins(0, 0, margins[2], margins[3])
                bbox.setContentsMargins(*margins)

            def remind_later(self, state, version=None):
                self.ignored[version] = not state

            def done(self, result):
                all_ignored = [version for version in self.ignored if self.ignored[version]]
                if all_ignored:
                    # don't use += or .extend() to guarantee that Settings.__setattr__
                    # will see that ignore_update has changed
                    ignore = settings.ignore_update + all_ignored
                    settings.ignore_update = ignore
                super().done(result)

        # keep reference to dialog because it's non-modal and would disappear otherwise
        d = NewerDialog(self.session.ui.main_window)
        self.newer_dialog = d
        d.show()
