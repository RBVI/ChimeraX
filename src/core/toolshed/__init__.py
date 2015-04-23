# vi: set expandtab ts=4 sw=4:

"""
The Toolshed provides an interface for finding installed
tool distributions as well as distributions available for
installation from a remote server.
The Toolshed can handle updating, installing and uninstalling
distributions while taking care of inter-distribution dependencies.

The Toolshed interface uses :py:mod:`distlib` heavily.
For example, `Distribution` instances from :py:mod:`distlib`
are tracked for both available and installed tools;
the :py:class:`distlib.locators.Locator` class is used for finding
an installed :py:class:`distlib.database.Distribution`.

Each Python distribution (Chimera uses :py:class:`distlib.wheel.Wheel`)
may contain multiple tools.
Metadata blocks in each distribution contain descriptions for tools.
Each tool is described by a 'Chimera-Tool' entry that consists of
seven fields separated by double colons (``::``).

1. ``Chimera-Tools`` : str constant
    Field identifying entry as tool metadata.
2. ``name`` : str
    Internal name of tool.  This must be unique across all tools.
3. ``module_name`` : str
    Name of module or package that implements the tool.
4. ``display_name`` : str
    Name of tool to display to users.
5. ``commands`` : str
    Comma-separated list of cli commands that the tool provides.
6. ``menu_categories`` : str
    Comma-separated list of menu categories in which the tool belong.
7. ``synopsis`` : str
    A short description of the tool.

Modules referenced in distribution metadata must define:

  ``register_command(command_name)``
    Called when delayed command line registration occurs.
    ``command_name`` is a string of the command to be registered.

  ``start_tool(session, ti, *args, **kw)``
    Called to create a tool instance.
    ``session`` is a :py:class:`~chimera.core.session.Session` instance for the current session.
    ``ti`` is a :py:class:`ToolInfo` instance for the tool to be started.

Attributes
----------
TOOLSHED_TOOL_INFO_ADDED : str
    Name of trigger fired when new tool metadata is registered.
    The trigger data is a :py:class:`ToolInfo` instance.
TOOLSHED_TOOL_INSTALLED : str
    Name of trigger fired when a new tool is installed.
    The trigger data is a :py:class:`ToolInfo` instance.
TOOLSHED_TOOL_UNINSTALLED : str
    Name of trigger fired when an installed tool is removed.
    The trigger data is a :py:class:`ToolInfo` instance.

.. note:
    The term 'installed' refers to tools whose corresponding Python
    module or package is installed on the local machine.  The term
    'available' refers to tools that are listed on a remote server
    but have not yet been installed on the local machine.

"""

# Toolshed trigger names
TOOLSHED_TOOL_INFO_ADDED = "tool info added"
TOOLSHED_TOOL_INSTALLED = "tool installed"
TOOLSHED_TOOL_UNINSTALLED = "tool uninstalled"


def _hack_distlib(f):
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
        v = f(*args, **kw)
        # Restore hacked name
        metadata.METADATA_FILENAME = save
        database.METADATA_FILENAME = save
        wheel.METADATA_FILENAME = save
        _debug("changing back METADATA_FILENAME", metadata.METADATA_FILENAME)
        return v
    return hacked_f


def _debug(*args, **kw):
    return


# Package constants


# Default URL of remote tool shed
_RemoteURL = "http://localhost:8080"
# Default name for toolshed cache and data directories
_Toolshed = "toolshed"
# Defaults names for installed chimera tools
_ChimeraBasePackage = "chimera"
_ChimeraCore = _ChimeraBasePackage + ".core"
_ChimeraToolboxPrefix = _ChimeraBasePackage + ".toolbox"


# Exceptions raised by Toolshed class


class ToolshedError(Exception):
    """Generic Toolshed error."""


class ToolshedUninstalledError(ToolshedError):
    """Uninstalled-tool error.
    
    This exception derives from :py:class:`ToolshedError` and is usually
    raised when trying to uninstall a tool that has not been installed."""


class ToolshedInstalledError(ToolshedError):
    """Tool-already-installed error.
    
    This exception derives from :py:class:`ToolshedError` and is usually
    raised when trying to install a tool that is already installed."""


class ToolshedUnavailableError(ToolshedError):
    """Tool-not-found error.
    
    This exception derives from ToolshedError and is usually
    raised when no Python distribution can be found for a tool."""


# Toolshed and ToolInfo are session-independent


class Toolshed:
    """Toolshed keeps track of the list of tool metadata, aka :py:class:`ToolInfo`.

    Tool metadata may be for "installed" tools, where their code
    is already downloaded from the remote server and installed
    locally, or "available" tools, where their code is not locally
    installed.
    
    Attributes
    ----------
    triggers : :py:class:`~chimera.core.triggerset.TriggerSet` instance
        Where to register handlers for toolshed triggers
    
    """

    def __init__(self, logger, appdirs,
                 rebuild_cache=False, check_remote=False, remote_url=None):
        """Initialize Toolshed instance.

        Parameters
        ----------
        logger : :py:class:`~chimera.core.logger.Logger` instance
            A logging object where warning and error messages are sent.
        appdirs : :py:class:`~chimera.core.appdirs.AppDirs` instance
            Location information about Chimera data and code directories.
        rebuild_cache : boolean
            True to ignore local cache of installed tool information and
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
        _debug("__init__", appdirs, rebuild_cache, check_remote, remote_url)
        if remote_url is None:
            self.remote_url = _RemoteURL
        else:
            self.remote_url = remote_url
        self._repo_locator = None
        self._inst_locator = None
        self._installed_tool_info = []
        self._available_tool_info = []
        self._all_installed_distributions = None

        # Compute base directories
        import os.path
        self._cache_dir = os.path.join(appdirs.user_cache_dir, _Toolshed)
        _debug("cache dir: %s" % self._cache_dir)
        self._data_dir = os.path.join(appdirs.user_data_dir, _Toolshed)
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
        self.triggers.add_trigger(TOOLSHED_TOOL_INFO_ADDED)
        self.triggers.add_trigger(TOOLSHED_TOOL_INSTALLED)
        self.triggers.add_trigger(TOOLSHED_TOOL_UNINSTALLED)

        # Reload the tool info list
        _debug("loading tools")
        self.reload(logger, check_remote=check_remote,
                    rebuild_cache=rebuild_cache)
        _debug("finished loading tools")

    def check_remote(self, logger):
        """Check remote shed for updated tool info.

        Parameters
        ----------
        logger : :py:class:`~chimera.core.logger.Logger` instance
            Logging object where warning and error messages are sent.
            
        Returns
        -------
        list of :py:class:`ToolInfo` instances
            List of tool metadata from remote server.
        
        """

        _debug("check_remote")
        if self._repo_locator is None:
            from .chimera_locator import ChimeraLocator
            self._repo_locator = ChimeraLocator(self.remote_url)
        distributions = self._repo_locator.get_distributions()
        ti_list = []
        for d in distributions:
            ti_list.extend(self._make_tool_info(d, False, logger))
            _debug("added remote distribution:", d)
        return ti_list

    def reload(self, logger, rebuild_cache=False, check_remote=False):
        """Discard and reread tool info.

        Parameters
        ----------
        logger : :py:class:`~chimera.core.logger.Logger` instance
            A logging object where warning and error messages are sent.
        rebuild_cache : boolean
            True to ignore local cache of installed tool information and
            rebuild it by scanning Python directories; False otherwise.
        check_remote : boolean
            True to check remote server for updated information;
            False to ignore remote server;
            None to use setting from user preferences.

        """

        _debug("reload", rebuild_cache, check_remote)
        for ti in self._installed_tool_info:
            ti.deregister_commands()
        self._installed_tool_info = []
        inst_ti_list = self._load_tool_info(logger, rebuild_cache=rebuild_cache)
        for ti in inst_ti_list:
            self.add_tool_info(ti)
            ti.register_commands()
        if check_remote:
            self._available_tool_info = []
            self._repo_locator = None
            remote_ti_list = self.check_remote(logger)
            for ti in remote_ti_list:
                self.add_tool_info(ti)
                # XXX: do we want to register commands so that we can
                # ask user whether to install tool when invoked?

    def tool_info(self, installed=True, available=False):
        """Return list of tool info.

        Parameters
        ----------
        installed : boolean
            True to include installed tool metadata in return value;
            False otherwise
        available : boolean
            True to include available tool metadata in return value;
            False otherwise

        Returns
        -------
        list of :py:class:`ToolInfo` instances
            Combined list of all selected types of tool metadata.
        """

        _debug("tool_info", installed, available)
        if installed and available:
            return self._installed_tool_info + self._available_tool_info
        elif installed:
            return self._installed_tool_info
        elif available:
            return self._available_tool_info
        else:
            return []

    def add_tool_info(self, ti):
        """Add metadata for a tool.

        Parameters
        ----------
        ti : :py:class:`ToolInfo` instance
            Must be a constructed instance, i.e., not an existing instance
            returned by :py:func:`tool_info`.

        .. note:
            A :py:const:`TOOLSHED_TOOL_INFO_ADDED` trigger is fired after the addition.
        
        """
        _debug("add_tool_info", ti)
        if ti.installed:
            container = self._installed_tool_info
        else:
            container = self._available_tool_info
        container.append(ti)
        self.triggers.activate_trigger(TOOLSHED_TOOL_INFO_ADDED, ti)

    def install_tool(self, ti, logger, system=False):
        """Install the tool by retrieving it from the remote shed.

        Parameters
        ----------
        ti : :py:class:`ToolInfo` instance
            Should be from the available tool list.
        system : boolean
            False to install tool only for the current user (default);
            True to install for everyone.
        logger : :py:class:`~chimera.core.logger.Logger` instance
            Logging object where warning and error messages are sent.

        Raises
        ------
        ToolshedInstalledError
            Raised if the tool is already installed.

        .. note:
            A :py:const:`TOOLSHED_TOOL_INSTALLED` trigger is fired after installation.
        
        """
        _debug("install_tool", ti)
        if ti.installed:
            raise ToolshedInstalledError("tool \"%s\" already installed"
                                         % ti.name)
        self._install_tool(ti, system, logger)
        self._write_cache(self._installed_tool_info, logger)
        self.triggers.activate_trigger(TOOLSHED_TOOL_INSTALLED, ti)

    def uninstall_tool(self, ti, logger):
        """Uninstall tool by removing the corresponding Python distribution.

        Parameters
        ----------
        ti : :py:class:`ToolInfo` instance
            Should be from the installed tool list.
        logger : :py:class:`~chimera.core.logger.Logger` instance
            Logging object where warning and error messages are sent.

        Raises
        ------
        ToolshedInstalledError
            Raised if the tool is not installed.

        .. note:
            A :py:const:`TOOLSHED_TOOL_UNINSTALLED` trigger is fired after package removal.
        
        """
        _debug("uninstall_tool", ti)
        self._uninstall_tool(ti, logger)
        self._write_cache(self._installed_tool_info, logger)
        self.triggers.activate_trigger(TOOLSHED_TOOL_UNINSTALLED, ti)

    def find_tool(self, name, installed=True, version=None):
        """Return a tool with the given name.

        Parameters
        ----------
        name : str
            Name (internal or display name) of the tool of interest.
        installed : boolean
            True to check only for installed tools; False otherwise.
        version : str
            None to find any version; specific string to check for
            one particular version.
            
        """
        _debug("find_tool", name, installed, version)
        if installed:
            container = self._installed_tool_info
        else:
            container = self._available_tool_info
        from distlib.version import NormalizedVersion as Version
        best_ti = None
        best_version = None
        for ti in container:
            if ti.name != name and ti.display_name != name:
                continue
            if version == ti.version:
                return ti
            if version is None:
                if best_ti is None:
                    best_ti = ti
                    best_version = Version(ti.version)
                else:
                    v = Version(ti.version)
                    if v > best_version:
                        best_ti = ti
                        best_version = v
        return best_ti

    #
    # End public API
    # All methods below are private
    #

    def _load_tool_info(self, logger, rebuild_cache=False):
        # Load tool info.  If not rebuild_cache, try reading
        # it from a cache file.  If we cannot use the cache,
        # read the information from the data directory and
        # try to create the cache file.
        _debug("_load_tool_info", rebuild_cache)
        if not rebuild_cache:
            tool_info = self._read_cache()
            if tool_info is not None:
                return tool_info
        self._scan_installed(logger)
        tool_info = []
        for d in self._inst_tool_dists:
            tool_info.extend(self._make_tool_info(d, True, logger))
        # NOTE: need to do something with toolboxes
        self._write_cache(tool_info, logger)
        return tool_info

    @_hack_distlib
    def _scan_installed(self, logger):
        # Scan installed packages for Chimera tools

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
            self._inst_tool_dists = set()
            logger.warning("\"%s\" distribution not found" % _ChimeraCore)
            return

        # Partition packages into core, tools and toolboxes
        from distlib.database import make_graph
        dg = make_graph(all_distributions)
        known_dists = set([core])
        self._inst_chimera_core = core
        self._inst_core = set([core])
        self._inst_tool_dists = set()
        self._all_installed_distributions = {_ChimeraCore: core}
        for d, label in dg.adjacency_list[core]:
            self._inst_core.add(d)
            self._all_installed_distributions[d.name] = d
        check_list = [core]
        while check_list:
            dist = check_list.pop()
            _debug("checking", dist)
            for d in dg.reverse_list[dist]:
                if d in known_dists:
                    continue
                known_dists.add(d)
                check_list.append(d)
                self._inst_tool_dists.add(d)
                self._all_installed_distributions[d.name] = d

    def _tool_cache(self, must_exist):
        """Return path to tool cache file."""
        _debug("_tool_cache", must_exist)
        if must_exist:
            import os
            os.makedirs(self._cache_dir, exist_ok=True)
        import os.path
        return os.path.join(self._cache_dir, "tool_info.cache")

    def _read_cache(self):
        """Read installed tool information from cache file.

        Returns boolean on whether cache file was read."""
        _debug("_read_cache")
        cache_file = self._tool_cache(False)
        if not self._is_cache_current(cache_file):
            return None
        import shelve
        import dbm
        try:
            s = shelve.open(cache_file, "r")
        except dbm.error:
            return None
        try:
            tool_info = [ToolInfo(*args, **kw) for args, kw in s["tool_info"]]
        except:
            return None
        finally:
            s.close()
        return tool_info

    def _is_cache_current(self, cache_file):
        """Check if cache is up to date."""
        _debug("_is_cache_current")
        import sys
        import os.path
        try:
            sys_timestamp = os.path.getmtime(os.path.join(sys.prefix, "timestamp"))
            cache_timestamp = os.path.getmtime(cache_file + ".timestamp")
        except FileNotFoundError:
            return False
        # TODO: check against user timestamp as well
        return cache_timestamp > sys_timestamp

    def _write_cache(self, tool_info, logger):
        """Write current tool information to cache file."""
        _debug("_write_cache", tool_info)
        cache_file = self._tool_cache(True)
        import shelve
        try:
            s = shelve.open(cache_file)
        except IOError as e:
            logger.error("\"%s\": %s" % (cache_file, str(e)))
        else:
            try:
                s["tool_info"] = [ti.cache_data() for ti in tool_info]
            finally:
                s.close()
        timestamp_file = cache_file + ".timestamp"
        with open(timestamp_file, "w") as f:
            import time
            print(time.ctime(), file=f)

    def _make_tool_info(self, d, installed, logger):
        """Convert distribution into a list of :py:class:`ToolInfo` instances."""
        name = d.name
        version = d.version
        md = d.metadata

        tools = []
        for classifier in md.dictionary["classifiers"]:
            parts = [v.strip() for v in classifier.split("::")]
            if parts[0] != "Chimera-Tool":
                continue
            if len(parts) != 7:
                logger.warning("Malformed Chimera-Tool line in %s skipped." % name)
                logger.warning("Expected 7 fields and got %d." % len(parts))
                continue
            kw = {"distribution_name": name, "distribution_version": version}
            # Name of tool
            tool_name = parts[1]
            # Name of module implementing tool
            kw["module_name"] = parts[2]
            # Display name of tool
            kw["display_name"] = parts[3]
            # CLI command names (just the first word)
            commands = parts[4]
            if commands:
                kw["command_names"] = [v.strip()
                                       for v in commands.split(',')]
            # Menu categories in which tool should appear
            categories = parts[5]
            if categories:
                kw["menu_categories"] = [v.strip()
                                         for v in categories.split(',')]
            # Synopsis of tool
            kw["synopsis"] = parts[6]
            tools.append(ToolInfo(tool_name, installed, **kw))
        return tools

    # Following methods are used for installing and removing
    # distributions

    def _install_tool(self, tool_info, system, logger):
        # Install a tool.  This entails:
        #  - finding all distributions that this one depends on
        #  - making sure things will be compatible if installed
        #  - installing all the distributions
        #  - updating any tool installation status
        _debug("_install_tool")
        want_update = []
        need_update = []
        self._install_dist_tool(tool_info, want_update, logger)
        self._install_cascade(want_update, need_update, logger)
        incompatible = self._install_check_incompatible(need_update, logger)
        if incompatible:
            return
        self._install_wheels(need_update, system, logger)
        # update tool installation status
        updated = set([d.name for d in need_update])
        keep = [ti for ti in self._installed_tool_info
                if ti._distribution_name not in updated]
        self._installed_tool_info = keep
        updated = set([(d.name, d.version) for d in need_update])
        if self._all_installed_distributions is not None:
            self._inst_path = None
            self._inst_locator = None
            self._all_installed_distributions = None
        import copy
        newly_installed = [copy.copy(ti) for ti in self._available_tool_info
                           if ti.distribution() in updated]
        for ti in newly_installed:
            ti.installed = True
            self.add_tool_info(ti)
            ti.register_commands()

    def _install_dist_core(self, want, logger):
        # Add Chimera core distribution to update list
        _debug("_install_dist_core")
        d = self._install_distribution(_ChimeraCore, None, logger)
        if d:
            want.append(d)

    def _install_dist_tool(self, tool_info, want, logger):
        # Add the distribution that provides the
        # given tool to update list
        _debug("_install_dist_tool", tool_info)
        if tool_info._distribution_name is None:
            raise ToolshedUnavailableError("no distribution information "
                                           "available for tool \"%s\""
                                           % tool_info.name)
        d = self._install_distribution(tool_info._distribution_name,
                                       tool_info._distribution_version, logger)
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
            for d, req_list in graph.missing.items():
                if len(req_list) == 1:
                    s = repr(req_list[0])
                else:
                    s = " and ".join(", ".join([repr(r) for r in req_list[:-1]]),
                                     repr(req_list[-1]))
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
        ordered = []    # ordered by least dependency
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
                import time, calendar
                d_mtime = calendar.timegm(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
                c_mtime = os.path.getmtime(dloc)
                print("distribution", time.ctime(d_mtime))
                print("cache", time.ctime(c_mtime))
                need_fetch = (d_mtime > c_mtime)
            if need_fetch:
                print("fetching wheel")
                try:
                    fn, headers = urlretrieve(url, dloc)
                except URLError as e:
                    logger.warning("cannot fetch %s: %s" % (url, str(e)))
                    continue
            else:
                print("using cached wheel")
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

    def _uninstall_tool(self, tool_info, logger):
        _debug("_uninstall", tool_info)
        dv = tool_info.distribution()
        name, version = dv
        all = self._get_all_installed_distributions(logger)
        d = all[name]
        if d.version != version:
            raise KeyError("distribution \"%s %s\" does not match tool version "
                           "\"%s\"" % (name, version, d.version))
        keep = []
        for ti in self._installed_tool_info:
            if ti.distribution() != dv:
                keep.append(ti)
            else:
                ti.deregister_commands()
        self._installed_tool_info = keep
        self._remove_distribution(d, logger)

    # End methods for installing and removing distributions


class ToolInfo:
    """Metadata about a tool, whether installed or available.

    A :py:class:`ToolInfo` instance stores the properties about a tool and
    can create a tool instance.

    Attributes
    ----------
    command_names : list of str
        List of cli command name registered for this tool.
    display_name : str
        The tool name to display in user interfaces.
    installed : boolean
        True if this tool is installed locally; False otherwise.
    menu_categories : list of str
        List of categories in which this tool belong.
    name : str
        The internal name of the tool.
    synopsis : readonly str
        Short description of this tool.
    version : readonly str
        Tool version (which is actually the same as the distribution version,
        so all tools from the same distribution share the same version).
    
    """

    def __init__(self, name, installed,
                 distribution_name=None,
                 distribution_version=None,
                 display_name=None,
                 module_name=None,
                 synopsis=None,
                 menu_categories=(),
                 command_names=()):
        """Initialize instance.

        Parameters
        ----------
        name : str
            Internal name for tool.
        installed : boolean
            Whether this tool is locally installed.
        display_name : str
            Tool nname to display in user interface.
        distribution_name : str
            Name of Python distribution that provided this tool.
        distribution_version : str
            Version of Python distribution that provided this tool.
        module_name : str
            Name of module implementing this tool.  Must be a dotted Python name.
        menu_categories : list of str
            List of menu categories in which this tool belong.
        command_names : list of str
            List of names of cli commands to register for this tool.

        """
        # Public attributes
        self.name = name
        self.installed = installed
        self.display_name = display_name or name
        self.menu_categories = menu_categories
        self.command_names = command_names

        # Private attributes
        self._distribution_name = distribution_name
        self._distribution_version = distribution_version
        self._module_name = module_name
        self._synopsis = synopsis

    @property
    def version(self):
        return self._distribution_version

    @property
    def synopsis(self):
        return self._synopsis or "no synopsis available"

    def __repr__(self):
        s = self.display_name
        if self.installed:
            s += " (installed)"
        else:
            s += " (available)"
        s += " [name: %s]" % self.name
        s += " [distribution: %s %s]" % (self._distribution_name,
                                         self._distribution_version)
        s += " [module: %s]" % self._module_name
        if self.menu_categories:
            s += " [category: %s]" % ','.join(self.menu_categories)
        if self.command_names:
            s += " [command line: %s]" % ','.join(self.command_names)
        return s

    def cache_data(self):
        """Return state data that can be used to recreate the instance.

        Returns
        -------
        2-tuple of (list, dict)
            List and dictionary suitable for passing to :py:class:`ToolInfo`.
        
        """
        args = (self.name, self.installed)
        kw = {
            "display_name": self.display_name,
            "menu_categories": self.menu_categories,
            "command_names": self.command_names,
            "synopsis": self._synopsis,
            "distribution_name": self._distribution_name,
            "distribution_version": self._distribution_version,
            "module_name": self._module_name,
        }
        return args, kw

    def distribution(self):
        """Return distribution information.
        
        Returns
        -------
        2-tuple of (str, str).
            Distribution name and version.

        """
        return self._distribution_name, self._distribution_version

    def register_commands(self):
        """Register commands with cli."""
        from chimera.core import cli
        for command_name in self.command_names:
            def cb(s=self, n=command_name):
                s._register_cmd(n)
            _debug("delay_registration", command_name)
            cli.delay_registration(command_name, cb)

    def _register_cmd(self, command_name):
        """Called when commands need to be really registered."""
        self._get_module().register_command(command_name)

    def deregister_commands(self):
        """Deregister commands with cli."""
        from chimera.core import cli
        for command_name in self.command_names:
            _debug("deregister_command", command_name)
            cli.deregister(command_name)

    def _get_module(self):
        """Return module for this tool."""
        if not self._module_name:
            raise ToolshedError("no module specified for tool \"%s\"" % self.name)
        import importlib
        m = importlib.import_module(self._module_name)
        _debug("_get_module", self._module_name, m)
        return m

    def start(self, session, *args, **kw):
        """Create and return a tool instance.

        Parameters
        ----------
        session : :py:class:`~chimera.core.session.Session` instance
            The session in which the tool will run.
        args : any
            Positional arguments to pass to tool instance initializer.
        kw : any
            Keyword arguments to pass to tool instance initializer.

        Returns
        -------
        :py:class:`~chimera.core.tools.ToolInstance` instance
            The registered running tool instance.

        Raises
        ------
        ToolshedUninstalledError
            If the tool is not installed.
        ToolshedError
            If the tool cannot be started.

        """
        if not self.installed:
            raise ToolshedUninstalledError("tool \"%s\" is not installed"
                                           % self.name)
        try:
            f = self._get_module().start_tool
        except (ImportError, AttributeError, TypeError, SyntaxError):
            raise ToolshedError("bad start callable specified for tool \"%s\""
                                % self.name)
        else:
            f(session, self, *args, **kw)

    def newer_than(self, ti):
        """Return whether this :py:class:`ToolInfo` instance is newer than given one

        Parameters
        ----------
        ti : :py:class:`ToolInfo` instance
            The instance to compare against

        Returns
        -------
        Boolean
            True if this instance is newer; False if 'ti' is newer.

        """
        from distlib.version import NormalizedVersion as Version
        return Version(self.version) > Version(ti.version)


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
