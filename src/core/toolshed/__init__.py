# vim: set expandtab ts=4 sw=4:

"""shed - Chimera 2 Tool Shed

The Tool Shed provides an interface for querying available and
out-of-date packages, and for updating, installing and uninstalling
packages while handling inter-package dependencies.

The '''shed''' interface uses '''distlib''' heavily.
For example, '''Distribution''' instances from '''distlib'''
are tracked for both available and installed packages; the
'''distlib''' '''Locator''' class is used for finding
'''Distribution'''s.

Distribution metadata blocks contain descriptions for tools:
  '''Chimera-Tools''' is a list of name of tools (preferably single words)
      Example: '''Chimera-Tools: MAVOpen MAV'''
  ``Tool-DisplayName`` is the name of a tool for display to user
      Example: '''MAVOpen-Name: 'Open MultAlign Viewer''''
  ``Tool-MenuCategories`` is a list of categories where tool is applicable
      Example: '''MAVOpen-MenuCategories: Sequence'''
  ``Tool-Commands`` is a list of CLI command names provided
      Example: '''MAVOpen-Commands: mav'''

Modules referenced in distribution metadata must define:
  '''register_command(command_name)'''
      Called when delayed command line registration occurs.
  '''start_tool(session, ti, *args, **kw) '''
      Called to create a tool instance.
    ``session`` is a core.Session instance.
    ``ti`` is a toolshed.ToolInfo instance.
"""


def _hack_distlib(f):
    def hacked_f(*args, **kw):
        # distlib and wheel packages disagree on the name for
        # the metadata file in wheels.  (wheel uses PEP345 while
        # distlib uses PEP427.)  distlib is backwards compatible,
        # so we hack the file name when we get distributions.
        from distlib import metadata
        save = metadata.METADATA_FILENAME
        metadata.METADATA_FILENAME = "metadata.json"
        v = f(*args, **kw)
        # Restore hacked name
        metadata.METADATA_FILENAME = save
        return v
    return hacked_f


def _debug(*args, **kw):
    return


# Package constants


# Default URL of remote tool shed
_RemoteURL = "http://localhost:8080"
# Default name for toolshed cache and data directories
_ToolShed = "toolshed"
# Defaults names for installed chimera tools
_ChimeraBasePackage = "chimera"
_ChimeraCore = _ChimeraBasePackage + ".core"
_ChimeraToolboxPrefix = _ChimeraBasePackage + ".toolbox"


# Exceptions raised by ToolShed class


class ToolShedError(Exception):
    """Generic ToolShed error."""


class ToolShedUninstalledError(ToolShedError):
    """Uninstalled-tool error."""


class ToolShedInstalledError(ToolShedError):
    """Tool-already-installed error."""


class ToolShedUnavailableError(ToolShedError):
    """Tool-not-found error."""


# ToolShed and ToolInfo are session-independent


class ToolShed:
    """ToolShed keeps track of the list of tool info.

    Tool info may be "installed", where their code
    is already downloaded from the remote shed and installed
    locally, or "available", where their code is not locally
    installed."""

    def __init__(self, logger, appdirs,
                 rebuild_cache=False, check_remote=False, remote_url=None):
        """Initialize shed using data from 'appdirs'.

        - ``logger`` is a logging object where warning and
          error messages are sent.
        - ``appdirs`` is an instance of '''appdirs.AppDirs'''
          containing location information about Chimera data
          and code directories.
        - ``rebuild_cache`` is a boolean indicating whether
          to ignore the local cache of installed tool
          information and rebuild it by scanning Python
          packages.
        - ``check_remote`` is a boolean indicating whether
          to check remote repository for updated information.
          If '''True''', the remote shed is queried;
          if '''False''', the check is not done;
          if '''None''', the check is done according to
          setting in '''preferences'''.
        - ``remote_url`` is a string with the URL of
          the remote toolshed.  If '''None''', a default
          URL is be used."""

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
        self._cache_dir = os.path.join(appdirs.user_cache_dir, _ToolShed)
        _debug("cache dir: %s" % self._cache_dir)
        self._data_dir = os.path.join(appdirs.user_data_dir, _ToolShed)
        _debug("data dir: %s" % self._data_dir)

        # Add directories to sys.path
        import os.path
        self._site_dir = os.path.join(self._data_dir, "site-packages")
        _debug("site dir: %s" % self._site_dir)
        import os
        os.makedirs(self._site_dir, exist_ok=True)
        import site
        site.addsitedir(self._site_dir)

        # Reload the tool info list
        _debug("loading tools")
        self.reload(logger, check_remote=check_remote,
                    rebuild_cache=rebuild_cache)
        _debug("finished loading tools")

    def check_remote(self, logger):
        """Check remote shed for updated tool info.

        - ``logger`` is a logging object where warning and
          error messages are sent."""

        _debug("check_remote")
        if self._repo_locator is None:
            from .chimera_locator import ChimeraLocator
            self._repo_locator = ChimeraLocator(self.remote_url)
        distributions = self._repo_locator.get_distributions()
        ti_list = []
        for d in distributions:
            ti_list.extend(_make_tool_info(logger, d, False))
            _debug("added remote distribution:", d)
        return ti_list

    def reload(self, logger, rebuild_cache=False, check_remote=False):
        """Discard and reread tool info.

        - ``logger``, ``check_remote`` and ``rebuild_cache``
          have the same meaning as in the constructor."""

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
            remote_ti_list = self.check_remote(logger)
            for ti in remote_ti_list:
                self.add_tool_info(ti)
                # XXX: do we want to register commands so that we can
                # ask user whether to install tool when invoked?

    def tool_info(self, installed=True, available=False):
        """Return list of tool info.

        - ``installed`` is a boolean indicating whether installed
          tools should be included in the returned list.
        - ``available`` is a boolean indicating whether available
          but uninstalled tools should be included."""

        _debug("tool_info", installed, available)
        if installed and available:
            return self._installed_tool_info + self._available_tool_info
        elif installed:
            return self._installed_tool_info
        elif available:
            return self._available_tool_info
        else:
            return []

    def add_tool_info(self, tool_info):
        """Add information for one tool.

        - ``tool_info`` is a constructed instance of '''ToolInfo''',
          i.e., not an existing instance returned by '''tool_info'''.
        A 'TOOLSHED_TOOL_INFO_ADDED' trigger is fired
        after the addition."""
        _debug("add_tool_info", tool_info)
        if tool_info.installed:
            container = self._installed_tool_info
        else:
            container = self._available_tool_info
        container.append(tool_info)
        # TODO: fire trigger

    def install_tool(self, tool_info, logger, system=False):
        """Install the tool by retrieving it from the remote shed.

        - ``tool_info`` should be from the available tool list.
          If the tool is already installed, a '''ToolShedError'''
          exception is raised.
        - ``system`` is a boolean that is False if the tool
          is installed for the current user (default), or
          True if installed for everyone.
        - ``logger`` is a logging object where warning and
          error messages are sent.
        A '''TOOLSHED_TOOL_INSTALLED''' trigger is fired
        after installation."""
        _debug("install_tool", tool_info)
        if tool_info.installed:
            raise ToolShedInstalledError("tool \"%s\" already installed"
                                         % tool_info.name)
        self._install_tool(tool_info, system, logger)
        self._write_cache(self._installed_tool_info, logger)
        # TODO: implement self._install_tool
        # TODO: fire trigger

    def uninstall_tool(self, tool_info, logger):
        """Uninstall the tool by removing the corresponding
        Python package.

        - ``tool_info`` should be from the installed list.
          If the tool is not installed, a '''ValueError'''
          exception is raised.
        - ``logger`` is a logging object where warning and
          error messages are sent.
        A '''TOOLSHED_TOOL_UNINSTALLED''' trigger is fired
        after package removal."""
        _debug("uninstall_tool", tool_info)
        self._uninstall_tool(tool_info, logger)
        self._write_cache(self._installed_tool_info, logger)
        # TODO: fire trigger

    def find_tool(self, name, installed=True, version=None):
        """Return a tool with the given name.

        - ``name`` is a string of the name (internal or
        display name) of the tool of interest."""
        _debug("find_tool", name, installed, version)
        if installed:
            container = self._installed_tool_info
        else:
            container = self._available_tool_info
        for ti in container:
            _debug("  find_tool check", ti.name, ti.display_name, ti.version)
            if ((ti.name == name or ti.display_name == name)
               and (version is None or version == ti.version)):
                return ti
        return None

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
            tool_info.extend(_make_tool_info(logger, d, True))
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
            from distlib.locators import DistPathLocator
            self._inst_locator = DistPathLocator(self._inst_path)
        _debug("_inst_path", self._inst_path)
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
            raise ToolShedUnavailableError("no distribution information "
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
            raise ToolShedUnavailableError("cannot find new distribution "
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
        _debug("all", all)
        all.update([(d.name, d) for d in need])
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
                try:
                    fn, headers = urlretrieve(url, dloc)
                except URLError as e:
                    logger.warning("cannot fetch %s: %s" % (url, str(e)))
                    continue
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
            raise ToolShedUninstalledError("trying to remove uninstalled "
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
        keep = [ti for ti in self._installed_tool_info if ti.distribution() != dv]
        self._installed_tool_info = keep
        self._remove_distribution(d, logger)

    # End methods for installing and removing distributions


class ToolInfo:
    """ToolInfo manages how to create an ToolInstance.

    An ToolInfo knows about the properties about a class
    of tools and can create an tool instance."""

    def __init__(self, name, installed,
                 distribution_name=None,
                 distribution_version=None,
                 display_name=None,
                 module_name=None,
                 synopsis=None,
                 menu_categories=(),
                 command_names=()):
        """Initialize tool info named 'name'.

        Supported keywords include:
        - ``distribution_name``: name of distribution that
          provided this tool (string)
        - ``display_name``: name to display in user interface
          for this tool (string)
        - ``module_name``: Name of module implementing the tool.
          Must be a dotted Python name.  (See module doc string.)
        - ``menu_categories``: list of categories (strings)
          in which tool belongs
        - ``command_names``: list of names of command (strings) in CLI
        """

        # Public attributes
        self.name = name
        self.installed = installed
        self.display_name = display_name or name
        self.menu_categories = menu_categories
        self.command_names = command_names
        self.synopsis = synopsis

        # Private attributes
        self._distribution_name = distribution_name
        self._distribution_version = distribution_version
        self._module_name = module_name

    @property
    def version(self):
        """Tool version number.

        This is the same as the tool distribution version number and
        is available as a read-only property."""
        return self._distribution_version

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
        """Return 2-tuple of (args, kw) that can be used
        to recreate with ToolInfo(*args, **kw)."""
        args = (self.name, self.installed)
        kw = {
            "display_name": self.display_name,
            "menu_categories": self.menu_categories,
            "command_names": self.command_names,
            "synopsis": self.synopsis,
            "distribution_name": self._distribution_name,
            "distribution_version": self._distribution_version,
            "module_name": self._module_name,
        }
        return args, kw

    def distribution(self):
        """Return distribution information as (name, version) tuple."""
        return self._distribution_name, self._distribution_version

    def synopsis(self):
        """Return short description of tool."""
        return self.synopsis or "no synopsis available"

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
            raise ToolShedError("no module specified for tool \"%s\"" % self.name)
        import importlib
        m = importlib.import_module(self._module_name)
        _debug("_get_module", self._module_name, m)
        return m

    def start(self, session, *args, **kw):
        """Create and return an tool instance.

        ``session`` is a Session instance in which the tool will run.
        ``args`` and 'kw' are passed as positional and keyword
        arguments to the ToolInstance constructor.

        If the tool is not installed,
        '''ToolShedUninstalledError''' is raised.
        If the tool cannot be started,
        '''ToolShedError''' is raised."""
        if not self.installed:
            raise ToolShedUninstalledError("tool \"%s\" is not installed"
                                           % self.name)
        try:
            f = self._get_module().start_tool
        except (ImportError, AttributeError, TypeError):
            raise ToolShedError("bad start callable specified for tool \"%s\""
                                % self.name)
        else:
            f(session, self, *args, **kw)


# Tools and ToolInstance are session-specific
from ..session import State
ADD_TOOL_INSTANCE = 'add tool instance'
REMOVE_TOOL_INSTANCE = 'remove tool instance'


class ToolInstance(State):
    """ToolInstance is the abstract base class for
    tool instance classes that implement actual functionality,
    in particular the '''session.State''' API.

    All session-related data are stored in ToolInstance instances,
    not in any ToolShed or ToolInfo instances."""

    def __init__(self, session, *args, **kw):
        """Initialize an ToolInstance.

        Supported keyword include:
        - ``session_data``: data read from session file; if present,
          this data overrides information from all other arguments
        """
        self.id = None
        # TODO: track.created(ToolInstance, [self])

    def delete(self):
        if self.id is not None:
            raise ValueError("tool instance is still in use")
        # TODO: track.deleted(ToolInstance, [self])


class Tools(State):
    """Tools is a session state manager for running tools."""
    # Most of this code is modeled after models.Models

    VERSION = 1     # snapshot version

    def __init__(self, session):
        """Initialize session state manager for ToolInstance instances."""
        import weakref
        self._session = weakref.ref(session)
        session.triggers.add_trigger(ADD_TOOL_INSTANCE)
        session.triggers.add_trigger(REMOVE_TOOL_INSTANCE)
        self._tool_instances = {}
        import itertools
        self._id_counter = itertools.count(1)

    def take_snapshot(self, session, flags):
        """Override State default method."""
        from ..session import unique_id
        data = {}
        for tid, ti in self._tool_instances.items():
            assert(isinstance(ti, ToolInstance))
            data[tid] = [unique_id(ti), ti.take_snapshot(session, flags)]
        return [self.VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        """Override State default method."""
        if version != self.VERSION or not data:
            raise RuntimeError("Unexpected version or data")

        for tid, [uid, [ti_version, ti_data]] in data.items():
            if phase == State.PHASE1:
                try:
                    cls = session.class_of_unique_id(uid, ToolInstance)
                except KeyError:
                    class_name = session.class_name_of_unique_id(uid)
                    session.log.warning("Unable to restore tool instance %s (%s)"
                                        % (id, class_name))
                    continue
                ti = cls("unnamed restored tool instance")
                ti.id = tid
                self.tool_instances[tid] = ti
                session.restore_unique_id(ti, uid)
            else:
                ti = session.unique_obj(uid)
            ti.restore_snapshot(phase, session, ti_version, ti_data)

    def reset_state(self):
        """Override State default method."""
        ti_list = self._tool_instances.values()
        self._tool_instances.clear()
        for ti in ti_list:
            ti.delete()

    def list(self):
        """Return list of running tools."""
        return list(self._tool_instances.values())

    def add(self, ti_list, id=None):
        """Add running tools to session."""
        session = self._session()   # resolve back reference
        for ti in ti_list:
            # TODO:
            # if id is not None
            #   ti.id = id
            # else:
            if True:
                ti.id = next(self._id_counter)
            self._tool_instances[ti.id] = ti
        session.triggers.activate_trigger(ADD_TOOL_INSTANCE, ti_list)

    def remove(self, ti_list):
        """Remove running tools from session."""
        session = self._session()   # resolve back reference
        session.triggers.activate_trigger(REMOVE_TOOL_INSTANCE, ti_list)
        for ti in ti_list:
            tid = ti.id
            if tid is None:
                # Not registered in a session
                continue
            ti.id = None
            del self._tool_instance[tid]


#
# Code in remainder of file are for internal use only
#


def _make_tool_info(logger, d, installed):
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


# Toolshed is a singleton.  Multiple calls to init returns the same instance.
_toolshed = None


def init(*args, debug=False, **kw):
    """Initialize toolshed.  The toolshed is a singleton, so
    the first call creates the instance and all subsequent
    calls return the same instance.  The toolshed debugging
    state is updated at each call.

    This function accepts all the arguments for the ``Toolshed``
    initializer.  In addition:

    - ``debug`` is a boolean value.  If true, debugging messages
      are sent to standard output.  Default value is false."""
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
        _toolshed = ToolShed(*args, **kw)
    return _toolshed
