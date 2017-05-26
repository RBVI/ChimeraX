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


from . import _debug
from . import _TIMESTAMP


def _hack_distlib(f):
    from functools import wraps

    @wraps(f)
    def hacked_f(self, logger, *args, **kw):
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
            v = f(self, logger, *args, **kw)
        finally:
            # Restore hacked name
            metadata.METADATA_FILENAME = save
            database.METADATA_FILENAME = save
            wheel.METADATA_FILENAME = save
            _debug("changing back METADATA_FILENAME", metadata.METADATA_FILENAME)
        return v
    return hacked_f


class InstalledBundleCache(list):

    @_hack_distlib
    def load(self, logger, cache_file=None, rebuild_cache=False, write_cache=True):
        """Load list of installed bundles.

        Bundles are ordered based on dependency with dependent
        bundles appearing later in the list."""

        #
        # Load bundle info.  If not rebuild_cache, try reading
        # it from a cache file.  If we cannot use the cache,
        # read the information from the data directory and
        # try to create the cache file.
        #
        _debug("InstalledBundleCache.load: rebuild_cache", rebuild_cache)
        if cache_file and not rebuild_cache:
            if self._read_cache(cache_file):
                _debug("InstalledBundleCache.load: using cached data")
                return
        #
        # Okay, no cache.  Go through all installed packages
        # and look for ChimeraX bundles
        #
        _debug("InstalledBundleCache.load: rebuilding cache")
        from distlib.database import DistributionPath
        inst_path = DistributionPath()
        dist_bundle_map = {}
        for d in inst_path.get_distributions():
            _debug("InstalledBundleCache.load: distribution %s" % d)
            bi = _make_bundle_info(d, True, logger)
            if bi is not None:
                _debug("InstalledBundleCache.load: bundle %s" % bi)
                dist_bundle_map[d] = bi
        #
        # The ordering of the bundles is important because we want
        # to call the initialize() method in the correct order.
        # To compute the ordering, we use distlib to compute a
        # dependency graph and then we iteratively loop through
        # distributions.  In each pass, we find distributions that
        # either do not depend on any other distribution or only
        # those that we have already removed from the list.  These
        # distributions are moved to an ORDERED bundle SET.
        # A pass must removed at least one distribution.  If not,
        # we have a circular reference.
        #
        from distlib.database import make_graph
        from ..orderedset import OrderedSet
        all_distributions = set(dist_bundle_map.keys())
        keep = OrderedSet()
        dg = make_graph(all_distributions)
        while all_distributions:
            can_move = []
            for d in all_distributions:
                for depends, _ in dg.adjacency_list[d]:
                    if depends in all_distributions:
                        break
                else:
                    # Nothing this distribution depends on
                    # is still on our list to check
                    can_move.append(d)
            if can_move:
                # Found some things to move
                keep.update(can_move)
                all_distributions.difference_update(can_move)
            else:
                # Remaining must depend on each other
                # (one or more dependency cycles)
                # XXX: For now, we just move them all to
                # the keep list and hope for the best
                logger.warning("Unexpected circular dependencies:",
                               ', '.join([str(d)
                                          for d in all_distributions]))
                keep.update(all_distributions)
                all_distributions.clear()
        for d in keep:
            self.append(dist_bundle_map[d])

        #
        # Save all our hard work
        #
        if cache_file and write_cache:
            _debug("InstalledBundleCache.load: write_cache")
            self._write_cache(cache_file, logger)

    def register_all(self, logger, session, package_map):
        """Register all installed bundles.

        Returns dictionary of package name to bundle instance."""

        for bi in self:
            for p in bi.packages:
                package_map[p] = bi
            bi.register(logger)
            if session is not None:
                bi.initialize(session)
        return package_map

    def deregister_all(self, logger, session, package_map):
        """Deregister all installed bundles."""

        for bi in reversed(self):
            for p in bi.packages:
                try:
                    del package_map[p]
                except KeyError:
                    pass
            if session is not None:
                bi.finish(session)
            bi.deregister(logger)

    def set_install_timestamp(self, per_user=False):
        """Set last installed timestamp."""

        import os
        if per_user:
            from chimerax import app_dirs
            directory = app_dirs.user_data_dir
        else:
            from chimerax import app_data_dir as directory
        timestamp_file = os.path.join(directory, _TIMESTAMP)
        with open(timestamp_file, 'w') as f:
            # Contents of file are never read, see _is_cache_newer()
            import time
            print(time.ctime(), file=f)

    #
    # Methods below are internal
    #
    def _read_cache(self, cache_file):
        """Read installed bundle information from cache file.

        Returns boolean on whether cache file was read."""
        _debug("InstalledBundleCache._read_cache:", cache_file)
        import filelock, json, os, sys
        from .info import BundleInfo
        try:
            lock_file = cache_file + '.lock'
            lock = filelock.FileLock(lock_file)
            with lock.acquire():
                if not lock.is_locked:
                    # As of filelock==2.0.8:
                    # On Unix, failing to create the lock file results
                    # in an exception, but on Windows the acquire fails
                    # silently but leaves the lock unlocked
                    raise OSError("cannot create lock file %r" % lock_file)
                f = open(cache_file, "r", encoding='utf-8')
                try:
                    with f:
                        data = f.readline()
                        data = json.loads(data)
                        if not isinstance(data[0], str):
                            _debug("InstalledBundleCache._read_cache: obsolete cache")
                            return None  # obsolete cache format
                        executable, mtime = data
                        if executable != sys.executable:
                            _debug("InstalledBundleCache._read_cache: different executable")
                            return None
                        if mtime != os.path.getmtime(executable):
                            _debug("InstalledBundleCache._read_cache: changed executable")
                            return None
                        if not self._is_cache_newer(cache_file):
                            return None
                        data = json.load(f)
                    self.extend([BundleInfo.from_cache_data(x) for x in data])
                    _debug("InstalledBundleCache._read_cache: %d bundles" % len(self))
                    return True
                except Exception as e:
                    _debug("InstalledBundleCache._read_cache: failed:", e)
                    return False
        except OSError as e:
            _debug("InstalledBundleCache._read_cache: failed os:", str(e))
            return False

    def _is_cache_newer(self, cache_file):
        """Check if cache is newer than timestamps."""
        _debug("_is_cache_newer")
        from chimerax import app_dirs, app_data_dir
        import os
        files = (
            (os.path.join(app_data_dir, _TIMESTAMP), True),
            (os.path.join(app_dirs.user_data_dir, _TIMESTAMP), False),
        )
        try:
            cache_time = os.path.getmtime(cache_file)
        except FileNotFoundError:
            return False
        for filename, required in files:
            try:
                time = os.path.getmtime(filename)
                if time > cache_time:
                    return False
            except FileNotFoundError:
                if required:
                    return False
        return True

    def _write_cache(self, cache_file, logger):
        """Write current bundle information to cache file."""
        _debug("InstalledBundleCache._write_cache", cache_file)
        import os, os.path, filelock, json, sys
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        lock = filelock.FileLock(cache_file + '.lock')
        with lock.acquire():
            try:
                f = open(cache_file, 'w', encoding='utf-8')
            except IOError as e:
                logger.error("\"%s\": %s" % (cache_file, str(e)))
            else:
                with f:
                    data = [sys.executable, os.path.getmtime(sys.executable)]
                    json.dump(data, f, ensure_ascii=False)
                    print(file=f)
                    json.dump([bi.cache_data() for bi in self], f,
                              ensure_ascii=False, check_circular=False)


#
# Class-independent utility functions available to other modules in package
#


def _extract_extra_keywords(kwds):
    result = {}
    all_kwds = [k.strip() for k in kwds.split(',')]
    for k in all_kwds:
        temp = [t.strip() for t in k.split(':', maxsplit=1)]
        if len(temp) == 1:
            result[temp[0]] = 'String'
        else:
            result[temp[0]] = temp[1]
    return result


def _report_difference(logger, before, after):
    bundles = {}
    for bi in before:
        bundles[bi.name] = [bi.version, None]
    for bi in after:
        try:
            versions = bundles[bi.name]
        except KeyError:
            bundles[bi.name] = [None, bi.version]
        else:
            versions[1] = bi.version
    messages = []
    for name in sorted(bundles.keys()):
        versions = bundles[name]
        if versions[0] is None:
            messages.append("Installed %s (%s)" % (name, versions[1]))
        elif versions[1] is None:
            messages.append("Removed %s (%s)" % (name, versions[0]))
        elif versions[0] != versions[1]:
            messages.append("Updated %s (from %s to %s)"
                            % (name, versions[0], versions[1]))
    if messages:
        logger.info('\n'.join(messages))
    else:
        logger.info("No change in list of installed bundles")


#
# Class-independent utility functions only used in this module
#


def _make_bundle_info(d, installed, logger):
    """Convert distribution into a list of :py:class:`BundleInfo` instances."""
    from .info import BundleInfo, ToolInfo, CommandInfo, SelectorInfo, FormatInfo
    name = d.name
    version = d.version
    md = d.metadata.dictionary

    if 'classifiers' not in md:
        _debug("InstalledBundleCache._make_bundle_info: no classifiers in %s" % d)
        return None

    bi = None
    kw = {"name": name, "version": version}
    try:
        kw['synopsis'] = md["summary"]
    except KeyError:
        _debug("InstalledBundleCache._make_bundle_info: no summary in %s" % d)
        return None
    kw['packages'] = _get_installed_packages(d, logger)
    for classifier in md["classifiers"]:
        parts = [v.strip() for v in classifier.split("::")]
        if parts[0] != 'ChimeraX':
            continue
        if parts[1] == 'Bundle':
            # ChimeraX :: Bundle :: categories :: session_versions :: module_name :: supercedes :: custom_init
            if bi is not None:
                logger.warning("Second ChimeraX :: Bundle line ignored.")
                break
            elif len(parts) != 7:
                logger.warning("Malformed ChimeraX :: Bundle line in %s skipped." % name)
                logger.warning("Expected 7 fields and got %d." % len(parts))
                continue
            categories, session_versions, module_name, supercedes, custom_init = parts[2:]
            kw["categories"] = [v.strip() for v in categories.split(',')]
            if session_versions:
                vs = [v.strip() for v in session_versions.split(',')]
                if len(vs) != 2:
                    logger.warning("Malformed ChimeraX :: Bundle line in %s skipped." % name)
                    logger.warning("Expected 2 version numbers and got %d." % len(vs))
                    continue
                try:
                    lo = int(vs[0])
                    hi = int(vs[1])
                except ValueError:
                    logger.warning("Malformed ChimeraX :: Bundle line in %s skipped." % name)
                    logger.warning("Found non-integer version numbers.")
                    continue
                if lo > hi:
                    logger.warning("Minimum version is greater than maximium.")
                    hi = lo
                kw["session_versions"] = range(lo, hi + 1)
            kw["api_package_name"] = module_name
            if supercedes:
                kw['supercedes'] = [v.strip() for v in supercedes.split(',')]
            if custom_init:
                kw["custom_init"] = (custom_init == "true")
            bi = BundleInfo(installed=installed, **kw)
            bi.path = d.path
        elif parts[1] == 'Tool':
            # ChimeraX :: Tool :: tool_name :: categories :: synopsis
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) != 5:
                logger.warning("Malformed ChimeraX :: Tool line in %s skipped." % name)
                logger.warning("Expected 5 fields and got %d." % len(parts))
                continue
            # Menu categories in which tool should appear
            name, categories, synopsis = parts[2:]
            if not categories:
                logger.warning("Missing tool categories")
                continue
            categories = [v.strip() for v in categories.split(',')]
            ti = ToolInfo(name, categories, synopsis)
            bi.tools.append(ti)
        elif parts[1] == 'Command':
            # ChimeraX :: Command :: name :: categories :: synopsis
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) != 5:
                logger.warning("Malformed ChimeraX :: Command line in %s skipped." % name)
                logger.warning("Expected 5 fields and got %d." % len(parts))
                continue
            name, categories, synopsis = parts[2:]
            if not categories:
                logger.warning("Missing command categories")
                continue
            categories = [v.strip() for v in categories.split(',')]
            ci = CommandInfo(name, categories, synopsis)
            bi.commands.append(ci)
        elif parts[1] == 'Selector':
            # ChimeraX :: Selector :: name :: synopsis
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) != 4:
                logger.warning("Malformed ChimeraX :: Selector line in %s skipped." % name)
                logger.warning("Expected 4 fields and got %d." % len(parts))
                continue
            name, synopsis = parts[2:]
            si = SelectorInfo(name, synopsis)
            bi.selectors.append(si)
        elif parts[1] == 'DataFormat':
            # ChimeraX :: DataFormat :: format_name :: nicknames :: category :: suffixes :: mime_types :: url :: dangerous :: icon :: synopsis :: encoding
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) not in [11, 12]:
                logger.warning("Malformed ChimeraX :: DataFormat line in %s skipped." % name)
                logger.warning("Expected 11 or 12 fields and got %d." % len(parts))
                continue
            if len(parts) == 12:
                name, nicknames, category, suffixes, mime_types, url, dangerous, icon, synopsis, encoding = parts[2:]
            else:
                encoding = None
                name, nicknames, category, suffixes, mime_types, url, dangerous, icon, synopsis = parts[2:]
            nicknames = [v.strip() for v in nicknames.split(',')] if nicknames else None
            suffixes = [v.strip() for v in suffixes.split(',')] if suffixes else None
            mime_types = [v.strip() for v in mime_types.split(',')] if mime_types else None
            # construct absolute path name of icon by looking
            # in package directory
            if icon:
                icon = bi.find_icon_path(icon)
            fi = FormatInfo(name=name, nicknames=nicknames,
                            category=category, suffixes=suffixes,
                            mime_types=mime_types, url=url, icon=icon,
                            dangerous=dangerous, synopsis=synopsis,
                            encoding=encoding)
            bi.formats.append(fi)
        elif parts[1] == 'Fetch':
            # ChimeraX :: Fetch :: database_name :: format_name :: prefixes :: example_id :: is_default
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) != 7:
                logger.warning("Malformed ChimeraX :: DataFormat line in %s skipped." % name)
                logger.warning("Expected 7 fields and got %d." % len(parts))
                continue
            database_name, format_name, prefixes, example_id, is_default = parts[2:]
            prefixes = [v.strip() for v in prefixes.split(',')] if prefixes else ()
            is_default = (is_default == 'true')
            bi.fetches.append((database_name, format_name, prefixes, example_id, is_default))
        elif parts[1] == 'Open':
            # ChimeraX :: Open :: format_name :: tag :: is_default :: keyword_arguments
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) not in [5, 6]:
                logger.warning("Malformed ChimeraX :: Open line in %s skipped." % name)
                logger.warning("Expected 5 or 6 fields and got %d." % len(parts))
                continue
            if len(parts) == 6:
                name, tag, is_default, kwds = parts[2:]
                kwds = _extract_extra_keywords(kwds)
            else:
                name, tag, is_default = parts[2:]
                kwds = None
            is_default = (is_default == 'true')
            try:
                fi = [fi for fi in bi.formats if fi.name == name][0]
            except (KeyError, IndexError):
                logger.warning("Unknown format name: %r." % name)
                continue
            fi.has_open = True
            fi.open_kwds = kwds
        elif parts[1] == 'Save':
            # ChimeraX :: Save :: format_name :: tag :: is_default :: keyword_arguments
            if bi is None:
                logger.warning('ChimeraX :: Bundle entry must be first')
                return None
            if len(parts) not in [5, 6]:
                logger.warning("Malformed ChimeraX :: Save line in %s skipped." % name)
                logger.warning("Expected 5 or 6 fields and got %d." % len(parts))
                continue
            if len(parts) == 6:
                name, tag, is_default, kwds = parts[2:]
                kwds = _extract_extra_keywords(kwds)
            else:
                name, tag, is_default = parts[2:]
                kwds = None
            is_default = (is_default == 'true')
            try:
                fi = [fi for fi in bi.formats if fi.name == name][0]
            except (KeyError, IndexError):
                logger.warning("Unknown format name: %r." % name)
                continue
            fi.has_save = True
            fi.save_kwds = kwds
    if bi is None:
        _debug("InstalledBundleCache._make_bundle_info: no ChimeraX bundle in %s" % d)
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

    #
    # If the bundle does not implement BundleAPI interface,
    # act as if it were not a bundle
    #
    from . import ToolshedError
    try:
        bi._get_api(logger)
    except ToolshedError as e:
        _debug("InstalledBundleCache._make_bundle_info: %s" % str(e))
        return None
    return bi


def _get_installed_packages(dist, logger):
    """Return set of tuples representing the packages in the distribution.

    For example, 'foo.bar' from foo/bar/__init__.py becomes ('foo', 'bar')
    """
    packages = []
    try:
        installed = dist.list_installed_files()
        for path, hash, size in installed:
            if not path.endswith('/__init__.py'):
                continue
            parts = path.split('/')
            packages.append(tuple(parts[:-1]))
    except:
        logger.warning("cannot get installed file list for %r" % dist.name)
        return []
    return packages
