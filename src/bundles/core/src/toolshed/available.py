# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===


from . import _debug
from .info import BundleInfo
from functools import cached_property

_CACHE_FILE = "available.json"
FORMAT_VERSION = 2


class _AvailableBundleInfo(BundleInfo):
    """Resolve catalog requirements and UI records only when requested."""

    def _set_ui_entries(self, data):
        # Keep the catalog's raw UI sections until a caller requests their
        # records. Startup provider registration needs none of these lists.
        for section, attribute in (('tool', 'tools'), ('command', 'commands'),
                                   ('selector', 'selectors')):
            entries = data.get(section)
            if entries:
                self.__dict__['_' + section + '_data'] = entries
                del self.__dict__[attribute]

    @cached_property
    def tools(self):
        from .info import ToolInfo
        records = [ToolInfo(name, fields.get('categories', []), fields.get('synopsis', ''))
                   for name, fields in self._tool_data.items()]
        return self.__dict__.setdefault('tools', records)

    @cached_property
    def commands(self):
        from .info import CommandInfo
        records = [CommandInfo(name, fields.get('categories', []), fields.get('synopsis', ''))
                   for name, fields in self._command_data.items()]
        return self.__dict__.setdefault('commands', records)

    @cached_property
    def selectors(self):
        from .info import SelectorInfo
        records = [SelectorInfo(name, fields.get('synopsis', ''),
                                fields.get('atomic', '').lower() != 'false')
                   for name, fields in self._selector_data.items()]
        return self.__dict__.setdefault('selectors', records)

    def _set_requirements(self, requirements):
        self._requirement_strings = tuple(requirements)
        self._parsed_requirements = {}

    def _iter_requires(self):
        # Once exposed, use the same mutable list callers have been given.
        if 'requires' in self.__dict__:
            yield from self.requires
            return
        for index, text in enumerate(self._requirement_strings):
            if index not in self._parsed_requirements:
                self._parsed_requirements[index] = _parse_requires(text)
            requirement = self._parsed_requirements[index]
            if requirement is not None:
                yield requirement

    @cached_property
    def requires(self):
        return list(self._iter_requires())


class AvailableBundleCache(list):

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.uninstallable = []
        self.toolshed_url = None
        self.format_version = 1
        self._index = {}  # provide access by bundle name

    def load(self, logger, toolshed_url):
        #
        # Load bundle info from ChimeraX toolshed using
        # json interface.
        #
        _debug("AvailableBundleCache.load: toolshed_url", toolshed_url)
        from chimerax import app_dirs
        from urllib.parse import urljoin, urlencode
        from . import chimerax_uuid
        params = [
            ("uuid", chimerax_uuid()),
            ("format_version", FORMAT_VERSION),
        ]
        url = urljoin(toolshed_url, "bundle/") + '?' + urlencode(params)
        _debug("AvailableBundleCache.load: url", url)
        from urllib.request import urlopen, Request
        from ..fetch import html_user_agent
        headers = {"User-Agent": html_user_agent(app_dirs)}
        request = Request(url, unverifiable=True, headers=headers)
        # pick short timeout because it limits how quickly ChimeraX can exit
        # when the toolshed can't be contacted
        with urlopen(request, timeout=12) as f:
            import json
            data = json.loads(f.read())
        data.insert(0, ['toolshed_url', toolshed_url])
        import os
        if self.cache_dir is not None:
            with open(os.path.join(self.cache_dir, _CACHE_FILE), 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, indent=0)
        try:
            from chimerax.registration import nag
        except ImportError:
            _debug("chimerax.registration import failed")
        else:
            _debug("extend registration")
            nag.extend_registration(logger)
        self._build_bundles(data)

    def load_from_cache(self):
        if self.cache_dir is None:
            raise FileNotFoundError("no bundle cache")
        import os
        with open(os.path.join(self.cache_dir, _CACHE_FILE), encoding='utf-8') as f:
            import json
            data = json.load(f)
            stat = os.fstat(f.fileno())
        import chimerax.core
        import packaging
        # A changed catalog or compatibility evaluator must start fresh.
        key = [1, chimerax.core.version, packaging.__version__,
               stat.st_mtime_ns, stat.st_size]
        checks = self._read_compatibility_cache(key)
        previous = checks.copy()
        self._build_bundles(data, checks)
        # Markers depend on the current environment, not just Core's version.
        persistent = {text: result for text, result in checks.items() if ';' not in text}
        if persistent != previous:
            self._write_compatibility_cache(key, persistent)

    def _read_compatibility_cache(self, key):
        import json
        import os
        try:
            with open(os.path.join(self.cache_dir, 'core-compatibility.json'),
                      encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, list) or len(data) != 2 or data[0] != key:
            return {}
        checks = data[1]
        if not isinstance(checks, dict) or any(
                not isinstance(text, str) or ';' in text
                or (result is not None and type(result) is not bool)
                for text, result in checks.items()):
            return {}
        return checks

    def _write_compatibility_cache(self, key, checks):
        import json
        import os
        from ..safesave import SaveTextFile
        try:
            with SaveTextFile(os.path.join(self.cache_dir, 'core-compatibility.json'),
                              encoding='utf-8') as f:
                json.dump([key, checks], f)
        except OSError:
            # This optional cache must not prevent loading the catalog.
            pass

    def _build_bundles(self, data, requirement_checks=None):
        import chimerax.core
        # A version string: SpecifierSet.contains() parses it if a requirement
        # is not already in requirement_checks, so a start with every check
        # cached does not import packaging.version.
        my_version = chimerax.core.version
        # Share compatibility decisions, never mutable Requirement objects.
        if requirement_checks is None:
            requirement_checks = {}
        for d in data:
            if isinstance(d, list):
                if d[0] == 'format_version':
                    self.format_version = d[1]
                elif d[0] == 'toolshed_url':
                    self.toolshed_url = d[1]
            else:
                b = _build_bundle(d, self.format_version)
                if not b:
                    continue
                if self._installable(b, my_version, requirement_checks):
                    self.append(b)
                else:
                    self.uninstallable.append(b)
                self._index[b.name] = b

    def _installable(self, b, my_version, requirement_checks=None):
        if requirement_checks is not None and 'requires' not in b.__dict__:
            for text in b._requirement_strings:
                if text not in requirement_checks:
                    requirement_checks[text] = _core_requirement_satisfied(
                        _parse_requires(text), my_version)
                result = requirement_checks[text]
                if result is not None:
                    return result
        else:
            # Once a caller exposes/edits requires, honor that bundle's own
            # objects.  Direct checks retain the existing prefix parse cache.
            for require in b._iter_requires():
                result = _core_requirement_satisfied(require, my_version)
                if result is not None:
                    return result
        return False

    def find_by_name(self, name):
        try:
            return self._index[name]
        except KeyError:
            return None


def has_cache_file(cache_dir):
    if cache_dir is None:
        return False
    import os
    return os.path.exists(os.path.join(cache_dir, _CACHE_FILE))


def _build_bundle(d, format_version=1):
    # "d" is a dictionary with (some of) the following keys:
    #    bundle_name: name of bundle (with "_", not "-")
    #    toolshed_name: name of toolshed name for bundle (version-independent)
    #    bundle: dictionary of general bundle information
    #    tool: dictionary of information for tools
    #    command: dictionary of information for commands
    #    selector: dictionary of information for selectors
    #    manager: dictionary of information for managers
    #    provider: dictionary of information for providers
    # Information was harvested by toolshed from submitted bundle using
    # both general wheel information and ChimeraX classifier fields.

    bundle_name = d["bundle_name"]
    #
    # First create BundleInfo instance
    #
    kw = {}
    kw["name"] = d["bundle_name"].replace("_", "-")
    try:
        # The '_' vs '-' problem is everywhere, so just try both
        try:
            bundle_d = d["bundle"][bundle_name]
        except KeyError:
            bundle_d = d["bundle"][kw["name"]]
        kw["version"] = bundle_d["version"]
    except KeyError:
        return None
    _debug("build available bundle", bundle_name, kw["version"])
    s = d.get("description", None)
    if s:
        kw["synopsis"] = s
    s = d.get("details", None)
    if s:
        kw["description"] = s
    _set_value(kw, bundle_d, "categories")
    _set_value(kw, bundle_d, "session_versions", _parse_session_versions)
    _set_value(kw, bundle_d, "api_package_name")
    _set_value(kw, bundle_d, "supersedes")
    _set_value(kw, bundle_d, "custom_init", lambda v: v == "true")
    bi = _AvailableBundleInfo(installed=False, **kw)

    if format_version >= 2:
        bi.release_file = d.get('release_file', None)

    #
    # Squirrel away requirements information
    #
    requires = bundle_d.get("requires", [])
    bi._set_requirements(requires)

    bi._set_ui_entries(d)

    #
    # Process manager information
    #
    try:
        manager_d = d["manager"]
    except KeyError:
        # No managers defined
        pass
    else:
        for manager_name, md in manager_d.items():
            # _debug("processing manager: %s" % manager_name)
            bi.managers[manager_name] = md

    #
    # Process provider information
    #
    try:
        provider_d = d["provider"]
    except KeyError:
        # No providers defined
        pass
    else:
        for provider_name, pd in provider_d.items():
            # _debug("processing provider: %s" % provider_name)
            bi.providers[provider_name] = pd

    #
    # Finished.  Return BundleInfo instance.
    #
    return bi


def _set_value(kw, d, key, post_process=None):
    try:
        v = d[key]
    except KeyError:
        return
    if post_process:
        try:
            kw[key] = post_process(v)
        except ValueError:
            pass
    else:
        kw[key] = v


def _parse_session_versions(sv):
    vs = [v.strip() for v in sv.split(',')]
    if len(vs) != 2:
        raise ValueError("bad session version format: %s" % repr(sv))
    lo = int(vs[0])
    hi = int(vs[1])
    if lo > hi:
        raise ValueError("bad session version values: %s" % repr(sv))
    return range(lo, hi + 1)


def _parse_requires(r):
    from packaging.requirements import Requirement, InvalidRequirement
    try:
        require = Requirement(r)
    except InvalidRequirement:
        return None
    return require


def _core_requirement_satisfied(require, my_version):
    # None means this is not a Core requirement.  False must stop the scan:
    # only the first Core requirement controls compatibility.
    if require is None or require.name != "ChimeraX-Core":
        return None
    if require.marker is not None and not require.marker.evaluate():
        return False
    return require.specifier.contains(my_version, prereleases=True)
