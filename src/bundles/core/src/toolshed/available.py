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


from . import _debug

_CACHE_FILE = "available.json"
FORMAT_VERSION = 2


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
        with urlopen(request, timeout=3) as f:
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
        self._build_bundles(data)

    def _build_bundles(self, data):
        from packaging.version import Version
        import chimerax.core
        my_version = Version(chimerax.core.version)
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
                if self._installable(b, my_version):
                    self.append(b)
                else:
                    self.uninstallable.append(b)
                self._index[b.name] = b

    def _installable(self, b, my_version):
        installable = False
        for require in b.requires:
            if require.name != "ChimeraX-Core":
                continue
            if require.marker is None:
                okay = True
            else:
                okay = require.marker.evaluate()
            installable = okay and require.specifier.contains(my_version, prereleases=True)
            break
        return installable

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
    from .info import BundleInfo
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
    bi = BundleInfo(installed=False, **kw)

    if format_version >= 2:
        bi.release_file = d.get('release_file', None)

    #
    # Squirrel away requirements information
    #
    requires = bundle_d.get("requires", [])
    bi.requires = [r for r in [_parse_requires(r) for r in requires] if r]

    #
    # Process tool information
    #
    try:
        tool_d = d["tool"]
    except KeyError:
        # No tools defined
        pass
    else:
        from .info import ToolInfo
        for tool_name, td in tool_d.items():
            # _debug("processing tool: %s" % tool_name)
            categories = td.get("categories", [])
            synopsis = td.get("synopsis", "")
            ti = ToolInfo(tool_name, categories, synopsis)
            bi.tools.append(ti)

    #
    # Process command information
    #
    try:
        cmd_d = d["command"]
    except KeyError:
        # No commands defined
        pass
    else:
        from .info import CommandInfo
        for cmd_name, cd in cmd_d.items():
            # _debug("processing command: %s" % cmd_name)
            categories = cd.get("categories", [])
            synopsis = cd.get("synopsis", "")
            ci = CommandInfo(cmd_name, categories, synopsis)
            bi.commands.append(ci)

    #
    # Process selector information
    #
    try:
        sel_d = d["selector"]
    except KeyError:
        # No selectors defined
        pass
    else:
        from .info import SelectorInfo
        for sel_name, sd in sel_d.items():
            # _debug("processing selector: %s" % sel_name)
            synopsis = sd.get("synopsis", "")
            atomic = sd.get("atomic", "").lower() != "false"
            si = SelectorInfo(sel_name, synopsis, atomic)
            bi.selectors.append(si)

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
