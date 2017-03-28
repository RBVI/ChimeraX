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


class AvailableBundleCache(list):

    def load(self, logger, toolshed_url):
        #
        # Load bundle info from ChimeraX toolshed using
        # json interface.
        #
        _debug("AvailableBundleCache.load: toolshed_url", toolshed_url)
        from urllib.parse import urljoin
        url = urljoin(toolshed_url, "bundle/")
        _debug("AvailableBundleCache.load: url", url)
        from urllib.request import urlopen
        with urlopen(url) as f:
            import json
            data = json.loads(f.read())
        for d in data:
            b = _build_bundle(d)
            if b:
                self.append(b)


def _build_bundle(d):
    # "d" is a dictionary with (some of) the following keys:
    #    bundle_name: name of bundle (with "_", not "-")
    #    toolshed_name: name of toolshed name for bundle (version-independent)
    #    bundle: dictionary of general bundle information
    #    tool: dictionary of information for tools
    #    command: dictionary of information for commands
    #    selector: dictionary of information for selectors
    #    dataformat: dictionary of information for data formats
    #    fetch: dictionary of information for fetching from databases
    #    open: dictionary of information for opening files
    #    save: dictionary of information for saving files
    # Information was harvested by toolshed from submitted bundle using
    # both general wheel information and ChimeraX classifier fields.

    bundle_name = d["bundle_name"]
    #
    # First create BundleInfo instance
    #
    from .info import BundleInfo
    kw = {}
    kw["name"] = d["bundle_name"].replace("_", "-")
    bundle_d = d["bundle"][bundle_name]
    try:
        kw["version"] = bundle_d["version"]
    except KeyError:
        return None
    s = d.get("description", None)
    if s:
        kw["synopsis"] = s
    s = d.get("details", None)
    if s:
        kw["description"] = s
    _set_value(kw, bundle_d, "categories")
    _set_value(kw, bundle_d, "session_versions", _parse_session_versions)
    _set_value(kw, bundle_d, "api_package_name")
    _set_value(kw, bundle_d, "supercedes")
    _set_value(kw, bundle_d, "custom_init", lambda v: v == "true")
    bi = BundleInfo(installed=False, **kw)

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
            _debug("processing tool: %s" % tool_name)
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
            _debug("processing command: %s" % cmd_name)
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
            _debug("processing selector: %s" % sel_name)
            synopsis = sd.get("synopsis", "")
            si = CommandInfo(sel_name, synopsis)
            bi.selectors.append(si)

    #
    # Process format information
    #
    format_map = {}
    try:
        fmt_d = d["dataformat"]
    except KeyError:
        # No data formats defined
        pass
    else:
        from .info import FormatInfo
        for fmt_name, fd in fmt_d.items():
            _debug("processing data format: %s" % fmt_name)
            nicknames = fd.get("nicknames", [])
            categories = fd.get("categories", [])
            suffixes = fd.get("suffixes", [])
            mime_types = fd.get("mime_types", [])
            url = fd.get("url", "")
            icon = fd.get("icon", "")
            dangerous = fd.get("dangerous", True)
            synopsis = fd.get("synopsis", "")
            encoding = fd.get("encoding", "")
            fi = CommandInfo(name=fmt_name, nicknames=nicknames,
                             category=categories, suffixes=suffixes,
                             mime_types=mime_types, url=url, icon=icon,
                             dangerous=dangerous, synopsis=synopsis,
                             encoding=encoding)
            format_map[fmt_name] = fi
            bi.formats.append(fi)

    #
    # Process fetch information
    #
    try:
        fetch_d = d["fetch"]
    except KeyError:
        # No fetch from database methods defined
        pass
    else:
        for db_name, fd in fetch_d.items():
            _debug("processing fetch: %s" % db_name)
            format_name = fd.get("format", "")
            prefixes = fd.get("prefixes", [])
            example = fd.get("example", "")
            is_default = fd.get("is_default", "") == "true"
            fi = (db_name, format_name, prefixes, example, is_default)
            bi.fetches.append(fi)

    #
    # Process open information
    #
    try:
        open_d = d["open"]
    except KeyError:
        # No open from database methods defined
        pass
    else:
        from .installed import _extract_extra_keywords
        for fmt_name, fd in open_d.items():
            _debug("processing open: %s" % filetype_name)
            try:
                fi = format_map[fmt_name]
            except KeyError:
                continue
            tag = fd.get("tag", "")
            is_default = fd.get("is_default", "") == "true"
            keywords = fd.get("keywords", None)
            if keywords:
                keywords = _extract_extra_keywords(keywords)
            fi.has_open = True
            fi.open_kwds = keywords

    #
    # Process save information
    #
    try:
        save_d = d["save"]
    except KeyError:
        # No save from database methods defined
        pass
    else:
        from .info import FormatInfo
        for fmt_name, fd in save_d.items():
            _debug("processing save: %s" % filetype_name)
            try:
                fi = format_map[fmt_name]
            except KeyError:
                continue
            tag = fd.get("tag", "")
            is_default = fd.get("is_default", "") == "true"
            keywords = fd.get("keywords", None)
            if keywords:
                keywords = _extract_extra_keywords(keywords)
            fi.has_open = True
            fi.open_kwds = keywords

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
    return range(lo, hi+1)

import re
_REReq = re.compile("""(?P<bundle>\S+)\s*\(>=(?P<version>\S+)\)""")

def _parse_requires(r):
    # Only handle requirements of form "bundle (>=version)" for now
    m = _REReq.match(r)
    if m is None:
        return None
    return (m.group("bundle"), m.group("version"))
