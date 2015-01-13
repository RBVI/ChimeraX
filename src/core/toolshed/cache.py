# vim: set expandtab ts=4 sw=4:

"""cache - Chimera 2 Tool Cache

The Tool Cache finds tools in the '''chimera2''' namespace package
and keeps track of some metadata and help file information.

A '''chimera2''' tool is defined as a package with a file named
'''toolinfo.json''', whose content is key-value pairs.
This module does not interpret the file contents but only
makes them available to callers, which can access the contents
using the package name.

Help files for tools are assumed to be HTML files in directories
named '''helpdir'''.  An '''index.html''' file must be present
for the directory to be considered a "real" help directory.
Callers can look up the help directory path using the package name.

This modules uses the '''shed''' module for finding the per-user
directory for storing cached data.
"""

#NAMESPACE_PACKAGE = "chimera2"
NAMESPACE_PACKAGE = "shedtest"    # For testing purposes
CACHE_FILENAME = "toolcache.json"
TOOLINFO_FILENAME = "toolinfo.json"
HELP_DIRNAME = "helpdir"
VERSION = 1

_tool_cache = None
_help_cache = None
_empty = {}

def get_tool_info(tool_name):
    if _tool_cache is None:
        init()
    return _tool_cache.get(tool_name, _empty)

def get_help_dir(tool_name):
    if _help_cache is None:
        init()
    return _help_cache.get(tool_name, None)

def init():
    """Initialize cache.  If the cache file is missing,
    construct and write out new cache."""
    global _tool_cache, _help_cache
    if _tool_cache is not None:
        return
    import shed
    shed.init()        # Safe to call multiple times
    try:
        with open(_cache_filename()) as f:
            import json
            cache = json.load(f)
        if not isinstance(cache, dict):
            raise ValueError("wrong type")
        if cache["version"] != VERSION:
            # Rebuild cache if not expected version
            raise ValueError("wrong version")
        _tool_cache = cache["tool"]
        _help_cache = cache["help"]
    except (IOError, ValueError, KeyError):
        # Cannot open file or parse error or missing version
        # Treat same as missing file
        remake()

def _cache_filename():
    import shed, os.path
    return os.path.join(shed.get_chimera_user_base(), CACHE_FILENAME)

def remake():
    global _tool_cache, _help_cache
    # fill in caches
    _tool_cache, _help_cache = _fill_caches(NAMESPACE_PACKAGE)
    if not _tool_cache and not _help_cache:
        return
    cache = {
        "version": VERSION,
        "tool": _tool_cache,
        "help": _help_cache,
    }
    with open(_cache_filename(), "w") as f:
        import json
        json.dump(cache, f)

def _fill_caches(pkg_name):
    import importlib
    pkg = importlib.import_module(pkg_name)
    tool_info = {}
    help_info = {}
    for p in pkg.__path__:
        _scan(p, tool_info, help_info)
    return tool_info, help_info

def _scan(top, tool_info, help_info):
    import os, os.path, json
    for dirpath, dirnames, filenames in os.walk(top):
        # convert path into package reference
        tool_name = (NAMESPACE_PACKAGE
                + dirpath.replace(top, "").replace(os.sep, '.'))
        if TOOLINFO_FILENAME in filenames:
            ti_name = os.path.join(dirpath, TOOLINFO_FILENAME)
            with open(ti_name) as f:
                try:
                    ti_data = json.load(f)
                except ValueError:
                    # Skip bad tool information file
                    pass
                else:
                    tool_info[tool_name] = ti_data
        if HELP_DIRNAME in dirnames:
            help_dir = os.path.join(dirpath, HELP_DIRNAME)
            help_file = os.path.join(help_dir, "index.html")
            if os.path.exists(help_file):
                help_info[tool_name] = help_dir

if __name__ == "__main__":
    init()
