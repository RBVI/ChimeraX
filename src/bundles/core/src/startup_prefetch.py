# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
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

"""
startup_prefetch: remember the files a startup reads
====================================================

The ChimeraX launcher names a per-executable list file in the
CHIMERAX_STARTUP_PREFETCH environment variable and, when the list exists,
asks the kernel to read those files ahead while Python starts.  This module
writes that list: the shared libraries, Python modules, and data files read
by the time the GUI is up, roughly in the order they were first needed.

Each line is a file name.  On Linux, a memory-mapped file (shared library,
font, ICU data) is followed by a tab and the byte ranges this process
touched, "offset+length,offset+length,...", since startup uses only part of
most libraries: about 70 MB of the 200 MB of the application's files a
start maps.  Reading only those ranges, from several threads, made a cold
start ~120 ms faster than asking the kernel to read whole files ahead.
"""

import os
import sys

ENV_VAR = "CHIMERAX_STARTUP_PREFETCH"

# macOS loads its own libraries from the dyld shared cache, which is always
# resident, so only list files that are read from disk.
_SYSTEM_PREFIXES = ("/System/", "/usr/lib/") if sys.platform == "darwin" else ()

# Qt reads the metadata of every plugin in these directories at startup,
# including plugins it never loads.
_QT_PLUGIN_DIRS = ("platforms", "imageformats", "iconengines", "styles")

# A startup script might open a large data file; don't read that ahead.
_MAX_DATA_FILE_SIZE = 16 * 1024 * 1024

_opened = None


def watch_opened_files():
    """Start remembering the files Python opens, until the list is recorded.

    Settings, toolshed caches, and histories are read with open(), which an
    audit hook sees.  Files Qt reads from C++ (icons, plugins) are found
    from the modules and libraries that were loaded instead.
    """
    global _opened
    if _opened is not None or not os.environ.get(ENV_VAR):
        return
    _opened = []
    opened = _opened

    def audit(event, args, opened=opened):
        if event == "open" and opened is _opened:
            path = args[0]
            if isinstance(path, (str, bytes)):
                opened.append(path)

    sys.addaudithook(audit)


def record_startup_files():
    """Write the list of files this startup used, if the launcher asked.

    The module list is taken now, on the calling (main) thread.  Finding the
    files and writing the list happens on a background thread.
    """
    global _opened
    path = os.environ.get(ENV_VAR)
    opened = _opened or []
    # Stop collecting: later opens are the user's work, not startup.
    _opened = None
    if not path:
        return None
    modules = [(name, getattr(m, "__file__", None)) for name, m in list(sys.modules.items())]
    import threading

    thread = threading.Thread(
        target=_write_list, args=(path, modules, opened), daemon=True,
        name="startup prefetch list")
    thread.start()
    return thread


def startup_files(modules, images=None, opened=(), search_path=None):
    """Return the files to prefetch, in a useful order without duplicates.

    Shared libraries come first since the earliest are needed before any
    Python module, then the .pth files site reads, Qt plugins, compiled
    Python modules, the data files Python opened, and the icons of the
    ChimeraX packages that were imported.  'modules' is a sequence of
    (module name, file name) in import order.
    """
    if images is None:
        images = loaded_images()
    if search_path is None:
        search_path = sys.path
    from importlib.util import cache_from_source

    files = []
    plugin_dirs = []
    for image in images:
        if not image or image.startswith(_SYSTEM_PREFIXES):
            continue
        files.append(image)
        parent = os.path.dirname(image)
        if os.path.basename(os.path.dirname(parent)) == "plugins":
            plugins = os.path.dirname(parent)
            for name in _QT_PLUGIN_DIRS:
                plugin_dirs.append(os.path.join(plugins, name))
            plugin_dirs.append(parent)
    for directory in search_path:
        files.extend(_directory_files(directory, ".pth"))
    for directory in plugin_dirs:
        files.extend(_directory_files(directory))
    package_dirs = []
    for name, filename in modules:
        if not filename:
            continue
        if filename.endswith(".py"):
            if name.startswith("chimerax.") and os.path.basename(filename) == "__init__.py":
                package_dirs.append(os.path.dirname(filename))
            try:
                filename = cache_from_source(filename)
            except (NotImplementedError, ValueError):
                continue
        files.append(filename)
    for filename in opened:
        filename = os.fsdecode(filename)
        if os.path.isabs(filename):
            try:
                if os.path.getsize(filename) <= _MAX_DATA_FILE_SIZE:
                    files.append(filename)
            except OSError:
                pass
    for directory in package_dirs:
        files.extend(_directory_files(os.path.join(directory, "icons")))

    seen = set()
    result = []
    for filename in files:
        if filename in seen:
            continue
        seen.add(filename)
        if os.path.isfile(filename):
            result.append(filename)
    return result


def _directory_files(directory, suffix=""):
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [os.path.join(directory, name) for name in names if name.endswith(suffix)]


def loaded_images():
    """Return the paths of loaded shared libraries in load order."""
    if sys.platform == "darwin":
        import ctypes

        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        names = (dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count()))
        return [os.fsdecode(name) for name in names if name]
    if sys.platform.startswith("linux"):
        images = []
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    fields = line.split(None, 5)
                    if len(fields) == 6 and fields[5].startswith("/"):
                        images.append(fields[5].rstrip("\n"))
        except OSError:
            pass
        return images
    return []


# Join touched ranges separated by less than this, to keep the list short.
_RANGE_GAP = 64 * 1024


def mapped_ranges():
    """Return {file name: [(offset, length), ...]} of the pages of each
    memory-mapped file that this process has touched (Linux only).

    /proc/self/pagemap says which pages of each mapping are present in this
    process, unlike mincore(), which also counts pages other processes or
    earlier sessions left in the page cache.
    """
    if not sys.platform.startswith("linux"):
        return {}
    page = os.sysconf("SC_PAGE_SIZE")
    pages = {}
    try:
        with open("/proc/self/maps") as maps, open("/proc/self/pagemap", "rb") as pagemap:
            for line in maps:
                fields = line.split(None, 5)
                if len(fields) < 6 or not fields[5].startswith("/"):
                    continue
                name = fields[5].rstrip("\n")
                if name.endswith(" (deleted)"):
                    continue
                start, end = (int(a, 16) for a in fields[0].split("-"))
                offset = int(fields[2], 16)
                npages = (end - start) // page
                pagemap.seek((start // page) * 8)
                entries = pagemap.read(npages * 8)
                present = pages.setdefault(name, set())
                first = offset // page
                # The top byte of each 64-bit entry: bit 63 present, bit 62 swapped.
                present.update(first + i for i, b in enumerate(entries[7::8]) if b & 0xC0)
    except OSError:
        return {}
    return {name: _page_spans(present, page) for name, present in pages.items() if present}


def _page_spans(pages, page_size, gap=_RANGE_GAP):
    """Byte ranges [(offset, length), ...] covering the given page numbers,
    joining ranges separated by less than 'gap' bytes."""
    spans = []
    for p in sorted(pages):
        o = p * page_size
        if spans and o - (spans[-1][0] + spans[-1][1]) < gap:
            spans[-1][1] = o + page_size - spans[-1][0]
        else:
            spans.append([o, page_size])
    return [tuple(span) for span in spans]


def _list_text(files, ranges):
    lines = []
    for f in files:
        spans = ranges.get(f)
        if spans:
            lines.append(f + "\t" + ",".join("%d+%d" % span for span in spans))
        else:
            lines.append(f)
    return "".join(line + "\n" for line in lines)


def _write_list(path, modules, opened=()):
    try:
        text = _list_text(startup_files(modules, opened=opened), mapped_ranges())
        try:
            with open(path, encoding="utf-8") as f:
                if f.read() == text:
                    return
        except (OSError, UnicodeDecodeError):
            pass
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write a new file and rename it so the launcher never reads part of a list.
        temp = "%s.%d.tmp" % (path, os.getpid())
        with open(temp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(temp, path)
    except Exception:
        # Prefetching is only an optimization.
        pass
