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
On macOS, the bin and MacOS entry points are hard links but existing launchers
hash their distinct paths. Record both lists when they name the same executable.

Each line is a file name.  On Linux, a memory-mapped file (shared library,
font, ICU data) is followed by a tab and the byte ranges this process
touched, "offset+length,offset+length,...", since startup uses only part of
most libraries: about 70 MB of the 200 MB of the application's files a
start maps.  Reading only those ranges, from several threads, made a cold
start ~120 ms faster than asking the kernel to read whole files ahead.

On Windows the DLLs come from EnumProcessModules and their touched ranges
from QueryWorkingSetEx.  A Windows DLL's ranges are offsets in its loaded
image, not in the file: the launcher prefetches them through an image
mapping.  Windows' own DLLs are left out, as are macOS's.
"""

import os
import sys

ENV_VAR = "CHIMERAX_STARTUP_PREFETCH"

# macOS system libraries are supplied by the dyld shared cache rather than
# ordinary individual library files. This list covers the app's separate files.
if sys.platform == "darwin":
    _SYSTEM_PREFIXES = ("/System/", "/usr/lib/")
elif sys.platform == "win32":
    # Windows' own DLLs are shared with every program and usually resident.
    _SYSTEM_PREFIXES = (os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows")) + os.sep,)
else:
    _SYSTEM_PREFIXES = ()


def _is_system(path):
    return bool(_SYSTEM_PREFIXES) and os.path.normcase(path).startswith(_SYSTEM_PREFIXES)

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

    On Windows only the DLLs loaded before the first extension module come
    first.  Each later DLL goes just before the first module of its package
    (Qt6Core.dll before PySide6, numpy.libs before numpy), and DLLs with no
    such module go last.  With every DLL first, the launcher's threads read
    ~80 MB of libraries before the Python modules needed at once, and a cold
    start in a Windows 11 VM was 0.5 s slower than without prefetching;
    in first-use order it was 0.75 s faster.
    """
    if images is None:
        images = loaded_images()
    if search_path is None:
        search_path = sys.path
    from importlib.util import cache_from_source

    files = []
    plugin_dirs = []
    windows = sys.platform == "win32"
    deferred = {}	# Windows: package key -> DLLs to read before its first module
    seen_extension = False
    for image in images:
        if not image or _is_system(image):
            continue
        if windows:
            if image.lower().endswith(".pyd"):
                # Extension modules are listed with the modules, in import order.
                seen_extension = True
            elif seen_extension:
                deferred.setdefault(_package_key(image, search_path), []).append(image)
            else:
                files.append(image)
        else:
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
        if windows:
            # Qt looks at its plugins when the application is created, after
            # Python and numpy start; read them with the other Qt files.
            for filename in _directory_files(directory):
                deferred.setdefault(_package_key(filename, search_path), []).append(filename)
        else:
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
        if deferred:
            files.extend(deferred.pop(_package_key(filename, search_path), ()))
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
    for libraries in deferred.values():
        files.extend(libraries)

    seen = set()
    result = []
    for filename in files:
        key = os.path.normcase(filename)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(filename):
            result.append(filename)
    return result


def _package_key(path, search_path):
    """The top-level package a file belongs to, for grouping a DLL with the
    modules that load it: (search path directory, "numpy") for numpy and
    numpy.libs, or (the file's directory, None) outside the search path."""
    path = os.path.normcase(os.path.abspath(path))
    best = None
    for directory in search_path:
        if not directory:
            continue
        directory = os.path.normcase(os.path.abspath(directory))
        # The longest match: site-packages, not the Lib directory above it.
        if path.startswith(directory + os.sep) and (best is None or len(directory) > len(best)):
            best = directory
    if best is not None:
        top = path[len(best) + 1:].split(os.sep, 1)[0]
        if top.endswith(".libs"):
            top = top[:-len(".libs")]
        if os.sep in path[len(best) + 1:]:
            return (best, top)
    directory = os.path.dirname(path)
    if os.path.basename(directory) == "__pycache__":
        directory = os.path.dirname(directory)
    return (directory, None)


def _with_sources(files):
    """Put each cached module's source file just before its .pyc, for the
    launcher to look up but not read: Python checks its size and time before
    using the .pyc, and on a cold start those metadata reads, one module
    at a time, took a Windows VM ~0.3 s.  Returns (files, set of sources).
    Windows only: on Linux these lookups are cheap even cold, and listing
    the sources made a cold start 10 +/- 9 ms slower."""
    if sys.platform != "win32":
        return files, set()
    result = []
    sources = set()
    for filename in files:
        directory, name = os.path.split(filename)
        if name.endswith(".pyc") and os.path.basename(directory) == "__pycache__":
            source = os.path.join(os.path.dirname(directory), name.split(".", 1)[0] + ".py")
            if source not in sources and os.path.isfile(source):
                result.append(source)
                sources.add(source)
        result.append(filename)
    return result, sources


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
    if sys.platform == "win32":
        return [path for path, base, size in _windows_modules()]
    return []


def _windows_modules():
    """[(path, base address, image size), ...] of this process's DLLs in load order."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process = wintypes.HANDLE(-1)	# GetCurrentProcess()
    enum = k32.K32EnumProcessModules
    enum.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD)]
    name = k32.K32GetModuleFileNameExW
    name.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]

    class MODULEINFO(ctypes.Structure):
        _fields_ = [("base", ctypes.c_void_p), ("size", wintypes.DWORD), ("entry", ctypes.c_void_p)]
    info = k32.K32GetModuleInformation
    info.argtypes = [wintypes.HANDLE, wintypes.HMODULE, ctypes.POINTER(MODULEINFO), wintypes.DWORD]
    count = 1024
    while True:
        handles = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not enum(process, handles, ctypes.sizeof(handles), ctypes.byref(needed)):
            return []
        if needed.value <= ctypes.sizeof(handles):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64
    modules = []
    buffer = ctypes.create_unicode_buffer(32768)
    for handle in handles[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        mi = MODULEINFO()
        if (name(process, handle, buffer, len(buffer))
                and info(process, handle, ctypes.byref(mi), ctypes.sizeof(mi))):
            modules.append((buffer.value, mi.base, mi.size))
    return modules


# Join touched ranges separated by less than this, to keep the list short.
_RANGE_GAP = 64 * 1024


def mapped_ranges():
    """Return {file name: [(offset, length), ...]} of the pages of each
    memory-mapped file that this process has touched (Linux only).

    /proc/self/pagemap says which pages of each mapping are present in this
    process, unlike mincore(), which also counts pages other processes or
    earlier sessions left in the page cache.
    """
    if sys.platform == "win32":
        try:
            return _windows_mapped_ranges()
        except Exception:
            return {}
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


def _windows_mapped_ranges():
    """Touched ranges of this process's non-system DLLs (Windows), as byte
    offsets in the loaded image rather than in the file.

    QueryWorkingSetEx says which pages of each image are in this process's
    working set.  The launcher maps each DLL as an image and prefetches
    these ranges of the view, which warms the image section the loader
    reuses.  Reading DLLs as file data warmed them only partly.
    """
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    page = 4096

    class WS(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("attributes", ctypes.c_size_t)]
    query = k32.K32QueryWorkingSetEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    process = wintypes.HANDLE(-1)
    ranges = {}
    for path, base, size in _windows_modules():
        if _is_system(path) or not base or not size:
            continue
        npages = (size + page - 1) // page
        entries = (WS * npages)()
        for i in range(npages):
            entries[i].address = base + i * page
        if not query(process, entries, ctypes.sizeof(entries)):
            continue
        pages = [i for i in range(npages) if entries[i].attributes & 1]	# Valid
        if pages:
            ranges[path] = _page_spans(pages, page)
    return ranges


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


def _list_text(files, ranges, metadata_only=()):
    lines = []
    for f in files:
        spans = ranges.get(f)
        if f in metadata_only:
            # A tab and no ranges: the launcher opens the file, reads nothing.
            lines.append(f + "\t")
        elif spans:
            lines.append(f + "\t" + ",".join("%d+%d" % span for span in spans))
        else:
            lines.append(f)
    return "".join(line + "\n" for line in lines)


def _launcher_list_name(executable):
    # Match the launcher's FNV-1a hash of realpath(executable).
    value = 0xcbf29ce484222325
    for byte in os.fsencode(os.path.realpath(executable)):
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return 'startup-prefetch-%016x' % value


def _list_destinations(path):
    """Keep both hard-linked macOS launchers supplied with the same list.

    Existing launchers key their list by pathname, so the bin and MacOS
    entry points use different files even when they are the same executable.
    Mirror only the launcher's default cache path, never a caller's override.
    """
    paths = [path]
    if sys.platform != 'darwin':
        return paths
    executable = os.path.realpath(sys.executable)
    parent = os.path.dirname(executable)
    entry = os.path.basename(parent)
    if entry not in ('bin', 'MacOS') or os.path.basename(executable) != 'ChimeraX':
        return paths
    cache_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Caches', 'ChimeraX')
    if os.path.abspath(path) != os.path.join(cache_dir, _launcher_list_name(executable)):
        return paths
    other = os.path.join(os.path.dirname(parent),
                        'MacOS' if entry == 'bin' else 'bin', 'ChimeraX')
    try:
        if not os.path.samefile(executable, other):
            return paths
    except OSError:
        return paths
    alternate = os.path.join(cache_dir, _launcher_list_name(other))
    if alternate != path:
        paths.append(alternate)
    return paths


def _write_list_text(path, text):
    try:
        with open(path, encoding='utf-8') as f:
            if f.read() == text:
                return
    except (OSError, UnicodeDecodeError):
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Rename each complete list so a concurrent launcher cannot read a partial one.
    temp = "%s.%d.tmp" % (path, os.getpid())
    # LF line endings on every platform; the launcher splits lines on LF.
    with open(temp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    os.replace(temp, path)


def _write_list(path, modules, opened=()):
    try:
        files, sources = _with_sources(startup_files(modules, opened=opened))
        text = _list_text(files, mapped_ranges(), sources)
        paths = _list_destinations(path)
    except Exception:
        # Prefetching is only an optimization.
        return
    for destination in paths:
        try:
            _write_list_text(destination, text)
        except Exception:
            # Failure to update one entry point must not suppress the other.
            pass
