#
# vi:set expandtab shiftwidth=4:
#
# Remove the parts of PySide6 that ChimeraX does not use.
#
# PySide6 ships every Qt module, plus developer tools, QML modules and
# translations.  ChimeraX and the bundles on the Toolshed only use a handful
# of modules, so we delete the rest after pip installs PySide6.
#
# To support another Qt module, add its Python binding to KEEP_BINDINGS.
# If it needs Qt libraries that are not in KEEP_LIBS, the check at the end
# of this script fails and lists them.
#
# Usage: python prune_pyside6.py [SITE_PACKAGES_DIR]
#
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

# Python binding modules to keep
KEEP_BINDINGS = {
    "QtCore", "QtGui", "QtWidgets",
    "QtNetwork",
    "QtMultimedia",                             # webcam bundle
    "QtWebEngineCore", "QtWebEngineWidgets",    # help viewer, HTML tools
    "QtWebChannel",                             # imported by QtWebEngine bindings
    "QtSvg", "QtPrintSupport",                  # qtconsole (shell tool)
    "QtOpenGLWidgets", "QtTest",                # optional in pyqtgraph (Toolshed)
}

# Qt libraries needed by the kept bindings, plugins and QtWebEngineProcess,
# without their platform decoration: QtCore.framework (macOS),
# libQt6Core.so.6 (Linux), Qt6Core.dll (Windows).
KEEP_LIBS = {
    "Concurrent", "Core", "DBus", "Gui", "Multimedia", "Network", "OpenGL",
    "OpenGLWidgets", "Positioning", "PrintSupport", "Qml", "QmlMeta",
    "QmlModels", "QmlWorkerScript", "Quick", "QuickWidgets", "Svg", "Test",
    "WebChannel", "WebEngineCore", "WebEngineWidgets", "Widgets",
    # Linux windowing systems
    "XcbQpa", "WaylandClient", "WlShellIntegration",
    "EglFSDeviceIntegration", "EglFsKmsSupport", "EglFsKmsGbmSupport",
}

# Plugin directories to keep; the others are only used by removed modules
KEEP_PLUGIN_DIRS = {
    "platforms", "platformthemes", "platforminputcontexts", "styles",
    "imageformats", "iconengines", "generic", "tls", "networkinformation",
    "multimedia", "printsupport", "permissions",
    # Linux windowing systems
    "xcbglintegrations", "egldeviceintegrations",
    "wayland-decoration-client", "wayland-graphics-integration-client",
    "wayland-shell-integration",
}

# Plugins in kept directories that need removed libraries
REMOVE_PLUGINS = re.compile(r"virtualkeyboard|qpdf", re.IGNORECASE)

# Developer tools, whether a bare executable, .exe or macOS .app
TOOLS = {
    "assistant", "designer", "linguist", "lrelease", "lupdate", "balsam",
    "balsamui", "qmlformat", "qmllint", "qmlls", "qsb", "svgtoqml", "rcc",
    "uic", "qmlcachegen", "qmlimportscanner", "qmltyperegistrar",
}

# Files only needed to build against PySide6
DEV_DIRS = {"include", "typesystems", "glue", "doc", "lib"}

QT_LIB_PATTERNS = (
    re.compile(r"Qt(\w+)\.framework"),      # macOS
    re.compile(r"libQt6(\w+)\.so[.\d]*"),   # Linux
    re.compile(r"Qt6(\w+)\.dll"),           # Windows
)


def qt_lib_name(filename):
    """Return the Qt library name ("Core") for a library filename, or None"""
    for pattern in QT_LIB_PATTERNS:
        m = pattern.fullmatch(filename)
        if m:
            return m.group(1)
    return None


def binding_name(filename):
    """Return the module name ("QtCore") for a binding filename, or None"""
    m = re.fullmatch(r"(Qt\w+)\.(abi3\.so|pyd|pyi)", filename)
    return m.group(1) if m else None


def qt_dir(pyside_dir):
    """Directory holding Qt's lib, plugins, qml... (PySide6 itself on Windows)"""
    d = os.path.join(pyside_dir, "Qt")
    return d if os.path.isdir(d) else pyside_dir


def lib_dir(pyside_dir):
    """Directory holding the Qt libraries (PySide6 itself on Windows)"""
    d = os.path.join(pyside_dir, "Qt", "lib")
    return d if os.path.isdir(d) else pyside_dir


def duplicate_libs(directory):
    """Versioned copies of shared libraries whose soname copy also exists.

    Wheels cannot contain symbolic links, so libavcodec.61.dylib and
    libavcodec.61.19.101.dylib (or libavcodec.so, .so.61, .so.61.19.101)
    are full copies.  The libraries are linked by the soname copy.
    """
    files = set(os.listdir(directory))
    dups = []
    for fn in files:
        m = re.fullmatch(r"(lib.+?)\.(\d+)(\.[\d.]+)\.dylib", fn)
        if m:
            soname = "%s.%s.dylib" % (m.group(1), m.group(2))
        else:
            m = re.fullmatch(r"(lib.+?\.so)(\.\d+)?([\d.]*)", fn)
            if not m or (m.group(2) and not m.group(3)):
                continue
            if m.group(2):
                soname = m.group(1) + m.group(2)
            else:
                # libfoo.so development link; find libfoo.so.N
                sonames = [f for f in files
                           if re.fullmatch(re.escape(fn) + r"\.\d+", f)]
                if len(sonames) != 1:
                    continue
                soname = sonames[0]
        if soname in files and _same_size(directory, fn, soname):
            dups.append(fn)
    return dups


def _same_size(directory, f1, f2):
    return (os.path.getsize(os.path.join(directory, f1))
            == os.path.getsize(os.path.join(directory, f2)))


def removals(pyside_dir, machine=None):
    """Return the paths under pyside_dir to remove"""
    remove = []
    qt = qt_dir(pyside_dir)
    libs = lib_dir(pyside_dir)

    for fn in os.listdir(pyside_dir):
        path = os.path.join(pyside_dir, fn)
        name = binding_name(fn)
        stem = os.path.splitext(fn)[0].lower()
        if name is not None:
            if name not in KEEP_BINDINGS:
                remove.append(path)
        elif stem in TOOLS or fn in DEV_DIRS:
            remove.append(path)
        elif fn.startswith(("libpyside6qml", "pyside6qml")):
            # only used by the QtQml and QtQuick bindings
            remove.append(path)
        elif fn.endswith(".lib"):
            # Windows import libraries
            remove.append(path)

    for fn in os.listdir(libs):
        name = qt_lib_name(fn)
        if name is not None and name not in KEEP_LIBS:
            remove.append(os.path.join(libs, fn))
    remove.extend(os.path.join(libs, fn) for fn in duplicate_libs(libs))

    plugins = os.path.join(qt, "plugins")
    if os.path.isdir(plugins):
        for d in os.listdir(plugins):
            path = os.path.join(plugins, d)
            if d not in KEEP_PLUGIN_DIRS:
                remove.append(path)
            elif os.path.isdir(path):
                remove.extend(os.path.join(path, fn) for fn in os.listdir(path)
                              if REMOVE_PLUGINS.search(fn))

    for d in ("qml", "metatypes"):
        path = os.path.join(qt, d)
        if os.path.isdir(path):
            remove.append(path)

    # Translations of Qt's own strings, which ChimeraX does not load.
    # QtWebEngine's locale .pak files in qtwebengine_locales are needed.
    translations = os.path.join(qt, "translations")
    if os.path.isdir(translations):
        remove.extend(os.path.join(translations, fn)
                      for fn in os.listdir(translations) if fn.endswith(".qm"))

    libexec = os.path.join(qt, "libexec")
    if os.path.isdir(libexec):
        remove.extend(os.path.join(libexec, fn) for fn in os.listdir(libexec)
                      if fn.lower() in TOOLS)

    # QtWebEngine resources for debug builds of Qt (Windows)
    resources = os.path.join(qt, "resources")
    if os.path.isdir(resources):
        remove.extend(os.path.join(resources, fn) for fn in os.listdir(resources)
                      if re.search(r"\.debug\.(pak|bin)$", fn))

    # macOS QtWebEngine has a V8 snapshot for each architecture
    resources = os.path.join(libs, "QtWebEngineCore.framework", "Resources")
    if machine and os.path.isdir(resources):
        for fn in os.listdir(resources):
            m = re.fullmatch(r"v8_context_snapshot\.(\w+)\.bin", fn)
            if m and m.group(1) != machine:
                remove.append(os.path.join(resources, fn))

    return remove


def remove_paths(paths):
    """Remove files and directories, returning the number of bytes freed"""
    freed = 0
    for path in paths:
        if os.path.isdir(path) and not os.path.islink(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    fp = os.path.join(root, fn)
                    if not os.path.islink(fp):
                        freed += os.path.getsize(fp)
            shutil.rmtree(path)
        elif os.path.lexists(path):
            if not os.path.islink(path):
                freed += os.path.getsize(path)
            os.remove(path)
    return freed


def bindings_present(pyside_dir):
    """Names of the binding modules in pyside_dir"""
    names = {binding_name(fn) for fn in os.listdir(pyside_dir)
             if fn.endswith((".abi3.so", ".pyd"))}
    names.discard(None)
    return names


def libs_present(pyside_dir):
    """Names of the Qt libraries in pyside_dir, plus "pyside6qml" """
    names = {qt_lib_name(fn) for fn in os.listdir(lib_dir(pyside_dir))}
    names.discard(None)
    if any(fn.startswith(("libpyside6qml", "pyside6qml"))
           for fn in os.listdir(pyside_dir)):
        names.add("pyside6qml")
    return names


def check_bindings(pyside_dir, removed):
    """Return {binding: removed bindings it imports}"""
    problems = {}
    for name in bindings_present(pyside_dir):
        for ext in (".abi3.so", ".pyd"):
            path = os.path.join(pyside_dir, name + ext)
            if os.path.exists(path):
                break
        with open(path, "rb") as f:
            data = f.read()
        imported = {m.decode() for m in
                    re.findall(rb"PySide6\.(Qt\w+)\x00", data)}
        missing = imported & removed
        if missing:
            problems[name] = missing
    return problems


def _is_binary(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    return (magic[:2] == b"MZ" or magic == b"\x7fELF"
            or magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                         b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"))


def linked_qt_libs(path):
    """Qt library names (and "pyside6qml") a binary links to,
    or None if there is no tool to list them"""
    if sys.platform == "darwin":
        cmd = ["otool", "-L", path]
        pattern = r"@rpath/(?:Qt(\w+)\.framework/|lib(pyside6qml)\.)"
    else:
        if shutil.which("objdump") is None:
            return None
        cmd = ["objdump", "-p", path]
        pattern = (r"NEEDED\s+lib(?:Qt6(\w+)\.so|(pyside6qml)\.)"
                   r"|DLL Name: (?:Qt6(\w+)\.dll|(pyside6qml)\.)")
    out = subprocess.run(cmd, capture_output=True, text=True,
                         errors="replace").stdout
    return {next(g for g in m.groups() if g)
            for m in re.finditer(pattern, out)}


def check_libs(pyside_dir, removed):
    """Return {binary: removed Qt libraries it links to}

    Only removed libraries count: some plugins in the wheels already link
    to libraries the wheels do not include."""
    problems = {}
    for root, _, files in os.walk(pyside_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if os.path.islink(path) or not _is_binary(path):
                continue
            linked = linked_qt_libs(path)
            if linked is None:
                print("prune_pyside6: objdump not found, not checking"
                      " Qt library dependencies")
                return {}
            missing = linked & removed
            if missing:
                problems[os.path.relpath(path, pyside_dir)] = missing
    return problems


def prune(site_packages):
    pyside_dir = os.path.join(site_packages, "PySide6")
    if not os.path.isdir(pyside_dir):
        print("prune_pyside6: no PySide6 in %s" % site_packages)
        return True
    bindings = bindings_present(pyside_dir)
    libs = libs_present(pyside_dir)
    machine = platform.machine() if sys.platform == "darwin" else None
    freed = remove_paths(removals(pyside_dir, machine))
    print("prune_pyside6: removed %.0f MB" % (freed / 1e6))
    removed_bindings = bindings - bindings_present(pyside_dir)
    removed_libs = libs - libs_present(pyside_dir)

    ok = True
    problems = check_bindings(pyside_dir, removed_bindings)
    for name, missing in sorted(problems.items()):
        print("prune_pyside6: %s imports removed bindings: %s;"
              " add them to KEEP_BINDINGS" % (name, " ".join(sorted(missing))),
              file=sys.stderr)
        ok = False
    problems = check_libs(pyside_dir, removed_libs)
    for path, missing in sorted(problems.items()):
        print("prune_pyside6: %s links to removed Qt libraries: %s;"
              " add them to KEEP_LIBS" % (path, " ".join(sorted(missing))),
              file=sys.stderr)
        ok = False
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        site_packages = sys.argv[1]
    else:
        site_packages = sysconfig.get_path("purelib")
    raise SystemExit(0 if prune(site_packages) else 1)
