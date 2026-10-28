#
# vi:set expandtab shiftwidth=4:
#
# Post-install cleanup up of application directory
#
import sys
import os
import platform
import re
import shutil
import subprocess
import sysconfig
from concurrent.futures import ThreadPoolExecutor


def clean_app(chimerax_root, keep_debug_info=False):
    """Clean application tree for distribution

    Returns whether it succeeded: prune_pyside6 fails if a Qt binding or
    library it removed turns out to be used.
    """
    site_packages = sysconfig.get_path("purelib")
    # Each of these removes a different part of the tree, so they run at once
    with ThreadPoolExecutor() as pool:
        pruned = pool.submit(prune_pyside6, site_packages)
        steps = [
            pool.submit(remove_package_tests, site_packages),
            pool.submit(remove_tkinter, sys.base_prefix,
                        sysconfig.get_path("stdlib"),
                        sysconfig.get_path("scripts")),
            pool.submit(remove_babel_locales, site_packages),
            pool.submit(remove_unused_pyopengl, site_packages),
        ]
        for step in steps:
            step.result()
        ok = pruned.result()
    # These look at every file, so they wait for the removals
    remove_pycache(chimerax_root)
    with ThreadPoolExecutor() as pool:
        steps = [pool.submit(remove_foreign_binaries, chimerax_root,
                             sys.platform, platform.machine())]
        if sys.platform == "darwin":
            # On macOS, remove_foreign_binaries only removes files that are
            # not Mach-O, and these are only Mach-O
            steps.append(pool.submit(thin_macos_binaries, chimerax_root,
                                     platform.machine()))
        for step in steps:
            step.result()
    # After remove_foreign_binaries, so other platforms' ELF files are gone
    if sys.platform == "linux" and not keep_debug_info:
        strip_debug_info(chimerax_root)
    return ok
    # TODO: figure this out
    #   removing scripts makes multiple "make installs" without
    #   a "make clean" inbetween fail and removes scripts that
    #   would be needed for developers to make documentation
    # remove_scripts(chimerax_root)


def prune_pyside6(site_packages):
    """Remove the Qt modules, tools and data ChimeraX does not use, with
    prereqs/pips/prune_pyside6.py, returning whether it succeeded

    It runs here rather than when PySide6 is installed, since nothing that
    runs before the end of the build depends on it.
    """
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "prereqs", "pips", "prune_pyside6.py")
    spec = importlib.util.spec_from_file_location("prune_pyside6", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prune(site_packages)


# PyOpenGL bindings nothing uses: OpenGL ES, GLUT, GLE, and the Tk widget
# (tkinter is removed too).  ChimeraX draws through chimerax.graphics.gl and
# uses PyOpenGL only for desktop GL, EGL, GLX, WGL and OSMesa.  PyOpenGL's
# platform code loads the GLES and GLUT shared libraries on demand but never
# imports these packages.
UNUSED_PYOPENGL = (
    "GLES1", "GLES2", "GLES3", "GLUT", "GLE", "Tk", "DLLS",
    os.path.join("raw", "GLES1"), os.path.join("raw", "GLES2"),
    os.path.join("raw", "GLES3"), os.path.join("raw", "GLUT"),
    os.path.join("raw", "GLE"),
)


def remove_unused_pyopengl(site_packages):
    """Remove the parts of PyOpenGL that ChimeraX does not use"""
    opengl = os.path.join(site_packages, "OpenGL")
    for name in UNUSED_PYOPENGL:
        shutil.rmtree(os.path.join(opengl, name), ignore_errors=True)


def remove_pycache(chimerax_root):
    """remove unwanted __pycache__ directories"""
    # cleanup -- remove __pycache__ directories
    cache_dirs = []
    for root, dirs, files in os.walk(chimerax_root):
        if "__pycache__" in dirs:
            cache_dirs.append(os.path.join(root, "__pycache__"))
    for d in cache_dirs:
        shutil.rmtree(d)


TEST_DIRS = ("tests", "test")


def imported_at_load(package_dir, name):
    """Check whether a module in package_dir imports its 'name' subpackage
    at the top level (e.g., tables/__init__.py imports tables.tests)"""
    pattern = re.compile(r"^(from \.%s\b|from \. import .*\b%s\b)" % (name, name),
                         re.MULTILINE)
    for fn in os.listdir(package_dir):
        if not fn.endswith(".py"):
            continue
        with open(os.path.join(package_dir, fn), encoding="utf-8",
                  errors="replace") as f:
            if pattern.search(f.read()):
                return True
    return False


def has_stubs(path):
    """Type stubs (e.g., jedi's typeshed copy of django.test) describe real
    modules and are not tests"""
    return any(fn.endswith(".pyi") for fn in os.listdir(path))


def remove_package_tests(site_packages):
    """Remove third-party packages' test suites"""
    test_dirs = []
    for root, dirs, files in os.walk(site_packages):
        if root == site_packages and "chimerax" in dirs:
            # ChimeraX bundles, including chimerax.test, are not third-party
            dirs.remove("chimerax")
        for name in TEST_DIRS:
            if name not in dirs:
                continue
            dirs.remove(name)
            path = os.path.join(root, name)
            if os.path.islink(path) or imported_at_load(root, name) \
                    or has_stubs(path):
                continue
            test_dirs.append(path)
    for d in test_dirs:
        shutil.rmtree(d)
    return test_dirs


# The standard library's Tk GUI modules
TK_MODULES = ("tkinter", "idlelib", "turtledemo", "turtle.py")
# _tkinter extension module, in lib-dynload or (Windows) DLLs
TK_EXTENSION = re.compile(r"_tkinter\b.*\.(so|pyd)$")
# Tcl/Tk libraries and script directories: libtcl9.0.dylib, libtcl9tk9.0.dylib,
# tcl9.0, tk9.0, itcl4.3.8, thread3.0.6 (macOS, Linux) and tcl86t.dll,
# tk86t.dll (Windows, whose scripts are all in a tcl directory)
TCL_LIBS = re.compile(r"(lib)?(tcl|tk|itcl|thread|tdbc\w*)\d.*")
TCL_DLLS = re.compile(r"(tcl|tk)\d+t?\.dll", re.IGNORECASE)


def remove_tkinter(prefix, stdlib, scripts):
    """Remove tkinter, IDLE, turtle and the Tcl/Tk they use

    ChimeraX's GUI is Qt.  The Tk dialogs left in Segger are from Chimera
    and are never shown.
    """
    paths = [os.path.join(stdlib, name) for name in TK_MODULES]
    for d in (os.path.join(stdlib, "lib-dynload"), os.path.join(prefix, "DLLs")):
        if os.path.isdir(d):
            paths.extend(os.path.join(d, fn) for fn in os.listdir(d)
                         if TK_EXTENSION.match(fn) or TCL_DLLS.fullmatch(fn))
    lib = os.path.join(prefix, "lib")
    if os.path.isdir(lib):
        paths.extend(os.path.join(lib, fn) for fn in os.listdir(lib)
                     if TCL_LIBS.fullmatch(fn))
    paths.append(os.path.join(prefix, "tcl"))
    if os.path.isdir(scripts):
        paths.extend(os.path.join(scripts, fn) for fn in os.listdir(scripts)
                     if fn.startswith("idle"))
    freed = 0
    for path in paths:
        if os.path.islink(path) or os.path.isfile(path):
            freed += os.lstat(path).st_size
            os.remove(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                freed += sum(os.lstat(os.path.join(root, fn)).st_size
                             for fn in files)
            shutil.rmtree(path)
    print(f"clean_app: removed {freed / 1e6:.0f} MB of tkinter and Tcl/Tk")
    return freed


# Babel's CLDR data for English and the root locale it inherits from
BABEL_KEEP_LOCALES = re.compile(r"(root|en|en_\w+)\.dat")


def remove_babel_locales(site_packages):
    """Remove Babel's locale data for languages other than English

    Only Sphinx uses Babel, to build ChimeraX's and some Toolshed bundles'
    (e.g., ISOLDE's) English documentation.  The other ~1000 locales are 29 MB.
    """
    locale_dir = os.path.join(site_packages, "babel", "locale-data")
    if not os.path.isdir(locale_dir):
        return 0
    freed = 0
    for fn in os.listdir(locale_dir):
        if not fn.endswith(".dat") or BABEL_KEEP_LOCALES.fullmatch(fn):
            continue
        path = os.path.join(locale_dir, fn)
        freed += os.lstat(path).st_size
        os.remove(path)
    print(f"clean_app: removed {freed / 1e6:.0f} MB of Babel locale data")
    return freed


ELF_MAGIC = b"\x7fELF"
MACHO_MAGICS = (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
                b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe")
FAT_MACHO_MAGICS = (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf")
# ELF e_machine values for the architectures ChimeraX is built for
ELF_MACHINES = {"x86_64": 62, "amd64": 62, "aarch64": 183, "arm64": 183}


def binary_format(path):
    """Return ("elf", is_64bit, e_machine), ("macho",), ("pe",) or None"""
    try:
        with open(path, "rb") as f:
            header = f.read(64)
            if header.startswith(ELF_MAGIC) and len(header) >= 20:
                order = "little" if header[5] == 1 else "big"
                return ("elf", header[4] == 2,
                        int.from_bytes(header[18:20], order))
            if header[:4] in MACHO_MAGICS:
                return ("macho",)
            # Java class files share the fat Mach-O magic number, but their
            # next four bytes are a version >= 45 instead of an arch count
            if header[:4] in FAT_MACHO_MAGICS \
                    and 0 < int.from_bytes(header[4:8], "big") < 45:
                return ("macho",)
            if header.startswith(b"MZ") and len(header) == 64:
                f.seek(int.from_bytes(header[60:64], "little"))
                if f.read(4) == b"PE\0\0":
                    return ("pe",)
    except OSError:
        pass
    return None


def is_foreign_binary(fmt, system, machine):
    """Check whether a binary_format() result cannot run on this platform"""
    if fmt is None:
        return False
    kind = fmt[0]
    if system == "darwin":
        # thin_macos_binaries() handles other architectures' Mach-O code
        return kind != "macho"
    if system == "win32":
        # Other architectures' PE files include launchers that setuptools
        # and pip pick from at run time, so keep them
        return kind != "pe"
    if kind != "elf":
        return True
    _, is_64bit, e_machine = fmt
    host_machine = ELF_MACHINES.get(machine.lower())
    return not is_64bit or (host_machine is not None
                            and e_machine != host_machine)


def remove_foreign_binaries(chimerax_root, system, machine):
    """Remove executables and libraries built for other platforms

    Wheels like PyOpenGL, openvr, pyopenxr and ChimeraX's own looking_glass
    ship every platform's libraries (and pip and setuptools ship Windows
    launchers), but only the current platform's can load.  Directories
    left empty are removed too.
    """
    freed = 0
    emptied = set()
    for root, dirs, files in os.walk(chimerax_root):
        for fn in files:
            path = os.path.join(root, fn)
            if os.path.islink(path):
                continue
            if is_foreign_binary(binary_format(path), system, machine):
                freed += os.lstat(path).st_size
                os.remove(path)
                emptied.add(root)
    # Deepest first, so a parent is only checked after its children
    for d in sorted(emptied, key=len, reverse=True):
        while d != chimerax_root and os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            d = os.path.dirname(d)
    print(f"clean_app: removed {freed / 1e6:.0f} MB of other platforms' binaries")
    return freed


def is_fat_macho(path):
    """Check for a universal binary's magic number (also used by Java class
    files, so confirm with lipo)"""
    try:
        with open(path, "rb") as f:
            return f.read(4) in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf")
    except OSError:
        # Such as a file remove_foreign_binaries has removed meanwhile
        return False


def thin_macos_binaries(chimerax_root, arch):
    """Remove other architectures from universal binaries

    The Intel and Arm builds are merged into a universal application later,
    so each build only needs its own architecture.  Wheels like PySide6's
    are universal2, which roughly doubles their size.
    """
    paths = []
    for root, dirs, files in os.walk(chimerax_root):
        for fn in files:
            path = os.path.join(root, fn)
            if not os.path.islink(path) and is_fat_macho(path):
                paths.append(path)

    def thin(path):
        """Return the number of bytes freed"""
        result = subprocess.run(["lipo", "-archs", path],
                                capture_output=True, text=True)
        if result.returncode != 0:
            return 0
        if arch not in result.stdout.split():
            print(f"clean_app: no {arch} code in {path}", file=sys.__stderr__)
            return 0
        size = os.path.getsize(path)
        thin_path = path + ".thin"
        subprocess.run(["lipo", "-thin", arch, path, "-output", thin_path],
                       check=True)
        shutil.copymode(path, thin_path)
        os.replace(thin_path, path)
        return size - os.path.getsize(path)

    # Two lipo runs per binary, one binary per CPU at a time
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        freed = sum(pool.map(thin, paths))
    print(f"clean_app: removed {freed / 1e6:.0f} MB of non-{arch} code")
    return freed


def elf_files(chimerax_root):
    """Return the ELF files under chimerax_root, as a list of each file's
    hard links"""
    links = {}
    for root, dirs, files in os.walk(chimerax_root):
        for fn in files:
            path = os.path.join(root, fn)
            if os.path.islink(path):
                continue
            fmt = binary_format(path)
            if fmt is None or fmt[0] != "elf":
                continue
            st = os.stat(path)
            links.setdefault((st.st_dev, st.st_ino), []).append(path)
    return list(links.values())


STRIP_SUFFIX = ".clean_app-strip"
PT_LOAD = 1


def load_segments_aligned(path):
    """Whether each PT_LOAD segment of an ELF file is one the dynamic loader
    accepts, its address and file offset agreeing modulo its alignment

    glibc refuses anything else with "ELF load command address/offset not
    properly aligned".
    """
    import struct
    with open(path, "rb") as f:
        header = f.read(64)
        if not header.startswith(ELF_MAGIC) or len(header) < 52:
            return False
        is_64bit = header[4] == 2
        order = "<" if header[5] == 1 else ">"
        if is_64bit:
            phoff, = struct.unpack_from(order + "Q", header, 32)
            phentsize, phnum = struct.unpack_from(order + "HH", header, 54)
            # p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz,
            # p_memsz, p_align
            fields, offset, vaddr, align = order + "IIQQQQQQ", 2, 3, 7
        else:
            phoff, = struct.unpack_from(order + "I", header, 28)
            phentsize, phnum = struct.unpack_from(order + "HH", header, 42)
            # p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz,
            # p_flags, p_align
            fields, offset, vaddr, align = order + "IIIIIIII", 1, 2, 7
        size = struct.calcsize(fields)
        if phentsize < size:
            return False
        f.seek(phoff)
        table = f.read(phentsize * phnum)
    if len(table) < phentsize * phnum:
        return False
    for i in range(phnum):
        ph = struct.unpack_from(fields, table, i * phentsize)
        if ph[0] == PT_LOAD and ph[align] > 1 \
                and (ph[vaddr] - ph[offset]) % ph[align] != 0:
            return False
    return True


def strip_debug_info(chimerax_root):
    """Remove debug information from Linux executables and libraries

    Wheels from PyPI ship with theirs, and ChimeraX's own code may have been
    compiled with it.  Symbols stay, so backtraces still have function names.
    Debug builds (DEBUG=1) keep it.

    This runs in the app's own Python, which has many of these libraries
    mapped, and strip (binutils 2.36 and later) rewrites a file in place,
    which crashes the process with SIGBUS or SIGSEGV.  So copies are stripped
    and renamed over the originals, leaving the mapped files untouched.

    Older strip (binutils 2.35 on RHEL 9) misplaces the segments of some
    libraries that auditwheel has rewritten with patchelf, such as numpy's
    OpenBLAS, and the loader then refuses them.  A copy whose segments the
    loader would refuse is thrown away, keeping the original.

    $STRIP, if set, is the strip to run instead of the one on $PATH.  CI uses
    it to run a newer one from gcc-toolset-15-binutils on Rocky.
    """
    strip = os.environ.get("STRIP") or "strip"
    groups = elf_files(chimerax_root)
    paths = [links[0] for links in groups]
    before = sum(os.path.getsize(p) for p in paths)
    batches = [groups[i:i + 32] for i in range(0, len(groups), 32)]

    def strip_batch(batch):
        copies = []
        for links in batch:
            copy = links[0] + STRIP_SUFFIX
            shutil.copy2(links[0], copy)
            copies.append(copy)
        result = subprocess.run([strip, "--strip-debug"] + copies,
                                capture_output=True, text=True)
        # strip goes on to the other files after one it can't handle,
        # leaving that copy as it was
        if result.returncode != 0:
            print(f"clean_app: {result.stderr.strip()}", file=sys.__stderr__)
        kept = []
        for links, copy in zip(batch, copies):
            if not load_segments_aligned(copy):
                os.remove(copy)
                kept.append(links[0])
                continue
            os.replace(copy, links[0])
            for link in links[1:]:
                os.link(links[0], link + STRIP_SUFFIX)
                os.replace(link + STRIP_SUFFIX, link)
        return kept

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        kept = [path for batch_kept in pool.map(strip_batch, batches)
                for path in batch_kept]
    for path in kept:
        print(f"clean_app: strip misaligned {path}, kept it unstripped",
              file=sys.__stderr__)
    freed = before - sum(os.path.getsize(p) for p in paths)
    print(f"clean_app: removed {freed / 1e6:.0f} MB of debug information")
    return freed


def remove_scripts(chimerax_root):
    """remove script's whose interpreter is not a system binary
    (eg., Python scripts with paths to "nonexisting" python)
    """
    # cleanup -- remove python shell scripts
    if sys.platform.startswith('win'):
        shutil.rmtree(f'{chimerax_root}/bin/Scripts')
    else:
        for fn in os.listdir(f'{chimerax_root}/bin'):
            filename = f'{chimerax_root}/bin/{fn}'
            if not os.path.isfile(filename):
                continue
            with open(filename, 'rb') as f:
                header = f.read(64)
            if header[0:2] != b'#!':
                continue
            program = header[2:].lstrip()
            if program.startswith(b'/bin/') or \
                    program.startswith(b'/usr/bin/'):
                continue
            os.remove(filename)


if __name__ == "__main__":
    bin_dir = os.path.dirname(sys.executable)
    if os.path.basename(bin_dir) != 'bin':
        print("Must be called as CHIMERAX/bin/python", file=sys.__stderr__)
        raise SystemExit(1)
    ok = clean_app(os.path.dirname(bin_dir),
                   keep_debug_info="--keep-debug-info" in sys.argv[1:])
    raise SystemExit(0 if ok else 1)
