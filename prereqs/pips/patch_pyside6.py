"""Make PySide6 / shiboken6 start faster.  Run with the application's Python:

    python patch_pyside6.py

Each step checks for the exact upstream text it replaces and stops with an
error if it is missing, so a PySide upgrade fails loudly instead of silently
skipping a patch.  Steps that were already applied are skipped.
"""
import os
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "ChimeraX:"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def replace_once(text, old, new, path):
    n = text.count(old)
    if n != 1:
        sys.exit(f"{path}: expected exactly one copy of {old[:60]!r}, found {n}")
    return text.replace(old, new)


def unpack_support_files(shiboken_dir):
    # PySide's supported way to put shibokensupport on disk, where its modules
    # get cached .pyc files instead of being compiled from an embedded zip.
    files_dir = os.path.join(shiboken_dir, "files.dir")
    # files.dir is not part of the wheel, so it survives reinstalling or
    # upgrading shiboken6.  Unpack it again if it came from another version.
    stamp = os.path.join(files_dir, "chimerax-shiboken-version")
    version = shiboken_version(shiboken_dir)
    try:
        with open(stamp) as f:
            current = f.read().strip() == version
    except OSError:
        current = False
    if not current and os.path.isdir(files_dir):
        import shutil
        shutil.rmtree(files_dir)
    if not os.path.isdir(files_dir):
        env = dict(os.environ, SBK_EMBED="0")
        subprocess.run([sys.executable, "-c", "import shiboken6.Shiboken"], env=env, check=True)
    if not os.path.isdir(files_dir):
        sys.exit(f"{files_dir} was not created")
    with open(stamp, "w") as f:
        f.write(version + "\n")
    return files_dir


def shiboken_version(shiboken_dir):
    for line in read(os.path.join(shiboken_dir, "__init__.py")).splitlines():
        if line.startswith("__version__ ="):
            return line.split("=", 1)[1].strip().strip("\"'")
    sys.exit(f"{shiboken_dir}/__init__.py: __version__ not found")


LOADER_LAZY = '''    put_into_package(PySide6.support.signature, lib)
    put_into_package(PySide6.support.signature, importhandler)


# ChimeraX: The modules below are only needed to compute signatures, report
# argument errors, or generate .pyi files, and importing them at startup costs
# tens of milliseconds (typing, inspect, ast, argparse, logging, many regexes).
# Load them, and publish them under PySide6.support as before, on first use.
_LAZY_SIGNATURE = ("mapping", "errorhandler", "layout", "parser")
_LAZY_LIB = ("enum_sig", "pyi_generator", "tool")
_signature_modules_loaded = False


def _mapping_placeholder():
    # libshiboken caches sys.modules["shibokensupport.signature.mapping"] and
    # its __dict__ the first time it reports an argument error, then calls
    # __dict__["update_mapping"] before calling seterror_argument().  So the
    # module object must exist from the start.  The real module code is run
    # into this same object (keeping the cached __dict__) on first use.
    import importlib.machinery
    import importlib.util
    name = "shibokensupport.signature.mapping"
    spec = importlib.machinery.PathFinder.find_spec(name, signature.__path__)
    module = importlib.util.module_from_spec(spec)

    def update_mapping(*args, **kwds):
        _load_signature_modules()
        return module.__dict__["update_mapping"](*args, **kwds)

    def __getattr__(attr):
        _load_signature_modules()
        try:
            return module.__dict__[attr]
        except KeyError:
            raise AttributeError(attr) from None

    module.update_mapping = update_mapping
    module.__getattr__ = __getattr__
    sys.modules[name] = module
    return module


def _load_signature_modules():
    global _signature_modules_loaded, mapping, errorhandler, layout, parser
    global enum_sig, pyi_generator, tool
    if _signature_modules_loaded:
        return
    _signature_modules_loaded = True
    module = sys.modules["shibokensupport.signature.mapping"]
    del module.__getattr__
    module.__spec__.loader.exec_module(module)
    mapping = signature.mapping = module
    from shibokensupport.signature import errorhandler, layout, parser
    from shibokensupport.signature.lib import enum_sig, pyi_generator, tool
    if "PySide6.support" in sys.modules:
        import PySide6.support
        for name in _LAZY_SIGNATURE:
            put_into_package(PySide6.support.signature, globals()[name])
        for name in _LAZY_LIB:
            put_into_package(PySide6.support.signature.lib, globals()[name])


def _normalize_enum_option():
    # Importing parser.py used to do this as a side effect, and libshiboken
    # and other code read the result.  Same rules as parser._get_flag_enum_option(),
    # without importing ast unless the environment value needs it.
    from shiboken6 import __version_info__ as ver
    flag = ver[:2] >= (6, 4)
    envname = "PYSIDE6_OPTION_PYTHON_ENUM"
    sysname = envname.lower()
    opt = os.environ.get(envname)
    if opt:
        opt = opt.lower()
        if opt in ("yes", "on", "true"):
            flag = True
        elif opt in ("no", "off", "false"):
            flag = False
        else:
            try:
                import ast
                flag = ast.literal_eval(opt)
            except Exception:
                flag = False
    elif hasattr(sys, sysname):
        flag = getattr(sys, sysname)
        if not isinstance(flag, int):
            flag = False
    setattr(sys, sysname, flag)
    os.environ[envname] = str(flag)
    # An unsupported value raises RuntimeError when parser.py is imported.
    if int(flag) == 0:
        _load_signature_modules()


def _lazy_getattr(names):
    def __getattr__(name):
        if name in names:
            _load_signature_modules()
            if name in globals():
                return globals()[name]
        raise AttributeError(name)
    return __getattr__


class _LazySignatureFinder:
    # "import PySide6.support.signature.mapping" must find the published module.
    _prefixes = ("PySide6.support.signature.", "shibokensupport.signature.")

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        if _signature_modules_loaded or not fullname.startswith(cls._prefixes):
            return None
        if fullname.rsplit(".", 1)[-1] not in _LAZY_SIGNATURE + _LAZY_LIB:
            return None
        _load_signature_modules()
        if fullname in sys.modules:
            import importlib.util
            return importlib.util.spec_from_loader(fullname, cls)
        return None

    @staticmethod
    def create_module(spec):
        return sys.modules[spec.name]

    @staticmethod
    def exec_module(module):
        pass
'''


def patch_loader(files_dir):
    path = os.path.join(files_dir, "shibokensupport", "signature", "loader.py")
    s = read(path)
    if MARK in s:
        return
    # Unused in loader.py, and traceback pulls in _colorize and linecache.
    s = replace_once(s, "import traceback\nimport types\n", "", path)
    for name, args, module, func in [
            ("pyside_type_init", "type_key, sig_strings", "parser", "pyside_type_init"),
            ("create_signature", "props, key", "layout", "create_signature"),
            ("seterror_argument", "args, func_name, info", "errorhandler", "seterror_argument"),
            ("make_helptext", "func", "errorhandler", "make_helptext")]:
        call = f"    return {module}.{func}("
        s = replace_once(s, f"def {name}({args}):\n{call}",
                         f"def {name}({args}):\n    _load_signature_modules()\n{call}", path)
    s = replace_once(s, """    put_into_package(PySide6.support.signature, mapping)
    put_into_package(PySide6.support.signature, errorhandler)
    put_into_package(PySide6.support.signature, layout)
    put_into_package(PySide6.support.signature, lib)
    put_into_package(PySide6.support.signature, parser)
    put_into_package(PySide6.support.signature, importhandler)
    put_into_package(PySide6.support.signature.lib, enum_sig)
    put_into_package(PySide6.support.signature.lib, pyi_generator)
    put_into_package(PySide6.support.signature.lib, tool)
""", LOADER_LAZY, path)
    s = replace_once(s, """from shibokensupport.signature import mapping
from shibokensupport.signature import errorhandler
from shibokensupport.signature import layout
from shibokensupport.signature import lib
from shibokensupport.signature import parser
from shibokensupport.signature import importhandler
from shibokensupport.signature.lib import enum_sig
from shibokensupport.signature.lib import pyi_generator
from shibokensupport.signature.lib import tool

import enum
""", """from shibokensupport.signature import lib
from shibokensupport.signature import importhandler

signature.__getattr__ = _lazy_getattr(_LAZY_SIGNATURE)
lib.__getattr__ = _lazy_getattr(_LAZY_LIB)
_mapping_placeholder()
_normalize_enum_option()
# The bootstrap executes this module twice; drop the first copy's finder.
sys.meta_path[:] = [f for f in sys.meta_path
                    if getattr(f, "__qualname__", None) != "_LazySignatureFinder"]
sys.meta_path.insert(0, _LazySignatureFinder)
""", path)
    write(path, s)


def patch_feature(files_dir):
    path = os.path.join(files_dir, "shibokensupport", "feature.py")
    s = read(path)
    if MARK in s:
        return
    s = replace_once(s, "import inspect\nimport sys\n", "import sys\n", path)
    s = replace_once(s, "def feature_import(name, *args, **kwargs):\n",
                     "def feature_import(name, *args, **kwargs):\n"
                     "    # ChimeraX: every import statement comes through here; only \"__feature__\"\n"
                     "    # imports need the caller's frame.\n"
                     "    if name != \"__feature__\":\n"
                     "        return None\n", path)
    s = replace_once(s, "    try:\n        source = inspect.getsource(module)",
                     "    import inspect\n    try:\n        source = inspect.getsource(module)", path)
    s = replace_once(s, "    # PYSIDE-1368: The `__name__` attribute does not need to exist in all modules.\n"
                        "    if hasattr(module, \"__name__\"):\n",
                     "    # ChimeraX: Until the first \"from __feature__ import\", __init__() has\n"
                     "    # not run and will classify every loaded module itself, so reading\n"
                     "    # each imported module's source here is wasted work.\n"
                     "    if not _is_initialized:\n"
                     "        return\n"
                     "    # PYSIDE-1368: The `__name__` attribute does not need to exist in all modules.\n"
                     "    if hasattr(module, \"__name__\"):\n", path)
    write(path, s)


def patch_shiboken_init(shiboken_dir):
    path = os.path.join(shiboken_dir, "__init__.py")
    s = read(path)
    if MARK in s:
        return
    start = s.find("# PYSIDE-932")
    end = s.find("from shiboken6.Shiboken import *")
    if start < 0 or end < start:
        sys.exit(f"{path}: preload block not found")
    s = (s[:start] + "# ChimeraX: shiboken6/files.dir is shipped unpacked, so the signature bootstrap\n"
         "# imports what it needs itself; no preloading needed.\n" + s[end:])
    write(path, s)


def patch_pyside_init(pyside_dir):
    path = os.path.join(pyside_dir, "__init__.py")
    s = read(path)
    if MARK in s:
        return
    # pathlib is only needed on Windows here.
    s = replace_once(s, "from pathlib import Path\n",
                     "# ChimeraX: import pathlib only where it is used (Windows).\n", path)
    s = replace_once(s, "def _additional_dll_directories(package_dir):\n",
                     "def _additional_dll_directories(package_dir):\n    from pathlib import Path\n", path)
    s = replace_once(s, "    pyside_package_dir = Path(__file__).parent.resolve()\n\n    if sys.platform == 'win32':\n",
                     "    if sys.platform == 'win32':\n        from pathlib import Path\n"
                     "        pyside_package_dir = Path(__file__).parent.resolve()\n", path)
    s = replace_once(s, "    for module in Path(__file__).resolve().parent.glob(pattern):\n",
                     "    from pathlib import Path\n    for module in Path(__file__).resolve().parent.glob(pattern):\n", path)
    write(path, s)


def patch_bootstrap(shiboken_dir, files_dir):
    # The bootstrap is a NUL-terminated source string inside libshiboken.
    # Overwrite it in place with a version that imports lazily, padded with
    # spaces to the original length so nothing else in the library moves.
    new = read(os.path.join(HERE, "pyside6_signature_bootstrap.py"))
    write(os.path.join(files_dir, "signature_bootstrap.py"), new)
    new_bytes = new.encode("utf-8")
    # libshiboken6.abi3.so.6.11 (Linux), libshiboken6.abi3.6.11.dylib (macOS),
    # shiboken6.abi3.dll (Windows)
    libs = [n for n in os.listdir(shiboken_dir)
            if n.startswith(("libshiboken6.", "shiboken6.abi3"))
            and not n.endswith((".lib", ".pyi"))]
    if len(libs) != 1:
        sys.exit(f"expected one libshiboken6 in {shiboken_dir}, found {libs}")
    path = os.path.join(shiboken_dir, libs[0])
    with open(path, "rb") as f:
        data = f.read()
    if MARK.encode() in data:
        return
    i = data.find(b"\ndef bootstrap():\n")
    if i < 0 or data.count(b"\ndef bootstrap():\n") != 1:
        sys.exit(f"{path}: embedded signature_bootstrap not found exactly once")
    start = data.rfind(b"\0", 0, i) + 1
    end = data.find(b"\0", i)
    old = data[start:end]
    if not old.startswith(b"# Copyright") or b"def find_incarnated_files():" not in old:
        sys.exit(f"{path}: embedded signature_bootstrap has an unexpected layout")
    if len(new_bytes) > len(old):
        sys.exit(f"replacement bootstrap is {len(new_bytes)} bytes, embedded one is {len(old)}")
    patched = data[:start] + new_bytes + b" " * (len(old) - len(new_bytes)) + data[end:]
    assert len(patched) == len(data)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(patched)
    os.chmod(tmp, os.stat(path).st_mode)
    os.replace(tmp, path)


def main():
    site = sysconfig.get_paths()["purelib"]
    shiboken_dir = os.path.join(site, "shiboken6")
    pyside_dir = os.path.join(site, "PySide6")
    files_dir = unpack_support_files(shiboken_dir)
    patch_loader(files_dir)
    patch_feature(files_dir)
    patch_shiboken_init(shiboken_dir)
    patch_pyside_init(pyside_dir)
    patch_bootstrap(shiboken_dir, files_dir)
    import compileall
    compileall.compile_dir(files_dir, quiet=1)
    compileall.compile_file(os.path.join(shiboken_dir, "__init__.py"), quiet=1)
    compileall.compile_file(os.path.join(pyside_dir, "__init__.py"), quiet=1)


if __name__ == "__main__":
    main()
