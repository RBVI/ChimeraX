"""Find the native Qt exports used by the experimental startup warmups.

Run with ChimeraX's Python, or pass --pyside-dir to inspect another wheel.
On Windows, run from a Visual Studio Developer Command Prompt. On macOS
and Linux, use the system nm and c++filt. The script reads export tables
without loading Qt and scans every slice of a universal macOS library.
"""

import argparse
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys


METHODS = {
    "QtGui": ("QFontMetrics::height() const", "QOpenGLContext::globalShareContext()"),
    "QtCore": ("QObject::moveToThread(QThread*)",),
}


def run(*args, input=None):
    return subprocess.run(args, input=input, capture_output=True, text=True,
                          errors="replace", check=True).stdout


def library_path(pyside_dir, module, platform):
    if platform == "win32":
        return pyside_dir / f"Qt6{module[2:]}.dll"
    if platform == "darwin":
        return pyside_dir / "Qt" / "lib" / f"{module}.framework" / module
    return pyside_dir / "Qt" / "lib" / f"libQt6{module[2:]}.so.6"


def windows_exports(path, methods):
    demangler = shutil.which("undname") or shutil.which("llvm-undname")
    if demangler is None:
        raise FileNotFoundError("undname or llvm-undname is required")
    prefixes = []
    for method in methods:
        owner, name = method.split("(", 1)[0].split("::")
        prefixes.append(f"?{name}@{owner}@@")
    for symbol in set(re.findall(r"(?m)^\s*\d+\s+[\da-fA-F]+\s+[\da-fA-F]+\s+(\?\S+)",
                                 run("dumpbin", "/nologo", "/exports", str(path)))):
        if symbol.startswith(tuple(prefixes)):
            for line in run(demangler, symbol).splitlines():
                yield symbol, line.strip().removeprefix('is :- ').strip('"')


def unix_exports(path, platform, arch):
    if platform == "darwin":
        output = run("nm", "-arch", arch, "-gUj", str(path))
        symbols = [name[1:] for name in output.splitlines() if name.startswith("__Z")]
    else:
        output = run("nm", "-D", "--defined-only", "--format=posix", str(path))
        symbols = [line.split()[0].split("@", 1)[0]
                   for line in output.splitlines() if line.startswith("_Z")]
    symbols = sorted(set(symbols))
    if not symbols:
        return []
    signatures = run("c++filt", input="\n".join(symbols) + "\n").splitlines()
    return zip(symbols, signatures, strict=True)


def find_matches(exports, method):
    normalized = re.sub(r"\s+", "", method)
    owner = method.split("::", 1)[0]
    matches = {}
    for symbol, signature in exports:
        start = re.search(r"(?<![\w:])" + re.escape(owner) + r"::", signature)
        if start is None:
            continue
        candidate = re.sub(r"\b(?:class|struct|__ptr64)\b|\s+", "", signature[start.start():])
        if candidate == normalized:
            matches[symbol] = signature
    return sorted(matches.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyside-dir", type=Path, help="path to the PySide6 package directory")
    args = parser.parse_args()
    platform = sys.platform
    if platform not in ("darwin", "linux", "win32"):
        parser.error(f"unsupported platform: {platform}")
    pyside_dir = args.pyside_dir
    if pyside_dir is None:
        spec = importlib.util.find_spec("PySide6")
        if spec is None or spec.origin is None:
            parser.error("PySide6 is unavailable; use ChimeraX's Python or pass --pyside-dir")
        pyside_dir = Path(spec.origin).parent

    failed = False
    for module, methods in METHODS.items():
        library = library_path(pyside_dir, module, platform)
        if not library.is_file():
            parser.error(f"Qt library not found: {library}")
        try:
            slices = run("lipo", "-archs", str(library)).split() if platform == "darwin" else [None]
            for arch in slices:
                exports = list(windows_exports(library, methods) if platform == "win32"
                               else unix_exports(library, platform, arch))
                print(f"{library}" + (f" [{arch}]" if arch else ""))
                for method in methods:
                    matches = find_matches(exports, method)
                    if len(matches) != 1:
                        failed = True
                        print(f"  {method}: {'ambiguous' if matches else 'not exported'}")
                    for symbol, signature in matches:
                        print(f"  {symbol}  {signature}")
        except (OSError, subprocess.CalledProcessError) as error:
            parser.error(getattr(error, "stderr", None) or str(error))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
