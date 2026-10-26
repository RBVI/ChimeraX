import argparse
import logging
import os
import sys

from chimerax.bundle_builder import xml_to_toml


def _get_builder(logger, path):
    if os.path.exists(os.path.join(os.path.abspath(path), "bundle_info.xml")):
        from chimerax.bundle_builder import BundleBuilder

        return BundleBuilder.from_path(logger, path)
    from chimerax.bundle_builder import BundleBuilderTOML

    return BundleBuilderTOML.from_path(logger, path)


def _build(path, debug=False, release=False):
    # Same as "devel build" but without starting ChimeraX, so wheels can be
    # built while other bundles are being installed
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger("bundle_builder")
    try:
        bb = _get_builder(logger, path)
        bb.make_wheel(debug=debug, release=release)
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    return 0


# What building a wheel imports that importing bundle_builder does not
_BUILD_MODULES = [
    "setuptools.command.bdist_wheel",
    "setuptools.command.build",
    "setuptools.command.build_ext",
    "setuptools.command.build_py",
    "setuptools.command.egg_info",
    "setuptools.command.install",
    "setuptools.command.install_egg_info",
    "setuptools.command.install_lib",
    "setuptools.command.install_scripts",
    "setuptools.command.sdist",
]


def _build_many(paths, jobs, debug=False, release=False):
    # Build each bundle in a child forked from this process once it has
    # imported everything building needs, so the import (most of the time a
    # pure Python bundle takes) is paid once instead of for every bundle.
    # Each child's output is printed when it finishes, so it is not mixed
    # with the others'.
    import importlib
    import tempfile

    if not hasattr(os, "fork"):
        import subprocess
        from concurrent.futures import ThreadPoolExecutor

        def build(path):
            cmd = [sys.executable, "-I", "-m", "chimerax.bundle_builder", "--build", "."]
            cmd += ["--debug"] * debug + ["--release"] * release
            r = subprocess.run(
                cmd, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            return path, r.returncode, r.stdout

        with ThreadPoolExecutor(jobs) as pool:
            results = list(pool.map(build, paths))
        for result in results:
            _report(result)
    else:
        for name in _BUILD_MODULES:
            importlib.import_module(name)
        from Cython.Build import cythonize

        cythonize([])
        results = []
        running = {}
        pending = list(paths)
        while pending or running:
            while pending and len(running) < jobs:
                path = pending.pop(0)
                log = tempfile.TemporaryFile()
                sys.stdout.flush()
                sys.stderr.flush()
                pid = os.fork()
                if pid == 0:
                    code = 1
                    try:
                        os.dup2(log.fileno(), 1)
                        os.dup2(log.fileno(), 2)
                        os.chdir(path)
                        code = _build(".", debug=debug, release=release)
                    finally:
                        sys.stdout.flush()
                        sys.stderr.flush()
                        os._exit(code)
                running[pid] = (path, log)
            pid, status = os.wait()
            path, log = running.pop(pid)
            log.seek(0)
            results.append((path, os.waitstatus_to_exitcode(status), log.read()))
            log.close()
            _report(results[-1])
    failed = [path for path, code, output in results if code != 0]
    if failed:
        print("Building failed in:", " ".join(failed), file=sys.stderr)
        return 1
    return 0


def _report(result):
    path, code, output = result
    sys.stdout.flush()
    sys.stdout.buffer.write(output)
    sys.stdout.flush()
    if code != 0:
        print("Building %s failed" % path, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ChimeraX Bundle Builder")
    parser.add_argument(
        "-c",
        "--convert",
        type=str,
        help="Convert a bundle_info.xml to a pyproject.toml file",
    )
    parser.add_argument(
        "--dynamic-version",
        action="store_true",
        help="When converting, make the version number dynamic [e.g. relocate it to src/__init__.py]",
    )
    parser.add_argument(
        "-b",
        "--build",
        nargs="+",
        metavar="PATH",
        help="Build a wheel for the bundle in each PATH without starting ChimeraX",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="With several --build paths, how many to build at once",
    )
    parser.add_argument(
        "--debug", action="store_true", help="When building, create a debug version"
    )
    parser.add_argument(
        "--release", action="store_true", help="When building, make a release build"
    )
    args = parser.parse_args()
    if args.convert:
        xml_to_toml(args.convert, dynamic_version=args.dynamic_version)
    if args.build and len(args.build) > 1:
        sys.exit(
            _build_many(
                args.build, args.jobs, debug=args.debug, release=args.release
            )
        )
    elif args.build:
        sys.exit(
            _build(
                args.build[0],
                debug=args.debug,
                release=args.release,
            )
        )
