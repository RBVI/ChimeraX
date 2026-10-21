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
Per-source-file parallel compilation for distutils compilers.

distutils compiles the sources of a library or extension one at a time, and
setuptools' ``build_ext -j`` only parallelizes across extensions. This module
replaces ``CCompiler.compile`` (and ``MSVCCompiler.compile`` on Windows) with
versions that compile each source file in a thread pool, the same approach
numpy.distutils used.

Since setuptools may build several extensions at once, and each of those
compiles its files in its own pool, all compiler processes share one global
semaphore so no more than ``jobs`` run at a time. The semaphore is only held
around the spawn of a single compiler process, which never waits on other
work, so the nested pools cannot deadlock.

Set CHIMERAX_BUILD_JOBS to override the number of jobs; 1 restores the stock
serial behavior.

CHIMERAX_BUILD_SPAWN_PREFIX, if set, is a command (such as "taskpolicy -c
utility") that the compiler, linker, and archiver are run under, leaving this
process itself unaffected.
"""

import importlib
import os
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# setuptools aliases the top-level "distutils" package to setuptools._distutils,
# but submodules get imported a second time under the "distutils." names, so
# there are two distinct copies of every compiler and exception class. Bundle
# builder uses distutils.ccompiler; setuptools itself may use either. Patch both.
_DISTUTILS_PACKAGES = ("distutils", "setuptools._distutils")


def _default_jobs():
    try:
        jobs = int(os.environ.get("CHIMERAX_BUILD_JOBS", ""))
    except ValueError:
        jobs = 0
    if jobs > 0:
        return jobs
    return os.cpu_count() or 1


jobs = 1
_job_semaphore = threading.Semaphore(1)


def set_jobs(n):
    global jobs, _job_semaphore
    jobs = max(1, n)
    _job_semaphore = threading.Semaphore(jobs)


set_jobs(_default_jobs())

# Two extensions may list the same source file and so produce the same object
# file; if they are built simultaneously, make one wait for the other instead
# of having two compilers write the same file.
_in_progress = set()
_in_progress_changed = threading.Condition()


@contextmanager
def _claimed(obj):
    key = os.path.abspath(obj)
    with _in_progress_changed:
        while key in _in_progress:
            _in_progress_changed.wait()
        _in_progress.add(key)
    try:
        with _job_semaphore:
            yield
    finally:
        with _in_progress_changed:
            _in_progress.discard(key)
            _in_progress_changed.notify_all()


def _run(func, items):
    if jobs == 1 or len(items) < 2:
        for item in items:
            func(item)
        return
    with ThreadPoolExecutor(max_workers=min(jobs, len(items))) as pool:
        futures = [pool.submit(func, item) for item in items]
        try:
            for future in futures:
                future.result()
        except BaseException:
            # Don't start compiling the rest of a build that has already failed
            pool.shutdown(cancel_futures=True)
            raise


def compile_libraries(libraries, logger, dependencies, debug=False):
    """Compile and link a bundle's C libraries.

    Compiling only needs headers, so all of the libraries' sources are compiled
    at once (still limited to ``jobs`` compilers in total).  Linking can need
    the libraries listed before, so that is done afterwards, in order.
    """
    def compile_objects(lib):
        return lib.compile_objects(logger, dependencies, lib.static, debug)

    if jobs == 1 or len(libraries) < 2:
        compiled = [compile_objects(lib) for lib in libraries]
    else:
        with ThreadPoolExecutor(max_workers=len(libraries)) as pool:
            compiled = list(pool.map(compile_objects, libraries))
    for lib, objects in zip(libraries, compiled):
        lib.link(objects, debug)


class BackgroundBuild:
    """Run a bundle's library and executable builds in a thread while setup()
    compiles its extension modules.

    Extension modules only need the libraries when they are linked, so setup()
    is given a build_ext (see ``build_ext_class``) that waits for this thread
    before linking, and before returning so that build_py, which runs after
    it, finds the libraries to package.
    """

    def __init__(self, work):
        self._work = work
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self._work()
        except BaseException as e:
            self._error = e

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def wait(self):
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("building the bundle's C libraries failed") from self._error

    def build_ext_class(self, base):
        background = self

        class build_ext(base):
            def build_extensions(self):
                link = self.compiler.link_shared_object

                def wait_then_link(*args, **kw):
                    background.wait()
                    return link(*args, **kw)

                self.compiler.link_shared_object = wait_then_link
                try:
                    super().build_extensions()
                finally:
                    del self.compiler.link_shared_object

            def run(self):
                super().run()
                background.wait()

        return build_ext


def _parallel_compile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=False,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    if not debug:
        # Python's CFLAGS include -g, which makes the compile 30-50% slower.
        # On macOS the debug info never reaches the wheel anyway (it stays in
        # the object files since nothing runs dsymutil); on Linux it only
        # bloats the wheel.  Debug builds still get it.
        extra_postargs = list(extra_postargs or []) + ["-g0"]
    # Same setup as CCompiler.compile; only the loop over objects differs
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    def compile_one(obj):
        src, ext = build[obj]
        with _claimed(obj):
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Largest sources first, so that one that takes long does not start last
    # and leave the build waiting on it alone
    def size(obj):
        try:
            return os.path.getsize(build[obj][0])
        except OSError:
            return 0

    to_build = [obj for obj in objects if obj in build]
    _run(compile_one, sorted(to_build, key=size, reverse=True))
    # Return *all* object filenames, not just the ones we just built.
    return objects


# MSVCCompiler overrides compile() with its own loop that spawns cl.exe for
# each file instead of calling _compile(), so the approach above can't reach
# it. Instead, while its compile() runs, intercept the cl.exe spawns (and only
# those; rc.exe and mc.exe have ordering dependencies) and record them, then
# run the recorded commands in parallel once it returns. The interception is
# thread-local because setuptools shares one compiler between the threads of
# build_ext -j.
_deferred = threading.local()


def _install_msvc(package):
    try:
        MSVCCompiler = importlib.import_module(package + "._msvccompiler").MSVCCompiler
        errors = importlib.import_module(package + ".errors")
    except (ImportError, AttributeError):
        return
    if getattr(MSVCCompiler.compile, "_chimerax_parallel", False):
        return
    CompileError, DistutilsExecError = errors.CompileError, errors.DistutilsExecError
    original_compile = MSVCCompiler.compile
    original_spawn = MSVCCompiler.spawn

    def spawn(self, cmd):
        pending = getattr(_deferred, "commands", None)
        if pending is not None and cmd and cmd[0] == self.cc:
            pending.append(list(cmd))
            return None
        return original_spawn(self, cmd)

    def compile(self, *args, **kw):
        _deferred.commands = []
        try:
            objects = original_compile(self, *args, **kw)
            commands = _deferred.commands
        finally:
            _deferred.commands = None

        def run(cmd):
            obj = next(arg[3:] for arg in cmd if arg.startswith("/Fo"))
            with _claimed(obj):
                try:
                    original_spawn(self, cmd)
                except DistutilsExecError as msg:
                    raise CompileError(msg)

        _run(run, commands)
        return objects

    compile._chimerax_parallel = True
    MSVCCompiler.spawn = spawn
    MSVCCompiler.compile = compile


def _install_spawn_prefix(base, prefix):
    if getattr(base.spawn, "_chimerax_prefixed", False):
        return
    original_spawn = base.spawn

    def spawn(self, cmd, **kw):
        return original_spawn(self, prefix + list(cmd), **kw)

    spawn._chimerax_prefixed = True
    base.spawn = spawn


def hide_console_windows():
    """Keep each compiler and linker run from opening a console window.

    On Windows, a console program started by a process without a console gets
    a window of its own, so "devel build" in ChimeraX, which has none, would
    flash one up for every compile and link.  subprocess makes the STARTUPINFO
    for each process from its module's STARTUPINFO class, so until the
    returned function is called, that class sets STARTF_USESHOWWINDOW; the
    default wShowWindow, SW_HIDE, then hides the window.
    """
    if sys.platform != "win32":
        return lambda: None
    import _winapi
    import subprocess

    original = subprocess.STARTUPINFO

    class HiddenWindowSTARTUPINFO(original):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.dwFlags |= _winapi.STARTF_USESHOWWINDOW

    subprocess.STARTUPINFO = HiddenWindowSTARTUPINFO

    def restore():
        subprocess.STARTUPINFO = original

    return restore


def install():
    """Patch the distutils compilers to compile source files in parallel.

    Safe to call more than once. Does nothing if only one job is allowed, or
    if this version of distutils doesn't have the private methods we rely on.
    """
    prefix = shlex.split(os.environ.get("CHIMERAX_BUILD_SPAWN_PREFIX", ""))
    if prefix and sys.platform != "win32":
        for package in _DISTUTILS_PACKAGES:
            try:
                base = importlib.import_module(package + ".ccompiler").CCompiler
            except (ImportError, AttributeError):
                continue
            _install_spawn_prefix(base, prefix)
    if jobs == 1:
        return
    for package in _DISTUTILS_PACKAGES:
        try:
            base = importlib.import_module(package + ".ccompiler").CCompiler
        except (ImportError, AttributeError):
            continue
        if hasattr(base, "_setup_compile") and hasattr(base, "_get_cc_args"):
            base.compile = _parallel_compile
        if sys.platform == "win32":
            _install_msvc(package)
