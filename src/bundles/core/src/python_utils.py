# vim: set et sw=4 sts=4:
"""Utilities for managing the Python executable ChimeraX depends on."""
import os
import sys

from contextlib import contextmanager


def _debug(*args, **kw):
    pass


def chimerax_python_executable():
    """Find the Python executable that comes with ChimeraX"""
    import sys
    from os.path import dirname, join, realpath
    chimerax = realpath(sys.executable)
    if sys.platform.startswith('win'):
        bin_dir = dirname(chimerax)
        return join(bin_dir, 'python.exe')
    if sys.platform.startswith('linux'):
        bin_dir = dirname(chimerax)
        v = sys.version_info
        return join(bin_dir, f'python{v.major}.{v.minor}')
    if sys.platform == 'darwin':
        bin_dir = join(dirname(dirname(chimerax)), 'bin')
        v = sys.version_info
        return join(bin_dir, f'python{v.major}.{v.minor}')
    # fallback to the ChimeraX executable (should never happen)
    # which can act like the python executable
    return chimerax


@contextmanager
def chimerax_user_base():
    """Make pip install packages to ChimeraX's customary PYTHONUSERBASE.

    Without this context manager, Python will install packages in the traditional
    user directory at, on macOS, ~/Library/Python/(version)/lib/python/site-packages
    instead of our location at ~/Library/Application Support/ChimeraX/(cx_version)
    """
    from chimerax import app_dirs
    old_pythonuserbase = os.environ.get('PYTHONUSERBASE', None)
    os.environ['PYTHONUSERBASE'] = app_dirs.user_data_dir
    yield
    if old_pythonuserbase is None:
        del os.environ['PYTHONUSERBASE']
    else:
        os.environ['PYTHONUSERBASE'] = old_pythonuserbase


def is_link(path):
    # Account for junctions on Windows
    if sys.platform == "win32":
        try:
            return bool(os.readlink(path))
        except OSError:
            return False
    else:
        return os.path.islink(path)


_pip_ignore_warnings = [
    "You are using pip version",
    "You should consider upgrading",
]


def _pip_has_warnings(content):
    for line in content.splitlines():
        if not line:
            continue
        for ignore in _pip_ignore_warnings:
            if ignore in line:
                break
        else:
            return True
    return False


def run_pip(command):
    # Note: uses PYTHONUSERBASE environment variable to ensure
    # the user site directory is the ChimeraX application location.
    import subprocess
    prog = chimerax_python_executable()
    pip_cmd = [prog] + subprocess._args_from_interpreter_flags() + ["-m", "pip"]
    # pip_cmd = [sys.executable, "-m", "pip"]
    from chimerax.core.python_utils import chimerax_user_base
    with chimerax_user_base():
        cp = subprocess.run(pip_cmd + command, capture_output=True)
    return cp


def run_logged_pip(command, logger):
    import sys
    _debug("_run_logged_pip command:", command)
    cp = run_pip(command)
    if cp.returncode != 0:
        output = cp.stdout.decode("utf-8", "backslashreplace")
        error = cp.stderr.decode("utf-8", "backslashreplace")
        _debug("_run_logged_pip return code:", cp.returncode, file=sys.__stderr__)
        _debug("_run_logged_pip output:", output, file=sys.__stderr__)
        _debug("_run_logged_pip error:", error, file=sys.__stderr__)
        s = output + error
        if "PermissionError" in s:
            raise PermissionError(s)
        else:
            raise RuntimeError(s)
    result = cp.stdout.decode("utf-8", "backslashreplace")
    err = cp.stderr.decode("utf-8", "backslashreplace")
    _debug("_run_logged_pip stdout:", result)
    _debug("_run_logged_pip stderr:", err)
    if logger and _pip_has_warnings(err):
        logger.warning("Errors may have occurred when running pip:")
        logger.warning("pip standard error:\n---\n%s---" % err)
        logger.warning("pip standard output:\n---\n%s---" % result)
    return result
