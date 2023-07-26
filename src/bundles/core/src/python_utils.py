# vim: set et sw=4 sts=4:

import glob
import os
import shutil
import sys

from contextlib import contextmanager

from .utils import make_link

"""Utilities for managing the Python executable ChimeraX depends on."""
def chimerax_python_executable():
    """Find the Python executable that comes with ChimeraX"""
    using_chimerax = "chimerax" in os.path.realpath(sys.executable).split(os.sep)[-1].lower()
    if not using_chimerax:
        exe = sys.executable
    else:
        exe_parted_out = os.path.realpath(sys.executable).split(os.sep)[:-1]
        exe_prefix = os.sep.join(["", os.path.join(*exe_parted_out)])
        if sys.version_info.minor < 11:
            old_cwd = os.getcwd()
            os.chdir(exe_prefix)
            cx_py_exe = glob.glob("[Pp]ython*")[0]
            os.chdir(old_cwd)
        else:
            cx_py_exe = glob.glob("[Pp]ython*", root_dir = exe_prefix)[0]
        exe = os.sep.join([exe_prefix, cx_py_exe])
    return exe

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
    os.environ['PYTHONUSERBASE'] = old_pythonuserbase or ''

def migrate_site_packages():
    """Ensure compliance with both ChimeraX and Python site package conventions

    ChimeraX's customary site-packages directory is, on macOS,
    ~/Library/Application Support/ChimeraX/(chimerax_release_ver)/site-packages
    where ~/Library/Application Support/ChimeraX/(chimerax_release_ver) is PYTHONUSERBASE.
    Despite differences, we try not to fight Python too much, and Python prefers to install
    packages to PYTHONUSERBASE/lib/python(python_ver)/site-packages.

    To ensure user scripts don't break, we symlink Python's preferred location to our
    preferred location. See Trac#8927
    """
    from chimerax import app_dirs
    user_site_packages = os.path.join(app_dirs.user_data_dir, "site-packages")
    lib = "lib"
    python = "python"
    if sys.platform == "win32":
        lib = ""
        python = "Python311"
    if sys.platform == "linux":
        python = "python3.11"
    real_site_dir = os.path.join(app_dirs.user_data_dir, lib, python, "site-packages")
    if os.path.exists(user_site_packages):
        if not is_link(user_site_packages):
            if os.path.exists(real_site_dir):
                if not os.listdir(user_site_packages):
                    shutil.rmtree(user_site_packages)
                    make_link(real_site_dir, user_site_packages)
                elif not os.listdir(real_site_dir):
                    shutil.move(user_site_packages, real_site_dir)
                    make_link(real_site_dir, user_site_packages)
                else:
                    # This path should be unreachable, since a user will always either transition
                    # from the old scheme and hit the else clause here or neither site-packages
                    # will exist and they'll hit the else clause of the outer if
                    raise RuntimeError("Populated ChimeraX site packages and Python site packages. Please report this bug!")
            else:
                os.makedirs(os.path.dirname(real_site_dir), exist_ok = True)
                shutil.move(user_site_packages, real_site_dir)
                make_link(real_site_dir, user_site_packages)
    else:
        os.makedirs(real_site_dir, exist_ok=True)
        make_link(real_site_dir, user_site_packages)

def is_link(path):
    # Account for junctions on Windows
    if sys.platform == "win32":
        try:
            return bool(os.readlink(path))
        except OSError:
            return False
    else:
        return os.path.islink(path)
