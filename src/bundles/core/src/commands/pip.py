# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os
import subprocess
import sys
import shutil
import glob

from packaging.requirements import Requirement, InvalidRequirement

from .cli import BoolArg, EnumOf, StringArg, CmdDesc

from ..session import Session
from ..errors import UserError

__all__ = ['pip', 'pip_desc', 'register_pip_commands']

def pip(
    session: Session
    , action: str = None
    , package: str = None
    , upgrade: bool = False
    , verbose: bool = False
):
    from chimerax import app_dirs
    using_chimerax = "chimerax" in sys.executable.split(os.sep)[-1].lower()
    old_pythonuserbase = os.environ.get('PYTHONUSERBASE', None)
    os.environ['PYTHONUSERBASE'] = app_dirs.user_data_dir
    if not using_chimerax:
        pip_cmd = [sys.executable]
    else:
        exe_parted_out = sys.executable.split(os.sep)[:-1]
        executable_prefix = os.sep.join(["", os.path.join(*exe_parted_out)])
        # TODO: in 3.11, glob gets a root_dir parameter that makes this unnecessary
        old_cwd = os.getcwd()
        os.chdir(executable_prefix)
        cx_py_exe = glob.glob("[Pp]ython*")[0]
        exe_parted_out.append(cx_py_exe)
        path_to_cx_py_exe = os.sep.join([executable_prefix, cx_py_exe])
        os.chdir(old_cwd)
        pip_cmd = [path_to_cx_py_exe]
    pip_cmd.extend(["-m", "pip"])
    if action == 'install':
        if not package:
            raise UserError("Can't possibly install an unspecified package.")
        else:
            if not package.endswith(".tar.gz"):
                try:
                    req = Requirement(package)
                except InvalidRequirement:
                    raise UserError("Can't install package: invalid requirement specified.")
            pip_cmd.extend(["install" , "--user"])
            if upgrade:
                pip_cmd.extend(["--upgrade"])
            pip_cmd.extend(["%s" % package])
            # If we don't add this flag then pip complains that distutils and sysconfig
            # don't report the same location for the user's site packages directory. The
            # error tells programmers to report the error to
            # https://github.com/pypa/pip/issues/10151
            pip_cmd.extend(["--no-warn-script-location"])
            # #8927 -- check if there's an existing site-packages directory and, if so,
            # move it to user_data_dir/lib/python3.x/site-packages, then symbolically
            # link it back to its old location
            user_site_packages = os.path.join(app_dirs.user_data_dir, "site-packages")
            if sys.platform == "win32":
                ver_info = "".join([str(x) for x in sys.version_info[:2]])
                py_ver = "".join(["Python", ver_info])
                real_site_dir = os.path.join(app_dirs.user_data_dir, py_ver, "site-packages")
            else:
                ver_info = ".".join([str(x) for x in sys.version_info[:2]])
                py_ver = "".join(["python", ver_info])
                real_site_dir = os.path.join(app_dirs.user_data_dir, "lib", py_ver, "site-packages")
            if os.path.exists(user_site_packages):
                if not os.path.islink(user_site_packages):
                    if os.path.exists(real_site_dir):
                        # This path should be unreachable, since a user will always either transition from
                        # the old scheme and hit the else clause here or neither site-packages will exist and
                        # they'll hit the else clause of the outer if
                        ...
                    else:
                        os.makedirs(os.path.dirname(real_site_dir))
                        shutil.move(user_site_packages, real_site_dir)
                        # On Windows, we have to use a junction since a symlink requires
                        # elevated privileges
                        if sys.platform == "win32":
                            subprocess.run('mklink /J "%s" "%s"' % (user_site_packages, real_site_dir), shell=True)
                        else:
                            os.symlink(real_site_dir, user_site_packages)
            else:
                os.makedirs(real_site_dir)
                os.symlink(real_site_dir, user_site_packages)
    elif action == 'uninstall':
        if not package:
            raise UserError("Can't possibly uninstall an unspecified package.")
        else:
            if package.lower().startswith("chimerax"):
                raise UserError("Uninstalling this package could compromise ChimeraX; refusing to execute.")
            else:
                pip_cmd.extend(["uninstall", "-y", "%s" % package])
    elif action == 'list':
        pip_cmd.extend(["list"])
    elif action == 'check':
        pip_cmd.extend(["check"])
    elif action == 'show':
        if not package:
            raise UserError("Can't possibly show an unspecified package.")
        pip_cmd.extend(["show", "%s" % package])
    elif action == 'search':
        if not package:
            raise UserError("Can't possibly search for an unspecified package.")
        pip_cmd.extend(["search", "%s" % package])
    elif action == 'debug':
       pip_cmd.extend(["debug"])
    else:
        raise UserError("Unsupported action. Please use one of [install, uninstall, list, check, show, search, debug].")
    if verbose:
        pip_cmd.extend(["--verbose"])
    res = subprocess.run(pip_cmd, capture_output=True)
    os.environ['PYTHONUSERBASE'] = old_pythonuserbase or ''
    session.logger.info(res.stdout.decode('utf-8'))
    session.logger.info(res.stderr.decode('utf-8'))

pip_desc = CmdDesc(
    required=[
        ("action", EnumOf(["install", "uninstall", "list", "check", "show", "search", "debug"]))
    ],
    optional=[
        ("package", StringArg)
    ],
    keyword=[
        ("upgrade", BoolArg)
        , ("verbose", BoolArg)
    ],
    synopsis="Call pip from within ChimeraX"
)

def register_command(logger):
    from chimerax.core.commands import register, create_alias
    register("pip", pip_desc, pip, logger=logger)
    create_alias("devel pip", "pip $*", logger=logger)
