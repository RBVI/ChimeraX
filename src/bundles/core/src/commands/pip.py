# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os

from packaging.requirements import Requirement, InvalidRequirement

from .cli import BoolArg, EnumOf, StringArg, CmdDesc

from ..session import Session
from ..errors import UserError
from ..python_utils import run_logged_pip

__all__ = ['pip', 'pip_desc', 'register_command']

def pip(
    session: Session
    , action: str = None
    , package: str = None
    , upgrade: bool = False
    , verbose: bool = False
):
    from chimerax import app_dirs
    pip_cmd = []
    if action == 'install':
        if not package:
            raise UserError("Can't possibly install an unspecified package.")
        else:
            if not package.endswith(".tar.gz"):
                try:
                    _ = Requirement(package)
                except InvalidRequirement:
                    raise UserError("Can't install package: invalid requirement specified.")
            pip_cmd.extend(["install" , "--user", "-qq"])
            if upgrade:
                pip_cmd.extend(["--upgrade"])
            # If we don't add this flag then pip complains that distutils and sysconfig
            # don't report the same location for the user's site packages directory. The
            # error tells programmers to report the error to
            # https://github.com/pypa/pip/issues/10151
            pip_cmd.extend(["--upgrade-strategy", "only-if-needed", "--no-warn-script-location"])
            pip_cmd.append(package)
    elif action == 'uninstall':
        if not package:
            raise UserError("Can't possibly uninstall an unspecified package.")
        else:
            if package.lower().startswith("chimerax"):
                raise UserError("Uninstalling this package could compromise ChimeraX; refusing to execute.")
            else:
                pip_cmd.extend(["uninstall", "-y", "%s" % package])
    elif action == 'list':
        pip_cmd.appendf("list")
    elif action == 'check':
        pip_cmd.appendf("check")
    elif action == 'show':
        if not package:
            raise UserError("Can't possibly show an unspecified package.")
        pip_cmd.extend(["show", "%s" % package])
    elif action == 'search':
        if not package:
            raise UserError("Can't possibly search for an unspecified package.")
        pip_cmd.extend(["search", "%s" % package])
    elif action == 'debug':
       pip_cmd.append("debug")
    else:
        raise UserError("Unsupported action. Please use one of [install, uninstall, list, check, show, search, debug].")
    if verbose:
        pip_cmd.append("--verbose")
    run_logged_pip(pip_cmd, session.logger)

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
