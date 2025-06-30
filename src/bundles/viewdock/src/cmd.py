# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
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

from chimerax.core.commands import CmdDesc, StringArg
from chimerax.atomic import AtomicStructuresArg


def viewdock(session, structures=None, name=None):
    from .tool import ViewDockTool
    return ViewDockTool(session, "ViewDock")

viewdock_desc = CmdDesc(optional=[("structures", AtomicStructuresArg),
                                  ("name", StringArg)])

command_map = {
    "viewdock": (viewdock, viewdock_desc),
}


def register_command(ci):
    try:
        func, desc = command_map[ci.name]
    except KeyError:
        raise ValueError("trying to register unknown command: %s" % ci.name)
    if desc.synopsis is None:
        desc.synopsis = ci.synopsis
    from chimerax.core.commands import register
    register(ci.name, desc, func)
