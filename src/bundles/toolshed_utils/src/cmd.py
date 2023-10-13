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

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, EnumOf
    from . import tool

    DialogTypeArg = EnumOf(*zip(*((dt, dt.name.lower()) for dt in tool.DialogType)))

    toolshed_updates_desc = CmdDesc(
        optional=[('dialog_type', DialogTypeArg)],
        synopsis='show updates for installed bundles')

    def toolshed_updates(session, dialog_type=None):
        tool.show(session, dialog_type)

    register("toolshed updates", toolshed_updates_desc, toolshed_updates, logger=logger)
