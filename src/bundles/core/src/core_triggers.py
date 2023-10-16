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

"""
core_triggers: access to core triggers
======================================

Some core triggers are registered elsewhere, such as in the View constructor.

"""

trigger_info = {
    "app quit": True,
    "begin restore session": False,
    "begin save session": False,
    "command failed": False,
    "command finished": False,
    "command started": False,
    "end restore session": False,
    "end save session": False,
    "frame drawn": True,
    "graphics update": True,
    "new frame": True,
    "shape changed": False,
}

def register_core_triggers(core_triggerset):
    for tn, rbh in trigger_info.items():
        core_triggerset.add_trigger(tn, remove_bad_handlers=rbh)
