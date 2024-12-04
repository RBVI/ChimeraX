# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
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
core_triggers: access to core triggers
======================================

Some core triggers are registered elsewhere, such as in the View constructor.

"""

APP_QUIT = "app quit"
BEGIN_RESTORE_SESSION = "begin restore session"
BEGIN_SAVE_SESSION = "begin save session"
COMMAND_FAILED = "command failed"
COMMAND_FINISHED = "command finished"
COMMAND_STARTED = "command started"
END_RESTORE_SESSION = "end restore session"
END_SAVE_SESSION = "end save session"
FRAME_DRAWN = "frame drawn"
GRAPHICS_UPDATE = "graphics update"
NEW_FRAME = "new frame"
SHAPE_CHANGED = "shape changed"

trigger_info = {
    APP_QUIT: True,
    BEGIN_RESTORE_SESSION: False,
    BEGIN_SAVE_SESSION: False,
    COMMAND_FAILED: False,
    COMMAND_FINISHED: False,
    COMMAND_STARTED: False,
    END_RESTORE_SESSION: False,
    END_SAVE_SESSION: False,
    FRAME_DRAWN: True,
    GRAPHICS_UPDATE: True,
    NEW_FRAME: True,
    SHAPE_CHANGED: False,
}


def register_core_triggers(core_triggerset):
    for tn, rbh in trigger_info.items():
        core_triggerset.add_trigger(tn, remove_bad_handlers=rbh)
