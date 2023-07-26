# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
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
