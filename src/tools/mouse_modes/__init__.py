# vim: set expandtab ts=4 sw=4:

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


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, bundle_info):
    from .gui import MouseModePanel
    return MouseModePanel.get_singleton(session)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'MouseModePanel':
        from . import gui
        return gui.MouseModePanel
    return None
