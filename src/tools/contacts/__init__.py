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
    # GUI started by command, so this is for restoring sessions
    if not session.ui.is_gui:
        return None
    from .gui import Plot
    return Plot(session, bundle_info)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, bundle_info):
    from . import cmd
    cmd.register_contacts()


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'Plot':
        from . import gui
        return gui.Plot
    return None
