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
    from . import cmd
    return cmd.get_singleton(session, create=True)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, bundle_info):
    from . import cmd
    from chimerax.core.commands import register, create_alias
    if command_name == "echo":
        create_alias("echo", "log text $*")
        return
    register(command_name, cmd.log_desc, cmd.log)


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'Log':
        from . import gui
        return gui.Log
    return None
