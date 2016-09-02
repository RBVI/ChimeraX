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

#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, bundle_info):
    return None


#
# 'register_command' is the delayed command registration callback
#
def register_command(command_name, bundle_info):
    from importlib import import_module
    if command_name.startswith('~'):
        module_name = "." + command_name[1:]
    else:
        module_name = "." + command_name
    try:
        m = import_module(module_name, __package__)
    except ImportError:
        print("cannot import %s from %s" % (module_name, __package__))
    else:
        m.initialize(command_name)


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    return None
