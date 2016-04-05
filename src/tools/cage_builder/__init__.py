# vim: set expandtab ts=4 sw=4:

def start_tool(session, bi):
    if not session.ui.is_gui:
        return
    from .gui import cage_builder_panel
    p = cage_builder_panel(session, bi)
    return p

#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, bundle_info):
    from . import cmd
    cmd.register_cage_command()
