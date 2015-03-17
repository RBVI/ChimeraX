# vi: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # GUI started by command
    pass

#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from chimera.core import cli
    from . import cmd
    cli.register('contacts', cmd.contact_desc, cmd.contact_command)
