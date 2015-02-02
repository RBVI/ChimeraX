# vim: set expandtab ts=4 sw=4:

#
# 'register_command' is called by the toolshed on start up
# 'start_tool' is called to start an instance of the tool
#


def start_tool(session, ti):
    return


def register_command(command_name):
    from . import cmd
    try:
        func = getattr(cmd, command_name)
        desc = getattr(cmd, command_name + "_desc")
    except AttributeError:
        print("unable to register command \"%s\"" % command_name)
    else:
        from chimera.core import cli
        cli.register(command_name, desc, func)
