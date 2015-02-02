# vim: set expandtab ts=4 sw=4:

#
# 'register_command' is called by the toolshed on start up
# 'start_tool' is called to start an instance of the tool
#


def start_tool(session, ti):
    return


def register_command(command_name):
    from . import cmd
    from chimera.core import cli
    desc_suffix = "_desc"
    for attr_name in dir(cmd):
        if not attr_name.endswith(desc_suffix):
            continue
        subcommand_name = attr_name[:-len(desc_suffix)]
        try:
            func = getattr(cmd, subcommand_name)
        except AttributeError:
            print("no function for \"%s\"" % subcommand_name)
            continue
        desc = getattr(cmd, attr_name)
        cli.register(command_name + ' ' + subcommand_name, desc, func)
