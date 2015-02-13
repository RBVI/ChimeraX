# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # This function is simple because we "know" we only provide
    # a single tool in the entire package, so we do not need to
    # look at the name in 'ti.name'
    from chimera.core import cli
    cmd = cli.Command(session, "log show", final=True)
    cmd.execute()


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from . import cmd
    from chimera.core import cli
    cli.register(command_name + " hide", cmd.hide_desc, cmd.hide)
    cli.register(command_name + " show", cmd.show_desc, cmd.show)
    cli.register(command_name + " test", cmd.test_desc, cmd.test)
