#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from . import test
    from chimera.core.commands import register, CmdDesc
    desc = CmdDesc(synopsis = 'Run through test sequence of commands to check for errors')
    register('test', desc, test.run_commands)
