# vim: set expandtab ts=4 sw=4:

#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, bundle_info):
    from . import cmd
    cmd.register_resfit_command()
