# vim: set expandtab ts=4 sw=4:

from .ksdssp import ss_assign

#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, bundle_info):
    from . import ksdssp
    ksdssp.register_command()

