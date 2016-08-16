# vim: set expandtab shiftwidth=4 softtabstop=4:

#
# 'register_command' is the delayed command registration callback
#
def register_command(command_name, bundle_info):
    from . import mlp
    mlp.register_mlp_command()
