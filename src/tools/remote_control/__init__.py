def start_tool(session, bundle_info):
    from .remotecmd import remote_control
    remote_control(session, enable = True)	# Start XMLRPC server

def register_command(command_name, bundle_info):
    from . import remotecmd
    remotecmd.register_remote_control_command()

