# vi: set expandtab shiftwidth=4 softtabstop=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    return None


#
# 'register_command' is the delayed command registration callback
#
def register_command(command_name):
    from importlib import import_module
    if command_name.startswith('~'):
        module_name = "." + command_name[1:]
    else:
        module_name = "." + command_name
    try:
        m = import_module(module_name, __package__)
    except ImportError:
        print("cannot import %s from %s" % (module_name, __package__))
    else:
        m.initialize(command_name)


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    return None
