# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, bundle_info):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name + " start",
                 cmd.start_desc, cmd.start_server)
        register(command_name + " port",
                 cmd.port_desc, cmd.report_port)
        register(command_name + " stop",
                 cmd.stop_desc, cmd.stop_server)

    @staticmethod
    def initialize(session, bi):
        # bundle-specific initialization (causes import)
        pass

    @staticmethod
    def finish(session, bi):
        # deinitialize bundle in session
        from . import cmd
        cmd.stop_server(session)

    @staticmethod
    def get_class(class_name):
        return None

bundle_api = _MyAPI()
