# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):
    # FIXME: remove unneeded methods

    @staticmethod
    def start_tool(session, bundle_info):
        # 'start_tool' is called to start an instance of the tool
        # If providing more than one tool in package,
        # look at the name in 'bundle_info.name' to see which is being started.

        from .tool import ToolUI
        # UI should register itself with tool state manager
        return ToolUI(session, bundle_info)

    @staticmethod
    def register_command(command_name, bundle_info):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name + " SUBCOMMAND_NAME",
                 cmd.subcommand_desc, cmd.subcommand_function)
        # TODO: Register more subcommands here

    @staticmethod
    def open_file(session, f, name, filespec=None, **kw):
        # 'open_file' is called by session code to open a file
        from . import cmd
        import os.path
        cmd.help(session, "file:" + os.path.realpath(filespec))
        return [], "Opened %s" % name

    @staticmethod
    def initialize(session, bi):
        # bundle-specific initialization (causes import)
        pass

    @staticmethod
    def finish(session, bi):
        # deinitialize bundle in session
        pass

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ToolUI':
            from . import tool
            return tool.ToolUI
        return None

bundle_api = _MyAPI()
