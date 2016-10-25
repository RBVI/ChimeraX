# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):
    # FIXME: only implement methods that the metadata says should be there

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class from bundle that
        # was saved in a session
        # FIXME: remove if not providing a tool
        # FIXME: rename if ToolUI has different class name
        if class_name == 'ToolUI':
            from . import tool
            return tool.ToolUI
        return None

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        # If providing more than one tool in package,
        # look at 'tool_name' to see which is being started.
        raise NotImplementedError  # FIXME: remove method if unneeded
        from .tool import ToolUI
        # UI should register itself with tool state manager
        return ToolUI(session, tool_name)

    @staticmethod
    def register_command(command_name):
        # 'register_command' is lazily called when the command is referenced
        raise NotImplementedError  # FIXME: remove method if unneeded
        from . import cmd
        from chimerax.core.commands import register
        register(command_name + " SUBCOMMAND_NAME",
                 cmd.subcommand_desc, cmd.subcommand_function)
        # TODO: Register more subcommands here

    @staticmethod
    def register_selector(selector_name):
        # 'register_selector' is lazily called when the selector is referenced
        raise NotImplementedError  # FIXME: remove method if unneeded

    @staticmethod
    def open_file(session, f, name, filespec=None, **kw):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        raise NotImplementedError     # FIXME: remove method if unneeded

    @staticmethod
    def save_file(session, name, **kw):
        # 'save_file' is called by session code to save a file
        raise NotImplementedError     # FIXME: remove method if unneeded

    @staticmethod
    def fetch_url(session, identifier, ignore_cache=False, database_name=None, format_name=None, **kw):
        # 'fetch_url' is called by session code to fetch data from a URL
        raise NotImplementedError     # FIXME: remove method if unneeded

    @staticmethod
    def initialize(session, bi):
        # bundle-specific initialization (causes import)
        raise NotImplementedError     # FIXME: remove method if unneeded

    @staticmethod
    def finish(session, bi):
        # deinitialize bundle in session (causes import)
        raise NotImplementedError

    @staticmethod
    def fetch_data(format, database, ident):
        # return function capable of fetching data format
        raise NotImplementedError     # FIXME: remove method if unneeded

    @staticmethod
    def open_data(format, tag):
        # return function capable of opening data format
        raise NotImplementedError  # FIXME: remove method if unneeded

    @staticmethod
    def save_data(format, tag):
        # return function capable of saving data format
        raise NotImplementedError  # FIXME: remove method if unneeded

bundle_api = _MyAPI()
