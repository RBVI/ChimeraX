# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    # Override method for starting tool
    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import SampleTool
        return SampleTool(session, tool_name, **kw)

    # Override method for registering commands
    @staticmethod
    def register_command(command_name, logger):
        # We expect that there is a function in "cmd"
        # corresponding to every registered command
        # in "setup.py.in" and that they are named
        # identically (except with '_' replacing spaces)
        from . import cmd
        from chimerax.core.commands import register
        base_name = command_name.replace(" ", "_")
        func = getattr(cmd, base_name)
        desc = getattr(cmd, base_name + "_desc")
        register(command_name, desc, func)

    # Override method for opening file
    @staticmethod
    def open_file(session, stream, format_name):
        # 'open_file' is called by session code to open a file;
        # returns (list of models, status message).
        #
        # The first argument must be named 'session'.
        # The second argument must be named either 'path' or 'stream'.
        # A 'path' argument will be bound to the path to the input file, 
        # which may be a temporary file; a 'stream' argument will be
        # bound to an file-like object in text or binary mode, depending
        # on the DataFormat ChimeraX classifier in bundle_info.xml.
        # If you want both, use 'stream' as the second argument and
        # add a 'file_name' argument, which will be bound to the
        # last (file) component of the path.
        from .io import open_xyz
        return open_xyz(session, stream)

    # Override method for saving file
    @staticmethod
    def save_file(session, path, format_name):
        """ 'save_file' is called by session code to save a file
        """
        raise NotImplementedError
        # Additional keywords may be added by listing them
        # in the "Save" ChimeraX classifier in bundle_info.xml.
        # Keywords listed there should match additional arguments
        # to this function.

    # Override method for retrieving entries from databases
    @staticmethod
    def fetch_from_database(session, identifier,
                            format=None,
                            ignore_cache=False,
                            **kw):
        # 'fetch_from_database' is called by session code to fetch
        # data with give identifier.
        raise NotImplementedError     # FIXME: remove method if unneeded

    # Override method for initialization function called each time
    # ChimeraX starts.  Only invoked if the custom initialization
    # flag is set in bundle_info.xml.
    @staticmethod
    def initialize(session, bi):
        # bundle-specific initialization (causes import)
        raise NotImplementedError     # FIXME: remove method if unneeded

    # Override method for finalization function.
    # Only invoked if the custom initialization
    # flag is set in bundle_info.xml.
    @staticmethod
    def finish(session, bi):
        # deinitialize bundle in session (causes import)
        raise NotImplementedError

    # Override method to support saving tools in sessions
    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class from bundle that
        # was saved in a session
        raise NotImplementedError
        # "class_name" should be the name of one of the tools
        # in this bundle, so code might look something like:
        if class_name == 'ToolUI':
            from . import tool
            return tool.ToolUI
        return None


bundle_api = _MyAPI()
