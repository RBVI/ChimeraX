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
    def open_file(session, stream, name):

    @staticmethod
    def open_file(session, stream_or_path, optional_format_name, optional_file_name, **kw):
        # 'open_file' is called by session code to open a file;
        # returns (list of models, status message).
        #
        # Second arg must be 'stream' or 'path'.  Depending on the name, either an
        # open data stream or a filesystem path will be provided.  The third and
        # fourth arguments are optional (remove 'optional_' from their names if you
        # provide them).  'format_name' will be the first nickname of the format
        # (reminder: the nickname of a format that provides no explicit nicknames
        # is the lower-case version of the full format name).  'file_name' is the
        # name of the input file, with compression suffix and path components
        # stripped.
        # 
        # You shouldn't actually use "**kw" but instead declare the actual keyword
        # args that your format supports (as per your bundle_info.xml file).
        from .io import open_xyz
        return open_xyz(session, stream)


bundle_api = _MyAPI()
