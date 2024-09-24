# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with CommandInfo instance
                        # instead of string

    # Override method for starting tool
    @staticmethod
    def start_tool(session, bi, ti, **kw):
        from .tool import SampleTool
        return SampleTool(session, ti.name, **kw)

    # Override method for registering commands
    @staticmethod
    def register_command(bi, ci, logger):
        # We expect that there is a function in "cmd"
        # corresponding to every registered command
        # in "setup.py.in" and that they are named
        # identically (except with '_' replacing spaces)
        from . import cmd
        from chimerax.core.commands import register
        command_name = ci.name
        base_name = command_name.replace(" ", "_")
        func = getattr(cmd, base_name)
        desc = getattr(cmd, base_name + "_desc")
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(command_name, desc, func)

    # Implement provider method for opening file
    @staticmethod
    def run_provider(session, name, mgr):
        # 'run_provider' is called by a manager to invoke the 
        # functionality of the provider.  Since the "data formats"
        # manager never calls run_provider (all the info it needs
        # is in the Provider tag), we know that only the "open
        # command" manager will call this function, and customize
        # it accordingly.
        #
        # The 'name' arg will be the same as the 'name' attribute
        # of your Provider tag, and mgr will be the corresponding
        # Manager instance
        #
        # For the "open command" manager, this method must return
        # a chimerax.open_command.OpenerInfo subclass instance.
        #
        # If your bundle also saved or fetched files, then it
        # need to distinguish what to do based on the 'name'
        # argument or by testing the 'mgr' argument against
        # session.open_command or session.fetch_command.  See
        # the developer tutorial here: 
        #   https://www.cgl.ucsf.edu/chimerax/docs/devel/tutorials/introduction.html#writing-bundles-in-seven-easy-steps
        # for more info
        from chimerax.open_command import OpenerInfo
        class XyzOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, **kw):
                # The 'open' method is called to open a file,
                # and must return a (list of models created,
                # status message) tuple.
                from .io import open_xyz
                return open_xyz(session, data)
        return XyzOpenerInfo()

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
