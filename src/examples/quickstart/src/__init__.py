# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with BundleInfo and
                        # CommandInfo instance instead of command name.
                        # start_tool called with BundleInfo and
                        # ToolInfo instance instead of tool_name.
    _URL = "http://www.cgl.ucsf.edu/chimerax/docs/quickstart/index.html"

    @classmethod
    def start_tool(cls, session, bi, ti, **kw):
        # Called from menu item
        cls._func(session)

    @classmethod
    def register_command(cls, bi, ci, logger):
        # Register command line command
        from chimerax.core.commands import register, CmdDesc
        desc = CmdDesc(synopsis=ci.synopsis)
        register(ci.name, desc, cls._func)

    @classmethod
    def _func(cls, session):
        from chimerax.core.commands import run
        run(session, "open %s" % cls._URL)

bundle_api = _MyAPI()
