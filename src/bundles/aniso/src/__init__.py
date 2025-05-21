# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _AnisoAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == '_StructureAnisoManager':
            from . import mgr
            return getattr(mgr, class_name)

    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import AnisoTool
        return AnisoTool(session, tool_name, **kw)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(logger, command_name)

bundle_api = _AnisoAPI()
