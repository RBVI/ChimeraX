# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _AnisoAPI(BundleAPI):

    """
    @staticmethod
    def get_class(class_name):
        from . import _data
        if class_name == 'NucleotideState':
            return _data.NucleotideState
        if class_name == 'Params':
            return _data.Params
        return None

    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import NucleotidesTool
        return NucleotidesTool(session, tool_name, **kw)
    """

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(logger)

bundle_api = _AnisoAPI()
