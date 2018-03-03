# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "Targets":
            from .cmd import target
            return target(session, name=ti.name)
        else:
            raise ValueError("trying to start unknown tool: %s" % ti.name)

    @staticmethod
    def register_command(bi, ci, logger):
        func_attr = ci.name.replace(' ', '_')
        desc_attr = func_attr + "_desc"
        from . import cmd
        func = getattr(cmd, func_attr)
        desc = getattr(cmd, desc_attr)
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        from chimerax.core.commands import register
        register(ci.name, desc, func)


    @staticmethod
    def get_class(class_name):
        if class_name in ["TargetsTool"]:
            from . import tool
            return getattr(tool, class_name, None)
        else:
            return None


bundle_api = _MyAPI()
