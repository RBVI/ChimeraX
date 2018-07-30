# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "Active Tools":
            from .tool import ActiveToolsTool
            return ActiveToolsTool(session, ti.name)
        else:
            raise ValueError("trying to start unknown tool: %s" % ti.name)


    @staticmethod
    def get_class(class_name):
        if class_name in ["BasicActionsTool"]:
            from . import tool
            return getattr(tool, class_name, None)
        else:
            return None


bundle_api = _MyAPI()
