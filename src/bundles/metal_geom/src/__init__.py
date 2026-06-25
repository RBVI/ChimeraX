# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MetalGeomAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import MetalGeomTool
        return MetalGeomTool(session, tool_name, **kw)

bundle_api = _MetalGeomAPI()
