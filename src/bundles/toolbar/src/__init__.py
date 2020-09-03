#
# vim: set expandtab shiftwidth=4 softtabstop=4:
#

from chimerax.core.toolshed import BundleAPI


class _ToolbarAPI(BundleAPI):

    api_version = 1

    # Override method
    @staticmethod
    def start_tool(session, bi, ti):
        from . import tool
        if ti.name == "Toolbar":
            return tool.get_toolbar_singleton(session)
        if ti.name == "Customize Toolbar":
            return tool.CustomizeTool(session, ti.name)
        raise ValueError("trying to start unknown tool: %s" % ti.name)

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize schemes manager"""
        if name == "toolbar":
            if hasattr(session, 'toolbar'):
                # TODO: does this happen?
                session.toolbar.clear()
            else:
                from .manager import ToolbarManager
                session.toolbar = ToolbarManager(session, name)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        """Run toolbar provider"""
        from . import providers
        providers.run_provider(session, name)


bundle_api = _ToolbarAPI()
