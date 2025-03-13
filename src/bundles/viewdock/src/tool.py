from chimerax.core.tools import ToolInstance

class ViewDockTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    def __init__(self, session, tool_name, structures):
        super().__init__(session, tool_name)
        self.display_name = "ViewDock"

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)

        self.setup(structures)

        self.tool_window.manage('side')

    def setup(self, structures):
        self.session.logger.info(f"ViewDockTool setup with structures: {structures}")