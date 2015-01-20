# vim: set expandtab ts=4 sw=4:


class ToolUI:

    SIZE = (500, 25)

    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("TOOL_NAME", "TOOL_CATEGORY",
                                      session, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        self.tool_window.manage(placement="bottom")
        # Add to running tool list for session (not required)
        session.tools.add([self])

    def OnEnter(self, event):
        session = self._session()  # resolve back reference
        # Handle event
