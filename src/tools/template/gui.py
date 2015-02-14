# vim: set expandtab ts=4 sw=4:

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
# Since ToolInstance derives from core.session.State, which
# is an abstract base class, ToolUI classes must implement
#   "take_snapshot" - return current state for saving
#   "restore_snapshot" - restore from given state
#   "reset_state" - reset to data-less state
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimera.core.tools import ToolInstance


class ToolUI(ToolInstance):

    SIZE = (500, 25)
    VERSION = 1

    def __init__(self, session):
        super().__init__(session)
        import weakref
        self._session = weakref.ref(session)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("TOOL_NAME", session, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        self.tool_window.manage(placement="bottom")
        # Add to running tool list for session (not required)
        session.tools.add([self])

    def OnEnter(self, event):
        session = self._session()  # resolve back reference
        # Handle event

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        version = self.VERSION
        data = {}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.VERSION or len(data) > 0:
            raise RuntimeError("unexpected version or data")
        from chimera.core.session import State
        if phase == State.PHASE1:
            # Restore all basic-type attributes
            pass
        else:
            # Resolve references to objects
            pass

    def reset_state(self):
        pass

    #
    # Override ToolInstance delete method to clean up
    #
    def delete(self):
        session = self._session()  # resolve back reference
        self.tool_window.shown = False
        self.tool_window.destroy()
        session.tools.remove([self])
        super().delete()

    def display(self, b):
        self.tool_window.shown = b

    def display_name(self):
        return "custom name for running tool"
