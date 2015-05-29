# vi: set expandtab shiftwidth=4 softtabstop=4:

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

    SESSION_ENDURING = False    # default
    SIZE = (500, 25)
    VERSION = 1

    def __init__(self, session, tool_info):
        super().__init__(session, tool_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Tool UI"
        # in this case), so only override if different name desired
        self.display_name = "custom name for running tool"
        if session.ui.is_gui:
            self.tool_window = session.ui.creat_main_tool_window(
                self, size=self.SIZE)
            parent = self.tool_window.ui_area
            # UI content code
            self.tool_window.manage(placement="bottom")
            # Add to running tool list for session if tool should be saved
            # in and restored from session and scenes
        session.tools.add([self])

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        version = self.VERSION
        data = {}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import RestoreError
        if version != self.VERSION or len(data) > 0:
            raise RestoreError("unexpected version or data")
        if phase == self.CREATE_PHASE:
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
        session = self.session()
        if session.ui.is_gui:
            self.tool_window.shown = False
            self.tool_window.destroy()
        self.session.tools.remove([self])
        super().delete()

    def display(self, b):
        self.tool_window.shown = b
