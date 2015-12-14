# vim: set expandtab shiftwidth=4 softtabstop=4:

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
# Since ToolInstance derives from core.session.State, which
# is an abstract base class, ToolUI classes must implement
#   "take_snapshot" - return current state for saving
#   "restore_snapshot_init" - restore from given state
#   "reset_state" - reset to data-less state
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance


class ToolUI(ToolInstance):

    SESSION_ENDURING = False
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    SIZE = (500, 25)

    def __init__(self, session, bundle_info, *, restoring=False):
        if not restoring:
            ToolInstance.__init__(self, session, bundle_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Tool UI"
        # in this case), so only override if different name desired
        self.display_name = "custom name for running tool"
        if session.ui.is_gui:
            from chimerax.core.ui import MainToolWindow
            self.tool_window = MainToolWindow(self, size=self.SIZE)
            self.tool_window.manage(placement="bottom")
            parent = self.tool_window.ui_area
            # TODO: UI content code goes here

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        data = {
            "ti": ToolInstance.take_snapshot(self, session, flags),
            "shown": self.tool_window.shown
        }
        return self.bundle_info.session_write_version, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        if version not in bundle_info.session_versions:
            from chimerax.core.state import RestoreError
            raise RestoreError("unexpected version")
        ti_version, ti_data = data["ti"]
        ToolInstance.restore_snapshot_init(
            self, session, bundle_info, ti_version, ti_data)
        self.__init__(session, bundle_info, restoring=True)
        self.display(data["shown"])

    def reset_state(self, session):
        pass
