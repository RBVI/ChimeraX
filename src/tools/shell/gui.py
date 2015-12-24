# vim: set expandtab shiftwidth=4 softtabstop=4:
import wx
from chimerax.core.tools import ToolInstance


class ShellUI(ToolInstance):

    # shell tool does not participate in sessions
    SESSION_ENDURING = True
    SESSION_SKIP = True
    SIZE = (500, 500)

    def __init__(self, session, bundle_info, *, restoring=False):
        if not restoring:
            ToolInstance.__init__(self, session, bundle_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Tool UI"
        # in this case), so only override if different name desired
        self.display_name = "ChimeraX Python Shell"
        from chimerax.core.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        from wx.py.shell import Shell
        self.shell = Shell(
            parent, -1, size=self.SIZE, locals={
                'session': session
            },
            introText='Use "session" to access the current session.')
        self.shell.redirectStdin(True)
        self.shell.redirectStdout(True)
        self.shell.redirectStderr(True)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.shell, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement=None)
        self.shell.setFocus()

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

    def delete(self):
        self.shell.redirectStdin(False)
        self.shell.redirectStdout(False)
        self.shell.redirectStderr(False)
        super().delete()
