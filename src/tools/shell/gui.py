# vim: set expandtab shiftwidth=4 softtabstop=4:
import wx
from chimerax.core.tools import ToolInstance


class ShellUI(ToolInstance):

    # shell tool does not participate in sessions
    SESSION_ENDURING = True
    SESSION_SKIP = True
    SIZE = (500, 500)

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Tool UI"
        # in this case), so only override if different name desired
        self.display_name = "ChimeraX Python Shell"
        from chimerax.core.ui.gui import MainToolWindow
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

    def delete(self):
        self.shell.redirectStdin(False)
        self.shell.redirectStdout(False)
        self.shell.redirectStderr(False)
        super().delete()
