# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

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
        from chimerax.core import window_sys
        if window_sys == "wx":
            kw = { 'size': (500, 500) }
        else:
            kw = {}
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, **kw)
        parent = self.tool_window.ui_area
        # UI content code
        if window_sys == "wx":
            import wx
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
        else:
            from ipykernel.ipkernel import IPythonKernel
            save_ns = IPythonKernel.user_ns
            IPythonKernel.user_ns = { 'session': session }
            from qtconsole.inprocess import QtInProcessKernelManager
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel()
            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            from qtconsole.rich_jupyter_widget import RichJupyterWidget
            self.shell = RichJupyterWidget(parent)
            def_banner = self.shell.banner
            self.shell.banner = "{}\nCurrent ChimeraX session available as 'session'.\n\n".format(
                def_banner)
            self.shell.kernel_manager = kernel_manager
            self.shell.kernel_client = kernel_client
            IPythonKernel.user_ns = save_ns

            from PyQt5.QtWidgets import QHBoxLayout
            layout = QHBoxLayout()
            layout.addWidget(self.shell)
            layout.setStretchFactor(self.shell, 1)
            parent.setLayout(layout)
        self.tool_window.manage(placement=None)
        if window_sys == "wx":
            self.shell.setFocus()

    def delete(self):
        from chimerax.core import window_sys
        if window_sys == "wx":
            self.shell.redirectStdin(False)
            self.shell.redirectStdout(False)
            self.shell.redirectStderr(False)
        super().delete()
