# vim: set expandtab ts=4 sw=4:

import wx

class MainWindow(wx.Frame):
    
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, title="Chimera 2")

        from wx.lib.agw.aui import AuiManager, AuiPaneInfo
        self.aui_mgr = AuiManager(self)
        self.aui_mgr.SetManagedWindow(self)

        gl_panel = self._create_opengl()
        self.aui_mgr.AddPane(gl_panel, AuiPaneInfo().Name("GL").CenterPane())

        self.aui_mgr.Update()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def _create_opengl(self):
        panel = wx.Panel(self)
        from wx.glcanvas import GLCanvas
        self.gl_canvas = GLCanvas(panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.gl_canvas, 1, wx.EXPAND)
        panel.SetSizerAndFit(sizer)
        return panel

    def OnClose(self, event):
        self.aui_mgr.UnInit()
        del self.aui_mgr
        self.Destroy()

app = wx.App()
main_window = MainWindow(None)
