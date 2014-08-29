# vim: set expandtab ts=4 sw=4:

import wx

class Hydra_App(wx.App):

    def __init__(self, argv, session):
        self.session = session
        wx.App.__init__(self)

    def OnInit(self):
        self.main_window = MainWindow(self.session)
        self.main_window.Show(True)
        self.SetTopWindow(self.main_window)
        return True

class _FileDrop(wx.FileDropTarget):

    def __init__(self, session):
        wx.FileDropTarget.__init__(self)
        self.session = session

    def OnDropFiles(self, x, y, file_names):
        from ...files.opensave import open_files
        open_files(file_names, self.session)
        self.session.main_window.show_graphics()

class MainWindow(wx.Frame):

    def __init__(self, session):
        wx.Frame.__init__(self, None, title="Chimera 2")

        from wx.lib.agw.aui import AuiManager, AuiPaneInfo
        self.aui_mgr = AuiManager(self)
        self.aui_mgr.SetManagedWindow(self)

        from .graphics import GraphicsWindow
        # View is a base class of Graphics Window
        self.view = GraphicsWindow(session, self)
        self.aui_mgr.AddPane(self.view, AuiPaneInfo().Name("GL").CenterPane())
        self.aui_mgr.Update()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.SetDropTarget(_FileDrop(session))

    def OnClose(self, event):
        self.aui_mgr.UnInit()
        del self.aui_mgr
        self.Destroy()
