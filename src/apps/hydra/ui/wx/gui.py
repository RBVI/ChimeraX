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

def start_event_loop(app):
    return app.MainLoop()

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

        session.main_window = self # needed for ToolWindow init
        from ..tool_api import ToolWindow
        self._text_window = ToolWindow("Messages", "General", session,
            placement="bottom", destroy_hides=True)
        from wx.html2 import WebView
        self._text = WebView.New(self._text_window.ui_area)
        self._text_window.shown = False

        self._shortcuts_enabled = False
        from .cmd_line import CmdLine
        self._command_line = CmdLine(session)

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.SetDropTarget(_FileDrop(session))

    def OnClose(self, event):
        self.aui_mgr.UnInit()
        del self.aui_mgr
        self.view.timer = None
        self.Destroy()

    def show(self):
        self.Show()

    def show_graphics(self):
        pass # always showing

    def showing_graphics(self):
        return True

    def showing_text(self):
        return self._text_window.shown

    def show_text(self, text=None, url=None, html=False, id=None,
            anchor_callback=None, open_links=False, scroll_to_end=False):
        t = self._text
        if text is not None:
            t.SetPage(text, "")
        elif url is not None:
            t.LoadPage(url)

        self.text_id = id
        t.shown = True
        if scroll_to_end:
            while t.ScrollPages(1):
                pass

def set_window_size(session, width = None, height = None):

    view = session.main_window.view
    if width is None and height is None:
        from .. import show_status, show_info
        msg = 'Graphics size %d, %d' % view.GetClientSize()
        show_status(msg)
        show_info(msg)
    else:
        view.SetClientSize(width, height)
