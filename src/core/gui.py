# vim: set expandtab ts=4 sw=4:

import wx


class UI(wx.App):

    def __init__(self, session):
        self.session = session
        wx.App.__init__(self)

        # splash screen
        import os.path
        splash_pic_path = os.path.join(os.path.dirname(__file__),
                                       "ui", "splash.jpg")
        import wx.lib.agw.advancedsplash as AS
        bitmap = wx.Bitmap(splash_pic_path, type=wx.BITMAP_TYPE_JPEG)

        class DebugSplash(AS.AdvancedSplash):
            def __init__(self, *args, **kw):
                def DebugPaint(*_args, **_kw):
                    self._actualPaint(*_args, **_kw)
                    self._painted = True
                self._actualPaint = self.OnPaint
                self.OnPaint = DebugPaint
                AS.AdvancedSplash.__init__(self, *args, **kw)
        self.splash = DebugSplash(None, bitmap=bitmap,
                                  agwStyle=AS.AS_CENTER_ON_SCREEN)
        splash_font = wx.Font(1, wx.SWISS, wx.NORMAL, wx.BOLD, False)
        splash_font.SetPointSize(40.0)
        self.splash.SetTextFont(splash_font)
        w, h = bitmap.GetSize()
        self.splash.SetTextPosition((0, int(0.9 * h)))
        self.splash.SetTextColour(wx.RED)
        self.splash.SetText("Initializing Chimera 2")
        self.splash._painted = False
        num_yields = 0
        while not self.splash._painted:
            wx.SafeYield()
            num_yields += 1

        self._keystroke_sinks = []

    def build(self):
        self.splash.Close()
        self.main_window = MainWindow(self, self.session.toolshed)
        self.main_window.Show(True)
        self.SetTopWindow(self.main_window)
        #from .ui.cmd_line import CmdLine
        #self.cmd_line = CmdLine(self.session)

    def deregister_for_keystrokes(self, sink, notfound_okay=False):
        """'undo' of register_for_keystrokes().  Use the same argument.
        """
        try:
            i = self._keystroke_sinks.index(sink)
        except ValueError:
            if not notfound_okay:
                raise
        else:
            self._keystroke_sinks = self._keystroke_sinks[:i] + \
                self._keystroke_sinks[i + 1:]

    def event_loop(self):
        self.MainLoop()

    def forward_keystroke(self, event):
        """forward keystroke from graphics window to most recent
           caller of 'register_for_keystrokes'
        """
        if self._keystroke_sinks:
            self._keystroke_sinks[-1].forwarded_keystroke(event)

    def register_for_keystrokes(self, sink):
        """'sink' is interested in receiving keystrokes from the main
           graphics window.  That object's 'forwarded_keystroke'
           method will be called with the keystroke event as the argument.
        """
        self._keystroke_sinks.append(sink)

    def splash_info(self, msg, step_num=None, num_steps=None):
        self.splash.SetText(msg)
        wx.SafeYield()

    def quit(self, confirm=True):
        self.main_window.close()


class MainWindow(wx.Frame):

    def __init__(self, ui, toolshed):
        wx.Frame.__init__(self, None, title="Chimera 2", size=(1000, 700))

        from wx.lib.agw.aui import AuiManager, AuiPaneInfo
        self.aui_mgr = AuiManager(self)
        self.aui_mgr.SetManagedWindow(self)

        self._build_graphics(ui)
        self._build_status()
        self._build_menus(toolshed)

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def close(self):
        self.aui_mgr.UnInit()
        del self.aui_mgr
        self.graphics_window.timer = None
        self.Destroy()

    def OnClose(self, event):
        self.close()

    def OnQuit(self, event):
        self.close()

    def _build_graphics(self, ui):
        from .ui.graphics import GraphicsWindow
        self.graphics_window = g = GraphicsWindow(self, ui)
        from wx.lib.agw.aui import AuiPaneInfo
        self.aui_mgr.AddPane(g, AuiPaneInfo().Name("GL").CenterPane())

    def _build_menus(self, toolshed):
        menu_bar = wx.MenuBar()
        self._populate_menus(menu_bar)
        self.SetMenuBar(menu_bar)
        """
        import sys
        if sys.platform == "darwin":
            self._populate_menus(wx.MenuBar.MacGetCommonMenuBar())
        """

    def _build_status(self):
        self.status_bar = self.CreateStatusBar(
            3, wx.STB_SIZEGRIP | wx.STB_SHOW_TIPS | wx.STB_ELLIPSIZE_MIDDLE
            | wx.FULL_REPAINT_ON_RESIZE)
        self.status_bar.SetStatusWidths([-24, -30, -2])
        self.status_bar.SetStatusText("Status", 0)
        self.status_bar.SetStatusText("Welcome to Chimera 2", 1)
        self.status_bar.SetStatusText("", 2)

    def _populate_menus(self, menu_bar):
        import sys
        if sys.platform != "darwin":
            file_menu = wx.Menu()
            item = file_menu.Append(wx.ID_EXIT, "Quit", "Quit application")
            menu_bar.Append(file_menu, "&File")
            self.Bind(wx.EVT_MENU, self.OnQuit, item)
