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

    def build(self, load_tools):
        self.splash.Close()
        self.main_window = MainWindow(self, self.session)
        self.main_window.Show(True)
        self.SetTopWindow(self.main_window)
        if load_tools:
            from .toolshed import ToolshedError
            for ti in self.session.toolshed.tool_info():
                try:
                    ti.start(self.session)
                except ToolshedError as e:
                    self.session.logger.info("Tool \"%s\" failed to start"
                                             % ti.name)
                    print("{}".format(e))

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

    def thread_safe(self, func, *args, **kw):
        """Call function 'func' in a thread-safe manner
        """
        wx.CallAfter(func, *args, **kw)


from .logger import PlainTextLog
class MainWindow(wx.Frame, PlainTextLog):

    def __init__(self, ui, session):
        wx.Frame.__init__(self, None, title="Chimera 2", size=(1000, 700))

        from wx.lib.agw.aui import AuiManager, AuiPaneInfo, EVT_AUI_PANE_CLOSE
        self.aui_mgr = AuiManager(self)
        self.aui_mgr.SetManagedWindow(self)

        self.pane_to_tool_window = {}

        self._build_graphics(ui)
        self._build_status()
        self._build_menus(session)

        session.logger.add_log(self)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(EVT_AUI_PANE_CLOSE, self.OnPaneClose)

    def close(self):
        self.aui_mgr.UnInit()
        del self.aui_mgr
        self.graphics_window.timer = None
        self.Destroy()

    def log(self, *args, **kw):
        return False

    def OnClose(self, event):
        self.close()

    def OnOpen(self, event, session):
        from . import io
        dlg = wx.FileDialog(self, "Open file",
            wildcard=io.wx_open_file_filter(all=True),
            style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST|wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return

        for p in dlg.GetPaths():
            session.models.open(p)

    def OnPaneClose(self, event):
        pane_info = event.GetPane()
        tool_window = self.pane_to_tool_window[pane_info.window]
        if tool_window.destroy_hides:
            tool_window.shown = False
            event.Veto()
        else:
            tool_window.destroy(from_destructor=True)

    def OnQuit(self, event):
        self.close()

    def status(self, msg, color, secondary):
        if self._initial_status_kludge == True:
            self._initial_status_kludge = False
            self.status_bar.SetStatusText("", 1)

        if secondary:
            secondary_text = msg
        else:
            secondary_text = self.status_bar.GetStatusText(1)
        secondary_size = wx.Window.GetTextExtent(self, secondary_text)
        self.status_bar.SetStatusWidths([-1, secondary_size.width, 0])

        color_db = wx.ColourDatabase()
        wx_color = color_db.Find(color)
        if not wx_color.IsOk:
            wx_color = wx_color.Find("black")
        self.status_bar.SetForegroundColour(wx_color)

        if secondary:
            self.status_bar.SetStatusText(msg, 1)
        else:
            self.status_bar.SetStatusText(msg, 0)

    def _build_graphics(self, ui):
        from .ui.graphics import GraphicsWindow
        self.graphics_window = g = GraphicsWindow(self, ui)
        from wx.lib.agw.aui import AuiPaneInfo
        self.aui_mgr.AddPane(g, AuiPaneInfo().Name("GL").CenterPane())

    def _build_menus(self, session):
        menu_bar = wx.MenuBar()
        self._populate_menus(menu_bar, session)
        self.SetMenuBar(menu_bar)

    def _build_status(self):
        # as a kludge, use 3 fields so that I can center the initial
        # "Welcome" text
        self.status_bar = self.CreateStatusBar(3,
            wx.STB_SIZEGRIP | wx.STB_SHOW_TIPS | wx.STB_ELLIPSIZE_MIDDLE
            | wx.FULL_REPAINT_ON_RESIZE)
        greeting = "Welcome to Chimera 2"
        greeting_size = wx.Window.GetTextExtent(self, greeting)
        self.status_bar.SetStatusWidths([-1, greeting_size.width, -1])
        self.status_bar.SetStatusText("", 0)
        self.status_bar.SetStatusText(greeting, 1)
        self.status_bar.SetStatusText("", 2)
        self._initial_status_kludge = True

    def _populate_menus(self, menu_bar, session):
        import sys
        file_menu = wx.Menu()
        menu_bar.Append(file_menu, "&File")
        item = file_menu.Append(wx.ID_OPEN, "Open...", "Open input file")
        self.Bind(wx.EVT_MENU, lambda evt, ses=session: self.OnOpen(evt, ses),
            item)
        if sys.platform != "darwin":
            item = file_menu.Append(wx.ID_EXIT, "Quit", "Quit application")
            self.Bind(wx.EVT_MENU, self.OnQuit, item)
        tools_menu = wx.Menu()
        categories = {}
        for ti in session.toolshed.tool_info():
            for cat in ti.menu_categories:
                categories.setdefault(cat, {})[ti.display_name] = ti
        for cat in sorted(categories.keys()):
            cat_menu = wx.Menu()
            tools_menu.Append(wx.ID_ANY, cat, cat_menu)
            cat_info = categories[cat]
            for tool_name in sorted(cat_info.keys()):
                ti = cat_info[tool_name]
                item = cat_menu.Append(wx.ID_ANY, tool_name)
                cb = lambda evt, ses=session, ti=ti: ti.start(ses)
                self.Bind(wx.EVT_MENU, cb, item)
        menu_bar.Append(tools_menu, "&Tools")
