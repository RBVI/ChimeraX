# vi: set expandtab ts=4 sw=4:

import wx


class UI(wx.App):

    def __init__(self, session):
        self.is_gui = True
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
        self.main_window = MainWindow(self, self.session)
        self.main_window.Show(True)
        self.SetTopWindow(self.main_window)

    def close_splash(self):
        self.splash.Close()

    def create_child_tool_window(self, tool_instance, title=None,
            size=None, destroy_hides=False):
        return self.main_window._create_child_tool_window(tool_instance,
            title, size, destroy_hides)

    def create_main_tool_window(self, tool_instance, size=None,
            destroy_hides=False):
        return self.main_window._create_main_tool_window(tool_instance, size,
            destroy_hides)

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
# This turns Python deprecation warnings into exceptions, useful for debugging.
#        import warnings
#        warnings.filterwarnings('error')

        redirect_stdio_to_logger(self.session.logger)
        self.MainLoop()
        self.session.logger.clear()

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
        self.session.logger.status("Exiting ...", blank_after=0)
        self.session.logger.clear()    # clear logging timers
        self.main_window.close()

    def thread_safe(self, func, *args, **kw):
        """Call function 'func' in a thread-safe manner
        """
        wx.CallAfter(func, *args, **kw)

def redirect_stdio_to_logger(logger):
    # Redirect stderr to log
    class LogStdout:
        def __init__(self, logger):
            self.logger = logger
            self.closed = False
        def write(self, s):
            self.logger.info(s, add_newline = False)
        def flush(self):
            return
    LogStderr = LogStdout
    import sys
    sys.orig_stdout = sys.stdout
    sys.stdout = LogStdout(logger)
    # TODO: Should raise an error dialog for exceptions, but traceback
    #       is written to stderr with a separate call to the write() method
    #       for each line, making it hard to aggregate the lines into one
    #       error dialog.
    sys.orig_stderr = sys.stderr
    sys.stderr = LogStderr(logger)

from .logger import PlainTextLog
class MainWindow(wx.Frame, PlainTextLog):

    def __init__(self, ui, session):
        wx.Frame.__init__(self, None, title="Chimera 2", size=(1000, 700))

        from wx.lib.agw.aui import AuiManager, AuiPaneInfo, EVT_AUI_PANE_CLOSE
        self.aui_mgr = AuiManager(self)
        self.aui_mgr.SetManagedWindow(self)

        self.tool_pane_to_window = {}
        self.tool_instance_to_windows = {}

        self._build_graphics(ui)
        self._build_status()
        self._build_menus(session)

        session.logger.add_log(self)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(EVT_AUI_PANE_CLOSE, self.on_pane_close)

    def close(self):
        self.aui_mgr.UnInit()
        del self.aui_mgr
        if self.graphics_window.timer is not None:
            self.graphics_window.timer.Stop()
        self.Destroy()

    def log(self, *args, **kw):
        return False

    def on_close(self, event):
        self.close()

    def on_edit(self, event, func):
        widget = self.FindFocus()
        if widget and hasattr(widget, func):
            getattr(widget, func)()
        else:
            event.Skip()

    def on_open(self, event, session):
        from . import io
        dlg = wx.FileDialog(self, "Open file",
            wildcard=io.wx_open_file_filter(all=True),
            style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST|wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return

        paths = dlg.GetPaths()
        session.models.open(paths)

    def on_pane_close(self, event):
        pane_info = event.GetPane()
        tool_window = self.tool_pane_to_window[pane_info.window]
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        destroy_hides = tool_window.destroy_hides
        if tool_window.destroy_hides:
            tool_window.shown = False
            event.Veto()
        else:
            del self.tool_pane_to_window[tool_window.ui_area]
            tool_window.destroy(from_destructor=True)
            all_windows.remove(tool_window)

        if is_main_window:
            for window in all_windows:
                if destroy_hides:
                    window.shown = False
                else:
                    del self.tool_pane_to_window[window.ui_area]
                    window.destroy(from_destructor=True)
            if not destroy_hides:
                del self.tool_instance_to_windows[tool_instance]

    def on_quit(self, event):
        self.close()

    def on_save(self, event, ses):
        self.save_dialog.display(self, ses)

    def status(self, msg, color, secondary):
        wx.CallAfter(self._main_thread_status, msg, color, secondary)

    def _build_graphics(self, ui):
        from .ui.graphics import GraphicsWindow
        self.graphics_window = g = GraphicsWindow(self, ui)
        from wx.lib.agw.aui import AuiPaneInfo
        self.aui_mgr.AddPane(g, AuiPaneInfo().Name("GL").CenterPane())
        from .ui.save_dialog import SaveDialog, ImageSaver
        self.save_dialog = SaveDialog(self)
        ImageSaver(self.save_dialog).register()

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

    def _create_child_tool_window(self, tool_instance, title, size,
            destroy_hides):
        if tool_instance not in self.tool_instance_to_windows:
            raise ValueError("Tool {} trying to create child window without "
                "first creating main window".format(tool_instance.display_name))
        if title is None:
            title = tool_instance.display_name
        tw = ToolWindow(tool_instance, title, self, size, destroy_hides)
        self.tool_pane_to_window[tw.ui_area] = tw
        self.tool_instance_to_windows[tool_instance].append(tw)
        return tw

    def _create_main_tool_window(self, tool_instance, size, destroy_hides):
        if tool_instance in self.tool_instance_to_windows:
            raise ValueError("Tool {} trying to create multiple main windows"
                .format(tool_instance.display_name))
        tw = ToolWindow(tool_instance, tool_instance.display_name, self, size,
            destroy_hides)
        self.tool_pane_to_window[tw.ui_area] = tw
        self.tool_instance_to_windows[tool_instance] = [tw]
        return tw

    def _main_thread_status(self, msg, color, secondary):
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

    def _populate_menus(self, menu_bar, session):
        import sys
        file_menu = wx.Menu()
        menu_bar.Append(file_menu, "&File")
        item = file_menu.Append(wx.ID_OPEN, "Open...\tCtrl+O", "Open input file")
        self.Bind(wx.EVT_MENU, lambda evt, ses=session: self.on_open(evt, ses),
            item)
        item = file_menu.Append(wx.ID_ANY, "Save...\tCtrl+S", "Save output file")
        self.Bind(wx.EVT_MENU, lambda evt, ses=session: self.on_save(evt, ses),
            item)
        item = file_menu.Append(wx.ID_EXIT, "Quit\tCtrl-Q", "Quit application")
        self.Bind(wx.EVT_MENU, self.on_quit, item)
        edit_menu = wx.Menu()
        menu_bar.Append(edit_menu, "&Edit")
        for wx_id, letter, func in [
                (wx.ID_CUT, "X", "Cut"),
                (wx.ID_COPY, "C", "Copy"),
                (wx.ID_PASTE, "V", "Paste")]:
            self.Bind(wx.EVT_MENU, lambda e, f=func: self.on_edit(e, f),
                edit_menu.Append(wx_id, "{}\tCtrl-{}".format(func, letter),
                "{} text".format(func)))
        tools_menu = wx.Menu()
        categories = {}
        for ti in session.toolshed.tool_info():
            for cat in ti.menu_categories:
                categories.setdefault(cat, {})[ti.display_name] = ti
        for cat in sorted(categories.keys()):
            if cat == "Hidden":
                continue
            cat_menu = wx.Menu()
            tools_menu.Append(wx.ID_ANY, cat, cat_menu)
            cat_info = categories[cat]
            for tool_name in sorted(cat_info.keys()):
                ti = cat_info[tool_name]
                item = cat_menu.Append(wx.ID_ANY, tool_name)
                cb = lambda evt, ses=session, ti=ti: ti.start(ses)
                self.Bind(wx.EVT_MENU, cb, item)
        menu_bar.Append(tools_menu, "&Tools")

    def _tool_window_request_shown(self, tool_window, shown):
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        tool_window._set_shown(shown)
        if is_main_window:
            for window in all_windows[1:]:
                window._set_shown(shown)

class ToolWindow:
    """An area that a tool can populate with widgets.

    Should not be created directly by the tool but instead should
    be created via the UI class, either its method creat_main_tool_window
    or create_child_tool_window."""

    placements = ["right", "left", "top", "bottom"]

    def __init__(self, tool_instance, title, main_window, size, destroy_hides):
        """ 'ui_area' is the parent to all the tool's widgets;
            Call 'manage' once the widgets are set up to put the
            tool into the main window.
        """
        try:
            self.__toolkit = _Wx(self, title, main_window, size, destroy_hides)
        except ImportError:
            # browser version
            raise NotImplementedError("Browser tool API not implemented")
        self.ui_area = self.__toolkit.ui_area
        self.tool_instance = tool_instance
        self.main_window = main_window

    def destroy(self, **kw):
        self.__toolkit.destroy(**kw)
        self.__toolkit = None

    def manage(self, placement, fixed_size = False):
        """ Tool will be docked into main window on the side indicated by
            'placement' (which should be a value from self.placements or None);
            if 'placement' is None, the tool will be detached from the main
            window.
        """
        self.__toolkit.manage(placement, fixed_size)

    def get_destroy_hides(self):
        return self.__toolkit.destroy_hides

    destroy_hides = property(get_destroy_hides)

    def get_shown(self):
        return self.__toolkit.shown

    def set_shown(self, shown):
        if shown == self.__toolkit.shown:
            return
        self.main_window._tool_window_request_shown(self, shown)

    shown = property(get_shown, set_shown)

    def _set_shown(self, shown):
        self.__toolkit.shown = shown

class _Wx:

    def __init__(self, tool_window, title, main_window, size, destroy_hides):
        import wx
        self.tool_window = tool_window
        self.title = title
        self.destroy_hides = destroy_hides
        self.main_window = mw = main_window
        wx_sides = [wx.RIGHT, wx.LEFT, wx.TOP, wx.BOTTOM]
        self.placement_map = dict(zip(self.tool_window.placements, wx_sides))
        from wx.lib.agw.aui import AUI_DOCK_RIGHT, AUI_DOCK_LEFT, \
            AUI_DOCK_TOP, AUI_DOCK_BOTTOM
        self.aui_side_map = dict(zip(wx_sides, [AUI_DOCK_RIGHT, AUI_DOCK_LEFT,
            AUI_DOCK_TOP, AUI_DOCK_BOTTOM]))
        if not mw:
            raise RuntimeError("No main window or main window dead")
        if size is None:
            size = wx.DefaultSize
        class WxToolPanel(wx.Panel):
            def __init__(self, parent, destroy_hides=destroy_hides, **kw):
                self._destroy_hides = destroy_hides
                wx.Panel.__init__(self, parent, **kw)

        self.ui_area = WxToolPanel(mw, name=title, size=size)
        mw.tool_pane_to_window[self.ui_area] = tool_window
        self._pane_info = None

    def destroy(self, from_destructor=False):
        if not self.tool_window:
            # already destroyed
            return
        if not from_destructor:
            del self.main_window.tool_pane_to_window[self.ui_area]
            self.ui_area.Destroy()
        # free up references
        self.tool_window = None
        self.main_window = None
        self._pane_info = None

    def manage(self, placement, fixed_size = False):
        import wx
        placements = self.tool_window.placements
        if placement is None:
            side = wx.RIGHT
        else:
            if placement not in placements:
                raise ValueError("placement value must be one of: {}, or None"
                    .format(", ".join(placements)))
            else:
                side = self.placement_map[placement]

        mw = self.main_window
        # commented out the layering code, since though it does make
        # the newly added tool larger since it doesn't share a layer,
        # it typically shrinks the graphics window, which is probably
        # a bigger downside
        """
        # find the outermost layer in that direction, and put it past that
        layer = -1
        aui_side = self.aui_side_map[side]
        for pane_info in mw.aui_mgr.GetAllPanes():
            if pane_info.dock_direction == aui_side:
                layer = max(layer, pane_info.dock_layer)
        """
        mw.aui_mgr.AddPane(self.ui_area, side, self.title)
        if fixed_size:
            mw.aui_mgr.GetPane(self.ui_area).Fixed()
        """
        mw.aui_mgr.GetPane(self.ui_area).Layer(layer+1)
        """
        mw.aui_mgr.Update()
        if placement is None:
           mw.aui_mgr.GetPane(self.ui_area).Float()

        if not self.destroy_hides:
            mw.aui_mgr.GetPane(self.ui_area).DestroyOnClose()

    def getShown(self):
        return self.ui_area.Shown

    def setShown(self, shown):
        if shown == self.ui_area.Shown:
            return
        aui_mgr = self.main_window.aui_mgr
        if shown:
            if self._pane_info:
                # has been hidden at least once
                aui_mgr.AddPane(self.ui_area, self._pane_info)
                self._pane_info = None
        else:
            self._pane_info = aui_mgr.GetPane(self.ui_area)
            aui_mgr.DetachPane(self.ui_area)
        aui_mgr.Update()

        self.ui_area.Shown = shown

    shown = property(getShown, setShown)
