# vi: set expandtab ts=4 sw=4:
"""
gui: Main Chimera2 user interface
==================================

The principal class that tool writers will use from this module is
:py:class:`MainToolWindow`, which is either instantiated directly, or
subclassed and instantiated to create the tool's main window.
Additional windows are created by calling that instance's
:py:meth:`MainToolWindow.create_child_window` method.

Rarely, methods are used from the :py:class:`UI` class to get
keystrokes typed to the main graphics window, or to execute code
in a thread-safe manner.  The UI instance is accessed as session.ui.
"""

import wx


class UI(wx.App):
    """Main Chimera2 user interface

       The only methods that tools might directly use are:

       register(/deregister)_for_keystrokes
        For the rare tool that might want to get keystrokes that are
        typed when focus is in the main graphics window

       thread_safe
        To execute a function in a thread-safe manner
       """


    def __init__(self, session):
        self.is_gui = True
        self.session = session
        wx.App.__init__(self)

        # splash screen
        import os.path
        splash_pic_path = os.path.join(os.path.dirname(__file__),
                                       "splash.jpg")
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
        self.splash.SetText("Initializing Chimera2")
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
        self.main_window.Raise()

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

    def remove_tool(self, tool_instance):
        self.main_window.remove_tool(tool_instance)

    def set_tool_shown(self, tool_instance, shown):
        self.main_window.set_tool_shown(tool_instance, shown)

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

from ..logger import PlainTextLog
class MainWindow(wx.Frame, PlainTextLog):

    def __init__(self, ui, session):
        # make main window 2/3 of full screen
        x, y = wx.DisplaySize()
        req_size = ((2*x)/3, (2*y)/3)
        wx.Frame.__init__(self, None, title="Chimera2", size=req_size)

        from wx.lib.agw.aui import AuiManager, EVT_AUI_PANE_CLOSE
        self.aui_mgr = AuiManager(self)
        self.aui_mgr.SetManagedWindow(self)
        self.aui_mgr.SetDockSizeConstraint(0.5, 0.4)

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
        from .. import io
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
        close_destroys = tool_window.close_destroys
        if is_main_window and close_destroys:
            tool_instance.delete()
            return
        if tool_window.close_destroys:
            del self.tool_pane_to_window[tool_window.ui_area]
            tool_window._destroy()
            all_windows.remove(tool_window)
        else:
            tool_window.shown = False
            event.Veto()

        if is_main_window:
            # close hides, since close destroys is handled above
            for window in all_windows:
                window._prev_shown = window.shown
                window.shown = False

    def on_quit(self, event):
        self.close()

    def on_save(self, event, ses):
        self.save_dialog.display(self, ses)

    def remove_tool(self, tool_instance):
        tool_windows = self.tool_instance_to_windows.get(tool_instance, None)
        if tool_windows:
            for tw in tool_windows:
                tw._mw_set_shown(False)
                del self.tool_pane_to_window[tw.ui_area]
                tw._destroy()
            del self.tool_instance_to_windows[tool_instance]

    def set_tool_shown(self, tool_instance, shown):
        tool_windows = self.tool_instance_to_windows.get(tool_instance, None)
        if tool_windows:
            tool_windows[0].shown = shown

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
        self.status_bar.Update()


    def _build_graphics(self, ui):
        from .graphics import GraphicsWindow
        self.graphics_window = g = GraphicsWindow(self, ui)
        from wx.lib.agw.aui import AuiPaneInfo
        self.aui_mgr.AddPane(g, AuiPaneInfo().Name("GL").CenterPane())
        from .save_dialog import MainSaveDialog, ImageSaver
        self.save_dialog = MainSaveDialog(self)
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
        greeting = "Welcome to Chimera2"
        greeting_size = wx.Window.GetTextExtent(self, greeting)
        self.status_bar.SetStatusWidths([-1, greeting_size.width, -1])
        self.status_bar.SetStatusText("", 0)
        self.status_bar.SetStatusText(greeting, 1)
        self.status_bar.SetStatusText("", 2)
        self._initial_status_kludge = True

    def _new_tool_window(self, tw):
        self.tool_pane_to_window[tw.ui_area] = tw
        self.tool_instance_to_windows.setdefault(tw.tool_instance,[]).append(tw)

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

        help_menu = wx.Menu()
        menu_bar.Append(help_menu, "&Help")
        for entry, topic in (('User Guide', 'user'),
                           ('Quick Start Guide', 'quickstart'),
                           ('Programming Guide', 'devel'),
                           ('PDB images command', 'pdbimages')):
            item = help_menu.Append(wx.ID_ANY, entry, "Show " + entry)
            def cb(evt, ses=session, t=topic):
                from chimera.core.commands import run
                run(ses, 'help sethome help:%s' % t)
            self.Bind(wx.EVT_MENU, cb, item)

    def _tool_window_destroy(self, tool_window):
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        if is_main_window:
            tool_instance.delete()
            return
        del self.tool_pane_to_window[tool_window.ui_area]
        tool_window._destroy()
        all_windows.remove(tool_window)

    def _tool_window_request_shown(self, tool_window, shown):
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        tool_window._mw_set_shown(shown)
        if is_main_window:
            for window in all_windows[1:]:
                if shown:
                    # if child window has a '_prev_shown' attr, then it was
                    # around when main window was closed/hidden, possibly
                    # show it and forget the _prev_shown attrs
                    if hasattr(window, '_prev_shown'):
                        if window._prev_shown:
                            window._mw_set_shown(True)
                        delattr(window, '_prev_shown')
                else:
                    window._mw_set_shown(False)

class ToolWindow:
    """An area that a tool can populate with widgets.

    This class is not used directly.  Instead, a tool makes its main
    window by instantiating the :py:class:`MainToolWindow` class
    (or a subclass thereof), and any subwindows by calling that class's
    :py:meth:`~MainToolWindow.create_child_window` method.

    The window's :py:attr:`ui_area` attribute is the parent to all the tool's
    widgets for this window.  Call :py:meth:`manage` once the widgets
    are set up to show the tool window in the main interface.
    """

    #: Where the window is placed in the main interface
    placements = ["right", "left", "top", "bottom"]

    #: Whether closing this window destroys it or hides it.
    #: If it destroys it and this is the main window, all the
    #: child windows will also be destroyedRolls
    close_destroys = True

    def __init__(self, tool_instance, title, size):
        self.tool_instance = tool_instance
        mw = tool_instance.session.ui.main_window
        try:
            self.__toolkit = _Wx(self, title, mw, size)
        except ImportError:
            # browser version
            raise NotImplementedError("Browser tool API not implemented")
        self.ui_area = self.__toolkit.ui_area
        mw._new_tool_window(self)

    def cleanup(self):
        """Perform tool-specific cleanup

        Override this method to perform additional actions needed when
        the window is destroyed"""
        pass

    def destroy(self):
        """Called to destroy the window (from non-UI code)

           Destroying a tool's main window will also destroy all its
           child windows.
        """
        self.tool_instance.session.ui.main_window._tool_window_destroy(self)

    def fill_context_menu(self, menu):
        """Add items to this tool window's context menu

        Override to add items to any context menu popped up over this window"""
        pass

    def manage(self, placement, fixed_size=False):
        """Show this tool window in the interface

        Tool will be docked into main window on the side indicated by
        `placement` (which should be a value from :py:attr:`placements`
        or None).  If `placement` is None, the tool will be detached
        from the main window.
        """
        self.__toolkit.manage(placement, fixed_size)

    def _get_shown(self):
        """Whether this window is hidden or shown"""
        return self.__toolkit.shown

    def _set_shown(self, shown):
        self.tool_instance.session.ui.main_window._tool_window_request_shown(
            self, shown)

    shown = property(_get_shown, _set_shown)

    def shown_changed(self, shown):
        """Perform actions when window hidden/shown

        Override to perform any actions you want done when the window
        is hidden (\ `shown` = False) or shown (\ `shown` = True)"""
        pass

    def _destroy(self):
        self.cleanup()
        self.__toolkit.destroy()
        self.__toolkit = None

    def _mw_set_shown(self, shown):
        self.__toolkit.shown = shown
        self.shown_changed(shown)

class MainToolWindow(ToolWindow):
    """Class used to generate tool's main UI window.

    The window's :py:attr:`ui_area` attribute is the parent to all the tool's
    widgets for this window.  Call :py:meth:`manage` once the widgets
    are set up to show the tool window in the main interface.

    Parameters
    ----------
    tool_instance : a :py:class:`~chimera.core.tools.ToolInstance` instance
        The tool creating this window.
    size : 2-tuple of ints, optional
        Requested size for the tool window, width by height, in pixels.
        If not specified, uses the window system default.
    """
    def __init__(self, tool_instance, size=None):
        super().__init__(tool_instance,
            tool_instance.display_name, size)

    def create_child_window(self, title, size=None, window_class=None):
        """Make additional tool window

        Parameters
        ----------
        title : str
            Text shown in the window's title bar.
        size
            Same as for :py:class:`MainToolWindow` constructor.
        window_class : :py:class:`ChildToolWindow` subclass, optional
            Class to instantiate to create the child window.
            Only needed if you want to override methods/attributes in
            order to change behavior.
            Defaults to :py:class:`ChildToolWindow`.
        """

        if window_class is None:
            window_class = ChildToolWindow
        elif not issubclass(window_class, ChildToolWindow):
            raise ValueError(
                "Child window class must inherit from ChildToolWindow")
        return window_class(self.tool_instance, title, size)

class ChildToolWindow(ToolWindow):
    """Child (*i.e.* additional) tool window

    Only created through use of
    :py:meth:`MainToolWindow.create_child_window` method.
    """
    def __init__(self, tool_instance, title, size=None):
        super().__init__(tool_instance, title, size)

class _Wx:

    def __init__(self, tool_window, title, main_window, size):
        import wx
        self.tool_window = tool_window
        self.title = title
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

        self.ui_area = wx.Panel(mw, name=title, size=size)
        self._pane_info = None

        self.ui_area.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)

    def destroy(self):
        if not self.tool_window:
            # already destroyed
            return
        self.ui_area.Destroy()
        # free up references
        self.tool_window = None
        self.main_window = None
        self._pane_info = None

    def manage(self, placement, fixed_size=False):
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
        notebook = None
        if side == wx.TOP:
            aui_side = self.aui_side_map[side]
            side_pis = []
            for pi in mw.aui_mgr.GetAllPanes():
                if pi.dock_direction == aui_side:
                    side_pis.append(pi)
            for side_pi in side_pis:
                if side_pi.IsNotebookControl():
                    notebook = side_pi
                    break
            if not notebook and side_pis:
                notebook = side_pis[0]
            if notebook:
                from wx.lib.agw.aui import AuiPaneInfo
                pane_info = AuiPaneInfo().Top().Caption(self.title)
        if notebook:
            mw.aui_mgr.AddPane4(self.ui_area, pane_info, notebook)
        else:
            mw.aui_mgr.AddPane(self.ui_area, side, self.title)
        pane_info = mw.aui_mgr.GetPane(self.ui_area)
        if fixed_size:
            pane_info.Fixed()
        """
        mw.aui_mgr.GetPane(self.ui_area).Layer(layer+1)
        """
        if placement is None:
            pane_info.Float()

        if side == wx.BOTTOM:
            pane_info.CaptionVisible(False)

        if side in (wx.TOP, wx.BOTTOM):
            pane_info.Layer(0)
        else:
            pane_info.Layer(1)

        # hack
        if self.tool_window.tool_instance.display_name == "Log":
            pane_info.dock_proportion = 5
        else:
            pane_info.dock_proportion = 1

        if self.tool_window.close_destroys:
            pane_info.DestroyOnClose()
        mw.aui_mgr.Update()

    def on_context_menu(self, event):
        menu = wx.Menu()
        self.tool_window.fill_context_menu(menu)
        if menu.GetMenuItemCount() > 0:
            menu.Append(wx.ID_SEPARATOR)
        help_id = wx.NewId()
        # TODO: once help system more fleshed out, look for help attribute
        # or method in tool window instance and do something appropriate...
        ti = self.tool_window.tool_instance
        if ti.help is not None:
            menu.Append(help_id, "Help")
            def help_func(event, tool_instance=ti):
                tool_instance.display_help()
            self.ui_area.Bind(wx.EVT_MENU, help_func, id=help_id)
        else:
            menu.Append(help_id, "No help available")
            menu.Enable(help_id, False)
        self.ui_area.PopupMenu(menu)
        menu.Destroy()

    def _get_shown(self):
        return self.ui_area.Shown

    def _set_shown(self, shown):
        aui_mgr = self.main_window.aui_mgr
        if shown == self.ui_area.Shown:
            if shown:
                #ensure it's on top
                aui_mgr.ShowPane(self.ui_area, True)
            return
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
        if shown:
            #ensure it's on top
            aui_mgr.ShowPane(self.ui_area, True)

    shown = property(_get_shown, _set_shown)
