# vim: set expandtab ts=4 sw=4:

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

"""
gui: Main ChimeraX user interface
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

from ..logger import PlainTextLog

# remove the build tree plugin path, and add install tree plugin path
import sys
mac = (sys.platform == 'darwin')
if mac:
    # The "plugins" directory can be in one of two places on Mac:
    # - if we built Qt and PyQt from source: Contents/lib/plugins
    # - if we used a wheel built using standard Qt: C/l/python3.5/site-packages/PyQt5/Qt/plugins
    # If the former, we need to set some environment variables so
    # that Qt can find itself.  If the latter, it "just works".
    import os.path
    from ... import app_lib_dir
    plugins = os.path.join(os.path.dirname(app_lib_dir), "plugins")
    if os.path.exists(plugins):
        from PyQt5.QtCore import QCoreApplication
        qlib_paths = [p for p in QCoreApplication.libraryPaths() if not str(p).endswith('plugins')]
        qlib_paths.append(os.path.join(os.path.dirname(app_lib_dir), "plugins"))
        QCoreApplication.setLibraryPaths(qlib_paths)
        import os
        fw_path = os.environ.get("DYLD_FRAMEWORK_PATH", None)
        if fw_path:
            os.environ["DYLD_FRAMEWORK_PATH"] = app_lib_dir + ":" + fw_path
        else:
            os.environ["DYLD_FRAMEWORK_PATH"] = app_lib_dir

from PyQt5.QtWidgets import QApplication
class UI(QApplication):
    """Main ChimeraX user interface

       The only methods that tools might directly use are:

       register(/deregister)_for_keystrokes
        For the rare tool that might want to get keystrokes that are
        typed when focus is in the main graphics window

       thread_safe
        To execute a function in a thread-safe manner
       """

    required_opengl_version = (3, 3)
    required_opengl_core_profile = True

    def __init__(self, session):
        self.is_gui = True
        self.already_quit = False
        self.session = session

        from .mousemodes import MouseModes
        self.mouse_modes = MouseModes(session)

        from PyQt5.QtCore import QObject, pyqtSlot
        class CxUrlHandler(QObject):
            def __init__(self, session):
                QObject.__init__(self)
                self.session = session

            @pyqtSlot("QUrl")
            def handle_help_schema(self, qurl):
                from ..commands import run
                run(self.session, "help %s" % qurl.toString(), log=False)

            @pyqtSlot("QUrl")
            def handle_cxcmd_schema(self, qurl):
                from ..commands import run
                from PyQt5.QtCore import QUrl
                cmd = qurl.toString(QUrl.RemoveScheme)
                run(self.session, cmd, log=True)

        self.cxUrlHandler = CxUrlHandler(session)
        from PyQt5.QtGui import QDesktopServices
        QDesktopServices.setUrlHandler("cxcmd", self.cxUrlHandler.handle_cxcmd_schema)
        QDesktopServices.setUrlHandler("help", self.cxUrlHandler.handle_help_schema)

        # for whatever reason, QtWebEngineWidgets has to be imported before a
        # QtCoreApplication is created...
        import PyQt5.QtWebEngineWidgets

        import sys
        QApplication.__init__(self, [sys.argv[0]])

        # redirect Qt log messages to our logger
        from ..logger import Log
        from PyQt5.QtCore import QtDebugMsg, QtInfoMsg, QtWarningMsg, QtCriticalMsg, QtFatalMsg
        qt_to_cx_log_level_map = {
            QtDebugMsg: Log.LEVEL_INFO,
            QtInfoMsg: Log.LEVEL_INFO,
            QtWarningMsg: Log.LEVEL_WARNING,
            QtCriticalMsg: Log.LEVEL_ERROR,
            QtFatalMsg: Log.LEVEL_ERROR,
        }
        from PyQt5.QtCore import qInstallMessageHandler
        def cx_qt_msg_handler(msg_type, msg_log_context, msg_string):
            log_level = qt_to_cx_log_level_map[int(msg_type)]
            if msg_string.strip().endswith(" null"):
                # downgrade Javascript errors
                log_level = Log.LEVEL_INFO
            self.session.logger.method_map[log_level](msg_string)
        qInstallMessageHandler(cx_qt_msg_handler)

        # splash screen
        import os.path
        splash_pic_path = os.path.join(os.path.dirname(__file__),
                                       "splash.jpg")
        from PyQt5.QtWidgets import QSplashScreen
        from PyQt5.QtGui import QPixmap
        self.splash = QSplashScreen(QPixmap(splash_pic_path))
        font = self.splash.font()
        font.setPointSize(40)
        self.splash.setFont(font)
        self.splash.show()
        self.splash_info("Initializing ChimeraX")

        self._keystroke_sinks = []

    def close_splash(self):
        pass

    def build(self):
        self.main_window = mw = MainWindow(self, self.session)
        # Clicking the graphics window sets the Qt focus to None because
        # the graphics window is a QWindow rather than a widget.  Then clicking
        # on other widgets can prevent the graphics window from getting key
        # strokes even though the focus remains None.  So redirect the main
        # main window key strokes when the focus window is None.
        def forward_from_top(event, self=self):
            if self.focusWidget() is None:
                self.forward_keystroke(event)
        mw.keyPressEvent = forward_from_top
        mw.graphics_window.keyPressEvent = self.forward_keystroke
        mw.show()
        self.splash.finish(mw)

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
        if self.already_quit:
            return
        redirect_stdio_to_logger(self.session.logger)
        self.exec_()
        self.session.logger.clear()

    def forward_keystroke(self, event):
        """forward keystroke from graphics window to most recent
           caller of 'register_for_keystrokes'

           up/down arrow keystrokes are not forwarded and instead
           promote/demote the graphics window selection
        """
        from PyQt5.QtCore import Qt
        if event.key() == Qt.Key_Up:
            self.session.selection.promote()
        elif event.key() == Qt.Key_Down:
            self.session.selection.demote()
        elif self._keystroke_sinks:
            self._keystroke_sinks[-1].forwarded_keystroke(event)

    def register_for_keystrokes(self, sink):
        """'sink' is interested in receiving keystrokes from the main
           graphics window.  That object's 'forwarded_keystroke'
           method will be called with the keystroke event as the argument.
        """
        self._keystroke_sinks.append(sink)

    def shift_key_down(self):
        modifiers = self.keyboardModifiers()
        from PyQt5.QtCore import Qt
        return modifiers & Qt.ShiftModifier

    def remove_tool(self, tool_instance):
        self.main_window.remove_tool(tool_instance)

    def set_tool_shown(self, tool_instance, shown):
        self.main_window.set_tool_shown(tool_instance, shown)

    def splash_info(self, msg, step_num=None, num_steps=None):
        from PyQt5.QtCore import Qt
        self.splash.showMessage(msg, Qt.AlignLeft|Qt.AlignBottom, Qt.red)
        self.processEvents()

    def quit(self, confirm=True):
        # called by exit command
        self.already_quit = True
        self.session.logger.status("Exiting ...", blank_after=0)
        self.session.logger.clear()    # clear logging timers
        self.closeAllWindows()

    def thread_safe(self, func, *args, **kw):
        """Call function 'func' in a thread-safe manner
        """
        from PyQt5.QtCore import QEvent
        class ThreadSafeGuiFuncEvent(QEvent):
            EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
            def __init__(self, func, args, kw):
                QEvent.__init__(self, self.EVENT_TYPE)
                self.func_info = (func, args, kw)
        self.postEvent(self.main_window, ThreadSafeGuiFuncEvent(func, args, kw))

# The surface format has to be set before QtGui is initialized
from PyQt5.QtGui import QSurfaceFormat
sf = QSurfaceFormat()
sf.setVersion(*UI.required_opengl_version)
sf.setDepthBufferSize(24)
if UI.required_opengl_core_profile:
    sf.setProfile(QSurfaceFormat.CoreProfile)
sf.setRenderableType(QSurfaceFormat.OpenGL)
QSurfaceFormat.setDefaultFormat(sf)

from PyQt5.QtWidgets import QMainWindow, QStatusBar, QStackedWidget, QLabel, QDesktopWidget, \
    QToolButton
class MainWindow(QMainWindow, PlainTextLog):

    def __init__(self, ui, session):
        QMainWindow.__init__(self)
        self.setWindowTitle("ChimeraX")
        # make main window 2/3 of full screen of primary display
        dw = QDesktopWidget()
        main_screen = dw.availableGeometry(dw.primaryScreen())
        self.resize(main_screen.width()*.67, main_screen.height()*.67)
        self.setDockOptions(self.dockOptions() | self.GroupedDragging)

        from PyQt5.QtCore import QSize
        class GraphicsArea(QStackedWidget):
            def sizeHint(self):
                return QSize(800, 800)

        self._stack = GraphicsArea(self)
        from .graphics import GraphicsWindow
        self.graphics_window = g = GraphicsWindow(self._stack, ui)
        self._stack.addWidget(g.widget)
        self._stack.setCurrentWidget(g.widget)
        self.setCentralWidget(self._stack)

        from .save_dialog import MainSaveDialog, ImageSaver
        self.save_dialog = MainSaveDialog(self)
        ImageSaver(self.save_dialog).register()

        self._hide_tools = False
        self.tool_instance_to_windows = {}

        self._build_status()
        self._populate_menus(session)

        # set icon for About dialog
        from chimerax import app_dirs as ad, app_data_dir
        import os.path
        icon_path = os.path.join(app_data_dir, "%s-icon512.png" % ad.appname)
        if os.path.exists(icon_path):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))
        
        session.logger.add_log(self)

        self.show()

    def adjust_size(self, delta_width, delta_height):
        cs = self.size()
        cww, cwh = cs.width(), cs.height()
        ww = cww + delta_width
        wh = cwh + delta_height
        self.resize(ww, wh)

    def close_request(self, tool_window, close_event):
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        close_destroys = tool_window.close_destroys
        if is_main_window and close_destroys:
            close_event.accept()
            tool_instance.delete()
            return
        if close_destroys:
            close_event.accept()
            tool_window._destroy()
            all_windows.remove(tool_window)
        else:
            close_event.ignore()
            tool_window.shown = False

        if is_main_window:
            # close hides, since close destroys is handled above
            for window in all_windows:
                window._prev_shown = window.shown
                window.shown = False

    def customEvent(self, event):
        # handle requests to execute GUI functions from threads
        func, args, kw = event.func_info
        func(*args, **kw)

    def file_open_cb(self, session):
        from PyQt5.QtWidgets import QFileDialog
        from .open_save import open_file_filter
        paths_and_types = QFileDialog.getOpenFileNames(filter=open_file_filter(all=True))
        paths, types = paths_and_types
        if not paths:
            return

        models = session.models.open(paths)
        if models and len(paths) == 1:
            # Remember in file history
            from ..filehistory import remember_file
            remember_file(session, paths[0], format=None, models=models)

    def file_save_cb(self, session):
        self.save_dialog.display(self, session)

    def file_quit_cb(self, session):
        session.ui.quit()

    def hide_other_tools(self, tool_instance):
        for ti, tool_windows in self.tool_instance_to_windows.items():
            if ti == tool_instance:
                continue
            for tw in tool_windows:
                if tw.title == "Command Line Interface":
                    # leave the command line as is
                    continue
                if tw.shown:
                    tw._mw_set_shown(False)
        
    def _get_hide_tools(self):
        return self._hide_tools

    def _set_hide_tools(self, ht):
        if ht == self._hide_tools:
            return

        # need to set _hide_tools attr first, since it will be checked in 
        # subsequent calls
        self._hide_tools = ht
        if ht == True:
            icon = self._contract_icon
            self._hide_tools_shown_states = states = {}
            for tool_windows in self.tool_instance_to_windows.values():
                for tw in tool_windows:
                    if tw.title == "Command Line Interface":
                        # leave the command line as is
                        continue
                    state = tw.shown
                    states[tw] = state
                    if state:
                        tw._mw_set_shown(False)
        else:
            icon = self._expand_icon
            for tw, state in self._hide_tools_shown_states.items():
                if state:
                    tw.shown = True
            self._hide_tools_shown_states.clear()

        self._global_hide_button.setIcon(icon)


    hide_tools = property(_get_hide_tools, _set_hide_tools)

    def log(self, *args, **kw):
        return False

    def remove_tool(self, tool_instance):
        tool_windows = self.tool_instance_to_windows.get(tool_instance, None)
        if tool_windows:
            for tw in tool_windows:
                tw._mw_set_shown(False)
                tw._destroy()
            del self.tool_instance_to_windows[tool_instance]

    def set_tool_shown(self, tool_instance, shown):
        tool_windows = self.tool_instance_to_windows.get(tool_instance, None)
        if tool_windows:
            tool_windows[0].shown = shown

    def status(self, msg, color, secondary):
        # prevent status message causing/allowing a redraw
        self.graphics_window.session.update_loop.block_redraw()
        self.statusBar().clearMessage()
        if secondary:
            label = self._secondary_status_label
        else:
            label = self._primary_status_label
        label.setText("<font color='" + color + "'>" + msg + "</font>")
        label.show()
        # Make status line update during long computations where event loop is not running.
        # Code that asks to display a status message does not expect arbitrary callbacks
        # to run.  This could cause timers and callbacks to run that could lead to errors.
        # User events are not processed since that could allow the user to delete data.
        # Ticket #407.
        from PyQt5.QtCore import QEventLoop
        self.graphics_window.session.ui.processEvents(QEventLoop.ExcludeUserInputEvents)
        self.graphics_window.session.update_loop.unblock_redraw()

    def _about(self, arg):
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        import os.path
        from .. import buildinfo
        from chimerax import app_dirs as ad
        fn = os.path.join(os.path.dirname(__file__), "about.html")
        with open(fn) as f:
            content = f.read()
        content = content.replace("VERSION", ad.version)
        content = content.replace("DATE", buildinfo.date.split()[0])
        self._about_dialog = QWebEngineView()
        self._about_dialog.setHtml(content)
        self._about_dialog.show()

    def _build_status(self):
        sb = QStatusBar(self)
        from PyQt5.QtWidgets import QSizePolicy
        sb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._primary_status_label = QLabel(sb)
        self._secondary_status_label = QLabel(sb)
        sb.addWidget(self._primary_status_label)
        sb.addPermanentWidget(self._secondary_status_label)
        self._global_hide_button = ghb = QToolButton(sb)
        from PyQt5.QtGui import QIcon
        import os.path
        cur_dir = os.path.dirname(__file__)
        self._expand_icon = QIcon(os.path.join(cur_dir, "expand1.png"))
        self._contract_icon = QIcon(os.path.join(cur_dir, "contract1.png"))
        ghb.setIcon(self._expand_icon)
        ghb.setCheckable(True)
        from PyQt5.QtWidgets import QAction
        but_action = QAction(ghb)
        but_action.setCheckable(True)
        but_action.toggled.connect(lambda checked: setattr(self, 'hide_tools', checked))
        but_action.setIcon(self._expand_icon)
        ghb.setDefaultAction(but_action)
        sb.addPermanentWidget(ghb)
        sb.showMessage("Welcome to Chimera X")
        self.setStatusBar(sb)

    def _new_tool_window(self, tw):
        if self.hide_tools:
            self._hide_tools_shown_states[tw] = True
            tw._mw_set_shown(False)
            tw.tool_instance.session.logger.status("Tool display currently suppressed",
                color="red", blank_after=7)
        self.tool_instance_to_windows.setdefault(tw.tool_instance,[]).append(tw)

    def _populate_menus(self, session):
        from PyQt5.QtWidgets import QAction

        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open input file")
        open_action.triggered.connect(lambda arg, s=self, sess=session: s.file_open_cb(sess))
        file_menu.addAction(open_action)
        save_action = QAction("&Save...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save output file")
        save_action.triggered.connect(lambda arg, s=self, sess=session: s.file_save_cb(sess))
        file_menu.addAction(save_action)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.setStatusTip("Quit ChimeraX")
        quit_action.triggered.connect(lambda arg, s=self, sess=session: s.file_quit_cb(sess))
        file_menu.addAction(quit_action)

        tools_menu = mb.addMenu("&Tools")
        categories = {}
        for bi in session.toolshed.bundle_info():
            for cat in bi.menu_categories:
                categories.setdefault(cat, {})[bi.display_name] = bi
        cat_keys = sorted(categories.keys())
        try:
            cat_keys.remove('Hidden')
        except ValueError:
            pass
        one_menu = len(cat_keys) == 1
        for cat in cat_keys:
            if one_menu:
                cat_menu = tools_menu
            else:
                cat_menu = tools_menu.addMenu(cat)
            cat_info = categories[cat]
            for tool_name in sorted(cat_info.keys()):
                bi = cat_info[tool_name]
                tool_action = QAction(tool_name, self)
                tool_action.setStatusTip(bi.synopsis)
                tool_action.triggered.connect(lambda arg, ses=session, bi=bi: bi.start(ses))
                cat_menu.addAction(tool_action)

        help_menu = mb.addMenu("&Help")
        for entry, topic in (('User Guide', 'user'),
                           ('Quick Start Guide', 'quickstart'),
                           ('Programming Manual', 'devel'),
                           ('Documentation Index', 'index.html'),
                           ):
            help_action = QAction(entry, self)
            help_action.setStatusTip("Show " + entry)
            def cb(arg, ses=session, t=topic):
                from chimerax.core.commands import run
                run(ses, 'help new_viewer help:%s' % t)
            help_action.triggered.connect(cb)
            help_menu.addAction(help_action)
        from chimerax import app_dirs as ad
        about_action = QAction("About %s %s" % (ad.appauthor, ad.appname), self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

    def add_custom_menu_entry(self, menu_name, entry_name, callback):
        '''
        Add a custom top level menu entry.  Currently you can not add to
        the standard ChimeraX menus but can create new ones.
        Callback function takes no arguments.
        '''
        mb = self.menuBar()
        from PyQt5.QtWidgets import QMenu, QAction
        menu = mb.findChild(QMenu, menu_name)
        add = (menu is None)
        if add:
            menu = QMenu(menu_name, mb)
            menu.setObjectName(menu_name)	# Need for findChild() above to work.
        
        action = QAction(entry_name, self)
        action.triggered.connect(lambda arg, cb = callback: callback())
        menu.addAction(action)
        if add:
            # Add menu after adding entry otherwise it is not shown on Mac.
            mb.addMenu(menu)

    def _tool_window_destroy(self, tool_window):
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        if is_main_window:
            tool_instance.delete()
            return
        tool_window._destroy()
        all_windows.remove(tool_window)

    def _tool_window_request_shown(self, tool_window, shown):
        if self.hide_tools:
            def set_shown(win, show):
                self._hide_tools_shown_states[win] = show
        else:
            set_shown = lambda win, show: win._mw_set_shown(show)
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        set_shown(tool_window, shown)
        if is_main_window:
            for window in all_windows[1:]:
                if shown:
                    # if child window has a '_prev_shown' attr, then it was
                    # around when main window was closed/hidden, possibly
                    # show it and forget the _prev_shown attrs
                    if hasattr(window, '_prev_shown'):
                        if window._prev_shown:
                            set_shown(window, True)
                        delattr(window, '_prev_shown')
                else:
                    set_shown(window, False)

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

    #: Where the window is placed in the main interface;
    #: 'side' is either left or right, depending on user preference
    placements = ["side", "right", "left", "top", "bottom"]

    #: Whether closing this window destroys it or hides it.
    #: If it destroys it and this is the main window, all the
    #: child windows will also be destroyed
    close_destroys = True

    def __init__(self, tool_instance, title):
        self.tool_instance = tool_instance
        mw = tool_instance.session.ui.main_window
        self.__toolkit = _Qt(self, title, mw)
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

    def fill_context_menu(self, menu, x, y):
        """Add items to this tool window's context menu,
           whose downclick occurred at position (x,y)

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

    def _get_title(self):
        if self.__toolkit is None:
            return ""
        return self.__toolkit.title

    def _set_title(self, title):
        if self.__toolkit is None:
            return
        self.__toolkit.set_title(title)
    set_title = _set_title

    title = property(_get_title, _set_title)

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
    tool_instance : a :py:class:`~chimerax.core.tools.ToolInstance` instance
        The tool creating this window.
    """
    def __init__(self, tool_instance):
        super().__init__(tool_instance, tool_instance.display_name)

    def create_child_window(self, title, window_class=None):
        """Make additional tool window

        Parameters
        ----------
        title : str
            Text shown in the window's title bar.
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
        return window_class(self.tool_instance, title)

class ChildToolWindow(ToolWindow):
    """Child (*i.e.* additional) tool window

    Only created through use of
    :py:meth:`MainToolWindow.create_child_window` method.
    """
    def __init__(self, tool_instance, title):
        super().__init__(tool_instance, title)

class _Qt:
    def __init__(self, tool_window, title, main_window):
        self.tool_window = tool_window
        self.title = title
        self.main_window = mw = main_window
        from PyQt5.QtCore import Qt
        # for now, 'side' equals 'right'
        qt_sides = [Qt.RightDockWidgetArea, Qt.RightDockWidgetArea, Qt.LeftDockWidgetArea,
            Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea]
        self.placement_map = dict(zip(self.tool_window.placements, qt_sides))
        if not mw:
            raise RuntimeError("No main window or main window dead")

        from PyQt5.QtWidgets import QDockWidget, QWidget
        self.dock_widget = dw = QDockWidget(title, mw)
        dw.closeEvent = lambda e, tw=tool_window, mw=mw: mw.close_request(tw, e)
        dw.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self.ui_area = QWidget(dw)
        self.ui_area.contextMenuEvent = lambda e, self=self: self.show_context_menu(e.globalPos())
        self.dock_widget.setWidget(self.ui_area)

    def destroy(self):
        if not self.tool_window:
            # already destroyed
            return
        # free up references
        self.tool_window = None
        self.main_window = None
        self.ui_area.destroy()
        self.dock_widget.destroy()

    def manage(self, placement, fixed_size=False):
        from PyQt5.QtCore import Qt
        placements = self.tool_window.placements
        if placement is None:
            side = Qt.RightDockWidgetArea
        else:
            if placement not in placements:
                raise ValueError("placement value must be one of: {}, or None"
                    .format(", ".join(placements)))
            else:
                side = self.placement_map[placement]

        mw = self.main_window
        mw.addDockWidget(side, self.dock_widget)
        if placement is None:
            self.dock_widget.setFloating(True)

        #QT disable: create a 'hide_title_bar' option
        if side == Qt.BottomDockWidgetArea:
            from PyQt5.QtWidgets import QWidget
            self.dock_widget.setTitleBarWidget(QWidget())

        if self.tool_window.close_destroys:
            self.dock_widget.setAttribute(Qt.WA_DeleteOnClose)

    def show_context_menu(self, event):
        from PyQt5.QtWidgets import QMenu, QAction
        menu = QMenu(self.ui_area)

        self.tool_window.fill_context_menu(menu, event.x(), event.y())
        if not menu.isEmpty():
            menu.addSeparator()
        ti = self.tool_window.tool_instance
        hide_tool_action = QAction("Hide this tool", self.ui_area)
        hide_tool_action.triggered.connect(lambda arg, ti=ti: ti.display(False))
        menu.addAction(hide_tool_action)
        hide_other_tools_action = QAction("Hide other tools", self.ui_area)
        hide_other_tools_action.triggered.connect(
            lambda arg, ti=ti, ho=self.main_window.hide_other_tools: ho(ti))
        menu.addAction(hide_other_tools_action)
        if ti.help is not None:
            help_action = QAction("Help", self.ui_area)
            help_action.setStatusTip("Show tool help")
            help_action.triggered.connect(lambda arg, ti=ti: ti.display_help())
            menu.addAction(help_action)
        else:
            no_help_action = QAction("No help available", self.ui_area)
            no_help_action.setEnabled(False)
            menu.addAction(no_help_action)
        menu.exec(event)

    def _get_shown(self):
        return not self.dock_widget.isHidden()

    def _set_shown(self, shown):
        # isHidden() is not to be trusted before the main window is shown
        # since it will return True even though the window _will_ be shown
        # once the main window shows, so comment out the optimization
        # until I can figure something out (showEvent and QTimer(0) both
        # seem to fire too early...)
        """
        if shown != self.dock_widget.isHidden():
            if shown:
                #ensure it's on top
                self.dock_widget.raise_()
            return
        """
        if shown:
            self.dock_widget.show()
            #ensure it's on top
            self.dock_widget.raise_()
        else:
            self.dock_widget.hide()

    shown = property(_get_shown, _set_shown)

    def set_title(self, title):
        self.dock_widget.setWindowTitle(title)

def redirect_stdio_to_logger(logger):
    # Redirect stderr to log
    class LogStdout:

        # Qt's error logging looks at the encoding of sys.stderr...
        encoding = 'utf-8'

        def __init__(self, logger):
            self.logger = logger
            self.closed = False

        def write(self, s):
            self.logger.session.ui.thread_safe(self.logger.info,
                                               s, add_newline = False)
            # self.logger.info(s, add_newline = False)

        def flush(self):
            return

        def isatty(self):
            return False
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

