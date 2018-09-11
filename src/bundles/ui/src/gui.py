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
ui.gui: Main ChimeraX graphical user interface
==============================================

The principal class that tool writers will use from this module is
:py:class:`MainToolWindow`, which is either instantiated directly, or
subclassed and instantiated to create the tool's main window.
Additional windows are created by calling that instance's
:py:meth:`MainToolWindow.create_child_window` method.

Rarely, methods are used from the :py:class:`UI` class to get
keystrokes typed to the main graphics window, or to execute code
in a thread-safe manner.  The UI instance is accessed as session.ui.
"""

from chimerax.core.logger import PlainTextLog

def initialize_qt():
    initialize_qt_plugins_location()
    initialize_qt_high_dpi_display_support()

def initialize_qt_plugins_location():
    # remove the build tree plugin path, and add install tree plugin path
    import sys
    mac = (sys.platform == 'darwin')
    if mac:
        # The "plugins" directory can be in one of two places on Mac:
        # - if we built Qt and PyQt from source: Contents/lib/plugins
        # - if we used a wheel built using standard Qt: C/l/python3.5/site-packages/PyQt5/Qt/plugins
        # If the former, we need to set some environment variables so
        # that Qt can find itself.  If the latter, it "just works",
        # though if there is a comma in the app name, the magic gets
        # screwed up, so explicitly set the path in that case too
        import os.path
        from chimerax import app_lib_dir
        plugins = os.path.join(os.path.dirname(app_lib_dir), "plugins")
        if not os.path.exists(plugins) and "," in app_lib_dir:
            # The comma character screws up the magic Qt plugin-finding code;
            # supply an explicit path in this case
            # To find site-packages look above __file__...
            dn = os.path.dirname
            plugins = os.path.join(dn(dn(dn(dn(__file__)))), "PyQt5/Qt/plugins")
        if os.path.exists(plugins):
            from PyQt5.QtCore import QCoreApplication
            qlib_paths = [p for p in QCoreApplication.libraryPaths() if not str(p).endswith('plugins')]
            qlib_paths.append(plugins)
            QCoreApplication.setLibraryPaths(qlib_paths)
            import os
            fw_path = os.environ.get("DYLD_FRAMEWORK_PATH", None)
            if fw_path:
                os.environ["DYLD_FRAMEWORK_PATH"] = app_lib_dir + ":" + fw_path
            else:
                os.environ["DYLD_FRAMEWORK_PATH"] = app_lib_dir

def initialize_qt_high_dpi_display_support():
    import sys
    # Fix text and button sizes on high DPI displays in Windows 10
    win = (sys.platform == 'win32')
    if win:
        from PyQt5.QtCore import QCoreApplication, Qt
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

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

    def __init__(self, session):
        self.is_gui = True
        self.has_graphics = True
        self.already_quit = False
        self.session = session

        from .settings import UI_Settings
        self.settings = UI_Settings(session, "ui")

        from .mousemodes import MouseModes
        self.mouse_modes = MouseModes(session)

        # for whatever reason, QtWebEngineWidgets has to be imported before a
        # QtCoreApplication is created...
        import PyQt5.QtWebEngineWidgets

        import sys
        QApplication.__init__(self, [sys.argv[0]])

        self.redirect_qt_messages()

        self._keystroke_sinks = []
        self._files_to_open = []

        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('ready')

    def redirect_qt_messages(self):
        
        # redirect Qt log messages to our logger
        from chimerax.core.logger import Log
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

    def show_splash(self):
        # splash screen
        import os.path
        splash_pic_path = os.path.join(os.path.dirname(__file__), "splash.jpg")
        from PyQt5.QtWidgets import QSplashScreen
        from PyQt5.QtGui import QPixmap
        self.splash = QSplashScreen(QPixmap(splash_pic_path))
        font = self.splash.font()
        font.setPointSize(40)
        self.splash.setFont(font)
        self.splash.show()
        self.splash_info("Initializing ChimeraX")

    def close_splash(self):
        pass

    def window_image(self):
        screen = self.primaryScreen()
        w = self.main_window
        w_id = w.winId()
#        g = w.geometry()  # Works on Mac, wrong origin on Windows
        g = w.rect()
        pixmap = screen.grabWindow(w_id, g.x(), g.y(), g.width(), g.height())
        im = pixmap.toImage()
        return im

    def build(self):
        self.main_window = mw = MainWindow(self, self.session)
        # key event forwarding from the main window itself seems to have
        # no benefit, and occasionally causes double command execution
        # for slow commands, so only forward from graphics window
        mw.graphics_window.keyPressEvent = self.forward_keystroke
        mw.rapid_access.keyPressEvent = self.forward_keystroke
        mw.show()
        mw.rapid_access_shown = True
        self.splash.finish(mw)
        # Register for tool installation/deinstallation so that
        # we can update the Tools menu
        from chimerax.core.toolshed import (TOOLSHED_BUNDLE_INSTALLED,
                                TOOLSHED_BUNDLE_UNINSTALLED,
                                TOOLSHED_BUNDLE_INFO_RELOADED)
        def handler(*args, mw=self.main_window, ses=self.session, **kw):
            mw.update_tools_menu(ses)
        triggers = self.session.toolshed.triggers
        triggers.add_handler(TOOLSHED_BUNDLE_INSTALLED, handler)
        triggers.add_handler(TOOLSHED_BUNDLE_UNINSTALLED, handler)
        triggers.add_handler(TOOLSHED_BUNDLE_INFO_RELOADED, handler)
        if self.autostart_tools:
            self.session.tools.start_tools(self.settings.autostart)

        self.triggers.activate_trigger('ready', None)

    def event(self, event):
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.FileOpen:
            if not hasattr(self, 'toolshed'):
                # Drop event might have started ChimeraX and it is not yet ready to open a file.
                # So remember file and startup script will open it when ready.
                self._files_to_open.append(event.file())
            else:
                _open_dropped_file(self.session, event.file())
            return True
        return QApplication.event(self, event)

    def open_pending_files(self, ignore_files = ()):
        # Note about ignore_files:  macOS 10.12 generates QFileOpenEvent for arguments specified
        # on the command-line, but are code also opens those files, so ignore files we already processed.
        for path in self._files_to_open:
            if path not in ignore_files:
                try:
                    _open_dropped_file(self.session, path)
                except Exception as e:
                    self.session.logger.warning('Failed opening file %s:\n%s' % (path, str(e)))
        self._files_to_open.clear()
                    
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
            from chimerax.core.commands import run
            run(self.session, 'select up')
        elif event.key() == Qt.Key_Down:
            from chimerax.core.commands import run
            run(self.session, 'select down')
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
        ses = self.session
        log = ses.logger
        log.status("Exiting ...", blank_after=0)
        log.clear()    # clear logging timers
        ses.triggers.activate_trigger('app quit', None)
        self.closeAllWindows()
        QApplication.quit()

    def thread_safe(self, func, *args, **kw):
        """Call function 'func' in a thread-safe manner
        """
        import threading
        if threading.main_thread() == threading.current_thread():
            func(*args, **kw)
            return
        from PyQt5.QtCore import QEvent
        class ThreadSafeGuiFuncEvent(QEvent):
            EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
            def __init__(self, func, args, kw):
                QEvent.__init__(self, self.EVENT_TYPE)
                self.func_info = (func, args, kw)
        self.postEvent(self.main_window, ThreadSafeGuiFuncEvent(func, args, kw))

    def timer(self, millisec, callback, *args, **kw):
        from PyQt5.QtCore import QTimer
        t = QTimer()
        def cb(callback=callback, args=args, kw=kw):
            callback(*args, **kw)
        t.timeout.connect(cb)
        t.setSingleShot(True)
        t.start(int(millisec))
        return t

    def cancel_timer(self, timer):
        timer.stop()

    def update_undo(self, undo_manager):
        self.main_window.update_undo(undo_manager)
        
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QLabel, QDesktopWidget, \
    QToolButton, QWidget
class MainWindow(QMainWindow, PlainTextLog):

    def __init__(self, ui, session):
        self.session = session
        QMainWindow.__init__(self)
        self.setWindowTitle("ChimeraX")
        # make main window 2/3 of full screen of primary display
        dw = QDesktopWidget()
        main_screen = dw.availableGeometry(dw.primaryScreen())
        self.resize(main_screen.width()*.67, main_screen.height()*.67)

        from PyQt5.QtCore import QSize
        class GraphicsArea(QStackedWidget):
            def sizeHint(self):
                return QSize(800, 800)

        self._stack = GraphicsArea(self)
        from .graphics import GraphicsWindow
        stereo = getattr(ui, 'stereo', False)
        if stereo:
            from chimerax.core.graphics import StereoCamera
            session.main_view.camera = StereoCamera()
        self.graphics_window = g = GraphicsWindow(self._stack, ui, stereo)
        self._stack.addWidget(g.widget)
        self.rapid_access = QWidget(self._stack)
        ra_bg_color = "#B8B8B8"
        font_size = 96
        new_user_text = [
            "<html>",
            "<body>",
            "<style>",
            "body {",
            "    background-color: %s;" % ra_bg_color,
            "}",
            ".banner-text {",
            "    font-size: %dpx;" % font_size,
            "    color: #3C6B19;",
            "    position: absolute;",
            "    top: 50%;",
            "    left: 50%;",
            "    transform: translate(-50%,-150%);",
            "}"
            ".help-link {",
            "    position: absolute;"
            "    top: 60%;",
            "    left: 50%;",
            "    transform: translate(-50%,-50%);",
            "}",
            "</style>",
            '<p class="banner-text">ChimeraX</p>',
            '<p class="help-link"><a href="cxcmd:help help:quickstart">Get started</a><p>',
            "</body>",
            "</html>"
        ]
        from .file_history import FileHistory
        fh = FileHistory(session, self.rapid_access, bg_color=ra_bg_color, thumbnail_size=(128,128),
            filename_size=15, no_hist_text="\n".join(new_user_text))
        self._stack.addWidget(self.rapid_access)
        self._stack.setCurrentWidget(g.widget)
        self.setCentralWidget(self._stack)
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        session.triggers.add_handler(ADD_MODELS, self._check_rapid_access)
        session.triggers.add_handler(REMOVE_MODELS, self._check_rapid_access)

        from .save_dialog import MainSaveDialog, ImageSaver
        self.save_dialog = MainSaveDialog(self)
        ImageSaver(self.save_dialog).register()

        self._hide_tools = False
        self.tool_instance_to_windows = {}
        self._fill_tb_context_menu_cbs = {}

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

        # Allow drag and drop of files onto app window.
        self.setAcceptDrops(True)

        self.show()

    def enable_stereo(self, stereo = True):
        '''
        Switching to a sequential stereo OpenGL context seems to require
        replacing the graphics window with a stereo compatible window on 
        Windows 10 with Qt 5.9.
        '''
        gw = self.graphics_window
        oc = gw.opengl_context
        if stereo == oc.stereo:
            return True	# Already using requested mode

        from .graphics import GraphicsWindow
        try:
            g = GraphicsWindow(self._stack, self.session.ui, stereo, oc)
        except:
            # Failed to create OpenGL context
            return False

        # Only destroy old graphics window after new one is made so clean-up
        # of old OpenGL context can be done.
        gw.destroy()

        self.graphics_window = g
        self._stack.addWidget(g.widget)
        self._stack.setCurrentWidget(g.widget)

        return True
    
    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            event.acceptProposedAction()
        
    def dropEvent(self, event):
        md = event.mimeData()
        paths = [url.toLocalFile() for url in md.urls()]
        for p in paths:
            _open_dropped_file(self.session, p)
        
    def add_tool_bar(self, tool, *tb_args, fill_context_menu_cb=None, **tb_kw):
        # need to track toolbars for checkbuttons in Tools->Toolbar
        retval = QMainWindow.addToolBar(self, *tb_args, **tb_kw)
        from PyQt5.QtWidgets import QToolBar
        for arg in tb_args:
            if isinstance(arg, QToolBar):
                tb = arg
                break
        else:
            tb = retval
        tb.visibilityChanged.connect(lambda vis, tb=tb: self._set_tool_checkbuttons(tb, vis))
        tb.contextMenuEvent = lambda e, self=self, tb=tb: self.show_tb_context_menu(tb, e)
        self._fill_tb_context_menu_cbs[tb] = (tool, fill_context_menu_cb)
        return tb

    def adjust_size(self, delta_width, delta_height):
        cs = self.size()
        cww, cwh = cs.width(), cs.height()
        ww = cww + delta_width
        wh = cwh + delta_height
        self.resize(ww, wh)

    def window_maximized(self):
        from PyQt5.QtCore import Qt
        return bool(self.windowState() & (Qt.WindowMaximized | Qt.WindowFullScreen))
    
    def closeEvent(self, event):
        # the MainWindow close button has been clicked
        event.accept()
        self.session.ui.quit()

    def close_request(self, tool_window, close_event):
        # closing a tool window has been requested
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

        def _qt_safe(session=session, paths=paths):
            from chimerax.core.commands import run, quote_if_necessary
            if len(paths) == 1:
                run(session, "open " + quote_if_necessary(paths[0]))
            else:
                # Open multiple files as a single batch.
                # TODO: Make open command handle this including saving in file history.
                session.models.open(paths)

        # Opening the model directly adversely affects Qt interfaces that show
        # as a result.  In particular, Multalign Viewer no longer gets hover
        # events correctly, nor tool tips.
        #
        # Using session.ui.thread_safe() doesn't help either(!)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, _qt_safe)

    def file_save_cb(self, session):
        self.save_dialog.display(self, session)

    def file_close_cb(self, session):
        from chimerax.core.commands import run
        run(session, 'close session')

    def file_quit_cb(self, session):
        session.ui.quit()

    def edit_undo_cb(self, session):
        session.undo.undo()

    def edit_redo_cb(self, session):
        session.undo.redo()

    def update_undo(self, undo_manager):
        self._set_undo(self.undo_action, "Undo", undo_manager.top_undo_name())
        self._set_undo(self.redo_action, "Redo", undo_manager.top_redo_name())

    def _set_undo(self, action, label, name):
        if name is None:
            action.setText(label)
            action.setEnabled(False)
        else:
            action.setText("%s %s" % (label, name))
            action.setEnabled(True)

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
            settings_dw = self.settings_ui_widget
            self._pref_dialog_state = not settings_dw.isHidden() and not settings_dw.isFloating()
            if self._pref_dialog_state:
                settings_dw.hide()
            for tool_windows in self.tool_instance_to_windows.values():
                for tw in tool_windows:
                    if tw.title == "Command Line Interface":
                        # leave the command line as is
                        continue
                    if tw.floating:
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
            if self._pref_dialog_state:
                self.settings_ui_widget.show()

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

    def _get_rapid_access_shown(self):
        return self._stack.currentWidget() == self.rapid_access

    def _set_rapid_access_shown(self, show):
        if show == (self._stack.currentWidget() == self.rapid_access):
            return

        ses = self.session
        from PyQt5.QtCore import QEventLoop
        if show:
            icon = self._ra_shown_icon
            self._stack.setCurrentWidget(self.rapid_access)
        else:
            icon = self._ra_hidden_icon
            self._stack.setCurrentWidget(self.graphics_window.widget)
        ses.update_loop.block_redraw()
        ses.ui.processEvents(QEventLoop.ExcludeUserInputEvents)
        ses.update_loop.unblock_redraw()

        but = self._rapid_access_button
        but.setChecked(show)
        but.defaultAction().setChecked(show)
        but.setIcon(icon)

    rapid_access_shown = property(_get_rapid_access_shown, _set_rapid_access_shown)

    def _check_chains_update_status(self, trig_name, changes):
        if not self.chains_menu_needs_update:
            self.chains_menu_needs_update = \
                changes.created_chains() or changes.num_deleted_chains() > 0

    def _check_rapid_access(self, *args):
        self.rapid_access_shown = len(self.session.models) == 0

    def show_tb_context_menu(self, tb, event):
        tool, fill_cb = self._fill_tb_context_menu_cbs[tb]
        show_context_menu(event, tool, fill_cb, True)

    def status(self, msg, color, secondary):
        self._status_bar.status(msg, color, secondary)

    def _about(self, arg):
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        import os.path
        from chimerax.core import buildinfo
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
        from .statusbar import _StatusBar
        self._status_bar = sbar = _StatusBar(self.session)
        sbar.pad_vert = 0.2	# Make text in main status bar a little smaller to match command-line
        sb = sbar.widget
        self._global_hide_button = ghb = QToolButton(sb)
        self._rapid_access_button = rab = QToolButton(sb)
        from .icons import get_qt_icon
        self._expand_icon = get_qt_icon("expand1")
        self._contract_icon = get_qt_icon("contract1")
        self._ra_shown_icon = get_qt_icon("lightning_day")
        self._ra_hidden_icon = get_qt_icon("lightning_night")
        ghb.setIcon(self._expand_icon)
        rab.setIcon(self._ra_shown_icon)
        ghb.setCheckable(True)
        rab.setCheckable(True)
        rab.setChecked(True)
        from PyQt5.QtWidgets import QAction
        ghb_action = QAction(ghb)
        rab_action = QAction(rab)
        ghb_action.setCheckable(True)
        rab_action.setCheckable(True)
        rab_action.setChecked(True)
        ghb_action.toggled.connect(lambda checked: setattr(self, 'hide_tools', checked))
        rab_action.toggled.connect(lambda checked: setattr(self, 'rapid_access_shown', checked))
        ghb_action.setIcon(self._expand_icon)
        rab_action.setIcon(self._ra_shown_icon)
        ghb.setDefaultAction(ghb_action)
        rab.setDefaultAction(rab_action)
        sb.addPermanentWidget(ghb)
        sb.addPermanentWidget(rab)
        sb.showMessage("Welcome to ChimeraX")
        self.setStatusBar(sb)

    def _dockability_change(self, tool_name, dockable):
        """Call back from 'ui dockable' command"""
        for ti, tool_windows in self.tool_instance_to_windows.items():
            if ti.tool_name == tool_name:
                for win in tool_windows:
                    win._mw_set_dockable(dockable)

    def _make_settings_ui(self, session):
        from .core_settings_ui import CoreSettingsPanel
        from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout
        self.settings_ui_widget = dw = QDockWidget("ChimeraX Settings", self)
        dw.closeEvent = lambda e, dw=dw: dw.hide()
        container = QWidget()
        CoreSettingsPanel(session, container)
        dw.setWidget(container)
        from PyQt5.QtCore import Qt
        self.addDockWidget(Qt.RightDockWidgetArea, dw)
        dw.hide()
        dw.setFloating(True)

    def _new_tool_window(self, tw):
        if self.hide_tools:
            self._hide_tools_shown_states[tw] = True
            tw._mw_set_shown(False)
            tw.tool_instance.session.logger.status("Tool display currently suppressed",
                color="red", blank_after=7)
        self.tool_instance_to_windows.setdefault(tw.tool_instance,[]).append(tw)

    def _populate_menus(self, session):
        from PyQt5.QtWidgets import QAction
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtCore import Qt

        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        file_menu.setObjectName("File")
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setToolTip("Open input file")
        open_action.triggered.connect(lambda arg, s=self, sess=session: s.file_open_cb(sess))
        file_menu.addAction(open_action)
        save_action = QAction("&Save...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Save output file")
        save_action.triggered.connect(lambda arg, s=self, sess=session: s.file_save_cb(sess))
        file_menu.addAction(save_action)
        save_action = QAction("&Close Session", self)
        save_action.setToolTip("Close session")
        save_action.triggered.connect(lambda arg, s=self, sess=session: s.file_close_cb(sess))
        file_menu.addAction(save_action)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.setToolTip("Quit ChimeraX")
        quit_action.triggered.connect(lambda arg, s=self, sess=session: s.file_quit_cb(sess))
        file_menu.addAction(quit_action)
        file_menu.setToolTipsVisible(True)

        edit_menu = mb.addMenu("&Edit")
        edit_menu.setObjectName("Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setEnabled(False)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(lambda arg, s=self, sess=session: s.edit_undo_cb(sess))
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setEnabled(False)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(lambda arg, s=self, sess=session: s.edit_redo_cb(sess))
        edit_menu.addAction(self.redo_action)

        select_menu = mb.addMenu("&Select")
        select_menu.setObjectName("Select")
        self._populate_select_menu(select_menu)

        self.tools_menu = mb.addMenu("&Tools")
        self.tools_menu.setToolTipsVisible(True)
        self.update_tools_menu(session)

        self.favorites_menu = mb.addMenu("Fa&vorites")
        self.favorites_menu.setToolTipsVisible(True)
        self._make_settings_ui(session)
        self.update_favorites_menu(session)

        help_menu = mb.addMenu("&Help")
        help_menu.setObjectName("Help")
        help_menu.setToolTipsVisible(True)
        for entry, topic, tooltip in (
                ('User Guide', 'user', 'Tutorials and user documentation'),
                ('Quick Start Guide', 'quickstart', 'Interactive ChimeraX basics'),
                ('Programming Manual', 'devel', 'How to develop ChimeraX tools'),
                ('Documentation Index', 'index.html', 'Access all documentarion'),
                ('Contact Us', 'contact.html', 'Report problems/issues; ask questions')):
            help_action = QAction(entry, self)
            help_action.setToolTip(tooltip)
            def cb(arg, ses=session, t=topic):
                from chimerax.core.commands import run
                run(ses, 'help help:%s' % t)
            help_action.triggered.connect(cb)
            help_menu.addAction(help_action)
        from chimerax import app_dirs as ad
        about_action = QAction("About %s %s" % (ad.appauthor, ad.appname), self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)
        help_menu.setObjectName("Help") # so custom-menu insertion can find it

    def _populate_select_menu(self, select_menu):
        from PyQt5.QtWidgets import QAction
        self.select_chains_menu = select_menu.addMenu("&Chains")
        self.select_chains_menu.aboutToShow.connect(self.update_select_chains_menu)
        self.select_chains_menu.setToolTipsVisible(True)
        self.chains_menu_needs_update = False
        from chimerax import atomic
        atom_triggers = atomic.get_triggers(self.session)
        atom_triggers.add_handler("changes", self._check_chains_update_status)

        chem_menu = select_menu.addMenu("Che&mistry")
        chem_menu.setObjectName("Chemistry")
        elements_menu = chem_menu.addMenu("&element")
        elements_menu.setObjectName("element")
        def sel_element(element_name):
            from chimerax.core.commands import run
            run(self.session, "select %s" % element_name)
        for element_name in ["C", "H", "N", "O", "P", "S"]:
            action = QAction(element_name, self)
            action.triggered.connect(lambda arg, *, en=element_name: self.select_by_mode(en))
            elements_menu.addAction(action)

        from chimerax.atomic import Element
        known_elements = [nm for nm in Element.names if len(nm) < 3]
        known_elements.sort()
        from math import sqrt
        num_menus = int(sqrt(len(known_elements)) + 0.5)
        incr = len(known_elements) / num_menus
        start_index = 0
        other_menu = elements_menu.addMenu("other")
        for i in range(num_menus):
            if i < num_menus-1:
                end_index = int((i+1) * incr + 0.5)
            else:
                end_index = len(known_elements) - 1
            submenu = other_menu.addMenu("%s-%s"
                % (known_elements[start_index], known_elements[end_index]))
            for en in known_elements[start_index:end_index+1]:
                action = QAction(en, self)
                action.triggered.connect(lambda arg, *, en=en: self.select_by_mode(en))
                submenu.addAction(action)
            start_index = end_index + 1

        self.select_mode_menu = select_menu.addMenu("mode")
        self.select_mode_menu.setObjectName("mode")
        mode_names =  ["replace", "add to", "subtract from", "intersect with"]
        self._select_mode_reminders = {k:v for k,v in zip(mode_names, 
            ["", " (+)", " (-)", " (\N{INTERSECTION})"])}
        for mode in mode_names:
            action = QAction(mode + self._select_mode_reminders[mode], self)
            self.select_mode_menu.addAction(action)
            action.triggered.connect(
                lambda arg, s=self, m=mode: s._set_select_mode(m))
        self._set_select_mode("replace")

    def select_by_mode(self, selector_text):
        mode = self.select_menu_mode
        if mode == "replace":
            cmd = "sel"
        elif mode == "add to":
            cmd = "sel add"
        elif mode == "subtract from":
            cmd = "sel subtract"
        else:
            cmd = "sel intersect"
        from chimerax.core.commands import run
        run(self.session, "%s %s" % (cmd, selector_text))

    def _set_select_mode(self, mode_text):
        self.select_menu_mode = mode_text
        self.select_mode_menu.setTitle("Menu mode: %s selection" % mode_text)
        mb = self.menuBar()
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtCore import Qt
        select_menu = mb.findChild(QMenu, "Select", Qt.FindDirectChildrenOnly)
        select_menu.setTitle("Select" + self._select_mode_reminders[mode_text])

    def update_favorites_menu(self, session):
        from PyQt5.QtWidgets import QAction
        self.favorites_menu.clear()
        self.favorites_menu.addSeparator()
        settings = QAction("Settings...", self)
        settings.setToolTip("Show/set ChimeraX settings")
        settings.triggered.connect(lambda arg, self=self: self.settings_ui_widget.show())
        self.favorites_menu.addAction(settings)

    def update_select_chains_menu(self):
        if not self.chains_menu_needs_update:
            return
        from chimerax.atomic import AtomicStructures, all_atomic_structures
        structures = AtomicStructures(all_atomic_structures(self.session))
        chain_info = {}
        for chain in structures.chains:
            key = chain.description if chain.description else chain.chain_id
            chain_info.setdefault(key, []).append(chain)
        chain_keys = list(chain_info.keys())
        chain_keys.sort()
        self.select_chains_menu.clear()
        def final_description(description):
            if len(description) < 110:
                return False, description
            return True, description[:50] + "..." + description[-50:]
        from PyQt5.QtWidgets import QAction
        for chain_key in chain_keys:
            chains = chain_info[chain_key]
            if len(chains) > 1:
                submenu = self.select_chains_menu.addMenu(chain_key)
                sep = submenu.addSeparator()
                chains.sort(key=lambda c: (c.structure.id, c.chain_id))
                collective_spec = ""
                for chain in chains:
                    if len(structures) > 1:
                        if chain.description:
                            shortened, final = final_description(chain.description)
                            label = "[%s] chain %s" % (chain.structure, chain.chain_id)
                        else:
                            label = "[%s]" % chain.structure
                            shortened = False
                    else:
                        # ...must be multiple identical descriptions...
                        label = "chain %s" % chain.chain_id
                        shortened = False
                    action = QAction(label, self)
                    spec = chain.string(style="command")
                    action.triggered.connect(lambda *, spec=spec: self.select_by_mode(spec))
                    submenu.addAction(action)
                    collective_spec += spec
                    if shortened:
                        submenu.setToolTipsVisible(True)
                        action.setToolTip(chain.description)
                action = QAction("all", self)
                action.triggered.connect(lambda *, spec=collective_spec: self.select_by_mode(spec))
                submenu.insertAction(sep, action)
            else:
                chain = chains[0]
                chain_id_text = str(chain)
                slash_index = chain_id_text.rfind('/')
                chain_id_text = chain_id_text[:slash_index] + " chain " \
                    + chain_id_text[slash_index+1:]
                if chain.description:
                    shortened, final = final_description(chain.description)
                    label = "%s (%s)" % (final, chain_id_text)
                else:
                    label = chain_id_text
                    shortened = False
                action = QAction(label, self)
                spec = chain.string(style="command")
                action.triggered.connect(lambda *, spec=spec: self.select_by_mode(spec))
                if shortened:
                    self.select_chains_menu.setToolTipsVisible(True)
                    action.setToolTip(chain.description)
                self.select_chains_menu.addAction(action)

        self.chains_menu_needs_update = False

    def update_tools_menu(self, session):
        self._checkbutton_tools = {}
        from PyQt5.QtWidgets import QMenu, QAction
        tools_menu = QMenu("&Tools")
        tools_menu.setToolTipsVisible(True)
        categories = {}
        self._tools_cache = set()
        for bi in session.toolshed.bundle_info(session.logger):
            for tool in bi.tools:
                self._tools_cache.add(tool)
                for cat in tool.categories:
                    categories.setdefault(cat, {})[tool.name] = (bi, tool)
        cat_keys = sorted(categories.keys())
        one_menu = len(cat_keys) == 1
        from chimerax.core.commands import run, quote_if_necessary
        active_tool_names = set([tool.display_name for tool in session.tools.list()])
        for cat in cat_keys:
            if one_menu:
                cat_menu = tools_menu
            else:
                cat_menu = tools_menu.addMenu(cat)
                cat_menu.setToolTipsVisible(True)
            cat_info = categories[cat]
            use_checkbuttons = cat == "Toolbar"
            for tool_name in sorted(cat_info.keys()):
                tool_action = QAction(tool_name, self)
                tool_action.setToolTip(cat_info[tool_name][1].synopsis)
                if use_checkbuttons:
                    tool_action.setCheckable(True)
                    tool_action.setChecked(tool_name in active_tool_names)
                    tool_action.triggered.connect(
                        lambda arg, ses=session, run=run, tool_name=tool_name:
                        run(ses, "toolshed %s %s" % (("show" if arg else "hide"),
                        quote_if_necessary(tool_name))))
                    self._checkbutton_tools[tool_name] = tool_action
                else:
                    tool_action.triggered.connect(
                        lambda arg, ses=session, run=run, tool_name=tool_name:
                        run(ses, "toolshed show %s" % quote_if_necessary(tool_name)))
                cat_menu.addAction(tool_action)
        def _show_toolshed(arg):
            from chimerax.help_viewer import show_url
            from chimerax.core import toolshed
            show_url(session, toolshed.get_toolshed().remote_url)
        more_tools = QAction("More Tools...", self)
        more_tools.setToolTip("Open ChimeraX Toolshed in Help Viewer")
        more_tools.triggered.connect(_show_toolshed)
        tools_menu.addAction(more_tools)
        mb = self.menuBar()
        old_action = self.tools_menu.menuAction()
        mb.insertMenu(old_action, tools_menu)
        mb.removeAction(old_action)
        self.tools_menu = tools_menu

    def _set_tool_checkbuttons(self, toolbar, visibility):
        if toolbar.windowTitle() in self._checkbutton_tools:
            self._checkbutton_tools[toolbar.windowTitle()].setChecked(visibility)

    def add_menu_entry(self, menu_names, entry_name, callback, tool_tip=None):
        '''
        Add a main menu entry.  Adding entries to the Select menu should normally be done
        with the add_select_menu_entry method instead, which allows selectors to honor the
        selection mode of that menu.

        Menus that are needed but that don't already exist (including top-level ones) will
        be created.  Callback function takes no arguments.  This method cannot be used to
        add entries to menus that are updated dynamically, such as Tools.
        '''
        menu = self._get_target_menu(self.menuBar(), menu_names)
        from PyQt5.QtWidgets import QAction
        action = QAction(entry_name, self)
        action.triggered.connect(lambda arg, cb = callback: cb())
        if tool_tip is not None:
            action.setToolTip(tool_tip)
        menu.addAction(action)

    def add_select_menu_entry(self, submenu_names, entry_name, selector_text, tool_tip=None):
        menu = self._get_target_menu(self.menuBar(), ["Select"] + submenu_names)
        self._add_select_menu_entry(menu, entry_name, selector_text, tool_tip)

    def add_select_menu_entries(self, submenu_names, entry_names, selector_texts, tool_tips=None):
        menu = self._get_target_menu(self.menuBar(), ["Select"] + submenu_names)
        if tool_tips is None:
            tool_tips = [None] * len(entry_names)
        for entry_name, selector_text, tool_tip in zip(entry_names, selector_texts, tool_tips):
            self._add_select_menu_entry(menu, entry_name, selector_text, tool_tip)

    def _add_select_menu_entry(self, menu, entry_name, selector_text, tool_tip):
        from PyQt5.QtWidgets import QAction
        action = QAction(entry_name, self)
        action.triggered.connect(lambda arg, st=selector_text: self.select_by_mode(st))
        if tool_tip is not None:
            action.setToolTip(tool_tip)
        menu.addAction(action)

    def _get_target_menu(self, parent_menu, menu_names):
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtCore import Qt
        for menu_name in menu_names:
            menu = parent_menu.findChild(QMenu, menu_name, Qt.FindDirectChildrenOnly)
            add = (menu is None)
            if add:
                menu = QMenu(menu_name, parent_menu)
                menu.setToolTipsVisible(True)
                menu.setObjectName(menu_name)	# Needed for findChild() above to work.
                if parent_menu == self.menuBar():
                    parent_menu.insertMenu(parent_menu.findChild(QMenu, "Help").menuAction(), menu)
                else:
                    parent_menu.addMenu(menu)
            parent_menu = menu
        return parent_menu

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

def _open_dropped_file(session, path):
    if not path:
        return
    from chimerax.core.commands import run, quote_if_necessary
    run(session, 'open %s' % quote_if_necessary(path))

from chimerax.core.logger import StatusLogger
class ToolWindow(StatusLogger):
    """An area that a tool can populate with widgets.

    This class is not used directly.  Instead, a tool makes its main
    window by instantiating the :py:class:`MainToolWindow` class
    (or a subclass thereof), and any subwindows by calling that class's
    :py:meth:`~MainToolWindow.create_child_window` method.

    The window's :py:attr:`ui_area` attribute is the parent to all the tool's
    widgets for this window.  Call :py:meth:`manage` once the widgets
    are set up to show the tool window in the main interface.

    The :py:attr:`close_destroys` keyword controls whether closing this window
    destroys it or hides it.  If it destroys it and this is the main window, all
    the child windows will also be destroyed.

    The :py:attr:`statusbar` keyword controls whether the tool will display
    status messages via an in-window statusbar, or via the main ChimeraX statusbar.
    In either case, the :py:meth:`status` method can be used to issue status
    messages.  It accepts the exact same arguments/keywords as the
    :py:meth:`~..logger.Logger.status` method in the :py:class:`~..logger.Logger` class.
    The resulting QStatusBar widget (or None if statusbar was False) will be
    available from the ToolWindow's "statusbar" in case you need to add widgets to it
    or otherwise customize it.

    """

    #: Where the window can be placed in the main interface;
    #: 'side' is either left or right, depending on user preference
    placements = ["side", "right", "left", "top", "bottom"]

    def __init__(self, tool_instance, title, *, close_destroys=True, statusbar=False):
        StatusLogger.__init__(self, tool_instance.session)
        self.tool_instance = tool_instance
        self.close_destroys = close_destroys
        mw = tool_instance.session.ui.main_window
        self.__toolkit = _Qt(self, title, statusbar, mw)
        self.ui_area = self.__toolkit.ui_area
        mw._new_tool_window(self)
        self._kludge = self.__toolkit

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
        self.session.ui.main_window._tool_window_destroy(self)

    def fill_context_menu(self, menu, x, y):
        """Add items to this tool window's context menu,
        whose downclick occurred at position (x,y)

        Override to add items to any context menu popped up over this window.

        Note that you have to keep references to the actions you add to the
        menu to avoid having then automatically destroyed and removed from the
        menu when this method returns.  You can use the menu itself to store 
        the reference, e.g. menu._ref1 = QAction(...)"""
        pass

    @property
    def floating(self):
        return self.__toolkit.dock_widget.isFloating()

    from PyQt5.QtCore import Qt
    def manage(self, placement, fixed_size=False, allowed_areas=Qt.AllDockWidgetAreas):
        """Show this tool window in the interface

        Tool will be docked into main window on the side indicated by
        `placement` (which should be a value from :py:attr:`placements`
        or None, or another tool window).  If `placement` is None, the tool will
        be detached from the main window.  If `placement` is another tool window,
        then those tools will be tabbed together.

        The tool window will be allowed to dock in the allowed_areas, the value
        of which is a bitmask formed from Qt's Qt.DockWidgetAreas flags.
        """
        if self.tool_instance.tool_name in self.session.ui.settings.undockable:
            from PyQt5.QtCore import Qt
            allowed_areas = Qt.NoDockWidgetArea
        self.__toolkit.manage(placement, allowed_areas, fixed_size)

    def _get_shown(self):
        """Whether this window is hidden or shown"""
        return self.__toolkit.shown

    def _set_shown(self, shown):
        self.session.ui.main_window._tool_window_request_shown(self, shown)

    shown = property(_get_shown, _set_shown)

    def shown_changed(self, shown):
        """Perform actions when window hidden/shown

        Override to perform any actions you want done when the window
        is hidden (\ `shown` = False) or shown (\ `shown` = True)"""
        pass

    def status(self, *args, **kw):
        if self._have_statusbar:
            StatusLogger.status(self, *args, **kw)
        else:
            self.session.logger.status(*args, **kw)

    @property
    def _have_statusbar(self):
        """Does this window have a QStatusBar widget"""
        tk = self.__toolkit
        return tk is not None and tk.status_bar is not None

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
        if self._have_statusbar:
            self.clear()
        self.__toolkit.destroy()
        self.__toolkit = None

    @property
    def _dock_widget(self):
        return self.__toolkit.dock_widget

    def _mw_set_dockable(self, dockable):
        self.__toolkit.dockable = dockable

    def _mw_set_shown(self, shown):
        self.__toolkit.shown = shown
        self.shown_changed(shown)

    def _prioritized_logs(self):
        return [self.__toolkit.status_bar]

    def _show_context_menu(self, event):
        # this routine needed as a kludge to allow QwebEngine to show
        # our own context menu
        self.__toolkit.show_context_menu(event)

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
    def __init__(self, tool_instance, **kw):
        super().__init__(tool_instance, tool_instance.display_name, **kw)

    def create_child_window(self, title, *, window_class=None, **kw):
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
        kw : Keywords to pass on to the tool window's constructor
        """

        if window_class is None:
            window_class = ChildToolWindow
        elif not issubclass(window_class, ChildToolWindow):
            raise ValueError(
                "Child window class must inherit from ChildToolWindow")
        return window_class(self.tool_instance, title, **kw)

class ChildToolWindow(ToolWindow):
    """Child (*i.e.* additional) tool window

    Only created through use of
    :py:meth:`MainToolWindow.create_child_window` method.
    """
    def __init__(self, tool_instance, title, **kw):
        super().__init__(tool_instance, title, **kw)

class _Qt:
    def __init__(self, tool_window, title, has_statusbar, main_window):
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

        from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout
        self.dock_widget = dw = QDockWidget(title, mw)
        dw.closeEvent = lambda e, tw=tool_window, mw=mw: mw.close_request(tw, e)
        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 1, 0, 0) # all zeros produces a complaint about -1 height
        self.ui_area = QWidget()
        self.ui_area.contextMenuEvent = lambda e, self=self: self.show_context_menu(e)
        layout.addWidget(self.ui_area)
        if has_statusbar:
            session = tool_window.tool_instance.session
            from .statusbar import _StatusBar
            self.status_bar = sbar = _StatusBar(session)
            layout.addWidget(sbar.widget)
        else:
            self.status_bar = None
        container.setLayout(layout)
        self.dock_widget.setWidget(container)

    def destroy(self):
        if not self.tool_window:
            # already destroyed
            return
        # free up references
        self.tool_window = None
        self.main_window = None
        self.ui_area.destroy()
        if self.status_bar:
            self.status_bar.destroy()
            self.status_bar = None
        self.dock_widget.destroy()

    def _get_dockable(self):
        from PyQt5.QtCore import Qt
        return self.dock_widget.allowedAreas() != Qt.NoDockWidgetArea

    def _set_dockable(self, dockable):
        from PyQt5.QtCore import Qt
        areas = Qt.AllDockWidgetAreas if dockable else Qt.NoDockWidgetArea
        self.dock_widget.setAllowedAreas(areas)
        if not dockable and not self.dock_widget.isFloating():
            self.dock_widget.setFloating(True)

    dockable = property(_get_dockable, _set_dockable)

    def manage(self, placement, allowed_areas, fixed_size=False):
        from PyQt5.QtCore import Qt
        placements = self.tool_window.placements
        if placement is None or isinstance(placement, ToolWindow):
            side = Qt.RightDockWidgetArea
        else:
            if placement not in placements:
                raise ValueError("placement value must be one of: {}, or None"
                    .format(", ".join(placements)))
            else:
                side = self.placement_map[placement]

        # With in-window status bar support now creating an additional layer
        # of containing widgets, the following updateGeometry call now seems
        # to be necessary to get the outermost widget to request the right size
        # (most noticeable for initially-undocked tools)
        self.ui_area.updateGeometry()
        mw = self.main_window
        if isinstance(placement, ToolWindow):
            mw.tabifyDockWidget(placement._dock_widget, self.dock_widget)
        else:
            mw.addDockWidget(side, self.dock_widget)
            if placement is None or allowed_areas == Qt.NoDockWidgetArea:
                self.dock_widget.setFloating(True)
        self.dock_widget.setAllowedAreas(allowed_areas)

        #QT disable: create a 'hide_title_bar' option
        if side == Qt.BottomDockWidgetArea:
            from PyQt5.QtWidgets import QWidget
            self.dock_widget.setTitleBarWidget(QWidget())

        if self.tool_window.close_destroys:
            self.dock_widget.setAttribute(Qt.WA_DeleteOnClose)

    def show_context_menu(self, event):
        show_context_menu(event, self.tool_window.tool_instance, self.tool_window.fill_context_menu,
            self.tool_window.tool_instance.tool_info in self.main_window._tools_cache)

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
            self.errors = "ignore"

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

def show_context_menu(event, tool_instance, fill_cb, autostartable):
    from PyQt5.QtWidgets import QMenu, QAction
    menu = QMenu()

    if fill_cb:
        fill_cb(menu, event.x(), event.y())
    if not menu.isEmpty():
        menu.addSeparator()
    ti = tool_instance
    hide_tool_action = QAction("Hide tool")
    hide_tool_action.triggered.connect(lambda arg, ti=ti: ti.display(False))
    menu.addAction(hide_tool_action)
    if ti.help is not None:
        help_action = QAction("Help")
        help_action.setStatusTip("Show tool help")
        help_action.triggered.connect(lambda arg, ti=ti: ti.display_help())
        menu.addAction(help_action)
    else:
        no_help_action = QAction("No help available")
        no_help_action.setEnabled(False)
        menu.addAction(no_help_action)
    session = ti.session
    if autostartable:
        autostart = ti.tool_name in session.ui.settings.autostart
        auto_action = QAction("Start at ChimeraX startup")
        auto_action.setCheckable(True)
        auto_action.setChecked(autostart)
        from chimerax.core.commands import run, quote_if_necessary
        auto_action.triggered.connect(
            lambda arg, ses=session, run=run, tool_name=ti.tool_name:
            run(ses, "ui autostart %s %s" % (("true" if arg else "false"),
            quote_if_necessary(ti.tool_name))))
        menu.addAction(auto_action)
    undockable = ti.tool_name in session.ui.settings.undockable
    dock_action = QAction("Dockable tool")
    dock_action.setCheckable(True)
    dock_action.setChecked(not undockable)
    from chimerax.core.commands import run, quote_if_necessary
    dock_action.triggered.connect(
        lambda arg, ses=session, run=run, tool_name=ti.tool_name:
        run(ses, "ui dockable %s %s" % (("true" if arg else "false"),
        quote_if_necessary(ti.tool_name))))
    menu.addAction(dock_action)
    menu.exec(event.globalPos())
