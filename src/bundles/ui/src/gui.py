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
gui: Main ChimeraX graphical user interface
===========================================

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
        self.main_window = None
        self.already_quit = False
        self.session = session

        from .settings import UI_Settings
        self.settings = UI_Settings(session, "ui")

        from chimerax.mouse_modes import MouseModes
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
            QtFatalMsg: Log.LEVEL_BUG,
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
        pixmap = w.grab()
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
        """Supported API.  Call function 'func' in a thread-safe manner
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

        from chimerax.core.core_settings import settings as core_settings
        sizing_scheme, size_data = core_settings.initial_window_size
        if sizing_scheme == "last used" and core_settings.last_window_size is None:
            sizing_scheme = "proportional"
            size_data = (0.67, 0.67)
        if sizing_scheme == "last used":
            width, height = core_settings.last_window_size
        elif sizing_scheme == "proportional":
            wf, hf = size_data
            dw = QDesktopWidget()
            main_screen_geom = ui.primaryScreen().availableGeometry()
            width, height = main_screen_geom.width()*wf, main_screen_geom.height()*hf
        else:
            width, height = size_data
        self.resize(width, height)

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

        from .open_folder import OpenFolderDialog
        self._open_folder = OpenFolderDialog(self, session)

        from .save_dialog import MainSaveDialog, ImageSaver
        self.save_dialog = MainSaveDialog(self)
        ImageSaver(self.save_dialog).register()

        self._hide_tools = False
        self.tool_instance_to_windows = {}
        self._fill_tb_context_menu_cbs = {}
        self._select_seq_dialog = self._select_zone_dialog = None
        self._presets_menu_needs_update = True
        session.presets.triggers.add_handler("presets changed",
            lambda *args, s=self: setattr(s, '_presets_menu_needs_update', True))

        self._build_status()
        self._populate_menus(session)

        # set icon for About dialog
        from chimerax import app_dirs as ad, app_data_dir
        import os.path
        icon_path = os.path.join(app_data_dir, "%s-icon512.png" % ad.appname)
        if os.path.exists(icon_path):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))

        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('resized')

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
        settings =  self.session.ui.settings
        if tool.tool_name in settings.tool_positions['toolbars']:
            version, placement, geom_info = settings.tool_positions['toolbars'][tool.tool_name]
            if placement is None:
                self.session.logger.info("Cannot restore toolbar as floating")
                #from PyQt5.QtCore import Qt
                #QMainWindow.addToolBar(self, Qt.NoToolBarArea, tb)
                #from PyQt5.QtCore import QRect
                #geometry = QRect(*geom_info)
                #tb.setGeometry(geometry)
            else:
                QMainWindow.addToolBar(self, placement, tb)
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
            ## The following commented-out open command doesn't get multiple volume-plane files
            ## to open as a single volume, whereas the uncommented code does
            #run(session, "open " + " ".join([quote_if_necessary(p) for p in paths]))
            if len(paths) == 1:
                run(session, "open " + quote_if_necessary(paths[0]))
            else:
                # TODO: Make open command handle this including saving in file history.
                suffixes = set(p[p.rfind('.'):] for p in paths)
                if len(suffixes) == 1:
                    # Files have same suffix, open as a single group
                    session.models.open(paths)
                else:
                    # Files have more than one suffix, open each at top-level model.
                    for p in paths:
                        session.models.open([p])

        # Opening the model directly adversely affects Qt interfaces that show
        # as a result.  In particular, Multalign Viewer no longer gets hover
        # events correctly, nor tool tips.
        #
        # Using session.ui.thread_safe() doesn't help either(!)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, _qt_safe)

    def folder_open_cb(self, session):
        self._open_folder.display(session)

    def file_save_cb(self, session):
        self.save_dialog.display(self, session)

    def file_close_cb(self, session):
        from chimerax.core.commands import run
        run(session, 'close session')

    def file_quit_cb(self, session):
        session.ui.quit()

    def edit_undo_cb(self, session):
        from chimerax.core.commands import run
        run(session, 'undo')

    def edit_redo_cb(self, session):
        from chimerax.core.commands import run
        run(session, 'redo')

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

    @property
    def hide_tools(self):
        return self._hide_tools

    @hide_tools.setter
    def hide_tools(self, ht):
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

    @property
    def rapid_access_shown(self):
        return self._stack.currentWidget() == self.rapid_access

    @rapid_access_shown.setter
    def rapid_access_shown(self, show):
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

    def _check_rapid_access(self, *args):
        self.rapid_access_shown = len(self.session.models) == 0

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        from chimerax.core.core_settings import settings as core_settings
        size = event.size()
        wh = (size.width(), size.height())
        core_settings.last_window_size = wh
        self.triggers.activate_trigger('resized', wh)

    def show_select_seq_dialog(self, *args):
        if self._select_seq_dialog is None:
            self._select_seq_dialog = SelSeqDialog(self.session)
        self._select_seq_dialog.show()
        self._select_seq_dialog.raise_()

    def show_select_zone_dialog(self, *args):
        if self._select_zone_dialog is None:
            self._select_zone_dialog = SelZoneDialog(self.session)
        self._select_zone_dialog.show()
        self._select_zone_dialog.raise_()

    def show_tb_context_menu(self, tb, event):
        tool, fill_cb = self._fill_tb_context_menu_cbs[tb]
        _show_context_menu(event, tool, fill_cb, True, tb)

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
        open_folder_action = QAction("Open DICOM Folder...", self)
        open_folder_action.setToolTip("Open data in folder")
        open_folder_action.triggered.connect(lambda arg, s=self, sess=session: s.folder_open_cb(sess))
        file_menu.addAction(open_folder_action)
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

        self.presets_menu = mb.addMenu("Presets")
        self.presets_menu.setToolTipsVisible(True)
        self.presets_menu.aboutToShow.connect(lambda ses=session: self._populate_presets_menu(ses))
        self._populate_presets_menu(session)

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

    def _populate_presets_menu(self, session):
        if not self._presets_menu_needs_update:
            return
        self.presets_menu.clear()
        preset_info = session.presets.presets_by_category
        self._presets_menu_needs_update = False
       
        if not preset_info:
            return
        
        if len(preset_info) == 1:
            self._uncategorized_preset_menu(session, preset_info)
        elif len(preset_info) + sum([len(v) for v in preset_info.values()]) < 20:
            self._inline_categorized_preset_menu(session, preset_info)
        else:
            self._rollover_categorized_preset_menu(session, preset_info)
    
    def _uncategorized_preset_menu(self, session, preset_info):
        for category, preset_names in preset_info.items():
            self._add_preset_entries(session, self.presets_menu, preset_names)

    def _inline_categorized_preset_menu(self, session, preset_info):
        from PyQt5.QtWidgets import QAction
        categories = self._order_preset_categories(preset_info.keys())
        for cat in categories:
            # need to force the category text to be shown, so can't use
            # addSection, which might not show it
            sep = QAction("— %s —" % menu_capitalize(cat), self.presets_menu)
            sep.setEnabled(False)
            self.presets_menu.addAction(sep)
            self._add_preset_entries(session, self.presets_menu, preset_info[cat], cat)

    def _rollover_categorized_preset_menu(self, session, preset_info):
        categories = self._order_preset_categories(preset_info.keys())
        for cat in categories:
            cat_menu = self.presets_menu.addMenu(menu_capitalize(cat))
            self._add_preset_entries(session, cat_menu, preset_info[cat], cat)

    def _order_preset_categories(self, categories):
        cats = list(categories)[:]
        cats.sort(key=lambda x: x.lower())
        return cats

    def _add_preset_entries(self, session, menu, preset_names, category=None):
        from PyQt5.QtWidgets import QAction
        from chimerax.core.commands import run, quote_if_necessary
        # the menu names may be instances of CustomSortString, so sort them
        # before applying menu_capitalize(); also 'preset_names' may be a keys view
        menu_names = list(preset_names)
        menu_names.sort(key=lambda x: x.lower())
        menu_names = [menu_capitalize(name) for name in menu_names]
        if category is None:
            cat_string = ""
        else:
            cat_string = quote_if_necessary(category.lower()) + " "
        for name in menu_names:
            action = QAction(name, menu)
            action.triggered.connect(lambda checked, ses=session, name=name, cat=cat_string:
                run(ses, "preset %s%s" % (cat, quote_if_necessary(name.lower()))))
            menu.addAction(action)

    def _populate_select_menu(self, select_menu):
        from PyQt5.QtWidgets import QAction
        sel_seq_action = QAction("Sequence...", self)
        select_menu.addAction(sel_seq_action)
        sel_seq_action.triggered.connect(self.show_select_seq_dialog)
        sel_zone_action = QAction("&Zone...", self)
        select_menu.addAction(sel_zone_action)
        sel_zone_action.triggered.connect(self.show_select_zone_dialog)
        sel_clear_action = QAction("&Clear", self)
        select_menu.addAction(sel_clear_action)
        from chimerax.core.commands import run
        sel_clear_action.triggered.connect(lambda *args, run=run, ses=self.session:
            run(ses, "sel clear"))

        self.select_mode_menu = select_menu.addMenu("mode")
        self.select_mode_menu.setObjectName("mode")
        mode_names =  ["replace", "add", "subtract", "intersect"]
        self._select_mode_reminders = {k:v for k,v in zip(mode_names,
            ["", " (+)", " (-)", " (\N{INTERSECTION})"])}
        for mode in mode_names:
            mode_action = QAction(mode.title() + self._select_mode_reminders[mode], self)
            self.select_mode_menu.addAction(mode_action)
            mode_action.triggered.connect(
                lambda arg, s=self, m=mode: s._set_select_mode(m))
        self._set_select_mode("replace")

    def select_by_mode(self, selector_text):
        """Supported API.  Select based on the selector 'selector_text' but honoring the current
           selection mode chosen in the Select menu.  Typically used by callbacks off the Selection menu"""
        mode = self.select_menu_mode
        if mode == "replace":
            cmd = "sel"
        elif mode == "add":
            cmd = "sel add"
        elif mode == "subtract":
            cmd = "sel subtract"
        else:
            cmd = "sel intersect"
        from chimerax.core.commands import run
        run(self.session, "%s %s" % (cmd, selector_text))

    def _set_select_mode(self, mode_text):
        self.select_menu_mode = mode_text
        self.select_mode_menu.setTitle("Menu Mode: %s" % mode_text.title())
        mb = self.menuBar()
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtCore import Qt
        select_menu = mb.findChild(QMenu, "Select", Qt.FindDirectChildrenOnly)
        select_menu.setTitle("Select" + self._select_mode_reminders[mode_text])

    def update_favorites_menu(self, session):
        from PyQt5.QtWidgets import QAction
        from chimerax.core.commands import run, quote_if_necessary
        # Due to Settings possibly being displayed in another menu (but the actions
        # still being in this menu), be tricky about clearing out menu
        prev_actions = self.favorites_menu.actions()
        if prev_actions:
            separator, settings = prev_actions[-2:]
            for action in prev_actions[:-2]:
                self.favorites_menu.removeAction(action)
        for fave in session.ui.settings.favorites:
            fave_action = QAction(fave, self)
            fave_action.triggered.connect(lambda arg, ses=session, run=run, fave=fave:
                run(ses, "toolshed show %s" % (quote_if_necessary(fave))))
            if prev_actions:
                self.favorites_menu.insertAction(separator, fave_action)
            else:
                self.favorites_menu.addAction(fave_action)
        if not prev_actions:
            self.favorites_menu.addSeparator()
            settings = QAction("Settings...", self)
            settings.setToolTip("Show/set ChimeraX settings")
            settings.triggered.connect(lambda arg, self=self: self.settings_ui_widget.show())
            self.favorites_menu.addAction(settings)

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

    def add_menu_entry(self, menu_names, entry_name, callback, *, tool_tip=None, insertion_point=None):
        '''Supported API.
        Add a main menu entry.  Adding entries to the Select menu should normally be done via
        the add_select_submenu method instead.  For details, see the doc string for that method.

        Menus that are needed but that don't already exist (including top-level ones) will
        be created.  The menu names (and entry name) can contain appropriate keyboard navigation
        markup.  Callback function takes no arguments.  This method cannot be used to add entries
        to menus that are updated dynamically, such as Tools.

        If 'insertion_point is specified, then the entry will be inserted before it.
        'insertion_point' can be a QAction, a string (menu item text with navigation markup removed)
        or an integer indicating a particular separator (top to bottom, numbering starting at 1).
        '''
        menu = self._get_target_menu(self.menuBar(), menu_names)
        from PyQt5.QtWidgets import QAction
        action = QAction(entry_name, self)
        action.triggered.connect(lambda arg, cb = callback: cb())
        if tool_tip is not None:
            action.setToolTip(tool_tip)
        if insertion_point is None:
            menu.addAction(action)
        else:
            menu.insertAction(self._get_menu_action(menu, insertion_point), action)
        return action

    def add_select_submenu(self, parent_menu_names, submenu_name, *, append=True):
        '''Supported API.
        Add a submenu (or get it if it already exists).  Any parent menus will be created as
        needed.  Menu names can contain keyboard navigation markup (the '&' character).
        'parent_menu_names' should not contain the Select menu itself. If 'append' is True then
        the menu will be appended at the end of any existing items, otherwise it will be
        alphabetized into them.

        The caller is responsible for filling out the menu with entries, separators, etc.
        Any further submenus should again use this call.  Entry or menu callbacks that
        actually make selections should use the select_by_mode(selector_text) method
        to make the selection, which will run the command appropriate to the current
        selection mode of the Select menu.

        The convenience method add_menu_selector, which takes a menu, a label, and an
        optional selector text (defaulting to the label) can be used to easily add items
        to the menu.
        '''

        insert_positions = [None, "Sequence..."] + [False if append else None] * len(parent_menu_names)
        return self._get_target_menu(self.menuBar(),
            ["Select"] + parent_menu_names + [submenu_name],
            insert_positions=insert_positions)

    def add_menu_selector(self, menu, label, selector_text=None, *, insertion_point=None):
        '''Supported API.
        Add an item to the given menu (which was probably obtained with the add_select_submenu
        method) which will make a selection using the given selector text (which should just
        be the text of the selector, not a full command) while honoring the current selection
        mode set in the Select menu.  If 'selector_text' is not given, it defaults to be the
        same as 'label'.  The label can have keyboard navigation markup.  If 'insertion_point'
        is specified, then the item will be inserted before it.  'insertion_point' can be a
        QAction, a string (menu item text with navigation markup removed) or an integer
        indicating a particular separator (top to bottom, numbering starting at 1).
        '''
        if selector_text is None:
            selector_text = remove_keyboard_navigation(label)
        from PyQt5.QtWidgets import QAction
        action = QAction(label, self)
        action.triggered.connect(lambda *, st=selector_text: self.select_by_mode(st))
        if insertion_point is None:
            menu.addAction(action)
        else:
            menu.insertAction(self._get_menu_action(menu, insertion_point), action)
        return action

    def _get_menu_action(self, menu, insertion_point):
        from PyQt5.QtWidgets import QAction
        if isinstance(insertion_point, QAction):
            return insertion_point
        sep_count = 0
        for menu_action in menu.actions():
            if menu_action.isSeparator():
                sep_count += 1
                if insertion_point == sep_count:
                    return menu_action
            elif isinstance(insertion_point, str) \
            and menu_action.text().lower().replace('&', '', 1) == insertion_point.lower():
                return menu_action
        raise ValueError("Requested menu insertion point (%s) not found" % str(insertion_point))

    def _get_target_menu(self, parent_menu, menu_names, *, insert_positions=None):
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtCore import Qt
        if insert_positions is None:
            insert_positions = [None]*len(menu_names)
        for menu_name, insert_pos in zip(menu_names, insert_positions):
            obj_name = remove_keyboard_navigation(menu_name)
            menu = parent_menu.findChild(QMenu, obj_name, Qt.FindDirectChildrenOnly)
            add = (menu is None)
            if add:
                if parent_menu == self.menuBar() and insert_pos is None:
                    insert_pos = "Help"
                if insert_pos is None:
                    # try to alphabetize; use an insert_pos of False to prevent this
                    existing_actions = parent_menu.actions()
                    names = [(obj_name, None)]
                    for ea in existing_actions:
                        names.append((remove_keyboard_navigation(ea.text()), ea))
                    names.sort(key=lambda n: n[0].lower())
                    i = names.index((obj_name, None))
                    if i < len(names) - 1:
                        insert_action = names[i+1][1]
                    else:
                        insert_pos = False
                elif insert_pos is not False:
                    simple_text_insert_pos = remove_keyboard_navigation(insert_pos)
                    for action in parent_menu.actions():
                        if remove_keyboard_navigation(action.text()) == simple_text_insert_pos:
                            insert_action = action
                            break
                    else:
                        # look for obj name
                        for child in parent_menu.children():
                            if child.objectName() == simple_text_insert_pos:
                                insert_action = child.menuAction()
                                break
                        else:
                            raise ValueError("Could not find insert point '%s' in %s" %
                                (insert_pos, ("menu %s" % parent_menu.title()
                                if isinstance(parent_menu, QMenu) else "main menubar")))
                # create here rather than earlier so that's it's not included in parent_menu.children()
                menu = QMenu(menu_name, parent_menu)
                menu.setToolTipsVisible(True)
                menu.setObjectName(obj_name)	# Needed for findChild() above to work.
                if insert_pos is False:
                    parent_menu.addMenu(menu)
                else:
                    parent_menu.insertMenu(insert_action, menu)
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
    """Supported API. An area that a tool can populate with widgets.

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
        ui = tool_instance.session.ui
        mw = ui.main_window
        self.__toolkit = _Qt(self, title, statusbar, mw)
        self.ui_area = self.__toolkit.ui_area
        # forward unused keystrokes (to the command line by default)
        self.ui_area.keyPressEvent = self._forward_keystroke
        mw._new_tool_window(self)
        self._kludge = self.__toolkit

    def cleanup(self):
        """Supported API. Perform tool-specific cleanup

        Override this method to perform additional actions needed when
        the window is destroyed"""
        pass

    def destroy(self):
        """Supported API. Called to destroy the window (from non-UI code)

        Destroying a tool's main window will also destroy all its
        child windows.
        """
        self.session.ui.main_window._tool_window_destroy(self)

    def fill_context_menu(self, menu, x, y):
        """Supported API. Add items to this tool window's context menu,
        whose downclick occurred at position (x,y)

        Override to add items to any context menu popped up over this window.

        Note that you need to specify the 'parent' argument of the QAction
        constructor (as 'menu') to avoid having the action automatically destroyed
        and removed from the menu when this method returns."""
        pass

    @property
    def floating(self):
        return self.__toolkit.dock_widget.isFloating()

    from PyQt5.QtCore import Qt
    window_placement_to_text = {
        Qt.RightDockWidgetArea: "right",
        Qt.LeftDockWidgetArea: "left",
        Qt.TopDockWidgetArea: "top",
        Qt.BottomDockWidgetArea: "bottom"
    }
    def manage(self, placement, fixed_size=False, allowed_areas=Qt.AllDockWidgetAreas):
        """Supported API. Show this tool window in the interface

        Tool will be docked into main window on the side indicated by
        `placement` (which should be a value from :py:attr:`placements` or 'side'
        or None, or another tool window).  If `placement` is "side", then the user-preferred
        side will be used.  If `placement` is None, the tool will
        be detached from the main window.  If `placement` is another tool window,
        then those tools will be tabbed together.

        The tool window will be allowed to dock in the allowed_areas, the value
        of which is a bitmask formed from Qt's Qt.DockWidgetAreas flags.
        """
        settings =  self.session.ui.settings
        tool_name = self.tool_instance.tool_name
        if tool_name in settings.undockable:
            from PyQt5.QtCore import Qt
            allowed_areas = Qt.NoDockWidgetArea
        geometry = None
        if tool_name in settings.tool_positions['windows'] and isinstance(self, MainToolWindow):
            version, placement, geom_info = settings.tool_positions['windows'][tool_name]
            if placement is not None:
                placement = self.window_placement_to_text[placement]
            if geom_info is not None:
                from PyQt5.QtCore import QRect
                geometry = QRect(*geom_info)
        self.__toolkit.manage(placement, allowed_areas, fixed_size, geometry)

    @property
    def shown(self):
        """Whether this window is hidden or shown"""
        return self.__toolkit.shown

    @shown.setter
    def shown(self, shown):
        self.session.ui.main_window._tool_window_request_shown(self, shown)

    def shown_changed(self, shown):
        """Supported API. Perform actions when window hidden/shown

        Override to perform any actions you want done when the window
        is hidden (\ `shown` = False) or shown (\ `shown` = True)"""
        pass

    def status(self, *args, **kw):
        """Supported API.  Show a status message for the tool."""
        if self._have_statusbar:
            StatusLogger.status(self, *args, **kw)
        else:
            self.session.logger.status(*args, **kw)

    @property
    def _have_statusbar(self):
        """Does this window have a QStatusBar widget"""
        tk = self.__toolkit
        return tk is not None and tk.status_bar is not None

    @property
    def title(self):
        """Supported API.  Get/change window title."""
        if self.__toolkit is None:
            return ""
        return self.__toolkit.title

    @title.setter
    def title(self, title):
        if self.__toolkit is None:
            return
        self.__toolkit.set_title(title)

    def _destroy(self):
        self.cleanup()
        if self._have_statusbar:
            self.clear()
        self.__toolkit.destroy()
        self.__toolkit = None

    @property
    def _dock_widget(self):
        return self.__toolkit.dock_widget

    def _forward_keystroke(self, event):
        # Exclude floating windows because they don't forward all keystrokes (e.g. Delete)
        # and because the Google sign-on (via the typically floating Help Viewer) forwards
        # _just_ the Return key (well, and shift/control/other non-printable)
        #
        # QLineEdits don't eat Return keys, so they may propagate to the
        # top widget; don't forward keys if the focus widget is a QLineEdit
        from PyQt5.QtWidgets import QLineEdit, QComboBox
        if not self.floating and not isinstance(self.ui_area.focusWidget(), (QLineEdit, QComboBox)):
            self.tool_instance.session.ui.forward_keystroke(event)

    def _mw_set_dockable(self, dockable):
        self.__toolkit.dockable = dockable

    def _mw_set_shown(self, shown):
        self.__toolkit.shown = shown
        self.shown_changed(shown)

    def _prioritized_logs(self):
        return [self.__toolkit.status_bar]

    def _show_context_menu(self, event):
        # this routine needed as a kludge to allow QWebEngine to show
        # our own context menu
        self.__toolkit.show_context_menu(event)

class MainToolWindow(ToolWindow):
    """Supported API. Class used to generate tool's main UI window.

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
        """Supported API. Make additional tool window

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
    """Supported API. Child (*i.e.* additional) tool window

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
        self.main_window.removeDockWidget(self.dock_widget)
        # free up references
        self.tool_window = None
        self.main_window = None
        self.dock_widget.widget().destroy()
        if self.status_bar:
            self.status_bar.destroy()
            self.status_bar = None
        self.dock_widget.destroy()

    @property
    def dockable(self):
        from PyQt5.QtCore import Qt
        return self.dock_widget.allowedAreas() != Qt.NoDockWidgetArea

    @dockable.setter
    def dockable(self, dockable):
        from PyQt5.QtCore import Qt
        areas = Qt.AllDockWidgetAreas if dockable else Qt.NoDockWidgetArea
        self.dock_widget.setAllowedAreas(areas)
        if not dockable and not self.dock_widget.isFloating():
            self.dock_widget.setFloating(True)

    def manage(self, placement, allowed_areas, fixed_size, geometry):
        # map 'side' to the user's preferred side
        from chimerax.core.core_settings import settings as core_settings
        from PyQt5.QtCore import Qt
        pref_area = Qt.RightDockWidgetArea if core_settings.default_tool_window_side == "right" \
            else Qt.LeftDockWidgetArea
        qt_sides = [pref_area, Qt.RightDockWidgetArea, Qt.LeftDockWidgetArea,
            Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea]
        self.placement_map = dict(zip(self.tool_window.placements, qt_sides))
        placements = self.tool_window.placements
        if placement is None or isinstance(placement, ToolWindow):
            side = Qt.RightDockWidgetArea
        else:
            try:
                side = self.placement_map[placement]
            except KeyError:
                raise ValueError("placement value must be one of: {}, or None"
                    .format(", ".join(placements)))

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
        if geometry is not None:
            self.dock_widget.setGeometry(geometry)
        self.dock_widget.setAllowedAreas(allowed_areas)

        #QT disable: create a 'hide_title_bar' option
        if side == Qt.BottomDockWidgetArea:
            from PyQt5.QtWidgets import QWidget
            self.dock_widget.setTitleBarWidget(QWidget())

        if self.tool_window.close_destroys:
            self.dock_widget.setAttribute(Qt.WA_DeleteOnClose)

    def show_context_menu(self, event):
        _show_context_menu(event, self.tool_window.tool_instance,
            self.tool_window.fill_context_menu,
            self.tool_window.tool_instance.tool_info in self.main_window._tools_cache,
            self.dock_widget if isinstance(self.tool_window, MainToolWindow) else None)

    @property
    def shown(self):
        return not self.dock_widget.isHidden()

    @shown.setter
    def shown(self, shown):
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

def _show_context_menu(event, tool_instance, fill_cb, autostartable, memorable):
    from PyQt5.QtWidgets import QMenu, QAction
    menu = QMenu()

    if fill_cb:
        fill_cb(menu, event.x(), event.y())
    if not menu.isEmpty():
        menu.addSeparator()
    ti = tool_instance
    hide_tool_action = QAction("Hide Tool")
    hide_tool_action.triggered.connect(lambda arg, ti=ti: ti.display(False))
    menu.addAction(hide_tool_action)
    if ti.help is not None:
        help_action = QAction("Help")
        help_action.setStatusTip("Show tool help")
        help_action.triggered.connect(lambda arg, ti=ti: ti.display_help())
        menu.addAction(help_action)
    else:
        no_help_action = QAction("No Help Available")
        no_help_action.setEnabled(False)
        menu.addAction(no_help_action)
    session = ti.session
    if autostartable:
        autostart = ti.tool_name in session.ui.settings.autostart
        auto_action = QAction("Start at ChimeraX Startup")
        auto_action.setCheckable(True)
        auto_action.setChecked(autostart)
        from chimerax.core.commands import run, quote_if_necessary
        auto_action.triggered.connect(
            lambda arg, ses=session, run=run, tool_name=ti.tool_name:
            run(ses, "ui autostart %s %s" % (("true" if arg else "false"),
            quote_if_necessary(ti.tool_name))))
        menu.addAction(auto_action)
        favorite = ti.tool_name in session.ui.settings.favorites
        fav_action = QAction("In Favorites Menu")
        fav_action.setCheckable(True)
        fav_action.setChecked(favorite)
        from chimerax.core.commands import run, quote_if_necessary
        fav_action.triggered.connect(
            lambda arg, ses=session, run=run, tool_name=ti.tool_name:
            run(ses, "ui favorite %s %s" % (("true" if arg else "false"),
            quote_if_necessary(ti.tool_name))))
        menu.addAction(fav_action)
    undockable = ti.tool_name in session.ui.settings.undockable
    dock_action = QAction("Dockable Tool")
    dock_action.setCheckable(True)
    dock_action.setChecked(not undockable)
    from chimerax.core.commands import run, quote_if_necessary
    dock_action.triggered.connect(
        lambda arg, ses=session, run=run, tool_name=ti.tool_name:
        run(ses, "ui dockable %s %s" % (("true" if arg else "false"),
        quote_if_necessary(ti.tool_name))))
    menu.addAction(dock_action)
    if memorable:
        position_action = QAction("Save Tool Position")
        position_action.setStatusTip("Use current docked side,"
            " or undocked size/position as default")
        position_action.triggered.connect(lambda arg, ui=session.ui, widget=memorable, ti=ti:
            _remember_tool_pos(ui, ti, widget))
        menu.addAction(position_action)
    menu.exec(event.globalPos())

def _remember_tool_pos(ui, tool_instance, widget):
    mw = ui.main_window
    from PyQt5.QtWidgets import QToolBar
    # need to copy _before_ modifying, so that default isn't also changed
    from copy import deepcopy
    remembered = deepcopy(ui.settings.tool_positions)
    if isinstance(widget, QToolBar):
        if widget.isFloating():
            from chimerax.core.errors import LimitationError
            raise LimitationError("Cannot currently save toolbars as floating")
        get_side = mw.toolBarArea
        mem_location = remembered['toolbars']
    else:
        get_side = mw.dockWidgetArea
        mem_location = remembered['windows']
    if widget.isFloating():
        side = None
        geom = widget.geometry()
        pos_info = (geom.x(), geom.y(), geom.width(), geom.height())
    else:
        side = get_side(widget)
        pos_info = None
    version = 1
    mem_location[tool_instance.tool_name] = (version, side, pos_info)
    ui.settings.tool_positions = remembered
    ui.settings.save()

def remove_keyboard_navigation(menu_label):
    if menu_label.count('&') == 1:
        return menu_label.replace('&', '')
    return menu_label.replace('&&', '&')

from PyQt5.QtWidgets import QDialog
class SelSeqDialog(QDialog):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.setWindowTitle("Select Sequence")
        self.setSizeGripEnabled(True)
        from PyQt5.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QLineEdit, QHBoxLayout, QLabel
        layout = QVBoxLayout()
        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("Sequence:"))
        self.edit = QLineEdit()
        self.edit.textChanged.connect(self._update_button_states)
        edit_layout.addWidget(self.edit)
        layout.addLayout(edit_layout)

        self.bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Apply | qbbox.Help)
        self.bbox.accepted.connect(self.search)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        self.bbox.button(qbbox.Apply).clicked.connect(self.search)
        from chimerax.core.commands import run
        self.bbox.helpRequested.connect(lambda run=run, ses=session: run(ses, "help help:user/findseq.html"))
        self._update_button_states(self.edit.text())
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def search(self, *args):
        from chimerax.core.commands import run, quote_if_necessary
        run(self.session, "sel seq %s" % quote_if_necessary(self.edit.text().strip()))

    def _update_button_states(self, text):
        enable = bool(text.strip())
        for button in [self.bbox.button(x) for x in [self.bbox.Ok, self.bbox.Apply]]:
            button.setEnabled(enable)

class SelZoneDialog(QDialog):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.setWindowTitle("Select Zone")
        self.setSizeGripEnabled(True)
        from PyQt5.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QLineEdit, QHBoxLayout, QLabel, \
            QCheckBox, QDoubleSpinBox
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select atoms/bonds that meet all chosen distance criteria:"))
        less_layout = QHBoxLayout()
        self.less_checkbox = QCheckBox("<")
        self.less_checkbox.setChecked(True)
        self.less_checkbox.stateChanged.connect(self._update_button_states)
        less_layout.addWidget(self.less_checkbox)
        self.less_spinbox = QDoubleSpinBox()
        self.less_spinbox.setValue(5.0)
        self.less_spinbox.setDecimals(3)
        self.less_spinbox.setSuffix("\N{ANGSTROM SIGN}")
        self.less_spinbox.setMaximum(9999.999)
        less_layout.addWidget(self.less_spinbox)
        less_layout.addWidget(QLabel("from the currently selected atoms"))
        layout.addLayout(less_layout)
        more_layout = QHBoxLayout()
        self.more_checkbox = QCheckBox(">")
        self.more_checkbox.stateChanged.connect(self._update_button_states)
        more_layout.addWidget(self.more_checkbox)
        self.more_spinbox = QDoubleSpinBox()
        self.more_spinbox.setValue(5.0)
        self.more_spinbox.setDecimals(3)
        self.more_spinbox.setSuffix("\N{ANGSTROM SIGN}")
        self.more_spinbox.setMaximum(9999.999)
        more_layout.addWidget(self.more_spinbox)
        more_layout.addWidget(QLabel("from the currently selected atoms"))
        layout.addLayout(more_layout)
        res_layout = QHBoxLayout()
        self.res_checkbox = QCheckBox("Apply criteria to whole residues")
        res_layout.addWidget(self.res_checkbox)
        layout.addLayout(res_layout)

        self.bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        self.bbox.accepted.connect(self.zone)
        self.bbox.button(qbbox.Apply).clicked.connect(self.zone)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        from chimerax.core.commands import run
        self.bbox.helpRequested.connect(lambda run=run, ses=session:
            run(ses, "help help:user/menu.html#selectzone"))
        self._update_button_states()
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def zone(self, *args):
        cmd = "select "
        char = ':' if self.res_checkbox.isChecked() else '@'
        if self.less_checkbox.isChecked():
            cmd += "sel %s< %g" % (char, self.less_spinbox.value())
            if self.more_checkbox.isChecked():
                cmd += ' & '
        if self.more_checkbox.isChecked():
            cmd += "sel %s> %g" % (char, self.more_spinbox.value())
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _update_button_states(self, *args):
        self.bbox.button(self.bbox.Ok).setEnabled(
            self.less_checkbox.isChecked() or self.more_checkbox.isChecked())

prepositions = set(["a", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to"])
def menu_capitalize(text):
    capped_words = []
    for word in text.split():
        if word[0] == '(':
            capped_words.append('(' + menu_capitalize(word[1:]))
        else:
            if word.lower() != word or (capped_words and word in prepositions):
                capped_words.append(word)
            else:
                capped_word = ""
                for frag in [x for part in word.split('/') for x in part.split('-')]:
                    capped_word += frag.capitalize()
                    if len(capped_word) < len(word):
                        capped_word += word[len(capped_word)]
                capped_words.append(capped_word)
    return " ".join(capped_words)

