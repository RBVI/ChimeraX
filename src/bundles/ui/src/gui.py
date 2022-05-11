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
    initialize_desktop_opengl()
    initialize_shared_opengl_contexts()
#    initialize_pyqt5_compatibility()

def initialize_qt_plugins_location():
    # remove the build tree plugin path, and add install tree plugin path
    import sys
    mac = (sys.platform == 'darwin')
    if mac:
        # The "plugins" directory can be in one of two places on Mac:
        # - if we built Qt and Qt from source:
        #      Contents/lib/plugins
        # - if we used a wheel built using standard Qt:
        #      Contents/lib/python3.5/site-packages/Qt/Qt/plugins
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
            plugins = os.path.join(dn(dn(dn(dn(__file__)))), "Qt/Qt/plugins")
        if os.path.exists(plugins):
            from Qt.QtCore import QCoreApplication
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
        from Qt.QtCore import QCoreApplication, Qt
        if not hasattr(Qt, 'AA_EnableHighDpiScaling'):
            return  # Qt6 does not have this setting
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

def initialize_desktop_opengl():
    # Need full OpenGL support
    from Qt import using_qt5
    if using_qt5:
        from Qt.QtCore import QCoreApplication, Qt
        QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)

def initialize_shared_opengl_contexts():
    # Mono and stereo opengl contexts need to share vertex buffers
    from Qt.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

def initialize_pyqt5_compatibility():
    '''
    Attempt to allow PyQt5 code to run using Qt.  This can work for
    simple uses of Qt but for more complex tools there is often some small
    differences between PyQt5 and Qt that cause errors.
    I think this compatibility code should not be used but I am leaving
    it in if we decide to temporarily use it during transition to Qt.
    '''

    # Matplotlib is looks for whether PyQt5.QtCore or Qt.QtCore
    # is present to choose backend. So have it decide before setting up
    # PyQt5 otherwise matplotlib will be broken looking for PyQt5 sip.
    from Qt import QtCore
    import matplotlib.backends.qt_compat

    # Add PyQt5 module which is Qt
    import Qt, sys
    sys.modules['PyQt5'] = Qt

    # Submodules also need to be added otherwise app does not initialize
    # apparently because Qt.QtWidgets and PyQt5.QtWidgets become two
    # different instantiations of the Qt.QtWidgets module.
    from Qt import QtCore, QtWidgets, Qt, QtGui
    sys.modules['PyQt5.QtCore'] = QtCore
    sys.modules['PyQt5.QtWidgets'] = QtWidgets
    sys.modules['PyQt5.Qt'] = Qt

    # SEQCROW imports things from wrong location, but works in PyQt5
    Qt.QIcon = QtGui.QIcon
    Qt.QStyle = QtWidgets.QStyle
    Qt.QClipboard = QtGui.QClipBoard

    # Added PyQt5 pyqtSignal which has same API as Qt Signal class.
    QtCore.pyqtSignal = QtCore.Signal

from Qt.QtWidgets import QApplication
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

        self._mouse_modes = None

        # for whatever reason, QtWebEngineWidgets has to be imported before a
        # QtCoreApplication is created...
        from Qt import qt_have_web_engine
        if qt_have_web_engine():
            import Qt.QtWebEngineWidgets

        from chimerax import app_dirs as ad
        QApplication.__init__(self, [ad.appname])

        from Qt import using_qt5
        if using_qt5:
            # Improve toolbar icon quality on retina displays
            from Qt.QtCore import Qt
            self.setAttribute(Qt.AA_UseHighDpiPixmaps)

        self.redirect_qt_messages()

        self._keystroke_sinks = []
        self._key_callbacks = {}	# Maps Qt key number to callback func(session, key_num).
        self._files_to_open = []

        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('ready')
        self.triggers.add_trigger('tool window show')
        self.triggers.add_trigger('tool window hide')

    @property
    def mouse_modes(self):
        # delay creation of mouse modes to allow mouse_modes bundle time to run
        # its custom init, which initializes its settings
        if self._mouse_modes is None:
            from chimerax.mouse_modes import MouseModes
            self._mouse_modes = MouseModes(self.session)
        return self._mouse_modes

    def redirect_qt_messages(self):

        # redirect Qt log messages to our logger
        from chimerax.core.logger import Log
        from Qt import using_qt5
        from Qt.QtCore import QtMsgType
        qt_to_cx_log_level_map = {
            QtMsgType.QtDebugMsg: Log.LEVEL_INFO,
            QtMsgType.QtInfoMsg: (Log.LEVEL_INFO if using_qt5 else None),
            QtMsgType.QtWarningMsg: Log.LEVEL_WARNING,
            QtMsgType.QtCriticalMsg: Log.LEVEL_ERROR,
            QtMsgType.QtFatalMsg: Log.LEVEL_BUG,
        }
        from Qt.QtCore import qInstallMessageHandler
        def cx_qt_msg_handler(msg_type, msg_log_context, msg_string):
            from Qt import using_qt6
            if (using_qt6 and
                (msg_string.startswith('delivering touch release to same window') or
                 msg_string.startswith('skipping QEventPoint'))):
                return	# Supress Qt 6.2 warnings
            if msg_type == QtMsgType.QtFatalMsg:
                import sys
                sys.__stderr__.write('Qt fatal error: %s\n' % msg_string)
            log_level = qt_to_cx_log_level_map[msg_type]
            if log_level is None:
                return
            if msg_string.strip().endswith(" null"):
                # downgrade Javascript errors
                log_level = Log.LEVEL_INFO
            self.session.logger.method_map[log_level](msg_string)
        qInstallMessageHandler(cx_qt_msg_handler)

    def window_image(self):
        '''
        Tests on macOS 10.14.5 show that QWidget.grab() gives a correct QPixmap
        even for undisplayed or iconified windows, except not for html widgets
        (e.g. Log, or file history) which come out blank.  Hidden windows also
        may render an image at the wrong size with a new layout (e.g. Model Panel).
        '''
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
            defunct_toolbars = set(["Density Map Toolbar", "Graphics Toolbar",
                "Molecule Display Toolbar", "Mouse Modes for Right Button"])
            final_autostart = [tool_name for tool_name in
                self.settings.autostart if tool_name not in defunct_toolbars]
            if final_autostart != self.settings.autostart:
                if "Toolbar" not in final_autostart:
                    final_autostart.append("Toolbar")
                self.settings.autostart = final_autostart
            self.session.tools.start_tools(final_autostart)

        self.triggers.activate_trigger('ready', None)

    def event(self, event):
        from Qt.QtCore import QEvent
        if event.type() == QEvent.Type.FileOpen:
            from chimerax.core.toolshed import get_toolshed
            if not hasattr(self, '_bad_drop_events'):
                # script running window_size() (which call processEvents)
                # can cause this to get called before open_pending_files()
                self._seen_bad_drops = getattr(self, '_seen_bad_drops', []) + [event.file()]
            elif event.file() in self._bad_drop_events:
                self._bad_drop_events.remove(event.file())
            elif get_toolshed() is None:
                # Drop event might have started ChimeraX and it is not yet ready to open a file.
                # So remember file and startup script will open it when ready.
                self._files_to_open.append(event.file())
            else:
                _open_dropped_file(self.session, event.file())
            return True
        return QApplication.event(self, event)

    def open_pending_files(self, ignore_files = ()):
        # Note about ignore_files:  macOS 10.12 generates QFileOpenEvent for arguments specified
        # on the command-line, but our code also opens those files, so ignore files we already processed.
        self._bad_drop_events = set(ignore_files)
        for bad_drop in getattr(self, '_seen_bad_drops', []):
            self._bad_drop_events.discard(bad_drop)
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
        self.exec()
        self.session.logger.clear()

    def forward_keystroke(self, event):
        """forward keystroke from graphics window to most recent
           caller of 'register_for_keystrokes'

           up/down arrow keystrokes are not forwarded and instead
           promote/demote the graphics window selection
        """
        from Qt.QtCore import Qt
        k = event.key()
        if self.key_intercepted(k):
            return
        elif k == Qt.Key.Key_Up:
            if not self.session.selection.empty():
                from chimerax.core.commands import run
                run(self.session, 'select up')
                return
            # Up arrow on an empty selection was probably intended for the command history...
        elif k == Qt.Key.Key_Down:
            from chimerax.core.commands import run
            run(self.session, 'select down')
            return
        if self._keystroke_sinks:
            self._keystroke_sinks[-1].forwarded_keystroke(event)
            # accepting the event prevents both the main Ui and tools from forwarding the same keystrokes
            event.setAccepted(True)

    def intercept_key(self, qt_key_number, callback):
        self._key_callbacks[qt_key_number] = callback

    def key_intercepted(self, key_num):
        if key_num in self._key_callbacks:
            f = self._key_callbacks[key_num]
            f(self.session, key_num)
            return True
        return False

    def register_for_keystrokes(self, sink):
        """'sink' is interested in receiving keystrokes from the main
           graphics window.  That object's 'forwarded_keystroke'
           method will be called with the keystroke event as the argument.
        """
        self._keystroke_sinks.append(sink)

    def shift_key_down(self):
        modifiers = self.keyboardModifiers()
        from Qt.QtCore import Qt
        return modifiers & Qt.ShiftModifier

    def remove_tool(self, tool_instance):
        self.main_window.remove_tool(tool_instance)
        # get garbage collection to break callback loops in deleted tools
        # that might be triggered by live tools (e.g. settings changes)
        # particularly since WA_DeleteOnClose can nuke the Qt side
        def _cleanup(s=self):
            import gc
            gc.collect()
            delattr(s, '_kludge_cleanup_timer')
        self._kludge_cleanup_timer = self.timer(100, _cleanup)

    def set_tool_shown(self, tool_instance, shown):
        self.main_window.set_tool_shown(tool_instance, shown)

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
        from Qt.QtCore import QEvent
        class ThreadSafeGuiFuncEvent(QEvent):
#            EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
            EVENT_TYPE = QEvent.Type(QEvent.Type.User)
            def __init__(self, func, args, kw):
                QEvent.__init__(self, self.EVENT_TYPE)
                self.func_info = (func, args, kw)
        self.postEvent(self.main_window, ThreadSafeGuiFuncEvent(func, args, kw))

    def timer(self, millisec, callback, *args, **kw):
        # Caller needs to hold a reference to the returned timer so it doesn't get
        # garbage collected immediately
        from Qt.QtCore import QTimer
        t = QTimer()
        def cb(*, callback=callback, args=args, kw=kw):
            callback(*args, **kw)
        t.timeout.connect(cb)
        t.setSingleShot(True)
        t.start(int(millisec))
        return t

    def cancel_timer(self, timer):
        timer.stop()

    def update_undo(self, undo_manager):
        self.main_window.update_undo(undo_manager)

from Qt.QtWidgets import QMainWindow, QStackedWidget, QLabel, QToolButton, QWidget
class MainWindow(QMainWindow, PlainTextLog):

    def __init__(self, ui, session):
        self.session = session
        QMainWindow.__init__(self)
        self.setWindowTitle("ChimeraX")

        sizing_scheme, size_data = session.ui.settings.initial_window_size
        if sizing_scheme == "last used" and session.ui.settings.last_window_size is None:
            sizing_scheme = "proportional"
            size_data = (0.67, 0.67)
        if sizing_scheme == "last used":
            width, height = session.ui.settings.last_window_size
        elif sizing_scheme == "proportional":
            wf, hf = size_data
            main_screen_geom = ui.primaryScreen().availableGeometry()
            width, height = main_screen_geom.width()*wf, main_screen_geom.height()*hf
        elif sizing_scheme == "fixed":
            width, height = size_data
        if sizing_scheme not in ["full screen", "maximized"]:
            self.resize(width, height)
        # going into full screen / maximized causes events to happen, so delay until we're more
        # fully initialized

        from Qt.QtCore import QSize
        class GraphicsArea(QStackedWidget):
            def sizeHint(self):
                return QSize(800, 800)

        self._stack = GraphicsArea(self)
        from .graphics import GraphicsWindow
        stereo = getattr(ui, 'stereo', False)
        if stereo:
            from chimerax.graphics import StereoCamera
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
        from Qt import qt_have_web_engine
        if qt_have_web_engine():
            from .file_history import FileHistory
            fh = FileHistory(session, self.rapid_access, bg_color=ra_bg_color, thumbnail_size=(128,128),
                             filename_size=15, no_hist_text="\n".join(new_user_text))
        self._stack.addWidget(self.rapid_access)
        self._stack.setCurrentWidget(g.widget)
        self.setCentralWidget(self._stack)
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        session.triggers.add_handler(ADD_MODELS, self._check_rapid_access)
        session.triggers.add_handler(REMOVE_MODELS, self._check_rapid_access)

        self._hide_tools = False
        self._hide_floating_tools = False
        self._pref_dialog_state = False
        self.tool_instance_to_windows = {}
        self._fill_tb_context_menu_cbs = {}
        self._select_seq_dialog = self._select_zone_dialog = self._define_selector_dialog = None
        self._select_contacts_dialog = self._set_label_height_dialog = None
        self._presets_menu_needs_update = True
        session.presets.triggers.add_handler("presets changed",
            lambda *args, s=self: setattr(s, '_presets_menu_needs_update', True))
        self._is_quitting = False
        self._color_dialog = None

        self._build_status()
        self._populate_menus(session)

        # set icon for About dialog
        from chimerax import app_dirs as ad, app_data_dir
        import os.path
        icon_path = os.path.join(app_data_dir, "%s-icon512.png" % ad.appname)
        if os.path.exists(icon_path):
            from Qt.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))

        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('resized')

        session.logger.add_log(self)

        # Allow drag and drop of files onto app window.
        self.setAcceptDrops(True)

        self._activated_window_index = 0
        self.set_hot_keys()

        # full screen works very poorly on Windows as of 6/16/20 (see ticket #3409)
        # so withdrawn in favor of just "maximized" for now
        #if sizing_scheme == "full screen":
        #    self.showFullScreen()
        if sizing_scheme == "maximized" or sizing_scheme == "full screen":
            self.showMaximized()
        else:
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
            return True    # Already using requested mode

        from .graphics import GraphicsWindow
        try:
            g = GraphicsWindow(self._stack, self.session.ui, stereo, oc)
        except Exception:
            # Failed to create OpenGL context
            return False

        # Only destroy old graphics window after new one is made so clean-up
        # of old OpenGL context can be done.
        gw.destroy()

        self.graphics_window = g
        g.keyPressEvent = self.session.ui.forward_keystroke
        self._stack.addWidget(g.widget)
        self._stack.setCurrentWidget(g.widget)

        return True

    def keyPressEvent(self, event):
        self.session.ui.forward_keystroke(event)

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
        from Qt.QtWidgets import QToolBar
        for arg in tb_args:
            if isinstance(arg, QToolBar):
                tb = arg
                break
        else:
            tb = retval
        tb.visibilityChanged.connect(lambda vis, *, tb=tb: self._set_tool_checkbuttons(tb, vis))
        tb.contextMenuEvent = lambda e, *, self=self, tb=tb: self.show_tb_context_menu(tb, e)
        self._fill_tb_context_menu_cbs[tb] = (tool, fill_context_menu_cb)
        settings =  self.session.ui.settings
        if tool.tool_name in settings.tool_positions['toolbars']:
            pos_info = settings.tool_positions['toolbars'][tool.tool_name]
            if type(pos_info) == dict:
                placement = pos_info['placement']
                geom_info = pos_info['geom_info']
                tab_info = pos_info['tab_info']
            else:
                version, *info = settings.tool_positions['toolbars'][tool.tool_name]
                if version == 1:
                    placement, geom_info = info
                    tab_info = []
                else:
                    placement, geom_info, tab_info = info
            if placement is None:
                self.session.logger.info("Cannot restore toolbar as floating")
                #from Qt.QtCore import Qt
                #QMainWindow.addToolBar(self, Qt.NoToolBarArea, tb)
                #from Qt.QtCore import QRect
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
        from Qt.QtCore import Qt
        return bool(self.windowState() & (Qt.WindowMaximized | Qt.WindowFullScreen))

    def changeEvent(self, event):
        t = event.type()
        from Qt.QtCore import QEvent
        if t == QEvent.Type.WindowStateChange:
            self.hide_floating_tools = self.isMinimized()

    def closeEvent(self, event):
        # the MainWindow close button has been clicked
        self._is_quitting = True
        event.accept()
        sbar = self._status_bar
        if sbar is not None:
            sbar.destroy()
            self._status_bar = None
        self.session.ui.quit()

    def close_request(self, tool_window, close_event = None):
        # closing a tool window has been requested
        if self._is_quitting:
            if close_event:
                close_event.accept()
            return
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        close_destroys = tool_window.close_destroys
        if is_main_window and close_destroys:
            if close_event:
                close_event.accept()
            tool_instance.delete()
            return
        if close_destroys:
            if close_event:
                close_event.accept()
            # _destroy will remove window from all_windows indirectly
            # via tw._destroy -> toolkit.destroy -> mw->_tool_window_destroyed
            tool_window._destroy()
        else:
            if close_event:
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
    def hide_floating_tools(self):
        return self._hide_floating_tools

    @hide_floating_tools.setter
    def hide_floating_tools(self, ht):
        if ht == self._hide_floating_tools:
            return

        # need to set _hide_floating_tools attr first, since it will be checked in
        # subsequent calls
        self._hide_floating_tools = ht
        if ht == True:
            self._hide_floating_tools_shown_states = states = {}
            settings_dw = self.settings_ui_widget
            self._pref_dialog_state = not settings_dw.isHidden() and settings_dw.isFloating()
            if self._pref_dialog_state:
                settings_dw.hide()
            for tool_windows in self.tool_instance_to_windows.values():
                for tw in tool_windows:
                    if not tw.floating:
                        continue
                    state = tw.shown
                    states[tw] = state
                    if state:
                        tw._mw_set_shown(False)
        else:
            for tw, state in self._hide_floating_tools_shown_states.items():
                if state:
                    tw._mw_set_shown(True)
            self._hide_floating_tools_shown_states.clear()
            if self._pref_dialog_state:
                self.settings_ui_widget.show()

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
                    if tw.hides_title_bar:
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
                    tw._mw_set_shown(True)
            self._hide_tools_shown_states.clear()
            if self._pref_dialog_state:
                self.settings_ui_widget.show()

        self._global_hide_button.setIcon(icon)

    def log(self, *args, **kw):
        return False

    def remove_tool(self, tool_instance):
        tool_windows = self.tool_instance_to_windows.get(tool_instance, None)
        if tool_windows:
            for tw in tool_windows[:]:
                tw._mw_set_shown(False)
                tw._destroy()
            del self.tool_instance_to_windows[tool_instance]

    def set_hot_keys(self):
        from Qt.QtGui import QKeySequence, QShortcut
        from Qt.QtCore import Qt
        sc = QShortcut(QKeySequence("Shift+Esc"), self, context=Qt.ShortcutContext.ApplicationShortcut)
        from chimerax.core.commands import run
        sc.activated.connect(lambda run=run, ses=self.session: run(ses, "ui hideFloating toggle"))
        sc = QShortcut(QKeySequence.StandardKey.NextChild, self, context=Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(lambda mw=self: mw._activate_next_window(1))
        sc = QShortcut(QKeySequence.StandardKey.PreviousChild, self, context=Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(lambda mw=self: mw._activate_next_window(-1))

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
        if show:
            icon = self._ra_shown_icon
            self._stack.setCurrentWidget(self.rapid_access)
        else:
            icon = self._ra_hidden_icon
            self._stack.setCurrentWidget(self.graphics_window.widget)

        but = self._rapid_access_button
        but.setChecked(show)
        but.defaultAction().setChecked(show)
        but.setIcon(icon)
        if self.session.in_script:
            # get the graphics to actually hide/show
            self.session.ui.processEvents()

    def _check_rapid_access(self, *args):
        self.rapid_access_shown = len(self.session.models) == 0

    def showEvent(self, event):
        QMainWindow.showEvent(self, event)
        if not hasattr(self, '_already_shown'):
            self._already_shown = True
            # Work around startup crash on Windows that appears to happen when
            # rapid access is shown too early, a likely Qt bug.  ChimeraX ticket #4698.
            self.rapid_access_shown = True

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        size = event.size()
        wh = (size.width(), size.height())
        self.session.ui.settings.last_window_size = wh
        self.triggers.activate_trigger('resized', wh)

    def moveEvent(self, event):
        # If window is moved to another screen with different pixel scale
        # then update the framebuffer size.  Bug #2251.
        gw = self.graphics_window
        r = gw.view.render
        if not hasattr(self, '_last_pixel_scale') or r.pixel_scale() != self._last_pixel_scale:
            self._last_pixel_scale = r.pixel_scale()
            r.set_default_framebuffer_size(gw.width(), gw.height())
            gw.view.redraw_needed = True

    def show_define_selector_dialog(self, *args):
        if self._define_selector_dialog is None:
            self._define_selector_dialog = DefineSelectorDialog(self.session)
        self._define_selector_dialog.show()
        self._define_selector_dialog.raise_()

    def show_select_contacts_dialog(self, *args):
        if self._select_contacts_dialog is None:
            self._select_contacts_dialog = SelContactsDialog(self.session)
        self._select_contacts_dialog.show()
        self._select_contacts_dialog.raise_()

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

    def show_set_label_height_dialog(self, *args):
        if self._set_label_height_dialog is None:
            self._set_label_height_dialog = LabelHeightDialog(self.session)
        self._set_label_height_dialog.show()
        self._set_label_height_dialog.raise_()

    def show_tb_context_menu(self, tb, event):
        tool, fill_cb = self._fill_tb_context_menu_cbs[tb]
        _show_context_menu(event, tool, None, fill_cb, True, tb)

    def status(self, msg, color, secondary):
        sbar = self._status_bar
        if sbar:
            sbar.status(msg, color, secondary)

    def show_statusbar(self, show):
        self._status_bar.show(show)

    def show_menubar(self, show):
        self.menuBar().setVisible(show)

    def show_fullscreen(self, show):
        if show:
            self.showFullScreen()
        else:
            self.showNormal()

    def _about(self, arg):
        from Qt.QtWebEngineWidgets import QWebEngineView
        from Qt.QtCore import QUrl
        import os.path
        from chimerax.core import buildinfo
        fn = os.path.join(os.path.dirname(__file__), "about.html")
        with open(fn) as f:
            content = f.read()
        content = content.replace("VERSION", buildinfo.version)
        content = content.replace("DATE", buildinfo.date.split()[0])
        self._about_dialog = QWebEngineView()
        self._about_dialog.setHtml(content, QUrl.fromLocalFile(
            os.path.dirname(os.path.realpath(__file__)) + os.sep))
        self._about_dialog.show()

    def _about_to_manage(self, tool_window, as_floating):
        if self.hide_tools and not as_floating:
            self.hide_tools = False
    _float_changed = _about_to_manage

    def _activate_next_window(self, increment):
        windows = [self]
        if not self.settings_ui_widget.isHidden():
            windows += [self.settings_ui_widget]
        for tws in self.tool_instance_to_windows.values():
            for tw in tws:
                if tw.floating and tw.shown:
                    windows.append(tw._dock_widget)
        self._activated_window_index = (self._activated_window_index + increment) % len(windows)
        from Qt.QtCore import Qt
        windows[self._activated_window_index].activateWindow()
        windows[self._activated_window_index].raise_()

    def _build_status(self):
        from .statusbar import _StatusBar
        self._status_bar = sbar = _StatusBar(self.session)
        sbar.status('Welcome to ChimeraX', 'blue')
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
        from Qt.QtGui import QAction
        ghb_action = QAction(ghb)
        rab_action = QAction(rab)
        ghb_action.setCheckable(True)
        rab_action.setCheckable(True)
        rab_action.setChecked(True)
        from chimerax.core.commands import run
        ghb_action.toggled.connect(lambda checked, run=run, ses=self.session: run(ses, 'ui windowfill toggle'))
        rab_action.toggled.connect(lambda checked: setattr(self, 'rapid_access_shown', checked))
        ghb_action.setIcon(self._expand_icon)
        rab_action.setIcon(self._ra_shown_icon)
        ghb.setDefaultAction(ghb_action)
        rab.setDefaultAction(rab_action)
        sb.addPermanentWidget(ghb)
        sb.addPermanentWidget(rab)
        self.setStatusBar(sb)

    def _dockability_change(self, tool_name, dockable):
        """Call back from 'ui dockable' command"""
        for ti, tool_windows in self.tool_instance_to_windows.items():
            if ti.tool_name == tool_name:
                for win in tool_windows:
                    win._mw_set_dockable(dockable)

    @property
    def settings_ui_widget(self):
        # this is a property in order to delay actual creation of the window as long as possible,
        # so that bundles can register settings options and the window will start large
        # enough to accomodate the registered options
        if self._settings_ui_widget is None:
            from Qt.QtWidgets import QDockWidget, QWidget
            dw = QDockWidget("ChimeraX Settings", self)
            dw.closeEvent = lambda e, *, dw=dw: dw.hide()
            from .core_settings_ui import CoreSettingsPanel
            container = QWidget()
            self._core_settings_panel = csp = CoreSettingsPanel(self.session, container)
            for cat, opt in self._accumulated_settings_options:
                csp.options_widget.add_option(cat, opt)
            self._accumulated_settings_options = []
            dw.setWidget(container)
            from Qt.QtCore import Qt
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)
            dw.setFloating(True)
            dw.hide()
            self._settings_ui_widget = dw
        return self._settings_ui_widget

    def add_settings_option(self, category, option):
        """For bundles that need/want to present their settings with the core ChimeraX settings
           rather than present their own settings UI"""
        if self._settings_ui_widget is None:
            self._accumulated_settings_options.append((category, option))
        else:
            self._core_settings_panel.options_widget.add_option(category, option)

    def show_settings(self, category=None):
        self.settings_ui_widget.show()
        if category is not None:
            self._core_settings_panel.show_category(category)

    def _new_tool_window(self, tw):
        self.tool_instance_to_windows.setdefault(tw.tool_instance,[]).append(tw)

    def _populate_menus(self, session):
        from Qt.QtGui import QKeySequence, QAction
        from Qt.QtCore import Qt
        from chimerax.core.commands import run

        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        file_menu.setObjectName("File")
        cd_action = QAction("Set &Working Folder...", self)
        cd_action.setToolTip("Set default folder for commands/tools to use when opening/saving files")
        cd_action.triggered.connect(lambda *, run=run, sess=session: run(sess, "cd browse"))
        file_menu.addAction(cd_action)
        close_action = QAction("&Close Session", self)
        close_action.setToolTip("Close session")
        close_action.triggered.connect(lambda *, s=self, sess=session: s.file_close_cb(sess))
        file_menu.addAction(close_action)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.setToolTip("Quit ChimeraX")
        quit_action.triggered.connect(lambda *, s=self, sess=session: s.file_quit_cb(sess))
        file_menu.addAction(quit_action)
        file_menu.setToolTipsVisible(True)

        edit_menu = mb.addMenu("&Edit")
        edit_menu.setObjectName("Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setEnabled(False)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(lambda *, s=self, sess=session: s.edit_undo_cb(sess))
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setEnabled(False)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(lambda *, s=self, sess=session: s.edit_redo_cb(sess))
        edit_menu.addAction(self.redo_action)

        select_menu = mb.addMenu("&Select")
        select_menu.setObjectName("Select")
        self._populate_select_menu(select_menu)

        actions_menu = mb.addMenu("&Actions")
        actions_menu.setObjectName("Actions")
        self._populate_actions_menu(actions_menu)

        self.tools_menu = mb.addMenu("&Tools")
        self.tools_menu.setToolTipsVisible(True)
        self.update_tools_menu(session)

        self._settings_ui_widget = None
        self._accumulated_settings_options = []
        settings = session.ui.settings
        self.add_settings_option("Window", InitWindowSizeOption("Initial overall window size",
            settings.initial_window_size, None, attr_name="initial_window_size", settings=settings,
            session=self.session, balloon="Initial overall size of ChimeraX window"))
        self.add_settings_option("Window", ToolSideOption("Default tool side",
            settings.default_tool_window_side, None, attr_name="default_tool_window_side", settings=settings,
            balloon="Which side of main window that new tool windows appear on by default"))
        from .options import BooleanOption
        self.add_settings_option("Window", BooleanOption("Tool windows start undocked",
            settings.auto_float_tools, None, attr_name="auto_float_tools", settings=settings,
            balloon="Tools (other than ones launched at ChimeraX startup) start undocked and undockable.\n"
            'A tool can be made dockable through its "Dockable Tool" context menu entry.'))

        self.favorites_menu = mb.addMenu("Fa&vorites")
        self.favorites_menu.setToolTipsVisible(True)
        self.update_favorites_menu(session)

        self.presets_menu = mb.addMenu("Presets")
        self.presets_menu.setToolTipsVisible(True)
        self.presets_menu.aboutToShow.connect(lambda *, ses=session: self._populate_presets_menu(ses))
        self._populate_presets_menu(session)

        help_menu = mb.addMenu("&Help")
        help_menu.setObjectName("Help")
        help_menu.setToolTipsVisible(True)
        for entry, location, tooltip in (
                ('User Guide', 'user', 'Tutorials and user documentation'),
                ('Quick Start Guide', 'quickstart', 'Interactive ChimeraX basics'),
                ('Tutorials', 'https://www.rbvi.ucsf.edu/chimerax/tutorials.html', 'Tutorials'),
                ('Programming Manual', 'devel', 'How to develop ChimeraX tools'),
                ('Documentation Index', 'index.html', 'Access all documentarion'),
                ('Citing ChimeraX', 'credits.html', 'How to cite ChimeraX in publications'),
                ('Contact Us', 'contact.html', 'Report problems/issues; ask questions')):
            help_action = QAction(entry, self)
            help_action.setToolTip(tooltip)
            cmd = ('open %s' % location) if location.startswith('http') else ('help help:%s' % location)
            def cb(*, ses=session, cmd=cmd):
                run(ses, cmd)
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

        from Qt.QtGui import QAction
        help_action = QAction("Add A Preset...", self)
        from chimerax.core.commands import run
        help_action.triggered.connect(lambda *, run=run, ses=session: run(ses,
            "help help:user/preferences.html#startup"))
        if not preset_info:
            self.presets_menu.addAction(help_action)
            return

        if len(preset_info) == 1:
            self._uncategorized_preset_menu(session, preset_info)
        elif len(preset_info) + sum([len(v) for v in preset_info.values()]) < 40:
            self._inline_categorized_preset_menu(session, preset_info)
        else:
            self._rollover_categorized_preset_menu(session, preset_info)
        self.presets_menu.addSeparator()
        self.presets_menu.addAction(help_action)

    def _uncategorized_preset_menu(self, session, preset_info):
        for category, preset_names in preset_info.items():
            self._add_preset_entries(session, self.presets_menu, preset_names)

    def _inline_categorized_preset_menu(self, session, preset_info):
        from Qt.QtGui import QAction
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
            cat_menu.setTearOffEnabled(True)
            self._add_preset_entries(session, cat_menu, preset_info[cat], cat)

    def _order_preset_categories(self, categories):
        cats = list(categories)[:]
        cats.sort(key=lambda x: x.lower())
        return cats

    def _add_preset_entries(self, session, menu, preset_names, category=None):
        from Qt.QtGui import QAction
        from chimerax.core.commands import run, StringArg
        # the menu names may be instances of CustomSortString, so sort them
        # before applying menu_capitalize(); also 'preset_names' may be a keys view
        menu_names = list(preset_names)
        menu_names.sort(key=lambda x: x.lower())
        menu_names = [menu_capitalize(name) for name in menu_names]
        if category is None:
            cat_string = ""
        else:
            cat_string = StringArg.unparse(category.lower()) + " "
        for name in menu_names:
            action = QAction(name, menu)
            action.triggered.connect(lambda *, ses=session, name=name, cat=cat_string:
                run(ses, "preset %s%s" % (cat, StringArg.unparse(name.lower()))))
            menu.addAction(action)

    def _populate_actions_menu(self, actions_menu):
        from Qt.QtGui import QAction
        from chimerax.core.commands import run, sel_or_all
        #
        # Atoms/Bonds...
        #
        atoms_bonds_menu = actions_menu.addMenu("Atoms/Bonds")
        action = QAction("Show", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="show %s target ab": run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        action = QAction("Show Only", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="hide #* target %s; show %s target ab":
            run(ses, cmd % (precise_target(ses), sel_or_all(ses, ['atoms', 'bonds']))))
        action = QAction("Hide", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="hide %s target %s":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds']), precise_target(ses))))

        # Atom Style submenu...
        atom_style_menu = atoms_bonds_menu.addMenu("Atom Style")
        style_info = [("Stick", "stick"), ("Ball && Stick", "ball"), ("Sphere", "sphere")]
        for menu_entry, style_name in style_info:
            action = QAction(menu_entry, self)
            atom_style_menu.addAction(action)
            action.triggered.connect(lambda *, run=run, ses=self.session,
                cmd="style %%s %s" % style_name: run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        rings_menu = atom_style_menu.addMenu("Ring Fill")
        rings_info = [("Thick", "thick"), ("Thin", "thin"), ("None", "off")]
        for menu_entry, ring_style in rings_info:
            action = QAction(menu_entry, self)
            rings_menu.addAction(action)
            action.triggered.connect(lambda *, run=run, ses=self.session,
                cmd="style %%s ringFill %s" % ring_style:
                run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        # end Atom Style submenu

        atoms_bonds_menu.addSeparator()

        action = QAction("Show Sidechain/Base", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="show %s target ab":
            run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues", restriction="sidechain")))
        action = QAction("Backbone Only", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="hide %s target %s; cartoon hide %s; show %s target ab":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues",
            restriction="(protein|nucleic)"), precise_target(ses),
            sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues"),
            sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues", restriction="backbone"))))
        action = QAction("Chain Trace Only", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="hide %s target %s; cartoon hide %s; show %s target ab":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues",
                restriction="(protein|nucleic)"),
            precise_target(ses), sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues"),
            sel_or_all(ses, ['atoms', 'bonds'], sel="sel-residues",
                restriction="((protein&@ca)|(nucleic&@p))"))))

        # Nucleotide Style submenu...
        nuc_menu = atoms_bonds_menu.addMenu("Nucleotide Style")
        nuc_info = [("Ladder", "ladder"), ("Stubs", "stubs"), ("Slab Base, Ribose Tube", "tube/slab"),
            ("Slab Base, Ribose Atoms", "slab"), ("Atoms (Filled Rings)", "fill"),
            ("Atoms (No Ring Fill)", "atoms")]
        for menu_entry, nuc_style in nuc_info:
            action = QAction(menu_entry, self)
            nuc_menu.addAction(action)
            action.triggered.connect(lambda *, run=run, ses=self.session,
                cmd="nucleotides %%s %s" % nuc_style:
                run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        # end Nucleotide Style submenu

        atoms_bonds_menu.addSeparator()

        action = QAction("Delete", self)
        atoms_bonds_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="delete atoms %s; delete bonds %s":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds']), sel_or_all(ses, ['atoms', 'bonds']))))

        #
        # Cartoon...
        #
        cartoon_menu = actions_menu.addMenu("Cartoon")
        action = QAction("Show", self)
        cartoon_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="cartoon %s": run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        action = QAction("Hide", self)
        cartoon_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="cartoon hide %s": run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        cartoon_menu.addSeparator()
        action = QAction("Rounded Edges", self)
        cartoon_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="cartoon style %s xsection oval modeHelix default":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds']))))
        action = QAction("Squared Edges", self)
        cartoon_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="cartoon style %s xsection rectangle modeHelix default":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds']))))
        action = QAction("Piped Edges", self)
        cartoon_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="cartoon style %s xsection oval; cartoon style %s xsection barbell modeHelix default":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds'], restriction="coil"),
                sel_or_all(ses, ['atoms', 'bonds']))))
        action = QAction("Tube Helices", self)
        cartoon_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            cmd="cartoon style %s modeHelix tube sides 20":
            run(ses, cmd % (sel_or_all(ses, ['atoms', 'bonds']))))

        #
        # Surface...
        #
        surface_menu = actions_menu.addMenu("Surface")
        action = QAction("Show", self)
        surface_menu.addAction(action)
        action.triggered.connect(lambda *, run=self._run_surf_command, cmd="surface %s": run(cmd))
        action = QAction("Hide", self)
        surface_menu.addAction(action)
        action.triggered.connect(lambda *, run=self._run_surf_command, cmd="surface hide %s": run(cmd))
        surface_menu.addSeparator()
        for style in ["solid", "mesh", "dot"]:
            action = QAction(style.capitalize(), self)
            surface_menu.addAction(action)
            action.triggered.connect(lambda *, run=self._run_surf_command,
                cmd="surface style %%s %s" % style: run(cmd, whole_surf=True))
        surface_menu.addSeparator()
        transparency_menu = surface_menu.addMenu("Transparency")
        for percent in range(0, 101, 10):
            action = QAction("%d%%" % percent, self)
            transparency_menu.addAction(action)
            action.triggered.connect(lambda *, run=self._run_surf_command,
                cmd="transparency %%s %d" % percent: run(cmd))

        #
        # Color...
        #
        color_menu = actions_menu.addMenu("Color")
        from Qt.QtGui import QColor, QPixmap, QIcon
        for spaced_name in [ "red", "orange red", "orange", "yellow", "lime", "forest green", "cyan",
                "light sea green", "blue", "cornflower blue", "medium blue", "purple", "hot pink",
                "magenta", "white", "light gray", "gray", "dark gray", "dim gray", "black"]:
            svg_name = "".join(spaced_name.split())
            color = QColor(svg_name)
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            action = QAction(icon, spaced_name.title(), self)
            color_menu.addAction(action)
            action.triggered.connect(lambda *, run=self._run_surf_command,
                cmd="color %%s %s" % spaced_name: run(cmd))
        color_menu.addSeparator()
        for menu_text, cmd_arg in [("By Heteroatom", "byhet"), ("By Element", "byelement")]:
            action = QAction(menu_text, self)
            color_menu.addAction(action)
            action.triggered.connect(lambda *, run=run, ses=self.session,
                cmd="color %%s %s" % cmd_arg: run(ses, cmd % sel_or_all(ses, ['atoms', 'bonds'])))
        action = QAction("Custom...", self)
        color_menu.addAction(action)
        action.triggered.connect(self.color_by_editor)
        color_menu.addSeparator()
        action = QAction("All Options...", self)
        color_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session:
                run(ses, "ui tool show 'Color Actions'"))

        #
        # Label...
        #
        label_menu = actions_menu.addMenu("Label")
        label_atoms_menu = label_menu.addMenu("Atoms")
        main_atom_label_info = [("Name", None), ("Element", "element"), ("IDATM Type", "idatm_type")]
        for menu_entry, attr_name in main_atom_label_info:
            action = QAction(menu_entry, self)
            label_atoms_menu.addAction(action)
            text = " attr %s" % attr_name if attr_name else ""
            action.triggered.connect(lambda *, run=run, ses=self.session, cmd="label %%s atoms%s"
                % text: run(ses, cmd % sel_or_all(ses, ['atoms'], allow_empty_spec=False)))
        action = QAction("Custom Text", self)
        label_atoms_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            fetch_text=self._get_label_text_arg: run(ses, "label %s atoms text %s"
            % (sel_or_all(ses, ['atoms'], allow_empty_spec=False), fetch_text())))
        label_atoms_other_menu = label_atoms_menu.addMenu("Other")
        def fill_other_menu(menu, main_attrs, class_obj, sel_name, *, run=run, sel_or_all=sel_or_all):
            menu.clear()
            from chimerax.core.attributes import MANAGER_NAME
            other_info = [(attr_name.replace('_', ' ').title(), attr_name)
                for attr_name in [attr for attr in self.session.get_state_manager(MANAGER_NAME).attributes_returning(class_obj, (int, float, str), none_okay=True) if attr not in main_attrs]]
            other_info.sort()
            for menu_entry, attr_name in other_info:
                if attr_name.startswith("num_") and attr_name != "num_alt_locs":
                    # most "num" attributes uninteresting for labeling purposes
                    continue
                action = QAction(menu_entry, self)
                menu.addAction(action)
                text = " attr %s" % attr_name
                action.triggered.connect(lambda *, run=run, ses=self.session, attr_name=attr_name,
                    cmd="label %%s %s attr %s" % (sel_name, attr_name):
                    run(ses, cmd % sel_or_all(ses, [sel_name], allow_empty_spec=False)))
        from chimerax.atomic import Atom, Residue
        label_atoms_other_menu.aboutToShow.connect(lambda *, menu=label_atoms_other_menu, main_attrs=
            set([attr_name if attr_name else "name" for label, attr_name in main_atom_label_info]),
            class_obj=Atom, fill=fill_other_menu: fill(menu, main_attrs, class_obj, "atoms"))
        action = QAction("Off", self)
        label_atoms_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session:
            run(ses, "~label %s atoms" % sel_or_all(ses, ['atoms'], allow_empty_spec=False)))

        label_residues_menu = label_menu.addMenu("Residues")
        main_residue_label_info = [("Name", "name"), ("Specifier", "label_specifier"),
                ("Name and Number", '"{0.name} {0.number}{0.insertion_code}"'),
                ("Chain, Name, Number", '"/{0.chain_id} {0.name} {0.number}{0.insertion_code}"'),
                ("1-Letter Code", "label_one_letter_code"),
                ("Code and Number",
                    '"{0.label_one_letter_code} {0.number}{0.insertion_code}"'),
                ("Chain, Code, Number",
                    '"/{0.chain_id} {0.label_one_letter_code} {0.number}{0.insertion_code}"'),
                 ]
        for menu_entry, cmd_arg in main_residue_label_info:
            action = QAction(menu_entry, self)
            label_residues_menu.addAction(action)
            if cmd_arg:
                if '{' in cmd_arg:
                    text = " text %s" % cmd_arg
                else:
                    text = " attr %s" % cmd_arg
            else:
                text = ""
            action.triggered.connect(lambda *, run=run, ses=self.session, cmd="label %%s%s"
                % text: run(ses, cmd % sel_or_all(ses, ['residues'], allow_empty_spec=False)))
        action = QAction("Custom Text", self)
        label_residues_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session,
            fetch_text=self._get_label_text_arg: run(ses, "label %s text %s"
            % (sel_or_all(ses, ['atoms'], allow_empty_spec=False), fetch_text())))
        label_residues_other_menu = label_residues_menu.addMenu("Other")
        label_residues_other_menu.aboutToShow.connect(lambda *, menu=label_residues_other_menu, main_attrs=
            set([attr_name for label, attr_name in main_residue_label_info if attr_name.isalnum()]),
            class_obj=Residue, fill=fill_other_menu: fill(menu, main_attrs, class_obj, "residues"))
        action = QAction("Off", self)
        label_residues_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session:
            run(ses, "~label %s residues" % sel_or_all(ses, ['residues'], allow_empty_spec=False)))
        action = QAction("Set Label Height", self)
        label_menu.addAction(action)
        action.triggered.connect(self.show_set_label_height_dialog)

        # misc...
        action = QAction("View", self)
        actions_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session:
            run(ses, "view" + ("" if ses.selection.empty() else " sel")))

        action = QAction("Set Pivot", self)
        actions_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session:
            run(ses, "cofr " + ("frontCenter" if ses.selection.empty() else "sel")))

        actions_menu.addSeparator()

        action = QAction("Inspect", self)
        actions_menu.addAction(action)
        action.triggered.connect(lambda *, run=run, ses=self.session:
            run(ses, "ui tool show 'Selection Inspector'"))

    def color_by_editor(self, *args):
        if not self._color_dialog:
            from Qt.QtWidgets import QColorDialog
            self._color_dialog = cd = QColorDialog(self)
            cd.setOption(cd.NoButtons, True)
            cd.setOption(cd.ShowAlphaChannel, True)
            from chimerax.core.commands import run, sel_or_all
            cd.currentColorChanged.connect(lambda clr, *, ses=self.session:
                run(ses, "color %s %s" % (sel_or_all(ses, ['atoms', 'bonds']),
                clr.name() + clr.name(clr.HexArgb)[1:3])))
            cd.destroyed.connect(lambda *, s=self: setattr(s, '_color_dialog', None))
        else:
            cd = self._color_dialog
            # On Mac, Qt doesn't realize when the color dialog has been hidden by the red 'X' button, so
            # "hide" it now so that Qt doesn't believe that the later show() is a no op.  Whereas on Windows
            # doing a hide followed by a show causes the dialog to jump back to it's original screen
            # position, so do the hide _only_ on Mac.
            import sys
            if sys.platform == "darwin":
                cd.hide()
        cd.show()

    def _run_surf_command(self, cmd, *, whole_surf=False):
        from chimerax.core.commands import run, sel_or_all, NoneSelectedError
        from chimerax.core.models import Surface
        if whole_surf:
            selector = sel_or_all(self.session, Surface, relevant_types=Surface)
        else:
            try:
                selector = sel_or_all(self.session, ['atoms', 'bonds'])
            except NoneSelectedError:
                try:
                    selector = sel_or_all(self.session, Surface)
                except NoneSelectedError:
                    from chimerax.core.errors import UserError
                    if self.session.selection.empty():
                        raise UserError("No atoms, bonds, or surfaces visible")
                    else:
                        raise UserError("No visible atoms, bonds, or surfaces selected")
            else:
                if "sel" not in selector:
                    # no visible atoms/bonds selected, see if any surfaces are
                    try:
                        surf_selector = sel_or_all(self.session, Surface)
                    except NoneSelectedError:
                        pass
                    else:
                        if "sel" in surf_selector:
                            selector = surf_selector
        run(self.session, cmd % selector)

    def _get_label_text_arg(self):
        from Qt.QtWidgets import QInputDialog
        from chimerax.core.commands import StringArg
        user_text, okay = QInputDialog.getText(self, "Custom Label Text", "Label:")
        if okay:
            return StringArg.unparse(user_text)
        from chimerax.core.errors import CancelOperation
        raise CancelOperation("Custom labeling cancelled")

    def _populate_select_menu(self, select_menu):
        from Qt.QtGui import QAction
        sel_seq_action = QAction("Sequence...", self)
        select_menu.addAction(sel_seq_action)
        sel_seq_action.triggered.connect(self.show_select_seq_dialog)
        sel_zone_action = QAction("&Zone...", self)
        select_menu.addAction(sel_zone_action)
        sel_zone_action.triggered.connect(self.show_select_zone_dialog)
        sel_contacts_action = QAction("Con&tacts...", self)
        select_menu.addAction(sel_contacts_action)
        sel_contacts_action.triggered.connect(self.show_select_contacts_dialog)
        from chimerax.core.commands import run
        for menu_label, cmd_args in [("&Clear", "clear"), ("&Invert", "~sel"), ("&All", ""),
                ("&Broaden", "up"), ("&Narrow", "down")]:
            action = QAction(menu_label, self)
            select_menu.addAction(action)
            action.triggered.connect(lambda *, run=run, ses=self.session, cmd="sel " + cmd_args:
                run(ses, cmd))

        self.select_mode_menu = select_menu.addMenu("mode")
        self.select_mode_menu.setObjectName("mode")
        mode_names =  ["replace", "add", "subtract", "intersect"]
        self._select_mode_reminders = {k:v for k,v in zip(mode_names,
            ["", " (+)", " (-)", " (\N{INTERSECTION})"])}
        for mode in mode_names:
            mode_action = QAction(mode.title() + self._select_mode_reminders[mode], self)
            self.select_mode_menu.addAction(mode_action)
            mode_action.triggered.connect(
                lambda *, s=self, m=mode: s._set_select_mode(m))
        self._set_select_mode("replace")

        selectors_menu = select_menu.addMenu("User-Defined Selectors")
        selectors_menu.setToolTipsVisible(True)
        selectors_menu.aboutToShow.connect(lambda *, menu=selectors_menu: self._populate_selectors_menu(menu))
        from chimerax.core.commands import run
        selectors_menu.triggered.connect(lambda name: self.select_by_mode(name.text()))
        def_selector_action = QAction("Define Selector...", self)
        select_menu.addAction(def_selector_action)
        def_selector_action.triggered.connect(self.show_define_selector_dialog)

    def _populate_selectors_menu(self, menu):
        names = []
        from chimerax.core.commands import list_selectors, is_selector_user_defined, get_selector
        from chimerax.core.objects import Objects
        for selector_name in list_selectors():
            if not is_selector_user_defined(selector_name):
                continue
            val = get_selector(selector_name)
            if isinstance(val, Objects) and val.empty():
                continue
            names.append(selector_name)
        menu.clear()
        if not names:
            menu.addAction("No user-defined selectors")
            menu.actions()[0].setEnabled(False)
            return
        names.sort()
        for name in names:
            menu.addAction(name)

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
        from Qt.QtWidgets import QMenu
        from Qt.QtCore import Qt
        select_menu = _find_child_menu(mb, "Select")
        select_menu.setTitle("Select" + self._select_mode_reminders[mode_text])

    def update_favorites_menu(self, session):
        from Qt.QtGui import QAction
        from chimerax.core.commands import run, StringArg
        # Due to Settings possibly being displayed in another menu (but the actions
        # still being in this menu), be tricky about clearing out menu
        prev_actions = self.favorites_menu.actions()
        if prev_actions:
            separator, settings = prev_actions[-2:]
            for action in prev_actions[:-2]:
                self.favorites_menu.removeAction(action)
        for fave in session.ui.settings.favorites:
            fave_action = QAction(fave, self)
            fave_action.triggered.connect(lambda *, ses=session, run=run, fave=fave:
                run(ses, "ui tool show %s" % (StringArg.unparse(fave))))
            if prev_actions:
                self.favorites_menu.insertAction(separator, fave_action)
            else:
                self.favorites_menu.addAction(fave_action)
        if not prev_actions:
            self.favorites_menu.addSeparator()
            settings = QAction("Settings...", self)
            settings.setToolTip("Show/set ChimeraX settings")
            settings.triggered.connect(lambda *, self=self: self.settings_ui_widget.show())
            self.favorites_menu.addAction(settings)

    def update_tools_menu(self, session):
        self._checkbutton_tools = {}
        from Qt.QtWidgets import QMenu
        from Qt.QtGui import QAction
        tools_menu = QMenu("&Tools", self.menuBar())
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
        from chimerax.core.commands import run, StringArg
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
                        lambda checked, *, ses=session, run=run, tool_name=tool_name:
                        run(ses, "ui tool %s %s" % (("show" if checked else "hide"),
                        StringArg.unparse(tool_name))))
                    self._checkbutton_tools[tool_name] = tool_action
                else:
                    tool_action.triggered.connect(
                        lambda *, ses=session, run=run, tool_name=tool_name:
                        run(ses, "ui tool show %s" % StringArg.unparse(tool_name)))
                cat_menu.addAction(tool_action)
        more_tools = QAction("More Tools...", self)
        more_tools.setToolTip("Open ChimeraX Toolshed in Help Viewer")
        more_tools.triggered.connect(
            lambda *, ses=session, run=run: run(ses, "toolshed show"))
        tools_menu.addAction(more_tools)
        # running tools will go below this...
        self._tools_menu_separator = tools_menu.addSection("Running Tools")
        tools_menu.aboutToShow.connect(self._update_running_tools)
        mb = self.menuBar()
        old_action = self.tools_menu.menuAction()
        mb.insertMenu(old_action, tools_menu)
        mb.removeAction(old_action)
        self.tools_menu = tools_menu

    def _update_running_tools(self, *args):
        # clear out old running tools
        seen_sep = False
        max_text_len = 0
        for action in self.tools_menu.actions():
            if seen_sep:
                self.tools_menu.removeAction(action)
            elif action == self._tools_menu_separator:
                seen_sep = True
            else:
                max_text_len = max(max_text_len, len(action.text()))
        ellipsis_threshold = max_text_len + 2 # account for rollover arrow
        from Qt.QtGui import QAction
        running_actions = []
        for tool_instance, tool_windows in self.tool_instance_to_windows.items():
            for tw in tool_windows:
                if not isinstance(tw, MainToolWindow):
                    continue
                if len(tw.title) > ellipsis_threshold:
                    front = int((ellipsis_threshold+1)/2)
                    back = ellipsis_threshold - front
                    action_text = tw.title[:front] + "\N{HORIZONTAL ELLIPSIS}" + tw.title[-back:]
                else:
                    action_text = tw.title
                tool_action = QAction(action_text, self)
                tool_action.setToolTip("%s %s tool" % (("Hide" if tw.shown else "Show"), tw.title))
                tool_action.setCheckable(True)
                tool_action.setChecked(tw.shown)
                tool_action.triggered.connect(lambda *, tw=tw: setattr(tw, 'shown', not tw.shown))
                running_actions.append(tool_action)
        running_actions.sort(key=lambda act: act.text())
        for action in running_actions:
            self.tools_menu.addAction(action)

    def _set_tool_checkbuttons(self, toolbar, visibility):
        if toolbar.windowTitle() in self._checkbutton_tools:
            self._checkbutton_tools[toolbar.windowTitle()].setChecked(visibility)

    def add_menu_entry(self, menu_names, entry_name, callback, *, tool_tip=None, insertion_point=None,
            shortcut=None):
        '''Supported API.
        Add a main menu entry.  Adding entries to the Select menu should normally be done via
        the add_select_submenu method instead.  For details, see the doc string for that method.

        Menus that are needed but that don't already exist (including top-level ones) will
        be created.  The menu names (and entry name) can contain appropriate keyboard navigation
        markup.  Callback function takes no arguments.  This method cannot be used to add entries
        to menus that are updated dynamically, such as Tools.

        If 'insertion_point is specified, then the entry will be inserted before it.
        'insertion_point' can be a QAction, a string (menu item text with navigation markup removed)
        an integer indicating a particular separator (top to bottom, numbering starting at 1),
        or False indicating that the entry should be placed at the top of the menu.
        '''
        menu = self._get_target_menu(self.menuBar(), menu_names)
        from Qt.QtGui import QAction
        action = QAction(entry_name, self)
        action.triggered.connect(callback)
        if tool_tip is not None:
            action.setToolTip(tool_tip)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if insertion_point is None:
            menu.addAction(action)
        elif insertion_point is False:
            existing_actions = menu.actions()
            if not existing_actions:
                menu.addAction(action)
            else:
                menu.insertAction(existing_actions[0], action)
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
        from Qt.QtGui import QAction
        action = QAction(label, self)
        action.triggered.connect(lambda *, st=selector_text: self.select_by_mode(st))
        if insertion_point is None:
            menu.addAction(action)
        else:
            menu.insertAction(self._get_menu_action(menu, insertion_point), action)
        return action

    def remove_menu(self, menu_names):
        menu = self._get_target_menu(self.menuBar(), menu_names)
        if menu:
            menu.deleteLater()

    def _get_menu_action(self, menu, insertion_point):
        from Qt.QtGui import QAction
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
        from Qt.QtWidgets import QMenu
        from Qt.QtCore import Qt
        if insert_positions is None:
            insert_positions = [None]*len(menu_names)
        for menu_name, insert_pos in zip(menu_names, insert_positions):
            obj_name = remove_keyboard_navigation(menu_name)
            menu = _find_child_menu(parent_menu, obj_name)
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
                menu.setObjectName(obj_name)    # Needed for findChild() above to work.
                if insert_pos is False:
                    parent_menu.addMenu(menu)
                else:
                    parent_menu.insertMenu(insert_action, menu)
            parent_menu = menu
        return parent_menu

    def _tool_window_destroy(self, tool_window):
        # via request from non-UI code to destroy window
        tool_instance = tool_window.tool_instance
        all_windows = self.tool_instance_to_windows[tool_instance]
        is_main_window = tool_window is all_windows[0]
        if is_main_window:
            tool_instance.delete()
            return
        tool_window._destroy()

    def _tool_window_destroyed(self, tool_window):
        # tool window (is about to be) destroyed, both via UI and non-UI code
        all_windows = self.tool_instance_to_windows[tool_window.tool_instance]
        all_windows.remove(tool_window)
        if tool_window in getattr(self, '_hide_tools_shown_states', {}):
            del self._hide_tools_shown_states[tool_window]


    def _tool_window_request_shown(self, tool_window, shown):
        if shown and not tool_window.floating:
            self.hide_tools = False
        if self.hide_tools and not shown and not tool_window.floating:
            self._hide_tools_shown_states[tool_window] = False
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
                    if self.hide_tools and not window.floating:
                        self._hide_tools_shown_states[window] = False
                    window._mw_set_shown(False)

def _find_child_menu(w, name):
    # Qt 5.15 does not support the options argument to QObject.findChild().
    #menu = w.findChild(QMenu, name, Qt.FindDirectChildrenOnly)
    from Qt.QtWidgets import QMenu
    cm = [c for c in w.children() if isinstance(c, QMenu) and c.objectName() == name]
    return cm[0] if cm else None

def _open_dropped_file(session, path):
    if not path:
        return
    from chimerax.core.commands import run, FileNameArg
    run(session, 'open %s' % FileNameArg.unparse(path))

from Qt.QtCore import Qt
keyboard_state_keys = set([Qt.Key.Key_CapsLock, Qt.Key.Key_NumLock, Qt.Key.Key_ScrollLock, Qt.Key.Key_AltGr])

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

    def __init__(self, tool_instance, title, *, close_destroys=True, hide_title_bar=False, statusbar=False):
        StatusLogger.__init__(self, tool_instance.session)
        self.tool_instance = tool_instance
        self.close_destroys = close_destroys
        ui = tool_instance.session.ui
        mw = ui.main_window
        self.__toolkit = _Qt(self, title, statusbar, hide_title_bar, mw, close_destroys)
        self.ui_area = self.__toolkit.ui_area
        # forward unused keystrokes (to the command line by default)
        self.ui_area.keyPressEvent = self._forward_keystroke
        mw._new_tool_window(self)

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

    def _get_floating(self):
        return self.__toolkit.dock_widget.isFloating()
    def _set_floating(self, floating):
        return self.__toolkit.dock_widget.setFloating(floating)
    floating = property(_get_floating, _set_floating)

    @property
    def hides_title_bar(self):
        return self.__toolkit.hide_title_bar


    # PyQt6 uses Python enums instead of ints
    dock_area_value = lambda x: x if isinstance(x, int) else x.value
    from Qt.QtCore import Qt
    window_placement_to_text = {
        dock_area_value(Qt.DockWidgetArea.RightDockWidgetArea): "right",
        dock_area_value(Qt.DockWidgetArea.LeftDockWidgetArea): "left",
        dock_area_value(Qt.DockWidgetArea.TopDockWidgetArea): "top",
        dock_area_value(Qt.DockWidgetArea.BottomDockWidgetArea): "bottom"
    }
    def manage(self, placement, fixed_size=False, allowed_areas=Qt.DockWidgetArea.AllDockWidgetAreas,
            initially_hidden=False):
        """Supported API. Show this tool window in the interface

        Tool will be docked into main window on the side indicated by
        `placement` (which should be a value from :py:attr:`placements` or 'side'
        or None, or another tool window).  If `placement` is "side", then the user-preferred
        side will be used.  If `placement` is None, the tool will
        be detached from the main window.  If `placement` is another tool window,
        then those tools will be tabbed together.

        If fixed_size is True then the vertical size of the panel will equal the
        requested size, otherwise the panel could be vertically larger or smaller than
        its layout requires.

        The tool window will be allowed to dock in the allowed_areas, the value
        of which is a bitmask formed from Qt's Qt.DockWidgetAreas flags.

        The tool will be displayed unless 'initially_hidden' is True.  This flag is needed because
        setting tool.shown to False after manage() will otherwise briefly show the tool.
        """
        ui = self.session.ui
        settings =  ui.settings
        tool_name = self.tool_instance.tool_name
        if settings.auto_float_tools and tool_name not in settings.autostart:
            if tool_name not in settings.undockable:
                settings.undockable = settings.undockable + [tool_name]
        from Qt.QtCore import Qt
        if tool_name in settings.undockable:
            allowed_areas = Qt.DockWidgetArea.NoDockWidgetArea
        geometry = None
        if tool_name in settings.tool_positions['windows'] and isinstance(self, MainToolWindow):
            pos_info = settings.tool_positions['windows'][tool_name]
            if type(pos_info) == dict:
                placement = pos_info['placement']
                geom_info = pos_info['geom_info']
                tab_info = pos_info['tab_info']
            else:
                version, *info = settings.tool_positions['windows'][tool_name]
                if version == 1:
                    placement, geom_info = info
                    tab_info = []
                else:
                    placement, geom_info, tab_info = info
            if placement is not None:
                for tabbed_with in tab_info:
                    for ti, tws in ui.main_window.tool_instance_to_windows.items():
                        if ti.tool_name == tabbed_with:
                            placement = tws[0]
                            break
                    else:
                        continue
                    break
                else:
                    placement = self.window_placement_to_text[self.__class__.dock_area_value(placement)]
            if geom_info is not None:
                from Qt.QtCore import QRect
                geometry = QRect(*geom_info)
        resize_docked = False
        place_floating = placement is None or (isinstance(placement, ToolWindow) and placement.floating)
        if not place_floating and placement in ('side', 'left', 'right'):
            # if the tool's sizeHint indicates that the graphics window would be forced to shrink by more
            # than 50% to accomodate; resize docked widget instead
            central_width = ui.main_window.centralWidget().size().width()
            overall_width = ui.main_window.size().width()
            graphics_width = overall_width - central_width
            if overall_width - self.ui_area.sizeHint().width() < graphics_width / 2:
                resize_docked = True
        ui.main_window._about_to_manage(self, place_floating)
        self.__toolkit.manage(placement, allowed_areas, fixed_size, geometry)
        if resize_docked:
            ui.main_window.resizeDocks([self._dock_widget], [overall_width - central_width], Qt.Horizontal)
        if initially_hidden:
            self.shown = False
        else:
            ui.triggers.activate_trigger('tool window show', self)

    @property
    def shown(self):
        """Whether this window is hidden or shown"""
        return self.__toolkit.shown

    @shown.setter
    def shown(self, shown):
        ui = self.session.ui
        ui.main_window._tool_window_request_shown(self, shown)
        tname = 'tool window show' if shown else 'tool window hide'
        ui.triggers.activate_trigger(tname, self)

    def shown_changed(self, shown):
        """Supported API. Perform actions when window hidden/shown

        Override to perform any actions you want done when the window
        is hidden (\ `shown` = False) or shown (\ `shown` = True)"""
        pass

    def shrink_to_fit(self):
        """Supported API. Resize the window to take up the minimum size needed by its contents.
           Typically used after hiding widgets.
        """
        dw = self.__toolkit.dock_widget
        if self.floating:
            resize = lambda dw=dw: dw.resize(0,0)
        else:
            dock_area = self.session.ui.main_window.dockWidgetArea(dw)
            from Qt.QtCore import Qt
            if dock_area == Qt.DockWidgetArea.LeftDockWidgetArea or dock_area == Qt.DockWidgetArea.RightDockWidgetArea:
                orientation = Qt.Vertical
            else:
                orientation = Qt.Horizontal
            resize = lambda mw=self.session.ui.main_window, dw=dw, orientation=orientation: \
                mw.resizeDocks([dw], [1], orientation)
        from Qt.QtCore import QTimer
        # 0 is empirically "too fast", as is 10 and 12, using 25 msec
        QTimer.singleShot(25, resize)

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
        self.ui_area = None
        self.tool_instance = None

    @property
    def _dock_widget(self):
        """This is for emergency access to the QDockWidget.  If you find yourself needing to use this,
           you should send mail to chimerax-users and request an enhancement to the ToolWindow API to
           directly support whatever you are using the dock widget for.
        """
        return self.__toolkit.dock_widget

    def _forward_keystroke(self, event):
        # Exclude floating windows because they don't forward all keystrokes (e.g. Delete)
        # and because the Google sign-on (via the typically floating Help Viewer) forwards
        # _just_ the Return key (well, and shift/control/other non-printable)
        #
        # QLineEdits don't eat Return keys, so they may propagate to the
        # top widget; don't forward keys if the focus widget is a QLineEdit
        #
        # Since forwarding keystrokes can shift the keyboard focus, don't forward keys that
        # are "unhandled" if those keys only change keyboard state (e.g. CapsLock).  Important
        # for the Python Shell retaining focus.
        from Qt.QtWidgets import QLineEdit, QComboBox
        if not self.floating and not isinstance(self.ui_area.focusWidget(), (QLineEdit, QComboBox)) \
        and event.key() not in keyboard_state_keys:
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

    Parameters:
        tool_instance: a :py:class:`~chimerax.core.tools.ToolInstance` instance
            The tool creating this window.
    """
    def __init__(self, tool_instance, **kw):
        super().__init__(tool_instance, tool_instance.display_name, **kw)

    def create_child_window(self, title: str, *, window_class: 'ChildToolWindow'=None, **kw):
        """Supported API. Make additional tool window

        Parameters:
            title: Text shown in the window's title bar.
            window_class: Class to instantiate to create the child window. Only
                          needed if you want to override methods/attributes in
                          order to change behavior.
            kw: Keywords to pass on to the tool window's constructor
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
    def __init__(self, tool_window, title, has_statusbar, hide_title_bar, main_window, close_destroys):
        self.tool_window = tool_window
        self.title = title
        self.hide_title_bar = hide_title_bar
        self.main_window = mw = main_window
        self._destroyed = False

        if not mw:
            raise RuntimeError("No main window or main window dead")

        from Qt.QtWidgets import QDockWidget, QWidget, QVBoxLayout
        self.dock_widget = dw = QDockWidget(title, mw)
        dw.closeEvent = lambda e, *, tw=tool_window, mw=mw: mw.close_request(tw, e)
        if close_destroys:
            from Qt.QtCore import Qt
            dw.setAttribute(Qt.WA_DeleteOnClose)
        dw.topLevelChanged.connect(self.float_changed)
        if hide_title_bar:
            self.dock_widget.setTitleBarWidget(QWidget())

        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 1, 0, 0) # all zeros produces a complaint about -1 height
        self.ui_area = QWidget()
        self.ui_area.contextMenuEvent = lambda e, *, self=self: self.show_context_menu(e)
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
        self._docked_window_flags = self.dock_widget.windowFlags()

    def destroy(self):
        if self._destroyed:
            return  # already destroyed
        self._destroyed = True
        from Qt.QtCore import Qt
        auto_delete = self.dock_widget.testAttribute(Qt.WA_DeleteOnClose)
        is_floating = self.dock_widget.isFloating()
        self.main_window._tool_window_destroyed(self.tool_window)
        self.main_window.removeDockWidget(self.dock_widget)
        # free up references
        self.tool_window = None
        self.main_window = None
        sbar = self.status_bar
        if sbar is not None:
            # apparently needs to be explicitly destroyed even if auto_delete is True
            sbar.destroy()
            self.status_bar = None
        if not auto_delete:
            # horrible hack to try to work around two different crashes, in 5.12:
            # 1) destroying floating window closed with red-X with immediate destroy()
            # 2) resize event to dead window if deleteLater() used
            if is_floating:
                self.dock_widget.deleteLater()
            else:
                self.dock_widget.destroy()

    @property
    def dockable(self):
        from Qt.QtCore import Qt
        return self.dock_widget.allowedAreas() != Qt.DockWidgetArea.NoDockWidgetArea

    @dockable.setter
    def dockable(self, dockable):
        from Qt.QtCore import Qt
        areas = Qt.DockWidgetArea.AllDockWidgetAreas if dockable else Qt.DockWidgetArea.NoDockWidgetArea
        self.dock_widget.setAllowedAreas(areas)
        if not dockable and not self.dock_widget.isFloating():
            self.dock_widget.setFloating(True)

    def float_changed(self, floating):
        if self.hide_title_bar:
            from Qt.QtWidgets import QWidget
            self.dock_widget.setTitleBarWidget(None if floating else QWidget())
        import sys
        if sys.platform == 'darwin':
            # Add iconify and maximize buttons to undocked tools.
            if floating:
                dw = self.dock_widget
                vis = dw.isVisible()
                from Qt.QtCore import Qt
                # Changing window type allows undocked tool to stack
                # below main window on but issues errors on macOS Big Sur.
                # See ChimeraX bug #453 for details.
                # window_flags = (Qt.CustomizeWindowHint | Qt.Window)
                window_flags = dw.windowFlags()
                button_flags = (Qt.WindowType.WindowMinimizeButtonHint |
                                Qt.WindowType.WindowMaximizeButtonHint |
                                Qt.WindowType.WindowCloseButtonHint)
                dw.setWindowFlags(window_flags | button_flags)
                if vis:
                    dw.show()
            else:
                self.dock_widget.setWindowFlags(self._docked_window_flags)
        self.main_window._float_changed(self.tool_window, floating)

    def manage(self, placement, allowed_areas, fixed_size, geometry):
        # map 'side' to the user's preferred side
        session = self.tool_window.tool_instance.session
        from Qt.QtCore import Qt
        pref_area = Qt.DockWidgetArea.RightDockWidgetArea if session.ui.settings.default_tool_window_side == "right" \
            else Qt.DockWidgetArea.LeftDockWidgetArea
        qt_sides = [pref_area, Qt.DockWidgetArea.RightDockWidgetArea, Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.TopDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea]
        self.placement_map = dict(zip(self.tool_window.placements, qt_sides))
        placements = self.tool_window.placements
        if placement is None or isinstance(placement, ToolWindow):
            side = Qt.DockWidgetArea.RightDockWidgetArea
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
            if placement is None or allowed_areas == Qt.DockWidgetArea.NoDockWidgetArea:
                self.dock_widget.setFloating(True)
        if geometry is not None:
            self.dock_widget.setGeometry(geometry)
        self.dock_widget.setAllowedAreas(allowed_areas)

        if fixed_size:
            # Always set vertical size to what sizeHint() asks for.
            from Qt.QtWidgets import QSizePolicy
            self.dock_widget.widget().setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def show_context_menu(self, event):
        _show_context_menu(event, self.tool_window.tool_instance, self.tool_window,
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
        #
        #if shown != self.dock_widget.isHidden():
        #    if shown:
        #        #ensure it's on top
        #        self.dock_widget.raise_()
        #    return
        #
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

def _show_context_menu(event, tool_instance, tool_window, fill_cb, autostartable, memorable):
    from Qt.QtWidgets import QMenu
    from Qt.QtGui import QAction
    menu = QMenu()

    if fill_cb:
        pos = event.pos()
        fill_cb(menu, pos.x(), pos.y())
    if not menu.isEmpty():
        menu.addSeparator()
    ti = tool_instance
    hide_tool_action = QAction("Hide Tool")
    hide_tool_action.triggered.connect(lambda *, ti=ti: ti.display(False))
    menu.addAction(hide_tool_action)
    help_url = getattr(tool_window, "help", None) or ti.help
    session = ti.session
    from chimerax.core.commands import run, StringArg
    if help_url is not None:
        help_action = QAction("Help")
        help_action.setStatusTip("Show tool help")
        help_action.triggered.connect(lambda *, ses=session, run=run, help_url=help_url:
            run(ses, "help %s" % help_url))
        menu.addAction(help_action)
    else:
        no_help_action = QAction("No Help Available")
        no_help_action.setEnabled(False)
        menu.addAction(no_help_action)
    if autostartable:
        autostart = ti.tool_name in session.ui.settings.autostart
        auto_action = QAction("Start at ChimeraX Startup")
        auto_action.setCheckable(True)
        auto_action.setChecked(autostart)
        auto_action.triggered.connect(
            lambda checked, *, ses=session, run=run, tool_name=ti.tool_name:
            run(ses, "ui autostart %s %s" % (("true" if checked else "false"),
            StringArg.unparse(ti.tool_name))))
        menu.addAction(auto_action)
        favorite = ti.tool_name in session.ui.settings.favorites
        fav_action = QAction("In Favorites Menu")
        fav_action.setCheckable(True)
        fav_action.setChecked(favorite)
        from chimerax.core.commands import run, StringArg
        fav_action.triggered.connect(
            lambda checked, *, ses=session, run=run, tool_name=ti.tool_name:
            run(ses, "ui favorite %s %s" % (("true" if checked else "false"),
            StringArg.unparse(ti.tool_name))))
        menu.addAction(fav_action)
    if memorable and tool_window.hides_title_bar and not tool_window.floating:
        undock_action = QAction("Undock")
        undock_action.triggered.connect(lambda *, dock_widget=memorable: dock_widget.setFloating(True))
        menu.addAction(undock_action)
    undockable = ti.tool_name in session.ui.settings.undockable
    dock_action = QAction("Dockable Tool")
    dock_action.setCheckable(True)
    dock_action.setChecked(not undockable)
    from chimerax.core.commands import run, StringArg
    dock_action.triggered.connect(
        lambda checked, *, ses=session, run=run, tool_name=ti.tool_name:
        run(ses, "ui dockable %s %s" % (("true" if checked else "false"),
        StringArg.unparse(ti.tool_name))))
    menu.addAction(dock_action)
    if memorable:
        position_action = QAction("Save Tool Position")
        position_action.setStatusTip("Use current docked side,"
            " or undocked size/position as default")
        position_action.triggered.connect(lambda *, ui=session.ui, widget=memorable, ti=ti:
            _remember_tool_pos(ui, ti, widget))
        menu.addAction(position_action)
    p = event.globalPos()  if hasattr(event, 'globalPos') else event.globalPosition().toPoint()
    if hasattr(menu, 'exec'):
        menu.exec(p)	# PyQt6
    else:
        menu.exec_(p)	# PyQt5

def _remember_tool_pos(ui, tool_instance, widget):
    mw = ui.main_window
    from Qt.QtWidgets import QToolBar
    # need to copy _before_ modifying, so that default isn't also changed
    from copy import deepcopy
    remembered = deepcopy(ui.settings.tool_positions)
    if isinstance(widget, QToolBar):
        if widget.isFloating():
            from chimerax.core.errors import LimitationError
            raise LimitationError("Cannot currently save toolbars as floating")
        get_side = mw.toolBarArea
        mem_location = remembered['toolbars']
        tab_info = []
    else:
        get_side = mw.dockWidgetArea
        mem_location = remembered['windows']
        tabbed_with = mw.tabifiedDockWidgets(widget)
        if tabbed_with:
            tab_info = []
            for ti, tws in mw.tool_instance_to_windows.items():
                if tws[0]._dock_widget in tabbed_with:
                    tab_info.append(ti.tool_name)
            mw.status('To save tabbed positions, use "Save Tool Position" on each tab', "blue", False)
        else:
            tab_info = []
    if widget.isFloating():
        side = None
        geom = widget.geometry()
        geom_info = (geom.x(), geom.y(), geom.width(), geom.height())
    else:
        # unlike PyQt, PySide needs cast to int
        side = int(get_side(widget))
        geom_info = None
    version = 3
    mem_location[tool_instance.tool_name] = {
        'version': version,
        'placement': side,
        'geom_info': geom_info,
        'tab_info': tab_info
    }
    ui.settings.tool_positions = remembered
    ui.settings.save()

def remove_keyboard_navigation(menu_label):
    if menu_label.count('&') == 1:
        return menu_label.replace('&', '')
    return menu_label.replace('&&', '&')

from Qt.QtWidgets import QDialog
class DefineSelectorDialog(QDialog):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.setWindowTitle("Define Selector")
        self.setSizeGripEnabled(True)
        from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QLineEdit, QHBoxLayout, QLabel
        layout = QVBoxLayout()
        def_layout = QHBoxLayout()
        def_layout.setSpacing(4)
        def_layout.addWidget(QLabel("Name"))
        from Qt.QtWidgets import QPushButton, QMenu
        self.cur_sel_text = "current selection"
        self.atom_spec_text = "target specifier"
        self.push_button = QPushButton(self.cur_sel_text)
        menu = QMenu(self)
        menu.triggered.connect(self._menu_cb)
        self.push_button.setMenu(menu)
        from Qt.QtGui import QAction
        for text in [self.cur_sel_text, self.atom_spec_text]:
            menu.addAction(text)
        def_layout.addWidget(self.push_button)
        self.atom_spec_edit = QLineEdit()
        self.atom_spec_edit.textChanged.connect(self._update_button_states)
        self.atom_spec_edit.hide()
        def_layout.addWidget(self.atom_spec_edit)
        def_layout.addWidget(QLabel("as"))
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._update_button_states)
        def_layout.addWidget(self.name_edit)
        layout.addLayout(def_layout)

        self.bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Apply | qbbox.Help)
        self.bbox.accepted.connect(self.def_selector)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        self.bbox.button(qbbox.Apply).clicked.connect(self.def_selector)
        from chimerax.core.commands import run
        self.bbox.helpRequested.connect(lambda *, run=run, ses=session:
            run(ses, "help help:user/menu.html#named-selections"))
        self._update_button_states()
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def _menu_cb(self, action):
        if action.text() == self.cur_sel_text:
            self.atom_spec_edit.hide()
        else:
            self.atom_spec_edit.show()
        self.push_button.setText(action.text())
        self._update_button_states()

    def def_selector(self, *args):
        from chimerax.core.commands import run, StringArg
        if self.push_button.text() == self.cur_sel_text:
            command = "name frozen"
            spec = "sel"
        else:
            command = "name"
            spec = self.atom_spec_edit.text()
        run(self.session, "%s %s %s" % (command, StringArg.unparse(self.name_edit.text()), spec))

    def _update_button_states(self, *args):
        enable = bool(self.name_edit.text().strip())
        if enable and self.push_button.text() == self.atom_spec_text:
            enable = bool(self.atom_spec_edit.text().strip())
        for button in [self.bbox.button(x) for x in [self.bbox.Ok, self.bbox.Apply]]:
            button.setEnabled(enable)

class SelSeqDialog(QDialog):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.setWindowTitle("Select Sequence")
        self.setSizeGripEnabled(True)
        from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QLineEdit, QHBoxLayout, QLabel
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
        self.bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help help:user/findseq.html"))
        self._update_button_states(self.edit.text())
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def search(self, *args):
        from chimerax.core.commands import run, StringArg
        run(self.session, "sel seq %s" % StringArg.unparse(self.edit.text().strip()))

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
        from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QLineEdit, QHBoxLayout, QLabel, \
            QCheckBox, QDoubleSpinBox, QPushButton, QMenu, QWidget, QTabWidget
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        target_area = QWidget()
        target_layout = QHBoxLayout()
        target_layout.setContentsMargins(0,0,0,0)
        target_layout.setSpacing(3)
        target_area.setLayout(target_layout)
        target_layout.addWidget(QLabel("Select"))
        self.target_button = QPushButton("atoms")
        menu = QMenu()
        menu.triggered.connect(lambda action: self.target_button.setText(action.text()))
        menu.addAction("atoms")
        menu.addAction("residues")
        self.target_button.setMenu(menu)
        target_layout.addWidget(self.target_button)
        target_layout.addWidget(QLabel(":"))
        layout.addWidget(target_area, alignment=Qt.AlignLeft)
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
        more_layout.addWidget(self.more_checkbox, alignment=Qt.AlignRight)
        self.more_spinbox = QDoubleSpinBox()
        self.more_spinbox.setValue(5.0)
        self.more_spinbox.setDecimals(3)
        self.more_spinbox.setSuffix("\N{ANGSTROM SIGN}")
        self.more_spinbox.setMaximum(9999.999)
        more_layout.addWidget(self.more_spinbox)
        more_layout.addWidget(QLabel("from the currently selected atoms"))
        layout.addLayout(more_layout)

        self.bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        self.bbox.accepted.connect(self.zone)
        self.bbox.button(qbbox.Apply).clicked.connect(self.zone)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        from chimerax.core.commands import run
        self.bbox.helpRequested.connect(lambda *, run=run, ses=session:
            run(ses, "help help:user/menu.html#selectzone"))
        self._update_button_states()
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def zone(self, *args):
        cmd = ""
        char = ':' if self.target_button.text() == "residues" else '@'
        if self.less_checkbox.isChecked():
            cmd += "sel %s< %g" % (char, self.less_spinbox.value())
            if self.more_checkbox.isChecked():
                cmd += ' & '
        if self.more_checkbox.isChecked():
            cmd += "sel %s> %g" % (char, self.more_spinbox.value())
        self.session.ui.main_window.select_by_mode(cmd)

    def _update_button_states(self, *args):
        self.bbox.button(self.bbox.Ok).setEnabled(
            self.less_checkbox.isChecked() or self.more_checkbox.isChecked())

class SelContactsDialog(QDialog):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.setWindowTitle("Select Contacts")
        self.setSizeGripEnabled(True)
        from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QRadioButton, QHBoxLayout, QLabel, \
            QButtonGroup, QDoubleSpinBox, QPushButton, QMenu, QWidget, QTabWidget, QGridLayout, QGroupBox
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Chains tab
        chains_widget = QWidget()
        chains_layout = QVBoxLayout()
        chains_layout.setContentsMargins(0,0,0,0)
        chains_layout.setSpacing(3)
        chains_widget.setLayout(chains_layout)
        lists_layout = QGridLayout()
        for col in (0,2,4):
            lists_layout.setColumnStretch(col, 1)
        lists_layout.addWidget(QLabel("Select contacts of:"), 0, 1, alignment=Qt.AlignCenter)
        from chimerax.atomic.widgets import ChainListWidget
        self.chains1 = ChainListWidget(self.session, autoselect=ChainListWidget.AUTOSELECT_FIRST)
        lists_layout.addWidget(self.chains1, 1, 1)
        lists_layout.addWidget(QLabel("with:"), 0, 3, alignment=Qt.AlignCenter)
        self.chains2 = ChainListWidget(self.session, autoselect=ChainListWidget.AUTOSELECT_FIRST,
            filter_func=lambda x, cl=self.chains1: x not in cl.value)
        lists_layout.addWidget(self.chains2, 1, 3)
        self.chains1.value_changed.connect(self.chains2.refresh)
        chains_layout.addLayout(lists_layout)
        select_layout = QHBoxLayout()
        select_layout.addStretch(1)
        select_layout.addWidget(QLabel("Select contacts in "))
        from Qt.QtWidgets import QPushButton, QMenu
        self.what_sel_button = QPushButton("both")
        select_layout.addWidget(self.what_sel_button)
        menu = QMenu(self.what_sel_button)
        from Qt.QtGui import QAction
        menu.addAction("both")
        menu.addAction("first")
        menu.triggered.connect(lambda action, but=self.what_sel_button: but.setText(action.text()))
        self.what_sel_button.setMenu(menu)
        select_layout.addWidget(QLabel(" chain(s)"))
        select_layout.addStretch(1)
        chains_layout.addLayout(select_layout)
        criteria_group = QGroupBox("")
        chains_layout.addWidget(criteria_group, alignment=Qt.AlignCenter)
        criteria_layout = QVBoxLayout()
        criteria_layout.setContentsMargins(0,0,0,0)
        criteria_layout.setSpacing(0)
        criteria_group.setLayout(criteria_layout)
        criteria_layout.addWidget(QLabel("Select residues with:"), alignment=Qt.AlignLeft)
        test_type_layout = QGridLayout()
        test_type_layout.setContentsMargins(0,0,0,0)
        test_type_layout.setSpacing(0)
        test_type_layout.setColumnStretch(2, 1)
        test_type_layout.setColumnMinimumWidth(0, 15)
        criteria_layout.addLayout(test_type_layout)
        self.criteria_button_group = QButtonGroup()
        self.buried_button = QRadioButton("")
        self.buried_button.setChecked(True)
        self.criteria_button_group.addButton(self.buried_button)
        test_type_layout.addWidget(self.buried_button, 0, 1)
        residue_area_layout = QHBoxLayout()
        residue_area_layout.addWidget(QLabel("buried solvent-accessible surface area >="))
        self.residue_spinbox = QDoubleSpinBox()
        from chimerax.interfaces import residue_area_default
        self.residue_spinbox.setValue(residue_area_default)
        self.residue_spinbox.setDecimals(1)
        self.residue_spinbox.setMinimum(0.0)
        self.residue_spinbox.setMaximum(9999.9)
        self.residue_spinbox.setAlignment(Qt.AlignHCenter)
        residue_area_layout.addWidget(self.residue_spinbox)
        residue_area_layout.addWidget(QLabel("\N{ANGSTROM SIGN}\N{SUPERSCRIPT TWO}"))
        test_type_layout.addLayout(residue_area_layout, 0, 2)
        self.distance_button = QRadioButton("")
        self.criteria_button_group.addButton(self.distance_button)
        test_type_layout.addWidget(self.distance_button, 1, 1)
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("atomic distance <="))
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setValue(3.5)
        self.distance_spinbox.setDecimals(1)
        self.distance_spinbox.setMinimum(0.0)
        self.distance_spinbox.setMaximum(9999.9)
        self.distance_spinbox.setAlignment(Qt.AlignHCenter)
        distance_layout.addWidget(self.distance_spinbox)
        distance_layout.addWidget(QLabel("\N{ANGSTROM SIGN}"))
        test_type_layout.addLayout(distance_layout, 1, 2)

        self.tabs.addTab(chains_widget, "Chains")

        # Atomic tab
        contacts_widget = QWidget()
        contacts_layout = QVBoxLayout()
        contacts_layout.setContentsMargins(0,0,0,0)
        contacts_layout.setSpacing(3)
        contacts_widget.setLayout(contacts_layout)
        from chimerax.clashes.gui import ContactsGUI
        self.contacts_gui = ContactsGUI(self.session, True, settings_name="select zone",
            action_phrase="select contacts", restrict="any", show_attr_name=False,
            show_checking_frequency=False, show_color=False, show_dashes=False, show_log=False,
            show_make_pseudobonds=False, show_name=False, show_radius=False, show_reveal=False,
            show_save_file=False, show_section_titles=False, show_select=False, show_set_attrs=False,
            show_show_dist=False)
        contacts_layout.addWidget(self.contacts_gui, alignment=Qt.AlignCenter)
        full_gui_but = QPushButton("Show Contacts tool")
        from chimerax.core.commands import run
        full_gui_but.clicked.connect(lambda *args, run=run, ses=self.session:
            run(ses, "ui tool show Contacts"))
        contacts_layout.addWidget(full_gui_but, alignment=Qt.AlignCenter)
        self.tabs.addTab(contacts_widget, "Atomic")

        self.bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        self.bbox.accepted.connect(self.contacts)
        self.bbox.button(qbbox.Apply).clicked.connect(self.contacts)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        from chimerax.core.commands import run
        self.bbox.helpRequested.connect(lambda *, run=run, ses=session:
            run(ses, "help help:user/selectcontacts.html"))
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def contacts(self, *args):
        from chimerax.core.commands import run
        if self.tabs.tabText(self.tabs.currentIndex()) == "Chains":
            chains1 = self.chains1.value
            chains2 = self.chains2.value
            if not chains1 or not chains2:
                from chimerax.core.errors import UserError
                raise UserError("Must select at least one chain from each list")
            chain_spec1 = "".join([c.atomspec for c in chains1])
            chain_spec2 = "".join([c.atomspec for c in chains2])
            if self.criteria_button_group.checkedButton() == self.buried_button:
                cmd = "interfaces select %s & ::polymer_type>0 " \
                    "contacting %s & ::polymer_type>0 areaCutoff 0" % (chain_spec1, chain_spec2)
                if self.what_sel_button.text() == "both":
                    cmd += " bothSides true"
                from chimerax.interfaces import residue_area_default
                buried_residue_area = self.residue_spinbox.value()
                if buried_residue_area != residue_area_default:
                    cmd += " interfaceResidueAreaCutoff %g" % buried_residue_area
            else:
                d = self.distance_spinbox.value()
                spec = "(%s & ::polymer_type>0 ) & ((%s & ::polymer_type>0 ) :<%g)" % (
                    chain_spec1, chain_spec2, d)
                if self.what_sel_button.text() == "both":
                    spec = "(%s) | ((%s & ::polymer_type>0 ) & ((%s & ::polymer_type>0 ) :<%g))" % (
                        spec, chain_spec2, chain_spec1, d)
                cmd = "sel " + spec
        else:
            cmd = "%s %s %s sel true make false" % self.contacts_gui.get_command()
        run(self.session, cmd)


class LabelHeightDialog(QDialog):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.setWindowTitle("Set Label Height")
        from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox, QLineEdit, QHBoxLayout, QLabel, \
            QCheckBox, QPushButton, QMenu, QWidget
        from Qt.QtCore import Qt
        from Qt.QtGui import QDoubleValidator, QAction
        layout = QVBoxLayout()
        height_area = QWidget()
        layout.addWidget(height_area)
        height_layout = QHBoxLayout()
        height_layout.setContentsMargins(0,0,0,0)
        height_layout.setSpacing(3)
        height_area.setLayout(height_layout)
        height_layout.addWidget(QLabel("Set all label heights to"))
        self.height_entry = QLineEdit()
        self.height_entry.setMaximumWidth(50)
        from chimerax.label.settings import settings
        self.height_entry.setText(str(settings.label_height))
        height_layout.addWidget(self.height_entry)
        self.unit_button = QPushButton("\N{ANGSTROM SIGN}")
        menu = QMenu(self.unit_button)
        menu.triggered.connect(lambda action: self.unit_button.setText(action.text()))
        menu.addAction(QAction("\N{ANGSTROM SIGN}", self.unit_button))
        menu.addAction(QAction("pixels", self.unit_button))
        self.unit_button.setMenu(menu)
        height_layout.addWidget(self.unit_button)

        self.bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        self.bbox.accepted.connect(self.set_height)
        self.bbox.button(qbbox.Apply).clicked.connect(self.set_height)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        self.bbox.button(self.bbox.Help).setEnabled(False)
        #from chimerax.core.commands import run
        #self.bbox.helpRequested.connect(lambda *, run=run, ses=session:
        #    run(ses, "help help:user/menu.html#selectzone"))
        layout.addWidget(self.bbox)
        self.setLayout(layout)

    def set_height(self, *args):
        from chimerax.core.errors import UserError
        in_pixels = self.unit_button.text() == 'pixels'
        if in_pixels:
            try:
                height = int(self.height_entry.text())
            except ValueError:
                raise UserError("Pixels must be an integer")
        else:
            try:
                height = float(self.height_entry.text())
            except ValueError:
                raise UserError("Height must be a number")
        if height <= 0:
            raise UserError("Height must be a positive number")
        if in_pixels:
            command = "label size %g height fixed" % height
        else:
            command = "label height %g" % height
        from chimerax.core.commands import run
        run(self.session, command)

def menu_capitalize(text):
    from chimerax.core.utils import titleize
    return titleize(text)

from .options import Option, EnumOption
class ToolSideOption(EnumOption):
    values = ("left", "right")

class InitWindowSizeOption(Option):

    def __init__(self, *args, session=None, **kw):
        self.session = session
        Option.__init__(self, *args, **kw)

    def get_value(self):
        size_scheme = self.push_button.text()
        if size_scheme == "last used":
            data = None
        elif size_scheme == "proportional":
            data = (self.w_proportional_spin_box.value()/100,
                self.h_proportional_spin_box.value()/100)
        else:
            data = (self.w_fixed_spin_box.value(), self.h_fixed_spin_box.value())
        return (size_scheme, data)

    def set_value(self, value):
        size_scheme, size_data = value
        self.push_button.setText(size_scheme)
        if size_scheme == "proportional":
            w, h = size_data
            data = (self.w_proportional_spin_box.setValue(w*100),
                self.h_proportional_spin_box.setValue(h*100))
        elif size_scheme == "fixed":
            w, h = size_data
            self.w_fixed_spin_box.setValue(w)
            self.h_fixed_spin_box.setValue(h)
        self._show_appropriate_widgets()

    value = property(get_value, set_value)

    def set_multiple(self):
        self.push_button.setText(self.multiple_value)

    def _make_widget(self, **kw):
        from Qt.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        self.widget.setLayout(layout)

        from Qt.QtWidgets import QPushButton, QMenu
        size_scheme, size_data = self.default
        self.push_button = QPushButton(size_scheme)
        menu = QMenu(self.widget)
        self.push_button.setMenu(menu)
        from Qt.QtGui import QAction
        menu = self.push_button.menu()
        for label in ("last used", "proportional", "fixed", "maximized"):
            action = QAction(label, self.push_button)
            action.triggered.connect(lambda *, s=self, lab=label: s._menu_cb(lab))
            menu.addAction(action)
        from Qt.QtCore import Qt
        layout.addWidget(self.push_button, 0, Qt.AlignLeft)

        self.fixed_widgets = []
        self.proportional_widgets = []
        w_pr_val, h_pr_val = 67, 67
        w_px_val, h_px_val = 1200, 750
        if size_scheme == "proportional":
            w_pr_val, h_pr_val = size_data
        elif size_scheme == "fixed":
            w_px_val, h_px_val = size_data
        from Qt.QtWidgets import QSpinBox, QWidget, QLabel
        self.nonmenu_widgets = QWidget()
        layout.addWidget(self.nonmenu_widgets)
        nonmenu_layout = QVBoxLayout()
        nonmenu_layout.setContentsMargins(0,0,0,0)
        nonmenu_layout.setSpacing(2)
        self.nonmenu_widgets.setLayout(nonmenu_layout)
        w_widgets = QWidget()
        nonmenu_layout.addWidget(w_widgets)
        w_layout = QHBoxLayout()
        w_widgets.setLayout(w_layout)
        w_layout.setContentsMargins(0,0,0,0)
        w_layout.setSpacing(2)
        self.w_proportional_spin_box = QSpinBox()
        self.w_proportional_spin_box.setMinimum(1)
        self.w_proportional_spin_box.setMaximum(100)
        self.w_proportional_spin_box.setValue(w_pr_val)
        self.w_proportional_spin_box.valueChanged.connect(lambda val, *, s=self: s.make_callback())
        w_layout.addWidget(self.w_proportional_spin_box)
        self.proportional_widgets.append(self.w_proportional_spin_box)
        self.w_fixed_spin_box = QSpinBox()
        self.w_fixed_spin_box.setMinimum(1)
        self.w_fixed_spin_box.setMaximum(1000000)
        self.w_fixed_spin_box.setValue(w_px_val)
        self.w_fixed_spin_box.valueChanged.connect(lambda val, *, s=self: s.make_callback())
        w_layout.addWidget(self.w_fixed_spin_box)
        self.fixed_widgets.append(self.w_fixed_spin_box)
        w_proportional_label = QLabel("% of screen width")
        w_layout.addWidget(w_proportional_label)
        self.proportional_widgets.append(w_proportional_label)
        w_fixed_label = QLabel("pixels wide")
        w_layout.addWidget(w_fixed_label)
        self.fixed_widgets.append(w_fixed_label)
        h_widgets = QWidget()
        nonmenu_layout.addWidget(h_widgets)
        h_layout = QHBoxLayout()
        h_widgets.setLayout(h_layout)
        h_layout.setContentsMargins(0,0,0,0)
        h_layout.setSpacing(2)
        self.h_proportional_spin_box = QSpinBox()
        self.h_proportional_spin_box.setMinimum(1)
        self.h_proportional_spin_box.setMaximum(100)
        self.h_proportional_spin_box.setValue(h_pr_val)
        self.h_proportional_spin_box.valueChanged.connect(lambda val, *, s=self: s.make_callback())
        h_layout.addWidget(self.h_proportional_spin_box)
        self.proportional_widgets.append(self.h_proportional_spin_box)
        self.h_fixed_spin_box = QSpinBox()
        self.h_fixed_spin_box.setMinimum(1)
        self.h_fixed_spin_box.setMaximum(1000000)
        self.h_fixed_spin_box.setValue(h_px_val)
        self.h_fixed_spin_box.valueChanged.connect(lambda val, *, s=self: s.make_callback())
        h_layout.addWidget(self.h_fixed_spin_box)
        self.fixed_widgets.append(self.h_fixed_spin_box)
        h_proportional_label = QLabel("% of screen height")
        h_layout.addWidget(h_proportional_label)
        self.proportional_widgets.append(h_proportional_label)
        h_fixed_label = QLabel("pixels high")
        h_layout.addWidget(h_fixed_label)
        self.fixed_widgets.append(h_fixed_label)

        self.current_fixed_size_label = QLabel()
        self.current_proportional_size_label = QLabel()
        nonmenu_layout.addWidget(self.current_fixed_size_label)
        nonmenu_layout.addWidget(self.current_proportional_size_label)
        self._update_current_size()

        self._show_appropriate_widgets()

    def _menu_cb(self, label):
        self.push_button.setText(label)
        self._show_appropriate_widgets()
        self.make_callback()

    def _show_appropriate_widgets(self):
        for w in self.proportional_widgets + self.fixed_widgets:
            w.hide()
        self.current_fixed_size_label.hide()
        self.current_proportional_size_label.hide()
        self.nonmenu_widgets.hide()
        size_scheme = self.push_button.text()
        if size_scheme == "proportional":
            self.nonmenu_widgets.show()
            for w in self.proportional_widgets:
                w.show()
            self.current_proportional_size_label.show()
        elif size_scheme == "fixed":
            self.nonmenu_widgets.show()
            for w in self.fixed_widgets:
                w.show()
            self.current_fixed_size_label.show()

    def _update_current_size(self, trig_name=None, wh=None):
        mw = getattr(self.session.ui, "main_window", None)
        if not mw:
            self.session.ui.triggers.add_handler('ready', self._update_current_size)
            return

        if wh is None:
            # this should only happen once...
            mw.triggers.add_handler('resized', self._update_current_size)
            window_width, window_height = mw.width(), mw.height()
        else:
            window_width, window_height = wh

        screen_geom = self.session.ui.primaryScreen().availableGeometry()
        screen_width, screen_height = screen_geom.width(), screen_geom.height()
        if not screen_width or not screen_height:
            return
        self.current_fixed_size_label.setText(
            "Current: %d wide, %d high" % (window_width, window_height))
        self.current_proportional_size_label.setText("Current: %d%% wide, %d%% high" % (
                int(100.0 * window_width / screen_width),
                int(100.0 * window_height / screen_height)))

def precise_target(session):
    sel_bonds = session.selection.items('bonds')
    if not sel_bonds:
        return 'a'
    sel_atoms = session.selection.items('atoms')
    if not sel_atoms:
        return 'b'
    a_bonds = sel_atoms[0].bonds
    if sel_bonds[0] - a_bonds:
        # some selected bonds have neither endpoint atom selected
        return 'ab'
    return 'a'
