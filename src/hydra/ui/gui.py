from .qt import QtCore, QtGui, QtOpenGL, QtWidgets

class MainWindow(QtWidgets.QMainWindow):
    '''
    Main application window including graphics, toolbar, command line, status line,
    and scrolled text log.
    '''
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle(self.tr("Hydra"))
        
        sb = QtWidgets.QStatusBar(self)
        self.setStatusBar(sb)
        self.status_update_interval = 0.2       # seconds
        self.last_status_update = 0

        class GraphicsArea(QtWidgets.QStackedWidget):
            def sizeHint(self):
                return QtCore.QSize(800,800)

        self.stack = st = GraphicsArea(self)
        from .view import View
        self.view = v = View(st)
        st.addWidget(v.widget)

#        self.text = e = QtGui.QTextEdit(st)
        self.text = e = QtWidgets.QTextBrowser(st)          # Handle clicks on anchors
        e.setReadOnly(True)
        e.anchorClicked.connect(self.anchor_callback)
        self.anchor_cb = None
        self.back_action = ba = QtWidgets.QAction(icon('back.png'), 'Go back in web browser', self)
        ba.triggered.connect(e.backward)
        self.forward_action = fa = QtWidgets.QAction(icon('forward.png'), 'Go forward in web browser', self)
        fa.triggered.connect(self.text.forward)

#        e.setAlignment(QtCore.Qt.AlignHCenter)
        # Use black background for text
#        p = QtGui.QPalette()
#        p.setColor(p.Text, QtGui.QColor(255,255,255))
#        p.setColor(p.Base, QtGui.QColor(0,0,0))
#        e.setPalette(p)
        st.addWidget(e)
        st.setCurrentWidget(v.widget)
        self.setCentralWidget(st)

        from . import shortcuts
        shortcuts.register_shortcuts(v)

        from . import commands
        commands.register_commands()

        self.create_toolbar()

        self.create_command_line()

        # Work around bug where initial window size limited to 2/3 of screen width and height
        # by qt adjustSize() routine.
#        lo = self.layout()
#        lo.setSizeConstraint(lo.SetFixedSize)

    def create_command_line(self):

        d = QtWidgets.QDockWidget('Command line', self)
        self.command_line = cline = QtWidgets.QLineEdit(d)
        cline.setFocusPolicy(QtCore.Qt.ClickFocus)
        d.setWidget(cline)
        d.setTitleBarWidget(QtWidgets.QWidget(d))   # No title bar
        d.setFeatures(d.NoDockWidgetFeatures)   # No close button
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d)
        cline.returnPressed.connect(self.command_entered)

    def focus_on_command_line(self):
        cline = self.command_line
#        self.view.setFocus(QtCore.Qt.OtherFocusReason)
        cline.setFocus(QtCore.Qt.OtherFocusReason)
#        self.releaseKeyboard()
#        cline.activateWindow()
#        cline.setEnabled(True)
#        self.view.setFocus(QtCore.Qt.MouseFocusReason)
#       cline.setFocus(QtCore.Qt.MouseFocusReason)

    def create_toolbar(self):

# TODO: tooltips take too long to show and don't hide when mouse moves off button on Mac.
#   Tests show toolbar no longer gets mouse move events once tool tip is shown. QTBUG-26669
        self.toolbar = toolbar = QtWidgets.QToolBar('Toolbar', self)
        toolbar.setFocusPolicy(QtCore.Qt.NoFocus)
#        toolbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
#        toolbar.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        a = self.add_shortcut_icon('select.png', 'Select model mouse mode', 'sl')
        self.left_toolbar_action = a
        self.add_shortcut_icon('contour.png', 'Adjust contour level mouse mode', 'ct')
        self.add_shortcut_icon('cubearrow.png', 'Resize map mouse mode', 'mp')
        self.add_shortcut_icon('move_h2o.png', 'Move selected mouse mode', 'mo')
        self.add_shortcut_icon('rotate_h2o.png', 'Rotate selected mouse mode', 'ro')
        toolbar.addSeparator()

        self.add_shortcut_icon('rabbithat.png', 'Show/hide models', 'sh')
        self.add_shortcut_icon('cube-outline.png', 'Show map outline box', 'ob')
        self.add_shortcut_icon('icecube.png', 'Make map transparent', 't5')
        toolbar.addSeparator()

        self.add_shortcut_icon('graphics.png', 'Show graphics window', 'gr')
        self.add_shortcut_icon('scenes.png', 'Show scenes', 'sc')
        self.add_shortcut_icon('log.png', 'Show command log', 'lg')
        self.add_shortcut_icon('commands.png', 'Show command history', 'ch')
        self.add_shortcut_icon('shortcut.png', 'List keyboard shortcuts', 'ks')
        self.add_shortcut_icon('book.png', 'Show manual', 'mn')
        toolbar.addSeparator()

        self.add_shortcut_icon('grid.png', 'Show recent sessions', 'rs')
        self.add_shortcut_icon('savesession.png', 'Save session', 'sv')

    def add_shortcut_icon(self, icon_file, descrip, shortcut):

        a = QtWidgets.QAction(icon(icon_file), descrip + ' (%s)' % shortcut, self)
        from .shortcuts import keyboard_shortcuts as ks
        a.triggered.connect(lambda a,ks=ks,s=shortcut: ks.run_shortcut(s))
        self.toolbar.addAction(a)
        return a

    def keyPressEvent(self, event):

#        print('got key', repr(event.text()))
#        cline = self.command_line
#        if cline.hasFocus():
            # TODO: Bug in Mac Qt 5.0.2 that calling setFocus() on command line when using
            # keyboard shortcut "cl" switches focus but does not deliver key events to the
            # command-line.  This code works around the problem.
#            if str(event.text()) != '\r':
#                cline.keyPressEvent(event)
#        else:
            if str(event.text()) == '\r':
                return
            from .shortcuts import keyboard_shortcuts as ks
            ks.key_pressed(event)

#        w = self.toolbar.widgetForAction(a)  # QToolButton
# TODO: show tool tip immediately

    def command_entered(self):
        cline = self.command_line
        text = cline.text()
        cline.selectAll()
        from . import commands
        commands.run_command(text)

    def showing_text(self):
        return self.stack.currentWidget() == self.text
    def show_text(self, text = None, html = False, id = None, anchor_callback = None,
                  open_links = False, scroll_to_end = False):
        t = self.text
        if not text is None:
            if html:
                t.setHtml(text)
            else:
                t.setPlainText(text)
        self.text_id = id
        self.stack.setCurrentWidget(t)
        self.anchor_cb = anchor_callback
        t.setOpenLinks(open_links)
        self.show_back_forward_buttons(open_links)
        if scroll_to_end:
            sb = t.verticalScrollBar()
            sb.setValue(sb.maximum())

    def anchor_callback(self, url):
        if self.anchor_cb:
            self.anchor_cb(url)
    def show_back_forward_buttons(self, show):
        tb = self.toolbar
        if show:
            tb.insertAction(self.left_toolbar_action, self.back_action)
            tb.insertAction(self.left_toolbar_action, self.forward_action)
        else:
            tb.removeAction(self.back_action)
            tb.removeAction(self.forward_action)
    def showing_graphics(self):
        return self.stack.currentWidget() == self.view.widget
    def show_graphics(self):
        self.stack.setCurrentWidget(self.view.widget)

    def show_status(self, msg, append = False):
        sb = self.statusBar()
        if append:
            msg = str(sb.currentMessage()) + msg
        sb.showMessage(sb.tr(msg))
#        sb.repaint()        # Does not draw.  Redraw in case long wait before return to event loop

        # Repaint status line by entering event loop
        from time import time
        t = time()
        if t > self.last_status_update + self.status_update_interval:
            self.last_status_update = t
            global app
            self.view.block_redraw()        # Avoid graphics redraw
            try:
                app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            finally:
                self.view.unblock_redraw()

def icon(filename):
    from os.path import dirname, join
    dir = dirname(__file__)
    path = join(dir, 'icons', filename)
    i = QtGui.QIcon(path)
    return i

def set_default_context(major_version, minor_version, profile):
    f = QtOpenGL.QGLFormat()
    f.setVersion(major_version, minor_version)
    f.setProfile(profile)
#    f.setStereo(True)
    QtOpenGL.QGLFormat.setDefaultFormat(f)

app = None
main_window = None
def show_main_window():
    '''
    Create the application main window and start the event loop.
    '''
    set_default_context(3, 2, QtOpenGL.QGLFormat.CoreProfile)
    import sys
    global app
    app = QtWidgets.QApplication(sys.argv)
#    d = app.desktop()
#    print('screen count', d.screenCount())
#    for s in range(d.screenCount()):
#        g = d.screenGeometry(s)
#        print('screen', s, 'size', g.width(), g.height(), 'top left', g.top(), g.left())
    # Seting icon does not work, mac qt 5.0.2.
    # Get Python launcher rocket icon in Dock.
    app.setWindowIcon(icon('reo.png'))
    w = MainWindow()
    global main_window
    main_window = w
#    w.view.setFocus(QtCore.Qt.OtherFocusReason)       # Get keyboard events on startup
    w.show()
    from ..file_io.history import history
    history.show_thumbnails()
    enable_exception_logging()
    redirect_stdout()
    status = app.exec_()
#    from . import leap
#    leap.quit_leap(w.view)
    history.write_history()
    sys.exit(status)

def show_status(msg, append = False):
    '''
    Show a status message at the bottom of the main window.
    '''
    main_window.show_status(msg, append)

def show_info(msg, color = None):
    '''
    Write information to the log window, typically command output.
    '''
    log_message(msg, color)

class Log:
    def __init__(self):
        self.html_text = ''
        self.image_number = 1
        self.image_directory = None
    def show(self):
        from .gui import main_window as mw
        if mw.showing_text() and mw.text_id == 'log':
            mw.show_graphics()
        else:
#            mw.show_text(self.html_text, html = True, id = "log", open_links = True)
            mw.show_text(self.html_text, html = True, id = "log", scroll_to_end = True)
    def append(self, text, color = None, html = False):
        if html:
            htext = text
        else:
            style = '' if color is None else ' style="color:%s;"' % color
            import cgi
            etext = cgi.escape(text)
            htext = '<pre%s>%s</pre>\n' % (style,etext)
        self.html_text += htext
    def insert_graphics_image(self):
        self.schedule_image_capture()
    def schedule_image_capture(self):
        global main_window
        v = main_window.view
        v.add_rendered_frame_callback(self.capture_image)
    def capture_image(self, show_height = 128, format = 'JPG'):
        global main_window
        v = main_window.view
        v.remove_rendered_frame_callback(self.capture_image)
        i = v.image()
        filename = 'img%04d.%s' % (self.image_number, format.lower())
        htmlfile = 'img%04d.html' % (self.image_number,)
        self.image_number += 1
        ldir = self.image_log_directory()
        from os.path import join
        path = join(ldir, filename)
        i.save(path, format)
#        hpath = join(ldir, htmlfile)
#        f = open(hpath, 'w')
#        f.write('<html><body><img src="%s"></body></html>\n' % path)
#        f.close()
# TODO: Shows binary text instead of image clicking link to jpg file.
#        htext = '<br><a href="%s" type="image/jpeg"><img src="%s" height=64></a><br>\n' % (path, path)
#        htext = '<br><a href="%s"><img src="%s" height=%d></a><br>\n' % (hpath, path, show_height)
        htext = '<br><img src="%s" height=%d><br>\n' % (path, show_height)
        self.html_text += htext
    def image_log_directory(self):
        if self.image_directory is None:
            from ..file_io import history
            d = history.user_settings_path()
            import tempfile
            self.image_directory = tempfile.TemporaryDirectory(dir = d)
        return self.image_directory.name

cmd_log = Log()
def show_log():
    global cmd_log
    cmd_log.show()

def log_message(msg, color = None, html = False):
    '''
    Write a message to the log window.  It can be plain text or html.
    '''
    global cmd_log
    cmd_log.append(msg, color, html)

def log_image():
    '''
    Take a snapshot of the graphics window and insert a thumbnail of it in the log window.
    '''
    global cmd_log
    cmd_log.insert_graphics_image()

def redirect_stdout():
    import sys
    sys.stdout_orig = sys.stdout
    class Log_Output:
        def __init__(self):
            self.text = ''
        def write(self, text):
            self.text += text
            if text.endswith('\n'):
                log_message(self.text.rstrip())
                self.text = ''
        def flush(self):
            if self.text:
                log_message(self.text.rstrip())
                self.text = ''
    sys.stdout = Log_Output()

def enable_exception_logging():
    import sys
    sys.excepthook = log_exception

def log_exception(type, value, traceback):
    from traceback import format_exception
    lines = format_exception(type, value, traceback)
    import cgi
    elines = tuple(cgi.escape(line) for line in lines)
    tb = '<p style="color:#A00000;">\n%s</p>' % '<br><br>'.join(elines)
    log_message(tb, html = True)
    show_log()
