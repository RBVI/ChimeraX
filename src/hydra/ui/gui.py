from .qt import QtCore, QtGui, QtOpenGL, QtWidgets, Qt

class MainWindow(QtWidgets.QMainWindow):
    '''
    Main application window including graphics, toolbar, command line, status line,
    and scrolled text log.
    '''
    def __init__(self, app, session, parent=None):
        self.app = app
        self.session = session

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
        self.view = v = View(session, st)
        st.addWidget(v.widget)

        class TextArea(QtWidgets.QTextBrowser):
            def keyPressEvent(self, event):
                if event.key() in (Qt.Key_Space, Qt.Key_Enter, Qt.Key_Return):
                    event.ignore()       # Make space and enter keys pass through to command-line.
                else:
                    QtWidgets.QTextBrowser.keyPressEvent(self, event)

        self.text = e = TextArea(st)

        # Create close button for text widget.
        self.close_text = ct = QtWidgets.QPushButton('X', e)
        ct.setStyleSheet("padding: 1px; min-width: 1em")
        ct.clicked.connect(lambda e: self.show_graphics())

        e.setReadOnly(True)
        e.anchorClicked.connect(self.anchor_callback)          # Handle clicks on anchors
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

        self.create_menus()
        self.create_toolbar()

        self.shortcuts_enabled = False
        self.command_line = cl = self.create_command_line()
        v.widget.setFocusProxy(cl)

        # Work around bug where initial window size limited to 2/3 of screen width and height
        # by qt adjustSize() routine.
#        lo = self.layout()
#        lo.setSizeConstraint(lo.SetFixedSize)

    def resizeEvent(self, e):
        s = e.size()
        ct = self.close_text
        ct.move(s.width()-ct.width()-5,5)

    def create_command_line(self):

        d = QtWidgets.QDockWidget('Command line', self)
        self.command_line_frame = w = QtWidgets.QWidget(d)
        hbox = QtWidgets.QHBoxLayout(w)
        hbox.setContentsMargins(0,0,0,0)
        t = QtWidgets.QLabel(' Command', w)
        hbox.addWidget(t)
        cline = Command_Line(w, self.session)
        hbox.addWidget(cline)
        w.setLayout(hbox)
        cline.setFocus(QtCore.Qt.OtherFocusReason)      # Set the initial focus to the command-line
        d.setWidget(w)
        d.setTitleBarWidget(QtWidgets.QWidget(d))   # No title bar
        d.setFeatures(d.NoDockWidgetFeatures)   # No close button
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d)
        cline.returnPressed.connect(self.command_entered)
        cline.textEdited.connect(self.command_text_changed)
        return cline

    def show_command_line(self, show):
        f = self.command_line_frame
        if show:
            f.show()
        else:
            f.hide()

    def focus_on_command_line(self):
        cline = self.command_line
        cline.setFocus(QtCore.Qt.OtherFocusReason)

    def create_menus(self):

#        self.menuBar = mb = QtWidgets.QMenuBar()
        mb = self.menuBar()

        from .shortcuts import standard_shortcuts
        scuts = standard_shortcuts(self.session)[0]
        mnames = []
        for sc in scuts:
            m = sc.menu
            if m and not m in mnames:
                mnames.append(m)
        menus = {}
        for mname in mnames:
            menus[mname] = mb.addMenu(mname)

        ks = self.session.keyboard_shortcuts
        for sc in scuts:
            m = sc.menu
            if m:
                a = QtWidgets.QAction(sc.description, self)
                a.triggered.connect(lambda a,ks=ks,s=sc.key_seq: ks.run_shortcut(s))
                menus[m].addAction(a)
                if sc.menu_separator:
                    menus[m].addSeparator()

#            a.setCheckable(True)
        
    def create_toolbar(self):

# TODO: tooltips take too long to show and don't hide when mouse moves off button on Mac.
#   Tests show toolbar no longer gets mouse move events once tool tip is shown. QTBUG-26669
        self.toolbar = toolbar = QtWidgets.QToolBar('Toolbar', self)
        toolbar.setFocusPolicy(QtCore.Qt.NoFocus)
#        toolbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
#        toolbar.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        modes = QtWidgets.QActionGroup(toolbar)
        modes.setExclusive(True)
        a = self.add_shortcut_icon('move.png', 'Movement mouse mode', 'mv')
        a.setCheckable(True)
        a.setChecked(True)
        modes.addAction(a)
        self.left_toolbar_action = a
        a = self.add_shortcut_icon('move_h2o.png', 'Move selected mouse mode', 'mo')
        a.setCheckable(True)
        modes.addAction(a)
        a = self.add_shortcut_icon('contour.png', 'Adjust contour level mouse mode', 'ct')
        a.setCheckable(True)
        modes.addAction(a)
        a = self.add_shortcut_icon('cubearrow.png', 'Resize map mouse mode', 'Mp')
        a.setCheckable(True)
        modes.addAction(a)
        a = self.add_shortcut_icon('vseries.png', 'Volume series mouse mode', 'vs')
        a.setCheckable(True)
        modes.addAction(a)
        toolbar.addSeparator()

        self.add_shortcut_icon('rabbithat.png', 'Show/hide models', 'mp')
        self.add_shortcut_icon('cube-outline.png', 'Show map outline box', 'ob')
        self.add_shortcut_icon('icecube.png', 'Make map transparent', 'tt')
        toolbar.addSeparator()

        self.add_shortcut_icon('graphics.png', 'Show graphics window', 'gr')
        self.add_shortcut_icon('scenes.png', 'Show scenes', 'sc')
        self.add_shortcut_icon('log.png', 'Show command log', 'lg')
        self.add_shortcut_icon('commands.png', 'Show command history', 'ch')
        self.add_shortcut_icon('shortcut.png', 'List keyboard shortcuts', 'ks')
        self.add_shortcut_icon('book.png', 'Show manual', 'mn')
        toolbar.addSeparator()

        self.add_shortcut_icon('grid.png', 'Show recent files', 'rf')
        self.add_shortcut_icon('savesession.png', 'Save session', 'sv')

    def add_shortcut_icon(self, icon_file, descrip, shortcut):

        a = QtWidgets.QAction(icon(icon_file), descrip + ' (%s)' % shortcut, self)
        ks = self.session.keyboard_shortcuts
        a.triggered.connect(lambda a,ks=ks,s=shortcut: ks.run_shortcut(s))
        self.toolbar.addAction(a)
        return a

    def enable_shortcuts(self, enable):
        color = 'rgb(230,255,230)' if enable else 'white'
        cl = self.command_line
        cl.setStyleSheet('QLineEdit {background: %s;}' % color)
        self.shortcuts_enabled = enable
        cl.setText('')

    def keyPressEvent(self, event):

        k = event.key()
        if k == Qt.Key_Escape:
            self.enable_shortcuts(not self.shortcuts_enabled)
            return

        if self.shortcuts_enabled and (k == Qt.Key_Return or k == Qt.Key_Enter):
            self.enable_shortcuts(False)
            return

        self.command_line.event(event)

    def command_text_changed(self, text):

        if self.shortcuts_enabled:
            ks = self.session.keyboard_shortcuts
            if ks.try_shortcut(text):
                self.command_line.setText('')

    def command_entered(self):
        cline = self.command_line
        text = cline.text()
        cline.selectAll()
        self.session.commands.run_command(text)

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
        self.show_back_forward_buttons(False)

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
            self.view.block_redraw()        # Avoid graphics redraw
            try:
                self.app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            finally:
                self.view.unblock_redraw()

class Command_Line(QtWidgets.QLineEdit):

    def __init__(self, parent, session):
        self.session = session
        QtWidgets.QLineEdit.__init__(self, parent)

    def keyPressEvent(self, event):
        t = event.text()
        k = event.key()
        ctrlp = b'\x10'.decode('utf-8')
        ctrln = b'\x0e'.decode('utf-8')
        ctrlk = b'\x0b'.decode('utf-8')
        ctrlb = b'\x02'.decode('utf-8')
        ctrlf = b'\x06'.decode('utf-8')
        from .qt import Qt
        if t in (ctrlp, ctrln):
            s = self.session
            s.main_window.enable_shortcuts(False)
            if t == ctrlp:
                s.commands.history.show_previous_command()
            elif t == ctrln:
                s.commands.history.show_next_command()
        elif t in (ctrlk, ctrlb, ctrlf):
            if t == ctrlk:
                self.setText(self.text()[:self.cursorPosition()])
            elif t == ctrlb:
                self.cursorBackward(False)
            elif t == ctrlf:
                self.cursorForward(False)
        elif k == Qt.Key_Escape:
            event.ignore()      # Handled by MainWindow.keyPressEvent()
            return
        elif k == Qt.Key_Up:
            self.session.promote_selection()
        elif k == Qt.Key_Down:
            self.session.demote_selection()
        else:
            QtWidgets.QLineEdit.keyPressEvent(self, event)
        event.accept()

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

class Hydra_App(QtWidgets.QApplication):

    def __init__(self, argv, session):
        fix_qt_plugin_path()
        QtWidgets.QApplication.__init__(self, argv)
        self.session = session
        self.setWindowIcon(icon('reo.png'))
        set_default_context(3, 2, QtOpenGL.QGLFormat.CoreProfile)
        self.main_window = MainWindow(self, session)

    def event(self, e):
        if e.type() == QtCore.QEvent.FileOpen:
            path = e.file()
            from ..file_io.opensave import open_files
            open_files([path], self.session)
            self.session.main_window.show_graphics()
            return True
        else:
            return QtWidgets.QApplication.event(self, e)

def fix_qt_plugin_path():
    # Remove plugin location set in QtCore library (qt_plugpath in binary) which points to build location
    # instead of install location.  Messes up application menu on Mac when run on build machine if build
    # path is used, and gives numerous warnings to start-up shell.  Installed Qt plugins are in PyQt5/plugins.
    libpaths = [p for p in QtCore.QCoreApplication.libraryPaths()
                if not str(p).endswith('plugins') or str(p).endswith('PyQt5/plugins')]
    QtCore.QCoreApplication.setLibraryPaths(libpaths)

def start_event_loop(app):
    status = app.exec_()
    return status

class Log:
    '''
    Log window for command output.
    '''
    def __init__(self, main_window):
        self.main_window = main_window
        self.html_text = ''
        self.thumbnail_size = 128       # Pixels
        self.keep_images = []
    def show(self, toggle = True):
        mw = self.main_window
        if mw.showing_text() and mw.text_id == 'log' and toggle:
            mw.show_graphics()
        else:
#            mw.show_text(self.html_text, html = True, id = "log", open_links = True)
            mw.show_text(self.html_text, html = True, id = "log", scroll_to_end = True)
    def log_message(self, text, color = None, html = False):
        if html:
            htext = text
        else:
            style = '' if color is None else ' style="color:%s;"' % color
            import cgi
            etext = cgi.escape(text)
            htext = '<pre%s>%s</pre>\n' % (style,etext)
        self.html_text += htext

    def insert_graphics_image(self, format = 'JPG'):
        mw = self.main_window
        v = mw.view
        s = self.thumbnail_size
        qi = v.image(s,s)
        # If we don't keep a reference to images, then displaying them causes a crash.
        self.keep_images.append(qi)
        n = len(self.keep_images)
        d = mw.text.document()
        uri = "file://image%d" % (n,)
        d.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(uri), qi)
        htext = '<br><img src="%s"><br>\n' % (uri,)
        self.html_text += htext

    def exceptions_to_log(self):
        import sys
        sys.excepthook = self.log_exception

    def log_exception(self, type, value, traceback):
        from traceback import format_exception
        lines = format_exception(type, value, traceback)
        import cgi
        elines = tuple(cgi.escape(line) for line in lines)
        tb = '<p style="color:#A00000;">\n%s</p>' % '<br><br>'.join(elines)
        self.log_message(tb, html = True)
        self.show(toggle = False)

    def stdout_to_log(self):
        import sys
        sys.stdout_orig = sys.stdout
        sys.stdout = self.output_stream()

    def output_stream(self):
        class Log_Output_Stream:
            def __init__(self, log):
                self.log = log
                self.text = ''
            def write(self, text):
                self.text += text
                if text.endswith('\n'):
                    self.log.log_message(self.text.rstrip())
                    self.text = ''
            def flush(self):
                if self.text:
                    self.log.log_message(self.text.rstrip())
                    self.text = ''
        return Log_Output_Stream(self)

def window_size_command(cmdname, args, session):

    from .commands import int_arg, parse_arguments
    req_args = ()
    opt_args = ((('width', int_arg),
                 ('height', int_arg),))
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    set_window_size(session, **kw)

def set_window_size(session, width = None, height = None):

    mw = session.main_window
    g = mw.stack
    if width is None and height is None:
        from . import show_status, show_info
        msg = 'Graphics size %d, %d' % (g.width(), g.height())
        show_status(msg)
        show_info(msg)
    else:
        # Have to resize main window.  Main window will not resize for central graphics window.
        wpad, hpad = mw.width()-g.width(), mw.height()-g.height()
        mw.resize(width + wpad, height + hpad)
