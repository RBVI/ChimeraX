from .qt import QtCore, QtGui, QtOpenGL, QtWidgets, Qt

class Hydra_App(QtWidgets.QApplication):

    def __init__(self, argv, session):
        fix_qt_plugin_path()
        QtWidgets.QApplication.__init__(self, argv)
        self.session = session
        from os.path import join
        session.bin_dir = join(self.applicationDirPath(), '..', 'Resources', 'bin')
        self.setWindowIcon(icon('reo.png'))
        set_default_context(3, 2, QtOpenGL.QGLFormat.CoreProfile)
        self.main_window = Main_Window(self, session)

    # This is a virtual method of QApplication being used here to detect file open
    # requests such as dragging a file onto the application window.
    def event(self, e):
        if e.type() == QtCore.QEvent.FileOpen:
            path = e.file()
            from ...files.opensave import open_files
            open_files([path], self.session)
            self.session.main_window.show_graphics()
            return True
        else:
            return QtWidgets.QApplication.event(self, e)

class Main_Window(QtWidgets.QMainWindow):
    '''
    Main application window including graphics, toolbar, command line, status line,
    and scrolled text log.
    '''
    def __init__(self, app, session, parent=None):
        self._qapp = app        	# Needed for redrawing status line.
        self.session = session

        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle(self.tr("Hydra"))
        
        sb = QtWidgets.QStatusBar(self)
        self.setStatusBar(sb)
        self.status_update_interval = 0.2       # seconds
        self._last_status_update = 0

        class GraphicsArea(QtWidgets.QStackedWidget):
            def sizeHint(self):
                return QtCore.QSize(800,800)

        self._stack = st = GraphicsArea(self)
        from .graphicswindow import Graphics_Window
        g = Graphics_Window(session, st)
        self.view = g   # View is a base class of Graphics_Window
        st.addWidget(g.widget)

        class TextArea(QtWidgets.QTextBrowser):
            def keyPressEvent(self, event):
                if event.key() in (Qt.Key_Space, Qt.Key_Enter, Qt.Key_Return):
                    event.ignore()       # Make space and enter keys pass through to command-line.
                else:
                    QtWidgets.QTextBrowser.keyPressEvent(self, event)

        self._text = e = TextArea(st)

        # Create close button for text widget.
        self._close_text = ct = QtWidgets.QPushButton('X', e)
        ct.setStyleSheet("padding: 1px; min-width: 1em")
        ct.clicked.connect(lambda e: self.show_graphics())

        e.setReadOnly(True)
        e.anchorClicked.connect(self._anchor_callback)          # Handle clicks on anchors
        self._anchor_cb = None
        self._back_action = ba = QtWidgets.QAction(icon('back.png'), 'Go back in web browser', self)
        ba.triggered.connect(e.backward)
        self._forward_action = fa = QtWidgets.QAction(icon('forward.png'), 'Go forward in web browser', self)
        fa.triggered.connect(e.forward)

#        e.setAlignment(QtCore.Qt.AlignHCenter)
        # Use black background for text
#        p = QtGui.QPalette()
#        p.setColor(p.Text, QtGui.QColor(255,255,255))
#        p.setColor(p.Base, QtGui.QColor(0,0,0))
#        e.setPalette(p)
        st.addWidget(e)
        st.setCurrentWidget(g.widget)
        self.setCentralWidget(st)

        self._create_menus()
        self._create_toolbar()

        self._shortcuts_enabled = False
        self._command_line = cl = self._create_command_line()
        g.widget.setFocusProxy(cl)

        # Work around bug where initial window size limited to 2/3 of screen width and height
        # by qt adjustSize() routine.
#        lo = self.layout()
#        lo.setSizeConstraint(lo.SetFixedSize)

    # QMainWindow virtual function called when user resizes window.
    def resizeEvent(self, e):
        s = e.size()
        ct = self._close_text
        ct.move(s.width()-ct.width()-5,5)

    # Virtual QMainWindow method used to receive key stroke events.
    def keyPressEvent(self, event):

        k = event.key()
        if k == Qt.Key_Escape:
            self.enable_shortcuts(not self._shortcuts_enabled)
            return

        if self._shortcuts_enabled and (k == Qt.Key_Return or k == Qt.Key_Enter):
            self.enable_shortcuts(False)
            return

        self._command_line.event(event)

    def graphics_size(self):
        st = self._stack
        return (st.width(), st.height())

    def show_command_line(self, show):
        f = self._command_line_frame
        if show:
            f.show()
        else:
            f.hide()
    def set_command_line_text(self, cmd):
        cline = self._command_line
        cline.clear()
        cline.insert(cmd)

    def enable_shortcuts(self, enable):
        '''Interpret key strokes as shortcuts.  The command-line changes color to light green when in shortcut mode.'''
        color = 'rgb(230,255,230)' if enable else 'white'
        cl = self._command_line
        cl.setStyleSheet('QLineEdit {background: %s;}' % color)
        self._shortcuts_enabled = enable
        cl.setText('')

    def showing_graphics(self):
        return self._stack.currentWidget() == self.view.widget
    def show_graphics(self):
        self._stack.setCurrentWidget(self.view.widget)
        self.show_back_forward_buttons(False)

    def showing_text(self):
        return self._stack.currentWidget() == self._text
    def show_text(self, text = None, url = None, html = False, id = None, anchor_callback = None,
                  open_links = False, scroll_to_end = False):
        '''Show specified HTML in the main panel of the main window.  This html panel covers the graphics.'''
        t = self._text
        if not text is None:
            if html:
                t.setHtml(text)
            else:
                t.setPlainText(text)
        elif not url is None:
            t.setSource(QtCore.QUrl(url))

        self.text_id = id
        self._stack.setCurrentWidget(t)
        self._anchor_cb = anchor_callback
        t.setOpenLinks(open_links)
        self.show_back_forward_buttons(open_links)
        if scroll_to_end:
            sb = t.verticalScrollBar()
            sb.setValue(sb.maximum())
    def register_html_image_identifier(self, uri, qimage):
        d = self._text.document()
        from . import qt
        qt.register_html_image_identifier(d, uri, qimage)
    def show_back_forward_buttons(self, show):
        '''Display toolbar arrow buttons for going back or forward when users follows links in html panel.'''
        tb = self.toolbar
        if show:
            tb.insertAction(self.left_toolbar_action, self._back_action)
            tb.insertAction(self.left_toolbar_action, self._forward_action)
        else:
            tb.removeAction(self._back_action)
            tb.removeAction(self._forward_action)

    def show_status(self, msg, append = False):
        '''Write a message on the status line.'''
        sb = self.statusBar()
        if append:
            msg = str(sb.currentMessage()) + msg
        sb.showMessage(sb.tr(msg))

        # Repaint status line by entering event loop
        from time import time
        t = time()
        if t > self._last_status_update + self.status_update_interval:
            self._last_status_update = t
#            sb.repaint()        # Does not draw.  Redraw in case long wait before return to event loop
#            self._qapp.sendPostedEvents(sb)        # Does not draw.
#            sb.paintEvent(QtGui.QPaintEvent(sb.visibleRegion()))       # Crashes
#            return

            self.view.block_redraw()        # Avoid graphics redraw
            try:
                # TODO: exclude user input events drops key strokes and mouse events that will never
                #   get processed, Qt 5.2.  Documentation claims these events are not dropped.
                self._qapp.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                # TODO: Processing all events is unacceptable since data can be changed or deleted whenever
                #       a status message is shown. Need a way to repaint without processing events.
                # self._qapp.processEvents(QtCore.QEventLoop.AllEvents)
            finally:
                self.view.unblock_redraw()

    def _create_command_line(self):

        d = QtWidgets.QDockWidget('Command line', self)
        self._command_line_frame = w = QtWidgets.QWidget(d)
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
        cline.returnPressed.connect(self._command_entered)
        cline.textEdited.connect(self._command_text_changed)
        return cline

    def _create_menus(self):

#        self.menuBar = mb = QtWidgets.QMenuBar()
        mb = self.menuBar()

        from ...commands.shortcuts import standard_shortcuts
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
        
    def _create_toolbar(self):

# TODO: tooltips take too long to show and don't hide when mouse moves off button on Mac.
#   Tests show toolbar no longer gets mouse move events once tool tip is shown. QTBUG-26669
        self.toolbar = toolbar = QtWidgets.QToolBar('Toolbar', self)
        toolbar.setFocusPolicy(QtCore.Qt.NoFocus)
#        toolbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
#        toolbar.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        modes = QtWidgets.QActionGroup(toolbar)
        modes.setExclusive(True)
        a = self._add_shortcut_icon('move.png', 'Movement mouse mode', 'mv')
        a.setCheckable(True)
        a.setChecked(True)
        modes.addAction(a)
        self.left_toolbar_action = a
        a = self._add_shortcut_icon('move_h2o.png', 'Move selected mouse mode', 'mo')
        a.setCheckable(True)
        modes.addAction(a)
        a = self._add_shortcut_icon('contour.png', 'Adjust contour level mouse mode', 'ct')
        a.setCheckable(True)
        modes.addAction(a)
        a = self._add_shortcut_icon('cubearrow.png', 'Resize map mouse mode', 'Mp')
        a.setCheckable(True)
        modes.addAction(a)
        a = self._add_shortcut_icon('vseries.png', 'Volume series mouse mode', 'vs')
        a.setCheckable(True)
        modes.addAction(a)
        toolbar.addSeparator()

        self._add_shortcut_icon('rabbithat.png', 'Show/hide models', 'mp')
        self._add_shortcut_icon('cube-outline.png', 'Show map outline box', 'ob')
        self._add_shortcut_icon('icecube.png', 'Make map transparent', 'tt')
        toolbar.addSeparator()

        self._add_shortcut_icon('graphics.png', 'Show graphics window', 'gr')
        self._add_shortcut_icon('scenes.png', 'Show scenes', 'sc')
        self._add_shortcut_icon('log.png', 'Show command log', 'lg')
        self._add_shortcut_icon('commands.png', 'Show command history', 'ch')
        self._add_shortcut_icon('shortcut.png', 'List keyboard shortcuts', 'ks')
        self._add_shortcut_icon('book.png', 'Show manual', 'mn')
        toolbar.addSeparator()

        self._add_shortcut_icon('grid.png', 'Show recent files', 'rf')
        self._add_shortcut_icon('savesession.png', 'Save session', 'sv')

    def _add_shortcut_icon(self, icon_file, descrip, shortcut):

        a = QtWidgets.QAction(icon(icon_file), descrip + ' (%s)' % shortcut, self)
        ks = self.session.keyboard_shortcuts
        a.triggered.connect(lambda a,ks=ks,s=shortcut: ks.run_shortcut(s))
        self.toolbar.addAction(a)
        return a

    def _command_text_changed(self, text):

        if self._shortcuts_enabled:
            ks = self.session.keyboard_shortcuts
            if ks.try_shortcut(text):
                self._command_line.setText('')

    def _command_entered(self):
        cline = self._command_line
        text = cline.text()
        cline.selectAll()
        self.session.commands.run_command(text)

    def _anchor_callback(self, url):
        if self._anchor_cb:
            self._anchor_cb(url.toString(url.PreferLocalFile))

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
            event.ignore()      # Handled by Main_Window.keyPressEvent()
            return
        elif k == Qt.Key_Up:
            self.session.promote_selection()
        elif k == Qt.Key_Down:
            self.session.demote_selection()
        else:
            QtWidgets.QLineEdit.keyPressEvent(self, event)
        event.accept()

def icon(filename):
    from os.path import dirname, join, split
    dir = split(dirname(__file__))[0]
    path = join(dir, 'icons', filename)
    i = QtGui.QIcon(path)
    return i

def set_default_context(major_version, minor_version, profile):
    f = QtOpenGL.QGLFormat()
    f.setVersion(major_version, minor_version)
    f.setProfile(profile)
#    f.setStereo(True)
    QtOpenGL.QGLFormat.setDefaultFormat(f)

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

def set_window_size(session, width = None, height = None):

    mw = session.main_window
    gw,gh = mw.graphics_size()
    if width is None and height is None:
        from .. import show_status, show_info
        msg = 'Graphics size %d, %d' % (gw, gh)
        show_status(msg)
        show_info(msg)
    else:
        # Have to resize main window.  Main window will not resize for central graphics window.
        wpad, hpad = mw.width()-gw, mw.height()-gh
        mw.resize(width + wpad, height + hpad)
