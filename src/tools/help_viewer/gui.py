# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# HelpUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
#
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance
import weakref

_targets = weakref.WeakValueDictionary()

class HelpUI(ToolInstance):

    SESSION_ENDURING = False    # default

    def __init__(self, session, bundle_info, target):
        ToolInstance.__init__(self, session, bundle_info, target)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Help UI"
        # in this case), so only override if different name desired
        from chimerax import app_dirs
        self.display_name = "%s Help Viewer" % app_dirs.appname
        self.target = target
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        self.on_page = None
        self.home_page = None
        # UI content code
        from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QAction, QLineEdit
        from PyQt5.QtGui import QIcon
        self.toolbar = tb = QToolBar()
        layout = QVBoxLayout()
        layout.addWidget(tb)
        parent.setLayout(layout)
        style = tb.style()
        self.back = QAction(style.standardIcon(style.SP_ArrowBack), "Back to previous page", tb)
        self.back.triggered.connect(self.page_back)
        self.back.setEnabled(False)
        tb.addAction(self.back)
        self.forward = QAction(style.standardIcon(style.SP_ArrowForward), "Next page", tb)
        self.forward.triggered.connect(self.page_forward)
        self.forward.setEnabled(False)
        tb.addAction(self.forward)
        import os.path
        d = os.path.dirname(__file__)
        icon_path = os.path.join(d, "home.png")
        self.home = QAction(QIcon(icon_path), "Home page", tb)
        self.home.triggered.connect(self.go_home)
        self.home.setEnabled(False)
        tb.addAction(self.home)
        icon_path = os.path.join(d, "zoom-plus.png")
        self.zoom_in = QAction(QIcon(icon_path), "Zoom in", tb)
        self.zoom_in.triggered.connect(self.page_zoom_in)
        font = self.zoom_in.font()
        font.setPointSize(48)
        self.zoom_in.setFont(font)
        tb.addAction(self.zoom_in)
        icon_path = os.path.join(d, "zoom-minus.png")
        self.zoom_out = QAction(QIcon(icon_path), "Zoom out", tb)
        self.zoom_out.setFont(font)
        self.zoom_out.triggered.connect(self.page_zoom_out)
        tb.addAction(self.zoom_out)
        self.search = QLineEdit("search")
        self.search.selectAll()
        tb.addWidget(self.search)

        from PyQt5.QtWebEngineWidgets import QWebEngineView
        class HelpWebView(QWebEngineView):
            def __init__(self, ses=session, bi=bundle_info):
                self.session = ses
                self.bundle_info = bi
                QWebEngineView.__init__(self)

            def createWindow(self, win_type):
                help_ui = HelpUI(self.session, self.bundle_info)
                return help_ui.help_window
        self.help_window = HelpWebView()
        layout.addWidget(self.help_window)
        def link_clicked(qurl, nav_type, is_main_frame):
            self.link_clicked(qurl)
            return False
        self.help_window.page().acceptNavigationRequest = link_clicked
        """
        from PyQt5.QtGui import QDesktopServices
        def t(*args):
            import sys
            print("url handler args", args, file=sys.__stderr__)
        #QDesktopServices.setUrlHandler("cxcmd", t)
        #QDesktopServices.setUrlHandler("cxcmd", self.link_clicked)
        QDesktopServices.setUrlHandler("cxcmd", self, "link_clicked")
        """
        self.help_window.loadFinished.connect(self.page_loaded)
        self.help_window.titleChanged.connect(
            lambda title: setattr(self.tool_window, 'title', title))
        self.search.returnPressed.connect(lambda s=self.search, hw=self.help_window:
            hw.findText(s.text()))

        self.tool_window.manage(placement=None)

    def show(self, url, set_home=False):
        from urllib.parse import urlparse, urlunparse
        parts = urlparse(url)
        url = urlunparse(parts)  # canonicalize
        self.on_page = url
        if set_home or not self.home_page:
            self.help_window.history().clear()
            self.home_page = url
            self.home.setEnabled(True)
            self.back.setEnabled(False)
            self.forward.setEnabled(False)
        from PyQt5.QtCore import QUrl
        self.help_window.setUrl(QUrl(url))

    # wx event handling

    def page_back(self, checked):
        self.help_window.history().back()

    def on_forward(self, event):
        self.help_window.GoForward()

    def page_forward(self, checked):
        self.help_window.history().forward()

    def on_home(self, event):
        self.show(self.home_page)
    go_home = on_home

    def page_zoom_in(self, checked):
        self.help_window.setZoomFactor(1.25 * self.help_window.zoomFactor())

    def page_zoom_out(self, checked):
        self.help_window.setZoomFactor(0.8 * self.help_window.zoomFactor())

    def delete(self):
        ToolInstance.delete(self)
        try:
            del _targets[self.target]
        except:
            pass


    def on_navigated(self, event):
        self.toolbar.EnableTool(self.back.GetId(),
                                self.help_window.CanGoBack())
        self.toolbar.EnableTool(self.forward.GetId(),
                                self.help_window.CanGoForward())

    def page_loaded(self, okay):
        page = self.help_window.page()
        history = self.help_window.history()
        self.back.setEnabled(history.canGoBack())
        self.forward.setEnabled(history.canGoForward())

    def on_navigating(self, event):
        session = self.session
        # Handle event
        url = event.GetURL()
        from urllib.parse import urlparse, urlunparse, unquote
        parts = urlparse(url)
        url = urlunparse(parts)  # canonicalize
        if url == self.on_page:
            # Do not Veto, because it stops page from being shown
            return
        url = unquote(url)
        event.Veto()
        if parts.scheme in ('cxcmd', 'help', 'file', 'http'):
            from .cmd import help
            help(session, topic=url, target=self.target)
            return
        # unknown scheme
        session.logger.error("Unknown URL scheme: '%s'" % parts.scheme)

    def link_clicked(self, qurl, *args):
        import sys
        print("link_clicked!", file=sys.__stderr__)
        print("link_clicked:", repr(qurl), args, file=sys.__stderr__)
        session = self.session
        if qurl.scheme() in ('cxcmd', 'help', 'file', 'http'):
            from .cmd import help
            help(session, topic=qurl.url(), target=self.target)
            return
        # unknown scheme
        session.logger.error("Unknown URL scheme: '%s'" % parts.scheme)

    def on_title_change(self, event):
        new_title = self.help_window.CurrentTitle
        self.tool_window.set_title(new_title)

    def on_new_window(self, event):
        session = self.session
        event.Veto()
        url = event.GetURL()
        target = event.GetTarget()
        # TODO: figure out why target is always None
        if not target:
            target = url
        use_help_viewer = url.startswith('help:') or target.startswith('help:')
        if use_help_viewer:
            from .cmd import help
            help(session, topic=url, target=target)
        else:
            import webbrowser
            webbrowser.open(url)

    @classmethod
    def get_viewer(cls, session, target=None):
        # TODO: reenable multiple target windows
        # if target is None:
        if 1:
            target = 'help'
        if target in _targets:
            return _targets[target]
        bundle_info = session.toolshed.find_bundle('help_viewer')
        viewer = HelpUI(session, bundle_info, target)
        _targets[target] = viewer
        return viewer
