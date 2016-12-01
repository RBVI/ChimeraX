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

    def __init__(self, session, target):
        tool_name = "Help Viewer"
        ToolInstance.__init__(self, session, tool_name)
        from chimerax import app_dirs
        self.target = target
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        self.on_page = None
        self.home_page = None
        # UI content code
        from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QAction, QLineEdit
        from PyQt5.QtGui import QIcon
        # from PyQt5.QtCore import Qt
        self.toolbar = tb = QToolBar()
        # tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        layout = QVBoxLayout()
        layout.addWidget(tb)
        parent.setLayout(layout)
        import os.path
        icon_dir = os.path.dirname(__file__)
            # attribute, text, tool tip, callback, enabled
        buttons = (
            ( "back", "Back", "Back to previous page", self.page_back, False ),
            ( "forward", "Forward", "Next page", self.page_forward, False ),
            ( "home", "Home", "Home page", self.page_home, True ),
            ( "zoom_in", "Zoom in", "Zoom in", self.page_zoom_in, True ),
            ( "zoom_out", "Zoom out", "Zoom out", self.page_zoom_out, True ),
            ( "reload", "Reload", "Reload page", self.page_reload, True ),
        )
        for attribute, text, tooltip, callback, enabled in buttons:
            icon_path = os.path.join(icon_dir, "%s.png" % attribute)
            setattr(self, attribute, QAction(QIcon(icon_path), text, tb))
            a = getattr(self, attribute)
            a.setToolTip(tooltip)
            a.triggered.connect(callback)
            a.setEnabled(enabled)
            tb.addAction(a)

        self.search = QLineEdit("search")
        self.search.selectAll()
        tb.addWidget(self.search)

        from chimerax.core.ui.widgets import HtmlView
        class HelpWebView(HtmlView):

            def __init__(self, parent, link_clicked_cb, ses=session):
                super().__init__(parent, schemes=['help', 'cxcmd'],
                                 interceptor=link_clicked_cb)
                self.session = ses

            def createWindow(self, win_type):
                # win_type is window, tab, dialog, backgroundtab
                help_ui = HelpUI.get_viewer(self.session)  # TODO: target
                return help_ui.help_window
        self.help_window = HelpWebView(parent, self.link_clicked)
        layout.addWidget(self.help_window)
        self.help_window.loadFinished.connect(self.page_loaded)
        self.help_window.titleChanged.connect(self.title_changed)
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
            self.back.setEnabled(False)
            self.forward.setEnabled(False)
        from PyQt5.QtCore import QUrl
        self.help_window.setUrl(QUrl(url))

    def page_back(self, checked):
        self.help_window.history().back()

    def page_forward(self, checked):
        self.help_window.history().forward()

    def page_home(self, checked):
        self.show(self.home_page)

    def page_zoom_in(self, checked):
        self.help_window.setZoomFactor(1.25 * self.help_window.zoomFactor())

    def page_zoom_out(self, checked):
        self.help_window.setZoomFactor(0.8 * self.help_window.zoomFactor())

    def page_reload(self, checked):
        self.help_window.reload()

    def delete(self):
        ToolInstance.delete(self)
        try:
            del _targets[self.target]
        except:
            pass

    def page_loaded(self, okay):
        page = self.help_window.page()
        history = self.help_window.history()
        self.back.setEnabled(history.canGoBack())
        self.forward.setEnabled(history.canGoForward())

    def title_changed(self, title):
        self.tool_window.title = title

    def link_clicked(self, request_info, *args):
        session = self.session
        qurl = request_info.requestUrl()
        scheme = qurl.scheme()
        if scheme in ('https', 'http', 'file'):
            return
        if scheme in ('cxcmd', 'help'):
            from .cmd import help
            #help(session, topic=qurl.url(), target=self.target)
            session.ui.thread_safe(help, session, topic=qurl.url(), target=self.target)
            return
        # unknown scheme
        session.logger.error("Unknown URL scheme in help viewer: '%s'" % scheme)

    @classmethod
    def get_viewer(cls, session, target=None):
        # TODO: reenable multiple target windows
        # if target is None:
        if 1:
            target = 'help'
        if target in _targets:
            return _targets[target]
        viewer = HelpUI(session, target)
        _targets[target] = viewer
        return viewer
