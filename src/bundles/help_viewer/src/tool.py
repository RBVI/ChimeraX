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
from chimerax.core.ui.widgets import ChimeraXHtmlView

_singleton = None
_help_path = None


def _qurl2text(qurl):
    # recreate help: version
    global _help_path
    if _help_path is None:
        from chimerax import app_data_dir
        import os
        from urllib.request import pathname2url
        _help_path = pathname2url(os.path.join(app_data_dir, 'docs'))
        if _help_path.startswith('///'):
            _help_path = _help_path[2:]
        if not _help_path.endswith('/'):
            _help_path += '/'
    if qurl.scheme() == 'file':
        path = qurl.path()
        if path.startswith(_help_path):
            path = path[len(_help_path):]
            if path.endswith("/index.html"):
                path = path[:-11]
            return "help:%s" % path
    return qurl.toString()


class _HelpWebView(ChimeraXHtmlView):

    def __init__(self, session, tool):
        super().__init__(session, tool.tabs)
        self.help_tool = tool

    def createWindow(self, win_type):  # noqa
        # win_type is window, tab, dialog, backgroundtab
        from PyQt5.QtWebEngineWidgets import QWebEnginePage
        background = win_type == QWebEnginePage.WebBrowserBackgroundTab
        return self.help_tool.create_tab(background=background)

    def deleteLater(self):
        self.help_tool = None
        super().deleteLater()


class HelpUI(ToolInstance):

    # do not close when opening session (especially if web page asked to open session)
    SESSION_ENDURING = True

    def __init__(self, session):
        tool_name = "Help Viewer"
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        # UI content code
        from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QAction, QLineEdit, QTabWidget, QShortcut
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt
        shortcuts = (
            (Qt.CTRL + Qt.Key_0, self.page_reset_zoom),
            (Qt.CTRL + Qt.Key_T, lambda: self.create_tab(empty=True)),
            (Qt.CTRL + Qt.Key_W, self.close_current_tab),
            (Qt.CTRL + Qt.Key_Tab, lambda: self.cycle_tab(1)),
            (Qt.CTRL + Qt.SHIFT + Qt.Key_Tab, lambda: self.cycle_tab(-1)),
            (Qt.CTRL + Qt.Key_1, lambda: self.tab_n(0)),
            (Qt.CTRL + Qt.Key_2, lambda: self.tab_n(1)),
            (Qt.CTRL + Qt.Key_3, lambda: self.tab_n(2)),
            (Qt.CTRL + Qt.Key_4, lambda: self.tab_n(3)),
            (Qt.CTRL + Qt.Key_5, lambda: self.tab_n(4)),
            (Qt.CTRL + Qt.Key_6, lambda: self.tab_n(5)),
            (Qt.CTRL + Qt.Key_7, lambda: self.tab_n(6)),
            (Qt.CTRL + Qt.Key_8, lambda: self.tab_n(7)),
            (Qt.CTRL + Qt.Key_9, lambda: self.tab_n(-1)),
        )
        for shortcut, callback in shortcuts:
            sc = QShortcut(shortcut, parent)
            sc.activated.connect(callback)
        self.toolbar = tb = QToolBar()
        # tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        layout = QVBoxLayout()
        layout.addWidget(tb)
        parent.setLayout(layout)
        import os.path
        icon_dir = os.path.dirname(__file__)
        # attribute, text, tool tip, callback, shortcut(s), enabled
        buttons = (
            ("back", "Back", "Back to previous page", self.page_back,
                Qt.Key_Back, False),
            ("forward", "Forward", "Next page", self.page_forward,
                Qt.Key_Forward, False),
            ("reload", "Reload", "Reload page", self.page_reload,
                Qt.Key_Reload, True),
            ("zoom_in", "Zoom in", "Zoom in", self.page_zoom_in,
                [Qt.CTRL + Qt.Key_Plus, Qt.Key_ZoomIn, Qt.CTRL + Qt.Key_Equal], True),
            ("zoom_out", "Zoom out", "Zoom out", self.page_zoom_out,
                [Qt.CTRL + Qt.Key_Minus, Qt.Key_ZoomOut], True),
            ("home", "Home", "Home page", self.page_home,
                Qt.Key_HomePage, True),
            (None, None, None, None, None, None),
            ("search", "Search", "Search in page", self.page_search,
                Qt.Key_Search, True),
        )
        for attribute, text, tooltip, callback, shortcut, enabled in buttons:
            if attribute is None:
                tb.addSeparator()
                continue
            icon_path = os.path.join(icon_dir, "%s.svg" % attribute)
            setattr(self, attribute, QAction(QIcon(icon_path), text, tb))
            a = getattr(self, attribute)
            a.setToolTip(tooltip)
            a.triggered.connect(callback)
            if shortcut:
                if isinstance(shortcut, list):
                    a.setShortcuts(shortcut)
                else:
                    a.setShortcut(shortcut)
            a.setEnabled(enabled)
            tb.addAction(a)

        self.url = QLineEdit()
        self.url.setPlaceholderText("url")
        self.url.setClearButtonEnabled(True)
        self.url.returnPressed.connect(self.go_to)
        tb.insertWidget(self.reload, self.url)

        self.search_terms = QLineEdit()
        self.search_terms.setClearButtonEnabled(True)
        self.search_terms.setPlaceholderText("search terms")
        self.search_terms.setMaximumWidth(200)
        self.search_terms.returnPressed.connect(self.page_search)
        tb.addWidget(self.search_terms)

        self.tabs = QTabWidget(parent)
        self.tabs.setTabsClosable(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setDocumentMode(False)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        layout.addWidget(self.tabs)

        self.tool_window.manage(placement=None)

    def create_tab(self, *, empty=False, background=False):
        w = _HelpWebView(self.session, self)
        self.tabs.addTab(w, "New Tab")
        if empty:
            from chimerax import app_dirs
            self.tool_window.title = app_dirs.appname
            from PyQt5.QtCore import Qt
            self.url.setFocus(Qt.ShortcutFocusReason)
        if not background:
            self.tabs.setCurrentWidget(w)
        w.loadFinished.connect(lambda okay, w=w: self.page_loaded(w, okay))
        w.titleChanged.connect(lambda title, w=w: self.title_changed(w, title))
        return w

    def show(self, url, *, new=False):
        from urllib.parse import urlparse, urlunparse
        parts = urlparse(url)
        if not parts.scheme:
            parts = list(parts)
            parts[0] = "http"
        url = urlunparse(parts)  # canonicalize
        if new or self.tabs.count() == 0:
            w = self.create_tab()
        else:
            w = self.tabs.currentWidget()
        from PyQt5.QtCore import QUrl
        w.setUrl(QUrl(url))
        self.display(True)

    def go_to(self):
        self.show(self.url.text())

    def page_back(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.history().back()

    def page_forward(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.history().forward()

    def page_home(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        history = w.history()
        hi = history.itemAt(0)
        self.show(_qurl2text(hi.url()))

    def page_zoom_in(self):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.setZoomFactor(1.25 * w.zoomFactor())

    def page_zoom_out(self):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.setZoomFactor(0.8 * w.zoomFactor())

    def page_reset_zoom(self):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.setZoomFactor(1)

    def page_reload(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.reload()

    def page_search(self):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.findText(self.search_terms.text())

    def delete(self):
        global _singleton
        _singleton = None
        # remove tabs before destroying tool to avoid segfault when exiting
        for i in reversed(range(self.tabs.count())):
            self.tabs.removeTab(i)
        ToolInstance.delete(self)

    def page_loaded(self, w, okay):
        if self.tabs.currentWidget() != w:
            return
        history = w.history()
        self.back.setEnabled(history.canGoBack())
        self.forward.setEnabled(history.canGoForward())
        self.url.setText(_qurl2text(w.url()))

    def title_changed(self, w, title):
        if self.tabs.currentWidget() == w:
            self.tool_window.title = title
        i = self.tabs.indexOf(w)
        self.tabs.setTabText(i, title)

    def tab_changed(self, i):
        if i >= 0:
            tab_text = self.tabs.tabText(i)
            if tab_text != "New Tab":
                self.tool_window.title = tab_text
                self.url.setText(_qurl2text(self.tabs.currentWidget().url()))
        else:
            # no more tabs
            self.display(False)

    def close_tab(self, i):
        w = self.tabs.widget(i)
        self.tabs.removeTab(i)
        w.deleteLater()

    def close_current_tab(self):
        i = self.tabs.currentIndex()
        if i != -1:
            self.close_tab(i)

    def cycle_tab(self, incr):
        i = self.tabs.currentIndex()
        if i == -1:
            return
        count = self.tabs.count()
        i = (i + incr) % count
        self.tabs.setCurrentIndex(i)

    def tab_n(self, n):
        count = self.tabs.count()
        if count == 0:
            return
        if n == -1:
            self.tabs.setCurrentIndex(count - 1)
        elif n < count:
            self.tabs.setCurrentIndex(n)

    @classmethod
    def get_viewer(cls, session, target=None):
        global _singleton
        if _singleton is None:
            _singleton = HelpUI(session)
        return _singleton
