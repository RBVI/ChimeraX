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


class _HelpWebView(ChimeraXHtmlView):

    def __init__(self, session, tool):
        super().__init__(session, tool.tabs)
        self.session = session
        self.help_tool = tool

    def createWindow(self, win_type):
        # win_type is window, tab, dialog, backgroundtab
        return self.help_tool.create_tab()


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
        from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QAction, QLabel, QLineEdit, QTabWidget
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
            ("back", "Back", "Back to previous page", self.page_back, False),
            ("forward", "Forward", "Next page", self.page_forward, False),
            ("reload", "Reload", "Reload page", self.page_reload, True),
            ("zoom_in", "Zoom in", "Zoom in", self.page_zoom_in, True),
            ("zoom_out", "Zoom out", "Zoom out", self.page_zoom_out, True),
            ("home", "Home", "Home page", self.page_home, True),
        )
        for attribute, text, tooltip, callback, enabled in buttons:
            icon_path = os.path.join(icon_dir, "%s.png" % attribute)
            setattr(self, attribute, QAction(QIcon(icon_path), text, tb))
            a = getattr(self, attribute)
            a.setToolTip(tooltip)
            a.triggered.connect(callback)
            a.setEnabled(enabled)
            tb.addAction(a)

        self.url = QLineEdit()
        tb.insertWidget(self.reload, self.url)

        label = QLabel("  Search:")
        font = label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        label.setFont(font)
        tb.addWidget(label)
        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("search terms")
        self.search.setMaximumWidth(200)
        tb.addWidget(self.search)
        self.tabs = QTabWidget(parent)
        self.tabs.setTabsClosable(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.tabs.tabCloseRequested.connect(self.tab_close)
        layout.addWidget(self.tabs)

        self.search.returnPressed.connect(self.page_search)
        self.url.returnPressed.connect(self.go_to)

        self.tool_window.manage(placement=None)

    def create_tab(self):
        w = _HelpWebView(self.session, self)
        self.tabs.addTab(w, "")
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
        self.show(hi.url().url())

    def page_zoom_in(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.setZoomFactor(1.25 * self.help_window.zoomFactor())

    def page_zoom_out(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.setZoomFactor(0.8 * self.help_window.zoomFactor())

    def page_reload(self, checked):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.reload()

    def page_search(self):
        w = self.tabs.currentWidget()
        if w is None:
            return
        w.findText(self.search.text())

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
        self.url.setText(w.url().url())

    def title_changed(self, w, title):
        if self.tabs.currentWidget() == w:
            self.tool_window.title = title
        i = self.tabs.indexOf(w)
        self.tabs.setTabText(i, title)

    def tab_changed(self, i):
        if i >= 0:
            self.tool_window.title = self.tabs.tabText(i)
        else:
            # no more tabs
            self.display(False)

    def tab_close(self, i):
        self.tabs.removeTab(i)

    @classmethod
    def get_viewer(cls, session, target=None):
        global _singleton
        if _singleton is None:
            _singleton = HelpUI(session)
        return _singleton
