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
from chimerax.ui.widgets.htmlview import (
    ChimeraXHtmlView, chimerax_intercept,
    create_chimerax_profile
)

_singleton = None


def _qurl2text(qurl):
    # recreate help: version
    from . import help_url_paths
    if qurl.scheme() == 'file':
        path = qurl.path()
        frag = qurl.fragment()
        if frag:
            frag = '#%s' % frag
        for hp in help_url_paths:
            if path.startswith(hp):
                path = path[len(hp):]
                if path.endswith("/index.html"):
                    path = path[:-11]
                return "help:%s%s" % (path, frag)
    return qurl.toString()


class _HelpWebView(ChimeraXHtmlView):

    def __init__(self, session, tool, profile):
        super().__init__(session, tool.tabs, size_hint=(840, 800), profile=profile)
        self.help_tool = tool

    def createWindow(self, win_type):  # noqa
        # win_type is window, tab, dialog, backgroundtab
        from Qt.QtWebEngineWidgets import QWebEnginePage
        background = win_type == QWebEnginePage.WebBrowserBackgroundTab
        return self.help_tool.create_tab(background=background)

    def deleteLater(self):
        self.help_tool = None
        super().deleteLater()

    def contextMenuEvent(self, event):
        # inpsired by qwebengine simplebrowser example
        from Qt.QtWebEngineWidgets import QWebEnginePage
        from Qt.QtCore import Qt
        page = self.page()
        # keep reference to menu, so it doesn't get deleted before being shown
        self._context_menu = menu = page.createStandardContextMenu()
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        action = page.action(QWebEnginePage.OpenLinkInThisWindow)
        actions = iter(menu.actions())
        for a in actions:
            if a == action:
                try:
                    before = next(actions)
                except StopIteration:
                    before = None
                menu.insertAction(before, page.action(QWebEnginePage.OpenLinkInNewTab))
                menu.insertAction(before, page.action(QWebEnginePage.OpenLinkInNewBackgroundTab))
                break
        # if not page.contextMenuData().selectedText():
        #    menu.addAction(page.action(QWebEnginePage.SavePage))
        menu.popup(event.globalPos())


class HelpUI(ToolInstance):

    # do not close when opening session (especially if web page asked to open session)
    SESSION_ENDURING = True

    help = "help:user/tools/helpviewer.html"

    def __init__(self, session):
        tool_name = "Help Viewer"
        ToolInstance.__init__(self, session, tool_name)
        self._pending_downloads = []
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        # UI content code
        from Qt.QtWidgets import QToolBar, QVBoxLayout, QAction, QLineEdit, QTabWidget, QShortcut, QStatusBar
        from Qt.QtGui import QIcon
        from Qt.QtCore import Qt
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
        layout.setContentsMargins(0, 1, 0, 0)
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
            ("new_tab", "New Tab", "New Tab", lambda: self.create_tab(empty=True),
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
                from Qt.QtGui import QKeySequence
                if isinstance(shortcut, list):
                    a.setShortcuts([QKeySequence(s) for s in shortcut])
                else:
                    a.setShortcut(QKeySequence(shortcut))
            a.setEnabled(enabled)
            tb.addAction(a)

        self.url = QLineEdit()
        self.url.setPlaceholderText("url")
        self.url.setClearButtonEnabled(True)
        self.url.returnPressed.connect(self.go_to)
        tb.insertWidget(self.reload, self.url)

        self.search_terms = QLineEdit()
        self.search_terms.setClearButtonEnabled(True)
        self.search_terms.setPlaceholderText("search in page")
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

        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)

        self.tool_window.manage(placement=None)

        self.profile = create_chimerax_profile(
            self.tabs, interceptor=self.intercept,
            download=self.download_requested)

    def status(self, message):
        self.status_bar.showMessage(message, 2000)

    def create_tab(self, *, empty=False, background=False):
        w = _HelpWebView(self.session, self, profile=self.profile)
        self.tabs.addTab(w, "New Tab")
        if empty:
            from chimerax import app_dirs
            self.tool_window.title = app_dirs.appname
            from Qt.QtCore import Qt
            self.url.setFocus(Qt.ShortcutFocusReason)
        if not background:
            self.tabs.setCurrentWidget(w)
        p = w.page()
        p.loadFinished.connect(lambda okay, w=w: self.page_loaded(w, okay))
        p.urlChanged.connect(lambda url, w=w: self.url_changed(w, url))
        p.titleChanged.connect(lambda title, w=w: self.title_changed(w, title))
        p.linkHovered.connect(self.link_hovered)
        p.authenticationRequired.connect(self.authorize)
        # TODO? p.iconChanged.connect(....)
        # TODO? p.iconUrlChanged.connect(....)
        # TODO? p.loadProgress.connect(....)
        # TODO? p.loadStarted.connect(....)
        # TODO? p.renderProcessTerminated.connect(....)
        # TODO? p.selectionChanged.connect(....)
        # TODO? p.windowCloseRequested.connect(....)
        return w

    def authorize(self, requestUrl, auth):
        from Qt.QtWidgets import QDialog, QGridLayout, QLineEdit, QLabel, QPushButton
        from Qt.QtCore import Qt
        class PasswordDialog(QDialog):

            def __init__(self, requestUrl, auth, parent=None):
                super().__init__(parent)
                self.setWindowTitle("ChimeraX: Authentication Required")
                self.setModal(True)
                self.auth = auth
                url = requestUrl.url()
                key = QLabel("\N{KEY}")
                font = key.font()
                font.setPointSize(2 * font.pointSize())
                key.setFont(font)
                self.info = QLabel(f'{url} is requesting your username and password.  The site says: "{auth.realm()}"')
                self.info.setWordWrap(True)
                user_name = QLabel("User name:")
                self.user_name = QLineEdit(self)
                password = QLabel("Password:")
                self.password = QLineEdit(self)
                self.password.setEchoMode(QLineEdit.Password)
                self.cancel_button = QPushButton('Cancel', self)
                self.cancel_button.clicked.connect(self.reject)
                self.ok_button = QPushButton('OK', self)
                self.ok_button.clicked.connect(self.accept)
                self.ok_button.setDefault(True)
                layout = QGridLayout(self)
                layout.setColumnStretch(1, 1)
                layout.addWidget(key, 0, 0, Qt.AlignCenter)
                layout.addWidget(self.info, 0, 1, 1, 3)
                layout.addWidget(user_name, 1, 0, Qt.AlignRight)
                layout.addWidget(self.user_name, 1, 1, 1, 3)
                layout.addWidget(password, 2, 0, Qt.AlignRight)
                layout.addWidget(self.password, 2, 1, 1, 3)
                layout.addWidget(self.cancel_button, 3, 2)
                layout.addWidget(self.ok_button, 3, 3)

            def reject(self):
                from Qt.QtNetwork import QAuthenticator
                import sip
                sip.assign(auth, QAuthenticator())
                return super().reject()

            def accept(self):
                self.auth.setUser(self.user_name.text())
                self.auth.setPassword(self.password.text())
                return super().accept()

        p = PasswordDialog(requestUrl, auth)
        p.exec_()

    def intercept(self, request_info, *args):
        # check for help:user and generate the index page if need be
        qurl = request_info.requestUrl()
        scheme = qurl.scheme()
        if scheme == 'file' and qurl.path().endswith(('/docs/user', '/docs/user/index.html')):
            import sys
            import os
            path = qurl.toLocalFile()
            from chimerax import app_dirs
            cached_index = os.path.join(app_dirs.user_cache_dir, 'docs', 'user', 'index.html')
            if not os.path.exists(cached_index):
                from .cmd import _generate_index
                from chimerax import app_data_dir
                path = os.path.join(app_data_dir, 'docs', 'user', 'index.html')
                new_path = _generate_index(path, self.session.logger)
                if new_path is not None:
                    if sys.platform == 'win32':
                        new_path = new_path.replace(os.path.sep, '/')
                        if os.path.isabs(new_path):
                            new_path = '/' + new_path
                    qurl.setPath(new_path)
                    request_info.redirect(qurl)
                    return

        tabs = self.tabs
        from Qt import qt_object_is_deleted
        if not qt_object_is_deleted(tabs):
            chimerax_intercept(request_info, *args, session=self.session,
                               view=tabs.currentWidget())

    def download_requested(self, item):
        # "item" is an instance of QWebEngineDownloadItem
        # print("HelpUI.download_requested", item)
        import os
        url_file = item.url().fileName()
        base, extension = os.path.splitext(url_file)
        # print("HelpUI.download_requested connect", item.mimeType(), extension)
        # Normally, we would look at the download type or MIME type,
        # but since neither one is set by the server, we look at the
        # download extension instead
        if extension == ".whl":
            if not base.endswith("x86_64"):
                # Since the file name encodes the package name and version
                # number, we make sure that we are using the right name
                # instead of whatever QWebEngine may want to use.
                # Remove _# which may be present if bundle author submitted
                # the same version of the bundle multiple times.
                parts = base.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    url_file = parts[0] + extension
            file_path = os.path.join(os.path.dirname(item.path()), url_file)
            import pkg_resources
            py_env = pkg_resources.Environment()
            dist = pkg_resources.Distribution.from_filename(file_path)
            if not py_env.can_add(dist):
                raise ValueError("unsupported wheel platform")
            item.setPath(file_path)
            # print("HelpUI.download_requested clean", file_path)
            try:
                # Guarantee that file name is available
                os.remove(file_path)
            except OSError:
                pass
            self._pending_downloads.append(item)
            self.session.logger.info("Downloading bundle %s" % url_file)
            item.finished.connect(self.download_finished)
        else:
            from Qt.QtWidgets import QFileDialog
            path, filt = QFileDialog.getSaveFileName(directory=item.path())
            if not path:
                return
            self.session.logger.info("Downloading file %s" % url_file)
            item.setPath(path)
        # print("HelpUI.download_requested accept", file_path)
        item.accept()

    def download_finished(self, *args, **kw):
        # print("HelpUI.download_finished", args, kw)
        finished = []
        pending = []
        for item in self._pending_downloads:
            if not item.isFinished():
                pending.append(item)
            else:
                finished.append(item)
        self._pending_downloads = pending
        import pkginfo
        from chimerax.ui.ask import ask
        for item in finished:
            item.finished.disconnect()
            filename = item.path()
            try:
                w = pkginfo.Wheel(filename)
            except Exception as e:
                self.session.logger.info("Error parsing %s: %s" % (filename, str(e)))
                self.session.logger.info("File saved as %s" % filename)
                continue
            if not _installable(w, self.session.logger):
                self.session.logger.info("Bundle saved as %s" % filename)
                continue
            how = ask(self.session,
                      "Install %s %s (file %s)?" % (w.name, w.version, filename),
                      ["install", "cancel"],
                      title="Toolshed")
            if how == "cancel":
                self.session.logger.info("Bundle installation canceled")
                continue
            self.session.toolshed.install_bundle(filename,
                                                 self.session.logger,
                                                 per_user=True,
                                                 session=self.session)

    def show(self, url, *, new_tab=False, html=None):
        from urllib.parse import urlparse, urlunparse
        parts = urlparse(url)
        if not parts.scheme:
            parts = list(parts)
            parts[0] = "http"
        url = urlunparse(parts)  # canonicalize
        if new_tab or self.tabs.count() == 0:
            w = self.create_tab()
        else:
            w = self.tabs.currentWidget()
        from Qt.QtCore import QUrl
        if html:
            w.setHtml(html, QUrl(url))
        else:
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
        ToolInstance.delete(self)

    def page_loaded(self, w, okay):
        if self.tabs.currentWidget() != w:
            return
        self.update_back_forward(w)

    def url_changed(self, w, url):
        if self.tabs.currentWidget() != w:
            return
        self.url.setText(_qurl2text(url))
        self.update_back_forward(w)

    def title_changed(self, w, title):
        if self.tabs.currentWidget() == w:
            self.tool_window.title = title
        i = self.tabs.indexOf(w)
        self.tabs.setTabText(i, title)

    def link_hovered(self, url):
        from Qt.QtCore import QUrl
        try:
            self.status(_qurl2text(QUrl(url)))
        except Exception:
            self.status(url)

    def tab_changed(self, i):
        if i >= 0:
            tab_text = self.tabs.tabText(i)
            if tab_text != "New Tab":
                self.tool_window.title = tab_text
                self.url.setText(_qurl2text(self.tabs.currentWidget().url()))
        else:
            # no more tabs
            self.display(False)
        self.update_back_forward()

    def close_tab(self, i):
        w = self.tabs.widget(i)
        self.tabs.removeTab(i)
        w.deleteLater()
        self.update_back_forward()

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

    def update_back_forward(self, w=None):
        if w is None:
            w = self.tabs.currentWidget()
        history = w.history()
        self.back.setEnabled(history.canGoBack())
        self.forward.setEnabled(history.canGoForward())

    @classmethod
    def get_viewer(cls, session, target=None):
        global _singleton
        if _singleton is None:
            _singleton = HelpUI(session)
        return _singleton


def _installable(w, logger):
    return True
    # TODO: check if installable to give better error message than
    #       downstream use of pip.
    import re
    from distutils.version import LooseVersion as Version
    import chimerax.core
    pat = re.compile(r'ChimeraX-Core \((?P<op>.*=)(?P<version>\d.*)\)')
    for req in w.requires_dist:
        m = pat.match(req)
        if m:
            op = m.group("op")
            version = m.group("version")
            if op == ">=":
                return Version(chimerax.core.version) >= Version(version)
            elif op == "==":
                return Version(chimerax.core.version) == Version(version)
            elif op == "<=":
                return Version(chimerax.core.version) <= Version(version)
            logger.info("Unsupported version comparison:", op)
            return False
    return True
