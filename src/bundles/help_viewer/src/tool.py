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
from Qt.QtWebEngineCore import QWebEngineUrlSchemeHandler
from Qt.QtWebEngineCore import QWebEngineDownloadRequest
from Qt import using_qt5, using_qt6

if using_qt5:
    from Qt.QtWebEngineWidgets import QWebEnginePage
elif using_qt6:
    from Qt.QtWebEngineCore import QWebEnginePage

_singleton = None
_sys_tags = None
_sys_info = None


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
        background = win_type == QWebEnginePage.WebBrowserBackgroundTab
        return self.help_tool.create_tab(background=background)

    def deleteLater(self):
        self.help_tool = None
        super().deleteLater()

    def contextMenuEvent(self, event):
        # inpsired by qwebengine simplebrowser example
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
        from Qt.QtWidgets import QToolBar, QVBoxLayout, QLineEdit, QTabWidget, QShortcut, QStatusBar, QProgressBar
        from Qt.QtGui import QIcon, QAction, QKeySequence
        from Qt.QtCore import Qt
        shortcuts = (
            (Qt.CTRL | Qt.Key_0, self.page_reset_zoom),
            (Qt.CTRL | Qt.Key_T, lambda: self.create_tab(empty=True)),
            (Qt.CTRL | Qt.Key_W, self.close_current_tab),
            (Qt.CTRL | Qt.Key_Tab, lambda: self.cycle_tab(1)),
            (Qt.CTRL | Qt.SHIFT | Qt.Key_Tab, lambda: self.cycle_tab(-1)),
            (Qt.CTRL | Qt.Key_1, lambda: self.tab_n(0)),
            (Qt.CTRL | Qt.Key_2, lambda: self.tab_n(1)),
            (Qt.CTRL | Qt.Key_3, lambda: self.tab_n(2)),
            (Qt.CTRL | Qt.Key_4, lambda: self.tab_n(3)),
            (Qt.CTRL | Qt.Key_5, lambda: self.tab_n(4)),
            (Qt.CTRL | Qt.Key_6, lambda: self.tab_n(5)),
            (Qt.CTRL | Qt.Key_7, lambda: self.tab_n(6)),
            (Qt.CTRL | Qt.Key_8, lambda: self.tab_n(7)),
            (Qt.CTRL | Qt.Key_9, lambda: self.tab_n(-1)),
        )
        for shortcut, callback in shortcuts:
            sc = QShortcut(QKeySequence(shortcut), parent)
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
                [Qt.CTRL | Qt.Key_Plus, Qt.Key_ZoomIn, Qt.CTRL | Qt.Key_Equal], True),
            ("zoom_out", "Zoom out", "Zoom out", self.page_zoom_out,
                [Qt.CTRL | Qt.Key_Minus, Qt.Key_ZoomOut], True),
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
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.tool_window.manage(placement=None)

        self.profile = create_chimerax_profile(
            self.tabs, interceptor=self.intercept,
            schemes=("installable",),
            handlers={"installable": _InstallableSchemeHandler(self.session)},
            download=self.download_requested)

    def status(self, message, timeout=2000):
        self.status_bar.showMessage(message, timeout)

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
        elif scheme == "installable":
            # dealt with in scheme handler
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
        url_file = item.suggestedFileName()
        base, extension = os.path.splitext(url_file)
        # print("HelpUI.download_requested connect", item.mimeType(), extension)
        # Normally, we would look at the download type or MIME type,
        # but since neither one is set by the server, we look at the
        # download extension instead
        if extension == ".whl":
            is_wheel = True
            # Since the file name encodes information about the bundle, we make
            # sure that we are using the original name instead of the name
            # QWebEngine generated to avoid conflicting with an existing download.
            item.setDownloadFileName(url_file)
            file_path = os.path.join(item.downloadDirectory(), url_file)
            try:
                # Guarantee that file name is available
                os.remove(file_path)
            except OSError:
                pass
            self.session.logger.info("Downloading bundle %s" % url_file)
            self.status("Downloading bundle %s" % url_file, 0)
        else:
            is_wheel = False
            from Qt.QtWidgets import QFileDialog
            file_path = os.path.join(item.downloadDirectory(), item.downloadFileName())
            path, filt = QFileDialog.getSaveFileName(directory=file_path)
            if not path:
                return
            self.session.logger.info("Downloading file %s" % url_file)
            self.status("Downloading file %s" % url_file, 0)
            dirname, filename = os.path.split(path)
            item.setDownloadDirectory(dirname)
            item.setDownloadFileName(filename)
        # print("HelpUI.download_requested accept", file_path)
        if using_qt5:
            item.downloadProgress.connect(self.download_progress)
            item.finished.connect(lambda *args, **kw: self.download_finished(*args, **kw, item=item, is_wheel=is_wheel))
        elif using_qt6:
            item.totalBytesChanged.connect(lambda: self.change_total_bytes(item=item))
            item.receivedBytesChanged.connect(lambda: self.change_received_bytes(item=item))
            item.isFinishedChanged.connect(lambda *args, **kw: self.download_finished(*args, **kw, item=item, is_wheel=is_wheel))
        item.accept()

    def download_progress(self, bytes_received = None, bytes_total = None):
        self.progress_bar.setVisible(True)
        if bytes_total is not None:
            self.progress_bar.setMaximum(bytes_total)
        if bytes_received is not None:
            self.progress_bar.setValue(bytes_received)
        self.progress_bar.update()

    def change_total_bytes(self, item):
        self.download_progress(bytes_total = item.totalBytes())

    def change_received_bytes(self, item):
        self.download_progress(bytes_received = item.receivedBytes())

    def download_finished(self, *args, item=None, is_wheel=False, **kw):
        if using_qt5:
            item.downloadProgress.disconnect()
            item.finished.disconnect()
        elif using_qt6:
            item.totalBytesChanged.disconnect()
            item.receivedBytesChanged.disconnect()
            item.isFinishedChanged.disconnect()
        self.progress_bar.setVisible(False)
        state = item.state()
        if using_qt5:
            if state == item.DownloadCompleted:
                self.status("Download finished")
            elif state == item.DownloadCancelled:
                self.status("Download cancelled")
                return
            elif state == item.DownloadInterrupted:
                self.status(f"Download interrupted: {item.interrupteReasonString()}")
                return
            else:
                self.status(f"Odd download state: {state}")
                return
        elif using_qt6:
            if state == QWebEngineDownloadRequest.DownloadCompleted:
                self.status("Download finished")
            elif state == QWebEngineDownloadRequest.DownloadCancelled:
                self.status("Download cancelled")
                return
            elif state == QWebEngineDownloadRequest.DownloadInterrupted:
                self.status(f"Download interrupted: {item.interrupteReasonString()}")
                return
            else:
                self.status(f"Odd download state: {state}")
                return
        if not is_wheel:
            return
        import os
        import pkginfo
        from chimerax.ui.ask import ask
        dirname = item.downloadDirectory()
        if os.altsep:
            dirname = dirname.replace(os.altsep, os.sep)
        filename = item.downloadFileName()
        path = os.path.join(dirname, filename)
        if not os.path.exists(path):
            self.session.logger.warning("Can't find saved bundle %s" % path)
            return
        try:
            wh = pkginfo.Wheel(path)
        except Exception as e:
            self.session.logger.info("Error parsing %s: %s" % (path, str(e)))
            self.session.logger.info("File saved as %s" % path)
            return
        if not _installable(wh, self.session.logger):
            self.session.logger.info("Bundle saved as %s" % path)
            return
        prefix = _install_or_upgrade(path, self.session).decode()
        reinstall = False
        action = "Install"
        if prefix in ("Downgrade", "Upgrade"):
            action = prefix
            prefix += " to"
        elif prefix == "Installed":
            action = prefix = "Reinstall"
            reinstall = True
        how = ask(self.session,
                  f"{prefix} {wh.name} {wh.version} (file {path})?",
                  [action, "cancel"],
                  title="Toolshed")
        if how == "cancel":
            self.session.logger.info("Bundle installation canceled")
            return
        self.session.toolshed.install_bundle(path,
                                             self.session.logger,
                                             per_user=True, reinstall=reinstall,
                                             session=self.session)
        self.reload_toolshed_tabs()

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
            if w is None:
                return
        history = w.history()
        self.back.setEnabled(history.canGoBack())
        self.forward.setEnabled(history.canGoForward())

    def reload_toolshed_tabs(self):
        toolshed_url = self.session.toolshed.remote_url
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            tab_url = w.url().url()
            if tab_url.startswith(toolshed_url):
                w.reload()

    @classmethod
    def get_viewer(cls, session, target=None, create=True):
        global _singleton
        if _singleton is None:
            if not create:
                return None
            _singleton = HelpUI(session)
        return _singleton


def _installable(wh, logger):
    # check if installable to give better error message than
    #       downstream use of pip.
    if not _compatible(wh.filename, logger):
        return False
    import chimerax.core
    core_bundle = chimerax.core.BUNDLE_NAME + ' '
    for r in wh.requires_dist:
        if not r.startswith(core_bundle):
            continue
        from packaging.requirements import Requirement
        from packaging.version import Version
        core_version = Version(chimerax.core.version)
        try:
            req = Requirement(r)
            return req.specifier.contains(core_version, prereleases=True)
        except Exception:
            logger.info(f"Unsupported ChimeraX-Core requirement: {repr(r)}")
            return False
    return True


def _compatible(filename, logger=None):
    # using wheel filename check if compatible with running ChimeraX
    from wheel_filename import parse_wheel_filename
    global _sys_tags, _sys_info
    if _sys_tags is None:
        from packaging import tags
        _sys_tags = list(tags.sys_tags())
        i, a, p = zip(*((s.interpreter, s.abi, s.platform) for s in _sys_tags))
        _sys_info = (set(i), set(a), set(p))

    def supported_tags(winfo):
        for interpreter in winfo.python_tags:
            for abi in winfo.abi_tags:
                for platform in winfo.platform_tags:
                    yield (interpreter, abi, platform)

    winfo = parse_wheel_filename(filename)
    for t in supported_tags(winfo):
        for s in _sys_tags:
            if t == (s.interpreter, s.abi, s.platform):
                return True
    if logger:
        if _sys_info[2].isdisjoint(winfo.platform_tags):
            logger.warning("bundle is for different platform")
        elif _sys_info[0].isdisjoint(winfo.python_tags):
            logger.warning("bundle is for different version of Python")
        elif _sys_info[1].isdisjoint(winfo.abi_tags):
            logger.warning("bundle uses incompatible Python ABI")
        else:
            logger.warning("bundle is incompatible")
    return False


def _install_or_upgrade(filename, session):
    from wheel_filename import parse_wheel_filename
    winfo = parse_wheel_filename(filename)
    bundle = session.toolshed.find_bundle(winfo.project, session.logger)
    if bundle is None:
        return b"Install"
    try:
        from packaging.version import Version
        b_ver = Version(bundle.version)
        wh_ver = Version(winfo.version)
        if b_ver == wh_ver:
            return b"Installed"
        if b_ver < wh_ver:
            return b"Upgrade"
        return b"Downgrade"
    except Exception:
        return b"Install"


class _InstallableSchemeHandler(QWebEngineUrlSchemeHandler):

    def __init__(self, session):
        self.session = session
        super().__init__()

    def requestStarted(self, request):
        from Qt.QtCore import QBuffer
        wheel_name = request.requestUrl().path()
        try:
            compatible = _compatible(wheel_name)
        except Exception:
            compatible = False
        reply = QBuffer(parent=self)
        request.destroyed.connect(reply.deleteLater)
        reply.open(QBuffer.WriteOnly)
        if not compatible:
            reply.write(b"Not compatible")
        else:
            reply.write(_install_or_upgrade(wheel_name, self.session))
        reply.close()
        request.reply(b"text/plain", reply)
