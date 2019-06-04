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

"""
:py:class:`ChimeraXHtmlView` provides a HTML window that understands
ChimeraX-specific schemes.  It is built on top of :py:class:`HtmlView`,
which provides scheme support.
"""

from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineCore import QWebEngineUrlSchemeHandler


def set_user_agent(profile):
    """Set profile's user agent"""
    from chimerax.core.fetch import html_user_agent
    from chimerax import app_dirs
    profile.setHttpUserAgent('%s %s' % (profile.httpUserAgent(), html_user_agent(app_dirs)))


class HtmlView(QWebEngineView):
    """
    HtmlView is a derived class from PyQt5.QtWebEngineWidgets.QWebEngineView
    that simplifies using custom schemes and intercepting navigation requests.

    HtmlView may be instantiated just like QWebEngineView, with additional
    keyword arguments:

    Parameters
    ----------
    size_hint :   a QSize compatible value, typically (width, height),
                  specifying the preferred initial size for the view.
                  Default None.
    interceptor : a callback function taking one argument, an instance
                  of QWebEngineUrlRequestInfo, invoked to handle navigation
                  requests.  Default None.
    schemes :     an iterable of custom schemes that will be used in the
                  view.  If schemes is specified, then interceptor will
                  be called when custom URLs are clicked.  Default None.
    download :    a callback function taking one argument, an instance
                  of QWebEngineDownloadItem, invoked when download is
                  requested.  Default None.
    profile :     the QWebEngineProfile to use.  If it is given, then
                  'interceptor', 'schemes', and 'download' parameters are
                  ignored because they are assumed to be already set in
                  the profile.  Default None.
    tool_window : if specified, ChimeraX context menu is displayed instead
                  of default context menu.  Default None.
    log_errors :  whether to log JavaScript error/warning/info messages
                  to ChimeraX console.  Default False.

    Attributes
    ----------
    profile :     the QWebEngineProfile used
    """

    def __init__(self, *args, size_hint=None, schemes=None,
                 interceptor=None, download=None, profile=None,
                 tool_window=None, log_errors=False, **kw):
        super().__init__(*args, **kw)
        self._size_hint = size_hint
        self._tool_window = tool_window
        if profile is not None:
            self._profile = profile
            self._private_profile = False
        else:
            p = self._profile = QWebEngineProfile(self.parent())
            self._private_profile = True
            set_user_agent(p)
            self._intercept = _RequestInterceptor(callback=self.intercept)
            self._intercept_cb = interceptor
            p.setRequestInterceptor(self._intercept)
            if schemes:
                self._schemes = [s.encode("utf-8") for s in schemes]
                self._scheme_handler = _SchemeHandler()
                for scheme in self._schemes:
                    p.installUrlSchemeHandler(scheme, self._scheme_handler)
            if download:
                p.downloadRequested.connect(download)
        page = _LoggingPage(self._profile, self, log_errors=log_errors)
        self.setPage(page)
        s = page.settings()
        s.setAttribute(s.LocalStorageEnabled, True)
        self.setAcceptDrops(False)

    def deleteLater(self):  # noqa
        """Supported API.  Schedule HtmlView instance for deletion at a safe time."""
        if self._private_profile:
            p = self._profile
            p.downloadRequested.disconnect()
            p.removeAllUrlSchemeHandlers()
            p.setRequestInterceptor(None)
        super().deleteLater()

    @property
    def profile(self):
        return self._profile

    def intercept(self, request_info, *args):
        import os
        qurl = request_info.requestUrl()
        scheme = qurl.scheme()
        if scheme == 'file':
            # If path exists or references html files included
            # in chimerax.ui, intercept and return
            import sys
            if sys.platform == "win32":
                full_path = qurl.path()
                # If URL path is absolute, remove the leading /.
                # If URL path include a drive, extract the non-drive
                # part for matching against /chimerax/ui/html/
                if full_path[0] == '/':
                    full_path = full_path[1:]
                drive, path = os.path.splitdrive(full_path)
                if not drive:
                    path = full_path = qurl.path()
            else:
                path = full_path = qurl.path()
            if os.path.exists(os.path.normpath(full_path)):
                return
            if path.startswith("/chimerax/ui/html/"):
                from chimerax import ui
                ui_dir = os.path.dirname(ui.__file__).replace(os.path.sep, '/')
                full_path = ui_dir + path[len("/chimerax/ui"):]
                if sys.platform == "win32":
                    # change C:/ to /C:/
                    full_path = '/' + full_path
                qurl.setPath(full_path)
                request_info.redirect(qurl)
                return
        if self._intercept_cb:
            return self._intercept_cb(request_info, *args)

    def sizeHint(self):  # noqa
        """Supported API.  Returns size hint as a :py:class:PyQt5.QtCore.QSize instance."""
        if self._size_hint:
            from PyQt5.QtCore import QSize
            return QSize(*self._size_hint)
        else:
            return super().sizeHint()

    def contextMenuEvent(self, event):
        """Private API. Send event to context menu handler."""
        if self._tool_window:
            self._tool_window._show_context_menu(event)
        else:
            super().contextMenuEvent(event)

    def setHtml(self, html, url=None):  # noqa
        """Supported API. Replace widget content.

        Parameters
        ----------
        html :   a string containing new HTML content.
        url :    a string containing URL corresponding to content.
        """
        from PyQt5.QtCore import QUrl
        self.setEnabled(False)
        if len(html) < 1000000:
            if url is None:
                url = QUrl()
            super().setHtml(html, url)
        else:
            try:
                tf = open(self._tf_name, "wb")
            except AttributeError:
                import tempfile
                import atexit
                tf = tempfile.NamedTemporaryFile(prefix="chbp", suffix=".html",
                                                 delete=False, mode="wb")
                self._tf_name = tf.name

                def clean(filename):
                    import os
                    try:
                        os.remove(filename)
                    except OSError:
                        pass
                atexit.register(clean, tf.name)
            tf.write(bytes(html, "utf-8"))
            # On Windows, we have to close the temp file before
            # trying to open it again (like loading HTML from it).
            tf.close()
            self.load(QUrl.fromLocalFile(self._tf_name))
        self.setEnabled(True)

    def setUrl(self, url):  # noqa
        """Supported API. Replace widget content.

        Parameters
        ----------
        url :    a string containing URL to new content.
        """
        if isinstance(url, str):
            from PyQt5.QtCore import QUrl
            url = QUrl(url)
        super().setUrl(url)

    def runJavaScript(self, script, *args):    # noqa
        """Supported API.  Run JavaScript using currently displayed HTML page.

        Parameters
        ----------
        script :    a string containing URL to new content.
        args :      additional arguments supported by
                    :py:method:`PyQt5.QtWebEngineWidgets.QWebEnginePage.runJavaScript`.
        """
        self.page().runJavaScript(script, *args)

class _LoggingPage(QWebEnginePage):

    Levels = {
        0: "info",
        1: "warning",
        2: "error",
    }

    def __init__(self, *args, log_errors=False, **kw):
        super().__init__(*args, **kw)
        self.__log = log_errors

    def javaScriptConsoleMessage(self, level, msg, lineNumber, sourceId):
        if not self.__log:
            return
        import os.path
        filename = os.path.basename(sourceId)
        print("JS console(%s:%d:%s): %s" % (filename, lineNumber,
                                            self.Levels[level], msg))

class _RequestInterceptor(QWebEngineUrlRequestInterceptor):

    def __init__(self, *args, callback=None, **kw):
        super().__init__(*args, **kw)
        self._callback = callback

    def interceptRequest(self, info):  # noqa
        # "info" is an instance of QWebEngineUrlRequestInfo
        if self._callback:
            self._callback(info)


class _SchemeHandler(QWebEngineUrlSchemeHandler):

    def requestStarted(self, request):  # noqa
        # "request" is an instance of QWebEngineUrlRequestJob
        # We do nothing because caller should be intercepting
        # custom URL navigation already
        pass


class ChimeraXHtmlView(HtmlView):
    """
    HTML window with ChimeraX-specific scheme support.

    The schemes are 'cxcmd' and 'help'.
    """

    def __init__(self, session, *args, **kw):
        self.session = session
        for k in ('schemes', 'interceptor', 'download', 'profile'):
            if k in kw:
                raise ValueError("Cannot override HtmlView's %s" % k)
        # don't share profiles, so interceptor is bound to this instance
        super().__init__(*args, schemes=('cxcmd', 'help'),
                         interceptor=self.link_clicked,
                         download=self.download_requested, **kw)
        self._pending_downloads = []

    def link_clicked(self, request_info, *args):
        import os
        qurl = request_info.requestUrl()
        scheme = qurl.scheme()
        if scheme == 'file':
            # Paths to existing and chimerax.ui have already been intercepted.
            # Treat all directories with help documentation as equivalent
            # to integrate bundle help with the main help.  That is, so
            # relative hrefs will find files in other help directories.
            import sys
            from chimerax.core import toolshed
            path = os.path.normpath(qurl.path())
            if sys.platform == "win32" and path[0] == os.path.sep:
                path = path[1:]
            help_directories = toolshed.get_help_directories()
            for hd in help_directories:
                if path.startswith(hd):
                    break
            else:
                return   # not in a help directory
            tail = path[len(hd) + 1:]
            for hd in help_directories:
                path = os.path.join(hd, tail)
                if os.path.exists(path):
                    break
            else:
                return  # not in another help directory
            path = os.path.sep + path
            if sys.platform == "win32":
                path = path.replace(os.path.sep, '/')
                if os.path.isabs(path):
                    path = '/' + path
            qurl.setPath(path)
            request_info.redirect(qurl)  # set requested url to good location
            return
        if scheme in ('cxcmd', 'help'):
            # originating_url = request_info.firstPartyUrl()  # doesn't work
            originating_url = self.url()
            from_dir = None
            if originating_url.isLocalFile():
                from_dir = os.path.dirname(originating_url.toLocalFile())

            def defer(session, topic, from_dir):
                prev_dir = None
                try:
                    if from_dir:
                        prev_dir = os.getcwd()
                        try:
                            os.chdir(from_dir)
                        except OSError as e:
                            prev_dir = None
                            session.logger.warning(
                                'Unable to change working directory: %s' % e)
                    if scheme == 'cxcmd':
                        cxcmd(session, topic)
                    elif scheme == 'help':
                        from chimerax.help_viewer.cmd import help
                        help(session, topic)
                finally:
                    if prev_dir:
                        os.chdir(prev_dir)
            self.session.ui.thread_safe(defer, self.session, qurl.url(qurl.None_), from_dir)
            return

    def download_requested(self, item):
        # "item" is an instance of QWebEngineDownloadItem
        # print("ChimeraXHtmlView.download_requested", item)
        import os
        url_file = item.url().fileName()
        base, extension = os.path.splitext(url_file)
        # print("ChimeraXHtmlView.download_requested connect", item.mimeType(), extension)
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
            # print("ChimeraXHtmlView.download_requested clean", file_path)
            try:
                # Guarantee that file name is available
                os.remove(file_path)
            except OSError:
                pass
            self._pending_downloads.append(item)
            self.session.logger.info("Downloading bundle %s" % url_file)
            item.finished.connect(self.download_finished)
        else:
            from PyQt5.QtWidgets import QFileDialog
            path, filt = QFileDialog.getSaveFileName(directory=item.path())
            if not path:
                return
            self.session.logger.info("Downloading file %s" % url_file)
            item.setPath(path)
        # print("ChimeraXHTMLView.download_requested accept", file_path)
        item.accept()

    def download_finished(self, *args, **kw):
        # print("ChimeraXHtmlView.download_finished", args, kw)
        finished = []
        pending = []
        for item in self._pending_downloads:
            if not item.isFinished():
                pending.append(item)
            else:
                finished.append(item)
        self._pending_downloads = pending
        for item in finished:
            item.finished.disconnect()
            filename = item.path()
            if not _installable(filename):
                self.session.logger.info("Bundle saved as %s" % filename)
                continue
            from chimerax.ui.ask import ask
            how = ask(self.session,
                      "Install %s for:" % filename,
                      ["just me", "all users", "cancel"],
                      title="Toolshed")
            if how == "cancel":
                self.session.logger.info("Bundle installation canceled")
                continue
            elif how == "just me":
                per_user = True
            else:
                per_user = False
            self.session.toolshed.install_bundle(filename,
                                                 self.session.logger,
                                                 per_user=per_user,
                                                 session=self.session)


def _installable(filename):
    import pkginfo, re
    from distutils.version import LooseVersion as Version
    import chimerax.core
    try:
        w = pkginfo.Wheel(filename)
    except:
        return False
    pat = re.compile(r'ChimeraX-Core \((?P<op>.*=)(?P<version>\d.*)\)')
    for req in w.requires_dist:
        m = pat.match(req)
        if m:
            op = m.group("op")
            version = m.group("version")
            if op == ">=":
                return Version(version) >= Version(chimerax.core.version)
            elif op == "==":
                return Version(version) == Version(chimerax.core.version)
            elif op == "<=":
                return Version(version) <= Version(chimerax.core.version)
            return False
    return True


def cxcmd(session, url):
    from urllib.parse import unquote
    cmd = url.split(':', 1)[1]  # skip cxcmd:
    cmd = unquote(cmd)  # undo expected quoting
    from chimerax.core.commands import run
    run(session, cmd)
