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


def create_profile(parent, schemes=None, interceptor=None, download=None):
    """
    Create a QWebEngineProfile.  The profile provides shared access to the
    files in the html directory in the chimerax.ui package by rewriting links
    to /chimerax/ui/html to that directory.

    Parameters
    ----------
    interceptor : a callback function taking one argument, an instance
                  of QWebEngineUrlRequestInfo, invoked to handle navigation
                  requests.  Default None.
    schemes :     an iterable of custom schemes that will be used in the
                  view.  If schemes is specified, then interceptor will
                  be called when custom URLs are clicked.  Default None.
    download :    a callback function taking one argument, an instance
                  of QWebEngineDownloadItem, invoked when download is
                  requested.  Default None.
    """
    profile = QWebEngineProfile(parent)
    set_user_agent(profile)

    def _intercept(request_info, *args, interceptor=interceptor):
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
        if interceptor:
            return interceptor(request_info, *args)

    profile._intercept = _RequestInterceptor(callback=_intercept)
    from PyQt5.QtCore import QT_VERSION
    if QT_VERSION < 0x050d00:
        profile.setRequestInterceptor(profile._intercept)
    else:
        # Qt 5.13 and later
        profile.setUrlRequestInterceptor(profile._intercept)
    if schemes:
        profile._schemes = [s.encode("utf-8") for s in schemes]
        profile._scheme_handler = _SchemeHandler()
        for scheme in profile._schemes:
            profile.installUrlSchemeHandler(scheme, profile._scheme_handler)
    if download:
        profile.downloadRequested.connect(download)
    return profile


def delete_profile(profile):
    """Cleanup profiles created by create_profile"""
    profile.downloadRequested.disconnect()
    if hasattr(profile, '_schemes'):
        profile.removeAllUrlSchemeHandlers()
        del profile._scheme_handler
        del profile._schemes
    from PyQt5.QtCore import QT_VERSION
    if QT_VERSION < 0x050d00:
        profile.setRequestInterceptor(None)
    else:
        profile.setUrlRequestInterceptor(None)


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

    require_native_window = False

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
            self._private_profile = True
            self._profile = create_profile(self.parent(), schemes, interceptor, download)
        page = _LoggingPage(self._profile, self, log_errors=log_errors)
        self.setPage(page)
        s = page.settings()
        s.setAttribute(s.LocalStorageEnabled, True)
        self.setAcceptDrops(False)

        if self.require_native_window:
            # This is to work around ChimeraX bug #2537 where the entire
            # GUI becomes blank with some 2019 Intel graphics drivers.
            self.winId()  # Force it to make a native window

    def deleteLater(self):  # noqa
        """Supported API.  Schedule HtmlView instance for deletion at a safe time."""
        if self._private_profile:
            profile = self._profile
            del self._profile
            delete_profile(profile)
        super().deleteLater()

    @property
    def profile(self):
        return self._profile

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
        # Disable and reenable to avoid QWebEngineView taking focus, QTBUG-52999 in Qt 5.7
        self.setEnabled(False)
        # HACK ALERT: to get around a QWebEngineView bug where HTML
        # source is converted into a "data:" link and runs into the
        # URL length limit.
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

    def __init__(self, session, parent, *args, schemes=None, interceptor=None, download=None, profile=None, **kw):
        self.session = session
        private_profile = profile is None
        if private_profile:
            bad_keywords = ('interceptor',)
        else:
            bad_keywords = ('schemes', 'interceptor', 'download')
        for k in bad_keywords:
            if k in kw:
                raise ValueError("Cannot override HtmlView's %s" % k)
        if private_profile:
            # don't share profiles, so interceptor is bound to this QWebEngineView instance

            def intercept(*args, session=session, view=self):
                chimerax_intercept(*args, view=view, session=session)
            profile = create_chimerax_profile(parent, schemes=schemes, interceptor=intercept, download=download)
        super().__init__(parent, *args, profile=profile, **kw)
        if private_profile:
            self._private_profile = True


def create_chimerax_profile(parent, schemes=None, interceptor=None, download=None):
    """
    Create QWebEngineProfile with ChimeraX-specific scheme support

    See :py:func:`create_profile` for argument types.  The interceptor should
    incorporate the :py:func:`chimerax_intercept` functionality.
    """
    if interceptor is None:
        raise ValueError("Excepted interceptor")
    if schemes is None:
        schemes = ('cxcmd', 'help')
    else:
        schemes += type(schemes)(('cxcmd', 'help'))
    return create_profile(parent, schemes, interceptor, download)


def chimerax_intercept(request_info, *args, session=None, view=None):
    """Interceptor for ChimeraX-specific schemes
    
    Parameters
    ----------
    request_info : QWebEngineRequestInfo
    session : a :py:class:`~chimerax.core.session.Session` instance
    view : a QWebEngineView instance or a function that returns the instance
    """
    # interceptor for 'cxcmd' and 'help'
    if session is None or view is None:
        raise ValueError("session and view must be set")
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
        if callable(view):
            originating_url = view().url()
        else:
            originating_url = view.url()
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
        session.ui.thread_safe(defer, session, qurl.url(qurl.None_), from_dir)
        return


def cxcmd(session, url):
    from urllib.parse import unquote
    cmd = url.split(':', 1)[1]  # skip cxcmd:
    cmd = unquote(cmd)  # undo expected quoting
    from chimerax.core.commands import run
    run(session, cmd)
