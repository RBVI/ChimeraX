# vim: set expandtab shiftwidth=4 softtabstop=4:

# HelpUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
#
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance
import weakref

_targets = weakref.WeakValueDictionary()


def _bitmap(filename, size):
    import os
    import wx
    image = wx.Image(os.path.join(os.path.dirname(__file__), filename))
    image = image.Scale(size.width, size.height, wx.IMAGE_QUALITY_HIGH)
    result = wx.Bitmap(image)
    return result


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
        from chimerax.core import window_sys
        if window_sys == "wx":
            kw = {'size': (500, 500)}
        else:
            kw = {}
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, **kw)
        parent = self.tool_window.ui_area
        self.on_page = None
        self.home_page = None
        # UI content code
        if window_sys == "wx":
            import wx
            from wx import html2
            self.on_page = None
            self.zoom_factor = html2.WEBVIEW_ZOOM_MEDIUM
            # buttons: back, forward, reload, stop, home, search bar
            self.toolbar = wx.ToolBar(parent, wx.ID_ANY)
            bitmap_size = wx.ArtProvider.GetNativeSizeHint(wx.ART_TOOLBAR)
            self.back = self.toolbar.AddTool(
                wx.ID_ANY, 'Back',
                wx.ArtProvider.GetBitmap(wx.ART_GO_BACK, wx.ART_TOOLBAR, bitmap_size),
                shortHelp="Go back to previously viewed page")
            self.toolbar.EnableTool(self.back.GetId(), False)
            self.forward = self.toolbar.AddTool(
                wx.ID_ANY, 'Forward',
                wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_TOOLBAR, bitmap_size),
                shortHelp="Go forward to previously viewed page")
            self.toolbar.EnableTool(self.forward.GetId(), False)
            self.home = self.toolbar.AddTool(
                wx.ID_ANY, 'Home',
                wx.ArtProvider.GetBitmap(wx.ART_GO_HOME, wx.ART_TOOLBAR, bitmap_size),
                shortHelp="Return to first page")
            self.zoom_in = self.toolbar.AddTool(
                wx.ID_ANY, 'Zoom In',
                wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_TOOLBAR, bitmap_size),
                shortHelp="magnify document")
            self.zoom_out = self.toolbar.AddTool(
                wx.ID_ANY, 'Zoom Out',
                wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_TOOLBAR, bitmap_size),
                shortHelp="minify document")
            self.toolbar.EnableTool(self.home.GetId(), False)
            self.toolbar.AddStretchableSpace()
            f = self.toolbar.GetFont()
            dc = wx.ScreenDC()
            dc.SetFont(f)
            em_width, _ = dc.GetTextExtent("m")
            search_bar = wx.ComboBox(self.toolbar, size=wx.Size(12 * em_width, -1))
            self.search = self.toolbar.AddControl(search_bar, "Search:")
            self.toolbar.EnableTool(self.search.GetId(), False)
            self.toolbar.Realize()
            self.toolbar.Bind(wx.EVT_TOOL, self.on_back, self.back)
            self.toolbar.Bind(wx.EVT_TOOL, self.on_forward, self.forward)
            self.toolbar.Bind(wx.EVT_TOOL, self.on_home, self.home)
            self.toolbar.Bind(wx.EVT_TOOL, self.on_zoom_in, self.zoom_in)
            self.toolbar.Bind(wx.EVT_TOOL, self.on_zoom_out, self.zoom_out)
            self.help_window = html2.WebView.New(parent, **kw)
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.toolbar, 0, wx.EXPAND)
            sizer.Add(self.help_window, 1, wx.EXPAND)
            parent.SetSizerAndFit(sizer)
            self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATED, self.on_navigated)
            self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating,
                                  id=self.help_window.GetId())
            self.help_window.Bind(html2.EVT_WEBVIEW_NEWWINDOW, self.on_new_window,
                                  id=self.help_window.GetId())
            self.help_window.Bind(html2.EVT_WEBVIEW_TITLE_CHANGED,
                                  self.on_title_change)
            self.help_window.EnableContextMenu()
            if self.help_window.CanSetZoomType(html2.WEBVIEW_ZOOM_TYPE_LAYOUT):
                self.help_window.SetZoomType(html2.WEBVIEW_ZOOM_TYPE_LAYOUT)
        else: # qt
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
            icon_path = os.path.join(os.path.dirname(__file__), "home.png")
            self.home = QAction(QIcon(icon_path), "Home page", tb)
            self.home.triggered.connect(self.go_home)
            self.home.setEnabled(False)
            tb.addAction(self.home)
            self.zoom_in = QAction("+", tb)
            self.zoom_in.triggered.connect(self.page_zoom_in)
            font = self.zoom_in.font()
            font.setPointSize(48)
            self.zoom_in.setFont(font)
            tb.addAction(self.zoom_in)
            self.zoom_out = QAction("-", tb)
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
                hw.findText(s.text(), hw.page().FindWrapsAroundDocument))

        self.tool_window.manage(placement=None)

    def show(self, url, set_home=False):
        from urllib.parse import urlparse, urlunparse
        parts = urlparse(url)
        url = urlunparse(parts)  # canonicalize
        self.on_page = url
        from chimerax.core import window_sys
        if window_sys == "wx":
            self.help_window.Stop()
            if set_home or not self.home_page:
                self.help_window.ClearHistory()
                self.home_page = url
                self.toolbar.EnableTool(self.home.GetId(), True)
                self.toolbar.EnableTool(self.back.GetId(), False)
                self.toolbar.EnableTool(self.forward.GetId(), False)
            self.help_window.LoadURL(url)
        else: # qt
            if set_home or not self.home_page:
                self.help_window.history().clear()
                self.home_page = url
                self.home.setEnabled(True)
                self.back.setEnabled(False)
                self.forward.setEnabled(False)
            from PyQt5.QtCore import QUrl
            self.help_window.setUrl(QUrl(url))

    # wx event handling

    def on_back(self, event):
        self.help_window.GoBack()

    def page_back(self, checked):
        self.help_window.history().back()

    def on_forward(self, event):
        self.help_window.GoForward()

    def page_forward(self, checked):
        self.help_window.history().forward()

    def on_home(self, event):
        self.show(self.home_page)
    go_home = on_home

    def on_zoom_in(self, event):
        from wx import html2
        if self.zoom_factor < html2.WEBVIEW_ZOOM_LARGEST:
            self.zoom_factor += 1
            self.help_window.SetZoom(self.zoom_factor)
        if self.zoom_factor == html2.WEBVIEW_ZOOM_LARGEST:
            self.toolbar.EnableTool(self.zoom_in.GetId(), False)
        self.toolbar.EnableTool(self.zoom_out.GetId(), True)

    def page_zoom_in(self, checked):
        self.help_window.setZoomFactor(1.25 * self.help_window.zoomFactor())

    def on_zoom_out(self, event):
        from wx import html2
        if self.zoom_factor > html2.WEBVIEW_ZOOM_TINY:
            self.zoom_factor -= 1
            self.help_window.SetZoom(self.zoom_factor)
        if self.zoom_factor == html2.WEBVIEW_ZOOM_TINY:
            self.toolbar.EnableTool(self.zoom_out.GetId(), False)
        self.toolbar.EnableTool(self.zoom_in.GetId(), True)

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
