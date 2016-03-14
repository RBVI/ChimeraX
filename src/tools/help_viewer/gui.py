# vim: set expandtab shiftwidth=4 softtabstop=4:

# HelpUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
#
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance


def _bitmap(filename, size):
    import os
    import wx
    image = wx.Image(os.path.join(os.path.dirname(__file__), filename))
    image = image.Scale(size.width, size.height, wx.IMAGE_QUALITY_HIGH)
    result = wx.Bitmap(image)
    return result


class HelpUI(ToolInstance):

    SESSION_ENDURING = False    # default

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Help UI"
        # in this case), so only override if different name desired
        self.display_name = "%s Help Viewer" % session.app_dirs.appname
        self.home_page = None
        from chimerax.core import window_sys
        if window_sys == "wx":
            kw = {'size': (500, 500)}
        else:
            kw = {}
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, **kw)
        parent = self.tool_window.ui_area
        # UI content code
        if window_sys == "wx":
            import wx
            from wx import html2
            self.zoom_factor = html2.WEBVIEW_ZOOM_MEDIUM
            # buttons: back, forward, reload, stop, home, search bar
            self.toolbar = wx.ToolBar(parent, wx.ID_ANY,
                                      style=wx.TB_DEFAULT_STYLE | wx.TB_TEXT)
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
            self.help_window.Bind(wx.EVT_CLOSE, self.on_close)
            self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATED, self.on_navigated)
            self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating,
                                  id=self.help_window.GetId())
            self.help_window.Bind(html2.EVT_WEBVIEW_NEWWINDOW, self.on_new_window,
                                  id=self.help_window.GetId())
            self.help_window.Bind(html2.EVT_WEBVIEW_TITLE_CHANGED,
                                  self.on_title_change)
            self.help_window.EnableContextMenu()
        else: # qt
            from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QAction
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
            from PyQt5.QtWebKitWidgets import QWebView
            class HelpWebView(QWebView):
                def __init__(self, ses=session, bi=bundle_info):
                    self.session = ses
                    self.bundle_info = bi
                    QWebView.__init__(self)

                def createWindow(self, win_type):
                    help_ui = HelpUI(self.session, self.bundle_info)
                    return help_ui.help_window
            self.help_window = HelpWebView()
            layout.addWidget(self.help_window)
            self.help_window.linkClicked.connect(self.link_clicked)
            self.help_window.loadFinished.connect(self.page_loaded)
            self.help_window.titleChanged.connect(self.tool_window.set_title)

        self.tool_window.manage(placement=None)

    def show(self, url, set_home=True):
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
        self.show(self.home_page, set_home=False)
    go_home = on_home

    def on_zoom_in(self, event):
        from wx import html2
        if self.zoom_factor < html2.WEBVIEW_ZOOM_LARGEST:
            self.zoom_factor += 1
            self.help_window.SetZoom(self.zoom_factor)
        else:
            self.toolbar.EnableTool(self.zoom_in.GetId(), False)
            self.toolbar.EnableTool(self.zoom_out.GetId(), True)

    def on_zoom_out(self, event):
        from wx import html2
        if self.zoom_factor > html2.WEBVIEW_ZOOM_TINY:
            self.zoom_factor -= 1
            self.help_window.SetZoom(self.zoom_factor)
        else:
            self.toolbar.EnableTool(self.zoom_out.GetId(), False)
            self.toolbar.EnableTool(self.zoom_in.GetId(), True)

    def on_close(self, event):
        self.session.logger.remove_log(self)

    def on_navigated(self, event):
        self.toolbar.EnableTool(self.back.GetId(),
                                self.help_window.CanGoBack())
        self.toolbar.EnableTool(self.forward.GetId(),
                                self.help_window.CanGoForward())

    def page_loaded(self, okay):
        page = self.help_window.page()
        page.setLinkDelegationPolicy(page.DelegateAllLinks)
        history = self.help_window.history()
        self.back.setEnabled(history.canGoBack())
        self.forward.setEnabled(history.canGoForward())

    def on_navigating(self, event):
        session = self.session
        # Handle event
        url = event.GetURL()
        if url.startswith("cxcmd:"):
            from urllib.parse import unquote
            from chimerax.core.commands import run
            event.Veto()
            cmd = unquote(url.split(':', 1)[1])
            # Insert command in command-line entry field
            for ti in session.tools.list():
                if ti.bundle_info.name == 'cmd_line':
                    ti.cmd_replace(cmd)
                    ti.on_enter(None)
                    break
            else:
                # no command line?!?
                run(session, cmd)
            return
        # TODO: check if http url is within ChimeraX docs
        # TODO: handle missing doc -- redirect to web server
        from urllib.parse import urlparse
        parts = urlparse(url)
        if parts.scheme == 'file':
            pass

    def link_clicked(self, qurl):
        if qurl.scheme() == "cxcmd":
            session = self.session
            cmd = qurl.path()
            for ti in session.tools.list():
                if ti.bundle_info.name == 'cmd_line':
                    ti.cmd_replace(cmd)
                    ti.on_enter(None)
                    break
            else:
                # no command line?!?
                from chimerax.core.commands import run
                run(session, cmd)
        else:
            self.help_window.setUrl(qurl)

    def on_title_change(self, event):
        new_title = self.help_window.CurrentTitle
        self.tool_window.set_title(new_title)

    def on_new_window(self, event):
        # TODO: create new help viewer tab or window
        event.Veto()
        url = event.GetURL()
        import webbrowser
        webbrowser.open(url)

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, HelpUI, 'help_viewer')
