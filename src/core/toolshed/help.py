# vim: set expandtab ts=4 sw=4:

InitialPage = """<html>
<head>
<title>Chimera2 Help Test</title>
<script>
function do_something() { window.location.href = "chimera2:do_something"; }
</script>
</head>
<body>
<h1><a href="chimera2:test">Button Test</a></h1>
<button onclick="do_something();">Do Something</button>
<h1>Help Links Test</h1>
<ul>
<li><a href="/chimera2/help/shedtest.tool0">Tool0</a></li>
<li><a href="chimera2/help/shedtest.core0">Core0</a></li>
</ul>
</body>
</html>"""
InitialURL = "http://chimera2.rbvi.ucsf.edu/chimera2/index.html"

ErrorPage = """<html>
<head><title>Chimera 2 Error Page</title></head>
<body>
<h1>Chimera 2 Error Page</h1>
<p>%s</p>
</body>"""
ErrorURL = "http://chimera2.rbvi.ucsf.edu/chimera2/error.html"

# HostPrefix must not include the trailing /
# HelpPrefix must include the trailing /
HostPrefix = "http://www.rbvi.ucsf.edu"
HelpPrefix = "/chimera2/help/"

def noprint(*args, **kw):
    return

import wx, sys
from wx import html2

class MyApp(wx.App):

    def OnInit(self):
        fr = HelpFrame("WxWidgets Benchmark")
        fr.Show(True)
        return True

class HelpFrame(wx.Frame):

    def __init__(self, title):
        wx.Frame.__init__(self, None, wx.ID_ANY, title,
                        size=wx.Size(600,600))
        self.menubar = self._make_menu()
        self.toolbar = self._make_toolbar()
        self.webview = self._make_webview()
        self.statusbar = self._make_statusbar()
        #self.webview.SetPage(InitialPage, InitialURL)
        #self.webview.LoadURL(self.home)

    def _make_menu(self):
        history_menu = wx.Menu()
        item = history_menu.Append(wx.ID_ANY, "&Report History",
                    "Report history status")
        self.Bind(wx.EVT_MENU, self._history_report, id=item.GetId())
        item = history_menu.Append(wx.ID_ANY, "&Clear History",
                    "Clear history")
        self.Bind(wx.EVT_MENU, self._history_clear, id=item.GetId())
        history_menu.AppendSeparator()
        history_menu.Append(wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self._exit, id=wx.ID_EXIT)

        help_menu = wx.Menu()
        help_menu.Append(wx.ID_ABOUT)
        self.Bind(wx.EVT_MENU, self._about, id=wx.ID_ABOUT)

        menubar = wx.MenuBar()
        menubar.Append(history_menu, "&History")
        menubar.Append(help_menu, "&Help")
        self.SetMenuBar(menubar)
        return menubar

    def _make_toolbar(self):
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        from wx import ArtProvider
        tsize = (24,24)
        tb.SetToolBitmapSize(tsize)
        back = ArtProvider.GetBitmap(wx.ART_GO_BACK,
                        wx.ART_TOOLBAR, tsize)
        self.back_button = tb.AddTool(wx.ID_ANY, "Back", back,
                    wx.NullBitmap, wx.ITEM_NORMAL,
                    "Back", "Go to previous page", None)
        self.Bind(wx.EVT_TOOL, self._go_back,
                    id=self.back_button.GetId())
        forward = ArtProvider.GetBitmap(wx.ART_GO_FORWARD,
                        wx.ART_TOOLBAR, tsize)
        self.forward_button = tb.AddTool(wx.ID_ANY, "Forward", forward,
                    wx.NullBitmap, wx.ITEM_NORMAL,
                    "Forward", "Go to next page", None)
        self.Bind(wx.EVT_TOOL, self._go_forward,
                    id=self.forward_button.GetId())
        home = ArtProvider.GetBitmap(wx.ART_GO_HOME,
                        wx.ART_TOOLBAR, tsize)
        home_button = tb.AddTool(wx.ID_ANY, "Home", home,
                    wx.NullBitmap, wx.ITEM_NORMAL,
                    "Home", "Go to home page", None)
        self.Bind(wx.EVT_TOOL, self._go_home,
                    id=home_button.GetId())

        tb.Realize()
        return tb

    def _make_webview(self):
        import os, os.path
        home_page = os.path.join(os.getcwd(), "helpdir", "index.html")
        from urllib.request import pathname2url
        self.home = "file:%s" % pathname2url(home_page)

        wv = html2.WebView.New(self, url=self.home)
        wv.EnableContextMenu(False)
        wv_id = wv.GetId()
        self.Bind(html2.EVT_WEBVIEW_NAVIGATING, self._on_navigating,
                            id=wv_id)
        self.Bind(html2.EVT_WEBVIEW_NAVIGATED, self._on_navigated,
                            id=wv_id)
        self.Bind(html2.EVT_WEBVIEW_ERROR, self._on_document_error,
                            id=wv_id)
        self.Bind(html2.EVT_WEBVIEW_LOADED, self._on_loaded,
                            id=wv_id)
        self.Bind(wx.EVT_UPDATE_UI, self._update_ui, id=wv_id)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(wv, 10, wx.EXPAND, 0)
        self.SetSizer(sizer)
        self.usingIE = (html2.WebViewBackendDefault ==
                        html2.WebViewBackendIE)
        if self.usingIE:
            self.handling_error = False
            self.document_missing = False
            self.error_args = None
        return wv

    def _make_statusbar(self):
        statusbar = self.CreateStatusBar()
        return statusbar

    def _go_forward(self, evt):
        if self.webview.CanGoForward():
            self.webview.GoForward()

    def _go_back(self, evt):
        if self.webview.CanGoBack():
            self.webview.GoBack()

    def _go_home(self, evt):
        self.webview.LoadURL(self.home)

    def _history_report(self, evt=None):
        noprint("_history_report", self.webview.CanGoBack(),
            self.webview.CanGoForward(), file=sys.stderr)
        for item in self.webview.GetBackwardHistory():
            print(" -", item.GetUrl(), file=sys.stderr)
        print(" *", self.webview.GetCurrentURL(), file=sys.stderr)
        for item in self.webview.GetForwardHistory():
            print(" +", item.GetUrl(), file=sys.stderr)
        print(file=sys.stderr)

    def _history_clear(self, evt=None):
        self.webview.ClearHistory()
        #self._update_ui()

    def _update_ui(self, evt=None):
        self.toolbar.EnableTool(self.back_button.GetId(),
                        self.webview.CanGoBack())
        self.toolbar.EnableTool(self.forward_button.GetId(),
                        self.webview.CanGoForward())
        sys.stderr.flush()

    def _exit(self, evt):
        self.Close()

    def _about(self, evt):
        wx.MessageBox("Help Viewer", "Chimera 2 Help Viewer",
                    wx.OK | wx.ICON_INFORMATION)

    def _on_navigating(self, evt):
        noprint("OnNavigating:", repr(evt.GetTarget()),
                    repr(evt.GetURL()),
                    file=sys.stderr)
        from urllib.parse import urlparse
        url = urlparse(evt.GetURL())
        if url.scheme == "chimera2":
            evt.Veto()
            # TODO: do something with the URL
        elif url.scheme == "error":
            if self.error_args:
                self.webview.SetPage(*self.error_args)
                self.error_args = None

    def _on_navigated(self, evt):
        noprint("OnNavigated:", evt.GetTarget(), evt.GetURL(),
                            file=sys.stderr)
        if self.usingIE:
            if not self.handling_error and self.document_missing:
                self.handling_error = True
                self._processMissing(self.document_missing)
                self.handling_error = False
            self.document_missing = False

    def _on_loaded(self, evt):
        self._history_report()
        #self._update_ui()

    def _on_document_error(self, evt):
        noprint("OnDocumentError:", repr(evt.GetTarget()),
                        repr(evt.GetURL()),
                        file=sys.stderr)
        if evt.GetInt() == html2.WEBVIEW_NAV_ERR_NOT_FOUND:
            # When a "not found" error is encountered,
            # WebViewBackendWebKit only sends the error event
            # and nothing else; it is also okay to update the
            # displayed page; so that's what we do here.
            # WebViewBackendIE sends both an error event and
            # a "navigated" event; updating the displayed page
            # here does nothing; so we save the problematic URL
            # and let the "navigated" event handler deal with it.
            badURL = evt.GetURL()
            if not self.usingIE:
                self._processMissing(badURL)
            else:
                self.document_missing = badURL

    def _processMissing(self, url):
        # We handle the case where a file: URL refers
        # to a missing file.  If the URL does not
        # contain the known Chimera 2 prefix, then
        # we display an error.  Otherwise, we use the
        # subsequent path component as the name of a
        # Chimera 2 package and check if it is installed.
        # If so, we replace the URL with one constructed
        # from the help cache and the given URL;
        # if not, we translate it to an http: URL.
        p = self._file_path(url)
        if p is None:
            msg = "<b>%s</b> is missing" % url
            self._show_error(ErrorPage % msg, ErrorURL)
            return
        parts = p.split(HelpPrefix, 1)
        if len(parts) != 2:
            msg = "<b>%s</b> is a path and is missing" % url
            self._show_error(ErrorPage % msg, ErrorURL)
            return
        path_parts = parts[1].split('/', 1)
        if len(path_parts) == 2:
            pkg, path = path_parts
        else:
            pkg = parts[1]
            path = "index.html"
        import cache
        helpdir = cache.get_help_dir(pkg)
        if helpdir is None:
            # TODO: convert to HTTP url if possible
            newURL = "%s%s%s" % (HostPrefix, HelpPrefix, parts[1])
            msg = "using http path <b>%s</b> (%s)" % (newURL, url)
            self._show_error(ErrorPage % msg, ErrorURL)
        else:
            import os.path
            full_path = os.path.join(helpdir, path)
            from urllib.request import pathname2url
            url = "file:" + pathname2url(full_path)
            self.webview.LoadURL(url)

    def _file_path(self, url):
        from urllib.parse import urlparse
        u = urlparse(url)
        if u.scheme == "file":
            return u.path
        # WebViewBackendIE returns a file path
        # rather than a file: URL so we apply
        # the "one-character-scheme-must-be-
        # a-drive-letter" heuristic
        if self.usingIE and len(u.scheme) == 1:
            from urllib.request import pathname2url
            return pathname2url(url)
        return None

    def _show_error(self, html, url):
        self.error_args = (html, url)
        self.webview.LoadURL("error:error")

if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
    del app
