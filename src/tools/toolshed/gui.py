# vim: set expandtab ts=4 sw=4:

TestPage = """<html>
<head>
<title>Toolshed Test Page</title>
<script>
function button_test() { window.location.href = "toolshed:button_test:arg"; }
</script>
</head>
<body>
<h1>Toolshed Test Page</h1>
<button onclick="button_test();">Button Test</button>
</body>
</html>"""

class ToolshedUI:

    SIZE = (800, 50)

    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("Toolshed", "General", session)
        parent = self.tool_window.ui_area
        from wx import html2
        import wx
        self.webview = html2.WebView.New(parent, wx.ID_ANY, size=self.SIZE)
        self.webview.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.OnNavigating,
                                                    id=self.webview.GetId())
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.webview, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.webview.SetPage(TestPage, "")
        self.tool_window.manage(placement="right")

    def OnNavigating(self, event):
        session = self._session()  # resolve back reference
        # Handle event
        url = event.GetURL()
        if url.startswith("toolshed:"):
            event.Veto()
            parts = url.split(':')
            method = getattr(self, parts[1])
            args = parts[2:]
            method(session, *args)

    def button_test(self, session, *args):
        session.logger.info("ToolshedUI.button_test: %s" % str(args))
