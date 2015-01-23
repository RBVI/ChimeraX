# vim: set expandtab ts=4 sw=4:

_PageTemplate = """<html>
<head>
<title>Chimera Toolshed</title>
<script>
function button_test() { window.location.href = "toolshed:button_test:arg"; }
</script>
<style>
.refresh { color: blue; font-size: 80%; font-family: monospace; }
.install { color: green; font-family: monospace; }
.remove { color: red; font-family: monospace; }
</style>
</head>
<body>
<h2>Chimera Toolshed</h2>
<!-- button onclick="button_test();">Button Test</button> -->
<h2>Installed Tools
    <a href="toolshed:refresh_installed" class="refresh">refresh</a></h2>
INSTALLED_TOOLS
<h2>Available Tools
    <a href="toolshed:refresh_available" class="refresh">refresh</a></h2>
AVAILABLE_TOOLS
</body>
</html>"""
_REMOVE_LINK = '<a href="toolshed:_remove_tool:%s" class="remove">remove</a>'
_INSTALL_LINK = '<a href="toolshed:_install_tool:%s" class="install">install</a>'

from chimera.core.toolshed import ToolInstance


class ToolshedUI(ToolInstance):

    SIZE = (800, 50)
    VERSION = 1

    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("Toolshed", "General", session)
        parent = self.tool_window.ui_area
        from wx import html2
        import wx
        self.webview = html2.WebView.New(parent, wx.ID_ANY, size=self.SIZE)
        self.webview.Bind(html2.EVT_WEBVIEW_NAVIGATING,
                          self._OnNavigating,
                          id=self.webview.GetId())
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.webview, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self._make_page()
        self.tool_window.manage(placement="right")
        session.tools.add([self])

    def _OnNavigating(self, event):
        session = self.session
        # Handle event
        url = event.GetURL()
        if url.startswith("toolshed:"):
            event.Veto()
            parts = url.split(':')
            method = getattr(self, parts[1])
            args = parts[2:]
            method(session, *args)

    def _make_page(self):
        session = self.session
        ts = session.toolshed
        from io import StringIO

        # installed
        s = StringIO()
        print("<ul>", file=s)
        ti_list = ts.tool_info(installed=True, available=False)
        if not ti_list:
            print("<li>No installed tools found.</li>", file=s)
        else:
            for ti in ti_list:
                link = _REMOVE_LINK % ti.name
                print("<li>%s - %s. %s</li>"
                      % (ti.display_name, ti.synopsis, link), file=s)
        print("</ul>", file=s)
        page = _PageTemplate.replace("INSTALLED_TOOLS", s.getvalue())

        # TODO: handle multiple versions of same tool

        # available
        s = StringIO()
        ti_list = ts.tool_info(installed=False, available=True)
        print("<ul>", file=s)
        print("<li>Remote URL: %s</li>" % ts.remote_url, file=s)
        if not ti_list:
            print("<li>No available tools found.</li>", file=s)
        else:
            for ti in ti_list:
                link = _INSTALL_LINK % ti.name
                print("<li>%s - %s. %s</li>"
                      % (ti.display_name, ti.synopsis, link), file=s)
        print("</ul>", file=s)
        page = page.replace("AVAILABLE_TOOLS", s.getvalue())

        self.webview.SetPage(page, "")

    def refresh_installed(self, session):
        # refresh list of installed tools
        from . import cmd
        cmd.ts_refresh(session, tool_type="installed")
        self._make_page()

    def refresh_available(self, session):
        # refresh list of available tools
        from . import cmd
        cmd.ts_refresh(session, tool_type="available")
        self._make_page()

    def _remove_tool(self, session, tool_name):
        # remove installed tool
        from . import cmd
        cmd.ts_remove(session, tool_name)
        self._make_page()

    def _install_tool(self, session, tool_name):
        # install available tool
        from . import cmd
        cmd.ts_install(session, tool_name)
        self._make_page()

    def button_test(self, session, *args):
        session.logger.info("ToolshedUI.button_test: %s" % str(args))

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        version = self.VERSION
        data = {"shown": self.tool_window.shown}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.VERSION:
            raise RuntimeError("unexpected version or data")
        from chimera.core.session import State
        if phase == State.PHASE1:
            # Restore all basic-type attributes
            pass
        else:
            # Resolve references to objects
            self.display(data["shown"])

    def reset_state(self):
        pass

    #
    # Override ToolInstance methods
    #
    def delete(self):
        session = self.session
        self.tool_window.shown = False
        self.tool_window.destroy()
        session.tools.remove([self])
        super().delete()

    def display(self, b):
        self.tool_window.shown = b
