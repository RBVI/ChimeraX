# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance

_PageTemplate = """<html>
<head>
<title>ChimeraX Toolshed</title>
<script>
function button_test() { window.location.href = "toolshed:button_test:arg"; }
</script>
<style>
.refresh { color: blue; font-size: 80%; font-family: monospace; }
.url { font-size: 80%; }
.install { color: green; font-family: monospace; }
.start { color: green; font-family: monospace; }
.update { color: blue; font-family: monospace; }
.remove { color: red; font-family: monospace; }
.show { color: green; font-family: monospace; }
.hide { color: blue; font-family: monospace; }
.kill { color: red; font-family: monospace; }
.button { padding: 0px 3px 0px 3px; border-radius: 5px; border: solid 1px #000000;
          text-decoration: none; text-shadow: 0 -1px rgba(0, 0, 0, 0.4);
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4), 0 1px 1px rgba(0, 0, 0, 0.2); }
.buttons { width: 25%; text-align: right; }
.name { font-weight: bold; width: 25%; }
.synopsis { }
table { width: 100%; }
th { text-align: left; }
h2 { text-align: center; }
.empty { text-align: center; }
</style>
</head>
<body>
<h2>Running Tools</h2>
RUNNING_TOOLS
<h2>Installed Tools
    <a href="toolshed:refresh_installed" class="refresh">refresh</a></h2>
INSTALLED_TOOLS
<h2>Available Tools
    <span class="url">(from TOOLSHED_URL)</span>
    <a href="toolshed:refresh_available" class="refresh">refresh</a></h2>
AVAILABLE_TOOLS
</body>
</html>"""
_START_LINK = '<a href="toolshed:_start_tool:%s" class="start button">start</a>'
_UPDATE_LINK = '<a href="toolshed:_update_tool:%s" class="update button">update</a>'
_REMOVE_LINK = '<a href="toolshed:_remove_tool:%s" class="remove button">remove</a>'
_INSTALL_LINK = '<a href="toolshed:_install_tool:%s" class="install button">install</a>'
_SHOW_LINK = '<a href="toolshed:_show_tool:%s" class="show button">show</a>'
_HIDE_LINK = '<a href="toolshed:_hide_tool:%s" class="hide button">hide</a>'
_KILL_LINK = '<a href="toolshed:_kill_tool:%s" class="kill button">kill</a>'
_ROW = (
    '<tr>'
    '<td class="buttons">%s</td>'
    '<td class="name">%s</td>'
    '<td class="synopsis">%s</td>'
    '</tr>')
_RUNNING_ROW = (
    '<tr>'
    '<td class="buttons">%s</td>'
    '<td class="name", colspan="0">%s</td>'
    '</tr>')


class ToolshedUI(ToolInstance):

    SESSION_ENDURING = True
    SIZE = (800, 50)

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        from chimerax.core import window_sys
        self.window_sys = window_sys
        if self.window_sys == "wx":
            from wx import html2
            import wx
            self.webview = html2.WebView.New(parent, wx.ID_ANY, size=self.SIZE)
            self.webview.EnableContextMenu(False)
            # self.webview.EnableHistory(False)
            self.webview.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.navigate, id=self.webview.GetId())
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.webview, 1, wx.EXPAND)
            parent.SetSizerAndFit(sizer)
        else:  # qt
            from PyQt5.QtWebKitWidgets import QWebView, QWebPage

            class HtmlWindow(QWebView):

                def sizeHint(self):   # NOQA
                    from PyQt5.QtCore import QSize
                    return QSize(*ToolshedUI.SIZE)

            self.webview = HtmlWindow(parent)
            from PyQt5.QtWidgets import QGridLayout
            layout = QGridLayout(parent)
            layout.addWidget(self.webview, 0, 0)
            parent.setLayout(layout)
            self.webview.page().setLinkDelegationPolicy(QWebPage.DelegateAllLinks)
            self.webview.linkClicked.connect(self.navigate)
        self.tool_window.manage(placement="right")
        from chimerax.core.tools import ADD_TOOL_INSTANCE, REMOVE_TOOL_INSTANCE
        self._handlers = [session.triggers.add_handler(ADD_TOOL_INSTANCE, self._make_page),
                          session.triggers.add_handler(REMOVE_TOOL_INSTANCE, self._make_page)]
        self._make_page()

    def navigate(self, data):
        session = self.session
        # Handle event
        if self.window_sys == "wx":
            # data is wx event
            url = data.GetURL()
            link_handled = data.Veto
        else:
            # data is QUrl
            url = data.toString()

            def link_handled():
                return False
        if url.startswith("toolshed:"):
            link_handled()
            parts = url.split(':')
            method = getattr(self, parts[1])
            args = parts[2:]
            method(session, *args)

    def _make_page(self, *args):
        session = self.session
        ts = session.toolshed
        tools = session.tools
        from io import StringIO
        page = _PageTemplate

        # TODO: handle multiple versions of available tools
        # TODO: add "update" link for installed tools

        # running
        def tool_key(t):
            return t.display_name
        s = StringIO()
        tool_list = tools.list()
        if not tool_list:
            print('<p class="empty">No running tools found.</p>', file=s)
        else:
            print("<table>", file=s)
            for t in sorted(tool_list, key=tool_key):
                show_link = _SHOW_LINK % t.id
                hide_link = _HIDE_LINK % t.id
                kill_link = _KILL_LINK % t.id
                links = "&nbsp;".join([show_link, hide_link, kill_link])
                print(_RUNNING_ROW % (links, t.display_name), file=s)
            print("</table>", file=s)
        page = page.replace("RUNNING_TOOLS", s.getvalue())

        # installed
        def bundle_key(bi):
            return bi.display_name
        s = StringIO()
        bi_list = ts.bundle_info(installed=True, available=False)
        if not bi_list:
            print('<p class="empty">No installed tools found.</p>', file=s)
        else:
            print("<table>", file=s)
            for bi in sorted(bi_list, key=bundle_key):
                start_link = _START_LINK % bi.name
                update_link = _UPDATE_LINK % bi.name
                remove_link = _REMOVE_LINK % bi.name
                links = "&nbsp;".join([start_link, update_link, remove_link])
                print(_ROW % (links, bi.display_name, bi.synopsis), file=s)
            print("</table>", file=s)
        page = page.replace("INSTALLED_TOOLS", s.getvalue())
        installed_bundles = dict([(bi.name, bi) for bi in bi_list])

        # available
        s = StringIO()
        bi_list = ts.bundle_info(installed=False, available=True)
        if not bi_list:
            print('<p class="empty">No available tools found.</p>', file=s)
        else:
            any_shown = False
            for bi in sorted(bi_list, key=bundle_key):
                try:
                    # If this bundle is already installed, do not display it
                    installed = installed_bundles[bi.name]
                    if installed.version == bi.version:
                        continue
                except KeyError:
                    pass
                if not any_shown:
                    print("<table>", file=s)
                    any_shown = True
                link = _INSTALL_LINK % bi.name
                print(_ROW % (link, bi.display_name, bi.synopsis), file=s)
            if any_shown:
                print("</table>", file=s)
            else:
                print('<p class="empty">All available tools are installed.</p>', file=s)
        page = page.replace("AVAILABLE_TOOLS", s.getvalue())
        page = page.replace("TOOLSHED_URL", ts.remote_url)

        if self.window_sys == "wx":
            self.webview.ClearHistory()
            self.webview.SetPage(page, "")
        else:
            self.webview.history().clear()
            self.webview.setHtml(page)

    def refresh_installed(self, session):
        # refresh list of installed tools
        from . import cmd
        cmd.ts_refresh(session, bundle_type="installed")
        self._make_page()

    def refresh_available(self, session):
        # refresh list of available tools
        from . import cmd
        cmd.ts_refresh(session, bundle_type="available")
        self._make_page()

    def _start_tool(self, session, tool_name):
        # start installed tool
        from . import cmd
        cmd.ts_start(session, tool_name)
        self._make_page()

    def _update_tool(self, session, tool_name):
        # update installed tool
        from . import cmd
        cmd.ts_update(session, tool_name)
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

    def _find_tool(self, session, tool_id):
        t = session.tools.find_by_id(int(tool_id))
        if t is None:
            raise RuntimeError("cannot find tool with id \"%s\"" % tool_id)
        return t

    def _show_tool(self, session, tool_id):
        self._find_tool(session, tool_id).display(True)
        self._make_page()

    def _hide_tool(self, session, tool_id):
        self._find_tool(session, tool_id).display(False)
        self._make_page()

    def _kill_tool(self, session, tool_id):
        self._find_tool(session, tool_id).delete()
        self._make_page()

    def button_test(self, session, *args):
        session.logger.info("ToolshedUI.button_test: %s" % str(args))

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, ToolshedUI, 'toolshed')

    #
    # Override ToolInstance methods
    #
    def delete(self):
        session = self.session
        for h in self._handlers:
            session.triggers.delete_handler(h)
        self._handlers = []
        self.tool_window = None
        super().delete()
