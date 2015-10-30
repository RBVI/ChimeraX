# vim: set expandtab shiftwidth=4 softtabstop=4:

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
# Since ToolInstance derives from core.session.State, which
# is an abstract base class, ToolUI classes must implement
#   "take_snapshot" - return current state for saving
#   "restore_snapshot" - restore from given state
#   "reset_state" - reset to data-less state
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimera.core.tools import ToolInstance


class ToolUI(ToolInstance):

    SESSION_ENDURING = False    # default
    SIZE = (500, 25)
    VERSION = 1

    def __init__(self, session, tool_info):
        super().__init__(session, tool_info)
        self.display_name = "STL Inspector"
        if session.ui.is_gui:
            from chimera.core.ui import MainToolWindow
            self.tool_window = MainToolWindow(self, size=self.SIZE)
            self.tool_window.manage(placement="right")
            parent = self.tool_window.ui_area
            # Tool specific code
            from wx import html2
            import wx
            self.webview = html2.WebView.New(parent, wx.ID_ANY, size=self.SIZE)
            self.webview.EnableContextMenu(False)
            self.webview.Bind(html2.EVT_WEBVIEW_NAVIGATING,
                              self._on_navigating,
                              id=self.webview.GetId())
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.webview, 1, wx.EXPAND)
            parent.SetSizerAndFit(sizer)
            self._refresh(self.session)
        session.tools.add([self])

    def _make_page(self):
        page_template = "<html><body>%s</body></html>"
        content_list = []
        for ti in self.ti_list:
            m = ti.model()
            n = ti.index()
            s = "<p>Model %s: index %d (of %d): coords %s</p>" % (m, n, m.num_triangles, ti.coords())
            content_list.append(s)
        content_list.append("<a href=\"stl:_refresh\">Refresh</a>")
        page = page_template % '\n'.join(content_list)
        self.webview.SetPage(page, "")

    def _on_navigating(self, event):
        session = self.session
        url = event.GetURL()
        #session.logger.info("_on_navigating: %s" % url)
        if url.startswith("stl:"):
            event.Veto()
            parts = url.split(':')
            method = getattr(self, parts[1])
            args = parts[2:]
            method(session, *args)

    def _refresh(self, session, *args):
        #session.logger.info("_refresh: %s" % repr(args))
        self.ti_list = []
        import random
        from chimera.core.stl import STLModel
        for m in session.models.list(type=STLModel):
            for i in random.sample(range(m.num_triangles), random.randint(1, 10)):
                self.ti_list.append(m.triangle_info(i))
        self._make_page()

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        data = {}
        return data

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import RestoreError
        if version != self.VERSION or len(data) > 0:
            raise RestoreError("unexpected version or data")
        if phase == self.CREATE_PHASE:
            # Restore all basic-type attributes
            pass
        else:
            # Resolve references to objects
            pass

    def reset_state(self, session):
        pass
