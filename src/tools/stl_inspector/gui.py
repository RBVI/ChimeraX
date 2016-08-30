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

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance


class ToolUI(ToolInstance):

    SESSION_ENDURING = False    # default
    SIZE = (500, 25)

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        self.display_name = "STL Inspector"
        self.ti_list = []
        if session.ui.is_gui:
            from chimerax.core.ui.gui import MainToolWindow
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
        # session.logger.info("_on_navigating: %s" % url)
        if url.startswith("stl:"):
            event.Veto()
            parts = url.split(':')
            method = getattr(self, parts[1])
            args = parts[2:]
            method(session, *args)

    def _refresh(self, session, *args):
        # session.logger.info("_refresh: %s" % repr(args))
        self.ti_list = []
        import random
        from chimerax.core.stl import STLModel
        for m in session.models.list(type=STLModel):
            for i in random.sample(range(m.num_triangles), random.randint(1, 10)):
                self.ti_list.append(m.triangle_info(i))
        self._make_page()

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        data = {
            'name': self.bundle_info.name,
            'tool state': ToolInstance.take_snapshot(self, session, flags),
            'triangles': self.ti_list,
            'version': self.bundle_info.session_write_version,
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        bundle_info = session.toolshed.find_bundle(data['name'])
        tui = ToolUI(session, bundle_info)
        tui.set_state_from_snapshot(session, data['tool state'])
        tui.ti_list = data['triangles']
        if session.ui.is_gui:
            tui._make_page()
