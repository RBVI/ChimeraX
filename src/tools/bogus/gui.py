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
#
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance


class bogusUI(ToolInstance):

    SIZE = (500, 25)

    _PageTemplate = """<html>
<head>
<title>Select Models</title>
<script>
function action(button) { window.location.href = "bogus:_action:" + button; }
</script>
<style>
.refresh { color: blue; font-size: 80%; font-family: monospace; }
</style>
</head>
<body>
<h2>Select Models
    <a href="bogus:_refresh" class="refresh">refresh</a></h2>
MODEL_SELECTION
<p>
ACTION_BUTTONS
</body>
</html>"""

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        self.display_name = "Open Models"
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        from wx import html2
        import wx
        self.webview = html2.WebView.New(parent, wx.ID_ANY, size=self.SIZE)
        self.webview.Bind(html2.EVT_WEBVIEW_NAVIGATING,
                          self._on_navigating,
                          id=self.webview.GetId())
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.webview, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")
        # Add triggers for model addition/removal
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        self._handlers = [session.triggers.add_handler(ADD_MODELS,
                                                       self._make_page),
                          session.triggers.add_handler(REMOVE_MODELS,
                                                       self._make_page)]
        self._make_page()

    def _make_page(self, *args):
        models = self.session.models
        from io import StringIO
        page = self._PageTemplate

        # Construct model selector
        s = StringIO()
        print("<select multiple=\"1\">", file=s)
        for model in models.list():
            name = '.'.join([str(n) for n in model.id])
            print("<option value=\"%s\">%s</option>" % (name, name), file=s)
        print("</select>", file=s)
        page = page.replace("MODEL_SELECTION", s.getvalue())

        # Construct action buttons
        s = StringIO()
        for action in ["BLAST"]:
            print("<button type=\"button\""
                  "onclick=\"action('%s')\">%s</button>" % (action, action),
                  file=s)
        page = page.replace("ACTION_BUTTONS", s.getvalue())

        # Update display
        self.webview.SetPage(page, "")

    def _on_navigating(self, event):
        session = self.session
        # Handle event
        url = event.GetURL()
        if url.startswith("bogus:"):
            event.Veto()
            parts = url.split(':')
            method = getattr(self, parts[1])
            args = parts[2:]
            method(session, *args)

    #
    # Callbacks from HTML
    #
    def _refresh(self, session):
        self._make_page()

    def _action(self, session, action):
        print("bogus action button clicked: %s" % action)
