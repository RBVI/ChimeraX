# vi: set expandtab shiftwidth=4 softtabstop=4:

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


class bogusUI(ToolInstance):

    SIZE = (500, 25)
    VERSION = 1

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

    def __init__(self, session, tool_info):
        super().__init__(session, tool_info)

        self.display_name = "Open Models"
        from chimera.core.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        from wx import html2
        import wx
        self.webview = html2.WebView.New(parent, wx.ID_ANY, size=self.SIZE)
        self.webview.Bind(html2.EVT_WEBVIEW_NAVIGATING,
                          self._OnNavigating,
                          id=self.webview.GetId())
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.webview, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")
        # Add triggers for model addition/removal
        from chimera.core.models import ADD_MODELS, REMOVE_MODELS
        self._handlers = [session.triggers.add_handler(ADD_MODELS,
                                                       self._make_page),
                          session.triggers.add_handler(REMOVE_MODELS,
                                                       self._make_page)]
        # Add to running tool list for session (not required)
        session.tools.add([self])
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
        for action in [ "BLAST" ]:
            print("<button type=\"button\""
                  "onclick=\"action('%s')\">%s</button>" % (action, action),
                  file=s)
        page = page.replace("ACTION_BUTTONS", s.getvalue())

        # Update display
        self.webview.SetPage(page, "")

    def _OnNavigating(self, event):
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

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        version = self.VERSION
        data = {}
        return [version, data]

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

    def reset_state(self):
        pass
