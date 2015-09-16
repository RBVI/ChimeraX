# vi: set expandtab shiftwidth=4 softtabstop=4:

# HelpUI should inherit from ToolInstance if they will be
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


def get_singleton(session, create=True):
    running = session.tools.find_by_class(HelpUI)
    if len(running) > 1:
        raise RuntimeError("too many help viewers running")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('help_viewer')
            return HelpUI(session, tool_info)
        else:
            return None
    else:
        return running[0]


class HelpUI(ToolInstance):

    SESSION_ENDURING = False    # default
    SIZE = (500, 500)
    VERSION = 1

    def __init__(self, session, tool_info):
        super().__init__(session, tool_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Tool UI"
        # in this case), so only override if different name desired
        self.display_name = "%s Help Viewer" % session.app_dirs.appname
        from chimera.core.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        import wx
        from wx import html2
        self.help_window = html2.WebView.New(parent, size=self.SIZE)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.help_window, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement=None)
        # Add to running tool list for session if tool should be saved
        # in and restored from session and scenes
        session.tools.add([self])
        self.help_window.Bind(wx.EVT_CLOSE, self.on_close)
        # self.help_window.Bind(html2.EVT_WEBVIEW_LOADED, self.on_load)
        self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating,
                              id=self.help_window.GetId())
        self.help_window.EnableContextMenu()

    def show(self, url):
        self.help_window.Stop()
        self.help_window.ClearHistory()
        self.help_window.LoadURL(url)

    # wx event handling

    def on_close(self, event):
        self.session.logger.remove_log(self)

    def on_load(self, event):
        # TODO: scroll to requested tag
        # scroll to bottom
        self.help_window.RunScript(
            "window.scrollTo(0, document.body.scrollHeight);")

    def on_navigating(self, event):
        session = self.session
        # Handle event
        url = event.GetURL()
        if url.startswith("ch2cmd:"):
            from chimera.core.commands import run
            event.Veto()
            cmd = url.split(':', 1)[1]
            run(session, cmd)
            return
        # TODO: check if http url is within chimera docs
        # TODO: handle missing doc -- redirect to web server

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
