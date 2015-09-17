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


def _bitmap(filename, size):
    import os
    import wx
    image = wx.Image(os.path.join(os.path.dirname(__file__), filename))
    image = image.Scale(size.width, size.height, wx.IMAGE_QUALITY_HIGH)
    result = wx.Bitmap(image)
    return result


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
        self.home_page = None
        from chimera.core.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        import wx
        # buttons: back, forward, reload, stop, home, search bar
        self.toolbar = wx.ToolBar(parent, wx.ID_ANY,
                                  style=wx.TB_DEFAULT_STYLE | wx.TB_TEXT)
        bitmap_size = wx.ArtProvider.GetNativeSizeHint(wx.ART_TOOLBAR)
        self.back = self.toolbar.AddTool(
            wx.ID_ANY, 'Back', _bitmap('back.png', bitmap_size),
            shortHelp="Go back to previously viewed page")
        self.toolbar.EnableTool(self.back.GetId(), False)
        self.forward = self.toolbar.AddTool(
            wx.ID_ANY, 'Forward', _bitmap('forward.png', bitmap_size),
            shortHelp="Go forward to previously viewed page")
        self.toolbar.EnableTool(self.forward.GetId(), False)
        self.home = self.toolbar.AddTool(
            wx.ID_ANY, 'Home', _bitmap('home.png', bitmap_size),
            shortHelp="Return to first page")
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
        from wx import html2
        self.help_window = html2.WebView.New(parent, size=self.SIZE)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        sizer.Add(self.help_window, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement=None)
        # Add to running tool list for session if tool should be saved
        # in and restored from session and scenes
        session.tools.add([self])
        self.help_window.Bind(wx.EVT_CLOSE, self.on_close)
        self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATED, self.on_navigated)
        self.help_window.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating,
                              id=self.help_window.GetId())
        self.help_window.EnableContextMenu()

    def show(self, url, set_home=True):
        self.help_window.Stop()
        if set_home or not self.home_page:
            self.help_window.ClearHistory()
            self.home_page = url
            self.toolbar.EnableTool(self.home.GetId(), True)
            self.toolbar.EnableTool(self.back.GetId(), False)
            self.toolbar.EnableTool(self.forward.GetId(), False)
        self.help_window.LoadURL(url)

    # wx event handling

    def on_back(self, event):
        self.help_window.GoBack()

    def on_forward(self, event):
        self.help_window.GoForward()

    def on_home(self, event):
        self.show(self.home_page, set_home=False)

    def on_close(self, event):
        self.session.logger.remove_log(self)

    def on_navigated(self, event):
        self.toolbar.EnableTool(self.back.GetId(),
                                self.help_window.CanGoBack())
        self.toolbar.EnableTool(self.forward.GetId(),
                                self.help_window.CanGoForward())

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
        from urllib.parse import urlparse
        parts = urlparse(url)
        if parts.scheme == 'file':
            pass

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
