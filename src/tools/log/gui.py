# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.tools import ToolInstance
from chimerax.core.logger import HtmlLog

context_menu_css = """
.context-menu {
    display: none;
    position: absolute;
    z-index: 100;
    border: solid 1px #000000;
    box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5);
    cursor: pointer;
}
.context-menu-items {
    list-style-type: none;
    padding: 0;
    margin: 0;
}
.context-menu-item a:link, a:visited {
    display: block;
    color: #000;
    background-color: #fff;
    opacity: 1;
    text-decoration: none;
    padding: 4px 8px;
}
.context-menu-item a:hover, a:active {
    background-color: #cccccc;
}
.context-menu-active {
    display: block;
}
"""

cxcmd_css = """
.cxcmd {
    display: block;
    font-weight: bold;
    margin-top: .5em;
    background-color: #ddd;
}
"""

context_menu_html = """
<nav id="context-menu" class="context-menu">
    <ul class="context-menu-items">
        <!--
        <li class="context-menu-item">
            <a href="log:copy" class="context-menu-link"> Copy </a>
        </li>
        -->
        <li class="context-menu-item">
            <a href="log:image" class="context-menu-link"> Insert image </a>
        </li>
        <li class="context-menu-item">
            <a href="log:save" class="context-menu-link"> Save </a>
        </li>
        <li class="context-menu-item">
            <a href="log:clear" class="context-menu-link"> Clear </a>
        </li>
        <hr style="margin:0;">
        <li class="context-menu-item">
            <a href="log:help" class="context-menu-link"> Help </a>
        </li>
    </ul>
</nav>
"""

context_menu_script = """
(function() {
    "use strict";

    var context_menu = document.querySelector(".context-menu");
    var context_menu_shown = false;
    var active_css = "context-menu-active";

    function show_context_menu() {
        if (!context_menu_shown) {
            context_menu_shown = true;
            context_menu.classList.add(active_css);
        }
    }

    function hide_context_menu() {
        if (context_menu_shown) {
            context_menu_shown = false;
            context_menu.classList.remove(active_css);
        }
    }

    function position_menu(menu, e) {
        var x = e.pageX;
        var y = e.pageY;

        menu.style.left = x + "px";
        menu.style.top = y + "px";
    }

    function init() {
        document.addEventListener("contextmenu", function (e) {
                e.preventDefault();
                show_context_menu();
                position_menu(context_menu, e);
        });

        document.addEventListener("click", function (e) {
                var button = e.which;
                if (button === 1)	// left button used
                        hide_context_menu();
        });

        context_menu.addEventListener("mouseleave", hide_context_menu);
    }

    init();
})();
"""


class Log(ToolInstance, HtmlLog):

    SESSION_ENDURING = True
    SIZE = (575, 500)
    help = "help:user/tools/log.html"

    def __init__(self, session, bundle_info, *, restoring=False):
        if not restoring:
            ToolInstance.__init__(self, session, bundle_info)
        self.warning_shows_dialog = True
        self.error_shows_dialog = True
        from chimerax.core.ui import MainToolWindow

        class LogWindow(MainToolWindow):
            close_destroys = False
        self.tool_window = LogWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        import wx
        wx.FileSystem.AddHandler(wx.MemoryFSHandler())
        from itertools import count
        self._image_count = count()
        from wx import html2
        self.log_window = html2.WebView.New(parent, size=self.SIZE)
        self.log_window.EnableContextMenu(True)
        self.log_window.EnableHistory(False)
        self.page_source = ""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.log_window, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")
        session.logger.add_log(self)
        self.log_window.Bind(wx.EVT_CLOSE, self.on_close)
        self.log_window.Bind(html2.EVT_WEBVIEW_LOADED, self.on_load)
        self.log_window.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating,
                             id=self.log_window.GetId())
        # self.log_window.SetPage(self.page_source, "")
        self.show_page_source()

    #
    # Implement logging
    #
    def log(self, level, msg, image_info, is_html):
        """Log a message

        Parameters documented in HtmlLog base class
        """

        import wx
        image, image_break = image_info
        if image:
            import io
            img_io = io.BytesIO()
            image.save(img_io, format='PNG')
            png_data = img_io.getvalue()
            import codecs
            bitmap = codecs.encode(png_data, 'base64')
            width, height = image.size
            img_src = '<img src="data:image/png;base64,%s" width=%d height=%d style="vertical-align:middle">' % (bitmap.decode('utf-8'), width, height)
            self.page_source += img_src
            if image_break:
                self.page_source += "<br>\n"
        else:
            if ((level == self.LEVEL_ERROR and self.error_shows_dialog) or
                    (level == self.LEVEL_WARNING and self.warning_shows_dialog)):
                if level == self.LEVEL_ERROR:
                    caption = "ChimeraX Error"
                    icon = wx.ICON_ERROR
                else:
                    caption = "ChimeraX Warning"
                    icon = wx.ICON_EXCLAMATION
                style = wx.OK | wx.OK_DEFAULT | icon | wx.CENTRE
                graphics = self.session.ui.main_window.graphics_window
                if is_html:
                    from chimerax.core.logger import html_to_plain
                    dlg_msg = html_to_plain(msg)
                else:
                    dlg_msg = msg
                if dlg_msg.count('\n') > 20:
                    # avoid excessively high error dialogs where
                    # both the bottom buttons and top controls
                    # may be off the screen!
                    lines = dlg_msg.split('\n')
                    dlg_msg = '\n'.join(lines[:10] + ["..."] + lines[-10:])
                dlg = wx.MessageDialog(graphics, dlg_msg,
                                       caption=caption, style=style)
                dlg.ShowModal()

            if not is_html:
                from html import escape
                msg = escape(msg)
                msg = msg.replace("\n", "<br>\n")

            if level == self.LEVEL_ERROR:
                msg = '<font color="red">' + msg + '</font>'
            elif level == self.LEVEL_WARNING:
                msg = '<font color="red">' + msg + '</font>'

            self.page_source += msg
        self.show_page_source()
        return True

    def show_page_source(self):
        css = context_menu_css + cxcmd_css
        self.log_window.SetPage("<style>%s</style>%s\n%s" % (
            css, context_menu_html, self.page_source), "")

    # wx event handling

    def on_close(self, event):
        self.session.logger.remove_log(self)

    def on_load(self, event):
        # scroll to bottom
        self.log_window.RunScript(
            "window.scrollTo(0, document.body.scrollHeight);")
        # setup context menu
        self.log_window.RunScript(context_menu_script)

    def on_navigating(self, event):
        session = self.session
        # Handle event
        url = event.GetURL()
        from urllib.parse import unquote
        url = unquote(url)
        if url.startswith("log:"):
            event.Veto()
            cmd = url.split(':', 1)[1]
            if cmd == 'help':
                self.display_help()
            elif cmd == 'clear':
                self.clear()
            elif cmd == 'copy':
                pass  # TODO
            elif cmd == 'save':
                from chimerax.core.ui.open_save import SaveDialog
                save_dialog = SaveDialog(
                    self.log_window, "Save Log", defaultFile="log",
                    wildcard="HTML files (*.html)|*.html",
                    add_extension=".html")
                import wx
                if save_dialog.ShowModal() == wx.ID_CANCEL:
                    return
                filename = save_dialog.GetPath()
                self.save(filename)
            elif cmd == 'image':
                from .cmd import log
                log(self.session, thumbnail=True)
            return
        elif url.startswith("cxcmd:"):
            from chimerax.core.commands import run
            event.Veto()
            cmd = url.split(':', 1)[1]
            run(session, cmd)
            return
        from urllib.parse import urlparse
        parts = urlparse(url)
        if parts.scheme in ('', 'help', 'file', 'http'):
            if parts.path == '/':
                # Ingore file:/// URL event that Mac generates
                # for each call to SetPage()
                return
            event.Veto()
            from chimerax.core.commands import run
            run(session, "help %s" % url, log=False)
            return
        # unknown scheme
        event.Veto()
        session.logger.error("Unknown URL scheme: '%s'" % parts.scheme)

    def clear(self):
        self.page_source = ""
        self.show_page_source()

    def save(self, path):
        from os.path import expanduser
        path = expanduser(path)
        f = open(path, 'w')
        f.write("<!DOCTYPE html>\n"
                "<html>\n"
                "<head>\n"
                "<title> ChimeraX Log </title>\n"
                "</head>\n"
                "<body>\n"
                "<h1> ChimeraX Log </h1>\n"
                "<style>\n"
                "%s"
                "</style>\n" % cxcmd_css)
        f.write(self.page_source)
        f.write("</body>\n"
                "</html>\n")
        f.close()

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        data = {"shown": self.tool_window.shown}
        return self.bundle_info.session_write_version, data

    @classmethod
    def restore_snapshot_new(cls, session, bundle_info, version, data):
        from .cmd import get_singleton
        return get_singleton(session)

    def restore_snapshot_init(self, session, bundle_info, version, data):
        if version not in bundle_info.session_versions:
            from chimerax.core.state import RestoreError
            raise RestoreError("unexpected version")
        self.display(data["shown"])

    def reset_state(self, session):
        self.tool_window.shown = True

    #
    # Override ToolInstance methods
    #
    def delete(self):
        self.session.logger.remove_log(self)
        super().delete()
