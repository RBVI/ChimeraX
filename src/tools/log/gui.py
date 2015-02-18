# vim: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance
from chimera.core.logger import HtmlLog


class Log(ToolInstance, HtmlLog):

    SIZE = (300, 500)
    STATE_VERSION = 1

    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("Log", session,
                                      size=self.SIZE, destroy_hides=True)
        parent = self.tool_window.ui_area
        import wx
        wx.FileSystem.AddHandler(wx.MemoryFSHandler())
        from itertools import count
        self._image_count = count()
        # WebView doesn't currently support memory file systems,
        # so use HtmlWindow
        from wx.html import HtmlWindow
        self.log_window = HtmlWindow(parent, size=self.SIZE)
        self.page_source = ""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.log_window, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")
        session.tools.add([self])
        session.logger.add_log(self)

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
            mem_name = "image_{}.png".format(next(self._image_count))
            img_io = io.BytesIO()
            image.save(img_io, format='PNG')
            png_data = img_io.getvalue()
            bitmap = wx.Bitmap.NewFromPNGData(png_data, len(png_data))
            wx.MemoryFSHandler.AddFile(mem_name, bitmap.ConvertToImage(),
                wx.BITMAP_TYPE_PNG)
            w, h = image.size
            self.page_source += '<img src="memory:{}" width={} height={}' \
                ' style="vertical-align:middle">'.format(mem_name, w, h)
            if image_break:
                self.page_source += "<br>"
        else:
            if level in (self.LEVEL_ERROR, self.LEVEL_WARNING):
                if level == self.LEVEL_ERROR:
                    caption = "Chimera 2 Error"
                    icon = wx.ICON_ERROR
                else:
                    caption = "Chimera 2 Warning"
                    icon = wx.ICON_EXCLAMATION
                style = wx.OK | wx.OK_DEFAULT | icon | wx.CENTRE
                graphics = self.session.ui.main_window.graphics_window
                if is_html:
                    from chimera.core.logger import html_to_plain
                    dlg_msg = html_to_plain(msg)
                else:
                    dlg_msg = msg
                dlg = wx.MessageDialog(graphics, dlg_msg,
                    caption=caption, style=style)
                dlg.ShowModal()

            if not is_html:
                from html import escape
                msg = escape(msg)
                msg = msg.replace("\n", "<br>")
            
            if level == self.LEVEL_ERROR:
                msg = '<font color="red">' + msg + '</font>'
            elif level == self.LEVEL_WARNING:
                msg = '<font color="red">' + msg + '</font>'

            self.page_source += msg
        self.log_window.SetPage(self.page_source)
        r = self.log_window.GetScrollRange(wx.VERTICAL)
        self.log_window.Scroll(0, r)
        return True

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        version = self.STATE_VERSION
        data = {"shown": self.tool_window.shown}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import State
        if phase == State.PHASE1:
            # All the action is in phase 2 because we do not
            # want to restore until all objects have been resolved
            pass
        else:
            self.display(data["shown"])

    def reset_state(self):
        self.tool_window.shown = True

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
        """Show or hide log."""
        self.tool_window.shown = b
