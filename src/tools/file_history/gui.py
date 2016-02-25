# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.tools import ToolInstance

class FilePanel(ToolInstance):

    SESSION_ENDURING = True
    SESSION_SKIP = True
    SIZE = (575, 500)
    help = "help:user/tools/filehistory.html"

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        self.thumbnail_size = (64,64)	# Pixels

        from chimerax.core.ui.gui import MainToolWindow
        class FilesWindow(MainToolWindow):
            close_destroys = False
        self.tool_window = FilesWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area

        from wx import html2
        self.file_history_window = fhw = html2.WebView.New(parent, size=self.SIZE)
        fhw.EnableHistory(False)

        import wx
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(fhw, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")

        fhw.Bind(wx.EVT_CLOSE, self.on_close)
        fhw.Bind(html2.EVT_WEBVIEW_LOADED, self.on_load)
        fhw.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating, id=fhw.GetId())

        from chimerax.core.filehistory import file_history
        file_history(session).remove_missing_files()

        self.update_html()

        t = session.triggers
        t.add_handler('file history changed', self.file_history_changed_cb)

    def history_html(self):
        from chimerax.core.filehistory import file_history
        files = file_history(self.session).files
        if len(files) == 0:
            html = '<html><body>No files in history</body></html>'
        else:
            lines = ['<html>', '<body>', '<style>', 'table { float:left; }', '</style>']
            w,h = self.thumbnail_size
            for f in reversed(files):
                name = f.short_name()
                import html
                cmd = html.escape(f.open_command())
                if f.image is None:
                    line = '<a href="cxcmd:%s">%s</a>' % (cmd, name)
                else:
                    line = ('<table>'
                            '<tr><td><a href="cxcmd:%s"><img src="data:image/jpeg;base64,%s" width=%d height=%d></a>'
                            '<tr><td align=center><a href="cxcmd:%s">%s</a>'
                            '</table></a>'
                            % (cmd, f.image, w, h, cmd, name))
                lines.append(line)
            lines.extend(['</body>', '</html>'])
            html = '\n'.join(lines)
        return html

    def update_html(self):
        html = self.history_html()
        self.file_history_window.SetPage(html, "")

    def file_history_changed_cb(self, name, data):
        # TODO: Only update if window shown.
        self.update_html()

    # wx event handling

    def on_close(self, event):
        pass

    def on_load(self, event):
        pass

    def on_navigating(self, event):

        url = event.GetURL()
        from urllib.parse import unquote
        url = unquote(url)
        if url.startswith("cxcmd:"):
            event.Veto()
            cmd = url.split(':', 1)[1]
            from chimerax.core.commands import run
            run(self.session, cmd)
        elif url == 'file:///':
            # show_page_source() causes this
            pass
        else:
            # unknown scheme
            event.Veto()
            self.session.logger.error("Unknown URL scheme: '%s'" % url)

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        return 1, None

    @classmethod
    def restore_snapshot_new(cls, session, bundle_info, version, data):
        pass

    def restore_snapshot_init(self, session, bundle_info, version, data):
        pass

    def reset_state(self, session):
        pass
