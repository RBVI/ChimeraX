# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.tools import ToolInstance

class FilePanel(ToolInstance):

    SESSION_ENDURING = True
    SESSION_SKIP = True
    SIZE = (575, 200)
    help = "help:user/tools/filehistory.html"

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        self.thumbnail_size = (64,64)	# Pixels
        self._default_image = None
        self._default_image_format = None

        from chimerax.core.ui.gui import MainToolWindow
        class FilesWindow(MainToolWindow):
            close_destroys = False

        from chimerax.core import window_sys
        self.window_sys = window_sys
        if self.window_sys == "wx":
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
        else: # Qt
            self.tool_window = FilesWindow(self)
            parent = self.tool_window.ui_area

            from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
            class HtmlWindow(QWebEngineView):
                def sizeHint(self):
                    from PyQt5.QtCore import QSize
                    return QSize(*FilePanel.SIZE)
            self.file_history_window = fhw = HtmlWindow(parent)
            # TODO: Don't take focus away from command-line.  This doesn't work with QWebEngineView, QT bug 52999.
            # from PyQt5.QtCore import Qt
            # fhw.setFocusPolicy(Qt.NoFocus)
            fhw.setEnabled(False)	# Prevent file history panel from taking keyboard focus.
            # Don't record html history as log changes.
            def clear_history(okay, fhw=fhw):
                fhw.history().clear()
            fhw.loadFinished.connect(clear_history)

            from PyQt5.QtWidgets import QGridLayout, QErrorMessage
            layout = QGridLayout(parent)
            layout.setContentsMargins(0,0,0,0)
            layout.addWidget(self.file_history_window, 0, 0)
            parent.setLayout(layout)
            self.tool_window.manage(placement="right")

            # TODO: The following link click binding is not working in Qt 5.6.
            # Instead the link dispatching is going through core/ui/gui.py handling href="cxcmd:<command>".
            def link_clicked(qurl, nav_type, is_main_frame):
                self.navigate(qurl.toString())
                return False
            fhw.page().acceptNavigationRequest = link_clicked

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
            # TODO: Qt 5.6.0 bug in QT-53414 makes web content > 2 Mbytes not display.
            hbytes, max_bytes = 0, 1500000
            for fi, f in enumerate(reversed(files)):
                name = limit_string(f.short_name(), 8)
                import html
                cmd = html.escape(f.open_command())
                if self.window_sys == 'qt':
                    # TODO: JPEG inline images cause page to be blank.
                    i = self.default_image('PNG') if f.image is None or hbytes > max_bytes else image_jpeg_to_png(f.image, (w,h))
                    img = '<img src="data:image/png;base64,%s" width=%d height=%d>' % (i, w, h)
                elif self.window_sys == 'wx':
                    i = self.default_image() if f.image is None else f.image
                    img = '<img src="data:image/jpeg;base64,%s" width=%d height=%d>' % (i, w, h)
                line = ('<table>'
                        '<tr><td><a href="cxcmd:%s">%s</a>'
                        '<tr><td align=center><a href="cxcmd:%s">%s</a>'
                        '</table></a>'
                        % (cmd, img, cmd, name))
                lines.append(line)
                hbytes += len(line)
            lines.extend(['</body>', '</html>'])
            html = '\n'.join(lines)
        return html

    def default_image(self, format = 'JPEG'):
        if self._default_image is None or self._default_image_format != format:
            from PIL import Image
            w,h = self.thumbnail_size
            i = Image.new("RGB", (w,h), "gray")
            import io
            im = io.BytesIO()
            i.save(im, format=format)
            bytes = im.getvalue()
            import codecs
            self._default_image = codecs.encode(bytes, 'base64').decode('utf-8')
            self._default_image_format = format
        return self._default_image
        
    def update_html(self):
        html = self.history_html()
        fhw = self.file_history_window
        if self.window_sys == 'wx':
            fhw.SetPage(html, "")
        else:
            fhw.setHtml(html)
#            fhw.setUrl(QUrl('file:///Users/goddard/Desktop/test.html'))  # Works with > 2Mb history html

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
        if url == 'file:///':
            # show_page_source() causes this in wx
            return
        event.Veto()
        self.navigate(url)

    def navigate(self, url):
        from urllib.parse import unquote
        url = unquote(url)
        if url.startswith("cxcmd:"):
            cmd = url.split(':', 1)[1]
            from chimerax.core.commands import run
            run(self.session, cmd)
        else:
            # unknown scheme
            self.session.logger.error("Unknown URL scheme: '%s'" % url)

def image_jpeg_to_png(image_jpeg_base64, size = None):
    '''Convert base64 encoded jpeg image to base64 encode PNG image.'''
    import codecs
    image_jpeg = codecs.decode(image_jpeg_base64.encode('utf-8'), 'base64')
    import io
    img_io = io.BytesIO(image_jpeg)
    from PIL import Image
    i = Image.open(img_io)
    if size is not None:
        i = i.resize(size)
    png_io = io.BytesIO()
    i.save(png_io, format='PNG')
    png_bytes = png_io.getvalue()
    image_png_base64 = codecs.encode(png_bytes, 'base64').decode('utf-8')
    return image_png_base64

def limit_string(s, n):
    if len(s) > n:
        return s[:n//2] + '...' + s[-(n//2):]
    return s
