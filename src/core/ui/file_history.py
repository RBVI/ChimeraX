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

class FileHistory:

    def __init__(self, session, parent, bg_color=None, thumbnail_size=(64,64), filename_size=8,
            no_hist_text='<html><body>No files in history</body></html>', **kw):
        self.thumbnail_size = thumbnail_size	# Pixels
        self.filename_size = filename_size	# Characters
        self._default_image = None
        self._default_image_format = None
        self.session = session
        self.bg_color = bg_color
        self.no_hist_text = no_hist_text

        self.file_history_window = fhw = HistoryWindow(session, parent, **kw)

        from PyQt5.QtWidgets import QGridLayout, QErrorMessage
        layout = QGridLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.file_history_window, 0, 0)
        parent.setLayout(layout)

        self.update_html()

        t = session.triggers
        t.add_handler('file history changed', self.file_history_changed_cb)

    def history_html(self):
        from ..filehistory import file_history
        files = file_history(self.session).files
        if len(files) == 0:
            html = self.no_hist_text
        else:
            lines = ['<html>', '<body>', '<style>', 'table { float:left; }']
            if self.bg_color:
                lines.extend(['body {', '    background-color: %s;' % self.bg_color, '}'])
            lines.append('</style>')
            w,h = self.thumbnail_size
            # TODO: Qt 5.6.0 bug in QT-53414 makes web content > 2 Mbytes not display.
            # Work-around code saves html to temp file.  Still limit html to < 50 Mbytes for performance.
            hbytes, max_bytes = 0, 50000000
            for fi, f in enumerate(reversed(files)):
                name = limit_string(f.short_name(), self.filename_size)
                descrip = f.path if f.database is None else '%s %s' % (f.database.upper(), f.path)
                from urllib import parse
                cmd = parse.quote(f.open_command())
                i = self.default_image('JPEG') if f.image is None or hbytes > max_bytes else f.image
                img = '<img src="data:image/jpeg;base64,%s" width=%d height=%d title="%s">' % (i, w, h, descrip)
                line = ('<table>'
                        '<tr><td><a href="cxcmd:%s">%s</a>'
                        '<tr><td align=center><a href="cxcmd:%s" title="%s">%s</a>'
                        '</table></a>'
                        % (cmd, img, cmd, descrip, name))
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
        fhw.setHtml(html)
#        fhw.setUrl(QUrl('file:///Users/goddard/Desktop/test.html'))  # Works with > 2Mb history html

    def file_history_changed_cb(self, name, data):
        # TODO: Only update if window shown.
        self.update_html()

def limit_string(s, n):
    if len(s) > n:
        return s[:n//2] + '...' + s[-(n//2):]
    return s

from .widgets import ChimeraXHtmlView
class HistoryWindow(ChimeraXHtmlView):
    def __init__(self, session, parent, **kw):
        super().__init__(session, parent, **kw)

        # Don't record html history as log changes.
        self.loadFinished.connect(self.clear_history)
        
    def clear_history(self, okay):
        self.history().clear()
        
    def contextMenuEvent(self, event):
        event.accept()
        cm = getattr(self, 'context_menu', None)
        if cm is None:
            from PyQt5.QtWidgets import QMenu
            cm = self.context_menu = QMenu()
            cm.addAction("Remove deleted files", self.remove_missing_files)
        cm.popup(event.globalPos())

    def remove_missing_files(self):
        from ..filehistory import file_history
        file_history(self.session).remove_missing_files()
