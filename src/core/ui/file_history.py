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

    def __init__(self, session, parent, bg_color=None, thumbnail_size=(64,64), filename_size=8, **kw):
        self.thumbnail_size = thumbnail_size	# Pixels
        self.filename_size = filename_size	# Characters
        self._default_image = None
        self._default_image_format = None
        self.session = session
        self.bg_color = bg_color

        from .widgets import ChimeraXHtmlView
        self.file_history_window = fhw = ChimeraXHtmlView(session, parent, **kw)
        # Don't record html history as log changes.
        def clear_history(okay, fhw=fhw):
            fhw.history().clear()
        fhw.loadFinished.connect(clear_history)

        from PyQt5.QtWidgets import QGridLayout, QErrorMessage
        layout = QGridLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.file_history_window, 0, 0)
        parent.setLayout(layout)

        from ..filehistory import file_history
        file_history(session).remove_missing_files()

        self.update_html()

        t = session.triggers
        t.add_handler('file history changed', self.file_history_changed_cb)

    def history_html(self):
        from ..filehistory import file_history
        files = file_history(self.session).files
        if len(files) == 0:
            html = '<html><body>No files in history</body></html>'
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
                import html
                cmd = html.escape(f.open_command())
                # TODO: JPEG inline images cause page to be blank.
                i = self.default_image('PNG') if f.image is None or hbytes > max_bytes else image_jpeg_to_png(f.image, (w,h))
                img = '<img src="data:image/png;base64,%s" width=%d height=%d>' % (i, w, h)
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
        fhw.setHtml(html)
#        fhw.setUrl(QUrl('file:///Users/goddard/Desktop/test.html'))  # Works with > 2Mb history html

    def file_history_changed_cb(self, name, data):
        # TODO: Only update if window shown.
        self.update_html()

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
