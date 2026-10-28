# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from Qt.QtCore import Qt, QSize, QRect, QPoint
from Qt.QtWidgets import QWidget, QLabel, QLayout, QVBoxLayout


class FileHistory:

    def __init__(self, session, parent, bg_color=None, thumbnail_size=(64,64), filename_size=8,
            no_hist_text=None, **kw):
        self.thumbnail_size = thumbnail_size	# Pixels
        self.filename_size = filename_size	# Characters
        show_full = session.ui.settings.file_history_show_full_filenames
        self._show_full_filenames = show_full
        self._default_image = None
        self._default_image_format = None
        self.session = session
        self.bg_color = bg_color
        if no_hist_text:
            self.no_hist_text = no_hist_text
        else:
            self.no_hist_text = 'No files in history'

        self.file_history_window = fhw = HistoryWindow(session, parent, **kw)

        from Qt.QtWidgets import QGridLayout
        layout = QGridLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.file_history_window, 0, 0)
        parent.setLayout(layout)

        self.update_html()

        t = session.triggers
        t.add_handler('file history changed', self.file_history_changed_cb)

    def history_html(self):
        from chimerax.core.filehistory import file_history
        files = file_history(self.session).files
        if len(files) == 0:
            html = self.no_hist_text
        else:
            lines = ['<style>', 'table { float:left; }']
            if self.bg_color:
                lines.extend(['body {', '    background-color: %s;' % self.bg_color, '}'])
            lines.append('</style>')
            w,h = self.thumbnail_size
            # TODO: Qt 5.6.0 bug in QT-53414 makes web content > 2 Mbytes not display.
            # Work-around code saves html to temp file.  Still limit html to < 50 Mbytes for performance.
            hbytes, max_bytes = 0, 50000000
            for fi, f in enumerate(reversed(files)):
                name = f.short_name()
                if not self._show_full_filenames:
                    name = limit_string(name, self.filename_size)
                descrip = f.path if f.database is None else '%s %s' % (f.database.upper(), f.path)
                from urllib import parse
                cmd = parse.quote(f.open_command())
                i = self.default_image('JPEG') if f.image is None or hbytes > max_bytes else f.image
                img = '<img src="data:image/jpeg;base64,%s" width=%d height=%d title="%s" draggable="false">' % (i, w, h, descrip)
                line = ('<table>'
                        '<tr><td><a href="cxcmd:%s">%s</a>'
                        '<tr><td align=center><a href="cxcmd:%s" title="%s">%s</a>'
                        '</table></a>'
                        % (cmd, img, cmd, descrip, name))
                lines.append(line)
                hbytes += len(line)
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
        html= f'<html><style>{self.session.ui.dark_css()}</style><body>' + self.history_html() + '</body></html>'
        fhw = self.file_history_window
        fhw.setHtml(html)
#        fhw.setUrl(QUrl('file:///Users/goddard/Desktop/test.html'))  # Works with > 2Mb history html

    def file_history_changed_cb(self, name, data):
        # TODO: Only update if window shown.
        self.update_html()

    def show_full_file_names(self, show):
        self._show_full_filenames = show
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
            from Qt.QtWidgets import QMenu
            cm = self.context_menu = QMenu(self)
            self._show_full_file_names = a = cm.addAction("Show full file names")
            a.setCheckable(True)
            a.triggered.connect(self._show_full_file_name_changed)
            cm.addAction("Remove deleted files", self.remove_missing_files)
            cm.addAction("Clear file history", self.clear_file_history)
        show_full = self.session.ui.settings.file_history_show_full_filenames
        self._show_full_file_names.setChecked(show_full)
        cm.popup(event.globalPos())

    def _show_full_file_name_changed(self, show):
        self.session.ui.settings.file_history_show_full_filenames = show
        self.session.ui.main_window.fh.show_full_file_names(show)
        
    def remove_missing_files(self):
        from chimerax.core.filehistory import file_history
        file_history(self.session).remove_missing_files()

    def clear_file_history(self):
        from chimerax.core.filehistory import file_history
        file_history(self.session).clear_file_history()

class TileArea(QWidget):
    '''Thumbnails laid out in rows that wrap to the widget width.'''
    def __init__(self, file_history, files, used_pixmaps):
        super().__init__()
        self._layout = layout = FlowLayout(self, spacing = 8)
        layout.setContentsMargins(4,0,4,0)
        self.tiles = [Tile(file_history, fs, used_pixmaps) for fs in files]
        for t in self.tiles:
            layout.addWidget(t)

class Tile(QWidget):
    '''Thumbnail image and file name.  Click to open.'''
    def __init__(self, file_history, fspec, used_pixmaps):
        super().__init__()
        self._file_history = file_history
        self.file_spec = fspec
        self._press_pos = None
        self._hover = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        descrip = fspec.path if fspec.database is None else '%s %s' % (fspec.database.upper(), fspec.path)
        self.setToolTip(descrip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3,3,3,3)
        layout.setSpacing(2)
        image = QLabel()
        image.setPixmap(file_history.thumbnail(fspec, used_pixmaps))
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image)
        name = QLabel(file_history.display_name(fspec))
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from Qt.QtGui import QPalette
        pal = name.palette()
        pal.setColor(QPalette.ColorRole.WindowText, pal.color(QPalette.ColorRole.Link))
        name.setPalette(pal)
        layout.addWidget(name)

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        if self._hover:
            from Qt.QtGui import QPainter, QPalette
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            c = self.palette().color(QPalette.ColorRole.Highlight)
            c.setAlpha(70)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(self.rect(), 4, 4)
            p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        pressed = self._press_pos is not None
        self._press_pos = None
        if (pressed and event.button() == Qt.MouseButton.LeftButton
                and self.rect().contains(event.position().toPoint())):
            self._file_history.open_files([self.file_spec])

class EmptyHistoryWindow(QWidget):
    '''Shown when there is no file history.  Optionally a large ChimeraX banner.'''
    def __init__(self, file_history, show_banner):
        super().__init__()
        self._file_history = file_history
        self._title = self._link = None
        layout = QVBoxLayout(self)
        if show_banner:
            from chimerax.core.colors import scheme_color
            from Qt.QtGui import QPalette, QColor
            self._title = title = QLabel('ChimeraX')
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pal = title.palette()
            pal.setColor(QPalette.ColorRole.WindowText, QColor(scheme_color('new_user_canvas_text')))
            title.setPalette(pal)
            from Qt.QtWidgets import QGraphicsDropShadowEffect
            shadow = QGraphicsDropShadowEffect(title)
            shadow.setBlurRadius(2)
            shadow.setOffset(0, 0)
            shadow.setColor(QColor('black'))
            title.setGraphicsEffect(shadow)
            self._link = link = QLabel('<a href="help:quickstart">Get started</a>')
            link.setAlignment(Qt.AlignmentFlag.AlignCenter)
            link.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
            link.linkActivated.connect(self._get_started)
            layout.addStretch(3)
            layout.addWidget(title)
            layout.addWidget(link)
            layout.addStretch(4)
        else:
            label = QLabel('No files in history')
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def _get_started(self, url):
        from chimerax.core.commands import run
        run(self._file_history.session, 'help %s' % url)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Scale banner text with window width.
        w = event.size().width()
        for label, frac in ((self._title, 0.1), (self._link, 0.02)):
            if label is not None:
                f = label.font()
                f.setPixelSize(max(12, int(frac * w)))
                label.setFont(f)

    def contextMenuEvent(self, event):
        event.accept()
        self._file_history.show_context_menu(event.globalPos())

class FlowLayout(QLayout):
    '''Lay out widgets left to right wrapping to new rows.'''
    def __init__(self, parent = None, spacing = 6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only = True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only = False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_height = r.x(), r.y(), 0
        spacing = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            if x > r.x() and x + hint.width() > r.right() + 1:
                x = r.x()
                y += row_height + spacing
                row_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + spacing
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y() + m.bottom()
