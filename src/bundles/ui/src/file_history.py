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

#
# Thumbnail view of the file history using native Qt widgets.
# Thumbnails can be grouped into user defined sections by dragging
# them or using the context menu.
#
from Qt.QtCore import Qt, QSize, QRect, QPoint, QTimer
from Qt.QtWidgets import (QWidget, QLabel, QLayout, QVBoxLayout, QStackedLayout,
                          QScrollArea, QFrame, QApplication)

_MIME_TYPE = 'application/x-chimerax-file-history-entry'

class FileHistory:

    def __init__(self, session, parent, bg_color=None, thumbnail_size=(64,64), filename_size=8,
            show_banner=False, size_hint=None):
        self.session = session
        self.thumbnail_size = thumbnail_size	# Pixels
        self.filename_size = filename_size	# Characters
        self._pixmaps = {}      # Map base64 JPEG string to QPixmap
        self._default_pixmap = None
        self._selected = set()	# (path, database) of selected files
        self._selection_anchor = None	# (path, database) for shift-click range selection
        self._show_banner = show_banner
        self._bg_color = bg_color

        self._history_window = HistoryWindow(self, size_hint)
        self._empty_window = None

        self._stack = layout = QStackedLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self._history_window)

        if bg_color is not None:
            self.set_background_color(bg_color)

        self.update()

        t = session.triggers
        t.add_handler('file history changed', self.file_history_changed_cb)

    @property
    def history(self):
        from chimerax.core.filehistory import file_history
        return file_history(self.session)

    def update(self):
        h = self.history
        if len(h.files) == 0 and len(h.sections) == 0:
            self._history_window.clear()
            if self._empty_window is None:
                self._empty_window = EmptyHistoryWindow(self, self._show_banner)
                self._stack.addWidget(self._empty_window)
                if self._bg_color is not None:
                    self.set_background_color(self._bg_color)
            self._stack.setCurrentWidget(self._empty_window)
        else:
            self._selected = set(k for k in self._selected if h.find_file(*k) is not None)
            used = {}
            self._history_window.fill(h, used)
            self._pixmaps = used	# Drop pixmaps for thumbnails no longer shown.
            self._stack.setCurrentWidget(self._history_window)

    def file_history_changed_cb(self, name, data):
        self.update()

    def set_background_color(self, color):
        self._bg_color = color
        from Qt.QtGui import QColor, QPalette
        for w in (self._history_window.viewport(), self._empty_window):
            if w is None:
                continue
            w.setAutoFillBackground(True)
            p = w.palette()
            p.setColor(QPalette.ColorRole.Window, QColor(color))
            w.setPalette(p)

    def show_full_file_names(self, show):
        self.session.ui.settings.file_history_show_full_filenames = show
        self.history.history_changed()	# Redraw all file history views

    def display_name(self, fspec):
        name = fspec.short_name()
        if not self.session.ui.settings.file_history_show_full_filenames:
            name = limit_string(name, self.filename_size)
        return name

    def thumbnail(self, fspec, used = None):
        image = fspec.image
        if image is None:
            return self.default_thumbnail()
        pixmap = self._pixmaps.get(image)
        if pixmap is None:
            from Qt.QtGui import QPixmap
            import base64
            pixmap = QPixmap()
            if not pixmap.loadFromData(base64.b64decode(image), 'JPEG'):
                return self.default_thumbnail()
            w,h = self.thumbnail_size
            if pixmap.width() != w or pixmap.height() != h:
                pixmap = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
        if used is not None:
            used[image] = pixmap
        return pixmap

    def default_thumbnail(self):
        if self._default_pixmap is None:
            from Qt.QtGui import QPixmap, QColor
            w,h = self.thumbnail_size
            self._default_pixmap = p = QPixmap(w, h)
            p.fill(QColor('gray'))
        return self._default_pixmap

    def open_files(self, fspecs):
        # Defer so the thumbnail widgets are not rebuilt while handling their mouse event.
        cmds = [fs.open_command() for fs in fspecs]
        from chimerax.core.commands import run
        def open_cb(session = self.session, cmds = cmds):
            for cmd in cmds:
                run(session, cmd)
        QTimer.singleShot(0, open_cb)

    def move_files(self, fspecs, section_name, before = None):
        self.clear_selection()
        try:
            self.history.move_files(fspecs, section_name, before)
        except ValueError as e:
            self.session.logger.error(str(e))

    def remove_files(self, fspecs):
        self.clear_selection()
        self.history.remove_files(fspecs)

    # ------------------------------------------------------------------------
    # Selection.  Ctrl-click (command-click on Mac) toggles a thumbnail,
    # shift-click selects a range, plain click opens the file.
    #
    def is_selected(self, fspec):
        return _key(fspec) in self._selected

    def selected_files(self):
        '''Selected files in display order.'''
        return [t.file_spec for t in self._history_window.tiles if self.is_selected(t.file_spec)]

    def clear_selection(self):
        if self._selected:
            self._selected.clear()
            self._show_selection()
        self._selection_anchor = None

    def tile_clicked(self, tile, modifiers):
        key = _key(tile.file_spec)
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if key in self._selected:
                self._selected.remove(key)
            else:
                self._selected.add(key)
            self._selection_anchor = key
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            keys = [_key(t.file_spec) for t in self._history_window.tiles]
            anchor = self._selection_anchor
            if anchor in keys:
                i0, i1 = sorted((keys.index(anchor), keys.index(key)))
                self._selected = set(keys[i0:i1+1])
            else:
                self._selected = {key}
                self._selection_anchor = key
        else:
            self.clear_selection()
            self.open_files([tile.file_spec])
            return
        self._show_selection()

    def _show_selection(self):
        for t in self._history_window.tiles:
            t.set_selected(self.is_selected(t.file_spec))

    def _action_files(self, tile):
        '''Files a drag or context menu on a tile acts on.'''
        if self.is_selected(tile.file_spec):
            return self.selected_files()
        self.clear_selection()
        return [tile.file_spec]

    def start_drag(self, tile):
        fspecs = self._action_files(tile)
        from Qt.QtCore import QMimeData
        from Qt.QtGui import QDrag
        import json
        md = QMimeData()
        keys = [[fs.path, fs.database] for fs in fspecs]
        md.setData(_MIME_TYPE, json.dumps(keys).encode('utf-8'))
        drag = QDrag(tile)
        drag.setMimeData(md)
        pixmap = self._drag_pixmap(tile.file_spec, len(fspecs))
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width()//2, pixmap.height()//2))
        drag.exec(Qt.DropAction.MoveAction)

    def _drag_pixmap(self, fspec, count):
        pixmap = self.thumbnail(fspec).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
        if count > 1:
            # Show number of files being dragged in a badge.
            from Qt.QtGui import QPainter, QColor
            p = QPainter(pixmap)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = QRect(pixmap.width()-26, 2, 24, 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(200, 30, 30))
            p.drawRoundedRect(r, 9, 9)
            p.setPen(QColor('white'))
            f = p.font()
            f.setBold(True)
            f.setPixelSize(11)
            p.setFont(f)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(count))
            p.end()
        return pixmap

    def files_from_mime_data(self, mime_data):
        if not mime_data.hasFormat(_MIME_TYPE):
            return []
        import json
        try:
            keys = json.loads(bytes(mime_data.data(_MIME_TYPE)).decode('utf-8'))
        except Exception:
            return []
        h = self.history
        fspecs = [h.find_file(path, database) for path, database in keys]
        return [fs for fs in fspecs if fs is not None]

    def show_context_menu(self, global_pos, tile = None, section = None):
        h = self.history
        from Qt.QtWidgets import QMenu
        menu = QMenu(self._history_window)
        if tile is not None:
            files = self._action_files(tile)
            n = len(files)
            what = '' if n == 1 else ' %d files' % n
            menu.addAction('Open' + what, lambda: self.open_files(files))
            move_menu = menu.addMenu('Move%s to section' % what)
            fsections = set(h.file_section(fs) for fs in files)
            for name in [None] + h.sections:
                a = move_menu.addAction(_section_title(name))
                a.setCheckable(True)
                a.setChecked(fsections == {name})
                a.triggered.connect(lambda checked, n=name: self.move_files(files, n))
            move_menu.addSeparator()
            move_menu.addAction('New section...', lambda: self.new_section(files))
            menu.addAction('Remove%s from history' % what, lambda: self.remove_files(files))
            menu.addSeparator()
        if section is not None and h.sections:
            name = section.section_name
            a = menu.addAction('Add new files to this section')
            a.setCheckable(True)
            a.setChecked(h.current_section == name)
            a.triggered.connect(lambda checked, n=name: self._set_current_section(n if checked else None))
            if name is not None:
                menu.addAction('Rename section...', lambda: self.rename_section(name))
                i = h.sections.index(name)
                up = menu.addAction('Move section up', lambda: h.move_section(name, i-1))
                up.setEnabled(i > 0)
                down = menu.addAction('Move section down', lambda: h.move_section(name, i+1))
                down.setEnabled(i < len(h.sections)-1)
                menu.addAction('Remove section', lambda: h.remove_section(name))
            menu.addSeparator()
        menu.addAction('New section...', lambda: self.new_section())
        menu.addSeparator()
        a = menu.addAction('Show full file names')
        a.setCheckable(True)
        a.setChecked(self.session.ui.settings.file_history_show_full_filenames)
        a.triggered.connect(self.show_full_file_names)
        menu.addAction('Remove deleted files', h.remove_missing_files)
        menu.addAction('Clear file history', h.clear_file_history)
        menu.exec(global_pos)

    def new_section(self, fspecs = None):
        from Qt.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self._history_window, 'New Section', 'Section name:')
        if not ok:
            return
        h = self.history
        try:
            name = h.add_section(name)
        except ValueError as e:
            self.session.logger.error(str(e))
            return
        if fspecs:
            self.move_files(fspecs, name)

    def rename_section(self, name):
        from Qt.QtWidgets import QInputDialog, QLineEdit
        new_name, ok = QInputDialog.getText(self._history_window, 'Rename Section', 'Section name:',
                                            QLineEdit.EchoMode.Normal, name)
        if not ok:
            return
        try:
            self.history.rename_section(name, new_name)
        except ValueError as e:
            self.session.logger.error(str(e))

    def _set_current_section(self, name):
        self.history.current_section = name

def limit_string(s, n):
    if len(s) > n:
        return s[:n//2] + '...' + s[-(n//2):]
    return s

def _key(fspec):
    return (fspec.path, fspec.database)

def _section_title(name):
    return 'Recent' if name is None else name

class HistoryWindow(QScrollArea):
    '''Scrolled area showing a heading and thumbnails for each section.'''
    def __init__(self, file_history, size_hint = None):
        super().__init__()
        self._file_history = file_history
        self._size_hint = size_hint
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tiles = []		# All thumbnails in display order

    def sizeHint(self):
        if self._size_hint is not None:
            return QSize(*self._size_hint)
        return super().sizeHint()

    def fill(self, history, used_pixmaps):
        fh = self._file_history
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6,6,6,6)
        layout.setSpacing(10)
        sections = history.sections
        # Only show section headings once the user has made a section.
        show_headings = len(sections) > 0
        current = history.current_section
        tiles = []
        for name in [None] + sections:
            files = history.section_files(name)
            s = SectionWidget(fh, name, files, show_headings, name == current, used_pixmaps)
            layout.addWidget(s)
            tiles.extend(s.tile_area.tiles)
        self.tiles = tiles
        layout.addStretch(1)

        scroll = self.verticalScrollBar().value()
        self._replace_content(content)
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(scroll))

    def clear(self):
        self.tiles = []
        self._replace_content(None)

    def mousePressEvent(self, event):
        # Click on background clears thumbnail selection.
        if event.button() == Qt.MouseButton.LeftButton:
            self._file_history.clear_selection()
        super().mousePressEvent(event)

    def _replace_content(self, content):
        # Delete the old widgets later since this may be called from one of their event handlers.
        old = self.takeWidget()
        if content is not None:
            self.setWidget(content)
            content.setAutoFillBackground(False)	# Show viewport background color
        if old is not None:
            old.hide()
            old.deleteLater()

    def contextMenuEvent(self, event):
        event.accept()
        tile = section = None
        w = QApplication.widgetAt(event.globalPos())
        while w is not None and w is not self:
            if tile is None and isinstance(w, Tile):
                tile = w
            if isinstance(w, SectionWidget):
                section = w
                break
            w = w.parentWidget()
        self._file_history.show_context_menu(event.globalPos(), tile, section)

class SectionWidget(QWidget):
    '''Optional heading and thumbnails for one section.  Thumbnails can be dropped on it.'''
    def __init__(self, file_history, section_name, files, show_heading, is_current, used_pixmaps):
        super().__init__()
        self._file_history = file_history
        self.section_name = section_name	# None for files not in a section
        self._drag_highlight = False
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)
        layout.setSpacing(4)
        if show_heading:
            from html import escape
            text = '<b>%s</b>' % escape(_section_title(section_name))
            if is_current:
                text += '&nbsp;&nbsp;<small>(new files added here)</small>'
            heading = QLabel(text)
            heading.setTextFormat(Qt.TextFormat.RichText)
            f = heading.font()
            f.setPointSizeF(f.pointSizeF() * 1.4)
            heading.setFont(f)
            layout.addWidget(heading)
        placeholder = 'Drag thumbnails here' if show_heading else None
        self.tile_area = TileArea(file_history, files, placeholder, used_pixmaps)
        layout.addWidget(self.tile_area)

    def paintEvent(self, event):
        if self._drag_highlight:
            from Qt.QtGui import QPainter, QPalette
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            c = self.palette().color(QPalette.ColorRole.Highlight)
            c.setAlpha(50)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(self.rect(), 6, 6)
            p.end()

    def _set_drag_highlight(self, highlight):
        if highlight != self._drag_highlight:
            self._drag_highlight = highlight
            self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_MIME_TYPE):
            event.acceptProposedAction()
            self._set_drag_highlight(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if not event.mimeData().hasFormat(_MIME_TYPE):
            event.ignore()
            return
        event.acceptProposedAction()
        if self.section_name is not None:
            # Files not in a section are ordered by time, so only show insert position in sections.
            ta = self.tile_area
            ta.show_drop_indicator(ta.mapFrom(self, event.position().toPoint()))

    def dragLeaveEvent(self, event):
        self._set_drag_highlight(False)
        self.tile_area.show_drop_indicator(None)

    def dropEvent(self, event):
        self._set_drag_highlight(False)
        ta = self.tile_area
        ta.show_drop_indicator(None)
        fh = self._file_history
        fspecs = fh.files_from_mime_data(event.mimeData())
        if not fspecs:
            event.ignore()
            return
        event.acceptProposedAction()
        before = None
        if self.section_name is not None:
            tile = ta.insertion_tile(ta.mapFrom(self, event.position().toPoint()))
            if tile is not None:
                before = tile.file_spec
        # Defer since moving the file rebuilds the widgets including this one.
        name = self.section_name
        QTimer.singleShot(0, lambda: fh.move_files(fspecs, name, before))

class TileArea(QWidget):
    '''Thumbnails laid out in rows that wrap to the widget width.'''
    def __init__(self, file_history, files, placeholder, used_pixmaps):
        super().__init__()
        self._placeholder = None if files else placeholder
        self._indicator = None	# (x, top, bottom) of drop position line
        self._layout = layout = FlowLayout(self, spacing = 8)
        layout.setContentsMargins(4,0,4,0)
        self.tiles = [Tile(file_history, fs, used_pixmaps) for fs in files]
        for t in self.tiles:
            layout.addWidget(t)
        if not files and placeholder:
            self.setMinimumHeight(40)

    def insertion_tile(self, pos):
        '''Tile a drop at this position would be placed before, or None for the end.'''
        for t in self.tiles:
            g = t.geometry()
            if pos.y() < g.top():
                return t
            if pos.y() <= g.bottom() and pos.x() < g.center().x():
                return t
        return None

    def show_drop_indicator(self, pos):
        if pos is None:
            self._indicator = None
        else:
            gap = self._layout.spacing() // 2
            before = self.insertion_tile(pos)
            if before is not None:
                g = before.geometry()
                x = g.left() - gap
            elif self.tiles:
                g = self.tiles[-1].geometry()
                x = g.right() + gap
            else:
                g = self.rect()
                x = 4
            self._indicator = (max(x, 1), g.top(), g.bottom())
        self.update()

    def paintEvent(self, event):
        if self._placeholder is None and self._indicator is None:
            return
        from Qt.QtGui import QPainter, QPalette, QPen
        p = QPainter(self)
        pal = self.palette()
        if self._placeholder:
            p.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
            p.drawText(self.rect().adjusted(8,0,0,0),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._placeholder)
        if self._indicator:
            x, top, bottom = self._indicator
            p.setPen(QPen(pal.color(QPalette.ColorRole.Highlight), 3))
            p.drawLine(x, top, x, bottom)
        p.end()

class Tile(QWidget):
    '''Thumbnail image and file name.  Click to open, drag to move to another section,
    ctrl-click or shift-click to select several.'''
    def __init__(self, file_history, fspec, used_pixmaps):
        super().__init__()
        self._file_history = file_history
        self.file_spec = fspec
        self._press_pos = None
        self._hover = False
        self._selected = file_history.is_selected(fspec)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        descrip = fspec.path if fspec.database is None else '%s %s' % (fspec.database.upper(), fspec.path)
        self.setToolTip(descrip)

        self._pixmap = file_history.thumbnail(fspec, used_pixmaps)
        # QPixmap sizes are physical pixels; layout and painting use logical pixels.
        ratio = self._pixmap.devicePixelRatioF()
        self._image_size = QSize(round(self._pixmap.width() / ratio),
                                 round(self._pixmap.height() / ratio))
        self._name = file_history.display_name(fspec)
        self.setAccessibleName(self._name)
        from Qt.QtGui import QPalette
        self._text_color = QApplication.palette('QLabel').color(QPalette.ColorRole.Link)

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(max(self._image_size.width(), fm.horizontalAdvance(self._name)) + 6,
                     self._image_size.height() + fm.height() + 8)

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_selected(self, selected):
        if selected != self._selected:
            self._selected = selected
            self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        from Qt.QtGui import QPainter, QPalette
        p = QPainter(self)
        if self._hover or self._selected:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            c = self.palette().color(QPalette.ColorRole.Highlight)
            c.setAlpha(160 if self._selected else 70)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(self.rect(), 4, 4)
        p.drawPixmap((self.width() - self._image_size.width()) // 2, 3, self._pixmap)
        p.setPen(self._text_color)
        text_rect = QRect(3, self._image_size.height() + 5, self.width() - 6, self.fontMetrics().height())
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._name)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        moved = (event.position().toPoint() - self._press_pos).manhattanLength()
        if moved < QApplication.startDragDistance():
            return
        self._press_pos = None
        self._file_history.start_drag(self)

    def mouseReleaseEvent(self, event):
        pressed = self._press_pos is not None
        self._press_pos = None
        if (pressed and event.button() == Qt.MouseButton.LeftButton
                and self.rect().contains(event.position().toPoint())):
            self._file_history.tile_clicked(self, event.modifiers())

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
