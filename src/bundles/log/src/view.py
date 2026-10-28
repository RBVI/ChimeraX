# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
Log view built on QTextBrowser, so the Log does not need QtWebEngine.

Messages are rendered incrementally: each message's document position is
remembered so that new messages are appended and a changed message (e.g.,
"[Repeated N time(s)]" compaction) only re-renders from that message on.
"""

import base64
from bisect import bisect_right

from Qt.QtCore import Qt, QBuffer, QByteArray, QEvent, QRectF, QSize
from Qt.QtGui import (
    QColor, QFont, QFontDatabase, QFontInfo, QImage, QPainter, QPalette, QPen,
    QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument, QTextFrameFormat,
)
from Qt.QtWidgets import QTextBrowser

from .parse import (
    tokenize, Command, Disclosure, select_command_variant, replace_inline_blocks, drop_redundant_breaks,
    css_length, css_border, starts_block, starts_table, strip_line_end,
)

TOGGLE_SCHEME = "cxlog-toggle"


class LogView(QTextBrowser):

    def __init__(self, parent, *, link_cb, command_color_cb, context_menu_cb=None, size_hint=None):
        """
        link_cb : called with a QUrl when a link other than a disclosure toggle is clicked
        command_color_cb : returns the background color name for command echoes
        context_menu_cb : called with the QContextMenuEvent instead of showing Qt's menu
        """
        super().__init__(parent)
        self._link_cb = link_cb
        self._command_color_cb = command_color_cb
        self._context_menu_cb = context_menu_cb
        self._size_hint = size_hint
        self.exec_links = False

        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setUndoRedoEnabled(False)
        doc = self.document()
        doc.setUndoRedoEnabled(False)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        doc.setDefaultStyleSheet(
            "a.no_underline { text-decoration: none; }\n"
            "pre, tt, code, kbd, samp { font-family: '%s'; }\n" % fixed
        )
        self.anchorClicked.connect(self._anchor_clicked)

        sb = self.verticalScrollBar()
        sb.rangeChanged.connect(self._range_changed)
        sb.valueChanged.connect(self._value_changed)
        self._pin_bottom = True

        self._source = ""
        self._starts = []
        # per rendered message: (document position, whether a new block was needed)
        self._rendered = []
        self._new_block = False
        # offsets of <details> whose open/closed state the user changed
        self._toggled = set()
        self._swatches = {}

    def set_content(self, source, starts, dirty_from, *, replaced=False, scroll_to_end=True):
        """Show log HTML 'source', whose messages begin at offsets 'starts'.
        Nothing before offset 'dirty_from' changed since the last call.
        'replaced' indicates the source is unrelated to the previous one."""
        if replaced:
            self._toggled.clear()
        self._source = source
        self._starts = starts
        sb = self.verticalScrollBar()
        value = sb.value()
        if dirty_from is not None:
            self._render(dirty_from)
        if scroll_to_end:
            self._pin_bottom = True
            sb.setValue(sb.maximum())
        else:
            self._pin_bottom = False
            sb.setValue(value)

    def rerender(self, *, scroll_to_end=False):
        """Re-render everything, e.g., after a settings or color change."""
        self.set_content(self._source, self._starts, 0, scroll_to_end=scroll_to_end)

    def selected_text(self):
        return self.textCursor().selection().toPlainText().replace('\u2028', '\n')

    #
    # Rendering
    #
    def _render(self, dirty_from):
        i = max(bisect_right(self._starts, dirty_from) - 1, 0)
        i = min(i, len(self._rendered))
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        if i < len(self._rendered):
            doc_pos, self._new_block = self._rendered[i]
            del self._rendered[i:]
            cursor.setPosition(doc_pos)
            cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
        else:
            cursor.movePosition(QTextCursor.MoveOperation.End)
        if i == 0:
            self._new_block = False
            cursor.setBlockFormat(QTextBlockFormat())
            cursor.setBlockCharFormat(QTextCharFormat())
        source, starts = self._source, self._starts
        for j in range(i, len(starts)):
            end = starts[j + 1] if j + 1 < len(starts) else len(source)
            self._rendered.append((cursor.position(), self._new_block))
            self._render_html(cursor, source[starts[j]:end], starts[j])
        cursor.endEditBlock()

    def _render_html(self, cursor, html, base):
        for piece in tokenize(html, base):
            if isinstance(piece, Command):
                self._insert_command(cursor, piece)
            elif isinstance(piece, Disclosure):
                self._insert_disclosure(cursor, piece)
            else:
                self._insert_html(cursor, piece.html)

    def _start_block(self, cursor, block_format=None, char_format=None):
        if block_format is None:
            block_format = QTextBlockFormat()
        if char_format is None:
            char_format = QTextCharFormat()
        if cursor.block().length() <= 1:
            # reuse empty block, e.g., the one following a table
            cursor.setBlockFormat(block_format)
            cursor.setBlockCharFormat(char_format)
        else:
            cursor.insertBlock(block_format, char_format)
        cursor.setCharFormat(char_format)

    def _prepare(self, html):
        html = select_command_variant(html, self.exec_links)
        html = drop_redundant_breaks(html)
        return replace_inline_blocks(html, self._swatch_html)

    def _insert_html(self, cursor, html):
        html, ends_line = strip_line_end(self._prepare(html))
        if not html.strip():
            if ends_line:
                if self._new_block:
                    # blank line
                    self._start_block(cursor)
                    cursor.insertText(" ")
                self._new_block = True
            return
        block_start = None
        if (self._new_block or starts_block(html)) and not starts_table(html):
            self._start_block(cursor)
            block_start = cursor.position()
        cursor.insertHtml(html)
        if block_start is not None and starts_block(html):
            self._restore_first_block_format(html, block_start)
        self._new_block = ends_line

    def _restore_first_block_format(self, html, position):
        # Inserting HTML into an existing block drops the formatting of the
        # inserted HTML's first block, such as a <p>'s margins, so reapply it.
        doc = QTextDocument()
        doc.setDefaultStyleSheet(self.document().defaultStyleSheet())
        doc.setHtml(html)
        first = doc.begin().blockFormat()
        fmt = QTextBlockFormat()
        for prop in (QTextBlockFormat.Property.BlockTopMargin, QTextBlockFormat.Property.BlockBottomMargin,
                     QTextBlockFormat.Property.BlockLeftMargin, QTextBlockFormat.Property.BlockRightMargin,
                     QTextBlockFormat.Property.BlockAlignment, QTextBlockFormat.Property.BackgroundBrush):
            if first.hasProperty(prop):
                fmt.setProperty(prop, first.property(prop))
        if not fmt.isEmpty():
            QTextCursor(self.document().findBlock(position)).mergeBlockFormat(fmt)

    def _swatch_html(self, props):
        """Draw an inline-block div, e.g., a color swatch, as an image."""
        em = QFontInfo(self.font()).pixelSize()
        width = css_length(props.get("width"), em) or em
        height = css_length(props.get("height"), em) or em
        border_width, border_color = css_border(props.get("border")) if "border" in props else (0, None)
        key = (width, height, props.get("background-color"), border_width, border_color)
        if key not in self._swatches:
            self._swatches[key] = self._swatch_image(*key)
        return self._swatches[key]

    def _swatch_image(self, width, height, background, border_width, border_color):
        scale = 2  # stay sharp on high resolution displays
        w, h = width + 2 * border_width, height + 2 * border_width
        image = QImage(max(round(w * scale), 1), max(round(h * scale), 1), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        p = QPainter(image)
        p.scale(scale, scale)
        bg = QColor(background) if background else QColor()
        if bg.isValid():
            p.fillRect(QRectF(border_width, border_width, width, height), bg)
        if border_width:
            color = QColor(border_color) if border_color else self.palette().color(QPalette.ColorRole.Text)
            pen = QPen(color, border_width)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            p.setPen(pen)
            half = border_width / 2
            p.drawRect(QRectF(half, half, w - border_width, h - border_width))
        p.end()
        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QBuffer.OpenModeFlag.WriteOnly)
        image.save(buf, "PNG")
        return '<img src="data:image/png;base64,%s" width=%d height=%d>' % (
            base64.b64encode(bytes(data)).decode("ascii"), round(w), round(h))

    def _insert_command(self, cursor, piece):
        block_format = QTextBlockFormat()
        block_format.setBackground(QColor(self._command_color_cb()))
        block_format.setTopMargin(6)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        self._start_block(cursor, block_format, char_format)
        # insertHtml() uses the fragment's character formats, not the cursor's
        cursor.insertHtml("<b>%s</b>" % self._prepare(piece.html))
        self._new_block = True

    def _insert_disclosure(self, cursor, piece):
        expanded = piece.open != (piece.offset in self._toggled)
        # A frame paints its background across its blocks' margins too,
        # which per-block backgrounds do not.
        frame_format = QTextFrameFormat()
        color = self._disclosure_color(piece.background)
        if color is not None:
            frame_format.setBackground(color)
        if cursor.block().length() <= 1:
            # Qt needs a block between a table or frame and this frame
            self._collapse_block(cursor)
        frame = cursor.insertFrame(frame_format)
        self._new_block = False
        summary = piece.summary if piece.summary is not None else "Details"
        arrow = "\u25BC" if expanded else "\u25B6"
        # look like the <summary> of a <details>, not a link
        text_color = self.palette().color(QPalette.ColorRole.Text).name()
        cursor.insertHtml('<a class="no_underline" style="color:%s" href="%s:%d">%s %s</a>'
                          % (text_color, TOGGLE_SCHEME, piece.offset, arrow, self._prepare(summary)))
        self._new_block = True
        if expanded:
            self._render_html(cursor, piece.content, piece.content_offset)
            self._collapse_trailing_margin(frame)
        # continue after the frame
        cursor.setPosition(frame.lastPosition() + 1)
        self._new_block = True

    def _collapse_block(self, cursor):
        """Make the (empty) block at the cursor as short as possible."""
        block_format = QTextBlockFormat()
        block_format.setLineHeight(0, QTextBlockFormat.LineHeightTypes.FixedHeight.value)
        cursor.setBlockFormat(block_format)
        char_format = QTextCharFormat()
        char_format.setFontPointSize(1)
        cursor.setBlockCharFormat(char_format)

    def _collapse_trailing_margin(self, frame):
        # Qt ignores the bottom margin of a frame's last block, and when the
        # frame ends with a table or nested frame, Qt adds an empty block after
        # it.  In a browser, the last child's bottom margin collapses through
        # <details> to below its background, so emulate that.
        last = self.document().findBlock(frame.lastPosition())
        if last.length() <= 1:
            self._collapse_block(QTextCursor(last))
            children = frame.childFrames()
            if children and children[-1].lastPosition() + 1 == last.position():
                # move the margin out of this frame rather than adding to it
                child_format = children[-1].frameFormat()
                margin = child_format.bottomMargin()
                child_format.setBottomMargin(0)
                children[-1].setFrameFormat(child_format)
            else:
                margin = 0
        else:
            margin = last.blockFormat().bottomMargin()
        if margin:
            frame_format = frame.frameFormat()
            frame_format.setBottomMargin(margin)
            frame.setFrameFormat(frame_format)

    def _disclosure_color(self, background):
        if background is None:
            return None
        color = QColor(background)
        if not color.isValid():
            # CSS system colors, such as ButtonFace, that Qt doesn't know
            color = self.palette().color(QPalette.ColorRole.Button)
        return color

    #
    # Event handling
    #
    def _anchor_clicked(self, qurl):
        if not qurl.scheme() and not qurl.path() and qurl.hasFragment():
            self.scrollToAnchor(qurl.fragment())
            return
        if qurl.scheme() == TOGGLE_SCHEME:
            try:
                offset = int(qurl.path())
            except ValueError:
                return
            self._toggled ^= {offset}
            self.set_content(self._source, self._starts, offset, scroll_to_end=False)
            return
        self._link_cb(qurl)

    def _range_changed(self, minimum, maximum):
        if self._pin_bottom:
            self.verticalScrollBar().setValue(maximum)

    def _value_changed(self, value):
        self._pin_bottom = value >= self.verticalScrollBar().maximum()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and self._rendered:
            self._swatches.clear()
            # command and disclosure colors depend on light/dark mode
            self.rerender()

    def contextMenuEvent(self, event):
        if self._context_menu_cb:
            self._context_menu_cb(event)
        else:
            super().contextMenuEvent(event)

    def sizeHint(self):
        if self._size_hint:
            return QSize(*self._size_hint)
        return super().sizeHint()

