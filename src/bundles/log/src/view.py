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

from bisect import bisect_right

from Qt.QtCore import QSize
from Qt.QtGui import QFontDatabase, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument
from Qt.QtWidgets import QTextBrowser

from .parse import (
    select_command_variant, drop_redundant_breaks, starts_block, starts_table, strip_line_end,
)


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

    def set_content(self, source, starts, dirty_from, *, scroll_to_end=True):
        """Show log HTML 'source', whose messages begin at offsets 'starts'.
        Nothing before offset 'dirty_from' changed since the last call."""
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
        self._insert_html(cursor, html)

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
        return drop_redundant_breaks(html)

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

    #
    # Event handling
    #
    def _anchor_clicked(self, qurl):
        if not qurl.scheme() and not qurl.path() and qurl.hasFragment():
            self.scrollToAnchor(qurl.fragment())
            return
        self._link_cb(qurl)

    def _range_changed(self, minimum, maximum):
        if self._pin_bottom:
            self.verticalScrollBar().setValue(maximum)

    def _value_changed(self, value):
        self._pin_bottom = value >= self.verticalScrollBar().maximum()

    def contextMenuEvent(self, event):
        if self._context_menu_cb:
            self._context_menu_cb(event)
        else:
            super().contextMenuEvent(event)

    def sizeHint(self):
        if self._size_hint:
            return QSize(*self._size_hint)
        return super().sizeHint()

