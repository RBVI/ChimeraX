# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from typing import Optional, Union

from Qt.QtCore import Signal
from Qt.QtWidgets import QMenu, QWidget, QApplication, QStyle, QStyleOptionViewItem, QAbstractItemView, QStyledItemDelegate
from Qt.QtGui import QTextDocument, QAbstractTextDocumentLayout

from Qt.QtGui import QTextDocument
from Qt.QtCore import QSize

from chimerax.core.settings import Settings
from chimerax.ui.widgets import ItemTable

class BlastResultsRow:
    """Takes in and stores a dictionary. This class only exists to coerce Python into hashing a dictionary."""

    # Save on memory by suppressing the internal class dictionary.
    # Only allocate these slots.
    __slots__ = ['_internal_dict']

    def __init__(self, row: dict):
        self._internal_dict = row

    def __getitem__(self, key):
        return self._internal_dict.get(key, "")


class BlastResultsTable(ItemTable):
    def __init__(self, control_widget: Union[QMenu, QWidget], default_cols, settings: 'BlastProteinResultsSettings', parent = Optional[QWidget]):
        super().__init__(
            column_control_info=(
                control_widget
                , settings
                , default_cols
                , False        # fallback default for column display
                , None         # display callback
                , None         # number of checkbox columns
                , True         # Whether to show global buttons
            )
            , parent=parent)

    def resizeColumns(self, max_size: int = 0):
        for col in self._columns:
            if self.columnWidth(self._columns.index(col)) > max_size:
                self.setColumnWidth(self._columns.index(col), max_size)

    def launch(self, *, select_mode=QAbstractItemView.SelectionMode.ExtendedSelection, session_info=None, suppress_resize=False):
        super().launch(select_mode=select_mode, session_info=session_info, suppress_resize=suppress_resize)
        for index, col in enumerate(self._columns):
            if col.title == "Ligand Formulas":
                self.setItemDelegateForColumn(index, HTMLColumnDelegate())
                break

class BlastProteinResultsSettings(Settings):
    EXPLICIT_SAVE = {BlastResultsTable.DEFAULT_SETTINGS_ATTR: {}}


# Modified from https://stackoverflow.com/a/44365155/12208118
class HTMLColumnDelegate(QStyledItemDelegate):

    def anchorAt(self, html, point):
        doc = QTextDocument()
        doc.setHtml(html)
        textLayout = doc.documentLayout()
        return textLayout.anchorAt(point)

    def paint(self, painter, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        if options.widget:
            style = options.widget.style()
        else:
            style = QApplication.style()

        doc = QTextDocument()
        doc.setHtml(options.text)
        options.text = ''

        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, options, painter)
        ctx = QAbstractTextDocumentLayout.PaintContext()

        textRect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, options)

        painter.save()

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        painter.translate(0, 0.5 * (options.rect.height() - doc.size().height()))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        doc = QTextDocument()
        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())

        return QSize(doc.idealWidth(), doc.size().height())
