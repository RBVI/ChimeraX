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
from Qt.QtWidgets import QMenu, QWidget
from chimerax.ui.widgets.item_table import ItemTable

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
    get_selection = Signal(list)

    def __init__(self, control_widget: Union[QMenu, QWidget], default_cols, settings: 'BlastProteinResultsSettings', parent = Optional[QWidget]):
        super().__init__(
            column_control_info=(
                control_widget
                , settings
                , default_cols
                , False          # fallback default for column display
                , None           # display callback
                , None           # number of checkbox columns
                , True          # Whether to show global buttons
            )
            , parent=parent)
        self.doubleClicked.connect(self.doubleclicked)

    def doubleclicked(self, _) -> list:
        self.get_selection.emit(self.selected)

    def resizeColumns(self, max_size:int = 0):
        for col in self._columns:
            if self.columnWidth(self._columns.index(col)) > max_size:
                self.setColumnWidth(self._columns.index(col), max_size)
