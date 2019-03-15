# vim: set expandtab ts=4 sw=4:

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

from PyQt5.QtWidgets import QListWidget
from PyQt5.QtCore import Qt
from chimerax.atomic import Structure, AtomicStructure

class StructureItems:
    column_title = "Structure"
    class_filter = Structure

    def __init__(self, session, *, list_func=None, key_func=None, filter_func=None, **kw):
        self.list_func = session.models.list if list_func is None else list_func

        filt = lambda s, cf=self.class_filter: isinstance(s, cf)
        self.filter_func = filt if filter_func is None else lambda s, f=filt, ff=filter_func: f(s) and ff(s)

        self.key_func = lambda s: s.id if key_func is None else key_func

        self._remaining_kw = kw

    def _item_names(self):
        self.item_map = {}
        self.value_map = {}
        structures = [s for s in self.list_func() if self.filter_func(s)]
        structure.sort(key=self.key_func)
        # for some subclasses, the item names may not be str(value)
        items = []
        for s in structures:
            self.item_map[s] = s
            self.value_map[s] = s
            items.append(s)
        return items

class AtomicStructureItems(StructureItems):
    class_filter = AtomicStructure

class ListWidgetBase(QListWidget):
    pass
class StructureListWidget(ListWidgetBase, StructureItems):
    pass
class AtomicStructureListWidget(ListWidgetBase, AtomicStructureItems):
    pass
