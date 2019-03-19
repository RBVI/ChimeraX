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
            self.item_map[str(s)] = s
            self.value_map[s] = str(s)
            items.append(s)
        return items

class AtomicStructureItems(StructureItems):
    class_filter = AtomicStructure

class StructureListBase:
    def __init__(self, session):
        if not hasattr(self, '_trigger_info'):
            from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
            from chimerax.atomic import get_triggers
            self._trigger_info = [
                (session.triggers, ADD_MODELS, self._items_change),
                (session.triggers, REMOVE_MODELS, self._items_change),
                (get_triggers(session), "changes", self._structure_rename)
            ]
            self._handlers = []

    def destroy(self):
        self._delete_handlers()
        try:
            self.item_map.clear()
            self.value_map.clear()
        except AttributeError:
            # may not exist if widget was never shown (hidden in a tab for instance)
            pass

    def hideEvent(self, event):
        self._delete_handlers()

    def refresh(self):
        # can be needed if 'filter_func' was specified
        self._items_change()

    def showEvent(self, event):
        if self.handlers:
            return
        for triggers, trig_name, cb in self._trigger_info:
            self._handlers.append(triggers.add_handler(trig_name, cb))
        self._items_change()

    def _delete_handlers(self):
        while self._handlers:
            self._handlers.pop().remove()

    def _sleep_check(self, do_callback=True):
        if not self.handlers and not hasattr(self, '_recursion'):
            # the list maintenance code is "asleep" (i.e. we're hidden); force an update
            self._recursion = True
            self._items_change(do_callback=do_callback)
            delattr(self, '_recursion')

    def _structure_rename(self, trig_name, changes):
        if 'name changed' in changes.structure_reasons():
            self._items_change()

class ListWidgetBase(StructureListBase, QListWidget):
    """Maintain a list of molecular items

       Keep list up to date as structures are opened and closed while keeping the selected item(s) the same

       'autoselect' keyword controls what happens when nothing would be selected.  If 'single', then if
       there is exactly one item it will be selected and otherwise the selection remains empty.  If 'all'
       then all items will become selected.  If None, the selection remains empty.  Default: 'all' for
       structures and None for chains.

       'selection_mode' controls how items are selected in the list widget.  Possible values are:
       'single', 'extended', and 'multi' (default 'extended') which correspond to QAbstractItemView::
       Single/Extended/MultiSelection respectively as per:
       https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

       'balloon_help' is the balloon help to provide with the list (if any); if none provided then
       some generic help for  the 'extended' selection mode is provided if applicable

       _item_names() needs to be implemented in subclasses (possibly by multiple inheritance)
    """

    extended_balloon_help = "Click to choose item\n" \
        "Drag to choose range\n" \
        "Control-click to toggle item\n" \
        "Shift-click to choose range\n  (starting from previous selected item)"

    def __init__(self, *args, autoselect='default', selection_mode='extended', balloon_help=None, **kw):
        self.autoselect = self.autoselect_default if autoselect == 'default' else autoselect
        QListWidget.__init__(self, *args, **kw)
        self.setSelectionMode({'single': self.SingleSelection, 'extended': self.ExtendedSelection,
            'multi': self.MultiSelection}[selection_mode])
        if balloon_help or selection_mode == 'extended':
            self.setToolTip(balloon_help if balloon_help else self.extended_balloon_help)
        StructureListBase.__init__(self)

    def destroy(self):
        StructureListBase.destroy(self)
        QListWidget.destroy(self)

    def _get_value(self):
        self._sleep_check()
        values = [self.item_map[si] for si in self.selectedItems()]
        if self.selectionMode() == 'single':
            return values[0] if values else None
        return values


class StructureListWidget(ListWidgetBase, StructureItems):
    pass
class AtomicStructureListWidget(ListWidgetBase, AtomicStructureItems):
    pass
