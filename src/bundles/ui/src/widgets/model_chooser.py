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
from PyQt5.QtCore import Qt, pyqtSignal
from chimerax.core.models import Model

class ModelItems:
    def __init__(self, list_func=None, key_func=None, filter_func=None, item_text_func=str,
            class_filter=Model, column_title="Model", **kw):
        self.list_func = self.session.models.list if list_func is None else list_func
        filt = lambda s, cf=class_filter: isinstance(s, cf)
        self.filter_func = filt if filter_func is None else lambda s, f=filt, ff=filter_func: f(s) and ff(s)
        self.key_func = lambda s, kf=key_func: s.id if kf is None else kf(s)
        self.item_text_func = item_text_func
        self.column_title = column_title

        super().__init__(**kw)

    def _item_names(self):
        self.item_map = {}
        self.value_map = {}
        values = [v for v in self.list_func() if self.filter_func(v)]
        values.sort(key=self.key_func)
        items = []
        for v in values:
            text = self.item_text_func(v)
            self.item_map[text] = v
            self.value_map[v] = text
            items.append(text)
        return items

class ModelListBase:
    def __init__(self, **kw):
        super().__init__(**kw)
        if not hasattr(self, '_trigger_info'):
            from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_ID_CHANGED, MODEL_NAME_CHANGED
            from chimerax.atomic import get_triggers
            self._trigger_info = [
                (self.session.triggers, ADD_MODELS, self._items_change),
                (self.session.triggers, REMOVE_MODELS, self._items_change),
                (self.session.triggers, MODEL_ID_CHANGED, self._items_change),
                (self.session.triggers, MODEL_NAME_CHANGED, self._items_change),
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
        if self._handlers:
            return
        for triggers, trig_name, cb in self._trigger_info:
            self._handlers.append(triggers.add_handler(trig_name, cb))
        self._items_change()

    def _delete_handlers(self):
        while self._handlers:
            self._handlers.pop().remove()

    def _sleep_check(self):
        if not self._handlers and not hasattr(self, '_recursion'):
            # the list maintenance code is "asleep" (i.e. we're hidden); force an update
            self._recursion = True
            self._items_change()
            delattr(self, '_recursion')

class ModelListWidgetBase(ModelListBase, QListWidget):
    """Maintain a list of models

       Keep list up to date as models are opened and closed while keeping the selected item(s) the same

       'autoselect' keyword controls what happens when nothing would be selected.  If 'single', then if
       there is exactly one item it will be selected and otherwise the selection remains empty.  If 'all'
       then all items will become selected.  If None, the selection remains empty.  Default: 'all'.

       'selection_mode' controls how items are selected in the list widget.  Possible values are:
       'single', 'extended', and 'multi' (default 'extended') which correspond to QAbstractItemView::
       Single/Extended/MultiSelection respectively as per:
       https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

       'balloon_help' is the balloon help to provide with the list (if any); if none provided then
       some generic help for  the 'extended' selection mode is provided if applicable

       _item_names() needs to be implemented in subclasses (possibly by multiple inheritance)
    """

    value_changed = pyqtSignal()

    extended_balloon_help = "Click to choose item\n" \
        "Drag to choose range\n" \
        "Control-click to toggle item\n" \
        "Shift-click to choose range\n  (starting from previous selected item)"

    def __init__(self, autoselect='default', selection_mode='extended', balloon_help=None, **kw):
        super().__init__(**kw)
        self.autoselect = self.autoselect_default if autoselect == 'default' else autoselect
        self.setSelectionMode({'single': self.SingleSelection, 'extended': self.ExtendedSelection,
            'multi': self.MultiSelection}[selection_mode])
        if balloon_help or selection_mode == 'extended':
            self.setToolTip(balloon_help if balloon_help else self.extended_balloon_help)

    def destroy(self):
        ModelListBase.destroy(self)
        QListWidget.destroy(self)

    @property
    def value(self):
        self._sleep_check()
        values = [self.item_map[si.text()] for si in self.selectedItems()]
        if self.selectionMode() == 'single':
            return values[0] if values else None
        return values

    @value.setter
    def value(self, val):
        self._sleep_check()
        if self.value == val:
            return
        self.clearSelection()
        self._select_value(val)
        self.value_changed.emit()

    def _items_change(self, *args):
        del_recursion = False
        if not hasattr(self, '_recursion'):
            self._recursion = True
            del_recursion = True
        prev_value = self.value
        sel = [si.text() for si in self.selectedItems()]
        item_names = self._item_names()
        filtered_sel = [s for s in sel if s in item_names]
        if self.autoselect and not filtered_sel:
            if (self.autoselect == "single" and len(item_names) == 1) or self.autoselect == "all":
                filtered_sel = item_names
        self.clear()
        self.addItems(item_names)
        if self.selectionMode() == 'single':
            if filtered_sel:
                next_value = self.item_map[filtered_sel[0]]
            else:
                next_value = None
        else:
            next_value = [self.item_map[fs] for fs in filtered_sel]
        if prev_value == next_value:
            self._select_value(next_value)
        else:
            self.value = next_value
        if del_recursion:
            delattr(self, '_recursion')

    def _select_value(self, val):
        if self.selectionMode() == 'single':
            if val is None:
                val_names = set()
            else:
                val_names = set([self.value_map[val]])
        else:
            val_names = set([self.value_map[v] for v in val])
        for i in range(self.count()):
            item = self.item(i)
            if item.text() in val_names:
                item.setSelected(True)

class ModelListWidget(ModelListWidgetBase, ModelItems):
    autoselect_default = "all"

    def __init__(self, session, **kw):
        self.session = session
        super().__init__(**kw)
