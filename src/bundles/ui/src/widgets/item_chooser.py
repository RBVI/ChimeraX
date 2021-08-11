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

from Qt.QtWidgets import QListWidget, QPushButton, QMenu
from Qt.QtCore import Qt, Signal

class ItemsGenerator:
    def __init__(self, list_func=lambda: [], key_func=lambda x: x, filter_func=lambda x: True,
            item_text_func=str, class_filter=None, **kw):
        self.list_func = list_func
        if class_filter:
            filter_func = lambda x, ff=filter_func, cf=class_filter: ff(x) and isinstance(x, cf)
        self.filter_func = filter_func
        self.key_func = key_func
        self.item_text_func = item_text_func

        super().__init__(**kw)

    def destroy(self):
        try:
            self.item_map.clear()
            self.value_map.clear()
        except AttributeError:
            # may not exist if widget was never shown (hidden in a tab for instance)
            pass

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

class ItemsUpdater:
    def __init__(self, trigger_info=None, **kw):
        super().__init__(**kw)
        self._handlers = []
        self._trigger_info = trigger_info

    def destroy(self):
        self._delete_handlers()
        self._destroyed = True

    def hideEvent(self, event):
        self._delete_handlers()

    def refresh(self):
        # can be needed if 'filter_func' was specified
        self._items_change()

    def showEvent(self, event):
        if self._handlers:
            return
        for triggers, trig_name in self._trigger_info:
            self._handlers.append(triggers.add_handler(trig_name, self._items_change))
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

class ItemListWidget(ItemsGenerator, ItemsUpdater, QListWidget):

    value_changed = Signal()

    autoselect_default = "all"
    from chimerax.mouse_modes import mod_key_info
    extended_balloon_help = "Click to choose item\n" \
        "Drag to choose range\n" \
        "%s-click to toggle item\n" \
        "Shift-click to choose range\n  (starting from previous selected item)" \
            % mod_key_info("control")[1].capitalize()

    def __init__(self, autoselect='default', selection_mode='extended', balloon_help=None, **kw):
        super().__init__(**kw)
        self.autoselect = self.autoselect_default if autoselect == 'default' else autoselect
        self.setSelectionMode({'single': self.SingleSelection, 'extended': self.ExtendedSelection,
            'multi': self.MultiSelection}[selection_mode])
        self.itemSelectionChanged.connect(self.value_changed.emit)
        if balloon_help or selection_mode == 'extended':
            self.setToolTip(balloon_help if balloon_help else self.extended_balloon_help)

    def destroy(self):
        ItemsGenerator.destroy(self)
        ItemsUpdater.destroy(self)
        QListWidget.destroy(self)

    @property
    def value(self):
        return self.get_value()

    def get_value(self):
        self._sleep_check()
        values = [self.item_map[si.text()] for si in self.selectedItems()]
        if self.selectionMode() == self.SingleSelection:
            return values[0] if values else None
        return values

    @value.setter
    def value(self, val):
        self.set_value(val)

    def set_value(self, val):
        self._sleep_check()
        if self.get_value() == val:
            return
        preblocked = self.signalsBlocked()
        if not preblocked:
            self.blockSignals(True)
        self.clearSelection()
        self._select_value(val)
        if not preblocked:
            self.blockSignals(False)
        self.itemSelectionChanged.emit()

    def _items_change(self, *args):
        if self.__dict__.get('_destroyed', False):
            return
        del_recursion = False
        if not hasattr(self, '_recursion'):
            self._recursion = True
            del_recursion = True
        sel = [si.text() for si in self.selectedItems()]
        prev_value = self.get_value()
        item_names = self._item_names()
        filtered_sel = [s for s in sel if s in item_names]
        if self.autoselect and not filtered_sel:
            if (self.autoselect == "single" and len(item_names) == 1) or self.autoselect == "all":
                filtered_sel = item_names
            elif self.autoselect == "first" and len(item_names) > 0:
                filtered_sel = item_names[:1]
        preblocked = self.signalsBlocked()
        if not preblocked:
            self.blockSignals(True)
        self.clear()
        self.addItems(item_names)
        if self.selectionMode() == self.SingleSelection:
            if filtered_sel:
                next_value = self.item_map[filtered_sel[0]]
            else:
                next_value = None
        else:
            next_value = [self.item_map[fs] for fs in filtered_sel]
        if prev_value == next_value:
            self._select_value(next_value)
            if not preblocked:
                self.blockSignals(False)
        else:
            if not preblocked:
                self.blockSignals(False)
            self.set_value(next_value)
            # if items were deleted, then the current selection could be empty when the previous
            # one was not, but the test in the value setter will think the value is unchanged
            # and not emit the changed signal, so check for that here
            if len(sel) > 0 and not next_value:
                self.itemSelectionChanged.emit()
        if del_recursion:
            delattr(self, '_recursion')

    def _select_value(self, val):
        if self.selectionMode() == self.SingleSelection:
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

def _process_model_kw(session, list_func=None, key_func=None, class_filter=None, trigger_info=None, **kw):
    from chimerax.core.models import Model
    kw['class_filter'] = Model if class_filter is None else class_filter
    kw['list_func'] = session.models.list if list_func is None else list_func
    kw['key_func'] = lambda s: s.id if key_func is None else key_func
    from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_ID_CHANGED, MODEL_NAME_CHANGED
    kw['trigger_info'] = [
        (session.triggers, ADD_MODELS),
        (session.triggers, REMOVE_MODELS),
        (session.triggers, MODEL_ID_CHANGED),
        (session.triggers, MODEL_NAME_CHANGED),
    ] if trigger_info is None else trigger_info
    return kw

class ModelListWidget(ItemListWidget):
    """Maintain a list of models

       Keep list up to date as models are opened and closed while keeping the selected item(s) the same

       'autoselect' keyword controls what happens when nothing would be selected.  If 'single', then if
       there is exactly one item it will be selected and otherwise the selection remains empty.  If 'all'
       then all items will become selected.  If 'first' the first item will be selected.  If None, the
       selection remains empty.  Default: 'all'.

       'selection_mode' controls how items are selected in the list widget.  Possible values are:
       'single', 'extended', and 'multi' (default 'extended') which correspond to QAbstractItemView::
       Single/Extended/MultiSelection respectively as per:
       https://doc.qt.io/qt-5/qabstractitemview.html#SelectionMode-enum

       'balloon_help' is the balloon help to provide with the list (if any); if none provided then
       some generic help for  the 'extended' selection mode is provided if applicable

       Do not access or set the value of this widget in trigger handlers that also update the widget.
       For generic models, those triggers are in session.triggers and are named ADD_MODELS, REMOVE_MODELS,
       MODEL_ID_CHANGED and MODEL_NAME_CHANGED
    """
    def __init__(self, session, **kw):
        super().__init__(**_process_model_kw(session, **kw))

class MenuButton(QPushButton):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.setMenu(QMenu(self))

class ItemMenuButton(ItemsGenerator, ItemsUpdater, MenuButton):

    value_changed = Signal()

    def __init__(self, autoselect_single_item=True, balloon_help=None, no_value_button_text="No item chosen",
            no_value_menu_text=None, special_items=[], **kw):
        self._autoselect_single = autoselect_single_item
        self._no_value_menu_text = no_value_menu_text
        self._no_value_button_text = no_value_button_text
        self._special_items = special_items
        super().__init__(**kw)
        self.menu().triggered.connect(self._sel_change)
        if balloon_help:
            self.setToolTip(balloon_help if balloon_help else self.extended_balloon_help)

    def destroy(self):
        ItemsGenerator.destroy(self)
        ItemsUpdater.destroy(self)
        MenuButton.destroy(self)

    @property
    def value(self):
        return self.get_value()

    # break out 'value' into subfunction so that we don't have to access the property ourselves
    # directly, which allows it to be overridden by a derived class
    def get_value(self):
        self._sleep_check()
        text = self.text()
        if text == self._no_value_button_text or not hasattr(self, 'item_map') or not text:
            return None
        if self._special_items:
            try:
                return self._special_items[[str(si) for si in self._special_items].index(text)]
            except ValueError:
                pass
        return self.item_map[text]

    @value.setter
    def value(self, val):
        self.set_value(val)

    def set_value(self, val):
        self._sleep_check()
        # if value is being set to a special item, it may not be safe to call 'self.value', so handle that
        if val in self._special_items:
            if val != self.text():
                self.setText(val)
                if not self.signalsBlocked():
                    self.value_changed.emit()
            return
        # if value is being set to None, it may not be safe to call 'self.value' either, so handle that too
        if (val is None and self.text() in [self._no_value_button_text, ""]) \
        or (val is not None and self.get_value() == val):
            if val is None and not self.text():
                self.setText(self._no_value_button_text)
            return
        if val is None or not self.value_map:
            self.setText(self._no_value_button_text)
        elif val in self._special_items:
            self.setText(str(val))
        else:
            self.setText(self.value_map[val])
        if not self.signalsBlocked():
            self.value_changed.emit()

    def _items_change(self, *args):
        if self.__dict__.get('_destroyed', False):
            return
        del_recursion = False
        if not hasattr(self, '_recursion'):
            self._recursion = True
            del_recursion = True
        prev_value = self.get_value()
        item_names = self._item_names()
        special_names = []
        if self._no_value_menu_text is not None:
            special_names.append(self._no_value_menu_text)
        if self._special_items:
            special_names.extend([str(si) for si in self._special_items])
        menu = self.menu()
        menu.clear()
        if special_names:
            for special_name in special_names:
                menu.addAction(special_name)
            menu.addSeparator()
        for item_name in item_names:
            menu.addAction(item_name)
        if prev_value not in self.value_map and prev_value not in self._special_items:
            if len(self.value_map) + len(self._special_items) == 1 and self._autoselect_single:
                # value setter may use previous value, so prevent that by setting to None first,
                # blocking the value_changed signal as we do so
                preblocked = self.signalsBlocked()
                if not preblocked:
                    self.blockSignals(True)
                self.set_value(None)
                if not preblocked:
                    self.blockSignals(False)
                self.set_value((list(self.value_map.keys()) + self._special_items)[0])
            else:
                self.set_value(None)
        elif prev_value in self.value_map:
            # item name (only) may have changed
            self.setText(self.value_map[prev_value])
        if del_recursion:
            delattr(self, '_recursion')

    def _sel_change(self, action):
        if action.text() == self._no_value_menu_text:
            next_text = self._no_value_button_text
        else:
            next_text = action.text()
        if self.text() != next_text:
            self.setText(next_text)
            if not self.signalsBlocked():
                self.value_changed.emit()

class ModelMenuButton(ItemMenuButton):
    """Maintain a popup menu of models

       Keep menu up to date as models are opened and closed while keeping the selected item(s) the same

       'autoselect_single_item' controls whether the only item in a menu is automatically selected or not.

       'no_value_button_text' is the text shown on the menu button when no item is selected for whatever
       reason.  In such cases, self.value will be None.

       If 'no_value_menu_text' is not None, then there will be an additional entry in the menu with that
       text, and choosing that menu item is treated as setting self.value to None.

       'balloon_help' is the balloon help to provide with the list (if any).

       Do not access or set the value of this widget in trigger handlers that also update the widget.
       For generic models, those triggers are in session.triggers and are named ADD_MODELS, REMOVE_MODELS,
       MODEL_ID_CHANGED and MODEL_NAME_CHANGED
    """
    def __init__(self, session, *, no_value_button_text="No model chosen", **kw):
        kw['no_value_button_text'] = no_value_button_text
        super().__init__(**_process_model_kw(session, **kw))
