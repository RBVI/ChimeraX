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
        self.list_func = self.filter_func = self.key_func = self.item_text_func = None
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

    AUTOSELECT_ALL = "all"
    AUTOSELECT_FIRST = "first"
    AUTOSELECT_FIRST_DISPLAYED = "first displayed"
    AUTOSELECT_LAST = "last"
    AUTOSELECT_LAST_DISPLAYED = "last displayed"
    AUTOSELECT_NONE = None
    AUTOSELECT_SINGLE = "single"
    autoselect_default = AUTOSELECT_ALL

    from chimerax.mouse_modes import mod_key_info
    extended_balloon_help = "Click to choose item\n" \
        "Drag to choose range\n" \
        "%s-click to toggle item\n" \
        "Shift-click to choose range\n  (starting from previous selected item)" \
            % mod_key_info("control")[1].capitalize()

    def __init__(self, *, autoselect='default', selection_mode='extended', balloon_help=None,
            **kw):
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
        self.itemSelectionChanged.disconnect()
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

    def set_value(self, val, *, delayed=False):
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
        if delayed:
            # allow all widgets to get to correct values before emitting signal
            _when_all_updated(self, self.itemSelectionChanged.emit)
        else:
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
        if not filtered_sel and self.autoselect != self.AUTOSELECT_NONE:
            # no previously selected items still listed -- use autoselect
            if self.autoselect == self.AUTOSELECT_ALL:
                filtered_sel = item_names
            elif self.autoselect == self.AUTOSELECT_SINGLE:
                if len(item_names) == 1:
                    filtered_sel = item_names
            elif self.autoselect == self.AUTOSELECT_FIRST:
                if len(item_names) > 0:
                    filtered_sel = item_names[:1]
            elif self.autoselect == self.AUTOSELECT_LAST:
                if len(item_names) > 0:
                    filtered_sel = item_names[-1:]
            elif self.autoselect in (self.AUTOSELECT_FIRST_DISPLAYED, self.AUTOSELECT_LAST_DISPLAYED):
                displayed = [v for v in [self._item_map[i] for i in self.items]
                    if getattr(v, 'display', True)]
                if len(displayed) > 0:
                    shown = [self.value_map[v] for v in displayed]
                    if self.auto_select == self.AUTOSELECT_FIRST_DISPLAYED:
                        filtered_sel = shown[:1]
                    else:
                        filtered_sel = shown[-1:]
                elif len(item_names) > 1:
                    if self.autoselect == self.AUTOSELECT_FIRST:
                        filtered_sel = item_names[:1]
                    else:
                        filtered_sel = item_names[-1:]
 
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
            self.set_value(next_value, delayed=True)
            # if items were deleted, then the current selection could be empty when the previous
            # one was not, but the test in the value setter will think the value is unchanged
            # and not emit the changed signal, so check for that here
            if len(sel) > 0 and not next_value:
                _when_all_updated(self, self.itemSelectionChanged.emit)
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

       'autoselect' controls what happens when nothing would be selected.  The possible values (which
       are class variables) are:
           - AUTOSELECT_ALL: all items are selected
           - AUTOSELECT_SINGLE: if there is only one item in the list, it will be selected
           - AUTOSELECT_NONE: select nothing
           - AUTOSELECT_FIRST: the first item in the list will be selected
           - AUTOSELECT_FIRST_DISPLAYED: select the first item whose 'display' attribute is True; if there
                is none, then the first item
           - AUTOSELECT_LAST, AUTOSELECT_LAST_DISPLAYED: analogous to the _FIRST_ values, except the last
                item instead of the first.
       The default is AUTOSELECT_ALL.

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

    AUTOSELECT_FIRST = "first"
    AUTOSELECT_FIRST_DISPLAYED = "first displayed"
    AUTOSELECT_LAST = "last"
    AUTOSELECT_LAST_DISPLAYED = "last displayed"
    AUTOSELECT_NONE = None
    AUTOSELECT_SINGLE = "single"
    autoselect_default = AUTOSELECT_SINGLE

    def __init__(self, *, autoselect="default", balloon_help=None,
            no_value_button_text="No item chosen", no_value_menu_text=None, special_items=[],
            autoselect_single_item=None, **kw):
        if autoselect == "default":
            if autoselect_single_item is not None:
                # the obsolete "autoselect_single_item keyword was explicitly used
                if autoselect_single_item:
                    autoselect = self.AUTOSELECT_SINGLE
                else:
                    autoselect = self.AUTOSELECT_NONE
            else:
                autoselect = self.autoselect_default
        self._autoselect = autoselect
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

    def set_value(self, val, *, delayed=False):
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
            if delayed:
                # allow all widgets to get to correct values before emitting signal...
                _when_all_updated(self, self.value_changed.emit)
            else:
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
            # previous selection no longer in menu -- use autoselect
            entries = list(self.value_map.keys())
            next_val = None
            if self._autoselect == self.AUTOSELECT_SINGLE:
                if len(entries) == 1:
                    next_val = entries[0]
            elif self._autoselect == self.AUTOSELECT_FIRST:
                if len(entries) > 0:
                    next_val = entries[0]
            elif self._autoselect == self.AUTOSELECT_LAST:
                if len(entries) > 0:
                    next_val = entries[-1]
            elif self._autoselect in (self.AUTOSELECT_FIRST_DISPLAYED, self.AUTOSELECT_LAST_DISPLAYED):
                displayed = [v for v in self.value_map.keys() if getattr(v, 'display', True)]
                if len(displayed) > 0:
                    if self._autoselect == self.AUTOSELECT_FIRST_DISPLAYED:
                        next_val = displayed[0]
                    else:
                        next_val = displayed[-1]
                elif len(entries) > 1:
                    if self._autoselect == self.AUTOSELECT_FIRST:
                        next_val = entries[0]
                    else:
                        next_val = entries[-1]
            if next_val is not None:
                # value setter may use previous value, so prevent that by setting to None first,
                # blocking the value_changed signal as we do so
                preblocked = self.signalsBlocked()
                if not preblocked:
                    self.blockSignals(True)
                self.set_value(None)
                if not preblocked:
                    self.blockSignals(False)
            self.set_value(next_val, delayed=True)
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

       'special_items' is a list of special additional menu items whose str() reprs will be added to the
       menu.

       'autoselect' controls what happens when nothing would be selected.  The possible values (which
       are class variables) are:
           - AUTOSELECT_SINGLE: if there is only one item in the menu, it will be selected
           - AUTOSELECT_NONE: select nothing
           - AUTOSELECT_FIRST: the first item in the menu will be selected
           - AUTOSELECT_FIRST_DISPLAYED: select the first item whose 'display' attribute is True; if there
                is none, the the first item
           - AUTOSELECT_LAST, AUTOSELECT_LAST_DISPLAYED: analogous to the _FIRST_ values, except the last
                item instead of the first.
       Special items are ignored for purposes of autoselection. The default is AUTOSELECT_SINGLE.

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

def _when_all_updated(widget, func):
    def check_and_execute(*, widget=widget, func=func):
        if not widget.__dict__.get("_destroyed", False):
            func()
    from Qt.QtCore import QTimer
    QTimer.singleShot(0, check_and_execute)
