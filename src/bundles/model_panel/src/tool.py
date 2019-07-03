# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.tools import ToolInstance


class ModelPanel(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    help = "help:user/tools/modelpanel.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "Models"
        self.settings = ModelPanelSettings(session, "ModelPanel")
        last = self.settings.last_use
        from time import time
        now = self.settings.last_use = time()
        short_titles = last != None and now - last < 777700 # about 3 months

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area
        from PyQt5.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QFrame, QPushButton
        self.tree = QTreeWidget()
        self.tree.keyPressEvent = session.ui.forward_keystroke
        self.tree.expanded.connect(self._ensure_id_width)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.tree)
        layout.setStretchFactor(self.tree, 1)
        parent.setLayout(layout)
        shown_title = "" if short_titles else "Shown"
        sel_title = "" if short_titles else "Select"
        self.tree.setHeaderLabels(["Name", "ID", " ", shown_title, sel_title])
        from chimerax.ui.icons import get_qt_icon
        self.tree.headerItem().setIcon(3, get_qt_icon("shown"))
        self.tree.headerItem().setToolTip(3, "Shown")
        self.tree.headerItem().setIcon(4, get_qt_icon("select"))
        self.tree.headerItem().setToolTip(4, "Selected")
        self.tree.setColumnWidth(self.NAME_COLUMN, 200)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setAnimated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.itemChanged.connect(self._tree_change_cb)
        buttons_layout = QVBoxLayout()
        layout.addLayout(buttons_layout)
        self._items = []
        for model_func in [close, hide, show, view]:
            button = QPushButton(model_func.__name__.capitalize())
            buttons_layout.addWidget(button)
            button.clicked.connect(lambda chk, self=self, mf=model_func, ses=session:
                mf([self.models[row] for row in [self._items.index(i)
                    for i in self.tree.selectedItems()]] or self.models, ses))
        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        session.triggers.add_handler(MODEL_DISPLAY_CHANGED, self._initiate_fill_tree)
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, \
            MODEL_ID_CHANGED, MODEL_NAME_CHANGED
        from chimerax.core.selection import SELECTION_CHANGED
        self.session.triggers.add_handler(SELECTION_CHANGED, self._initiate_fill_tree)
        self.session.triggers.add_handler(ADD_MODELS, self._initiate_fill_tree)
        self.session.triggers.add_handler(REMOVE_MODELS, self._initiate_fill_tree)
        self.session.triggers.add_handler(MODEL_ID_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, always_rebuild=True))
        self.session.triggers.add_handler(MODEL_NAME_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, refresh=True))
        from chimerax import atomic
        atomic.get_triggers().add_handler("changes", self._changes_cb)
        self._frame_drawn_handler = None
        tw.manage(placement="side")
        tw.shown_changed = self._shown_changed

    NAME_COLUMN = 0
    ID_COLUMN = 1
    COLOR_COLUMN = 2
    SHOWN_COLUMN = 3
    SELECT_COLUMN = 4
    
    def _shown_changed(self, shown):
        if shown:
            # Update panel when it is shown.
            self._initiate_fill_tree(refresh=True)

    @classmethod
    def get_singleton(self, session):
        from chimerax.core import tools
        return tools.get_singleton(session, ModelPanel, 'Model Panel', create=False)

    def _changes_cb(self, trigger_name, changes):
        reasons = changes.atom_reasons()
        if "color changed" in reasons or 'display changed' in reasons:
            self._initiate_fill_tree()

    def _ensure_id_width(self, *args):
        # ensure that the newly visible model id isn't just "..."
        self.tree.resizeColumnToContents(self.ID_COLUMN)

    def _initiate_fill_tree(self, *args, always_rebuild=False, refresh=False):
        # in order to allow molecules to be drawn as quickly as possible,
        # delay the update of the tree until the 'frame drawn' trigger fires,
        # unless no models are open, in which case update immediately because
        # Rapid Access will come up and 'frame drawn' may not fire for awhile.
        # Also do immediately if the cause is a model-name change, since the
        # frame-drawn trigger may not fire for awhile
        if len(self.session.models) == 0 or refresh:
            if self._frame_drawn_handler is not None:
                self.session.triggers.remove_handler(self._frame_drawn_handler)
            self._fill_tree(always_rebuild=always_rebuild)
        elif self._frame_drawn_handler is None:
            self._frame_drawn_handler = self.session.triggers.add_handler("frame drawn",
                lambda *args, ft=self._fill_tree, ar=always_rebuild: ft(always_rebuild=ar))
        elif always_rebuild:
            self.session.triggers.remove_handler(self._frame_drawn_handler)
            self._frame_drawn_handler = self.session.triggers.add_handler("frame drawn",
                lambda *args, ft=self._fill_tree: ft(always_rebuild=True))

    def _fill_tree(self, *, always_rebuild=False):
        if not self.displayed():
            # Don't update panel when it is hidden.
            return
        self.tree.blockSignals(True) # particularly itemChanged
        update = self._process_models() and not always_rebuild
        if not update:
            expanded_models = { i._model : i.isExpanded()
                                for i in self._items if hasattr(i, '_model')}
            self.tree.clear()
            self._items = []
        all_selected_models = self.session.selection.models(all_selected=True)
        part_selected_models = self.session.selection.models()
        from PyQt5.QtWidgets import QTreeWidgetItem, QPushButton
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QColor
        item_stack = [self.tree.invisibleRootItem()]
        for model in self.models:
            model_id, model_id_string, bg_color, display, name, selected, part_selected = \
                self._get_info(model, all_selected_models, part_selected_models)
            len_id = len(model_id)
            if update:
                if len_id == len(item_stack):
                    # first child
                    item = item_stack[-1].child(0)
                    item_stack.append(item)
                else:
                    # sibling
                    parent, previous_child = item_stack[len_id-1:len_id+1]
                    item = parent.child(parent.indexOfChild(previous_child)+1)
                    item_stack[len_id:] = [item]
            else:
                parent = item_stack[0] if len(item_stack) == 1 else item_stack[len_id-1]
                item = QTreeWidgetItem(parent)
                item._model = model
                item_stack[len_id:] = [item]
                self._items.append(item)
                if bg_color is not False:
                    from chimerax.ui.widgets import MultiColorButton
                    but = MultiColorButton(has_alpha_channel=True, max_size=(16,16))
                    def set_single_color(rgba, m=model):
                        for cm in m.all_models():
                            cm.single_color = rgba
                    but.color_changed.connect(set_single_color)
                    but.set_color(bg_color)
                    self.tree.setItemWidget(item, self.COLOR_COLUMN, but)
                
                    
                
            item.setText(self.ID_COLUMN, model_id_string)
            bg = item.background(self.ID_COLUMN)
            if bg_color is False:
                bg.setStyle(Qt.NoBrush)
            else:
                but = self.tree.itemWidget(item, self.COLOR_COLUMN)
                if but is not None:
                    but.set_color(bg_color)
            item.setBackground(self.COLOR_COLUMN, bg)
            if display is not None:
                item.setCheckState(self.SHOWN_COLUMN, Qt.Checked if display else Qt.Unchecked)
            if selected:
                item.setCheckState(self.SELECT_COLUMN, Qt.Checked)
            elif part_selected:
                item.setCheckState(self.SELECT_COLUMN, Qt.PartiallyChecked)
            else:
                item.setCheckState(self.SELECT_COLUMN, Qt.Unchecked)
            item.setText(self.NAME_COLUMN, name)
            if not update:
                # Expand new top-level displayed models, or if previously expanded
                if hasattr(model, 'model_panel_show_expanded'):
                    expand_default = model.model_panel_show_expanded
                else:
                    expand_default = (model.display
                                      and len(model.id) <= 1
                                      and len(model.child_models()) <= 10)
                expand = expanded_models.get(model, expand_default)
                if expand:
                    self.tree.expandItem(item)
        for i in range(1,self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)
        self.tree.blockSignals(False)

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _get_info(self, obj, all_selected_models, part_selected_models):
        model_id = obj.id
        model_id_string = obj.id_string
        bg_color = self._model_color(obj)
        display = obj.display
        name = getattr(obj, "name", "(unnamed)")
        selected = obj in all_selected_models
        part_selected = selected or obj in part_selected_models
        return model_id, model_id_string, bg_color, display, name, selected, part_selected

    def _header_click_cb(self, index):
        if index == 0:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()

    def _label_click(self, event):
        if event.Col == self.ID_COLUMN:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()
        event.Skip()

    def _model_color(self, model):
        return model.single_color

    def _process_models(self):
        models = self.session.models.list()
        sorted_models = sorted(models, key=lambda m: m.id)
        final_models = list(sorted_models)
        update = True if hasattr(self, 'models') and final_models == self.models else False
        self.models = final_models
        return update

    def _tree_change_cb(self, item, column):
        from PyQt5.QtCore import Qt
        model = self.models[self._items.index(item)]
        if column == self.SHOWN_COLUMN:
            command_name = "show" if item.checkState(self.SHOWN_COLUMN) == Qt.Checked else "hide"
            run(self.session, "%s #!%s models" % (command_name, model.id_string))
        elif column == self.SELECT_COLUMN:
            prefix = "" if item.checkState(self.SELECT_COLUMN) == Qt.Checked else "~"
            run(self.session, prefix + "select #" + model.id_string)

from chimerax.core.settings import Settings
class ModelPanelSettings(Settings):
    AUTO_SAVE = {
        'last_use': None
    }

from chimerax.core.commands import run, concise_model_spec
def close(models, session):
    from chimerax.core.models import Model
    run(session, "close %s" %
        concise_model_spec(session, [m for m in models if isinstance(m, Model)]))

def hide(models, session):
    run(session, "hide %s target m" % concise_model_spec(session, models))

_mp = None
def model_panel(session, tool_name):
    global _mp
    if _mp is None:
        _mp = ModelPanel(session, tool_name)
    return _mp

def show(models, session):
    run(session, "show %s target m" % concise_model_spec(session, models))

def view(objs, session):
    from chimerax.core.models import Model
    models = [o for o in objs if isinstance(o, Model)]
    run(session, "view %s clip false" % concise_model_spec(session, models))

