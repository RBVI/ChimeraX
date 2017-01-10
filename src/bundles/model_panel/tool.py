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

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
# TODO: Changing display_name to be different from tool_name breaks toolshed hide command.
        self.display_name = "Models"
        self.settings = ModelPanelSettings(session, "ModelPanel")
        last = self.settings.last_use
        from time import time
        now = self.settings.last_use = time()
        short_titles = last != None and now - last < 777700 # about 3 months

        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area
        from PyQt5.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QFrame, QPushButton
        self.tree = QTreeWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.tree)
        layout.setStretchFactor(self.tree, 1)
        parent.setLayout(layout)
        title = "S" if short_titles else "Shown"
        self.tree.setHeaderLabels(["ID", " ", title, "Name"])
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setAnimated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.itemClicked.connect(self._tree_change_cb)
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
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        self.session.triggers.add_handler(ADD_MODELS, self._initiate_fill_tree)
        self.session.triggers.add_handler(REMOVE_MODELS, self._initiate_fill_tree)
        self.session.triggers.add_handler("atomic changes", self._changes_cb)
        self._frame_drawn_handler = None
        tw.manage(placement="side")
        tw.shown_changed = self._shown_changed

    def _shown_changed(self, shown):
        if shown:
            # Update panel when it is shown.
            self._fill_tree()

    @classmethod
    def get_singleton(self, session):
        from chimerax.core import tools
        return tools.get_singleton(session, ModelPanel, 'Model Panel', create=False)

    def _changes_cb(self, trigger_name, changes):
        reasons = changes.atom_reasons()
        if "color changed" in reasons or 'display changed' in reasons:
            self._initiate_fill_tree()

    def _initiate_fill_tree(self, *args):
        # in order to allow molecules to be drawn as quickly as possible,
        # delay the update of the tree until the 'frame drawn' trigger fires
        if self._frame_drawn_handler is None:
            self._frame_drawn_handler = self.session.triggers.add_handler(
                "frame drawn", self._fill_tree)

    def _fill_tree(self, *args):
        if not self.displayed():
            # Don't update panel when it is hidden.
            return
        update = self._process_models()
        if not update:
            expanded_models = { i._model : i.isExpanded()
                                for i in self._items if hasattr(i, '_model')}
            self.tree.clear()
            self._items = []
        from PyQt5.QtWidgets import QTreeWidgetItem, QPushButton
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QColor
        item_stack = [self.tree.invisibleRootItem()]
        for model in self.models:
            model_id, model_id_string, bg_color, display, name = self._get_info(model)
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
                parent = item_stack[len_id-1]
                item = QTreeWidgetItem(parent)
                item._model = model
                item_stack[len_id:] = [item]
                self._items.append(item)
                if bg_color is not None:
                    from chimerax.core.ui.widgets import ColorButton
                    but = ColorButton(has_alpha_channel=True, max_size=(16,16))
                    def set_single_color(rgba, m=model):
                        for cm in m.all_models():
                            cm.single_color = rgba
                    but.color_changed.connect(set_single_color)
                    but.set_color(bg_color)
                    self.tree.setItemWidget(item, 1, but)
            item.setText(0, model_id_string)
            bg = item.background(1)
            if bg_color is None:
                bg.setStyle(Qt.NoBrush)
            else:
                but = self.tree.itemWidget(item, 1)
                if but is not None:
                    but.set_color(bg_color)
            item.setBackground(1, bg)
            if display is not None:
                item.setCheckState(2, Qt.Checked if display else Qt.Unchecked)
            item.setText(3, name)
            if not update:
                # Expand new top-level displayed models, or if previously expanded
                expand = expanded_models.get(model, (model.display and len(model.id) <= 1))
                if expand:
                    self.tree.expandItem(item)
        for i in range(1,self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _get_info(self, obj):
        model_id = obj.id
        model_id_string = obj.id_string()
        bg_color = self._model_color(obj)
        display = obj.display
        name = getattr(obj, "name", "(unnamed)")
        return model_id, model_id_string, bg_color, display, name

    def _header_click_cb(self, index):
        if index == 0:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()

    def _left_click(self, event):
        if event.Col == 2:
            model = self.models[event.Row]
            model.display = not model.display
        event.Skip()

    def _label_click(self, event):
        if event.Col == 0:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()
        event.Skip()

    def _model_color(self, model):
        return model.single_color

    def _process_models(self):
        models = self.session.models.list()
        tree_models = []
        sorted_models = sorted(models, key=lambda m: m.id)
        from chimerax.core.atomic import AtomicStructure
        final_models = []
        for model in sorted_models:
            final_models.append(model)
        update = True if hasattr(self, 'models') and final_models == self.models else False
        self.models = final_models
        return update

    def _tree_change_cb(self, item, column):
        if column != 2:
            # not the shown check box
            return
        from PyQt5.QtCore import Qt
        self.models[self._items.index(item)].display = item.checkState(2) == Qt.Checked

from chimerax.core.settings import Settings
class ModelPanelSettings(Settings):
    AUTO_SAVE = {
        'last_use': None
    }

def close(models, session):
    from chimerax.core.models import Model
    session.models.close([m for m in models if isinstance(m, Model)])

def hide(models, session):
    for m in models:
        m.display = False

_mp = None
def model_panel(session, tool_name):
    global _mp
    if _mp is None:
        _mp = ModelPanel(session, tool_name)
    return _mp

def show(models, session):
    for m in models:
        m.display = True

def view(objs, session):
    from chimerax.core.models import Model
    models = [o for o in objs if isinstance(o, Model)]
    from chimerax.core.objects import Objects
    view_objects = Objects(models=models)
    for model in models:
        if getattr(model, 'atoms', None):
            view_objects.add_atoms(model.atoms)
    from chimerax.core.commands.view import view
    view(session, view_objects)

