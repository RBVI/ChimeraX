# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.tools import ToolInstance


class ModelPanel(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        self.display_name = "Models"
        from chimerax.core.ui.gui import MainToolWindow

        class ModelPanelWindow(MainToolWindow):
            close_destroys = False

        self.settings = ModelPanelSettings(session, "ModelPanel")
        last = self.settings.last_use
        from time import time
        now = self.settings.last_use = time()
        short_titles = last != None and now - last < 777700 # about 3 months

        self.tool_window = ModelPanelWindow(self)
        parent = self.tool_window.ui_area
        from PyQt5.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QFrame, QPushButton
        self.tree = QTreeWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.tree)
        layout.setStretchFactor(self.tree, 1)
        parent.setLayout(layout)
        title = "S" if short_titles else "Shown"
        self.tree.setHeaderLabels(["ID", " ", title, "Name"])
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.itemClicked.connect(self._tree_change_cb)
        buttons_layout = QVBoxLayout()
        layout.addLayout(buttons_layout)
        for model_func in [close, hide, show, view]:
            button = QPushButton(model_func.__name__.capitalize())
            buttons_layout.addWidget(button)
            button.clicked.connect(lambda chk, self=self, mf=model_func, ses=session:
                mf([self.models[row] for row in [self._items.index(i)
                    for i in self.tree.selectedItems()]] or self.models, ses))

        from chimerax.core.graphics import Drawing
        Drawing.triggers.add_handler('display changed', self._initiate_fill_tree)
        self._fill_tree()
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        self.session.triggers.add_handler(ADD_MODELS, self._initiate_fill_tree)
        self.session.triggers.add_handler(REMOVE_MODELS, self._initiate_fill_tree)
        self.session.triggers.add_handler("atomic changes", self._changes_cb)
        self._frame_drawn_handler = None
        self.tool_window.manage(placement="right")

    @classmethod
    def get_singleton(self, session):
        from chimerax.core import tools
        return tools.get_singleton(session, ModelPanel, 'model panel', create=False)

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
        update = self._process_models()
        if not update:
            self.tree.clear()
            self._items = []
        from PyQt5.QtWidgets import QTreeWidgetItem
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QColor
        item_stack = [self.tree.invisibleRootItem()]
        for model in self.models:
            if update:
                if len(model.id) == len(item_stack):
                    # first child
                    item = item_stack[-1].child(0)
                    item_stack.append(item)
                else:
                    # sibling
                    len_id = len(model.id)
                    parent, previous_child = item_stack[len_id-1:len_id+1]
                    item = parent.child(parent.indexOfChild(previous_child)+1)
                    item_stack[len(model.id):] = [item]
            else:
                parent = item_stack[len(model.id)-1]
                item = QTreeWidgetItem(parent)
                item_stack[len(model.id):] = [item]
                self._items.append(item)
            item.setText(0, model.id_string())
            bg_color = self._model_color(model)
            bg = item.background(1)
            if bg_color is None:
                bg.setStyle(Qt.NoBrush)
            else:
                bg.setStyle(Qt.SolidPattern)
                bg.setColor(QColor(*bg_color))
            item.setBackground(1, bg)
            item.setCheckState(2, Qt.Checked if model.display else Qt.Unchecked)
            item.setText(3, getattr(model, "name", "(unnamed)"))
        for i in range(1,self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _model_color(self, model):
        return model.single_color

    def _process_models(self):
        models = self.session.models.list()
        tree_models = []
        chains = []
        sorted_models = sorted(models, key=lambda m: m.id)
        update = True if hasattr(self, 'models') and sorted_models == self.models else False
        self.models = sorted_models
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
    session.models.close(models)

def hide(models, session):
    for m in models:
        m.display = False

_mp = None
def model_panel(session, bundle_info):
    global _mp
    if _mp is None:
        _mp = ModelPanel(session, bundle_info)
    return _mp

def show(models, session):
    for m in models:
        m.display = True

def view(models, session):
    from chimerax.core.objects import Objects
    view_objects = Objects(models=models)
    for model in models:
        if getattr(model, 'atoms', None):
            view_objects.add_atoms(model.atoms)
    from chimerax.core.commands.view import view
    view(session, view_objects)

