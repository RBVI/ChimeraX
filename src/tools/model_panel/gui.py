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

        from chimerax.core import window_sys
        kw = {'size': (200, 250) } if window_sys == "wx" else {}
        self.tool_window = ModelPanelWindow(self, **kw)
        parent = self.tool_window.ui_area
        if window_sys == "wx":
            self._fill_tree = self.wx_fill_table
            import wx
            import wx.grid
            self.table = wx.grid.Grid(parent, size=(200, 150))
            self.table.CreateGrid(5, 4)
            self.table.SetColLabelValue(0, "ID")
            self.table.SetColSize(0, 25)
            self.table.SetColLabelValue(1, " ")
            self.table.SetColSize(1, -1)
            title = "S" if short_titles else "Shown"
            self.table.SetColLabelValue(2, title)
            self.table.SetColSize(2, -1)
            self.table.SetColFormatBool(2)
            self.table.SetColLabelValue(3, "Name")
            self.table.HideRowLabels()
            self.table.SetDefaultCellAlignment(wx.ALIGN_CENTRE, wx.ALIGN_BOTTOM)
            self.table.EnableEditing(False)
            self.table.SelectionMode = wx.grid.Grid.GridSelectRows
            self.table.CellHighlightPenWidth = 0
            self.table.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self._label_click)
            self.table.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, self._left_click)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(self.table, 1, wx.EXPAND)
            button_sizer = wx.BoxSizer(wx.VERTICAL)
            for model_func in [close, hide, show, view]:
                button = wx.Button(parent, label=model_func.__name__.capitalize())
                button.Bind(wx.EVT_BUTTON, lambda e, self=self, mf=model_func, ses=session:
                    mf([self.models[i] for i in self.table.SelectedRows] or self.models, ses))
                button_sizer.Add(button, 0, wx.EXPAND)
            sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_VERTICAL)
            parent.SetSizerAndFit(sizer)
            self._sort_breadth_first = False
        else:
            self._fill_tree = self.qt_fill_tree
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
            self.tree.setAnimated(True)
            self.tree.setUniformRowHeights(True)
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

    def qt_fill_tree(self, *args):
        update = self._process_models()
        if not update:
            self.tree.clear()
            self._items = []
        from PyQt5.QtWidgets import QTreeWidgetItem
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
                item_stack[len_id:] = [item]
                self._items.append(item)
            item.setText(0, model_id_string)
            bg = item.background(1)
            if bg_color is None:
                bg.setStyle(Qt.NoBrush)
            else:
                bg.setStyle(Qt.SolidPattern)
                bg.setColor(QColor(*bg_color))
            item.setBackground(1, bg)
            if display is not None:
                item.setCheckState(2, Qt.Checked if display else Qt.Unchecked)
            item.setText(3, name)
        for i in range(1,self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def wx_fill_table(self, *args):
        models = self.session.models.list()
        order = (lambda m: (len(m.id),m.id)) if self._sort_breadth_first else (lambda m: m.id)
        sorted_models = sorted(models, key=order)
        replace = True if hasattr(self, 'models') and sorted_models == self.models else False
        self.models = sorted_models
        import wx
        # prevent repaints untill the end of this method...
        locker = wx.grid.GridUpdateLocker(self.table)
        if not replace:
            nr = self.table.NumberRows
            if nr:
                self.table.DeleteRows(0, nr)
            self.table.AppendRows(len(models))
        for i, model in enumerate(self.models):
            if not replace:
                self.table.SetCellValue(i, 0, model.id_string())
            self.table.SetCellBackgroundColour(i, 1, self._model_color(model))
            self.table.SetCellValue(i, 2, "1" if model.display else "")
            self.table.SetCellValue(i, 3, getattr(model, "name", "(unnamed)"))
        self.table.AutoSizeColumns()
        del locker

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _get_info(self, obj):
        from chimerax.core.atomic import Chain
        if isinstance(obj, Chain):
            chain_string = '/%s' % obj.chain_id
            model_id = obj.structure.id + (chain_string,)
            model_id_string = obj.structure.id_string() + chain_string
            bg_color = None
            display = None
            name = obj.description or "chain %s" % obj.chain_id
        else:
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
        chains = []
        sorted_models = sorted(models, key=lambda m: m.id)
        from chimerax.core.atomic import AtomicStructure
        final_models = []
        for model in sorted_models:
            final_models.append(model)
            if isinstance(model, AtomicStructure):
                final_models.extend([c for c in model.chains])
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
def model_panel(session, bundle_info):
    global _mp
    if _mp is None:
        _mp = ModelPanel(session, bundle_info)
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
    from chimerax.core.atomic import Chain
    for o in objs:
        if isinstance(o, Chain):
            view_objects.add_atoms(o.existing_residues.atoms)
    from chimerax.core.commands.view import view
    view(session, view_objects)

