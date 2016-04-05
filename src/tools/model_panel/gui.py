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
        else:
            from PyQt5.QtWidgets import QTableWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
                QFrame, QPushButton
            self.table = QTableWidget(5, 4)
            layout = QHBoxLayout()
            layout.addWidget(self.table)
            layout.setStretchFactor(self.table, 1)
            parent.setLayout(layout)
            title = "S" if short_titles else "Shown"
            self.table.setHorizontalHeaderLabels(["ID", " ", title, "Name"])
            self.table.verticalHeader().hide()
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.resizeColumnsToContents()
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setShowGrid(False)
            self.table.itemClicked.connect(self._table_change_cb)
            self.table.horizontalHeader().sectionClicked.connect(self._header_click_cb)
            buttons_layout = QVBoxLayout()
            layout.addLayout(buttons_layout)
            for model_func in [close, hide, show, view]:
                button = QPushButton(model_func.__name__.capitalize())
                buttons_layout.addWidget(button)
                button.clicked.connect(lambda chk, self=self, mf=model_func, ses=session:
                    # Qt returns each row index N times (N == # of columns)...
                    mf([self.models[row] for row in set([i.row()
                        for i in self.table.selectedIndexes()])] or self.models, ses))

        from chimerax.core.graphics import Drawing
        Drawing.triggers.add_handler('display changed', self._initiate_fill_table)
        self._sort_breadth_first = False
        self._fill_table()
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        self.session.triggers.add_handler(ADD_MODELS, self._initiate_fill_table)
        self.session.triggers.add_handler(REMOVE_MODELS, self._initiate_fill_table)
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
            self._initiate_fill_table()

    def _initiate_fill_table(self, *args):
        # in order to allow molecules to be drawn as quickly as possible,
        # delay the update of the table until the 'frame drawn' trigger fires
        if self._frame_drawn_handler is None:
            self._frame_drawn_handler = self.session.triggers.add_handler(
                "frame drawn", self._fill_table)

    def _fill_table(self, *args):
        models = self.session.models.list()
        order = (lambda m: (len(m.id),m.id)) if self._sort_breadth_first else (lambda m: m.id)
        sorted_models = sorted(models, key=order)
        replace = True if hasattr(self, 'models') and sorted_models == self.models else False
        self.models = sorted_models
        from chimerax.core import window_sys
        if window_sys == "wx":
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
        else:
            if not replace:
                self.table.clearContents()
            from PyQt5.QtWidgets import QTableWidgetItem
            from PyQt5.QtCore import Qt
            from PyQt5.QtGui import QColor
            for i, model in enumerate(self.models):
                if not replace:
                    self.table.insertRow(i)
                    for col in range(4):
                        self.table.setItem(i, col, QTableWidgetItem())
                self.table.item(i, 0).setText(model.id_string())
                bg_color = self._model_color(model)
                bg = self.table.item(i, 1).background()
                if bg_color is None:
                    bg.setStyle(Qt.NoBrush)
                else:
                    bg.setStyle(Qt.SolidPattern)
                    bg.setColor(QColor(*bg_color))
                self.table.item(i, 1).setBackground(bg)
                self.table.item(i, 2).setCheckState(Qt.Checked if model.display else Qt.Unchecked)
                self.table.item(i, 3).setText(getattr(model, "name", "(unnamed)"))
            self.table.resizeColumnsToContents()

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _header_click_cb(self, index):
        if index == 0:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_table()

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
            self._fill_table()
        event.Skip()

    def _model_color(self, model):
        return model.single_color

    def _table_change_cb(self, item):
        if item.column() != 2:
            # not the shown check box
            return
        from PyQt5.QtCore import Qt
        self.models[item.row()].display = item.checkState() == Qt.Checked

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

