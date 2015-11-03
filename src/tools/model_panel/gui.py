# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimera.core.tools import ToolInstance


class ModelPanel(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    SIZE = (200, 250)

    def __init__(self, session, tool_info, *, restoring=False):
        if not restoring:
            ToolInstance.__init__(self, session, tool_info)
        self.display_name = "Models"
        from chimera.core.ui import MainToolWindow

        class ModelPanelWindow(MainToolWindow):
            close_destroys = False

        self.tool_window = ModelPanelWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        import wx
        import wx.grid
        self.table = wx.grid.Grid(parent, size=(200, 150))
        self.table.CreateGrid(5, 3)
        self.table.SetColLabelValue(0, "ID")
        self.table.SetColSize(0, 25)
        self.table.SetColLabelValue(1, " ")
        self.table.SetColSize(1, -1)
        self.table.SetColLabelValue(2, "Name")
        self.table.HideRowLabels()
        self.table.SetDefaultCellAlignment(wx.ALIGN_CENTRE, wx.ALIGN_BOTTOM)
        self.table.EnableEditing(False)
        self.table.SelectionMode = wx.grid.Grid.GridSelectRows
        self.table.CellHighlightPenWidth = 0
        self._fill_table()
        from chimera.core.models import ADD_MODELS, REMOVE_MODELS
        self.session.triggers.add_handler(ADD_MODELS, self._initiate_fill_table)
        self.session.triggers.add_handler(REMOVE_MODELS, self._initiate_fill_table)
        self.session.triggers.add_handler("atomic changes", self._changes_cb)
        self._frame_drawn_handler = None
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.table, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        data = {
            "ti": ToolInstance.take_snapshot(self, session, flags),
            "shown": self.tool_window.shown
        }
        return self.tool_info.session_write_version, data

    def restore_snapshot_init(self, session, tool_info, version, data):
        if version not in tool_info.session_versions:
            from chimera.core.state import RestoreError
            raise RestoreError("unexpected version")
        ti_version, ti_data = data["ti"]
        ToolInstance.restore_snapshot_init(
            self, session, tool_info, ti_version, ti_data)
        self.__init__(session, tool_info, restoring=True)
        self.display(data["shown"])

    def reset_state(self, session):
        pass

    def _changes_cb(self, trigger_name, data):
        reasons = data["Atom"].reasons
        if "color changed" in reasons or 'display changed' in reasons:
            self._initiate_fill_table()

    def _initiate_fill_table(self, *args):
        # in order to allow molecules to be drawn as quickly as possible,
        # delay the update of the table until the 'frame drawn' trigger fires
        if self._frame_drawn_handler is None:
            self._frame_drawn_handler = self.session.triggers.add_handler(
                "frame drawn", self._fill_table)

    def _fill_table(self, *args):
        # prevent repaints untill the end of this method...
        import wx
        locker = wx.grid.GridUpdateLocker(self.table)
        nr = self.table.NumberRows
        if nr:
            self.table.DeleteRows(0, nr)
        models = self.session.models.list()
        self.table.AppendRows(len(models))
        models = sorted(models, key=lambda m: m.id)
        for i, model in enumerate(models):
            self.table.SetCellValue(i, 0, model.id_string())
            self.table.SetCellBackgroundColour(i, 1, self._model_color(model))
            self.table.SetCellValue(i, 2, getattr(model, "name", "(unnamed)"))
        self.table.AutoSizeColumns()
        del locker

        self._frame_drawn_handler = None
        from chimera.core.triggerset import DEREGISTER
        return DEREGISTER

    def _model_color(self, model):
        # should be done generically
        residues = getattr(model, 'residues', None)
        if residues:
            ribbon_displays = residues.ribbon_displays
            if ribbon_displays.any():
                return most_common_color(residues.filter(ribbon_displays).ribbon_colors)
        atoms = getattr(model, 'atoms', None)
        if atoms:
            shown = atoms.filter(atoms.displays)
            if shown:
                return most_common_color(shown.colors)
        return None


def most_common_color(colors):
    import numpy
    as32 = colors.view(numpy.int32).reshape((len(colors),))
    unique, indices, counts = numpy.unique(as32, return_index=True, return_counts=True)
    if counts[0] < len(colors) / 10:
        return None
    return colors[indices[0]]
