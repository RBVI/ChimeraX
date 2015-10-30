# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimera.core.tools import ToolInstance


class ModelPanel(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    SIZE = (200, 250)
    VERSION = 1

    def __init__(self, session, tool_info):
        super().__init__(session, tool_info)
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
        self.session.triggers.add_handler(ADD_MODELS, self._fill_table)
        self.session.triggers.add_handler(REMOVE_MODELS, self._fill_table)
        self.session.triggers.add_handler("atomic changes", self._changes_cb)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.table, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        version = self.VERSION
        data = {}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import RestoreError
        if version != self.VERSION or len(data) > 0:
            raise RestoreError("unexpected version or data")
        if phase == self.CREATE_PHASE:
            # Restore all basic-type attributes
            pass
        else:
            # Resolve references to objects
            pass

    def reset_state(self):
        pass

    def _changes_cb(self, trigger_name, data):
        reasons = data["Atom"].reasons
        if "color changed" in reasons or 'display changed' in reasons:
            self._fill_table()

    def _fill_table(self, *args):
        import wx.grid
        # prevent repaints untill the end of this method...
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

    def _model_color(self, model):
        # should be done generically
        residues = getattr(model, 'residues', None)
        if residues:
            ribbon_displays = residues.ribbon_displays
            if ribbon_displays.any():
                ribbon_colors = residues.filter(ribbon_displays).ribbon_colors
                if (ribbon_colors == ribbon_colors[0]).all():
                    import wx
                    return wx.Colour(*tuple(ribbon_colors[0]))
        atoms = getattr(model, 'atoms', None)
        if atoms:
            shown = atoms.filter(atoms.displays)
            shown_carbons = shown.filter(shown.element_numbers == 6)
            if shown_carbons:
                colors = shown_carbons.colors
                # are they all the same?
                if (colors == colors[0]).all():
                    import wx
                    return wx.Colour(*tuple(colors[0]))
        return None
