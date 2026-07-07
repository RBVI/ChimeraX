# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from Qt.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QGraphicsView, QGraphicsScene
from Qt.QtGui import QBrush, QColor, QPen, QPolygonF
from Qt.QtCore import Qt, QPointF

from chimerax.core.commands import plural_of
from chimerax.core.errors import UserError

from .gui import _md_tool_windows

class RMSDMapLauncher:
    def __init__(self, main_tool_window, launcher_window, structure):
        self.main_tool_window = main_tool_window
        self.tool_window = tw = launcher_window
        #tw.help = "help:user/commands/coordset.html#clustering"
        def cleanup(lcd=self):
            inst = lcd.tool_window.tool_instance
            from .gui import _remove_tool_window
            _remove_tool_window(inst, "rmsd map launcher")
            delattr(lcd.tool_window, 'cleanup')
        tw.cleanup = cleanup
        self.session = structure.session
        self.structure = structure
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        tw.ui_area.setLayout(layout)

        title = QLabel("Create RMSD map of trajectory against itself")
        title.setAlignment(Qt.AlignCenter)
        title.setFrameStyle(QLabel.Box | QLabel.Raised)
        title.setLineWidth(2)
        title.setMidLineWidth(3)
        layout.addWidget(title)

        self.settings = settings = LaunchRMSDMapSettings(self.session, "launch RMSD map")
        from chimerax.ui.options import OptionsPanel, IntOption, BooleanOption, EnumOption, FloatOption
        options_panel = OptionsPanel(sorting=False, scrolled=False, contents_margins=(2,2,2,2))
        cs_ids = structure.coordset_ids
        min_cs = min(cs_ids)
        max_cs = max(cs_ids)
        self.start_opt = IntOption("Starting frame:", min_cs, None, min=min_cs, max=max_cs)
        options_panel.add_option(self.start_opt)
        self.step_opt = IntOption("Step size:", 1 + int(len(cs_ids)/300), None, min=1, max=max_cs)
        options_panel.add_option(self.step_opt)
        self.end_opt = IntOption("Ending frame:", max_cs, None, min=min_cs, max=max_cs)
        options_panel.add_option(self.end_opt)
        self.low_bound_opt = FloatOption("Lower RMSD threshold (white):", settings.low_rmsd, None,
            min=0.0, max=999.999)
        options_panel.add_option(self.low_bound_opt)
        self.high_bound_opt = FloatOption("Upper RMSD threshold (black):", settings.high_rmsd, None,
            min=0.0, max=999.999)
        options_panel.add_option(self.high_bound_opt)
        self.sel_opt = BooleanOption("Restrict map to current selection, if any:", True, None)
        options_panel.add_option(self.sel_opt)
        self.solvent_opt = BooleanOption("Ignore solvent and non-metal ions:", True, None)
        options_panel.add_option(self.solvent_opt)
        self.hyd_opt = BooleanOption("Ignore hydrogens:", True, None)
        options_panel.add_option(self.hyd_opt)
        self.ligand_opt = BooleanOption("Ignore ligands:", False, None)
        options_panel.add_option(self.ligand_opt)
        from .manager import get_plotting_manager
        mgr = get_plotting_manager(self.session)
        self.metal_opt = EnumOption("Ignore metal ions:", "alkali", None, values=mgr.exclude_info["metals"])
        options_panel.add_option(self.metal_opt)
        self.recolor_opt = BooleanOption("Auto-recolor for contrast:", settings.auto_recolor, None)
        options_panel.add_option(self.recolor_opt)
        layout.addWidget(options_panel)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_rmsd_map)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.launch_rmsd_map(apply=True))
        bbox.rejected.connect(tw.destroy)
        if getattr(tw, 'help', None):
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=self.session: run(ses, "help " + tw.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(None)

    def launch_rmsd_map(self, *, apply=False):
        start = self.start_opt.value
        step = self.step_opt.value
        end = self.end_opt.value
        sel = self.sel_opt.value
        low = self.low_bound_opt.value
        high = self.high_bound_opt.value
        solvent = self.solvent_opt.value
        hyd = self.hyd_opt.value
        ligand = self.ligand_opt.value
        metal = self.metal_opt.value
        recolor = self.recolor_opt.value
        if not apply:
            self.tool_window.destroy()
        inst = self.main_tool_window.tool_instance
        inst_window_info = _md_tool_windows.setdefault(inst, {})
        map_results = inst_window_info.setdefault("RMSD map", [])
        map_results.append(
            RMSDMap(self.main_tool_window.create_child_window("RMSD Map", statusbar=True),
                self.structure, start, step, end, sel, low, high, solvent, hyd, ligand, metal, recolor))

from chimerax.core.settings import Settings
class LaunchRMSDMapSettings(Settings):
    AUTO_SAVE = {
        "auto_recolor": True,
        "low_rmsd": 0.5,
        "high_rmsd": 3.0,
    }

def show_rmsd_map_launcher(main_tool_window, structure):
    inst = main_tool_window.tool_instance
    inst_window_info = _md_tool_windows.setdefault(inst, {})
    try:
        rmsd_map_launcher = inst_window_info["rmsd map launcher"]
    except KeyError:
        rmsd_map_launcher = inst_window_info["rmsd map launcher"] = RMSDMapLauncher(main_tool_window,
            main_tool_window.create_child_window("Get RMSD Map Parameters"), structure)

    rmsd_map_launcher.tool_window.shown = True

class RMSDMap:
    title_fmt = "%g-%g RMSD Map"

    def __init__(self, results_window, structure, start_frame, step, end_frame, use_sel, min_rmsd,
            max_rmsd, ignore_bulk, ignore_hyds, ignore_ligand, metal_ions, recolor):
        self.tool_window = tw = results_window
        self.title = self.title_fmt % (min_rmsd, max_rmsd)
        #tw.help = "help:user/commands/coordset.html#clusterdialog"
        def cleanup(lcd=self):
            inst = lcd.tool_window.tool_instance
            _md_tool_windows[inst]["RMSD map"].remove(self)
            delattr(lcd.tool_window, 'cleanup')
        tw.cleanup = cleanup
        self.session = structure.session
        self.structure = structure
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)
        '''
        table_data = []
        table_rgbas = []
        from chimerax.core.colors import distinguish_from
        # color the same trajectory consistently...
        seed = structure.num_coordsets * structure.num_residues + structure.num_atoms
        for clustering in clusterings:
            entry = TableEntry(clustering, self)
            entry.rgba = distinguish_from([(1.0,1.0,1.0,1.0)] + table_rgbas, seed=seed)
            table_rgbas.append(entry.rgba)
            table_data.append(entry)
        from chimerax.ui.widgets import ItemTable
        class ShortTable(ItemTable):
            def sizeHint(self):
                sh = super().sizeHint()
                h = sh.height()
                if h > 500:
                    sh.setHeight(sh.height() // 2)
                return sh
        self.table = table = ShortTable()
        # Putting color first makes the rows about twice as high as needed (and column titles bold!)
        members_col = table.add_column("Members", "num_frames")
        table.add_column("Color", "rgba8", format=table.COL_FORMAT_OPAQUE_COLOR, title_display=False)
        table.add_column("Representative Frame", "representative")
        table.data = table_data
        table.launch()
        table.sort_by(members_col, table.SORT_DESCENDING)
        table.selection_changed.connect(self._update_scene)
        layout.addWidget(table, alignment=Qt.AlignHCenter, stretch=1)

        self.scene = QGraphicsScene()
        class ResizingView(QGraphicsView):
            def resizeEvent(self, event, *,
                    _height=self.scene_pixel_height, _width=self.scene_aspect*self.scene_pixel_height):
                super().resizeEvent(event)
                self.fitInView(0.0, 0.0, _width, _height)
        self.view = ResizingView(self.scene)
        if clusterings:
            # not a session restore
            self._setup_scene()
            self._update_indicator()
        layout.addWidget(self.view)
        '''

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Save | qbbox.Close | qbbox.Help)
        bbox.rejected.connect(tw.destroy)
        #bbox.accepted.connect(self._show_save_clustering_dialog)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.session:
            run(ses, "help " + self.tool_window.help))
        # Setting buttons' default and autoDefault properties to False doesn't seem to actually
        # do anything on Mac, so use this horrible kludge
        b = bbox.addButton("", qbbox.ActionRole)
        b.setDefault(True)
        b.hide()
        # Put the buttons below the status bar
        tw.ui_area.parent().layout().addWidget(bbox)

        tw.manage(None)

    '''
    def restore_session_info(self, session_info):
        entries = []
        for entry_info in session_info['table_data']:
            entry = TableEntry(entry_info['clustering'], self)
            entry.rgba = entry_info['rgba']
            entries.append(entry)
        self.table.data = entries
        # do these two calls before restoring the table state, since that might change the selected row
        # and cause a redraw
        self._setup_scene()
        self._update_indicator()
        self.table.process_session_info(session_info['table_state'])

    def session_info(self):
        return {
            'structure': self.structure,
            'table_data': [entry.session_info() for entry in self.table.data],
            'table_state': self.table.session_info(),
        }

    def _show_save_clustering_dialog(self):
        from Qt.QtWidgets import QFileDialog
        fname = QFileDialog.getSaveFileName(self.tool_window.ui_area, "Save Clustering Information")[0]
        if fname:
            from chimerax.io import open_output
            with open_output(fname, encoding='utf-8') as f:
                print("# one cluster per line; first frame on each line is representative", file=f)
                for entry in self.table.sorted_data:
                    print(" ".join([str(entry.clustering.representative)] + [str(f)
                        for f in entry.clustering.frames if f != entry.clustering.representative]), file=f)

    def _setup_scene(self):
        # Have to allow for the fact that the clustering may not involve all frames of the trajectory
        scene_width = self.scene_pixel_height * self.scene_aspect
        scene_height = self.scene_pixel_height
        self.scene.setSceneRect(0.0, 0.0, scene_width, scene_height)
        fns = []
        for row in self.table.data:
            fns.extend(row.clustering.frames)
        fns.sort()
        self.fn_index = fn_index = { fn: i for i, fn in enumerate(fns) }
        self.index_fn = { i: fn for fn, i in fn_index.items() }
        self.unit_x = unit_x = scene_width / len(fns)
        to_x = lambda fn: unit_x * fn_index[fn]
        pen = QPen(Qt.NoPen)
        for row in self.table.data:
            row.rects = rects = []
            first_fn = last_fn = None
            brush = QBrush(QColor(*row.rgba8))
            for fn in row.clustering.frames:
                if first_fn is None:
                    first_fn = last_fn = fn
                elif fn == last_fn + 1:
                    last_fn = fn
                else:
                    rects.append(self.scene.addRect(to_x(first_fn), scene_height / 2.0,
                        to_x(last_fn) - to_x(first_fn) + unit_x, scene_height, brush=brush, pen=pen))
                    first_fn = last_fn = fn
            if first_fn is not None:
                rects.append(self.scene.addRect(to_x(first_fn), scene_height / 2.0,
                    to_x(fn) - to_x(first_fn) + unit_x, scene_height, brush=brush, pen=pen))

        self.scene_text = self.scene.addSimpleText("Choose in above table to show cluster")
        text_rect = self.scene_text.boundingRect()
        cx, cy = text_rect.x() + text_rect.width()/2, text_rect.y() + text_rect.height()/2
        self.scene_text.moveBy(scene_width/2 - cx, scene_height/4 - cy)

        self.indicator = self.scene.addPolygon(QPolygonF([QPointF(*args) for args in [
            (0.0, 0.0), (11.5, 0.0), (5.75, 10.0)
            ]]))
        self.indicator.setZValue(1.0)

        self.view.setMouseTracking(True)
        self.scene.mouseMoveEvent = self._mouse_move_event
        self.scene.mousePressEvent = self._mouse_press_event

    def _mouse_move_event(self, event):
        scene_x = event.scenePos().x()
        scene_width = self.scene_pixel_height * self.scene_aspect
        import math
        index = min(max(0, math.floor(scene_x / self.unit_x)), len(self.index_fn)-1)
        fn = self.index_fn[index]
        self.tool_window.status("Frame %d" % fn)
        if event.buttons() == Qt.LeftButton:
           self.structure.active_coordset_id = fn

    def _mouse_press_event(self, event):
        if event.buttons() & Qt.LeftButton == 0:
            return
        scene_x = event.scenePos().x()
        scene_width = self.scene_pixel_height * self.scene_aspect
        import math
        index = min(max(0, math.floor(scene_x / self.unit_x)), len(self.index_fn)-1)
        fn = self.index_fn[index]
        self.tool_window.status("Frame %d" % fn)
        self.structure.active_coordset_id = fn

    def _update_indicator(self, *args):
        fn = self.structure.active_coordset_id
        if fn not in self.fn_index:
            self.indicator.hide()
            return
        self.indicator.show()
        fn_x = self.unit_x * (self.fn_index[fn] + 0.5)
        self.indicator.moveBy(fn_x - (self.indicator.pos().x() + 5.75), 0.0)

    def _update_scene(self, *args):
        sel_rows = set(self.table.selected)
        if sel_rows:
            self.scene_text.hide()
        else:
            self.scene_text.show()
        scene_height = self.scene_pixel_height
        for row in self.table.data:
            if row in sel_rows:
                y = 0.0
                height = scene_height
            else:
                y = scene_height / 2.0
                height = scene_height / 2.0
            for rect_item in row.rects:
                rect = rect_item.rect()
                rect_item.setRect(rect.x(), y, rect.width(), height)
        if len(sel_rows) == 1:
           self.structure.active_coordset_id = sel_rows.pop().representative

def show_cluster_results(main_tool_window, structure, clusterings):
    inst = main_tool_window.tool_instance
    inst_window_info = _md_tool_windows.setdefault(inst, {})
    results_dialogs = inst_window_info.setdefault("cluster results", [])
    results_dialogs.append(
        ClusterResults(main_tool_window.create_child_window("Clustering Results", statusbar=True),
            structure, clusterings))

def cluster_dialog_session_info(main_tool_window):
    inst = main_tool_window.tool_instance
    inst_windows = _md_tool_windows.get(inst, {})
    try:
        clusterings = inst_windows["cluster results"]
    except KeyError:
        return None
    return [clustering.session_info() for clustering in clusterings]

def restore_cluster_info(main_tool_window, info):
    inst = main_tool_window.tool_instance
    inst_windows = _md_tool_windows.setdefault(inst, {})
    try:
        results_dialogs = inst_windows["cluster results"]
    except KeyError:
        results_dialogs = inst_windows["cluster results"] = []
    for session_info in info:
        results = ClusterResults(main_tool_window.create_child_window("Clustering Results", statusbar=True),
            session_info.pop('structure'), [])
        results.restore_session_info(session_info)
        results_dialogs.append(results)
    '''
