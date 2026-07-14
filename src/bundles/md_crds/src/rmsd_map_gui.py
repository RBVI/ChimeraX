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
        def cleanup(self=self):
            inst = self.tool_window.tool_instance
            from .gui import _remove_tool_window
            _remove_tool_window(inst, "rmsd map launcher")
            delattr(self.tool_window, 'cleanup')
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

        self.settings = settings = RMSDMapSettings(self.session, "RMSD map")
        from chimerax.ui.options import OptionsPanel, IntOption, BooleanOption, EnumOption, FloatOption
        options_panel = OptionsPanel(sorting=False, scrolled=False, contents_margins=(2,2,2,2))
        cs_ids = structure.coordset_ids
        min_cs = min(cs_ids)
        max_cs = max(cs_ids)
        self.start_opt = IntOption("Starting frame:", min_cs, None, min=min_cs, max=max_cs)
        options_panel.add_option(self.start_opt)
        from .util import default_step
        self.step_opt = IntOption("Step size:", default_step(len(cs_ids)), None, min=1, max=max_cs)
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
        self.solution_opt = BooleanOption("Ignore solvent and non-metal ions:", True, None)
        options_panel.add_option(self.solution_opt)
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
        # Put the buttons below the status bar
        tw.ui_area.parent().layout().addWidget(bbox)

        tw.manage(None)

    def launch_rmsd_map(self, *, apply=False):
        start = self.start_opt.value
        step = self.step_opt.value
        end = self.end_opt.value
        use_sel = self.sel_opt.value
        low = self.low_bound_opt.value
        high = self.high_bound_opt.value
        exclude_solution = self.solution_opt.value
        exclude_hydrogens = self.hyd_opt.value
        exclude_ligands = self.ligand_opt.value
        exclude_metals = self.metal_opt.value
        recolor = self.recolor_opt.value
        atoms = self.structure.atoms
        if use_sel and atoms.selecteds.any():
            atoms = atoms.filter(atoms.selecteds)
        from .util import analysis_atoms, analysis_frames
        atoms = analysis_atoms(atoms, exclude_solution, exclude_hydrogens, exclude_ligands, exclude_metals)
        from chimerax.ui import tool_user_error
        if not atoms:
            return tool_user_error("No atoms remain after filtering")
        frames = analysis_frames(self.structure, start, end, step)
        if not frames:
            return tool_user_error("No frames match start/step/end")

        num_frames = len(frames)
        from math import sqrt
        import numpy
        rmsds = numpy.zeros((num_frames, num_frames), float)
        structure = atoms[0].structure
        with structure.suppress_coordset_change_notifications():
            for i, fn1 in enumerate(frames):
                self.tool_window.status("Computing RMSDS for frame %d/%d" % (i+1, num_frames))
                structure.active_coordset_id = fn1
                coords1 = atoms.coords
                for j in range(i+1, num_frames):
                    structure.active_coordset_id = frames[j]
                    diff = atoms.coords - coords1
                    rmsds[i,j] = rmsds[j,i] = sqrt(numpy.sum(diff * diff) / len(atoms))
        self.tool_window.status("Computed RMSDs; showing map")

        if not apply:
            self.tool_window.destroy()
        inst = self.main_tool_window.tool_instance
        inst_window_info = _md_tool_windows.setdefault(inst, {})
        map_results = inst_window_info.setdefault("RMSD map", [])
        map_results.append(
            RMSDMap(self.session, self.main_tool_window.create_child_window("RMSD Map", statusbar=True),
                rmsds, frames, low, high, recolor, self.settings))

from chimerax.core.settings import Settings
class RMSDMapSettings(Settings):
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
            main_tool_window.create_child_window("Get RMSD Map Parameters", statusbar=True), structure)

    rmsd_map_launcher.tool_window.shown = True

class RMSDMap:
    title_fmt = "%.2g-%.2g RMSD Map"

    def __init__(self, session, results_window, rmsds, frames, min_rmsd, max_rmsd, recolor, settings):
        self.tool_window = tw = results_window
        self.session = session
        self.rmsds = rmsds
        self.frames = frames
        self.settings = settings
        if recolor:
            rmsds_1D = rmsds.flatten()
            import numpy
            sorted_rmsds = numpy.sort(rmsds_1D)
            self.min_rmsd = sorted_rmsds[round(len(sorted_rmsds)/3)]
            self.max_rmsd = sorted_rmsds[round(2*len(sorted_rmsds)/3)]
        else:
            self.min_rmsd, self.max_rmsd = min_rmsd, max_rmsd
        self.set_title()
        #tw.help = "help:user/commands/coordset.html#clusterdialog"
        def cleanup(self=self):
            inst = self.tool_window.tool_instance
            _md_tool_windows[inst]["RMSD map"].remove(self)
            for cid in self._mouse_handlers:
                self.canvas.mpl_disconnect(cid)
            self._mouse_handlers.clear()
            delattr(self.tool_window, 'cleanup')
        tw.cleanup = cleanup
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)

        from matplotlib.colors import LinearSegmentedColormap as LSColormap
        # color map dictionary so that low values are white and high values black
        cm_dict = {
            'red': ((0.0, 1.0, 1.0),
                    (1.0, 0.0, 0.0)),
            'green': ((0.0, 1.0, 1.0),
                    (1.0, 0.0, 0.0)),
            'blue': ((0.0, 1.0, 1.0),
                    (1.0, 0.0, 0.0)),
        }
        cmap = LSColormap('rmsds', cm_dict, 256)
        cmap.set_under(color='white')
        cmap.set_over(color='black')

        from matplotlib.backends.backend_qtagg import FigureCanvas
        from matplotlib.figure import Figure
        self.canvas = canvas = FigureCanvas(Figure())
        layout.addWidget(canvas, stretch=1)
        self._mouse_handlers = [
            canvas.mpl_connect('motion_notify_event', self._mouse_event),
            canvas.mpl_connect('button_press_event', self._mouse_event),
        ]
        figure = canvas.figure
        axis = figure.subplots()
        axis.tick_params(direction='out')
        from matplotlib.ticker import MaxNLocator
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.fixed_mpl_kw = {
            'cmap': cmap,
            'origin': 'lower',
            'extent': (0, len(frames), 0, len(frames)),
        }
        im = axis.imshow(rmsds, vmin = self.min_rmsd, vmax = self.max_rmsd, **self.fixed_mpl_kw)
        canvas.draw_idle()

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

    def set_title(self):
        self.tool_window.title = self.title_fmt % (self.min_rmsd, self.max_rmsd)

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
    '''

    def _mouse_event(self, event):
        from matplotlib.backend_bases import MouseButton
        if event.name == "button_press_event":
            if event.button != MouseButton.LEFT:
                return
        elif event.name == "motion_notify_event":
            if event.xdata is None or event.ydata is None:
                self.tool_window.status('')
                return
            # ensure that index at extreme right/top remains in range
            mpl_to_index = lambda data, nf=len(self.frames): min(int(data), nf-1)
            xi, yi = mpl_to_index(event.xdata), mpl_to_index(event.ydata)
            self.tool_window.status("Frames %d/%d: RMSD %.3f"
                % (self.frames[xi], self.frames[yi], self.rmsds[xi,yi]))
        else:
            raise ValueError("Unexpected Matplotlib event: %s" % event.name)
        '''
        for plot in self.plots:
            if event.canvas == plot:
                if not event.inaxes:
                    break
                cs_id = round(event.xdata)
                if cs_id != self.structure.active_coordset_id:
                    # rather than directly check if the ID is valid (there could be many coord sets)
                    # just try to set it and catch the error
                    try:
                        self.structure.active_coordset_id = cs_id
                    except IndexError:
                        # non-existent
                        pass
                break
        '''
'''
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
