# vim: set expandtab ts=4 sw=4:

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

from chimerax.core.tools import ToolInstance
class ProfileGridsTool(ToolInstance):
    """ Viewer displays a multiple sequence alignment as a grid/table """

    help = "help:user/tools/profilegrid.html"

    def __init__(self, session, tool_name, alignment=None):
        """ if 'alignment' is None, then we are being restored from a session and
            _finalize_init will be called later.
        """

        ToolInstance.__init__(self, session, tool_name)
        if alignment is None:
            return
        self._finalize_init(alignment)

    def _finalize_init(self, alignment, *, session_data=None):
        self.alignment = alignment
        from . import subcommand_name
        alignment.attach_viewer(self, subcommand_name=subcommand_name)
        from . import settings
        self.settings = settings.init(self.session)
        from chimerax.core.utils import titleize
        self.display_name = titleize(self.alignment.description) + " [ID: %s]" % self.alignment.ident
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=True, statusbar=True)
        self.tool_window._dock_widget.setMouseTracking(True)
        self.tool_window.fill_context_menu = self.fill_context_menu
        self.status = self.tool_window.status
        parent = self.tool_window.ui_area
        parent.setMouseTracking(True)
        from .grid_canvas import GridCanvas
        if session_data is None:
            grid_data, weights = self.compute_grid(alignment.seqs)
        else:
            grid_data, weights = session_data
        self.grid_canvas = GridCanvas(parent, self, self.alignment, grid_data, weights)
        #self._seq_rename_handlers = {}
        #for seq in self.alignment.seqs:
        #    self._seq_rename_handlers[seq] = seq.triggers.add_handler("rename",
        #        self.region_browser._seq_renamed_cb)

        from Qt.QtCore import Qt
        self.tool_window.manage(None, allowed_areas=Qt.DockWidgetArea.AllDockWidgetAreas)

    def alignment_notification(self, note_name, note_data):
        alignment = self.alignment
        if note_name == alignment.NOTE_DESTROYED:
            self.delete()
            return
        '''
        if note_name == alignment.NOTE_MOD_ASSOC:
            if hasattr(self, 'associations_tool'):
                self.associations_tool._assoc_mod(note_data)
        elif note_name == alignment.NOTE_PRE_DEL_SEQS:
            self.region_browser._pre_remove_lines(note_data)
            for seq in note_data:
                if seq in self._feature_browsers:
                    self._feature_browsers[seq].tool_window.destroy()
                    del self._feature_browsers[seq]
        elif note_name == alignment.NOTE_DESTROYED:
            self.delete()
        elif note_name == alignment.NOTE_COMMAND:
            from .cmd import run
            run(self.session, self, note_data)
        '''

        self.grid_canvas.alignment_notification(note_name, note_data)

    def compute_grid(self, seqs):
        import os
        num_cpus = os.cpu_count()
        if num_cpus is None:
            num_cpus = 1
        from ._profile_grids import compute_profile
        weights = [getattr(seq, 'weight', 1.0) for seq in seqs]
        grid_data = compute_profile([seq.cpp_pointer for seq in seqs], weights, num_cpus)
        # the returned grid data is (num positions x num symbols), which is the transpose of what we
        # display, so to reduce confusion in the code, transpose it
        import numpy
        return numpy.transpose(grid_data), weights

    def delete(self):
        self.grid_canvas.destroy()
        self.alignment.detach_viewer(self)
        ToolInstance.delete(self)

    def fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction
        choose_seq_menu = menu.addMenu("Choose Cells For Sequence")
        choose_seq_menu.aboutToShow.connect(lambda *, s=self, m=choose_seq_menu:
            s._fill_choose_menu(m, "", s.alignment.seqs))
        cell_menu = menu.addMenu("Chosen Cells")
        action = QAction("Change in Prevalence...", cell_menu)
        action.triggered.connect(lambda *args, f=self.grid_canvas.prevalence_from_cells: f())
        cell_menu.addAction(action)
        action = QAction("List Sequence Names", cell_menu)
        action.triggered.connect(lambda *args, f=self.grid_canvas.list_from_cells: f())
        cell_menu.addAction(action)
        alignment_menu = cell_menu.addMenu("New Alignment")
        viewers = self.session.alignments.registered_viewers("alignment")
        viewers.sort()
        for viewer in viewers:
            alignment_menu.addAction(viewer.title())
        alignment_menu.triggered.connect(
            lambda action, f=self.grid_canvas.alignment_from_cells: f(action.text().lower()))
        cell_menu.setEnabled(bool(self.grid_canvas.chosen_cells))

        self.alignment.add_headers_menu_entry(menu)

        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(self.show_settings)
        menu.addAction(settings_action)

    def show_settings(self):
        if not hasattr(self, "settings_tool"):
            from .settings_tool import SettingsTool
            self.settings_tool = SettingsTool(self,
                self.tool_window.create_child_window("Profile Grid Settings", close_destroys=False))
            self.settings_tool.tool_window.manage(None)
        self.settings_tool.tool_window.shown = True

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(data['alignment'], session_data=(data['grid data'], data['weights']))
        inst.grid_canvas.restore_state(data['grid canvas'])
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'alignment': self.alignment,
            'grid data': self.grid_canvas.grid_data,
            'weights': self.grid_canvas.weights,
            'grid canvas': self.grid_canvas.state()
        }
        return data

    def _fill_choose_menu(self, menu, prefix, seqs):
        menu.clear()
        target_menu_size = 10
        if len(seqs) <= 1.5 * target_menu_size:
            for seq in sorted(seqs, key=lambda seq: seq.name.lower()):
                action = menu.addAction(seq.name)
                action.triggered.connect(lambda *args, gc=self.grid_canvas, seq=seq: gc.choose_from_seq(seq))
            return
        prev_boxes = None
        num_chars = 1
        prefix_len = len(prefix)
        while True:
            boxes = {}
            for seq in seqs:
                boxes.setdefault(seq.name[prefix_len:prefix_len+num_chars], []).append(seq)
            if prev_boxes is None:
                prev_boxes = boxes
                num_chars += 1
                continue
            if abs(len(prev_boxes) - target_menu_size) < abs(len(boxes) - target_menu_size):
                best_boxes = prev_boxes
                break
            prev_boxes = boxes
            num_chars += 1

        for addition in sorted(list(best_boxes.keys()), key=lambda add: add.lower()):
            box = best_boxes[addition]
            if len(box) == 1:
                seq = box[0]
                action = menu.addAction(seq.name)
                action.triggered.connect(lambda *args, gc=self.grid_canvas, seq=seq:
                    gc.choose_from_seq(seq))
            else:
                submenu = menu.addMenu(prefix + addition + '...')
                submenu.aboutToShow.connect(lambda *, s=self, m=submenu, prefix=prefix+addition, seqs=box:
                    s._fill_choose_menu(m, prefix, seqs))
