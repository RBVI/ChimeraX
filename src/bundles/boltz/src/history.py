# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Panel for listing submitted Boltz predictions and opening structures.
#
from chimerax.core.tools import ToolInstance
class BoltzHistoryPanel(ToolInstance):
    help = 'help:user/tools/boltz.html'

    def __init__(self, session, tool_name = 'Boltz History', predictions_directory = None):

        self.results = None
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self.fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Predictions table
        self._predictions_table = pt = self._create_predictions_table(parent, predictions_directory)
        layout.addWidget(pt)
        
        # Options panel
        options = self._create_options_pane(parent)
        layout.addWidget(options)

        # Buttons
        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Open structures', self._open_structures),
                         ('Options', self._show_or_hide_options),
                         ('Help', self._show_help)],
                        spacing = 2)
        bf.setContentsMargins(0,5,0,0)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")
        
    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Boltz History', create=create)

    # ---------------------------------------------------------------------------
    #
    def _create_predictions_table(self, parent, predictions_directory = None):
        if predictions_directory is None:
            predictions_directory = self._default_predictions_directory()
        frame = BoltzPredictionsTable(parent, predictions_directory)
        return frame
    
    # ---------------------------------------------------------------------------
    #
    def _create_options_pane(self, parent, trim = True, alignment_cutoff_distance = 2.0):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)

        from chimerax.ui.widgets import EntriesRow

        # Results directory
        rd = EntriesRow(f, 'Predictions directory', '',
                        ('Browse', self._choose_predictions_directory))
        self._predictions_directory = dir = rd.values[0]
        dir.pixel_width = 250
        dir.value = self._default_predictions_directory()
        dir.return_pressed.connect(self._directory_changed)

        aos = EntriesRow(f, True, 'Align opened structures')
        self._align = aos.values[0]
        
        return p

    # ---------------------------------------------------------------------------
    #
    def _directory_changed(self):
        self._predictions_table._update(self._predictions_directory.value)

    # ---------------------------------------------------------------------------
    #
    def _choose_predictions_directory(self):
        from .boltz_gui import _existing_directory
        dir = _existing_directory(self._predictions_directory.value)
        if not dir:
            dir = _existing_directory(self._default_predictions_directory())
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(parent,
                                                caption = f'Boltz predictions directory',
                                                directory = dir,
                                                options = QFileDialog.Option.ShowDirsOnly)
        if path:
            self._predictions_directory.value = path
            self._directory_changed()
        
    # ---------------------------------------------------------------------------
    #
    def _default_predictions_directory(self):
        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)
        dir_pattern = settings.boltz_results_location
        from os.path import dirname, expanduser
        dir = dirname(dir_pattern)
        dir = expanduser(dir)
        return dir

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()
            
    # ---------------------------------------------------------------------------
    #
    def _open_structures(self):
        pdirs = self._predictions_table.selected		# PredictionDirectory instances
        for pdir in pdirs:
            pdir.open_structures(self.session, align = self._align.value)
        if len(pdirs) == 0:
            msg = 'Click lines in the predictions table and then press Open.'
            self.session.logger.error(msg)

    # ---------------------------------------------------------------------------
    #
    @property
    def _selected_rows(self):
        rt = self._results_table
        return rt.selected if rt else []

    # ---------------------------------------------------------------------------
    #
    def fill_context_menu(self, menu, x, y):
        if self._results_table:
            from Qt.QtGui import QAction
            act = QAction("Save CSV or TSV File...", parent=menu)
            act.triggered.connect(lambda *args, tab=self._predictions_table: tab.write_values())
            menu.addAction(act)

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)
    
    # ---------------------------------------------------------------------------
    # Session saving.
    #
    @property
    def SESSION_SAVE(self):
        return len(self._predictions_table.data) > 0

    def take_snapshot(self, session, flags):
        data = {'predictions_directory': self._predictions_table.predictions_directory,
                'align': self._align.value,
                'version': '1'}
        return data

    # ---------------------------------------------------------------------------
    # Session restore
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        print (data)
        bpp = boltz_predictions_panel(session, create = True)
        pdir = data['predictions_directory']
        bpp._predictions_directory.value = pdir
        if pdir != bpp._default_predictions_directory():
            bpp._predictions_table._update(pdir)
        bpp._align.value = data['align']
        return bpp

# -----------------------------------------------------------------------------
#
from chimerax.ui.widgets import ItemTable
class BoltzPredictionsTable(ItemTable):
    def __init__(self, parent, predictions_directory):
        self.predictions_directory = predictions_directory
        ItemTable.__init__(self, parent = parent)
        self.add_column('Name', 'name')
        col_structures = self.add_column('Structures', 'structure_count', format = '%d')
        col_runtime = self.add_column('Time (sec)', 'run_time', format = '%.0f')
        col_status = self.add_column('Status', 'status')
        col_date = self.add_column('Date', 'submit_date')
#        self.add_column('Description', 'description', justification = 'left')
     
        self.data = self._prediction_rows(predictions_directory)
        self.launch()
        self.sort_by(col_date, self.SORT_DESCENDING)
#        self.setColumnWidth(self.columns.index(col_runtime), 120)
        self.setAutoScroll(False)  # Otherwise click on item scrolls horizontally
        from Qt.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)  # Don't resize whole panel width

    def _prediction_rows(self, predictions_directory):
        from os.path import join
        from os import listdir
        subdirs = [join(predictions_directory, dir) for dir in listdir(predictions_directory)]
        rows = [PredictionDirectory(dir) for dir in subdirs if _is_prediction_directory(dir)]
        return rows

    def _update(self, predictions_directory):
        from os.path import exists
        if not exists(predictions_directory):
            from chimerax.core.errors import UserError
            raise UserError(f'Boltz predictions directory does not exist: "{predictions_directory}"')
        self.predictions_directory = predictions_directory
        self.data = self._prediction_rows(predictions_directory)
        
def _is_prediction_directory(directory):
        from os.path import join, exists
        return exists(join(directory, 'command'))
    
# -----------------------------------------------------------------------------
#
class PredictionDirectory:
    def __init__(self, directory):
        self.directory = directory
        from os.path import basename, join, getctime, exists
        name = basename(directory)
        if name.startswith('boltz_'):
            name = name.removeprefix('boltz_')
        self.name = name
        submit_time = getctime(join(directory, 'command'))
        self.submit_date = _seconds_to_date(submit_time)
        struct_files = self._structure_files()
        self.structure_count = len(struct_files)
        stdout_path = join(directory, 'stdout')
        have_stdout = exists(stdout_path)
        done_time = getctime(stdout_path) if have_stdout else submit_time
        self.run_time = done_time - submit_time  # Seconds
        self.status = 'done' if struct_files else ('failed' if have_stdout else '')

    def _structure_files(self):
        struct_files = []
        from os import listdir
        rdirs = [rdir for rdir in listdir(self.directory) if rdir.startswith('boltz_results_')]
        from os.path import join, exists
        for rdir in rdirs:
            pdir = join(self.directory, rdir, 'predictions')
            if exists(pdir):
                for pname in listdir(pdir):
                    sdir = join(pdir, pname)
                    spaths = [join(sdir, file) for file in listdir(sdir) if file.endswith('.cif')]
                    struct_files.extend(spaths)
        return struct_files
        
    def open_structures(self, session, align = True):
        models = []
        paths = self._structure_files()
        for path in paths:
            structures, status = session.open_command.open_data(path, log_info = False)
            models.extend(structures)

        from chimerax.core.commands import log_equivalent_command, quote_path_if_necessary
        paths = [quote_path_if_necessary(path) for path in paths]
        log_equivalent_command(session, f'open {" ".join(paths)}')
        session.models.add(models)

        if align and len(models) > 1:
            # Align models to first model
            spec = models[0].atomspec
            from chimerax.core.commands import run, concise_model_spec
            specs = concise_model_spec(session, models[1:])
            run(session, f'matchmaker {specs} to {spec} log false')
                
        return models

# -----------------------------------------------------------------------------
#
def _seconds_to_date(seconds_from_epoch):        
    from datetime import datetime
    dt_object = datetime.fromtimestamp(seconds_from_epoch)
    date_string = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return date_string

# -----------------------------------------------------------------------------
#
def show_predictions_panel(session):
    bpp = boltz_predictions_panel(session, create = True)
    bpp.display(True)
    return bpp

# -----------------------------------------------------------------------------
#
def boltz_predictions_panel(session, create = False):
    return BoltzHistoryPanel.get_singleton(session, create=create)
