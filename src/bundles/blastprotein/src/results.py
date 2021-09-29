# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to app copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from string import capwords
from typing import Dict, List

from Qt.QtCore import Qt, QThread, Signal, Slot

from Qt.QtWidgets import QWidget, QVBoxLayout, QAbstractItemView
from Qt.QtWidgets import QPushButton, QAction, QLabel

from chimerax.atomic import Sequence
from chimerax.alphafold.match import _log_alphafold_sequence_info
from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.core.settings import Settings
from chimerax.core.tools import ToolInstance
from chimerax.ui.gui import MainToolWindow

from .databases import AvailableDBsDict
from .datatypes import BlastParams, SeqId
from .widgets import LabelledProgressBar, BlastResultsTable, BlastResultsRow

_settings = None

_instance_map = {} # Map of blastprotein results names to results instances

def find_match(instance_name):
    if instance_name is None:
        if len(_instance_map) == 1:
            return instance_map.values()[0]
        if len(_instance_map) > 1:
            raise UserError("no name specified with multiple active blastprotein instances")
        else:
            raise UserError("no active blastprotein instance")
    try:
        return _instance_map[instance_name]
    except KeyError:
        raise UserError("no blastprotein instance named \"%s\"" % instance_name)

class BlastProteinResults(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:/user/tools/blastprotein.html"

    def __init__(self, session, tool_name, **kw):
        self._instance_name = tool_name
        _instance_map[self._instance_name] = self
        self.display_name = "Blast Protein Results [name: %s]" % self._instance_name
        # TODO When and how does this need to be incremented?
        self._viewer_index = 1
        self.job = kw.pop('job', None)
        self.params: BlastParams = kw.pop('params', None)
        self._hits = kw.pop('hits', None)
        self._sequences: Dict[int, SeqId] = kw.pop('sequences', None)
        self._table_session_data = kw.pop('table_session_data', None)
        self._from_restore = kw.pop('from_restore', False)
        super().__init__(session, self.display_name, **kw)
        self._build_ui()

    def _build_ui(self):
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        global _settings
        if _settings is None:
            class _BlastProteinResultsSettings(Settings):
                EXPLICIT_SAVE = { BlastResultsTable.DEFAULT_SETTINGS_ATTR: {} }
            _settings = _BlastProteinResultsSettings(self.session, "Blastprotein")
        self.main_layout = QVBoxLayout()
        self.control_widget = QWidget(parent)
        #self.align_button = QPushButton("Load and Align Selection", parent)

        param_str = ", ".join(
            [": ".join([str(label), str(value)]) for label, value in self.params._asdict().items()]
        )
        self.param_report = QLabel("".join(["Query Parameters: {", param_str, "}"]), parent)
        self.control_widget.setVisible(False)

        default_cols = {key: True for key in AvailableDBsDict[self.params.database].default_cols}
        self.table = BlastResultsTable(self.control_widget, default_cols, _settings, parent)

        self.progress_bar = LabelledProgressBar(parent)

        self.main_layout.addWidget(self.param_report)
        self.main_layout.addWidget(self.table)
        self.main_layout.addWidget(self.control_widget)
        self.main_layout.addWidget(self.progress_bar)

        if not self._from_restore:
            self.worker = BlastResultsWorker(self.session, self.job)

            self.worker.parsing_results.connect(self.parsing_results)
            self.worker.standard_output.connect(self.stdout_to_report)
            self.worker.job_failed.connect(self.job_failed)

            self.worker.set_progress_maxval.connect(self._set_progress_bar_maxval)
            self.worker.finished_processing_hits.connect(self._on_finished_processing_hits)
            self.worker.processed_result.connect(self._increment_progress_bar_results)
            self.worker.waiting_for_info.connect(self._update_progress_bar_text)
            self.worker.processed_row.connect(self._increment_progress_bar_hits)
            self.worker.finished_adding_rows.connect(self._on_finished_processing_rows)
            self.worker.report_hits.connect(self._on_report_hits_signal)
            self.worker.report_sequences.connect(self._on_report_sequences_signal)
            self.worker.start()

        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.get_selection.connect(self.load)
        self.tool_window.fill_context_menu = self.fill_context_menu

        if self._from_restore:
            self._on_report_hits_signal(self._hits)

        self.tool_window.ui_area.setLayout(self.main_layout)
        self.tool_window.manage('side')


    def fill_context_menu(self, menu, x, y):
        seq_action = QAction("Load and Align Selections", menu)
        seq_view_action = QAction("Show Selections in Sequence Viewer", menu)
        seq_action.triggered.connect(lambda: self.load(self.table.selected))
        seq_view_action.triggered.connect(lambda: self._show_mav(self.table.selected))
        menu.addAction(seq_action)
        menu.addAction(seq_view_action)

    def _show_params(self, params):
        self.session.logger.info(params)

    #
    # Worker Callbacks
    #
    def parsing_results(self):
        self.session.logger.info("Parsing BLAST results.")

    def stdout_to_report(self, output):
        self.session.logger.error(output)

    def job_failed(self, error):
        raise UserError("BlastProtein failed: %s" % error)

    def _increment_progress_bar_results(self):
        self._increment_progress_bar("Results")

    def _increment_progress_bar_hits(self):
        self._increment_progress_bar("Hits")

    def _increment_progress_bar(self, itype):
        self._progress_bar_step = self.progress_bar.value + 1
        self.progress_bar.setValue(self._progress_bar_step)
        self._set_progress_bar_progress_text(itype, self._progress_bar_step)

    def _set_progress_bar_maxval(self, val):
        # We subtract one to account for the fact that the list contains the query
        # in position 0.
        self.progress_bar.setMaximum(val - 1)
        self.places = len(str(val))
        self.max_val = val - 1
        self._set_progress_bar_progress_text("Results", 0)

    def _on_finished_processing_hits(self):
        self.progress_bar.setValue(0)
        self._set_progress_bar_progress_text("Hits", 0)

    def _on_finished_processing_rows(self):
        self._unload_progress_bar()

    def _on_report_sequences_signal(self, sequences):
        self._sequences = sequences

    def _on_report_hits_signal(self, items):
        self._hits = items
        db = AvailableDBsDict[self.params.database]
        try:
            columns = list(items[0].keys())[::-1]
            columns = list(filter(lambda x: x not in db.excluded_cols, columns))
        except IndexError:
            if not self._from_restore:
                self.session.logger.warning("BlastProtein returned no results")
            self._unload_progress_bar()
        else:
            for string in columns:
                kwdict = {}
                #if string not in db.default_cols:
                #    kwdict['display'] = False
                # Decide how the title should be formatted
                kwdict['header_justification'] = 'center'
                # Format the title for display
                newstr = string
                newstr = capwords(" ".join(newstr.split('_')))
                newstr = newstr.replace('Id', 'ID')

                self.table.add_column(newstr, data_fetch=lambda x, i=string: x[i], **kwdict)
            # Convert dicts to objects (they're hashable)
            self.table.data = [BlastResultsRow(item) for item in items]
            self.table.sortByColumn(columns.index('score'), Qt.DescendingOrder)
            if self._from_restore:
                self.table.launch(session_info=self._table_session_data)
            else:
                self.table.launch()
            self.control_widget.setVisible(True)
            self._unload_progress_bar()

    def _set_progress_bar_progress_text(self, itype, curr_value):
        self._update_progress_bar_text(" ".join(["Processing", itype, '{0:>{width}}/{1:>{width}}'.format(curr_value, self.max_val, width=self.places)]))

    def _update_progress_bar_text(self, text):
        self.progress_bar.text = text

    def _unload_progress_bar(self):
        self.main_layout.removeWidget(self.progress_bar)
        self.progress_bar.deleteLater()
        self.progress_bar = None

    #
    # Code for loading (and spatially matching) a match entry
    #
    def load(self, selections: list['BlastResultsRow']) -> None:
        """Load the model from the results database.
        """
        db = AvailableDBsDict[self.params.database]
        for row in selections:
            code = row[db.fetchable_col]
            models, chain_id = db.load_model(
                self.session, code, self.params.chain
            )
            if not models:
                return
            if not self.params.chain:
                run(self.session, "select clear")
            else:
                if db.name == 'alphafold':
                    ...
                #    self._log_alphafold(models)
                else:
                    for m in models:
                        db.display_model(self.session, self.params.chain, m, chain_id)

    def _log_alphafold(self, models):
        if not self.params.chain:
            query_name = self.parser.true_name or 'query'
            query_seq = Sequence(name = query_name, characters = self.parser.query_seq)
            for m in models:
                _log_alphafold_sequence_info(m, query_seq)

    #
    # Code for displaying matches as multiple sequence alignment
    #
    def _show_mav(self, selections) -> None:
        """
        Collect the names and sequences of selected matches. All sequences
        should have the same length because they include the gaps inserted by
        the BLAST alignment.
        """
        ids = [hit['id'] for hit in selections]
        ids.insert(0,0)
        names = []
        seqs = []
        for sid in ids:
            name, seq = self._sequences[sid]
            names.append(name)
            seqs.append(seq)
        # Find columns that are gaps in all sequences and remove them.
        all_gaps = set()
        for i in range(len(seqs[0])):
            for seq in seqs:
                if seq[i].isalpha():
                    break
            else:
                all_gaps.add(i)
        if all_gaps:
            for i in range(len(seqs)):
                seq = seqs[i]
                new_seq = ''.join([seq[n] for n in range(len(seq))
                                   if n not in all_gaps])
                seqs[i] = new_seq
        # Generate multiple sequence alignment file
        # Ask sequence viewer to display alignment
        seqs = [Sequence(name=name, characters=seqs[i])
                for i, name in enumerate(names)]
        inst_name = "None"
        if self._instance_name:
            inst_name = self._instance_name
        name = "%s [%d]" % (inst_name, self._viewer_index)
        # Ensure that the next time the user launches the same command that a
        # unique index gets shown.
        self._viewer_index += 1
        self.session.alignments.new_alignment(seqs, name)


    #
    # Snapshots
    #
    @classmethod
    def from_snapshot(cls, session, data):
        """Initializer to be used when restoring ChimeraX sessions."""
        # Data from version 1 snapshots is prefixed by _, so need to add it in
        # for backwards compatibility.

        sequences_dict = {}
        # We use get with a default value of 2 because for a few weeks in August and
        # September 2021 the daily builds did not save snapshots with a version number.
        # We can remove this when sufficient time has passed.
        if data.get('version', 2) == 1:
            sequences_dict = data['_sequences']
            data['params'] = BlastParams(*[x[1] for x in data['_params']])
            data['tool_name'] = data['_instance_name'] + str(data['_viewer_index'])
            data['results'] = data['_hits']
            data['table_session'] = None
        else:
            sequences = data['sequences']
            for (key, hit_name, sequence) in sequences:
                sequences_dict[key] = SeqId(hit_name, sequence)
            data['params'] = BlastParams(*list(data['params'].values()))

        return cls(
            session, data['tool_name'], hits = data['results']
            , sequences = sequences_dict, params = data['params']
            , table_session_data = data['table_session'], from_restore=True
        )

    @classmethod
    def restore_snapshot(cls, session, data):
        return BlastProteinResults.from_snapshot(session, data)

    def take_snapshot(self, session, flags):
        data = {
            'version': 2
            , 'ToolUI': ToolInstance.take_snapshot(self, session, flags)
            , 'table_session': self.table.session_info()
            , 'params': self.params._asdict()
            , 'tool_name': self.tool_instance_name
            , 'results': self._hits
            , 'sequences': [(key
                           , self._sequences[key][0]
                           , self._sequences[key][1]
                           ) for key in self._sequences.keys()]
        }
        return data

class BlastResultsWorker(QThread):
    standard_output = Signal()
    job_failed = Signal(str)
    parsing_results = Signal()
    set_progress_maxval = Signal(object)
    processed_result = Signal()
    finished_processing_hits = Signal()
    processed_row = Signal()
    finished_adding_rows = Signal()
    waiting_for_info = Signal(str)
    report_hits = Signal(list)
    report_sequences = Signal(dict)

    def __init__(self, session, job):
        super().__init__()
        self.session = session
        self.job = job

    @Slot()
    def run(self):
        self._get_and_process_results()

    @Slot()
    def stop(self):
        pass

    def _get_and_process_results(self):
        """Fetch results from plato and process them for BlastProteinResults."""
        out = self.job.get_stdout()
        if out:
            # Originally sent to logger.error
            self.standard_output.emit("Standard output:\n" + out)
        if not self.job.exited_normally():
            err = self.job.get_stderr()
            self.job_failed.emit(err)
        else:
            self.waiting_for_info.emit("Downloading Results")
            results = self.job.get_file(self.job.RESULTS_FILENAME)
            try:
                self.parsing_results.emit()
                self.job._database.parse("query", self.job.seq, results)
            except Exception as e:
                err = self.job.get_stderr()
                self.job_failed.emit(err + str(e))
            else:
                self._ref_atomspec = self.job.atomspec
                blast_results = self.job._database
                hits = []
                self._sequences = {}
                if blast_results is not None:
                    query_match = blast_results.parser.matches[0]
                    if self._ref_atomspec:
                        name = self._ref_atomspec
                    else:
                        name = query_match.name
                    self._sequences[0] = (name, query_match.sequence)
                    match_chains = {}
                    sequence_only_hits = {}
                    self.set_progress_maxval.emit(len(blast_results.parser.matches))
                    for n, m in enumerate(blast_results.parser.matches[1:]):
                        sid = n + 1
                        hit = {"id":sid, "evalue":m.evalue, "score":m.score,
                               "description":m.description}
                        if m.match:
                            hit["name"] = m.match
                            match_chains[m.match] = hit
                        else:
                            hit["name"] = m.name
                            sequence_only_hits[m.name] = hit
                        hits.append(hit)
                        self._sequences[sid] = SeqId(hit["name"], m.sequence)
                        self.processed_result.emit()
                    # TODO: Make what this function does more explicit. It works on the
                    # hits that are in match_chain's hit dictionary, but that's not
                    # immediately clear.
                    self.waiting_for_info.emit("Postprocessing Hits")
                    blast_results.add_info(self.session, match_chains, sequence_only_hits)
                    self.finished_processing_hits.emit()
                self._hits = hits
                self.report_hits.emit(self._hits)
                self.report_sequences.emit(self._sequences)
