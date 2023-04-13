# vim: set expandtab shiftwidth=4 softtabstop=4:

#  === UCSF ChimeraX Copyright ===
#  Copyright 2022 Regents of the University of California.
#  All rights reserved.  This software provided pursuant to a
#  license agreement containing restrictions on its disclosure,
#  duplication and use.  For details see:
#  https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
#  This notice must be embedded in or attached to all copies,
#  including partial copies, of the software or any revisions
#  or derivations thereof.
#  === UCSF ChimeraX Copyright ===
from collections import defaultdict
from string import capwords
from typing import Dict

from Qt.QtCore import Qt, QThread, Signal, Slot
from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QAbstractItemView
    , QLabel, QPushButton, QHBoxLayout
    , QCheckBox
)
from Qt.QtGui import QAction

from chimerax.atomic import Sequence
from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.core.tools import ToolInstance
from chimerax.ui.gui import MainToolWindow
from chimerax.help_viewer import show_url

from ..data_model import AvailableDBsDict, Match, parse_blast_results
from ..utils import BlastParams, SeqId, _instance_generator
from .widgets import (
    BlastResultsTable, BlastResultsRow
    , BlastProteinResultsSettings
)

_settings = None

_instance_map = {} # Map of blastprotein results names to results instances

def find_match(instance_name):
    if instance_name is None:
        if len(_instance_map) == 1:
            return _instance_map.values()[0]
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
    help = "help:user/tools/blastprotein.html#results"

    def __init__(self, session, tool_name, **kw):
        display_name = "Blast Protein Results [name: %s]" % tool_name
        super().__init__(session, tool_name)
        self.display_name = display_name
        self._instance_name = tool_name
        _instance_map[self._instance_name] = self
        self._viewer_index = 1
        self._from_restore = kw.pop('from_restore', False)
        self.params: BlastParams = kw.pop('params', None)

        # from_job
        self.job = kw.pop('job', None)

        # from_pull
        self._sequence = kw.pop('sequence', None)
        self._results = kw.pop('results', None)

        # from_snapshot
        self._hits = kw.pop('hits', None)
        self._best_hits = None
        self._sequences: Dict[int, Match] = kw.pop('sequences', None)
        self._table_session_data = kw.pop('table_session_data', None)
        self.tool_window = None

        self._build_ui()

    @classmethod
    def from_job(cls, session, tool_name, params, job):
        return cls(
            session, tool_name, from_restore = False, params=params
            , job=job
        )

    @classmethod
    def from_pull(cls, session, tool_name, params, sequence, results):
        return cls(
            session, tool_name, from_restore = False, params=params
            , sequence=sequence, results=results
        )

    @classmethod
    def from_snapshot(cls, session, data):
        """Initializer to be used when restoring ChimeraX sessions."""
        sequences_dict = {}
        version = data.get('version', 3)
        if version == 1:
            # Data from version 1 snapshots is prefixed by _, so need to add it in
            # for backwards compatibility.
            sequences_dict = data['_sequences']
            data['params'] = BlastParams(*[x[1] for x in data['_params']])
            data['tool_name'] = data['_instance_name'] + str(data['_viewer_index'])
            data['results'] = data['_hits']
            data['table_session'] = None
        # The version 2 getter exists because for a few weeks in August and
        # September 2021 the daily builds did not save snapshots with a version number.
        # We can remove this when sufficient time has passed.
        elif version == 2:
            sequences = data['sequences']
            for (key, hit_name, sequence) in sequences:
                sequences_dict[key] = SeqId(hit_name, sequence)
            data['params'] = BlastParams(*list(data['params'].values()))
        else:
            sequences = data['sequences']
            for (key, hit_name, saved_seq_dict) in sequences:
                keys = list(saved_seq_dict.keys())
                # Fix up keys that don't match current initializer
                keys[1] = 'match_id'
                keys[2] = 'desc'
                keys[3] = 'score'
                values = list(saved_seq_dict.values())
                sequences_dict[key] = (hit_name, Match(**dict(zip(keys, values))))
            data['params'] = BlastParams(*list(data['params'].values()))
        return cls(
            session, data['tool_name'], from_restore=True, params = data['params']
            , hits = data['results']
            , sequences = sequences_dict
            , table_session_data = data['table_session']
        )

    #
    # Snapshots
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        # Increment the counter on the Blast name generator each time a blast
        # instance with a default name is restored, so that subsequent blast
        # jobs have the correct default name if needed
        if data['tool_name'].startswith('bp'):
            next(_instance_generator)
        return BlastProteinResults.from_snapshot(session, data)

    def take_snapshot(self, session, flags):
        data = {
            'version': 3
            , 'ToolUI': ToolInstance.take_snapshot(self, session, flags)
            , 'table_session': self.table.session_info()
            , 'params': self.params._asdict()
            , 'tool_name': self._instance_name
            , 'results': self._hits
            , 'sequences': [(key
                             , self._sequences[key][0]
                             , vars(self._sequences[key][1])
                             ) for key in self._sequences.keys()]
        }
        return data

    #
    # Utilities
    #
    def _format_column_title(self, title: str):
        if title == 'e-value':
            return 'E-Value'
        if title == 'uniprot_id':
            return 'UniProt ID'
        if title == 'mgnify_id':
            return 'MGnify ID'
        new_title = capwords(" ".join(title.split('_')))
        new_title = new_title.replace('Id', 'ID')
        return new_title

    def _make_settings_dict(self, db):
        defaults = {
            self._format_column_title(title): True for title in db.default_cols
        }
        return defaults

    def _format_param_str(self):
        labels = list(self.params._asdict().keys())
        values = list(self.params._asdict().values())
        try:
            model_no = int(float(values[0].split('/')[0][1:]))
        except AttributeError: # AlphaFold can be run without a model
            model_no = None
        try:
            chain = ''.join(['/', values[0].split('/')[1]])
        except AttributeError: # There won't be a selected chain either
            chain = None
        if model_no:
            try:
                model_formatted = ''.join([self.job.model_name, chain])
            except (KeyError, AttributeError):
                model_formatted = None
        else:
            model_formatted = None
        values[0] = model_formatted
        param_str = ", ".join(
            [": ".join([str(label), str(value)]) for label, value in zip(labels, values)]
        )
        return param_str

    #
    # UI
    #
    def _build_ui(self):
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        global _settings
        if _settings is None:
            _settings = BlastProteinResultsSettings(self.session, "Blastprotein")
        self.main_layout = QVBoxLayout()
        self.control_widget = QWidget(parent)
        self.buttons_label = QLabel("For chosen entries:", parent=parent)
        self.load_buttons_widget = QWidget(parent)
        self.load_button_container = QHBoxLayout()
        self.load_button_container.addWidget(self.buttons_label)
        self.load_button_container.addStretch()

        param_str = self._format_param_str()
        self.param_report = QLabel(" ".join(["Query:", param_str]), parent)
        self.control_widget.setVisible(False)

        default_cols = self._make_settings_dict(AvailableDBsDict[self.params.database])
        self.table = BlastResultsTable(self.control_widget, default_cols, _settings, parent)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.get_selection.connect(self.load)
        self.tool_window.fill_context_menu = self.fill_context_menu

        self.load_button = QPushButton("Load Structures", parent=self.load_buttons_widget)
        self.seq_view_button = QPushButton("Show Sequence Alignment", parent=self.load_buttons_widget)
        self.load_db_button = QPushButton("Open Database Webpage", parent=self.load_buttons_widget)
        self.load_button_container.addWidget(self.load_button)
        self.load_button_container.addWidget(self.seq_view_button)
        self.load_button_container.addWidget(self.load_db_button)
        self.load_button.clicked.connect(lambda: self.load(self.table.selected))
        self.seq_view_button.clicked.connect(lambda: self._show_mav(self.table.selected))
        self.load_db_button.clicked.connect(lambda: self.load_sequence(self.table.selected))

        self.main_layout.addWidget(self.param_report)
        self.main_layout.addWidget(self.table)
        self.main_layout.addWidget(self.control_widget)
        self.load_buttons_widget.setLayout(self.load_button_container)
        self.show_best_matching_container = QWidget(parent=parent)
        self.show_best_matching_layout = QHBoxLayout()
        self.only_best_matching = QCheckBox("List only best chain per PDB", parent=parent)
        self.only_best_matching.stateChanged.connect(self._on_best_matching_state_changed)
        self.show_best_matching_layout.addStretch()
        self.show_best_matching_layout.addWidget(self.only_best_matching)
        self.show_best_matching_container.setLayout(self.show_best_matching_layout)
        self.show_best_matching_container.setVisible(False)
        self.main_layout.addWidget(self.show_best_matching_container)
        self.main_layout.addWidget(self.load_buttons_widget)

        for layout in [self.main_layout, self.show_best_matching_layout]:
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(2)

        if self._from_restore:
            self._on_report_hits_signal(self._hits)
        else:
            if(self.job):
                results = None
                sequence = self.job.seq
                atomspec = self.job.atomspec
            else:
                results = self._results
                sequence = self._sequence
                atomspec = self.params.chain

            self.worker = BlastResultsWorker(
                self.session, self.job, self.params.database
                , results, sequence, atomspec
            )
            self.connect_worker_callbacks(self.worker)
            self.worker.start()

        self.tool_window.ui_area.closeEvent = self.closeEvent
        self.tool_window.ui_area.setLayout(self.main_layout)

    def _on_best_matching_state_changed(self):
        state = self.only_best_matching.checkState()
        if state.name == "Checked":
            if self._best_hits is None:
                chains = defaultdict(str)
                for hit in self._hits:
                    chain, homotetramer = hit["name"].split('_')
                    if chain not in chains:
                        chains[chain] = hit
                    else:
                        old_homotetramer = chains[chain]["name"].split('_')[1]
                        best_homotetramer = sorted([homotetramer, old_homotetramer])[0]
                        if best_homotetramer == homotetramer:
                            chains[chain] = hit
                self._best_hits = list(chains.values())
                self.table.data = [BlastResultsRow(item) for item in self._best_hits]
            else:
                self.table.data = [BlastResultsRow(item) for item in self._best_hits]
        else:
            self.table.data = [BlastResultsRow(item) for item in self._hits]

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.terminate()
        self.tool_window.ui_area.close()
        self.tool_window = None

    def fill_context_menu(self, menu, x, y):
        seq_action = QAction("Load Structures", menu)
        seq_view_action = QAction("Show Sequence Alignment", menu)
        load_from_db_action = QAction("Open Database Webpage", menu)
        seq_action.triggered.connect(lambda: self.load(self.table.selected))
        seq_view_action.triggered.connect(lambda: self._show_mav(self.table.selected))
        load_from_db_action.triggered.connect(lambda: self.load_sequence(self.table.selected))
        menu.addAction(seq_action)
        menu.addAction(seq_view_action)
        menu.addAction(load_from_db_action)

    #
    # Worker Callbacks
    #
    def connect_worker_callbacks(self, worker):
        worker.job_failed.connect(self.job_failed)
        worker.parse_failed.connect(self.parse_failed)
        worker.parsing_results.connect(self.parsing_results)
        worker.report_hits.connect(self._on_report_hits_signal)
        worker.report_sequences.connect(self._on_report_sequences_signal)

    def job_failed(self, error):
        self.session.logger.error("BlastProtein failed: %s" % error)

    def parse_failed(self, error):
        self.session.logger.error("Parsing BlastProtein results failed: %s" % error)

    def parsing_results(self):
        self.session.logger.status("Parsing BLAST results.")

    def _on_report_sequences_signal(self, sequences):
        self._sequences = sequences

    def _on_report_hits_signal(self, items):
        if items:
            self.tool_window.manage('side')
            try:
                items = sorted(items, key = lambda i: i['e-value'])
                for index, item in enumerate(items):
                    item['hit_#'] = index + 1
                self._hits = items
                db = AvailableDBsDict[self.params.database]
                try:
                    # Compute the set of unique column names
                    columns = set()
                    for item in items:
                        columns.update(list(item.keys()))
                    # Sort the columns so that defaults come first
                    columns = list(filter(lambda x: x not in db.excluded_cols, columns))
                    nondefault_cols = list(filter(lambda x: x not in db.default_cols, columns))
                    columns = list(db.default_cols)
                    columns.extend(nondefault_cols)
                except IndexError:
                    if not self._from_restore:
                        self.session.logger.warning("BlastProtein returned no results")
                else:
                    # Convert dicts to objects (they're hashable)
                    self.table.data = [BlastResultsRow(item) for item in items]
                    for string in columns:
                        title = self._format_column_title(string)
                        self.table.add_column(title, data_fetch=lambda x, i=string: x[i])
                    self.table.sortByColumn(columns.index('e-value'), Qt.AscendingOrder)
                    if self._from_restore:
                        self.table.launch(session_info=self._table_session_data, suppress_resize=True)
                    else:
                        self.table.launch(suppress_resize=True)
                    self.table.resizeColumns(max_size = 100) # pixels
                    self.control_widget.setVisible(True)
                    if db.name == "pdb":
                        self.show_best_matching_container.setVisible(True)
            except RuntimeError:
                # The user closed the window before the results came back.
                # TODO: Investigate the layer of abstraction in ui/src/gui.py that makes
                # QWidget a member variable of a tool rather than the tool itself, so that
                # QWidget.closeEvent() can be used to do this cleanly.
                pass
        else:
            self.session.logger.warning("BLAST search returned no results.")
            self.tool_window.destroy()

    # Show a sequence-only hit's webpage
    def load_sequence(self, selections: list['BlastResultsRow']) -> None:
        db = AvailableDBsDict[self.params.database]
        db_url = db.database_url
        urls = []
        for row in selections:
            code = row[db.fetchable_col]
            urls.append(db_url % code)
        show_url(self.session, urls[0])
        for url in urls[1:]:
            show_url(self.session, url, new_tab=True)


    # Loading (and spatially matching) a match entry
    def load(self, selections: list['BlastResultsRow']) -> None:
        """Load the model from the results database.
        """
        db = AvailableDBsDict[self.params.database]
        for row in selections:
            code = row[db.fetchable_col]
            if self.params.database in ["alphafold", "esmfold"]:
                models, chain_id = db.load_model(
                    self.session, code, self.params.chain, self.params._asdict().get("version", "1")
                )
            else:
                models, chain_id = db.load_model(
                    self.session, code, self.params.chain
                )
            if not models:
                return
            if not self.params.chain:
                if not db.name == 'alphafold':
                    run(self.session, "select clear")
                else:
                    self._log_alphafold(models)
            else:
                for m in models:
                    db.display_model(self.session, self.params.chain, m, chain_id)

    def _log_alphafold(self, models):
        query_match = self._sequences[0][1]
        query_seq = Sequence(name = 'query', characters = query_match.h_seq)
        from chimerax.alphafold.match import _similarity_table_html
        for m in models:
            # TODO: Would be nice if all models were in one log table.
            msg = _similarity_table_html(m.chains[0], query_seq, m.database.id)
            m.session.logger.info(msg, is_html = True)

    # Code for displaying matches as multiple sequence alignment
    def _show_mav(self, selections) -> None:
        """
        Collect the names and sequences of selected matches. All sequences
        should have the same length because they include the gaps inserted by
        the BLAST alignment.
        """
        ids = [hit['id'] for hit in selections]
        ids.insert(0, 0)
        names = []
        seqs = []
        for sid in ids:
            name, match = self._sequences[sid]
            names.append(name)
            seqs.append(match.sequence)
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


class BlastResultsWorker(QThread):
    job_failed = Signal(str)
    parse_failed = Signal(str)
    parsing_results = Signal()

    report_hits = Signal(list)
    report_sequences = Signal(dict)

    def __init__(self, session, job, database = None, results = None, sequence = None, atomspec = None):
        super().__init__()
        self.session = session
        self.job = job
        self.results = results
        self.database = database
        self.sequence = sequence
        self.atomspec = atomspec

    @Slot()
    def run(self):
        if self.job:
            if self.job.status == 'failed':
                self.job_failed.emit('failed')
                self.exit(1)
            if self.job.status == 'deleted':
                self.job_failed.emit('deleted')
                self.exit(1)
            if self.job.status == 'canceled':
                self.job_failed.emit('canceled')
                self.exit(1)
            try:
                self.results = self.job.get_results()
            except Exception as e:
                # TODO: Get info from backend as to why
                self.job_failed.emit(str(e))
                self.exit(1)
        self._parse_results(self.database, self.results, self.sequence, self.atomspec)

    @Slot()
    def stop(self):
        pass

    def _parse_results(self, db, results, sequence, atomspec):
        try:
            self.parsing_results.emit()
            hits, sequences = parse_blast_results(db, results, sequence, atomspec)
            self.report_hits.emit(hits)
            self.report_sequences.emit(sequences)
        except Exception as e:
            self.parse_failed.emit(str(e))
