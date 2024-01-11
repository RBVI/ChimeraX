# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import string
from typing import Dict, Optional, Union

from Qt.QtWidgets import (
    QPushButton, QSizePolicy
    , QVBoxLayout, QHBoxLayout, QComboBox
    , QWidget, QSpinBox, QAbstractSpinBox
    , QStackedWidget, QPlainTextEdit
    , QLineEdit
)

from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.core.session import Session
from chimerax.core.tools import ToolInstance

from chimerax.atomic import Residue
from chimerax.atomic.widgets import ChainMenuButton

from chimerax.ui import MainToolWindow

from ..data_model import (
    AvailableDBs, AvailableMatrices, CurrentDBVersions
)
from ..utils import make_instance_name
from .widgets import BlastProteinFormWidget

class BlastProteinTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:user/tools/blastprotein.html"

    def __init__(self, session: Session, *
                 , sequences: Optional[Union[str, list[str]]] = None
                 , uniprot_id: Optional[str] = None
                 , chain: Optional[str] = None
                 , db: str = AvailableDBs[0]
                 , seqs: Optional[int] = 100
                 # Guards against changes in list order
                 , matrix: str = AvailableMatrices[AvailableMatrices.index("BLOSUM62")]
                 , cutoff: Optional[int] = -3
                 , version: Optional[int] = None
                 , instance_name: Optional[str] = None):
        self.display_name = "Blast Protein"
        super().__init__(session, self.display_name)

        self._protein_chain = chain
        self._last_menu_option = None
        self._uniprot_id = uniprot_id
        self._sequences = sequences
        self._current_database = db
        self._num_sequences = seqs
        self._current_matrix = matrix
        self._cutoff = cutoff
        self._version = version

        self.menu_widgets: Dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self):
        """
        Build the BlastProtein Qt GUI.

        Args:
            chain:
            database:
            num_sequences:
            matrix:
            cutoff:

        Parameters are exposed from the start so that they may be correctly set
        if the BlastProtein GUI is spawned as the result of a blastprotein command
        being entered into the ChimeraX command line.
        """
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        self.main_layout = QVBoxLayout()
        self.input_container_row1 = QWidget(parent)
        self.input_container_row2 = QWidget(parent)
        self.input_container_row3 = QWidget(parent)
        self.input_container_row4 = QWidget(parent)
        self.input_container_row5 = QWidget(parent)
        self.menu_layout_row1 = QHBoxLayout()
        self.menu_layout_row2 = QHBoxLayout()
        self.menu_layout_row3 = QHBoxLayout()
        self.menu_layout_row4 = QHBoxLayout()
        self.menu_layout_row5 = QHBoxLayout()
        self.menu_layout_row5.insertStretch(0,5)

        # Row 1
        self.menu_widgets['matrices'] = BlastProteinFormWidget("Matrix", QComboBox, self.input_container_row1)
        self.menu_widgets['cutoff'] = BlastProteinFormWidget("Cutoff 1e", QSpinBox, self.input_container_row1)
        self.menu_widgets['sequences'] = BlastProteinFormWidget("# Sequences", QSpinBox, self.input_container_row1)

        self.menu_widgets['cutoff'].input_widget.setRange(-100, 100)
        self.menu_widgets['cutoff'].input_widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.menu_widgets['sequences'].input_widget.setRange(1, 5000)
        self.menu_widgets['sequences'].input_widget.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # Row 2
        self.menu_widgets['chain'] = BlastProteinFormWidget("Query", input_widget=None, parent=self.input_container_row2)
        self.menu_widgets['_chain_button'] = ChainMenuButton(
            self.session, no_value_button_text = "No chain chosen", parent=self.menu_widgets['chain']
            , special_items=["UniProt ID", "Raw Sequence"]
            , filter_func = lambda c: c.polymer_type == Residue.PT_AMINO
        )
        self.menu_widgets['chain'].input_widget = self.menu_widgets['_chain_button']
        self.menu_widgets['database'] = BlastProteinFormWidget("Database", QComboBox, self.input_container_row2)
        self.menu_widgets['version'] = BlastProteinFormWidget("Version", QSpinBox, self.input_container_row2)
        self.menu_widgets['version'].input_widget.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # There is absolutely a fancy way to do this with QStackedWidget, but this is faster
        # Row 3
        self.menu_widgets['uniprot_input'] = QLineEdit(parent = self.input_container_row3)
        self.menu_widgets['uniprot_input'].setPlaceholderText("UniProt ID")
        self.input_container_row3.hide()
        # Row 4
        self.menu_widgets['seq_input'] = QPlainTextEdit(parent=self.input_container_row4)
        self.menu_widgets['seq_input'].setPlaceholderText("Input Sequence")
        self.input_container_row4.hide()

        # Row 5
        self.menu_widgets['help'] = QPushButton("Help", self.input_container_row5)
        self.menu_widgets['apply'] = QPushButton("Apply", self.input_container_row5)
        self.menu_widgets['reset'] = QPushButton("Reset", self.input_container_row5)
        self.menu_widgets['close'] = QPushButton("Close", self.input_container_row5)
        self.menu_widgets['ok'] = QPushButton("OK", self.input_container_row5)

        for widget in ['help', 'apply', 'reset', 'close', 'ok']:
            self.menu_widgets[widget].setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Lay the menu out
        self.menu_layout_row1.addWidget(self.menu_widgets['matrices'])
        self.menu_layout_row1.addWidget(self.menu_widgets['cutoff'])
        self.menu_layout_row1.addWidget(self.menu_widgets['sequences'])

        self.menu_layout_row2.addWidget(self.menu_widgets['chain'])
        self.menu_layout_row2.addWidget(self.menu_widgets['database'])
        self.menu_layout_row2.addWidget(self.menu_widgets['version'])

        self.menu_layout_row3.addWidget(self.menu_widgets['uniprot_input'], 25)
        self.menu_layout_row3.addStretch(75)
        self.menu_layout_row4.addWidget(self.menu_widgets['seq_input'])

        self.menu_layout_row5.addWidget(self.menu_widgets['help'])
        self.menu_layout_row5.addWidget(self.menu_widgets['apply'])
        self.menu_layout_row5.addWidget(self.menu_widgets['reset'])
        self.menu_layout_row5.addWidget(self.menu_widgets['close'])
        self.menu_layout_row5.addWidget(self.menu_widgets['ok'])

        # Functionalize the menu
        self.menu_widgets['database'].input_widget.addItems(AvailableDBs)
        self.menu_widgets['matrices'].input_widget.addItems(AvailableMatrices)
        self.menu_widgets['chain'].input_widget.value_changed.connect(self._on_chain_menu_changed)
        self.menu_widgets['sequences'].input_widget.valueChanged.connect(self._on_num_sequences_changed)
        self.menu_widgets['cutoff'].input_widget.valueChanged.connect(self._on_cutoff_value_changed)
        self.menu_widgets['database'].input_widget.currentIndexChanged.connect(self._on_database_changed)

        self.menu_widgets['help'].clicked.connect(lambda *, run=run, ses=self.session: run(ses, " ".join(["open", self.help])))
        self.menu_widgets['apply'].clicked.connect(self._run_blast_job)
        self.menu_widgets['reset'].clicked.connect(self._reset_options)
        self.menu_widgets['close'].clicked.connect(self.delete)
        self.menu_widgets['ok'].clicked.connect(self._run_and_close)

        # Fill in blastprotein's default arguments or snapshot values
        if self._uniprot_id:
            self.menu_widgets['chain'].input_widget.value = "UniProt ID"
            self._last_menu_option = "UniProt ID"
            self.menu_widgets['uniprot_input'].setText(self._uniprot_id)
        elif self._sequences:
            self.menu_widgets['chain'].input_widget.value = "Raw Sequence"
            self._last_menu_option = "Raw Sequence"
            self.menu_widgets['seq_input'].setPlainText(self._sequences)
        else:
            pass
        self.menu_widgets['database'].input_widget.setCurrentIndex(AvailableDBs.index(self._current_database))
        self.menu_widgets['sequences'].input_widget.setValue(self._num_sequences)
        self.menu_widgets['matrices'].input_widget.setCurrentIndex(AvailableMatrices.index(self._current_matrix))
        self.menu_widgets['cutoff'].input_widget.setValue(self._cutoff)

        self.menu_widgets['version'].label.hide()
        self.menu_widgets['version'].input_widget.hide()
        if self._current_database == "esmfold":
            self.menu_widgets['version'].input_widget.setRange(0, CurrentDBVersions[self._current_database])
            self.menu_widgets['version'].input_widget.show()
            self.menu_widgets['version'].label.show()
        if self._current_database == "alphafold":
            self.menu_widgets['version'].input_widget.setRange(1, CurrentDBVersions[self._current_database])
            self.menu_widgets['version'].input_widget.show()
            self.menu_widgets['version'].label.show()
        else:
            self.menu_widgets['version'].input_widget.setRange(1, CurrentDBVersions[self._current_database])
        self.menu_widgets['version'].input_widget.valueChanged.connect(self._on_version_changed)

        self.input_container_row1.setLayout(self.menu_layout_row1)
        self.input_container_row2.setLayout(self.menu_layout_row2)
        self.input_container_row3.setLayout(self.menu_layout_row3)
        self.input_container_row4.setLayout(self.menu_layout_row4)
        self.input_container_row5.setLayout(self.menu_layout_row5)
        self.main_layout.addWidget(self.input_container_row1)
        self.main_layout.addWidget(self.input_container_row2)
        self.main_layout.addWidget(self.input_container_row3)
        self.main_layout.addWidget(self.input_container_row4)
        self.main_layout.addWidget(self.input_container_row5)
        self.main_layout.addStretch()

        for layout in [self.main_layout, self.menu_layout_row5]:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        for layout in [self.menu_layout_row3, self.menu_layout_row4]:
            layout.setContentsMargins(2, 2, 2, 2)
        for layout in [self.menu_layout_row1, self.menu_layout_row2]:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

        self.tool_window.ui_area.setLayout(self.main_layout)
        self.tool_window.manage('side')

    #
    # Data population and action callbacks for menu items
    #
    def _reset_options(self) -> None:
        self.menu_widgets['chain'].input_widget.value = None
        self.menu_widgets['database'].input_widget.setCurrentIndex(AvailableDBs.index('pdb'))
        self.menu_widgets['sequences'].input_widget.setValue(100)
        self.menu_widgets['matrices'].input_widget.setCurrentIndex(AvailableMatrices.index('BLOSUM62'))
        self.menu_widgets['cutoff'].input_widget.setValue(-3)
        self.menu_widgets['version'].input_widget.setValue(1)

    def _run_blast_job(self) -> None:
        blast_input_type = chain = self.menu_widgets['chain'].input_widget.get_value()
        blast_input = None
        if blast_input_type == "Raw Sequence":
            blast_input = self.menu_widgets['seq_input'].toPlainText().translate(str.maketrans('', '', string.whitespace))
        elif blast_input_type == "UniProt ID":
            blast_input = self.menu_widgets['uniprot_input'].text().translate(str.maketrans('', '', string.whitespace))
        else: # it's a chain
            try:
                blast_input = chain.string().split(" ")[-1]
            except:
                err = "Cannot run BLAST without some kind of sequence."
                if len(self.session.models) == 0:
                    err = err + " " + "Please open a model and select a chain, or input sequences or a UniProt ID."
                raise UserError(err)
        cmd_text = [
            "blastprotein"
            # When two or more models are displayed the protein name is inserted
            # before the Model #/Chain Name. Grabbing list[-1] always gets either
            # the only element or the Model #/Chain Name since it comes last.
            , blast_input
            , "database", self.menu_widgets['database'].input_widget.currentText()
            , "cutoff"
            , "".join(["1e", str(self._cutoff)])
            , "matrix", self.menu_widgets['matrices'].input_widget.currentText()
            , "maxSeqs", str(self._num_sequences)
            , "version", str(self._version)
            , "name", make_instance_name()
        ]
        run(self.session, " ".join(cmd_text))

    def _run_and_close(self) -> None:
        self._run_blast_job()
        self.delete()

    def _on_chain_menu_changed(self) -> None:
        chain = self.menu_widgets['chain'].input_widget.get_value()
        if chain == "UniProt ID":
            if chain != self._last_menu_option:
                self.menu_widgets['uniprot_input'].setText("")
                self._last_menu_option = chain
            self.input_container_row4.hide()
            self.menu_widgets['uniprot_input'].show()
            self.input_container_row3.show()
            self.tool_window.shrink_to_fit()
        elif chain == "Raw Sequence":
            if chain != self._last_menu_option:
                self.menu_widgets['seq_input'].setPlainText("")
                self._last_menu_option = chain
            self.input_container_row3.hide()
            self.menu_widgets['seq_input'].show()
            self.input_container_row4.show()
            self.tool_window.shrink_to_fit()
        else:
            try:
                self.input_container_row3.hide()
                self.menu_widgets['seq_input'].hide()
                self.tool_window.shrink_to_fit()
                self.menu_widgets['uniprot_input'].setText("")
                self.menu_widgets['seq_input'].setPlainText("")
                self._last_menu_option = chain.string().split(" ")[-1]
            except:
                # Maybe it changed because the last model was closed
                pass

    def _on_num_sequences_changed(self, value) -> None:
        self._num_sequences = value

    def _on_cutoff_value_changed(self, value) -> None:
        self._cutoff = value

    def _on_database_changed(self, _) -> None:
        self._current_database = self.menu_widgets['database'].input_widget.currentText()
        if self._current_database == "alphafold":
            self.menu_widgets['version'].input_widget.setRange(1, CurrentDBVersions[self._current_database])
            self.menu_widgets['version'].label.show()
            self.menu_widgets['version'].input_widget.show()
        elif self._current_database == "esmfold":
            self.menu_widgets['version'].input_widget.setRange(0, CurrentDBVersions[self._current_database])
            self.menu_widgets['version'].label.show()
            self.menu_widgets['version'].input_widget.show()
        else:
            self.menu_widgets['version'].input_widget.hide()
            self.menu_widgets['version'].label.hide()
        self.menu_widgets['version'].input_widget.setValue(CurrentDBVersions[self._current_database])
        self._version = CurrentDBVersions[self._current_database]

    def _on_version_changed(self, version) -> None:
        self._version = version

    #
    # Uncategorized
    #
    # TODO: Are these functions used?
    def _write_fasta(self, f, name, seq) -> None:
        print(name, len(seq))
        print(">", name, file=f)
        block_size = 60
        for i in range(0, len(seq), block_size):
            print(seq[i:i + block_size], file=f)

    def job_failed(self, job, error):
        raise UserError("BlastProtein failed: %s" % error)

    #
    # Saving / Restoring Sessions
    #
    @classmethod
    def from_snapshot(cls, session, data):
        if data['version'] == 2 or 'version' not in data:
            tmp = cls(
                session
                , chain = data['_protein_chain']
                , db = data["_current_database"]
                , seqs = data["_num_sequences"]
                , matrix = data["_current_matrix"]
                , cutoff = data["_cutoff"]
            )
        else:
            if data["_blast_input_type"] == "UniProt ID":
                tmp = cls(
                    session
                    , uniprot_id = data['_blast_input']
                    , db = data["_current_database"]
                    , seqs = data["_num_sequences"]
                    , matrix = data["_current_matrix"]
                    , cutoff = data["_cutoff"]
                )
            elif data["_blast_input_type"] == "Raw Sequence":
                tmp = cls(
                    session
                    , sequences = data['_blast_input']
                    , db = data["_current_database"]
                    , seqs = data["_num_sequences"]
                    , matrix = data["_current_matrix"]
                    , cutoff = data["_cutoff"]
                )
            elif data['_blast_input_type'] == "Chain":
                tmp = cls(
                    session
                    , chain = data['_blast_input']
                    , db = data["_current_database"]
                    , seqs = data["_num_sequences"]
                    , matrix = data["_current_matrix"]
                    , cutoff = data["_cutoff"]
                )
            else:
                 tmp = cls(
                    session
                    , db = data["_current_database"]
                    , seqs = data["_num_sequences"]
                    , matrix = data["_current_matrix"]
                    , cutoff = data["_cutoff"]
                )
        return tmp

    def take_snapshot(self, session, flags):
        blast_input_type = self.menu_widgets['chain'].input_widget.get_value()
        if blast_input_type == "UniProt ID":
            blast_input = self.menu_widgets['uniprot_input'].text().translate(str.maketrans('', '', string.whitespace))
        elif blast_input_type == "Raw Sequence":
            blast_input = self.menu_widgets['seq_input'].toPlainText().translate(str.maketrans('', '', string.whitespace))
        elif blast_input_type == "No chain chosen":
            blast_input_type = None
            blast_input = None
        else: # it's a chain
            try:
                blast_input = blast_input_type.string().split(" ")[-1]
            except AttributeError:
                blast_input = None
            blast_input_type = "Chain"
        data = {
            "version": 3,
            "_super": super().take_snapshot(session, flags),
            "_blast_input_type": blast_input_type,
            "_blast_input": blast_input,
            "_current_database": self._current_database,
            "_num_sequences": self._num_sequences,
            "_current_matrix": self._current_matrix,
            "_cutoff": self._cutoff
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        return BlastProteinTool.from_snapshot(session, data)
