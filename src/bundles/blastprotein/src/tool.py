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
import json
from typing import Any, Callable, Dict, List, Optional, Union

from Qt.QtWidgets import QPushButton
from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout
from Qt.QtWidgets import QComboBox, QLabel, QWidget
from Qt.QtWidgets import QSpinBox, QAbstractSpinBox

from chimerax.atomic.widgets import ChainMenuButton
from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.core.session import Session
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.options import IntOption, OptionsPanel
from chimerax.ui.options import Option

from .databases import AvailableDBs, AvailableMatrices
from .widgets import BlastProteinFormWidget
from .utils import make_instance_name

class BlastProteinTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:/user/tools/blastprotein.html"

    def __init__(self, session: Session, str, *
                 , chain: Optional[str] = None, db: str = AvailableDBs[0]
                 , seqs: Optional[int] = 100
                 # Guards against changes in list order
                 , matrix: str = AvailableMatrices[AvailableMatrices.index("BLOSUM62")]
                 , cutoff: Optional[int] = -3, instance_name: Optional[str] = None):
        self.display_name = "Blast Protein"
        super().__init__(session, self.display_name)

        self._protein_chain = chain
        self._current_database = db
        self._num_sequences = seqs
        self._current_matrix = matrix
        self._cutoff = cutoff

        self.menu_widgets: Dict[str, Union[QWidget, Option]] = {}
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

        main_layout = QVBoxLayout()
        input_container_row1 = QWidget(parent)
        input_container_row2 = QWidget(parent)
        menu_layout_row1 = QHBoxLayout()
        menu_layout_row2 = QHBoxLayout()

        self.menu_widgets['chain'] = ChainMenuButton(self.session, no_value_button_text = "No chain chosen", parent=input_container_row1)

        self.menu_widgets['database'] = BlastProteinFormWidget("Database", QComboBox, input_container_row1)

        self.menu_widgets['sequences'] = BlastProteinFormWidget("# Sequences", QSpinBox, input_container_row1)
        self.menu_widgets['sequences'].input_widget().setRange(1, 5000)
        self.menu_widgets['sequences'].input_widget().setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.menu_widgets['matrices'] = BlastProteinFormWidget("Matrix", QComboBox, input_container_row1)

        self.menu_widgets['cutoff'] = BlastProteinFormWidget("Cutoff 1e", QSpinBox, input_container_row2)
        self.menu_widgets['cutoff'].input_widget().setRange(-100, 100)
        self.menu_widgets['cutoff'].input_widget().setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.menu_widgets['start'] = QPushButton("BLAST", input_container_row2)

        # Lay the menu out
        menu_layout_row1.addWidget(self.menu_widgets['chain'])
        menu_layout_row1.addWidget(self.menu_widgets['database'])
        menu_layout_row1.addWidget(self.menu_widgets['sequences'])

        menu_layout_row2.addWidget(self.menu_widgets['matrices'])
        menu_layout_row2.addWidget(self.menu_widgets['cutoff'])
        menu_layout_row2.addWidget(self.menu_widgets['start'])

        # Functionalize the menu
        self.menu_widgets['database'].input_widget().addItems(AvailableDBs)
        self.menu_widgets['matrices'].input_widget().addItems(AvailableMatrices)
        self.menu_widgets['start'].clicked.connect(self._blast_pressed)
        self.menu_widgets['sequences'].input_widget().valueChanged.connect(self._on_num_sequences_changed)
        self.menu_widgets['cutoff'].input_widget().valueChanged.connect(self._on_cutoff_value_changed)

        # Fill in blastprotein's default arguments or snapshot values
        self.menu_widgets['chain'].value = self._protein_chain
        self.menu_widgets['database'].input_widget().setCurrentIndex(AvailableDBs.index(self._current_database))
        self.menu_widgets['sequences'].input_widget().setValue(self._num_sequences)
        self.menu_widgets['matrices'].input_widget().setCurrentIndex(AvailableMatrices.index(self._current_matrix))
        self.menu_widgets['cutoff'].input_widget().setValue(self._cutoff)

        input_container_row1.setLayout(menu_layout_row1)
        input_container_row2.setLayout(menu_layout_row2)
        main_layout.addWidget(input_container_row1)
        main_layout.addWidget(input_container_row2)

        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)
        self.tool_window.ui_area.setLayout(main_layout)
        self.tool_window.manage('side')

    #
    # Data population and action callbacks for menu items
    #
    def _run_blast_job(self) -> None:
        try:
            chain = self.menu_widgets['chain'].get_value().string().split(" ")[-1]
        except:
            err = "Cannot run BLAST without a chain."
            if len(self.session.models) == 0:
                err = err + " " + "Please open a model and select a chain."
            raise UserError(err)
        else:
            cmd_text = [
                "blastprotein"
                # When two or more models are displayed the protein name is inserted
                # before the Model #/Chain Name. Grabbing list[-1] always gets either
                # the only element or the Model #/Chain Name since it comes last.
                , chain
                , "database", self.menu_widgets['database'].input_widget().currentText()
                , "cutoff"
                , "".join(["1e", str(self._cutoff)])
                , "matrix", self.menu_widgets['matrices'].input_widget().currentText()
                , "maxSeqs", str(self._num_sequences)
                , "name", make_instance_name()
            ]
            run(self.session, " ".join(cmd_text))


    def _blast_pressed(self) -> None:
        self._run_blast_job()

    def _on_num_sequences_changed(self, value) -> None:
        self._num_sequences = value

    def _on_cutoff_value_changed(self, value) -> None:
        self._cutoff = value

    #
    # Uncategorized
    #
    # TODO: Are these functions used?
    def _write_fasta(self, f, name, seq) -> None:
        print(name, len(seq))
        print(">", name, file=f)
        block_size = 60
        for i in range(0, len(seq), block_size):
            print(seq[i:i+block_size], file=f)

    def job_failed(self, job, error):
        raise UserError("BlastProtein failed: %s" % error)

    #
    # Saving / Restoring Sessions
    #
    @classmethod
    def from_snapshot(cls, session, data):
        tmp = cls(
            session
            , chain = data['_protein_chain']
            , db = data["_current_database"]
            , seqs = data["_num_sequences"]
            , matrix = data["_current_matrix"]
            , cutoff = data["_cutoff"]
        )
        return tmp

    def take_snapshot(self, session, flags):
        data = {
            "version": 2,
            "_super": super().take_snapshot(session, flags),
            "_protein_chain": self._protein_chain,
            "_current_database": self._current_database,
            "_num_sequences": self._num_sequences,
            "_current_matrix": self._current_matrix,
            "_cutoff": self._cutoff
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        return BlastProteinTool.from_snapshot(session, data)
