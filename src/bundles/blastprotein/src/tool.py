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

_default_instance_prefix = "bp"
_instance_map = {} # Map of blastprotein results names to results instances

def _make_instance_name():
    n = 1
    while True:
        instance_name = _default_instance_prefix + str(n)
        if instance_name not in _instance_map:
            return instance_name
        n += 1

def find(instance_name):
    return _instance_map.get(instance_name, None)

def find_match(instance_name):
    if instance_name is None:
        if len(_instance_map) == 1:
            for name, inst in _instance_map.items():
                return inst
        if len(_instance_map) > 1:
            raise UserError("no name specified with multiple "
                            "active blastprotein instances")
        else:
            raise UserError("no active blastprotein instance")
    try:
        return _instance_map[instance_name]
    except KeyError:
        raise UserError("no blastprotein instance named \"%s\"" % instance_name)


class BlastProteinFormWidget(QWidget):
    def __init__(self, label, input_widget):
        super().__init__()
        layout = QFormLayout()
        self.__label = QLabel(label)
        self.__input_widget = input_widget()
        layout.setWidget(0, QFormLayout.LabelRole, self.__label)
        layout.setWidget(0, QFormLayout.FieldRole, self.__input_widget)
        self.setLayout(layout)

    def input_widget(self) -> QWidget:
        return self.__input_widget

    def label(self) -> QLabel:
        return self.__label

class BlastProteinTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:/user/tools/blastprotein.html"

    def __init__(self, session: Session, tool_name: str, *
                 , chain: Optional[str] = None, db: str = AvailableDBs[0]
                 , seqs: Optional[int] = 100
                 # Guards against changes in list order
                 , matrix: str = AvailableMatrices[AvailableMatrices.index("BLOSUM62")]
                 , cutoff: Optional[int] = -3, instance_name: Optional[str] = None):
        super().__init__(session, tool_name)

        if instance_name is None:
            instance_name = _make_instance_name()
        _instance_map[instance_name] = self
        self._instance_name = instance_name
        self._instance_name_formatted = "[name: %s]" % instance_name
        self._initialized = False
        self._blast_results = None
        self._viewer_index = 1

        self._protein_chain = chain
        self._current_database = db
        self._num_sequences = seqs
        self._current_matrix = matrix
        self._cutoff = cutoff

        self.display_name = "Blastprotein" + " " + self._instance_name_formatted
        self.menu_widgets: Dict[str, Union[QWidget, Option]] = {}
        self.tool_window = MainToolWindow(self)
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
        main_layout = QVBoxLayout()
        input_container_row1 = QWidget()
        menu_layout_row1 = QHBoxLayout()
        input_container_row2 = QWidget()
        menu_layout_row2 = QHBoxLayout()

        self.menu_widgets['chain'] = ChainMenuButton(self.session, no_value_button_text = "No chain chosen")

        self.menu_widgets['database'] = BlastProteinFormWidget("Database", QComboBox)

        self.menu_widgets['sequences'] = BlastProteinFormWidget("# Sequences", QSpinBox)
        self.menu_widgets['sequences'].input_widget().setRange(1, 5000)
        self.menu_widgets['sequences'].input_widget().setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.menu_widgets['matrices'] = BlastProteinFormWidget("Matrix", QComboBox)

        self.menu_widgets['cutoff'] = BlastProteinFormWidget("Cutoff 1e", QSpinBox)
        self.menu_widgets['cutoff'].input_widget().setRange(-100, 100)
        self.menu_widgets['cutoff'].input_widget().setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.menu_widgets['start'] = QPushButton("BLAST")

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
                , "name", str(self._instance_name)
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
        instance_name = data.get("instance_name", _make_instance_name())
        tmp = cls(
            session
            , instance_name
            , chain = data['_protein_chain']
            , db = data["_current_database"]
            , seqs = data["_num_sequences"]
            , matrix = data["_current_matrix"]
            , cutoff = data["_cutoff"]
        )
        tmp._viewer_index = data.get("_viewer_index", 1)
        return tmp

    def take_snapshot(self, session, flags):
        data = {
            "version": 1,
            "_super": super().take_snapshot(session, flags),
            "_instance_name": self._instance_name,
            "_viewer_index": self._viewer_index,
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
