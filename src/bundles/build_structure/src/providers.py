# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QRadioButton, QLineEdit, QWidget
from PyQt5.QtWidgets import QCheckBox, QSizePolicy, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from chimerax.core.errors import UserError

def fill_widget(name, widget):
    if name == "atom":
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.addWidget(QLabel("Place helium atom at:"), alignment=Qt.AlignCenter)
        button_area = QWidget()
        layout.addWidget(button_area, alignment=Qt.AlignCenter)
        button_layout = QGridLayout()
        button_area.setLayout(button_layout)
        button = QRadioButton("Center of view")
        button.setChecked(True)
        button.setObjectName("atom centered")
        button_layout.addWidget(button, 0, 0, 1, 7, alignment=Qt.AlignLeft)
        button_layout.addWidget(QRadioButton(""), 1, 0, alignment=Qt.AlignLeft)
        for i, axis in enumerate("xyz"):
            button_layout.addWidget(QLabel(axis + ":"), 1, i*2+1, alignment=Qt.AlignRight)
            coord_entry = QLineEdit()
            coord_entry.setText("0")
            coord_entry.setFixedWidth(50)
            coord_entry.setValidator(QDoubleValidator())
            coord_entry.setObjectName(axis + " coord")
            button_layout.addWidget(coord_entry, 1, 2*(i+1), alignment=Qt.AlignLeft)
        tip = QLabel("Use 'Modify Structure' section to change element type and add bonded atoms")
        tip.setWordWrap(True)
        # specify alignment within the label itself (instead of the layout) so that the label
        # is given the full width of the layout to work with, otherwise you get unneeded line
        # wrapping
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)
        res_name_area = QWidget()
        layout.addWidget(res_name_area, alignment=Qt.AlignCenter)
        res_name_layout = QHBoxLayout()
        res_name_area.setLayout(res_name_layout)
        res_name_layout.addWidget(QLabel("Residue name:"))
        res_name_entry = QLineEdit()
        res_name_entry.setText("UNL")
        res_name_entry.setFixedWidth(50)
        res_name_entry.setObjectName("res name")
        res_name_layout.addWidget(res_name_entry)
        check_box = QCheckBox("Select placed atom")
        check_box.setChecked(True)
        check_box.setObjectName("select atom")
        layout.addWidget(check_box, alignment=Qt.AlignCenter)
        layout.addStretch(1)
    elif name == "peptide":
        layout = QVBoxLayout()
        layout.setContentsMargins(3,0,3,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.addWidget(QLabel("Peptide Sequence", alignment=Qt.AlignCenter))
        peptide_seq = QTextEdit()
        peptide_seq.setObjectName("peptide sequence")
        layout.addWidget(peptide_seq, stretch=1)
        tip = QLabel("'Apply' button will bring up dialog for setting"
            "\N{GREEK CAPITAL LETTER PHI}/\N{GREEK CAPITAL LETTER PSI} angles")
        tip.setWordWrap(True)
        # specify alignment within the label itself (instead of the layout) so that the label
        # is given the full width of the layout to work with, otherwise you get unneeded line
        # wrapping
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

def process_widget(name, widget):
    from chimerax.core.commands import StringArg
    args = []
    if name == "atom":
        button = widget.findChild(QRadioButton, "atom centered")
        if not button.isChecked():
            coords = []
            for axis in "xyz":
                coord_entry = widget.findChild(QLineEdit, axis + " coord")
                if not coord_entry.hasAcceptableInput():
                    raise UserError("%s coordinate must be a number" % axis)
                coords.append(coord_entry.text().strip())
            args.append("pos " + ','.join(coords))
        res_name_entry = widget.findChild(QLineEdit, "res name")
        res_name = res_name_entry.text().strip()
        if not res_name:
            raise UserError("Residue name must not be empty/blank")
        if res_name != "UNL":
            args.append("res %s" % StringArg.unparse(res_name))
        check_box = widget.findChild(QCheckBox, "select atom")
        if not check_box.isChecked():
            args.append("select false")
    return " ".join(args)

from chimerax.core.commands import register, CmdDesc, Command
command_registry = None
def process_command(session, name, structure, substring):
    # all the commands use the trick that the structure arg is temporarily made available in the
    # global namespace as '_structure'
    global _structure, command_registry
    _structure = structure
    try:
        if command_registry is None:
            # register commands to private registry
            from chimerax.core.commands.cli import RegisteredCommandInfo
            command_registry = RegisteredCommandInfo()

            from chimerax.core.commands import Float3Arg, StringArg, BoolArg, \
                Float2Arg, DynamicEnum

            # atom
            register("atom",
                CmdDesc(
                    keyword=[("position", Float3Arg), ("res_name", StringArg),
                        ("select", BoolArg)],
                    synopsis="place helium atom"
                ), shim_place_atom, registry=command_registry)

            # peptide
            from chimerax.core.commands import Annotation
            class RepeatableFloat2Arg(Annotation):
                allow_repeat = True
                parse = Float2Arg.parse
                unparse = Float2Arg.unparse

            register("peptide",
                CmdDesc(
                    required=[("sequence", StringArg), ("phi_psis", RepeatableFloat2Arg)],
                    keyword=[("position", Float3Arg), ("chain_id", StringArg),
                        ("rot_lib", DynamicEnum(session.rotamers.library_names))],
                    synopsis="construct peptide from sequence"
                ), shim_place_peptide, registry=command_registry)

        cmd = Command(session, registry=command_registry)
        cmd.run(name + ' ' + substring, log=False)
    finally:
        _structure = None

def shim_place_atom(session, position=None, res_name="UNL", select=True):
    from .start import place_helium
    a = place_helium(_structure, res_name, position=position)
    if select:
        session.selection.clear()
        a.selected = True
    return a

def shim_place_peptide(session, sequence, phi_psis, **kw):
    from .start import place_peptide
    return place_peptide(_structure, sequence, phi_psis, **kw)
