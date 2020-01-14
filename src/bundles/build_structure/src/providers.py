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

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QRadioButton, QLineEdit, QWidget, QHBoxLayout
from PyQt5.QtWidgets import QCheckBox, QSizePolicy
from PyQt5.QtGui import QDoubleValidator
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
        tip.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        layout.addWidget(tip, alignment=Qt.AlignCenter)
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
            args.append("xyz " + ','.join(coords))
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

