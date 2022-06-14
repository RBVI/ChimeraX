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

from chimerax.core.tools import ToolInstance
from chimerax.core.errors import UserError

_rrd = None
def renumber_residues_dialog(session, tool_name):
    global _rrd
    if _rrd is None:
        _rrd = RenumberResiduesDialog(session, tool_name)
    return _rrd

class RenumberResiduesDialog(ToolInstance):

    #help = "help:user/tools/rotamers.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QGroupBox, QButtonGroup, QRadioButton
        from Qt.QtWidgets import QCheckBox
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        type_box = QGroupBox("Renumber")
        layout.addWidget(type_box)
        box_layout = QVBoxLayout()
        box_layout.setContentsMargins(10,0,10,0)
        type_box.setLayout(box_layout)
        self.button_group = QButtonGroup()
        self.button_group.buttonClicked.connect(self._type_changed)
        self.selected_residues_button = QRadioButton("selected residues")
        box_layout.addWidget(self.selected_residues_button, alignment=Qt.AlignLeft)
        self.button_group.addButton(self.selected_residues_button)
        chain_layout = QHBoxLayout()
        chain_layout.setContentsMargins(0,0,0,0)
        box_layout.addLayout(chain_layout, stretch=1)
        chain_button = QRadioButton("chains:")
        chain_layout.addWidget(chain_button, alignment=Qt.AlignLeft)
        self.button_group.addButton(chain_button)
        chain_button.setChecked(True)
        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(session)
        chain_layout.addWidget(self.chain_list, alignment=Qt.AlignLeft, stretch=1)
        self.seq_numbering = QCheckBox("Chain renumbering starts at beginning of sequence\n"
            "(if known) even if initial residues are missing")
        from chimerax.ui import shrink_font
        shrink_font(self.seq_numbering, 0.9)
        box_layout.addWidget(self.seq_numbering, alignment=Qt.AlignHCenter|Qt.AlignTop)
        self.seq_numbering.setChecked(True)

        self.relative = QCheckBox("Maintain relative numbering within renumbered residues")
        layout.addWidget(self.relative, alignment=Qt.AlignCenter)
        self.relative.setChecked(True)

        from chimerax.ui.options import IntOption, OptionsPanel
        opts = OptionsPanel(scrolled=False, contents_margins=(0,0,3,3))
        layout.addWidget(opts, alignment=Qt.AlignCenter)
        self.numbering_start = IntOption("Starting from:", 1, None)
        opts.add_option(self.numbering_start)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        #bbox.accepted.connect(self.launch_rotamers)
        #bbox.button(qbbox.Apply).clicked.connect(self.launch_rotamers)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        #bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def delete(self):
        global _rrd
        _rrd = None
        super().delete()

    def _type_changed(self, button):
        self.seq_numbering.setEnabled(button != self.selected_residues_button)
