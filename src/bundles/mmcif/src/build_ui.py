# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from Qt.QtWidgets import QLabel, QGridLayout, QLineEdit
from Qt.QtWidgets import QSizePolicy
from Qt.QtCore import Qt
from chimerax.core.errors import UserError
from chimerax.build_structure import StartStructureProvider


class CCDProvider(StartStructureProvider):
    def command_string(self, widget):
        ccd_edit = widget.findChild(QLineEdit, "CCD")
        text = ccd_edit.text().strip()
        if not text:
            raise UserError("No CCD ID given")
        from chimerax.core.commands import StringArg
        return "open %s from ccd" % StringArg.unparse(text)

    def fill_parameters_widget(self, widget):
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(QLabel("CCD ID:"), 1, 0, alignment=Qt.AlignRight)
        ccd_edit = QLineEdit()
        ccd_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(ccd_edit, 1, 1)
        ccd_edit.setObjectName("CCD")
        hint = QLabel('Input a <a href="https://www.wwpdb.org/data/ccd">PDB Chemical Component Dictionary</a> ID')
        hint.setOpenExternalLinks(True)
        layout.addWidget(hint, 2, 0, 1, 2, alignment=Qt.AlignCenter)
        layout.setRowStretch(3, 1)

    def execute_command(self, structure, args):
        raise UserError('There is no "build start ccd" command.'
            '  Use "open ccd:<CCD identifier>" instead.')
