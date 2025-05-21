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

from Qt.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QRadioButton, QLineEdit, QWidget, QHBoxLayout
from Qt.QtWidgets import QCheckBox, QSizePolicy
from Qt.QtCore import Qt
from chimerax.core.errors import UserError
from chimerax.build_structure import StartStructureProvider

class IupacProvider(StartStructureProvider):
    def command_string(self, widget):
        iupac_edit = widget.findChild(QLineEdit, "IUPAC")
        text = iupac_edit.text().strip()
        if not text:
            raise UserError("No IUPAC name given")
        from chimerax.core.commands import StringArg
        return "open %s from iupac" % StringArg.unparse(text)

    def fill_parameters_widget(self, widget):
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(QLabel("IUPAC name:"), 1, 0, alignment=Qt.AlignRight)
        iupac_edit = QLineEdit()
        iupac_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(iupac_edit, 1, 1)
        iupac_edit.setObjectName("IUPAC")
        from .iupac import fetcher_info
        providers = " or ".join(['<a href="%s">%s</a>' % (info[3], info[2]) for info in fetcher_info])
        acknowledgement = QLabel('IUPAC support courtesy of %s' % providers)
        acknowledgement.setWordWrap(True)
        acknowledgement.setOpenExternalLinks(True)
        layout.addWidget(acknowledgement, 2, 0, 1, 2)
        layout.setRowStretch(3, 1)

    def execute_command(self, structure, args):
        raise UserError('There is no "build start iupac" command.'
            '  Use "open iupac:<IUPAC name>" instead.')
