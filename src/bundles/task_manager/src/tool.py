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

from chimerax.core.tools import ToolInstance

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

from chimerax.atomic import Residue
from chimerax.atomic.widgets import ChainMenuButton

from chimerax.ui import MainToolWindow

from ..data_model import (
    AvailableDBs, AvailableMatrices, CurrentDBVersions
)
from ..utils import make_instance_name
from .widgets import BlastProteinFormWidget

class TaskManager(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = "help:user/tools/taskmanager.html"

    def __init__(self, session):
        self.display_name = "Task Manager"
        super().__init__(session, self.display_name)
        self._build_ui()

    def _build_ui(self):
        ...

