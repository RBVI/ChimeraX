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

from typing import Optional, Union

from Qt.QtWidgets import QMenu, QWidget

from chimerax.core.settings import Settings
from chimerax.ui.widgets import ItemTable

class DICOMTable(ItemTable):
    def __init__(
        self, control_widget: Union[QMenu, QWidget], settings: 'DICOMSettings', parent = Optional[QWidget]
    ):
        super().__init__(
            column_control_info=(
                control_widget
                , settings
                , {}
                , True        # fallback default for column display
                , None         # display callback
                , None         # number of checkbox columns
                , False         # Whether to show global buttons
            )
            , parent=parent
        )

class DICOMTableSettings(Settings):
    EXPLICIT_SAVE = {DICOMTable.DEFAULT_SETTINGS_ATTR: {}}
