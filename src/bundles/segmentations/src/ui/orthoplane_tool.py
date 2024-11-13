# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
from chimerax.core.tools import ToolInstance

from chimerax.segmentations.segmentation_tracker import get_tracker
from chimerax.segmentations.types import Direction, Axis
from chimerax.segmentations.ui.orthoplanes import PlaneViewer, PlaneViewerManager
from chimerax.ui import MainToolWindow

from Qt.QtWidgets import QVBoxLayout

class OrthoplaneTool(ToolInstance):
    help = "help:user/tools/orthoplanetool.html"
    SESSION_ENDURING = True

    def __init__(self, session=None, name="Orthoplane Viewer"):
        super().__init__(session, name)
        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.segmentation_tracker = get_tracker()
        self._orthoplane_manager = PlaneViewerManager(session)
        self.parent = self.tool_window.ui_area
        self.parent.setLayout(QVBoxLayout())
        self.viewer = PlaneViewer(self.parent, self._orthoplane_manager, self.session, Axis.AXIAL)
        self.parent.layout().addWidget(self.viewer.container)
        self.tool_window.manage("side")

    def delete(self):
        self.viewer.close()
        super().delete()
