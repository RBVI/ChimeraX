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

from Qt.QtWidgets import QVBoxLayout, QWidget
from Qt.QtGui import QImage, QPainter
from Qt.QtCore import Qt

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
        self.parent.grab = self._grab_window
        self.tool_window.manage("side")

    def _grab_window(self):
        rest_of_window = QWidget.grab(self.viewer.parent)
        graphics_area = self.viewer.widget.grab()
        total_size = self.viewer.parent.size()
        widget_size = self.viewer.widget.size()
        painter = QPainter(rest_of_window)
        resized_opengl_image = graphics_area.scaled(widget_size.width(), widget_size.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawImage(self.viewer.widget.x() + self.viewer.container.x(), self.viewer.widget.y() + self.viewer.container.y(), resized_opengl_image)
        painter.end()
        return rest_of_window


    def delete(self):
        self.viewer.close()
        super().delete()
