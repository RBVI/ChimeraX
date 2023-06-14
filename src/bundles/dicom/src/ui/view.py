from chimerax.core.commands import StringArg, CmdDesc, register, BoolArg
from chimerax.core.models import REMOVE_MODELS
from chimerax.map import Volume
from chimerax.nifti import NiftiGrid
from chimerax.nrrd import NRRDGrid
from ..dicom import DicomGrid
from .orthoplanes import PlaneViewer, PlaneViewerManager, Axis
from Qt.QtWidgets import QWidget, QGridLayout

medical_types = [DicomGrid, NiftiGrid, NRRDGrid]


class FourUpView(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self._graphics_area = session.ui.main_window.graphicsArea()

        self._newMWLayout = QGridLayout(parent=self)

        self._orthoplane_manager = PlaneViewerManager(session)

        self._orthoplane_top_left = PlaneViewer(self, self._orthoplane_manager, session, Axis.AXIAL)
        self._orthoplane_bottom_left = PlaneViewer(self, self._orthoplane_manager, session, Axis.CORONAL)
        self._orthoplane_bottom_right = PlaneViewer(self, self._orthoplane_manager, session, Axis.SAGITTAL)

        self._newMWLayout.addWidget(self._orthoplane_top_left.container, 0, 0)
        self._newMWLayout.addWidget(session.ui.main_window.graphicsArea(), 0, 1)
        self._newMWLayout.addWidget(self._orthoplane_bottom_left.container, 1, 0)
        self._newMWLayout.addWidget(self._orthoplane_bottom_right.container, 1, 1)

        self._newMWLayout.setContentsMargins(0, 0, 0, 0)
        self._newMWLayout.setSpacing(0)

        self.setLayout(self._newMWLayout)

    @property
    def graphics_area(self):
        return self._graphics_area

    def register_segmentation_tool(self, tool):
        self._orthoplane_manager.register_segmentation_tool(tool)

    def clear_segmentation_tool(self):
        self._orthoplane_manager.clear_segmentation_tool()

    def toggle_guidelines(self):
        self._orthoplane_manager.toggle_guidelines()

    def add_segmentation(self, seg):
        self._orthoplane_manager.add_segmentation(seg)

    def segmentation_tool_open(self):
        return self._orthoplane_manager.have_seg_tool
